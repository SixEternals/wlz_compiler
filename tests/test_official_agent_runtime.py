"""Runtime behavior tests for work/official_triton_agent (no network, fake LLM/executor)."""

import importlib.util
import hashlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from wlz_optimizer.budget import BudgetController, BudgetLimits

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "work" / "official_triton_agent"


def _load_module(alias: str, filename: str, stubs: dict):
    spec = importlib.util.spec_from_file_location(alias, AGENT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {alias: module, **stubs}):
        spec.loader.exec_module(module)
    return module


def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _FakeArray(list):
    def sum(self):
        return sum(self)

    def __truediv__(self, denominator):
        return [value / denominator for value in self]


def load_genetic_operators():
    return _load_module(
        "official_runtime_genetic_operators",
        "genetic_operators.py",
        {
            "config": _stub("config", EAConfig=type("EAConfig", (), {})),
            "llm_interface": _stub("llm_interface", LLMInterface=type("LLMInterface", (), {})),
        },
    )


GENETIC_OPS = load_genetic_operators()
Individual = GENETIC_OPS.Individual


def load_evolutionary_algorithm():
    return _load_module(
        "official_runtime_evolutionary_algorithm",
        "evolutionary_algorithm.py",
        {
            "config": _stub("config", EAConfig=type("EAConfig", (), {})),
            "numpy": _stub(
                "numpy",
                arange=lambda start, stop, step, dtype=None: _FakeArray(
                    range(start, stop, step)
                ),
                random=SimpleNamespace(
                    choice=lambda n, size=1, replace=False, p=None: list(range(size))
                ),
            ),
            "genetic_operators": _stub(
                "genetic_operators",
                GeneticOperators=type("GeneticOperators", (), {}),
                Individual=Individual,
                _ancestor_ids=GENETIC_OPS._ancestor_ids,
            ),
            "executor": _stub(
                "executor",
                TritonExecutor=type("TritonExecutor", (), {}),
                EvaluationResult=type("EvaluationResult", (), {}),
            ),
        },
    )


def load_optimizer_agent():
    dummy = lambda name: type(name, (), {})
    return _load_module(
        "official_runtime_optimizer_agent",
        "optimizer_agent.py",
        {
            "config": _stub("config", EAConfig=dummy("EAConfig")),
            "llm_interface": _stub("llm_interface", LLMInterface=dummy("LLMInterface")),
            "contract_executor": _stub(
                "contract_executor",
                ContractCheckingExecutor=dummy("ContractCheckingExecutor"),
                MultiCaseContractExecutor=dummy("MultiCaseContractExecutor"),
            ),
            "executor": _stub("executor", TritonExecutor=dummy("TritonExecutor")),
            "genetic_operators": _stub("genetic_operators", GeneticOperators=dummy("GeneticOperators")),
            "evolutionary_algorithm": _stub(
                "evolutionary_algorithm", EvolutionaryAlgorithm=dummy("EvolutionaryAlgorithm")
            ),
        },
    )


def load_main():
    return _load_module(
        "official_runtime_main",
        "main.py",
        {
            "config": _stub("config", EAConfig=type("EAConfig", (), {})),
            "optimizer_agent": _stub(
                "optimizer_agent",
                TritonOptimizerAgent=type("TritonOptimizerAgent", (), {}),
            ),
        },
    )


class FakePrompt:
    @classmethod
    def from_messages(cls, messages):
        return cls()

    def __or__(self, other):
        return other


def load_llm_interface():
    langchain_core = _stub("langchain_core")
    prompts = _stub("langchain_core.prompts", ChatPromptTemplate=FakePrompt)
    langchain_core.prompts = prompts
    return _load_module(
        "official_runtime_llm_interface",
        "llm_interface.py",
        {
            "config": _stub("config", EAConfig=type("EAConfig", (), {})),
            "httpx": _stub("httpx", Client=lambda **kwargs: None),
            "langchain_openai": _stub("langchain_openai", ChatOpenAI=lambda **kwargs: None),
            "langchain_core": langchain_core,
            "langchain_core.prompts": prompts,
        },
    )


class FakeExecutor:
    def __init__(self, fitness: float = 0.5):
        self.fitness = fitness
        self.timeouts = []

    def evaluate(self, code, timeout=None):
        self.timeouts.append(timeout)
        return SimpleNamespace(
            fitness=self.fitness, success=True, speedup=1.0, execution_time=0.1, error=None
        )


class FakeGeneticOps:
    def __init__(self, tokens_used: int = 0, fail: bool = False):
        self.llm = SimpleNamespace(total_tokens_used=tokens_used)
        self.fail = fail
        self.mutation_calls = 0
        self.crossover_calls = 0

    def mutate(self, individual):
        self.mutation_calls += 1
        if self.fail:
            raise RuntimeError("llm down")
        return Individual(
            code=individual.code + "\n# mutated",
            metadata={
                "parent": individual.id,
                "operation": "mutation",
                "lineage": GENETIC_OPS._ancestor_ids(individual),
            },
        )

    def crossover(self, a, b):
        self.crossover_calls += 1
        if self.fail:
            raise RuntimeError("llm down")
        return Individual(
            code=a.code + "\n# crossed",
            metadata={"parents": [a.id, b.id], "operation": "crossover"},
        )


def make_config(**overrides):
    base = dict(
        population_size=2,
        max_generations=2,
        crossover_rate=0.0,
        mutation_rate=0.0,
        elite_ratio=0.5,
        max_total_tokens=200_000,
        timeout_seconds=123,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class BudgetEnforcementTests(unittest.TestCase):
    def setUp(self):
        self.ea_module = load_evolutionary_algorithm()

    def test_token_budget_stops_before_any_evaluation(self):
        executor = FakeExecutor()
        genetic_ops = FakeGeneticOps(tokens_used=100)
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            executor,
            make_config(population_size=1, max_generations=3, max_total_tokens=100),
        )
        best = ea.run(["seed code"])
        self.assertEqual(best.code, "seed code")
        self.assertEqual(ea.generation, 0)
        self.assertEqual(genetic_ops.mutation_calls, 0)
        self.assertEqual(genetic_ops.crossover_calls, 0)
        self.assertEqual(executor.timeouts, [])

    def test_deadline_stops_evolution_and_returns_best(self):
        executor = FakeExecutor()
        genetic_ops = FakeGeneticOps()
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            executor,
            make_config(population_size=1, max_generations=3),
        )
        best = ea.run(["seed code"], deadline_seconds=1e-9)
        self.assertEqual(best.code, "seed code")
        self.assertEqual(ea.generation, 0)
        self.assertEqual(genetic_ops.mutation_calls, 0)
        self.assertEqual(genetic_ops.crossover_calls, 0)
        self.assertEqual(executor.timeouts, [])

    def test_run_without_deadline_completes_and_passes_timeout(self):
        executor = FakeExecutor()
        ea = self.ea_module.EvolutionaryAlgorithm(
            FakeGeneticOps(),
            executor,
            make_config(max_generations=1),
        )
        ea.run(["seed a", "seed b"])
        self.assertEqual(ea.generation, 1)
        self.assertTrue(executor.timeouts)
        self.assertTrue(all(t == 123 for t in executor.timeouts))

    def test_evaluation_timeout_is_capped_by_shared_wall_budget(self):
        now = [0.0]
        budget = BudgetController(
            BudgetLimits(100, 300), clock=lambda: now[0]
        )
        now[0] = 299.0
        genetic_ops = FakeGeneticOps()
        genetic_ops.llm.budget_controller = budget
        genetic_ops.llm.budget_denial_reason = None
        executor = FakeExecutor()
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            executor,
            make_config(population_size=1, timeout_seconds=123),
        )
        ea.population = [Individual(code="seed code")]

        ea._evaluate_population()

        self.assertEqual(executor.timeouts, [1.0])

    def test_evolve_stops_breeding_when_token_budget_is_exhausted(self):
        genetic_ops = FakeGeneticOps(tokens_used=100)
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            FakeExecutor(),
            make_config(max_total_tokens=100, crossover_rate=1.0, mutation_rate=1.0),
        )
        ea.population = [
            Individual(code="a", fitness=0.8, metadata={"evaluated": True}),
            Individual(code="b", fitness=0.7, metadata={"evaluated": True}),
        ]
        ea.best_individual = ea.population[0]
        original_ids = [ind.id for ind in ea.population]

        ea.evolve_generation()

        self.assertEqual([ind.id for ind in ea.population], original_ids)
        self.assertEqual(genetic_ops.crossover_calls, 0)
        self.assertEqual(genetic_ops.mutation_calls, 0)

    def test_shared_budget_hard_stop_prevents_all_search_work(self):
        budget = BudgetController(BudgetLimits(100, 300))
        reservation = budget.reserve(1, 1, 0, 0, use_reserve=True).reservation
        budget.mark_uncertain(reservation)
        genetic_ops = FakeGeneticOps(tokens_used=2)
        genetic_ops.llm.budget_controller = budget
        genetic_ops.llm.budget_denial_reason = None
        executor = FakeExecutor()
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            executor,
            make_config(population_size=5, max_total_tokens=100),
        )

        best = ea.run(["seed code"])

        self.assertEqual(best.code, "seed code")
        self.assertEqual(genetic_ops.mutation_calls, 0)
        self.assertEqual(genetic_ops.crossover_calls, 0)
        self.assertEqual(executor.timeouts, [])
        self.assertEqual(budget.snapshot().stop_reason, "unknown_in_flight_call")

    def test_preexisting_reservation_denial_prevents_all_search_work(self):
        budget = BudgetController(BudgetLimits(100, 300))
        llm = TokenAccountingTests._make_interface(
            self,
            SimpleNamespace(content="unused"),
            budget,
            max_tokens=128,
        )
        with self.assertRaisesRegex(RuntimeError, "budget denied"):
            llm.generate("prompt")
        genetic_ops = FakeGeneticOps()
        genetic_ops.llm = llm
        executor = FakeExecutor()
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            executor,
            make_config(population_size=5, max_total_tokens=100),
        )

        best = ea.run(["seed code"])

        self.assertEqual(best.code, "seed code")
        self.assertEqual(genetic_ops.mutation_calls, 0)
        self.assertEqual(genetic_ops.crossover_calls, 0)
        self.assertEqual(executor.timeouts, [])
        self.assertLess(budget.snapshot().used_tokens, budget.limits.token_limit)
        self.assertEqual(llm.budget_denial_reason, "token_limit")

    def test_shared_budget_ignores_legacy_deadline_and_token_counters(self):
        budget = BudgetController(BudgetLimits(100, 300))
        genetic_ops = FakeGeneticOps(tokens_used=100)
        genetic_ops.llm.budget_controller = budget
        genetic_ops.llm.budget_denial_reason = None
        executor = FakeExecutor()
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            executor,
            make_config(
                population_size=1,
                max_generations=0,
                max_total_tokens=100,
            ),
        )

        best = ea.run(["seed code"], deadline_seconds=1e-9)

        self.assertEqual(best.code, "seed code")
        self.assertEqual(executor.timeouts, [123])
        self.assertIsNone(budget.snapshot().stop_reason)

    def test_reservation_denial_stops_after_failed_generation_call(self):
        budget = BudgetController(BudgetLimits(100, 300))
        llm = TokenAccountingTests._make_interface(
            self,
            SimpleNamespace(content="unused"),
            budget,
            max_tokens=128,
        )

        class BudgetDenyingGeneticOps:
            def __init__(self):
                self.llm = llm
                self.mutation_calls = 0
                self.crossover_calls = 0

            def mutate(self, individual):
                self.mutation_calls += 1
                self.llm.generate("prompt")
                return individual

            def crossover(self, first, second):
                self.crossover_calls += 1
                return first

        genetic_ops = BudgetDenyingGeneticOps()
        executor = FakeExecutor()
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            executor,
            make_config(population_size=5, max_total_tokens=100),
        )

        best = ea.run(["seed code"])

        self.assertEqual(best.code, "seed code")
        self.assertEqual(genetic_ops.mutation_calls, 1)
        self.assertEqual(genetic_ops.crossover_calls, 0)
        self.assertEqual(executor.timeouts, [])
        self.assertEqual(budget.snapshot().used_tokens, 0)
        self.assertEqual(llm.budget_denial_reason, "token_limit")
        self.assertEqual(
            llm.call_history[-1]["budget_denial_reason"], "token_limit"
        )

    def test_unknown_inflight_stops_after_failed_generation_call(self):
        budget = BudgetController(BudgetLimits(1_000, 300))
        llm = TokenAccountingTests._make_interface(
            self,
            SimpleNamespace(content="unused"),
            budget,
            max_tokens=128,
        )

        def timeout(_):
            raise TimeoutError("timed out")

        llm.llm = SimpleNamespace(invoke=timeout)

        class UncertainGeneticOps:
            def __init__(self):
                self.llm = llm
                self.mutation_calls = 0
                self.crossover_calls = 0

            def mutate(self, individual):
                self.mutation_calls += 1
                self.llm.generate("prompt")
                return individual

            def crossover(self, first, second):
                self.crossover_calls += 1
                return first

        genetic_ops = UncertainGeneticOps()
        executor = FakeExecutor()
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            executor,
            make_config(population_size=5, max_total_tokens=1_000),
        )

        best = ea.run(["seed code"])

        snapshot = budget.snapshot()
        self.assertEqual(best.code, "seed code")
        self.assertEqual(genetic_ops.mutation_calls, 1)
        self.assertEqual(genetic_ops.crossover_calls, 0)
        self.assertEqual(executor.timeouts, [])
        self.assertEqual(snapshot.stop_reason, "unknown_in_flight_call")
        self.assertGreater(snapshot.used_tokens, 0)
        self.assertLess(snapshot.used_tokens, snapshot.limits["token_limit"])


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.ea_module = load_evolutionary_algorithm()

    def test_mutation_failure_during_init_falls_back_to_seed(self):
        ea = self.ea_module.EvolutionaryAlgorithm(
            FakeGeneticOps(fail=True), FakeExecutor(), make_config(max_generations=0)
        )
        best = ea.run(["seed code"])
        self.assertEqual(len(ea.population), 2)
        self.assertEqual([ind.code for ind in ea.population], ["seed code", "seed code"])
        self.assertEqual(best.code, "seed code")

    def test_crossover_failure_falls_back_to_better_parent(self):
        ea = self.ea_module.EvolutionaryAlgorithm(
            FakeGeneticOps(fail=True),
            FakeExecutor(),
            make_config(crossover_rate=1.0),
        )
        strong = Individual(code="strong", fitness=0.7, metadata={"evaluated": True})
        weak = Individual(code="weak", fitness=0.3, metadata={"evaluated": True})
        ea.population = [strong, weak]
        ea.best_individual = strong
        ea.evolve_generation()
        self.assertEqual(len(ea.population), 2)
        for ind in ea.population:
            self.assertIn(ind.code, ("strong", "weak"))


class PromptFeedbackTests(unittest.TestCase):
    def test_failure_context_reaches_next_mutation_and_is_inherited(self):
        ea_module = load_evolutionary_algorithm()

        class FailingExecutor:
            def evaluate(self, code, timeout=None):
                return SimpleNamespace(
                    fitness=0.0,
                    success=False,
                    speedup=0.0,
                    execution_time=0.0,
                    error="runtime error",
                )

        class RecordingGeneticOps(FakeGeneticOps):
            def __init__(self):
                super().__init__()
                self.seen_contexts = []

            def mutate(self, individual):
                context = individual.metadata.get("prompt_context")
                self.seen_contexts.append(context)
                child = super().mutate(individual)
                if context is not None:
                    child.metadata["prompt_context"] = dict(context)
                return child

        genetic_ops = RecordingGeneticOps()
        ea = ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            FailingExecutor(),
            make_config(
                population_size=1,
                max_generations=1,
                mutation_rate=1.0,
            ),
        )
        ea.run(["seed code"])

        self.assertEqual(len(genetic_ops.seen_contexts), 1)
        self.assertEqual(
            dict(genetic_ops.seen_contexts[0]["failure_category_counts"]),
            {"runtime_error": 1},
        )
        self.assertEqual(genetic_ops.seen_contexts[0]["evaluation_count"], 1)

        class CapturingLlm:
            current_model = "deepseek-v4-pro"
            requires_prompt_brace_escaping = False

            def __init__(self):
                self.calls = []

            def generate(self, prompt, **kwargs):
                self.calls.append({"prompt": prompt, **kwargs})
                return "def kernel(x):\n    return x\n"

        llm = CapturingLlm()
        operators = GENETIC_OPS.GeneticOperators(
            llm, SimpleNamespace(llm_models=[llm.current_model])
        )
        parent = Individual(
            code="def kernel(x):\n    return x\n",
            metadata={"prompt_context": genetic_ops.seen_contexts[0]},
        )
        child = operators.mutate(parent)

        self.assertEqual(child.metadata["prompt_context"], parent.metadata["prompt_context"])
        self.assertIn("Failure Category Counts: runtime_error=1", llm.calls[0]["prompt"])


class EvolutionSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.ea_module = load_evolutionary_algorithm()

    def test_mu_plus_lambda_preserves_strong_parent(self):
        ea = self.ea_module.EvolutionaryAlgorithm(
            FakeGeneticOps(), FakeExecutor(fitness=0.5), make_config()
        )
        strong = Individual(code="strong", fitness=1.5, metadata={"evaluated": True})
        weak = Individual(code="weak", fitness=0.2, metadata={"evaluated": True})
        ea.population = [strong, weak]
        ea.best_individual = strong

        ea.evolve_generation()

        self.assertIn(strong, ea.population)
        self.assertEqual(len(ea.population), 2)

    def test_select_parents_with_one_positive_fitness_returns_distinct_pair(self):
        ea = self.ea_module.EvolutionaryAlgorithm(
            FakeGeneticOps(), FakeExecutor(), make_config(population_size=3)
        )
        ea.population = [
            Individual(code="positive", fitness=1.0),
            Individual(code="zero-a"),
            Individual(code="zero-b"),
        ]

        parent1, parent2 = ea.select_parents()

        self.assertNotEqual(parent1.id, parent2.id)

    def test_clone_path_creates_independent_individual(self):
        ea = self.ea_module.EvolutionaryAlgorithm(
            FakeGeneticOps(), FakeExecutor(), make_config()
        )
        better = Individual(code="better", fitness=0.9, model_used="model-a")
        other = Individual(code="other", fitness=0.1)

        child = ea._breed_one((better, other))
        child.code = "changed"

        self.assertNotIn(child.id, {better.id, other.id})
        self.assertEqual(better.code, "better")
        self.assertEqual(child.metadata["operation"], "clone")
        self.assertEqual(child.model_used, "model-a")

    def test_crossover_then_mutation_increments_generation_once(self):
        ea = self.ea_module.EvolutionaryAlgorithm(
            FakeGeneticOps(),
            FakeExecutor(),
            make_config(crossover_rate=1.0, mutation_rate=1.0),
        )
        ea.generation = 4
        parents = (Individual(code="a", generation=4), Individual(code="b", generation=4))

        child = ea._breed_one(parents)

        self.assertEqual(child.generation, 5)

    def test_mutated_crossover_child_retains_both_parent_ids(self):
        ea = self.ea_module.EvolutionaryAlgorithm(
            FakeGeneticOps(),
            FakeExecutor(),
            make_config(crossover_rate=1.0, mutation_rate=1.0),
        )
        parent1 = Individual(code="a")
        parent2 = Individual(code="b")

        child = ea._breed_one((parent1, parent2))
        ancestry = {child.metadata["parent"], *child.metadata["lineage"]}

        self.assertTrue({parent1.id, parent2.id}.issubset(ancestry))


class ConfigDefaultsTests(unittest.TestCase):
    def test_default_population_size_is_ten(self):
        module = _load_module("official_runtime_config", "config.py", {})
        self.assertEqual(module.EAConfig().population_size, 10)

    def test_default_baseline_path_is_anchored_to_agent_directory(self):
        module = _load_module("official_runtime_config_path", "config.py", {})
        expected = AGENT_DIR / "baseline" / "baseline.json"
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                config = module.EAConfig()
            finally:
                os.chdir(previous)
        self.assertEqual(Path(config.baseline_json), expected)
        self.assertTrue(Path(config.baseline_json).is_file())


class MainResultSemanticsTests(unittest.TestCase):
    def test_functional_success_is_independent_of_zero_fitness(self):
        module = load_main()
        result = {
            "best_fitness": 0.0,
            "top5_codes": [{"evaluation_status": {"success": True}}],
        }

        self.assertTrue(module._functional_success(result))

    def test_functional_success_fails_closed_without_explicit_success(self):
        module = load_main()
        for result in (
            {"best_fitness": 1.0, "top5_codes": []},
            {"best_fitness": 1.0, "top5_codes": [{}]},
            {
                "best_fitness": 1.0,
                "top5_codes": [{"evaluation_status": {"success": False}}],
            },
        ):
            with self.subTest(result=result):
                self.assertFalse(module._functional_success(result))

    def test_batch_exit_requires_every_kernel_to_succeed(self):
        module = load_main()

        class Config:
            debug = False
            llm_models = ["deepseek-v4-pro"]
            model_switch_prob = 0.2

            @staticmethod
            def print_config():
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            for kernel in ("op_a", "op_b"):
                kernel_dir = input_dir / kernel
                kernel_dir.mkdir(parents=True)
                (kernel_dir / f"{kernel}.py").write_text("x = 1\n", encoding="utf-8")
            args = SimpleNamespace(
                debug=False,
                input_dir=str(input_dir),
                output_dir=str(root / "output"),
                population_size=None,
                max_generations=None,
                kernel=None,
            )

            for outcomes, expected_code in (([True, True], 0), ([True, False], 1)):
                with self.subTest(outcomes=outcomes):
                    with (
                        patch.object(module, "parse_args", return_value=args),
                        patch.object(module, "EAConfig", return_value=Config()),
                        patch.object(
                            module, "optimize_single_kernel", side_effect=outcomes
                        ),
                        patch("builtins.print"),
                        self.assertRaises(SystemExit) as raised,
                    ):
                        module.main()
                    self.assertEqual(raised.exception.code, expected_code)


class TopKTests(unittest.TestCase):
    def test_top_k_dedupes_and_excludes_baseline(self):
        module = load_optimizer_agent()
        agent = module.TritonOptimizerAgent.__new__(module.TritonOptimizerAgent)
        agent._baseline_code = "baseline"
        agent.ea = SimpleNamespace(
            population=[
                SimpleNamespace(code="opt1", fitness=1.5, metadata={"success": True}),
                SimpleNamespace(code="baseline", fitness=1.4, metadata={"success": True}),
                SimpleNamespace(code="opt1\n", fitness=1.3, metadata={"success": True}),
                SimpleNamespace(code="opt2", fitness=1.2, metadata={"success": True}),
                SimpleNamespace(code=" baseline ", fitness=1.1, metadata={"success": True}),
                SimpleNamespace(code="opt2", fitness=1.0, metadata={"success": True}),
            ]
        )
        top = agent._get_top_k(5)
        self.assertEqual([ind.code for ind in top], ["opt1", "opt2"])

    def test_top_k_excludes_failures_but_keeps_successful_zero_fitness(self):
        module = load_optimizer_agent()
        agent = module.TritonOptimizerAgent.__new__(module.TritonOptimizerAgent)
        agent._baseline_code = "baseline"
        agent.ea = SimpleNamespace(
            population=[
                SimpleNamespace(
                    code="failed", fitness=2.0, metadata={"success": False}
                ),
                SimpleNamespace(code="unknown", fitness=1.0, metadata={}),
                SimpleNamespace(
                    code="zero", fitness=0.0, metadata={"success": True}
                ),
            ]
        )

        top = agent._get_top_k(5)

        self.assertEqual([ind.code for ind in top], ["zero"])

    def test_optimize_reports_first_exportable_candidate_as_best(self):
        module = load_optimizer_agent()
        agent = module.TritonOptimizerAgent.__new__(module.TritonOptimizerAgent)
        failed = SimpleNamespace(
            code="failed", fitness=1.0, metadata={"success": False}, id="bad"
        )
        zero = SimpleNamespace(
            code="zero",
            fitness=0.0,
            metadata={"success": True, "speedup": 0.0},
            id="good",
        )
        agent.ea = SimpleNamespace(
            run=lambda seeds, deadline_seconds=None: failed,
            generation=1,
            population=[failed, zero],
        )
        agent.llm = SimpleNamespace(get_stats=lambda: {"call_count": 0})
        agent.optimization_history = []

        result = agent.optimize(["baseline"], max_time=1)

        self.assertEqual(result["best_code"], "zero")
        self.assertEqual(result["best_provenance"]["id"], "good")
        self.assertEqual(result["top5_codes"][0]["code"], "zero")

    def test_optimize_passes_max_time_as_deadline(self):
        module = load_optimizer_agent()
        agent = module.TritonOptimizerAgent.__new__(module.TritonOptimizerAgent)
        best = SimpleNamespace(code="best", fitness=1.0, metadata={}, generation=0, id="x")
        run_kwargs = {}

        def fake_run(seeds, deadline_seconds=None):
            run_kwargs["deadline_seconds"] = deadline_seconds
            return best

        agent.ea = SimpleNamespace(run=fake_run, generation=0, population=[best])
        agent.llm = SimpleNamespace(get_stats=lambda: {"call_count": 0})
        agent.optimization_history = []
        result = agent.optimize(["baseline"], max_time=42)
        self.assertEqual(run_kwargs["deadline_seconds"], 42)
        self.assertEqual(result["best_code"], "best")


class ProvenanceTests(unittest.TestCase):
    @staticmethod
    def _case_evidence():
        facts = {
            "python_version": "3.11.15",
            "machine": "aarch64",
            "system": "Linux",
            "release": "test",
            "packages": {
                "torch": "2.7.1",
                "torch-npu": "2.7.1.post4",
                "triton": "3.2.0",
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                facts, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 2,
            "kind": "multi-case-real-evaluation",
            "official_aggregate": False,
            "aggregation": {
                "execution_time_us": "sum-completed-case-time-v1",
                "speedup": "minimum-successful-case-speedup-v1",
                "fitness": "minimum-successful-case-fitness-v1",
            },
            "case_results": [
                {
                    "case_id": case_id,
                    "status": "passed",
                    "success": True,
                    "baseline_time_us": execution_time * (1.0 + speedup),
                    "execution_time_us": execution_time,
                    "speedup": speedup,
                    "fitness": fitness,
                    "test_file": f"test_kernel_{case_id}.py",
                    "test_sha256": "a" * 64,
                    "baseline_code_sha256": "b" * 64,
                    "profile": {
                        "schema_version": 1,
                        "kind": "msprof-op-observation",
                        "path_base": "executor_work_dir",
                        "run_directory_id": f"run-case-{case_id}",
                        "csv_path": (
                            f"performance/kernel/run-case-{case_id}/"
                            f"OPPROF_case_{case_id}/OpBasicInfo.csv"
                        ),
                        "csv_sha256": "c" * 64,
                        "parser_rule": (
                            "op-basic-info:first-exact-op-name:"
                            "task-duration-us:v1"
                        ),
                        "parse_status": "parsed",
                        "kernel_name": "kernel",
                        "target_row_index": 1,
                        "execution_time_us": execution_time,
                        "toolchain_fingerprint": {
                            "facts": facts,
                            "sha256": fingerprint,
                        },
                    },
                }
                for case_id, execution_time, speedup, fitness in (
                    (1, 5.0, 0.5, 0.5),
                    (2, 7.5, 0.4, 0.4),
                )
            ],
        }

    def test_optimize_and_save_results_preserve_candidate_provenance(self):
        module = load_optimizer_agent()
        agent = module.TritonOptimizerAgent.__new__(module.TritonOptimizerAgent)
        case_evidence = self._case_evidence()
        candidate = Individual(
            code="candidate code",
            fitness=0.4,
            generation=2,
            id="candidate-1",
            model_used="deepseek-v4-pro",
            metadata={
                "parent": "parent-1",
                "lineage": ["seed-1"],
                "operation": "mutation",
                "mutation_type": "local_rewrite",
                "prompt_id": "prompt-1",
                "evaluated": True,
                "success": True,
                "speedup": 0.4,
                "execution_time": 12.5,
                "evaluation_evidence": case_evidence,
                "error": "private raw error must not be serialized",
                "prompt_context": {
                    "failure_category_counts": [("runtime_error", 1)],
                },
            },
        )
        agent._baseline_code = "baseline"
        agent.ea = SimpleNamespace(
            population=[candidate],
            generation=2,
            run=lambda seeds, deadline_seconds=None: candidate,
        )
        agent.llm = SimpleNamespace(get_stats=lambda: {"call_count": 1})
        agent.budget = None
        agent._budget_run_started = False
        agent.optimization_history = []

        result = agent.optimize(["baseline"], max_time=1)
        item = result["top5_codes"][0]
        self.assertEqual(item["id"], "candidate-1")
        self.assertEqual(len(item["code_hash"]), 64)
        self.assertEqual(item["parent_ids"], ["parent-1"])
        self.assertEqual(item["lineage"], ["seed-1"])
        self.assertEqual(item["mutation_kind"], "local_rewrite")
        self.assertEqual(item["model_used"], "deepseek-v4-pro")
        self.assertEqual(item["evaluation_status"]["success"], True)
        self.assertEqual(item["evaluation_evidence"], case_evidence)
        self.assertNotIn("error", item)

        with tempfile.TemporaryDirectory() as tmp:
            agent.save_results(tmp, "demo")
            stats = json.loads((Path(tmp) / "demo_stats.json").read_text(encoding="utf-8"))
        saved = stats["top5_summary"][0]
        self.assertEqual(saved["code_hash"], item["code_hash"])
        self.assertEqual(saved["parent_ids"], ["parent-1"])
        self.assertEqual(saved["evaluation_evidence"], case_evidence)
        self.assertNotIn("private raw error", json.dumps(stats))

    def test_candidate_provenance_accepts_single_case_v2_and_legacy_v1(self):
        module = load_optimizer_agent()
        evidence = self._case_evidence()
        evidence["case_results"] = evidence["case_results"][:1]
        individual = SimpleNamespace(
            code="candidate",
            fitness=0.5,
            generation=1,
            id="candidate",
            model_used="deepseek-v4-pro",
            metadata={
                "evaluated": True,
                "success": True,
                "speedup": 0.5,
                "execution_time": 5.0,
                "evaluation_evidence": evidence,
            },
        )
        provenance = module.TritonOptimizerAgent._candidate_provenance(individual)
        self.assertEqual(provenance["evaluation_evidence"], evidence)

        legacy = self._case_evidence()
        legacy["schema_version"] = 1
        for case in legacy["case_results"]:
            case.pop("profile")
        individual.fitness = 0.4
        individual.metadata.update({
            "speedup": 0.4,
            "execution_time": 12.5,
            "evaluation_evidence": legacy,
        })
        legacy_provenance = module.TritonOptimizerAgent._candidate_provenance(individual)
        self.assertEqual(legacy_provenance["evaluation_evidence"], legacy)

    def test_candidate_provenance_rejects_untrusted_evaluation_evidence(self):
        module = load_optimizer_agent()
        individual = SimpleNamespace(
            code="candidate",
            fitness=0.4,
            generation=1,
            id="candidate",
            model_used="deepseek-v4-pro",
            metadata={
                "evaluated": True,
                "success": True,
                "speedup": 0.4,
                "execution_time": 12.5,
                "evaluation_evidence": {
                    "schema_version": 1,
                    "kind": "multi-case-real-evaluation",
                    "official_aggregate": False,
                    "aggregation": {},
                    "case_results": [],
                    "raw_error": "must not escape",
                },
            },
        )

        provenance = module.TritonOptimizerAgent._candidate_provenance(individual)

        self.assertNotIn("evaluation_evidence", provenance)
        self.assertNotIn("must not escape", json.dumps(provenance))

        individual.metadata["evaluation_evidence"] = self._case_evidence()
        individual.metadata["execution_time"] = 12.0
        inconsistent = module.TritonOptimizerAgent._candidate_provenance(individual)
        self.assertNotIn("evaluation_evidence", inconsistent)

        individual.metadata["execution_time"] = 12.5
        for label, mutate in (
            ("boolean schema", lambda evidence: evidence.update(schema_version=True)),
            (
                "boolean case id",
                lambda evidence: evidence["case_results"][0].update(case_id=True),
            ),
            (
                "absolute profile path",
                lambda evidence: evidence["case_results"][0]["profile"].update(
                    csv_path="/tmp/OpBasicInfo.csv"
                ),
            ),
            (
                "profile time mismatch",
                lambda evidence: evidence["case_results"][0]["profile"].update(
                    execution_time_us=4.0
                ),
            ),
            (
                "fingerprint mismatch",
                lambda evidence: evidence["case_results"][0]["profile"]
                ["toolchain_fingerprint"].update(sha256="d" * 64),
            ),
            (
                "unknown profile field",
                lambda evidence: evidence["case_results"][0]["profile"].update(
                    raw_log="must not escape"
                ),
            ),
            (
                "speedup formula mismatch",
                lambda evidence: evidence["case_results"][0].update(speedup=0.6),
            ),
        ):
            with self.subTest(label=label):
                evidence = self._case_evidence()
                mutate(evidence)
                individual.metadata["evaluation_evidence"] = evidence
                rejected = module.TritonOptimizerAgent._candidate_provenance(individual)
                self.assertNotIn("evaluation_evidence", rejected)

    def test_save_results_without_candidate_fails_before_writing(self):
        module = load_optimizer_agent()
        agent = module.TritonOptimizerAgent.__new__(module.TritonOptimizerAgent)
        agent.optimization_history = [{
            "best_code": "baseline",
            "best_fitness": 0.0,
            "speedup": 0.0,
            "generations": 0,
            "time_elapsed": 0.0,
            "llm_stats": {"call_count": 0},
            "top5_codes": [],
            "best_provenance": {},
        }]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "No non-baseline candidate"):
                agent.save_results(tmp, "demo")
            self.assertEqual(list(Path(tmp).iterdir()), [])


class CleanCodeTests(unittest.TestCase):
    RAW = "```python\nx = (1.0 +\n    2.5 * y)\n```\n"

    def test_genetic_operators_clean_code_preserves_numeric_lines(self):
        ops = GENETIC_OPS.GeneticOperators(
            SimpleNamespace(current_model="deepseek-v4-pro"),
            SimpleNamespace(llm_models=["deepseek-v4-pro"]),
        )
        cleaned = ops._clean_code(self.RAW)
        self.assertIn("2.5 * y)", cleaned)
        self.assertNotIn("```", cleaned)

    def test_llm_interface_clean_code_preserves_numeric_lines(self):
        module = load_llm_interface()
        llm = module.LLMInterface.__new__(module.LLMInterface)
        cleaned = llm._clean_code_markers(self.RAW)
        self.assertIn("2.5 * y)", cleaned)
        self.assertNotIn("```", cleaned)


class OfficialLlmAllowlistTests(unittest.TestCase):
    def test_init_rejects_any_non_allowlisted_configured_model_before_client(self):
        module = load_llm_interface()
        config = SimpleNamespace(
            budget_controller=None,
            llm_models=["deepseek-v4-pro", "deepseek-r1"],
        )

        with patch.object(module.LLMInterface, "_init_llm") as init_llm:
            with self.assertRaisesRegex(ValueError, "official competition allowlist"):
                module.LLMInterface(config)

        init_llm.assert_not_called()

    def test_explicit_initial_model_and_switch_target_fail_closed(self):
        module = load_llm_interface()
        config = SimpleNamespace(
            budget_controller=None,
            llm_models=["deepseek-v4-pro"],
        )

        with patch.object(module.LLMInterface, "_init_llm") as init_llm:
            with self.assertRaisesRegex(ValueError, "official competition allowlist"):
                module.LLMInterface(config, model_name="not-an-official-model")
            llm = module.LLMInterface(config)
            with self.assertRaisesRegex(ValueError, "official competition allowlist"):
                llm.switch_model("deepseek-r1")

        self.assertEqual(llm.current_model, "deepseek-v4-pro")
        self.assertEqual(init_llm.call_count, 1)


class TokenAccountingTests(unittest.TestCase):
    def _make_interface(self, response, budget=None, max_tokens=128):
        module = load_llm_interface()
        llm = module.LLMInterface.__new__(module.LLMInterface)
        llm.config = SimpleNamespace(max_llm_tokens=max_tokens)
        llm.current_model = "deepseek-v4-pro"
        llm.budget_controller = budget
        llm.total_tokens_used = budget.snapshot().used_tokens if budget else 0
        llm.call_count = 0
        llm.call_history = []
        llm.llm = SimpleNamespace(invoke=lambda _: response)
        return llm

    def test_generate_uses_real_usage_metadata(self):
        response = SimpleNamespace(content="x = 1", usage_metadata={"total_tokens": 1234})
        llm = self._make_interface(response)
        llm.generate("prompt words", system_msg="system words")
        self.assertEqual(llm.total_tokens_used, 1234)

    def test_generate_estimate_fallback_includes_system_msg(self):
        response = SimpleNamespace(content="x y z")
        llm = self._make_interface(response)
        llm.generate("a b", system_msg="s1 s2 s3")
        # 3 (system) + 2 (prompt) + 3 (completion) words
        self.assertEqual(llm.total_tokens_used, 8)


class OfficialLlmBudgetTests(unittest.TestCase):
    def _make_interface(self, response, budget, max_tokens=128):
        return TokenAccountingTests._make_interface(
            self, response, budget, max_tokens
        )

    def test_init_accepts_injected_budget_and_restored_usage(self):
        budget = BudgetController(BudgetLimits(10_000, 300))
        reservation = budget.reserve(1, 9, 0, 1).reservation
        budget.commit(reservation, 7)
        module = load_llm_interface()
        config = SimpleNamespace(
            budget_controller=budget,
            llm_models=["deepseek-v4-pro"],
        )

        with patch.object(module.LLMInterface, "_init_llm"):
            llm = module.LLMInterface(config)

        self.assertIs(llm.budget_controller, budget)
        self.assertEqual(llm.total_tokens_used, 7)

    def test_llm_client_does_not_disable_tls_verification(self):
        module = load_llm_interface()
        captured = {}

        def fake_client(**kwargs):
            captured.update(kwargs)
            return object()

        config = SimpleNamespace(
            budget_controller=None,
            llm_models=["deepseek-v4-pro"],
            api_url="https://api.deepseek.com/v1",
            api_key="test-key",
            llm_temperature=0.2,
            max_llm_tokens=128,
        )
        with patch.object(module.httpx, "Client", side_effect=fake_client):
            with patch.object(module, "ChatOpenAI", return_value=object()):
                module.LLMInterface(config)

        self.assertNotEqual(captured.get("verify"), False)

    def test_success_commits_real_usage(self):
        budget = BudgetController(BudgetLimits(10_000, 300))
        response = SimpleNamespace(content="x = 1", usage_metadata={"total_tokens": 42})
        llm = self._make_interface(response, budget)

        llm.generate("prompt", system_msg="system")

        snapshot = budget.snapshot()
        self.assertEqual(snapshot.used_tokens, 42)
        self.assertEqual(snapshot.in_flight_calls, 0)
        self.assertEqual(llm.total_tokens_used, 42)

    def test_missing_usage_commits_full_reservation(self):
        budget = BudgetController(BudgetLimits(10_000, 300))
        llm = self._make_interface(SimpleNamespace(content="x = 1"), budget)

        llm.generate("prompt", system_msg="system")

        snapshot = budget.snapshot()
        self.assertGreater(snapshot.used_tokens, llm.config.max_llm_tokens)
        self.assertEqual(snapshot.used_tokens, llm.call_history[-1]["tokens"])
        self.assertEqual(snapshot.in_flight_calls, 0)

    def test_budget_denial_skips_invoke(self):
        budget = BudgetController(BudgetLimits(10, 300))
        llm = self._make_interface(SimpleNamespace(content="unused"), budget)

        def unexpected_invoke(_):
            self.fail("invoke must not run after budget denial")

        llm.llm = SimpleNamespace(invoke=unexpected_invoke)
        with self.assertRaisesRegex(RuntimeError, "budget denied"):
            llm.generate("prompt")

        self.assertEqual(budget.snapshot().used_tokens, 0)
        self.assertEqual(budget.snapshot().in_flight_calls, 0)

    def test_failed_inflight_call_is_marked_uncertain(self):
        budget = BudgetController(BudgetLimits(10_000, 300))
        llm = self._make_interface(SimpleNamespace(content="unused"), budget)

        def timeout(_):
            raise TimeoutError("timed out")

        llm.llm = SimpleNamespace(invoke=timeout)
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            llm.generate("prompt")

        snapshot = budget.snapshot()
        self.assertEqual(snapshot.stop_reason, "unknown_in_flight_call")
        self.assertEqual(snapshot.in_flight_calls, 0)
        self.assertGreater(snapshot.used_tokens, llm.config.max_llm_tokens)


class IndividualIdTests(unittest.TestCase):
    def test_ids_are_unique_and_12_hex_chars(self):
        ids = [Individual(code="c").id for _ in range(200)]
        self.assertEqual(len(set(ids)), 200)
        for ind_id in ids:
            self.assertEqual(len(ind_id), 12)
            int(ind_id, 16)  # valid hex


class P0ResidualTests(unittest.TestCase):
    def setUp(self):
        self.ea_module = load_evolutionary_algorithm()

    def test_consecutive_mutation_lineage(self):
        genetic_ops = FakeGeneticOps()
        ancestor = Individual(code="a")

        child = genetic_ops.mutate(ancestor)
        grandchild = genetic_ops.mutate(child)
        ancestry = {
            grandchild.metadata["parent"],
            *grandchild.metadata.get("lineage", []),
        }

        self.assertIn(ancestor.id, ancestry)

    def test_breed_one_budget_stops_before_mutation(self):
        class BudgetExhaustingGeneticOps(FakeGeneticOps):
            def crossover(self, a, b):
                child = super().crossover(a, b)
                self.llm.total_tokens_used = 101
                return child

        genetic_ops = BudgetExhaustingGeneticOps()
        ea = self.ea_module.EvolutionaryAlgorithm(
            genetic_ops,
            FakeExecutor(),
            make_config(
                crossover_rate=1.0,
                mutation_rate=1.0,
                max_total_tokens=100,
            ),
        )

        child = ea._breed_one((Individual(code="a"), Individual(code="b")))

        self.assertNotEqual(child.metadata.get("operation"), "mutation")

    def test_initialize_population_respects_budget(self):
        ea = self.ea_module.EvolutionaryAlgorithm(
            FakeGeneticOps(),
            FakeExecutor(),
            make_config(population_size=5),
        )
        ea._deadline = time.time() - 1

        ea.initialize_population(["seed"])

        self.assertLess(len(ea.population), 5)


def load_budgeted_optimizer_agent():
    class FakeLlm:
        def __init__(self, config):
            self.budget_controller = config.budget_controller
            self.total_tokens_used = self.budget_controller.snapshot().used_tokens
            self.call_count = 0

        def get_stats(self):
            return {
                "call_count": self.call_count,
                "total_tokens": self.total_tokens_used,
            }

    class FakeContractExecutor:
        def __init__(self, **kwargs):
            pass

    class FakeGeneticOperators:
        def __init__(self, llm, config):
            self.llm = llm

    class FakeEvolutionaryAlgorithm:
        def __init__(self, genetic_ops, executor, config):
            self.llm = genetic_ops.llm
            self.generation = 0
            self.population = []
            self.observed_tokens = None

        def run(self, seed_codes, deadline_seconds=None):
            self.observed_tokens = self.llm.total_tokens_used
            best = SimpleNamespace(
                code=seed_codes[0],
                fitness=0.0,
                metadata={},
                generation=0,
                id="seed",
            )
            self.population = [best]
            return best

    return _load_module(
        "official_runtime_budgeted_optimizer_agent",
        "optimizer_agent.py",
        {
            "config": _stub("config", EAConfig=type("EAConfig", (), {})),
            "llm_interface": _stub("llm_interface", LLMInterface=FakeLlm),
            "contract_executor": _stub(
                "contract_executor",
                ContractCheckingExecutor=FakeContractExecutor,
                MultiCaseContractExecutor=lambda cases: cases[0][1],
            ),
            "executor": _stub(
                "executor", TritonExecutor=type("TritonExecutor", (), {})
            ),
            "genetic_operators": _stub(
                "genetic_operators", GeneticOperators=FakeGeneticOperators
            ),
            "evolutionary_algorithm": _stub(
                "evolutionary_algorithm",
                EvolutionaryAlgorithm=FakeEvolutionaryAlgorithm,
            ),
        },
    )


class OptimizerAgentBudgetTests(unittest.TestCase):
    def _make_agent(self):
        module = load_budgeted_optimizer_agent()
        config = SimpleNamespace(
            max_total_tokens=100,
            baseline_json="unused.json",
        )
        agent = module.TritonOptimizerAgent(config)
        with tempfile.TemporaryDirectory() as tmp:
            test_path = Path(tmp) / "test_kernel.py"
            test_path.write_text("def test_kernel():\n    pass\n", encoding="utf-8")
            with patch.object(module, "get_baseline_from_json", return_value=1.0):
                agent.setup(
                    "baseline",
                    str(test_path),
                    kernel_name="kernel",
                    work_dir=tmp,
                )
        return agent

    def test_optimizer_agent_injects_budget_into_llm(self):
        agent = self._make_agent()
        self.assertIs(agent.llm.budget_controller, agent.budget)

    def test_optimizer_agent_budget_stops_optimize(self):
        agent = self._make_agent()
        reservation = agent.budget.reserve(1, 1, 0, 0, use_reserve=True).reservation
        agent.budget.commit(reservation, 101)

        result = agent.optimize(["seed"], max_time=42)

        self.assertEqual(result["generations"], 0)
        self.assertGreaterEqual(agent.ea.observed_tokens, 100)
        self.assertLessEqual(agent.budget.snapshot().remaining_seconds, 42)

    def test_optimizer_agent_starts_wall_budget_on_first_optimize(self):
        agent = self._make_agent()
        state = agent.budget.snapshot()
        state["elapsed_seconds"] = agent.budget.limits.wall_time_seconds
        state["stop_reason"] = "wall_time_limit"
        agent.budget.restore(state)

        agent.optimize(["seed"], max_time=42)

        snapshot = agent.budget.snapshot()
        self.assertIsNone(snapshot.stop_reason)
        self.assertGreater(snapshot.remaining_seconds, 40)
        self.assertLessEqual(snapshot.remaining_seconds, 42)

    def test_optimizer_agent_does_not_restart_wall_budget(self):
        agent = self._make_agent()
        agent.optimize(["seed"], max_time=42)
        state = agent.budget.snapshot()
        state["elapsed_seconds"] = agent.budget.limits.wall_time_seconds - 10
        agent.budget.restore(state)

        agent.optimize(["seed"], max_time=42)

        self.assertLessEqual(agent.budget.snapshot().remaining_seconds, 10)

    def test_optimizer_agent_preserves_unknown_inflight_stop(self):
        agent = self._make_agent()
        reservation = agent.budget.reserve(1, 1, 0, 0, use_reserve=True).reservation
        agent.budget.mark_uncertain(reservation)

        agent.optimize(["seed"], max_time=42)

        self.assertEqual(
            agent.budget.snapshot().stop_reason, "unknown_in_flight_call"
        )

    def test_exhausted_budget_stops_real_ea_end_to_end(self):
        agent = self._make_agent()
        agent.config.population_size = 5
        agent.config.max_generations = 3
        agent.config.timeout_seconds = 1
        calls = {"mutation": 0, "crossover": 0, "evaluation": 0}

        class GuardedGeneticOps:
            def __init__(self, llm):
                self.llm = llm

            def mutate(self, individual):
                calls["mutation"] += 1
                return individual

            def crossover(self, first, second):
                calls["crossover"] += 1
                return first

        class GuardedExecutor:
            def evaluate(self, code, timeout=None):
                calls["evaluation"] += 1
                return SimpleNamespace(
                    fitness=1.0,
                    success=True,
                    speedup=1.0,
                    execution_time=0.0,
                    error=None,
                )

        ea_module = load_evolutionary_algorithm()
        agent.ea = ea_module.EvolutionaryAlgorithm(
            GuardedGeneticOps(agent.llm), GuardedExecutor(), agent.config
        )
        reservation = agent.budget.reserve(1, 1, 0, 0, use_reserve=True).reservation
        agent.budget.commit(reservation, 101)

        result = agent.optimize(["seed"], max_time=42)

        self.assertEqual(result["best_code"], "seed")
        self.assertEqual(result["generations"], 0)
        self.assertEqual(result["top5_codes"], [])
        self.assertEqual(calls, {"mutation": 0, "crossover": 0, "evaluation": 0})


if __name__ == "__main__":
    unittest.main()
