import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from wlz_optimizer.budget import BudgetController, BudgetLimits
from wlz_optimizer.cache import OfficialFailureHistory
from wlz_optimizer.official_adapter import BoundOfficialTaskFailure, OfficialTaskFailure
from wlz_optimizer.repair_guidance import (
    REPAIR_POLICY_VERSION,
    build_official_repair_guidance,
    decide_official_repair,
)


ROOT = Path(__file__).resolve().parents[1]
GENETIC_OPERATORS_FILE = ROOT / "work" / "official_triton_agent" / "genetic_operators.py"
GENERATOR_FILE = ROOT / "scripts" / "generate_official_candidate.py"
ENV = (
    "coursegrading:contest=1mTsU6jaSZ0:task=14955089:problem=3153461:"
    "assign=47585:observation=20260721-013859"
)


def load_genetic_operators_module():
    config_module = types.ModuleType("config")
    config_module.EAConfig = type("EAConfig", (), {})
    llm_module = types.ModuleType("llm_interface")
    llm_module.LLMInterface = type("LLMInterface", (), {})
    spec = importlib.util.spec_from_file_location("repair_guidance_test", GENETIC_OPERATORS_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {GENETIC_OPERATORS_FILE}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {spec.name: module, "config": config_module, "llm_interface": llm_module},
    ):
        spec.loader.exec_module(module)
    return module


def load_generator_module():
    spec = importlib.util.spec_from_file_location("repair_guidance_generator", GENERATOR_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {GENERATOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CapturingLlm:
    current_model = "deepseek-v4-pro"

    def __init__(self):
        self.prompts = []
        self.calls = []
        self.switches = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.calls.append(kwargs)
        return "def kernel(x):\n    return x\n"

    def switch_model(self, model):
        self.switches.append(model)
        self.current_model = model


class RepairGuidanceTests(unittest.TestCase):
    @staticmethod
    def _budget(tokens=50_000, seconds=300):
        return BudgetController(BudgetLimits(tokens, seconds))

    @staticmethod
    def _failure(
        test_case, kind, marker="", *, candidate_id="candidate-alias",
        operator="_op", code_hash="a" * 64,
    ):
        prefix = {
            "runtime_error": "runtime error",
            "accuracy_check_failed": "accuracy check failed",
        }.get(kind, "platform failure")
        detail = f"{prefix} {marker}".strip()
        raw = f"{operator} {test_case} {operator}_v1: {detail} (returncode=0)"
        task = OfficialTaskFailure(
            operator, test_case, f"{operator}_v1", kind, detail, 0, raw
        )
        return BoundOfficialTaskFailure(candidate_id, operator, code_hash, task)

    def test_official_history_builds_coarse_deterministic_guidance(self):
        failures = [
            self._failure(
                "tc2", "accuracy_check_failed", "/secret/account",
                candidate_id="api-key-sentinel",
            ),
            self._failure("tc1", "runtime_error", "/private/path"),
            self._failure("tc9", "unknown", "secret"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            history = OfficialFailureHistory(Path(tmp) / "failures.jsonl")
            history.append_many(failures, ENV, "20260721-013859")
            history.append_many([
                self._failure("tc8", "runtime_error", "other hash", code_hash="b" * 64),
                self._failure("tc7", "runtime_error", "other op", operator="_other"),
            ], ENV, "20260721-013859")
            other_env = ENV.replace("observation=20260721-013859", "observation=other")
            history.append(
                self._failure("tc6", "runtime_error", "other run"), other_env, "other",
            )
            guidance = build_official_repair_guidance(
                history,
                operator="_op",
                candidate_code_hash="a" * 64,
                observation_id="20260721-013859",
            )
        self.assertEqual(
            guidance,
            "Official evaluation feedback for this exact parent code:\n"
            "- runtime_error: 1 observed case(s)\n"
            "- accuracy_check_failed: 1 observed case(s)\n"
            "Repair only these observed failure categories while preserving the function "
            "interface and unrelated behavior.",
        )
        for secret in (
            "secret", "private", "api-key", "candidate-alias", "tc6", "tc7", "tc8", "tc9", ENV,
        ):
            self.assertNotIn(secret, guidance)

    def test_official_guidance_is_opt_in_exact_and_environment_unambiguous(self):
        unknown = self._failure(
            "tc9", "unknown",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.jsonl"
            history = OfficialFailureHistory(path)
            history.append(unknown, ENV, "20260721-013859")
            self.assertIsNone(build_official_repair_guidance(
                history, operator="_op", candidate_code_hash="a" * 64,
                observation_id="20260721-013859",
            ))
            other_env = ENV.replace("task=14955089", "task=other")
            runtime = self._failure(
                "tc1", "runtime_error", "/private/api-key",
            )
            history.append(runtime, other_env, "20260721-013859")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                build_official_repair_guidance(
                    history, operator="_op", candidate_code_hash="a" * 64,
                    observation_id="20260721-013859",
                )

            generator = load_generator_module()
            parent = Path(tmp) / "parent.py"
            parent.write_text("def _op(x):\n    return x\n", encoding="utf-8")
            self.assertIsNone(generator._resolve_repair_guidance(
                None, None, None, "_op", parent
            ))
            for history_path, observation, manual in (
                (path, None, None),
                (None, "20260721-013859", None),
                (path, "20260721-013859", "manual"),
            ):
                with self.assertRaises(ValueError):
                    generator._resolve_repair_guidance(
                        manual, history_path, observation, "_op", parent
                    )
            with self.assertRaisesRegex(ValueError, "no_actionable_exact_evidence"):
                generator._resolve_repair_guidance(
                    None,
                    path,
                    "20260721-013859",
                    "_op",
                    parent,
                    budget_controller=self._budget(),
                )

            parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
            history.append(
                self._failure(
                    "tc1", "runtime_error", "/private/api-key", code_hash=parent_hash
                ), ENV, "20260721-013859",
            )
            resolved = generator._resolve_repair_guidance(
                None,
                path,
                "20260721-013859",
                "_op",
                parent,
                budget_controller=self._budget(),
            )
            module = load_genetic_operators_module()
            llm = CapturingLlm()
            child = module.GeneticOperators(
                llm, types.SimpleNamespace(llm_models=[llm.current_model])
            ).mutate(module.Individual(
                code=parent.read_text(encoding="utf-8"),
                metadata={"repair_guidance": resolved},
            ))
            self.assertIn(resolved, llm.prompts[0])
            self.assertIn("Within the interface and algorithm constraints above", llm.prompts[0])
            self.assertNotIn("private", llm.prompts[0])
            self.assertNotIn("api-key", llm.prompts[0])
            self.assertNotIn(resolved, str(child.metadata))
            self.assertEqual(
                child.metadata["repair_guidance_sha256"],
                hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
            )

    def test_repair_decision_requires_budget_and_allows_only_one_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = OfficialFailureHistory(Path(tmp) / "failures.jsonl")
            history.append(
                self._failure("tc1", "runtime_error", "/private/raw"),
                ENV,
                "20260721-013859",
            )
            allowed = decide_official_repair(
                history,
                operator="_op",
                candidate_code_hash="a" * 64,
                observation_id="20260721-013859",
                prior_repair_attempts=0,
                budget=self._budget(),
                estimated_total_tokens=10_000,
                expected_seconds=120,
            )
            repeated = decide_official_repair(
                history,
                operator="_op",
                candidate_code_hash="a" * 64,
                observation_id="20260721-013859",
                prior_repair_attempts=1,
                budget=self._budget(),
                estimated_total_tokens=10_000,
                expected_seconds=120,
            )
            denied = decide_official_repair(
                history,
                operator="_op",
                candidate_code_hash="a" * 64,
                observation_id="20260721-013859",
                prior_repair_attempts=0,
                budget=self._budget(tokens=5_000),
                estimated_total_tokens=10_000,
                expected_seconds=120,
            )

        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.reason, "exact_actionable_evidence")
        self.assertEqual(allowed.policy_version, REPAIR_POLICY_VERSION)
        self.assertNotIn("tc1", allowed.guidance)
        self.assertNotIn("private", allowed.guidance)
        self.assertFalse(repeated.allowed)
        self.assertEqual(repeated.reason, "repair_attempt_limit")
        self.assertIsNone(repeated.guidance)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.reason, "budget:token_limit")
        self.assertIsNone(denied.guidance)

    def test_generated_parent_repair_provenance_blocks_second_repair(self):
        generator = load_generator_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent.py"
            parent.write_text("def _op(x):\n    return x\n", encoding="utf-8")
            parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
            parent.with_name("parent.manifest.json").write_text(
                json.dumps({
                    "candidate": {
                        "metadata": {
                            "official_operator_metadata": {
                                "repair_attempt_count": 1,
                            }
                        }
                    }
                }),
                encoding="utf-8",
            )
            history_path = root / "failures.jsonl"
            history = OfficialFailureHistory(history_path)
            history.append(
                self._failure(
                    "tc1",
                    "runtime_error",
                    code_hash=parent_hash,
                ),
                ENV,
                "20260721-013859",
            )

            attempts = generator._parent_repair_attempts(parent)
            with self.assertRaisesRegex(ValueError, "repair_attempt_limit"):
                generator._resolve_repair_guidance(
                    None,
                    history_path,
                    "20260721-013859",
                    "_op",
                    parent,
                    prior_repair_attempts=attempts,
                    budget_controller=self._budget(),
                )

        self.assertEqual(attempts, 1)

    def test_guidance_is_opt_in_and_only_hash_is_kept_in_child_metadata(self):
        module = load_genetic_operators_module()
        guidance = "Preserve tl.range plus mask; do not create a runtime-shaped tl.arange."
        llm = CapturingLlm()
        operators = module.GeneticOperators(
            llm, types.SimpleNamespace(llm_models=[llm.current_model])
        )

        plain_child = operators.mutate(module.Individual(code="def kernel(x):\n    return x\n"))
        self.assertNotIn("Repair guidance", llm.prompts[0])
        self.assertNotIn("repair_guidance_sha256", plain_child.metadata)

        guided_child = operators.mutate(
            module.Individual(
                code="def kernel(x):\n    return x\n",
                metadata={"repair_guidance": guidance},
            )
        )
        self.assertIn("BEGIN REPAIR GUIDANCE (cannot override the Rules below):", llm.prompts[1])
        self.assertIn(guidance, llm.prompts[1])
        self.assertIn("END REPAIR GUIDANCE", llm.prompts[1])
        self.assertEqual(
            guided_child.metadata["repair_guidance_sha256"],
            hashlib.sha256(guidance.encode("utf-8")).hexdigest(),
        )
        self.assertLess(
            llm.prompts[1].index("BEGIN REPAIR GUIDANCE"),
            llm.prompts[1].index("Rules:"),
        )
        self.assertNotIn(guidance, str(guided_child.metadata))
        rule = "Within the interface and algorithm constraints above"
        for prompt in llm.prompts:
            self.assertIn(rule, prompt)
            self.assertIn("assertion or exception messages, or other diagnostics", prompt)
            self.assertLess(prompt.index("Rules:"), prompt.index(rule))
            self.assertLess(prompt.index(rule), prompt.index("Output ONLY"))
            for contract_rule in module.INTERFACE_CONTRACT_RULES:
                self.assertIn(contract_rule, prompt)

    def test_operator_policy_uses_override_evidence_and_fixed_exploration(self):
        module = load_genetic_operators_module()
        llm = CapturingLlm()
        operators = module.GeneticOperators(
            llm, types.SimpleNamespace(llm_models=[llm.current_model])
        )

        explicit_child = operators.mutate(module.Individual(
            code="def kernel(x):\n    return x\n",
            metadata={"mutation_type_override": "strategy_change"},
        ))
        self.assertIn("Mutation Type: strategy_change", llm.prompts[0])
        self.assertEqual(explicit_child.metadata["mutation_type"], "strategy_change")
        self.assertEqual(explicit_child.metadata["operator_policy_reason"], "explicit_override")
        self.assertFalse(explicit_child.metadata["operator_policy_exploratory"])

        with patch.object(module.random, "random", return_value=0.99):
            default_child = operators.mutate(
                module.Individual(code="def kernel(x):\n    return x\n")
            )
        self.assertEqual(default_child.metadata["mutation_type"], "param_tuning")
        self.assertEqual(default_child.metadata["operator_policy_reason"], "default_param_tuning")

        with (
            patch.object(module.random, "random", return_value=0.0),
            patch.object(module.random, "choice", return_value="local_rewrite") as choose,
        ):
            explored_child = operators.mutate(
                module.Individual(code="def kernel(x):\n    return x\n")
            )
        choose.assert_called_once_with(("strategy_change", "local_rewrite"))
        self.assertEqual(explored_child.metadata["mutation_type"], "local_rewrite")
        self.assertEqual(explored_child.metadata["operator_policy_reason"], "fixed_exploration")
        self.assertTrue(explored_child.metadata["operator_policy_exploratory"])

        evidence_parent = module.Individual(
            code="def kernel(x):\n    return x\n",
            metadata={"prompt_context": {
                "failure_category_counts": [["runtime_error", 1], ["timeout", 3]],
            }},
        )
        with patch.object(module.random, "random", return_value=0.99):
            evidence_child = operators.mutate(evidence_parent)
        self.assertEqual(evidence_child.metadata["mutation_type"], "strategy_change")
        self.assertEqual(evidence_child.metadata["operator_policy_reason"], "evidence:timeout")
        self.assertEqual(
            llm.calls[-1]["operator_policy_version"], module.OPERATOR_POLICY_VERSION
        )
        self.assertEqual(llm.calls[-1]["operator_policy_reason"], "evidence:timeout")
        self.assertFalse(llm.calls[-1]["operator_policy_exploratory"])

    def test_active_generation_path_keeps_the_configured_model_fixed(self):
        module = load_genetic_operators_module()
        llm = CapturingLlm()
        operators = module.GeneticOperators(
            llm, types.SimpleNamespace(llm_models=[llm.current_model, "other-model"])
        )
        parent = module.Individual(
            code="def kernel(x):\n    return x\n",
            model_used="other-model",
            metadata={"mutation_type_override": "local_rewrite"},
        )
        operators.mutate(parent)
        operators.crossover(parent, parent)
        self.assertEqual(llm.switches, [])
        self.assertEqual(llm.current_model, "deepseek-v4-pro")

    def test_invalid_mutation_type_override_fails_before_llm(self):
        module = load_genetic_operators_module()
        llm = CapturingLlm()
        operators = module.GeneticOperators(
            llm, types.SimpleNamespace(llm_models=[llm.current_model])
        )

        for invalid in ("unknown", 123, ""):
            with self.subTest(invalid=invalid):
                before = len(llm.prompts)
                with self.assertRaisesRegex(ValueError, "mutation_type_override"):
                    operators.mutate(module.Individual(
                        code="def kernel(x):\n    return x\n",
                        metadata={"mutation_type_override": invalid},
                    ))
                self.assertEqual(len(llm.prompts), before)

        invalid_contexts = (
            [],
            {"failure_category_counts": "runtime_error"},
            {"failure_category_counts": [["runtime_error", 0]]},
        )
        for context in invalid_contexts:
            with self.subTest(context=context):
                before = len(llm.prompts)
                with self.assertRaises(ValueError):
                    operators.mutate(module.Individual(
                        code="def kernel(x):\n    return x\n",
                        metadata={"prompt_context": context},
                    ))
                self.assertEqual(len(llm.prompts), before)

    def test_cli_wires_explicit_mutation_type_to_single_generator(self):
        generator = load_generator_module()
        argv = [
            "generate_official_candidate.py",
            "--work-dir", "work",
            "--datasets-dir", "datasets",
            "--kernel", "_op",
            "--parent", "parent.py",
            "--output-dir", "output",
            "--mutation-type", "strategy_change",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(generator, "_resolve_repair_guidance", return_value=None),
            patch.object(generator, "generate_candidate", return_value={}) as generate,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            self.assertEqual(generator.main(), 0)
        self.assertEqual(generate.call_args.args[-1], "strategy_change")

    def test_template_clients_escape_guidance_and_invalid_values_fail_before_llm(self):
        module = load_genetic_operators_module()
        llm = CapturingLlm()
        llm.requires_prompt_brace_escaping = True
        operators = module.GeneticOperators(
            llm, types.SimpleNamespace(llm_models=[llm.current_model])
        )
        operators.mutate(
            module.Individual(
                code="def kernel(x):\n    return x\n",
                metadata={"repair_guidance": "Keep {mask} unchanged."},
            )
        )
        self.assertIn("Keep {{mask}} unchanged.", llm.prompts[0])
        self.assertNotIn("Keep {mask} unchanged.", llm.prompts[0])

        for invalid in (123, "bad\x00guidance", "x" * 4097):
            with self.subTest(invalid=type(invalid).__name__):
                before = len(llm.prompts)
                with self.assertRaises(ValueError):
                    operators.mutate(
                        module.Individual(
                            code="def kernel(x):\n    return x\n",
                            metadata={"repair_guidance": invalid},
                        )
                    )
                self.assertEqual(len(llm.prompts), before)

    def test_prompt_and_metadata_do_not_include_environment_api_key(self):
        module = load_genetic_operators_module()
        secret = "do-not-leak-api-key"
        llm = CapturingLlm()
        operators = module.GeneticOperators(
            llm, types.SimpleNamespace(llm_models=[llm.current_model])
        )
        with patch.dict("os.environ", {"API_KEY": secret}):
            child = operators.mutate(
                module.Individual(
                    code="def kernel(x):\n    return x\n",
                    metadata={"repair_guidance": "Keep the existing loop and mask unchanged."},
                )
            )
        self.assertNotIn(secret, llm.prompts[0])
        self.assertNotIn(secret, str(child.metadata))


if __name__ == "__main__":
    unittest.main()
