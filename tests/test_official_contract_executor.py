import importlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_WORK = ROOT / "work" / "official_triton_agent"


def load_main_module():
    path = OFFICIAL_WORK / "main.py"
    spec = importlib.util.spec_from_file_location("official_multi_case_main_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    config = types.ModuleType("config")
    config.EAConfig = type("EAConfig", (), {})
    optimizer = types.ModuleType("optimizer_agent")
    optimizer.TritonOptimizerAgent = type("TritonOptimizerAgent", (), {})
    with patch.dict(
        sys.modules,
        {"config": config, "optimizer_agent": optimizer},
    ):
        spec.loader.exec_module(module)
    return module


class OfficialContractExecutorTests(unittest.TestCase):
    def test_loader_returns_all_contiguous_numbered_cases(self) -> None:
        module = load_main_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_id in (1, 2, 3):
                (root / f"test_kernel_{case_id}.py").write_text(
                    "pass\n", encoding="utf-8"
                )

            cases = module.find_test_files(root, "kernel")
            self.assertEqual([case_id for case_id, _ in cases], [1, 2, 3])

            (root / "test_kernel_2.py").unlink()
            with self.assertRaisesRegex(ValueError, "Non-contiguous"):
                module.find_test_files(root, "kernel")

            (root / "test_kernel_4.py").write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported test case"):
                module.find_test_files(root, "kernel")

    def test_multi_case_executor_requires_all_cases_and_uses_weakest_score(self) -> None:
        sys.path.insert(0, str(OFFICIAL_WORK))
        try:
            module = importlib.import_module("contract_executor")
        finally:
            sys.path.remove(str(OFFICIAL_WORK))

        class FakeExecutor:
            def __init__(self, result, baseline_time):
                self.result = result
                self.timeouts = []
                self.baseline_time = baseline_time
                self.baseline_code = "def kernel():\n    pass\n"
                self.test_code_path = Path(__file__)

            def evaluate(self, code, timeout=1200):
                self.timeouts.append(timeout)
                return self.result

        def profile(duration, run_id):
            return {
                "schema_version": 1,
                "kind": "msprof-op-observation",
                "path_base": "executor_work_dir",
                "run_directory_id": run_id,
                "csv_path": f"performance/kernel/{run_id}/OPPROF_x/OpBasicInfo.csv",
                "csv_sha256": "a" * 64,
                "parser_rule": "op-basic-info:first-exact-op-name:task-duration-us:v1",
                "parse_status": "parsed",
                "kernel_name": "kernel",
                "target_row_index": 1,
                "execution_time_us": duration,
                "toolchain_fingerprint": {
                    "facts": {
                        "python_version": "3.11.15",
                        "machine": "aarch64",
                        "system": "Linux",
                        "release": "test",
                        "packages": {
                            "torch": "2.7.1",
                            "torch-npu": "2.7.1.post4",
                            "triton": "3.2.0",
                        },
                    },
                    "sha256": "b" * 64,
                },
            }

        first = FakeExecutor(
            module.EvaluationResult(
                True, 2.0, 0.5, 0.5,
                evidence={"profile": profile(2.0, "run-one"), "raw": "drop"},
            ),
            3.0,
        )
        second = FakeExecutor(
            module.EvaluationResult(
                True, 3.0, 0.2, 0.2,
                evidence={"profile": profile(3.0, "run-two")},
            ),
            3.6,
        )
        executor = module.MultiCaseContractExecutor([(2, second), (1, first)])
        with patch.object(module.time, "monotonic", side_effect=[0.0, 1.0, 2.0]):
            result = executor.evaluate("candidate", timeout=10)

        self.assertTrue(result.success)
        self.assertEqual(result.execution_time, 5.0)
        self.assertEqual((result.speedup, result.fitness), (0.2, 0.2))
        self.assertEqual(first.timeouts, [9.0])
        self.assertEqual(second.timeouts, [8.0])
        self.assertFalse(result.evidence["official_aggregate"])
        self.assertEqual(result.evidence["schema_version"], 2)
        self.assertEqual(
            [item["case_id"] for item in result.evidence["case_results"]], [1, 2]
        )
        self.assertEqual(
            [item["baseline_time_us"] for item in result.evidence["case_results"]],
            [3.0, 3.6],
        )
        self.assertTrue(
            all(item["test_sha256"] for item in result.evidence["case_results"])
        )
        self.assertEqual(
            [item["profile"]["run_directory_id"] for item in result.evidence["case_results"]],
            ["run-one", "run-two"],
        )
        self.assertNotIn("raw", result.evidence["case_results"][0]["profile"])

        second.result = module.EvaluationResult(False, 0.0, 0.0, 0.0, "wrong")
        failed = executor.evaluate("candidate")
        self.assertFalse(failed.success)
        self.assertEqual(failed.execution_time, 2.0)
        self.assertIn("Test case 2 failed", failed.error)
        self.assertEqual(
            [item["status"] for item in failed.evidence["case_results"]],
            ["passed", "failed"],
        )
        self.assertIsNone(failed.evidence["case_results"][-1]["profile"])

        with patch.object(module.time, "monotonic", side_effect=[0.0, 11.0]):
            not_run = executor.evaluate("candidate", timeout=10)
        self.assertEqual(
            not_run.evidence["case_results"][0]["status"],
            "not_run_budget_exhausted",
        )
        self.assertIsNone(not_run.evidence["case_results"][0]["profile"])

    def test_ea_binds_evaluation_evidence_to_candidate(self) -> None:
        path = OFFICIAL_WORK / "evolutionary_algorithm.py"
        spec = importlib.util.spec_from_file_location("official_evidence_ea", path)
        if spec is None or spec.loader is None:
            self.fail(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        config_module = types.ModuleType("config")
        config_module.EAConfig = type("EAConfig", (), {})
        genetic_module = types.ModuleType("genetic_operators")
        genetic_module.GeneticOperators = type("GeneticOperators", (), {})
        genetic_module.Individual = type("Individual", (), {})
        genetic_module._ancestor_ids = lambda individual: []
        executor_module = types.ModuleType("executor")
        executor_module.TritonExecutor = type("TritonExecutor", (), {})
        executor_module.EvaluationResult = type("EvaluationResult", (), {})
        numpy_module = types.ModuleType("numpy")
        with patch.dict(sys.modules, {
            "config": config_module,
            "genetic_operators": genetic_module,
            "executor": executor_module,
            "numpy": numpy_module,
        }):
            spec.loader.exec_module(module)

        evidence = {"schema_version": 1, "case_results": [{"case_id": 1}]}
        result = types.SimpleNamespace(
            fitness=0.5,
            success=True,
            speedup=0.5,
            execution_time=2.0,
            error=None,
            evidence=evidence,
        )
        individual = types.SimpleNamespace(code="code", fitness=0.0, metadata={})
        genetic_ops = types.SimpleNamespace(
            llm=types.SimpleNamespace(total_tokens_used=0)
        )
        ea = module.EvolutionaryAlgorithm(
            genetic_ops,
            types.SimpleNamespace(evaluate=lambda code, timeout=None: result),
            types.SimpleNamespace(timeout_seconds=10, max_total_tokens=100),
        )
        ea.population = [individual]

        ea._evaluate_population()

        self.assertEqual(individual.metadata["evaluation_evidence"], evidence)

    def test_contract_compares_api_ast_but_allows_body_changes(self) -> None:
        sys.path.insert(0, str(OFFICIAL_WORK))
        try:
            module = importlib.import_module("contract_executor")
        finally:
            sys.path.remove(str(OFFICIAL_WORK))

        baseline = (
            '@compile(mode="fast")\n'
            'def kernel(x: Tensor, scale: float = 1.0) -> Output:\n'
            '    return x\n'
        )
        mismatches = {
            "default": baseline.replace("1.0", "2.0"),
            "parameter annotation": baseline.replace("x: Tensor", "x: Other"),
            "return annotation": baseline.replace("-> Output", "-> Other"),
            "decorator argument": baseline.replace('mode="fast"', 'mode="safe"'),
        }
        for label, candidate in mismatches.items():
            with self.subTest(label=label):
                self.assertIsNotNone(
                    module.interface_contract_error(baseline, candidate)
                )

        body_only_change = baseline.replace("return x", "return x * scale")
        self.assertIsNone(module.interface_contract_error(baseline, body_only_change))

    def test_invalid_candidate_is_rejected_before_official_executor(self) -> None:
        sys.path.insert(0, str(OFFICIAL_WORK))
        try:
            module = importlib.import_module("contract_executor")
        finally:
            sys.path.remove(str(OFFICIAL_WORK))

        baseline = "\n".join(
            [
                "import triton",
                "import triton.language as tl",
                "@triton.jit",
                "def kernel(x, BLOCK_SIZE: tl.constexpr):",
                "    return",
            ]
        )
        candidate = baseline.replace(
            "@triton.jit",
            "@triton.autotune(configs=[], key=[])\n@triton.jit",
        )

        with tempfile.TemporaryDirectory() as tmp:
            executor = module.ContractCheckingExecutor(
                baseline_time=1.0,
                test_code_path=str(Path(tmp) / "test_kernel.py"),
                config=types.SimpleNamespace(),
                kernel_name="kernel",
                work_dir=Path(tmp),
                baseline_code=baseline,
            )
            delegated_result = module.EvaluationResult(True, 1.0, 0.0, 0.0)
            with patch.object(module.TritonExecutor, "evaluate", return_value=delegated_result) as delegate:
                rejected = executor.evaluate(candidate)
                accepted = executor.evaluate(baseline)

        self.assertFalse(rejected.success)
        self.assertIn("decorators differ from baseline", rejected.error)
        delegate.assert_called_once_with(baseline, timeout=1200)
        self.assertIs(accepted, delegated_result)

    def test_optimizer_setup_uses_contract_checking_executor(self) -> None:
        sys.path.insert(0, str(OFFICIAL_WORK))
        try:
            contract_module = importlib.import_module("contract_executor")
        finally:
            sys.path.remove(str(OFFICIAL_WORK))

        config_module = types.ModuleType("config")
        config_module.EAConfig = type("EAConfig", (), {})
        llm_module = types.ModuleType("llm_interface")
        llm_module.LLMInterface = lambda config: types.SimpleNamespace()
        genetic_module = types.ModuleType("genetic_operators")
        genetic_module.GeneticOperators = lambda llm, config: types.SimpleNamespace()
        evolution_module = types.ModuleType("evolutionary_algorithm")
        evolution_module.EvolutionaryAlgorithm = (
            lambda genetic_ops, executor, config: types.SimpleNamespace()
        )

        optimizer_path = OFFICIAL_WORK / "optimizer_agent.py"
        spec = importlib.util.spec_from_file_location("official_optimizer_setup_test", optimizer_path)
        if spec is None or spec.loader is None:
            self.fail(f"Cannot load {optimizer_path}")
        optimizer_module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                spec.name: optimizer_module,
                "config": config_module,
                "llm_interface": llm_module,
                "genetic_operators": genetic_module,
                "evolutionary_algorithm": evolution_module,
                "contract_executor": contract_module,
            },
        ):
            spec.loader.exec_module(optimizer_module)

        config = types.SimpleNamespace(baseline_json="unused.json")
        agent = optimizer_module.TritonOptimizerAgent(config)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(optimizer_module, "get_baseline_from_json", return_value=1.0):
                agent.setup(
                    baseline_code="def kernel(x):\n    return\n",
                    test_code="pass\n",
                    kernel_name="kernel",
                    work_dir=tmp,
                )

        self.assertIsInstance(agent.executor, contract_module.MultiCaseContractExecutor)
        self.assertEqual(agent.executor.case_ids, (1,))
        inner = agent.executor.cases[0][1]
        self.assertEqual(inner.baseline_code, "def kernel(x):\n    return\n")
        self.assertEqual(inner.test_code_path.name, "test_kernel.py")

    def test_optimizer_setup_binds_each_case_to_its_baseline(self) -> None:
        sys.path.insert(0, str(OFFICIAL_WORK))
        try:
            contract_module = importlib.import_module("contract_executor")
        finally:
            sys.path.remove(str(OFFICIAL_WORK))

        optimizer_module = self._load_optimizer_with_contract(contract_module)
        config = types.SimpleNamespace(baseline_json="unused.json")
        agent = optimizer_module.TritonOptimizerAgent(config)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for case_id in (1, 2):
                path = Path(tmp) / f"test_kernel_{case_id}.py"
                path.write_text("pass\n", encoding="utf-8")
                paths.append((case_id, str(path)))
            with patch.object(
                optimizer_module,
                "get_baseline_from_json",
                side_effect=lambda _, __, case_id: {1: 1.5, 2: 2.5}[case_id],
            ):
                agent.setup(
                    baseline_code="def kernel(x):\n    return\n",
                    kernel_name="kernel",
                    work_dir=tmp,
                    test_cases=paths,
                )

        self.assertIsInstance(
            agent.executor, contract_module.MultiCaseContractExecutor
        )
        self.assertEqual(agent.executor.case_ids, (1, 2))
        self.assertEqual(
            [executor.baseline_time for _, executor in agent.executor.cases],
            [1.5, 2.5],
        )

    @staticmethod
    def _load_optimizer_with_contract(contract_module):
        config_module = types.ModuleType("config")
        config_module.EAConfig = type("EAConfig", (), {})
        llm_module = types.ModuleType("llm_interface")
        llm_module.LLMInterface = lambda config: types.SimpleNamespace()
        genetic_module = types.ModuleType("genetic_operators")
        genetic_module.GeneticOperators = lambda llm, config: types.SimpleNamespace()
        evolution_module = types.ModuleType("evolutionary_algorithm")
        evolution_module.EvolutionaryAlgorithm = lambda *args: types.SimpleNamespace()
        path = OFFICIAL_WORK / "optimizer_agent.py"
        spec = importlib.util.spec_from_file_location("official_multi_case_setup", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "config": config_module,
            "llm_interface": llm_module,
            "genetic_operators": genetic_module,
            "evolutionary_algorithm": evolution_module,
            "contract_executor": contract_module,
        }):
            spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
