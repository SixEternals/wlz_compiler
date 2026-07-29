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


class OfficialContractExecutorTests(unittest.TestCase):
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
            test_file = Path(tmp) / "test_kernel.py"
            test_file.write_text("pass\n", encoding="utf-8")
            with patch.object(optimizer_module, "get_baseline_from_json", return_value=1.0):
                agent.setup(
                    baseline_code="def kernel(x):\n    return\n",
                    test_code=str(test_file),
                    kernel_name="kernel",
                    work_dir=tmp,
                )

        self.assertIsInstance(agent.executor, contract_module.ContractCheckingExecutor)
        self.assertEqual(agent.executor.baseline_code, "def kernel(x):\n    return\n")


if __name__ == "__main__":
    unittest.main()
