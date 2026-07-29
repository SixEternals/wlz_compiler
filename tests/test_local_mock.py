import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.cache import EvaluationCache
from wlz_optimizer.executors import LocalExecutor
from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.schemas import Candidate, EvalContext, EvaluationResult


class LocalExecutorTests(unittest.TestCase):
    def _candidate(self, code: str) -> Candidate:
        return Candidate(
            id="demo-candidate",
            op_name="demo_op",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=[],
            generation=0,
            mutation_kind="baseline",
            model_used=None,
            prompt_id=None,
            status="created",
            score=None,
            metadata={},
        )

    def test_valid_triton_static_code_passes(self) -> None:
        code = "\n".join(
            [
                "import triton",
                "import triton.language as tl",
                "",
                "@triton.jit",
                "def kernel(x):",
                "    return",
                "",
                "def wrapper(x):",
                "    return x",
                "",
            ]
        )
        ctx = EvalContext(
            op_name="demo_op",
            input_dir=ROOT,
            output_dir=ROOT,
            required_functions=["wrapper"],
        )
        result = LocalExecutor().evaluate(self._candidate(code), ctx)

        self.assertTrue(result.passed)
        self.assertEqual(result.status, "local_static_pass")
        self.assertIsNone(result.speedup)
        self.assertIsNotNone(result.proxy_score)

    def test_vllm_triton_utils_import_style_passes(self) -> None:
        code = "\n".join(
            [
                "from vllm.triton_utils import tl, triton",
                "",
                "@triton.jit",
                "def kernel(x):",
                "    return",
                "",
                "def wrapper(x):",
                "    return x",
                "",
            ]
        )
        ctx = EvalContext(
            op_name="demo_op",
            input_dir=ROOT,
            output_dir=ROOT,
            required_functions=["wrapper"],
        )
        result = LocalExecutor().evaluate(self._candidate(code), ctx)

        self.assertTrue(result.passed)
        self.assertEqual(result.metadata["imports_ok"], True)

    def test_syntax_error_is_classified(self) -> None:
        code = "import triton\ndef wrapper(:\n    pass\n"
        ctx = EvalContext(
            op_name="demo_op",
            input_dir=ROOT,
            output_dir=ROOT,
            required_functions=["wrapper"],
        )
        result = LocalExecutor().evaluate(self._candidate(code), ctx)

        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "syntax_fail")
        self.assertEqual(result.proxy_score, 0.0)

    def test_candidate_signature_must_match_baseline(self) -> None:
        baseline_code = "\n".join(
            [
                "import triton",
                "import triton.language as tl",
                "",
                "@triton.jit",
                "def kernel(x, BLOCK_SIZE: tl.constexpr):",
                "    return",
                "",
            ]
        )
        candidate_code = baseline_code.replace(
            "def kernel(x, BLOCK_SIZE: tl.constexpr):",
            "def kernel(x, total_num_blocks, BLOCK_SIZE: tl.constexpr):",
        )

        with tempfile.TemporaryDirectory() as tmp:
            baseline_file = Path(tmp) / "kernel.py"
            baseline_file.write_text(baseline_code, encoding="utf-8")
            ctx = EvalContext(
                op_name="demo_op",
                input_dir=Path(tmp),
                output_dir=Path(tmp),
                required_functions=["kernel"],
                baseline_file=baseline_file,
            )
            result = LocalExecutor().evaluate(self._candidate(candidate_code), ctx)

        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "signature_fail")
        self.assertIn("kernel", result.metadata["signature_mismatches"])
        self.assertIn("differs from baseline", result.error_message)

    def test_candidate_decorators_must_match_baseline(self) -> None:
        baseline_code = "\n".join(
            [
                "import triton",
                "import triton.language as tl",
                "",
                "@triton.jit",
                "def kernel(x, BLOCK_SIZE: tl.constexpr):",
                "    return",
                "",
            ]
        )
        candidate_code = baseline_code.replace(
            "@triton.jit",
            "@triton.autotune(configs=[], key=[])\n@triton.jit",
        )

        with tempfile.TemporaryDirectory() as tmp:
            baseline_file = Path(tmp) / "kernel.py"
            baseline_file.write_text(baseline_code, encoding="utf-8")
            ctx = EvalContext(
                op_name="demo_op",
                input_dir=Path(tmp),
                output_dir=Path(tmp),
                required_functions=["kernel"],
                baseline_file=baseline_file,
            )
            result = LocalExecutor().evaluate(self._candidate(candidate_code), ctx)

        self.assertFalse(result.passed)
        self.assertTrue(result.metadata["signature_ok"])
        self.assertFalse(result.metadata["launch_contract_ok"])
        self.assertEqual(result.error_type, "launch_contract_fail")
        self.assertIn("kernel", result.metadata["decorator_mismatches"])

    def test_runtime_dependent_arange_is_rejected(self) -> None:
        code = "\n".join(
            [
                "import triton",
                "import triton.language as tl",
                "",
                "@triton.jit",
                "def kernel(x, counts, BLOCK_SIZE: tl.constexpr):",
                "    num_blocks = tl.load(counts)",
                "    aligned = num_blocks - (num_blocks % BLOCK_SIZE)",
                "    for i in range(0, aligned, BLOCK_SIZE):",
                "        tl.store(x + i, 0)",
                "    remainder = num_blocks - aligned",
                "    offsets = tl.arange(0, remainder)",
                "    tl.store(x + aligned + offsets, offsets)",
                "",
            ]
        )
        ctx = EvalContext(
            op_name="demo_op",
            input_dir=ROOT,
            output_dir=ROOT,
            required_functions=["kernel"],
        )
        result = LocalExecutor().evaluate(self._candidate(code), ctx)

        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "triton_semantic_fail")
        errors = result.metadata["triton_semantic_errors"]
        self.assertEqual([item["code"] for item in errors], ["dynamic_tl_arange"])
        self.assertIn("remainder", errors[0]["runtime_symbols"])

    def test_runtime_python_range_is_not_rejected_by_narrow_gate(self) -> None:
        code = "\n".join(
            [
                "import triton",
                "import triton.language as tl",
                "",
                "@triton.jit",
                "def kernel(x, BLOCK_SIZE: tl.constexpr):",
                "    pid = tl.program_id(0)",
                "    for i in range(pid):",
                "        tl.store(x + i, i)",
                "",
            ]
        )
        ctx = EvalContext(
            op_name="demo_op",
            input_dir=ROOT,
            output_dir=ROOT,
            required_functions=["kernel"],
        )
        result = LocalExecutor().evaluate(self._candidate(code), ctx)

        self.assertTrue(result.passed)
        self.assertEqual(result.metadata["triton_semantic_errors"], [])

    def test_hardcoded_cuda_device_context_is_rejected_for_ascend_target(self) -> None:
        base = "\n".join(
            [
                "import torch",
                "import triton",
                "import triton.language as tl",
                "@triton.jit",
                "def kernel(x):",
                "    return",
                "def wrapper(x):",
                "    with torch.npu.device(x.device.index):",
                "        return x",
            ]
        )
        ctx = EvalContext(
            op_name="demo_op",
            input_dir=ROOT,
            output_dir=ROOT,
            required_functions=["wrapper"],
        )
        cuda = LocalExecutor().evaluate(
            self._candidate(base.replace("torch.npu.device", "torch.cuda.device")), ctx
        )

        self.assertFalse(cuda.passed)
        self.assertEqual(cuda.error_type, "target_device_fail")
        self.assertEqual(
            [item["code"] for item in cuda.metadata["target_device_errors"]],
            ["hardcoded_cuda_device_context"],
        )
        self.assertTrue(LocalExecutor().evaluate(self._candidate(base), ctx).passed)

        dynamic = base.replace("torch.npu.device", "device_context")
        string_only = base.replace(
            "with torch.npu.device(x.device.index):",
            "note = 'torch.cuda.device(x.device.index)'\n    if note:",
        )
        self.assertTrue(LocalExecutor().evaluate(self._candidate(dynamic), ctx).passed)
        self.assertTrue(LocalExecutor().evaluate(self._candidate(string_only), ctx).passed)

    def test_real_state_passing_cuda_context_is_rejected(self) -> None:
        path = (
            ROOT
            / "output/real-agent-candidates/_state_passing_fwd_kernel/d3ab8399.py"
        )
        code = path.read_text(encoding="utf-8")
        result = LocalExecutor().evaluate(
            self._candidate(code),
            EvalContext(
                op_name="_state_passing_fwd_kernel",
                input_dir=ROOT,
                output_dir=ROOT,
                required_functions=["_state_passing_fwd"],
            ),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "target_device_fail")


class LocalMockRunnerTests(unittest.TestCase):
    def test_runner_writes_output_contract_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "datasets"
            output_dir = tmp_path / "output" / "mock"
            op_dir = input_dir / "demo_op"
            op_dir.mkdir(parents=True)

            (op_dir / "demo_op.py").write_text(
                "\n".join(
                    [
                        "import torch",
                        "import triton",
                        "import triton.language as tl",
                        "",
                        "@triton.jit",
                        "def demo_kernel(x, BLOCK_SIZE: tl.constexpr):",
                        "    offsets = tl.arange(0, BLOCK_SIZE)",
                        "    tl.store(x + offsets, offsets)",
                        "",
                        "def demo_op(x):",
                        "    return x",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (op_dir / "test_demo_op_1.py").write_text(
                "from demo_op import demo_op\n\n"
                "def test_demo_op():\n"
                "    assert demo_op(1) == 1\n",
                encoding="utf-8",
            )

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_local_mock.py"),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--population-size",
                "4",
                "--generations",
                "1",
            ]

            first = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("cache_hits=", first.stdout)
            cache_lines_after_first = (output_dir / "cache.jsonl").read_text(encoding="utf-8").splitlines()

            second = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertIn("cache_hits=8", second.stdout)
            cache_lines_after_second = (output_dir / "cache.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(cache_lines_after_first), len(cache_lines_after_second))

            self.assertTrue((output_dir / "candidates").is_dir())
            self.assertTrue((output_dir / "top5").is_dir())
            self.assertTrue((output_dir / "manifest.json").is_file())
            self.assertTrue((output_dir / "cache.jsonl").is_file())
            self.assertTrue((output_dir / "run_report.md").is_file())

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["operator_count"], 1)
            self.assertEqual(manifest["summary"]["candidate_count"], 8)
            self.assertEqual(manifest["summary"]["cache_hits"], 8)
            self.assertLessEqual(len(manifest["top5"]["demo_op"]), 5)


class EvaluationCacheKeyTests(unittest.TestCase):
    def _candidate(self, mutation_kind: str = "baseline") -> "Candidate":
        code = "import triton\n"
        return Candidate(
            id=f"cache-key-{mutation_kind}",
            op_name="demo_op",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=[],
            generation=0,
            mutation_kind=mutation_kind,
            model_used=None,
            prompt_id=None,
            status="created",
            score=None,
            metadata={},
        )

    def _result(self, candidate: "Candidate") -> "EvaluationResult":
        return EvaluationResult(
            candidate_id=candidate.id,
            executor="local_static_proxy",
            status="local_static_pass",
            passed=True,
            correctness_ok=None,
            compile_ok=None,
            latency_ms=None,
            baseline_ms=None,
            speedup=None,
            proxy_score=0.1,
            error_type=None,
            error_message=None,
            metadata={},
        )

    def test_key_covers_mutation_kind_and_baseline_content(self) -> None:
        base = EvaluationCache.make_key("op", "hash", "kind", "env")
        self.assertNotEqual(
            base, EvaluationCache.make_key("op", "hash", "kind", "env", "block_size_hint")
        )
        self.assertNotEqual(
            base, EvaluationCache.make_key("op", "hash", "kind", "env", "", "baseline-hash")
        )

    def test_baseline_change_invalidates_cached_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.py"
            baseline.write_text("def kernel():\n    return 1\n", encoding="utf-8")
            cache = EvaluationCache(Path(tmp) / "cache.jsonl")
            candidate = self._candidate()
            cache.put(candidate, self._result(candidate), "kind", "env", baseline)
            self.assertIsNotNone(cache.get(candidate, "kind", "env", baseline))
            baseline.write_text("def kernel():\n    return 2\n", encoding="utf-8")
            self.assertIsNone(cache.get(candidate, "kind", "env", baseline))

    def test_mutation_kind_change_misses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = EvaluationCache(Path(tmp) / "cache.jsonl")
            candidate = self._candidate("baseline")
            cache.put(candidate, self._result(candidate), "kind", "env")
            other = self._candidate("block_size_hint")
            self.assertIsNone(cache.get(other, "kind", "env"))


if __name__ == "__main__":
    unittest.main()
