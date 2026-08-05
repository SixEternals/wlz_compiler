"""Focused tests for local Ascend correctness evidence production."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_local_ascend_correctness import _fingerprint, _prepare_compatibility_shims, run_correctness


class LocalAscendCorrectnessTests(unittest.TestCase):
    def test_vllm_triton_utils_shim_is_local_and_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            shims = _prepare_compatibility_shims(
                work, b"from vllm.triton_utils import tl, triton\n", b""
            )
            self.assertEqual(shims, ["vllm.triton_utils"])
            self.assertEqual(
                (work / "vllm" / "triton_utils.py").read_text(encoding="utf-8"),
                "import triton\nimport triton.language as tl\n",
            )

    def test_runs_baseline_and_candidate_and_fails_wrong_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = "demo_op"
            op_dir = root / "datasets" / operator
            op_dir.mkdir(parents=True)
            baseline = op_dir / f"{operator}.py"
            test = op_dir / f"test_{operator}_1.py"
            candidate = root / "candidates" / operator / "candidate.py"
            manifest = candidate.with_name("candidate.manifest.json")
            candidate.parent.mkdir(parents=True)
            baseline.write_text("def value():\n    return 1\n", encoding="utf-8")
            candidate.write_text("def value():\n    return 1 + 0\n", encoding="utf-8")
            test.write_text(
                f"from {operator}_1 import value\nassert value() == 1\n", encoding="utf-8"
            )

            def write_manifest() -> None:
                payload = {
                    "candidate_path": str(candidate),
                    "candidate": {
                        "id": "candidate",
                        "op_name": operator,
                        "code_hash": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                        "status": "static_pass",
                    },
                }
                manifest.write_text(json.dumps(payload), encoding="utf-8")

            write_manifest()
            environment = _fingerprint(
                {
                    "python_executable": sys.executable,
                    "python_version": "test",
                    "machine": "test",
                    "torch": "test",
                    "torch_npu": "test",
                    "triton": "test",
                    "npu_available": True,
                    "device_name": "Ascend910B4-test",
                }
            )
            output = root / "evidence" / "candidate.correctness.json"
            passed = run_correctness(
                root,
                root / "datasets",
                operator,
                candidate,
                manifest,
                Path(sys.executable),
                output,
                10.0,
                environment,
            )

            self.assertTrue(passed["qualified_for_local_performance"])
            self.assertEqual(passed["correctness_status"], "passed")
            self.assertEqual(passed["compile_status"], "unknown")
            self.assertEqual(len(passed["cases"]), 1)
            self.assertEqual(passed["case_count"], 1)
            self.assertEqual(passed["visible_case_count"], 1)
            self.assertEqual(passed["test_selection"]["kind"], "dataset_visible")
            for role in ("baseline_run", "candidate_run"):
                record = passed["cases"][0][role]
                self.assertEqual(record["status"], "passed")
                self.assertEqual(record["prepared_test_sha256"], passed["cases"][0]["test_sha256"])
                self.assertTrue((root / record["stdout_path"]).is_file())

            holdout = run_correctness(
                root,
                root / "datasets",
                operator,
                candidate,
                manifest,
                Path(sys.executable),
                root / "evidence" / "holdout.correctness.json",
                10.0,
                environment,
                explicit_tests=(test,),
            )
            self.assertTrue(holdout["qualified_for_local_performance"])
            self.assertEqual(holdout["case_count"], 1)
            self.assertEqual(holdout["visible_case_count"], 0)
            self.assertEqual(holdout["test_selection"], {
                "kind": "explicit",
                "paths": ["datasets/demo_op/test_demo_op_1.py"],
            })
            self.assertEqual(
                holdout["limitations"][0],
                "Only explicitly supplied local tests are covered; organizer cases remain unknown.",
            )

            candidate.write_text("def value():\n    return 2\n", encoding="utf-8")
            write_manifest()
            failed = run_correctness(
                root,
                root / "datasets",
                operator,
                candidate,
                manifest,
                Path(sys.executable),
                root / "evidence" / "wrong.correctness.json",
                10.0,
                environment,
            )
            self.assertFalse(failed["qualified_for_local_performance"])
            self.assertEqual(failed["cases"][0]["candidate_run"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
