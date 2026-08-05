"""Focused tests for the serial local Ascend paired benchmark."""

import hashlib
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_local_ascend_correctness import _fingerprint
from scripts.run_local_ascend_paired_benchmark import run_paired_benchmark


class LocalAscendPairedBenchmarkTests(unittest.TestCase):
    def test_abba_order_and_ratio_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op = "demo_op"
            dataset = root / "datasets" / op
            candidate_dir = root / "candidates" / op
            dataset.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            baseline = dataset / f"{op}.py"
            test = dataset / f"test_{op}_1.py"
            candidate = candidate_dir / "candidate.py"
            manifest = candidate_dir / "candidate.manifest.json"
            baseline.write_text("def value():\n    return 1\n", encoding="utf-8")
            candidate.write_text("def value():\n    return 1 + 0\n", encoding="utf-8")
            test.write_text(f"from {op} import value\nassert value() == 1\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "candidate_path": str(candidate),
                        "candidate": {
                            "id": "candidate",
                            "op_name": op,
                            "code_hash": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                            "status": "static_pass",
                        },
                    }
                ),
                encoding="utf-8",
            )
            facts = {
                "python_executable": sys.executable,
                "python_version": "test",
                "machine": "test",
                "torch": "test",
                "torch_npu": "test",
                "triton": "test",
                "npu_available": True,
                "device_name": "Ascend910B4-test",
            }
            environment = _fingerprint(facts)
            correctness = root / "correctness.json"
            correctness.write_text(
                json.dumps(
                    {
                        "artifact_kind": "local-ascend-correctness-evaluation",
                        "qualified_for_local_performance": True,
                        "correctness_status": "passed",
                        "evidence_scope": "local_ascend_910b4_derived_shape_matrix_not_official",
                        "operator": op,
                        "environment": environment,
                        "candidate": {
                            "source_path": candidate.relative_to(root).as_posix(),
                            "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                            "manifest_path": manifest.relative_to(root).as_posix(),
                            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        },
                        "baseline": {
                            "source_path": baseline.relative_to(root).as_posix(),
                            "source_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                        },
                        "cases": [
                            {
                                "test_path": test.relative_to(root).as_posix(),
                                "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                                "status": "passed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            fake = root / "fake-msprof.py"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import csv, pathlib, sys\n"
                "out = pathlib.Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('--output=')))\n"
                "op = next(x.split('=', 1)[1] for x in sys.argv if x.startswith('--kernel-name=')) + '_mix_aic'\n"
                "out = out / 'OPPROF_fake'; out.mkdir(parents=True)\n"
                "role = __import__('os').environ['WLZ_PAIRED_ROLE']\n"
                "duration = 100 if role == 'baseline' else 95\n"
                "with (out / 'OpBasicInfo.csv').open('w', newline='') as h:\n"
                "    w = csv.DictWriter(h, fieldnames=['Op Name','Task Duration(us)','Device Id','Current Freq']); w.writeheader(); w.writerow({'Op Name':op,'Task Duration(us)':duration,'Device Id':0,'Current Freq':1650})\n"
                "print('fake profiler complete')\n",
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

            # The runner currently uses the role environment only for the fake profiler.
            report = run_paired_benchmark(
                root, correctness, Path(sys.executable), root / "paired.json",
                str(fake), pairs=2, timeout=10.0, environment=environment,
            )

            self.assertTrue(report["qualified_for_local_performance"])
            self.assertEqual(
                report["evidence_scope"],
                "local_ascend_910b4_derived_shape_matrix_not_official",
            )
            self.assertEqual(report["input_seed"], 0)
            self.assertEqual(report["cases"][0]["sequence"], ["baseline", "candidate", "candidate", "baseline"])
            self.assertEqual(report["cases"][0]["input_seed"], 0)
            self.assertEqual(
                {run["input_seed"] for run in report["cases"][0]["baseline_runs"] + report["cases"][0]["candidate_runs"]},
                {0},
            )
            self.assertEqual(
                {run["profile_op_name"] for run in report["cases"][0]["baseline_runs"] + report["cases"][0]["candidate_runs"]},
                {"demo_op_mix_aic"},
            )
            self.assertAlmostEqual(report["cases"][0]["candidate_over_baseline_ratio"], 0.95)

    def test_missing_profile_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op = "demo_op"
            dataset = root / "datasets" / op
            candidate_dir = root / "candidates" / op
            dataset.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            baseline = dataset / f"{op}.py"
            test = dataset / f"test_{op}_1.py"
            candidate = candidate_dir / "candidate.py"
            manifest = candidate_dir / "candidate.manifest.json"
            baseline.write_text("def value():\n    return 1\n", encoding="utf-8")
            candidate.write_text("def value():\n    return 1 + 0\n", encoding="utf-8")
            test.write_text(f"from {op} import value\nassert value() == 1\n", encoding="utf-8")
            manifest.write_text(json.dumps({
                "candidate_path": str(candidate),
                "candidate": {"id": "candidate", "op_name": op,
                              "code_hash": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                              "status": "static_pass"},
            }), encoding="utf-8")
            facts = {"python_executable": sys.executable, "python_version": "test",
                     "machine": "test", "torch": "test", "torch_npu": "test",
                     "triton": "test", "npu_available": True,
                     "device_name": "Ascend910B4-test"}
            environment = _fingerprint(facts)
            correctness = root / "correctness.json"
            correctness.write_text(json.dumps({
                "artifact_kind": "local-ascend-correctness-evaluation",
                "qualified_for_local_performance": True, "correctness_status": "passed",
                "operator": op, "environment": environment,
                "candidate": {"source_path": candidate.relative_to(root).as_posix(),
                              "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                              "manifest_path": manifest.relative_to(root).as_posix(),
                              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
                "baseline": {"source_path": baseline.relative_to(root).as_posix(),
                             "source_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()},
                "cases": [{"test_path": test.relative_to(root).as_posix(),
                           "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                           "status": "passed"}],
            }), encoding="utf-8")
            fake = root / "fake-msprof.py"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "out = pathlib.Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('--output=')))\n"
                "(out / 'OPPROF_fake').mkdir(parents=True)\n",
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            report = run_paired_benchmark(
                root, correctness, Path(sys.executable), root / "paired.json",
                str(fake), pairs=2, timeout=10.0, environment=environment,
            )
            self.assertFalse(report["qualified_for_local_performance"])
            self.assertIsNone(report["cases"][0]["candidate_over_baseline_ratio"])


if __name__ == "__main__":
    unittest.main()
