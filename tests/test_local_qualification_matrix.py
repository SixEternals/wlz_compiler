"""Focused tests for the local Ascend qualification matrix."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_local_ascend_correctness import _fingerprint
from scripts.build_local_qualification_matrix import (
    _candidate_change_class,
    _rank_candidates,
    build_matrix,
)


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalQualificationMatrixTests(unittest.TestCase):
    def test_selection_excludes_faster_neutral_candidates(self) -> None:
        evaluations = [
            {
                "candidate_id": "neutral",
                "candidate_over_baseline_ratio": 0.8,
                "change_class": "ast_equivalent",
                "qualified": True,
            },
            {
                "candidate_id": "real",
                "candidate_over_baseline_ratio": 0.9,
                "change_class": "substantive",
                "qualified": True,
            },
        ]

        qualified, substantive = _rank_candidates(evaluations)

        self.assertEqual(qualified[0]["candidate_id"], "neutral")
        self.assertEqual(substantive[0]["candidate_id"], "real")

    def test_change_class_uses_ast_and_explicit_neutral_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            equivalent = root / "equivalent.py"
            changed = root / "changed.py"
            manifest = root / "candidate.manifest.json"
            _write(baseline, "def op(x):\n    return x\n")
            _write(equivalent, "# comment only\ndef op(x):\n    return x\n")
            _write(changed, "def op(x):\n    return x + 1\n")
            _write(manifest, json.dumps({"candidate": {"mutation_kind": "mutation"}}))

            self.assertEqual(
                _candidate_change_class(baseline, equivalent, manifest),
                "ast_equivalent",
            )
            self.assertEqual(
                _candidate_change_class(baseline, changed, manifest),
                "substantive",
            )
            _write(
                manifest,
                json.dumps(
                    {"candidate": {"mutation_kind": "existing_variant_identifier_only"}}
                ),
            )
            self.assertEqual(
                _candidate_change_class(baseline, changed, manifest),
                "declared_neutral",
            )

    def test_only_complete_hash_bound_paired_evidence_qualifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = root / "datasets"
            candidates = root / "candidates"
            for operator in ("op_a", "op_b"):
                op_dir = datasets / operator
                _write(op_dir / f"{operator}.py", f"def {operator}(x):\n    return x\n")
                _write(op_dir / f"test_{operator}_1.py", "print('All tests passed!')\n")
                _write(
                    candidates / operator / "candidate.py",
                    f"def {operator}(x):\n    return x + 0\n",
                )

            operator = "op_a"
            baseline = datasets / operator / f"{operator}.py"
            test = datasets / operator / f"test_{operator}_1.py"
            source = candidates / operator / "candidate.py"
            manifest = candidates / operator / "candidate.manifest.json"
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            _write(
                manifest,
                json.dumps(
                    {
                        "candidate": {
                            "id": "candidate",
                            "op_name": operator,
                            "code_hash": source_sha256,
                            "status": "static_pass",
                        }
                    }
                ),
            )

            def run_record(name: str, duration: float) -> dict:
                run_dir = root / "profiles" / name
                csv_path = run_dir / "OpBasicInfo.csv"
                csv_path.parent.mkdir(parents=True)
                with csv_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "Op Name", "Task Duration(us)", "Device Id", "Current Freq",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "Op Name": operator,
                            "Task Duration(us)": duration,
                            "Device Id": 0,
                            "Current Freq": 1650,
                        }
                    )
                log_path = run_dir / "get_prof.log"
                seconds = {"b1": 0, "c1": 1, "c2": 2, "b2": 3}[name]
                _write(
                    log_path,
                    f"2026-01-01 00:00:0{seconds} [INFO] start\n"
                    "All tests passed!\n"
                    "Profiling running finished. All task success.\n"
                    f"Op Name: {operator}\n",
                )
                return {
                    "csv_path": csv_path.relative_to(root).as_posix(),
                    "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                    "log_path": log_path.relative_to(root).as_posix(),
                    "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                    "duration_us": duration,
                    "frequency_mhz": 1650,
                }

            baseline_runs = [run_record("b1", 100.0), run_record("b2", 102.0)]
            candidate_runs = [run_record("c1", 99.0), run_record("c2", 101.0)]
            environment = {
                "cann_version": "9.0.0",
                "device_name": "Ascend910B4-1",
                "machine": "aarch64",
                "python_executable": "/usr/local/bin/python",
                "python_version": "3.11.0",
                "torch": "2.7.1+cpu",
                "torch_npu": "2.7.1",
                "triton": "3.2.0",
            }
            environment["fingerprint_sha256"] = hashlib.sha256(
                json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            sidecar = {
                "schema_version": 1,
                "artifact_kind": "local-ascend-candidate-evaluation",
                "operator": operator,
                "evidence_scope": "local-ascend-910b4-msprof-not-official-evaluation",
                "environment": environment,
                "candidate": {
                    "id": "candidate",
                    "source_path": source.relative_to(root).as_posix(),
                    "source_sha256": source_sha256,
                    "manifest_path": manifest.relative_to(root).as_posix(),
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                },
                "sources": {
                    "baseline_path": baseline.relative_to(root).as_posix(),
                    "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                    "test_path": test.relative_to(root).as_posix(),
                },
                "correctness": {
                    "status": "local_public_case_1_passed",
                    "completion_marker": "All tests passed!",
                    "candidate_public_test_run_count": 2,
                    "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                },
                "profiler": {
                    "baseline_runs": baseline_runs,
                    "candidate_runs": candidate_runs,
                    "candidate_over_baseline_ratio": 100.0 / 101.0,
                },
            }
            path = candidates / operator / "candidate.ascend-evaluation.json"
            _write(path, json.dumps(sidecar))

            paired_operator = "op_b"
            paired_dir = candidates / paired_operator
            paired_baseline = datasets / paired_operator / f"{paired_operator}.py"
            paired_test = datasets / paired_operator / f"test_{paired_operator}_1.py"
            paired_candidate = paired_dir / "candidate.py"
            paired_manifest = paired_dir / "candidate.manifest.json"
            paired_source_hash = hashlib.sha256(paired_candidate.read_bytes()).hexdigest()
            _write(
                paired_manifest,
                json.dumps(
                    {
                        "candidate": {
                            "id": "candidate",
                            "op_name": paired_operator,
                            "code_hash": paired_source_hash,
                            "status": "static_pass",
                        }
                    }
                ),
            )
            paired_facts = {
                "python_executable": "/usr/local/bin/python",
                "python_version": "3.11.0",
                "machine": "aarch64",
                "torch": "2.7.1+cpu",
                "torch_npu": "2.7.1.post4",
                "triton": "3.2.0",
                "npu_available": True,
                "device_name": "Ascend910B4-1",
            }
            paired_environment = _fingerprint(paired_facts)
            paired_correctness_path = (
                root / "output" / "local-correctness" / paired_operator / "candidate.correctness.json"
            )
            paired_correctness = {
                "artifact_kind": "local-ascend-correctness-evaluation",
                "operator": paired_operator,
                "environment": paired_environment,
                "qualified_for_local_performance": True,
                "correctness_status": "passed",
                "candidate": {"source_sha256": paired_source_hash},
                "baseline": {
                    "source_sha256": hashlib.sha256(paired_baseline.read_bytes()).hexdigest()
                },
                "cases": [
                    {
                        "test_path": paired_test.relative_to(root).as_posix(),
                        "test_sha256": hashlib.sha256(paired_test.read_bytes()).hexdigest(),
                        "status": "passed",
                    }
                ],
            }
            _write(paired_correctness_path, json.dumps(paired_correctness))
            paired_runs = {"baseline": [], "candidate": []}
            for index, (role, duration) in enumerate(
                (("baseline", 100.0), ("candidate", 95.0), ("candidate", 96.0), ("baseline", 101.0)),
                1,
            ):
                run_dir = (
                    root / "output" / "local-paired" / paired_operator
                    / "candidate.paired" / "case-1" / f"{index:02d}-{role}"
                )
                csv_path = run_dir / "profile.csv"
                log_path = run_dir / "get_prof.log"
                _write(csv_path, "profile\n")
                _write(log_path, "profiler ok\n")
                paired_runs[role].append(
                    {
                        "role": role,
                        "status": "passed",
                        "returncode": 0,
                        "source_sha256": (
                            hashlib.sha256(paired_baseline.read_bytes()).hexdigest()
                            if role == "baseline" else paired_source_hash
                        ),
                        "test_sha256": hashlib.sha256(paired_test.read_bytes()).hexdigest(),
                        "csv_path": csv_path.relative_to(root).as_posix(),
                        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                        "log_path": log_path.relative_to(root).as_posix(),
                        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                        "duration_us": duration,
                        "device_id": 0,
                        "frequency_mhz": 1650,
                        "input_seed": 0,
                    }
                )
            paired_path = (
                root / "output" / "local-paired" / paired_operator / "candidate.paired.json"
            )
            paired_payload = {
                "schema_version": 1,
                "artifact_kind": "local-ascend-paired-benchmark",
                "evidence_scope": "local_ascend_910b4_currently_visible_cases_not_official",
                "operator": paired_operator,
                "correctness_artifact_path": paired_correctness_path.relative_to(root).as_posix(),
                "correctness_artifact_sha256": hashlib.sha256(
                    paired_correctness_path.read_bytes()
                ).hexdigest(),
                "candidate": {
                    "id": "candidate",
                    "source_path": paired_candidate.relative_to(root).as_posix(),
                    "source_sha256": paired_source_hash,
                    "manifest_path": paired_manifest.relative_to(root).as_posix(),
                    "manifest_sha256": hashlib.sha256(paired_manifest.read_bytes()).hexdigest(),
                },
                "baseline": {
                    "source_path": paired_baseline.relative_to(root).as_posix(),
                    "source_sha256": hashlib.sha256(paired_baseline.read_bytes()).hexdigest(),
                },
                "environment": paired_environment,
                "sequence": ["baseline", "candidate", "candidate", "baseline"],
                "input_seed": 0,
                "cases": [
                    {
                        "test_path": paired_test.relative_to(root).as_posix(),
                        "test_sha256": hashlib.sha256(paired_test.read_bytes()).hexdigest(),
                        "sequence": ["baseline", "candidate", "candidate", "baseline"],
                        "input_seed": 0,
                        "baseline_runs": paired_runs["baseline"],
                        "candidate_runs": paired_runs["candidate"],
                        "candidate_over_baseline_ratio": 95.5 / 100.5,
                        "status": "passed",
                    }
                ],
                "candidate_over_baseline_ratio_limit": 1.03,
                "worst_case_ratio": 95.5 / 100.5,
                "qualified_for_local_performance": True,
            }
            _write(paired_path, json.dumps(paired_payload))

            report = build_matrix(root, datasets, candidates, 1.03)
            rows = {row["operator"]: row for row in report["operators"]}

            self.assertEqual(report["summary"]["qualified_count"], 2)
            self.assertEqual(report["summary"]["substantive_selected_count"], 2)
            self.assertEqual(rows["op_a"]["qualification_status"], "qualified")
            self.assertEqual(rows["op_a"]["compile_status"], "unknown")
            self.assertEqual(rows["op_b"]["qualification_status"], "qualified")
            self.assertEqual(rows["op_b"]["selection_status"], "substantive_candidate_ready")
            self.assertEqual(rows["op_b"]["best_candidate_id"], "candidate")
            self.assertAlmostEqual(
                rows["op_b"]["evaluations"][0]["candidate_over_baseline_ratio"],
                95.5 / 100.5,
            )

            del paired_payload["cases"][0]["candidate_runs"][0]["input_seed"]
            _write(paired_path, json.dumps(paired_payload))
            unseeded = {
                row["operator"]: row
                for row in build_matrix(root, datasets, candidates, 1.03)["operators"]
            }[paired_operator]
            self.assertIn("paired_input_seed_unverified", unseeded["blocking_reasons"])
            paired_payload["cases"][0]["candidate_runs"][0]["input_seed"] = 0

            paired_payload["cases"][0]["baseline_runs"][0]["duration_us"] = 100.0
            paired_payload["cases"][0]["baseline_runs"][1]["duration_us"] = 100.0
            paired_payload["cases"][0]["candidate_runs"][0]["duration_us"] = 104.0
            paired_payload["cases"][0]["candidate_runs"][1]["duration_us"] = 104.0
            paired_payload["cases"][0]["candidate_over_baseline_ratio"] = 1.04
            paired_payload["cases"][0]["status"] = "failed"
            paired_payload["worst_case_ratio"] = 1.04
            paired_payload["qualified_for_local_performance"] = False
            _write(paired_path, json.dumps(paired_payload))
            slower = {
                row["operator"]: row
                for row in build_matrix(root, datasets, candidates, 1.03)["operators"]
            }[paired_operator]
            self.assertEqual(slower["qualification_status"], "incomplete")
            self.assertIn("performance_ratio_above_limit", slower["blocking_reasons"])

            (root / baseline_runs[0]["log_path"]).write_text("tampered\n", encoding="utf-8")
            failed = build_matrix(root, datasets, candidates, 1.03)["operators"][0]
            self.assertEqual(failed["qualification_status"], "incomplete")
            self.assertIn("legacy_ratio_unverified", failed["blocking_reasons"])

            (root / candidate_runs[0]["log_path"]).write_text(
                "All tests passed!\n", encoding="utf-8"
            )
            failed = build_matrix(root, datasets, candidates, 1.03)["operators"][0]
            self.assertEqual(failed["correctness_status"], "unknown")
            self.assertIn("legacy_correctness_log_unverified", failed["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
