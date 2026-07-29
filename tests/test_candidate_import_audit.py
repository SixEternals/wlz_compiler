import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from wlz_optimizer.hash_utils import sha256_text

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "audit_existing_candidate_imports.py"
    spec = importlib.util.spec_from_file_location("candidate_import_audit_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExistingCandidateImportAuditTests(unittest.TestCase):
    def _write_candidate(self, root: Path, operator: str, candidate_id: str, code: str) -> None:
        datasets = root / "datasets" / operator
        datasets.mkdir(parents=True, exist_ok=True)
        (datasets / f"{operator}.py").write_text(
            f"def {operator}(x):\n    return x\n", encoding="utf-8"
        )
        candidates = root / "candidates" / operator
        candidates.mkdir(parents=True, exist_ok=True)
        (candidates / f"{candidate_id}.py").write_text(code, encoding="utf-8")
        manifest = {
            "candidate": {
                "id": candidate_id,
                "op_name": operator,
                "code_hash": sha256_text(code),
            }
        }
        (candidates / f"{candidate_id}.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_audits_import_success_and_failure_without_mutation(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_candidate(root, "op_a", "good", "def op_a(x):\n    return x\n")
            self._write_candidate(
                root,
                "op_b",
                "bad",
                "raise TypeError('import failed')\ndef op_b(x):\n    return x\n",
            )
            before = sorted(
                path.read_bytes() for path in (root / "candidates").rglob("*") if path.is_file()
            )
            report = module.audit_candidates(root / "candidates", root / "datasets")
            after = sorted(
                path.read_bytes() for path in (root / "candidates").rglob("*") if path.is_file()
            )

        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["status_counts"], {"imported": 1, "import_error": 1})
        self.assertEqual(
            report["recovery_counts"],
            {"requires_candidate_repair": 2},
        )
        self.assertEqual(before, after)
        self.assertEqual(
            {(item["operator"], item["status"]) for item in report["records"]},
            {("op_a", "imported"), ("op_b", "import_error")},
        )
        by_operator = {item["operator"]: item for item in report["records"]}
        self.assertFalse(by_operator["op_a"]["static_passed"])
        self.assertTrue(by_operator["op_a"]["target_device_ok"])

    def test_hash_mismatch_is_reported_without_running_candidate(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_candidate(root, "op_a", "bad-hash", "def op_a(x):\n    return x\n")
            manifest_path = root / "candidates/op_a/bad-hash.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["candidate"]["code_hash"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = module.audit_candidates(root / "candidates", root / "datasets")

        self.assertEqual(report["status_counts"], {"manifest_error": 1})
        self.assertEqual(report["recovery_counts"], {"requires_manifest_repair": 1})
        self.assertEqual(report["records"][0]["error_type"], "CodeHashMismatch")

    def test_recovery_recommendations_are_fail_closed(self) -> None:
        module = load_module()
        cases = (
            ({"status": "imported", "static_passed": True}, "eligible_for_manifest_backfill_review"),
            (
                {
                    "status": "import_error",
                    "static_passed": True,
                    "error_type": "ModuleNotFoundError",
                },
                "requires_target_environment_recheck",
            ),
            (
                {"status": "import_error", "static_passed": True, "error_type": "TypeError"},
                "requires_manual_review",
            ),
            ({"status": "imported", "static_passed": False}, "requires_candidate_repair"),
            ({"status": "manifest_error"}, "requires_manifest_repair"),
        )
        for record, expected in cases:
            with self.subTest(record=record):
                self.assertEqual(module._recovery_recommendation(record), expected)

    def test_backfill_preserves_failure_and_requires_exact_dependency_recheck(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_candidate(root, "op_a", "child", "def op_a(x):\n    return x\n")
            manifest_path = root / "candidates/op_a/child.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "static_evaluation": {"passed": True},
                    "import_evaluation": {
                        "status": "import_error",
                        "phase": "module_import",
                        "error_type": "ModuleNotFoundError",
                        "error_message": "No module named 'torch'",
                    },
                    "rejection_error": "Generated candidate failed import gate: missing torch",
                }
            )
            manifest["candidate"]["status"] = "rejected"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            record = {
                "manifest_path": str(manifest_path),
                "code_hash": manifest["candidate"]["code_hash"],
                "status": "imported",
                "phase": "module_import",
                "static_passed": True,
            }

            module._backfill_import_pass(record, "/target/python")
            recovered = json.loads(manifest_path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "not an eligible"):
                module._backfill_import_pass(record, "/target/python")

        self.assertEqual(recovered["candidate"]["status"], "static_pass")
        self.assertIsNone(recovered["rejection_error"])
        self.assertEqual(recovered["import_evaluation"]["status"], "imported")
        self.assertEqual(recovered["import_evaluation_history"][0]["error_type"], "ModuleNotFoundError")
        self.assertEqual(recovered["recovery_history"][0]["python_executable"], "/target/python")


if __name__ == "__main__":
    unittest.main()
