import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from wlz_optimizer.hash_utils import sha256_text

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "assemble_local_admission.py"
    spec = importlib.util.spec_from_file_location("assemble_local_admission_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AssembleLocalAdmissionTests(unittest.TestCase):
    def _fixture(self, root: Path, *, derived_import: bool = False):
        operator = "op_a"
        candidate_dir = root / "candidates" / operator
        candidate_dir.mkdir(parents=True)
        code = (
            "import definitely_missing_torch\ndef op_a(x):\n    return x\n"
            if derived_import
            else "def op_a(x):\n    return x\n"
        )
        candidate_id = "localv-test123456"
        candidate_path = candidate_dir / f"{candidate_id}.py"
        candidate_path.write_text(code, encoding="utf-8")
        candidate_hash = sha256_text(code)
        baseline_hash = "b" * 64
        manifest_path = candidate_dir / f"{candidate_id}.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "candidate": {
                        "id": candidate_id,
                        "op_name": operator,
                        "code_hash": candidate_hash,
                        "status": "static_pass",
                    },
                    "parent_sha256": baseline_hash,
                    "rejection_error": None,
                    "static_evaluation": {"passed": True},
                    "import_evaluation": {"status": "not_run", "phase": "not_run"},
                }
            ),
            encoding="utf-8",
        )

        def evidence(path: Path, test_sha: str):
            path.write_text(
                json.dumps(
                    {
                        "baseline": {"source_sha256": baseline_hash},
                        "candidate": {
                            "id": candidate_id,
                            "source_sha256": candidate_hash,
                        },
                        "operator": operator,
                        "correctness_status": "passed",
                        "process_status": "passed",
                        "cases": [
                            {
                                "status": "passed",
                                "test_path": f"tests/{test_sha}.py",
                                "test_sha256": test_sha,
                                "baseline_run": {"status": "passed"},
                                "candidate_run": (
                                    {
                                        "role": "candidate",
                                        "prepared_source_sha256": candidate_hash,
                                        "python_executable": "/opt/ascend/bin/python",
                                        "returncode": 0,
                                        "status": "passed",
                                    }
                                    if derived_import
                                    else {"status": "passed"}
                                ),
                            }
                        ],
                        "environment": {"fingerprint_sha256": "f" * 64},
                    }
                ),
                encoding="utf-8",
            )

        visible = root / "visible.json"
        holdout = root / "holdout.json"
        evidence(visible, "1" * 64)
        evidence(holdout, "2" * 64)
        return manifest_path, visible, holdout, candidate_id

    def test_assembles_import_and_holdout_admission(self):
        module = _module()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            manifest, visible, holdout, candidate_id = self._fixture(root)
            output = module.assemble(manifest, visible, holdout, root / "assembled")
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["import_evaluation"]["status"], "imported")
            self.assertTrue(document["correctness_evaluation"]["eligible_for_performance"])
            self.assertEqual(document["holdout_evaluation"]["candidate_id"], candidate_id)
            self.assertFalse(document["holdout_evaluation"]["used_for_search"])
            self.assertTrue((output.with_name(f"{candidate_id}.py")).is_file())
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8")).get("correctness_evaluation"), None)

    def test_rejects_tampered_evidence(self):
        module = _module()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            manifest, visible, holdout, _ = self._fixture(root)
            document = json.loads(holdout.read_text(encoding="utf-8"))
            document["candidate"]["source_sha256"] = "0" * 64
            holdout.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate identity mismatch"):
                module.assemble(manifest, visible, holdout, root / "assembled")

    def test_derives_import_from_ascend_correctness(self):
        module = _module()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            manifest, visible, holdout, _ = self._fixture(root, derived_import=True)
            original = module.run_candidate
            module.run_candidate = lambda request: (_ for _ in ()).throw(
                AssertionError("default interpreter should not be used")
            )
            try:
                output = module.assemble(manifest, visible, holdout, root / "assembled")
            finally:
                module.run_candidate = original
            imported = json.loads(output.read_text(encoding="utf-8"))["import_evaluation"]
            self.assertEqual(imported["status"], "imported")
            self.assertEqual(imported["evidence_source"], "candidate_correctness_run")
            self.assertEqual(imported["python_executable"], "/opt/ascend/bin/python")

    def test_rejects_mixed_ascend_interpreters_for_derived_import(self):
        module = _module()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            manifest, visible, holdout, _ = self._fixture(root, derived_import=True)
            for path, executable in (
                (visible, "/opt/ascend/bin/python"),
                (holdout, "/other/python"),
            ):
                document = json.loads(path.read_text(encoding="utf-8"))
                document["cases"][0]["candidate_run"]["python_executable"] = executable
                path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed interpreters"):
                module.assemble(manifest, visible, holdout, root / "assembled")

    def test_requires_distinct_holdout(self):
        module = _module()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            manifest, visible, _, _ = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "distinct"):
                module.assemble(manifest, visible, visible, root / "assembled")


if __name__ == "__main__":
    unittest.main()
