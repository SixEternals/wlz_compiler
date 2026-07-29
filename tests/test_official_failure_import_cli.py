import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/import_official_failures.py"
TARGET = {
    "contest_id": "1mTsU6jaSZ0",
    "task_id": "14955089",
    "assignment_id": "47585",
    "problem_id": "3153461",
    "task_title": "2026年编译挑战赛-基于进化算法的Triton自动优化系统",
    "task_url": (
        "https://course.educg.net/pages/contest/contest_submit.jsp?"
        "contestID=1mTsU6jaSZ0&taskID=14955089&my=false&contestCID=0"
    ),
}


class OfficialFailureImportCliTests(unittest.TestCase):
    def _fixture(
        self, root: Path, run_name: str = "20260721-013859-overlay-smoke"
    ) -> Path:
        run = root / run_name
        run.mkdir()
        base_zip = root / "base.zip"
        base_zip.write_bytes(b"base artifact identity")
        base_sha = hashlib.sha256(base_zip.read_bytes()).hexdigest()
        old_code, preserved_code, replacement_code = "old", "preserved", "replacement"
        candidates = {
            "_replaced": ("new-id", replacement_code),
            "_preserved": ("base-id", preserved_code),
        }
        archive = root / "submission.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for operator, (candidate_id, code) in candidates.items():
                output.writestr(f"output/{operator}/{operator}_v1.py", code)
                output.writestr(
                    f"output/{operator}/{operator}_stats.json",
                    json.dumps({"top5_summary": [{"id": candidate_id}]}),
                )
        with zipfile.ZipFile(archive) as output:
            entries = output.namelist()
        base_selections = [
            {"operator": "_replaced", "candidate_id": "old-id",
             "candidate_sha256": hashlib.sha256(old_code.encode()).hexdigest()},
            {"operator": "_preserved", "candidate_id": "base-id",
             "candidate_sha256": hashlib.sha256(preserved_code.encode()).hexdigest()},
        ]
        (root / "base.manifest.json").write_text(json.dumps({
            "artifact_path": base_zip.name,
            "artifact_sha256": base_sha,
            "selections": base_selections,
        }), encoding="utf-8")
        artifact_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        (root / "submission.manifest.json").write_text(json.dumps({
            "artifact_path": archive.name,
            "artifact_sha256": artifact_sha,
            "archive_entries": entries,
            "base_artifact": {"path": str(base_zip.relative_to(ROOT)), "sha256": base_sha},
            "replacements": [{
                "operator": "_replaced", "candidate_id": "new-id",
                "candidate_sha256": hashlib.sha256(replacement_code.encode()).hexdigest(),
            }],
        }), encoding="utf-8")
        metadata = {
            "submission_time": "2026-07-21T01:38:59+08:00",
            "target": TARGET,
            "result_levels": {"upload_accepted": True, "platform_run_completed": True},
            "official_result": {"failed_tasks": 2},
            "artifact": {
                "filename": archive.name,
                "local_path": str(archive.relative_to(ROOT)),
                "sha256": artifact_sha,
                "size_bytes": archive.stat().st_size,
            },
        }
        (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (run / "raw-result.txt").write_text(
            "=== 失败任务 ===\n"
            "_replaced tc1 _replaced_v1: runtime error (returncode=0)\n"
            "_preserved tc2 _preserved_v1: accuracy check failed (returncode=0)\n",
            encoding="utf-8",
        )
        return run

    def test_completion_timestamp_directory_keeps_submission_observation_id(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            run = self._fixture(root, "20260721-014811-overlay-smoke")
            metadata_path = run / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["completion_observed_at"] = "2026-07-21T01:48:11+08:00"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            (run / "raw-result.txt").write_text(
                "=== 失败任务 ===\n"
                "_replaced tc1 _replaced_v1: msprof no data (log tail: first\n"
                "second\n"
                ") (returncode=0)\n"
                "_preserved tc2 _preserved_v1: accuracy check failed (returncode=0)\n",
                encoding="utf-8",
            )
            history = root / "failures.jsonl"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--run-dir", str(run),
                 "--history", str(history)],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            self.assertEqual(json.loads(result.stdout)["observation_id"], "20260721-013859")
            records = [json.loads(line) for line in history.read_text().splitlines()]
            replaced = next(record for record in records if record["operator"] == "_replaced")
            self.assertEqual(replaced["task_failure"]["failure_kind"], "profiling_no_data")
            self.assertIn("second", replaced["task_failure"]["detail"])

    def test_overlay_run_import_is_idempotent_and_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            run = self._fixture(root)
            history = root / "failures.jsonl"
            command = [sys.executable, str(SCRIPT), "--run-dir", str(run),
                       "--history", str(history)]
            first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
            second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
            first_payload, second_payload = json.loads(first.stdout), json.loads(second.stdout)
            self.assertEqual(
                set(first_payload),
                {"action", "appended", "existing", "failures", "history_keys", "observation_id"},
            )
            self.assertEqual((first_payload["appended"], first_payload["existing"]), (2, 0))
            self.assertEqual((second_payload["appended"], second_payload["existing"]), (0, 2))
            records = [json.loads(line) for line in history.read_text().splitlines()]
            self.assertEqual({entry["candidate_id"] for entry in records}, {"new-id", "base-id"})
            self.assertEqual(
                {entry["candidate_id"]: entry["candidate_code_hash"] for entry in records},
                {
                    "new-id": hashlib.sha256(b"replacement").hexdigest(),
                    "base-id": hashlib.sha256(b"preserved").hexdigest(),
                },
            )

            before = history.read_bytes()
            base_zip = root / "base.zip"
            base_bytes = base_zip.read_bytes()
            base_zip.write_bytes(b"tampered")
            rejected = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("base artifact SHA-256 mismatch", rejected.stderr)
            self.assertEqual(history.read_bytes(), before)
            base_zip.write_bytes(base_bytes)

            manifest_path = root / "submission.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifact_path"] = "../../submission.zip"
            manifest_path.write_text(json.dumps(manifest))
            rejected = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("manifest artifact path mismatch", rejected.stderr)
            self.assertEqual(history.read_bytes(), before)

            metadata_path = run / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["result_levels"]["platform_run_completed"] = False
            metadata_path.write_text(json.dumps(metadata))
            rejected = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("not a completed accepted submission", rejected.stderr)
            self.assertEqual(history.read_bytes(), before)

    def test_untrusted_metadata_is_rejected_before_history_creation(self) -> None:
        cases = {
            "wrong-target": "fixed CourseGrading task",
            "duplicate-json": "Duplicate JSON key",
            "outside-path": "must remain inside the repository",
            "bool-failure-count": "failure count mismatch",
            "wrong-failure-count": "failure count mismatch",
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            for name, message in cases.items():
                with self.subTest(name=name):
                    case_root = root / name
                    case_root.mkdir()
                    run = self._fixture(case_root)
                    metadata_path = run / "metadata.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if name == "wrong-target":
                        metadata["target"]["task_id"] = "wrong"
                    elif name == "outside-path":
                        metadata["artifact"]["local_path"] = "/tmp/outside.zip"
                    elif name == "bool-failure-count":
                        metadata["official_result"]["failed_tasks"] = True
                    elif name == "wrong-failure-count":
                        metadata["official_result"]["failed_tasks"] = 3
                    content = json.dumps(metadata)
                    if name == "duplicate-json":
                        content = content.replace(
                            '"task_id": "14955089"',
                            '"task_id": "14955089", "task_id": "14955089"',
                        )
                    metadata_path.write_text(content, encoding="utf-8")
                    history = case_root / "failures.jsonl"
                    result = subprocess.run(
                        [sys.executable, str(SCRIPT), "--run-dir", str(run),
                         "--history", str(history)],
                        cwd=ROOT, text=True, capture_output=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(history.exists())


if __name__ == "__main__":
    unittest.main()
