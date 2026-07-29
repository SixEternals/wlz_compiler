import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.official_adapter import BOUND_EVALUATION_KIND

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/import_official_evaluation.py"


class OfficialEvaluationHistoryCliTests(unittest.TestCase):
    def test_append_replay_and_identity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = "def demo_op(x):\n    return x\n"
            code_hash = sha256_text(code)
            (root / "candidate-1.py").write_text(code, encoding="utf-8")
            manifest = root / "candidate-1.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "candidate": {
                            "id": "candidate-1",
                            "op_name": "demo_op",
                            "code_hash": code_hash,
                            "parent_ids": ["seed"],
                            "generation": 1,
                            "mutation_kind": "mutation",
                            "model_used": "test-model",
                            "prompt_id": "prompt-1",
                            "status": "static_pass",
                            "score": None,
                            "metadata": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
            envelope = root / "envelope.json"
            envelope_data = {
                "schema_version": 1,
                "artifact_kind": BOUND_EVALUATION_KIND,
                "operator": "demo_op",
                "candidate_id": "candidate-1",
                "candidate_code_hash": code_hash,
                "evaluation": {
                    "success": True,
                    "execution_time": 2500.0,
                    "speedup": 0.6,
                    "fitness": 0.6,
                    "error": None,
                },
            }
            envelope.write_text(json.dumps(envelope_data), encoding="utf-8")
            history = root / "official.jsonl"
            common = [
                sys.executable,
                str(SCRIPT),
                "--candidate-manifest",
                str(manifest),
                "--history",
                str(history),
                "--env-fingerprint",
                "ascend-a2-task-14955089",
            ]
            base = [
                *common,
                "--observation-id",
                "official-run-20260721-013859",
            ]

            first = subprocess.run(
                [*base, "append", "--envelope", str(envelope), "--baseline-time-us", "4000"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            second = subprocess.run(
                [*base, "append", "--envelope", str(envelope), "--baseline-time-us", "4000"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            replay = subprocess.run(
                [*base, "replay"], cwd=ROOT, text=True, capture_output=True, check=True
            )
            listed = subprocess.run(
                [*common, "list"], cwd=ROOT, text=True, capture_output=True, check=True
            )

            self.assertEqual(json.loads(first.stdout)["action"], "appended")
            self.assertEqual(json.loads(second.stdout)["action"], "existing")
            self.assertEqual(json.loads(replay.stdout)["action"], "replayed")
            listed_payload = json.loads(listed.stdout)
            self.assertEqual(listed_payload["action"], "listed")
            self.assertEqual(listed_payload["count"], 1)
            self.assertEqual(
                listed_payload["observations"][0]["metadata"]["observation_id"],
                "official-run-20260721-013859",
            )
            self.assertEqual(
                json.loads(replay.stdout)["result"]["metadata"]["observation_id"],
                "official-run-20260721-013859",
            )
            self.assertEqual(len(history.read_text(encoding="utf-8").splitlines()), 1)
            stored = json.loads(history.read_text(encoding="utf-8"))
            self.assertEqual(stored["observation_id"], "official-run-20260721-013859")

            envelope_data["candidate_code_hash"] = "0" * 64
            envelope.write_text(json.dumps(envelope_data), encoding="utf-8")
            rejected = subprocess.run(
                [*base, "append", "--envelope", str(envelope)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("candidate_code_hash", rejected.stderr)
            self.assertEqual(len(history.read_text(encoding="utf-8").splitlines()), 1)

            missing_id = subprocess.run(
                [*common, "replay"],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(missing_id.returncode, 2)


if __name__ == "__main__":
    unittest.main()
