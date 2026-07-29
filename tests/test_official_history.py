import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from wlz_optimizer.cache import OfficialEvaluationHistory
from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.official_adapter import (
    BOUND_EVALUATION_KIND,
    adapt_bound_official_evaluation,
)
from wlz_optimizer.schemas import Candidate


class OfficialEvaluationHistoryTests(unittest.TestCase):
    def _candidate(self) -> Candidate:
        code = "def demo_op(x):\n    return x\n"
        return Candidate(
            id="candidate-1",
            op_name="demo_op",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=["seed"],
            generation=1,
            mutation_kind="mutation",
            model_used="test-model",
            prompt_id="prompt-1",
            status="static_pass",
            score=None,
        )

    def _envelope(self, candidate: Candidate, speedup: float = 0.6) -> dict:
        return {
            "schema_version": 1,
            "artifact_kind": BOUND_EVALUATION_KIND,
            "operator": candidate.op_name,
            "candidate_id": candidate.id,
            "candidate_code_hash": candidate.code_hash,
            "evaluation": {
                "success": True,
                "execution_time": 2500.0,
                "speedup": speedup,
                "fitness": speedup,
                "error": None,
            },
        }

    def test_append_is_idempotent_and_replay_preserves_official_unknowns(self) -> None:
        candidate = self._candidate()
        envelope = self._envelope(candidate)
        result = adapt_bound_official_evaluation(
            candidate, envelope, baseline_time_us=4000.0
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "official.jsonl"
            history = OfficialEvaluationHistory(path)
            key = history.append(
                candidate, envelope, result, "ascend-a2-task-14955089", "run-001"
            )
            self.assertEqual(
                history.append(
                    candidate, envelope, result, "ascend-a2-task-14955089", "run-001"
                ),
                key,
            )
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

            replayed = OfficialEvaluationHistory(path).replay(
                candidate, "ascend-a2-task-14955089", "run-001"
            )

        self.assertIsNotNone(replayed)
        self.assertTrue(replayed.metadata["history_replay"])
        self.assertEqual(replayed.metadata["observation_id"], "run-001")
        self.assertIsNone(replayed.compile_ok)
        self.assertIsNone(replayed.correctness_ok)
        self.assertIsNone(candidate.score)

    def test_conflicting_result_and_corrupt_history_fail_closed(self) -> None:
        candidate = self._candidate()
        first_envelope = self._envelope(candidate)
        first_result = adapt_bound_official_evaluation(candidate, first_envelope)
        second_envelope = self._envelope(candidate, speedup=0.7)
        second_result = adapt_bound_official_evaluation(candidate, second_envelope)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "official.jsonl"
            history = OfficialEvaluationHistory(path)
            history.append(candidate, first_envelope, first_result, "ascend-a2", "run-001")
            history.append(candidate, second_envelope, second_result, "ascend-a2", "run-002")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(
                history.replay(candidate, "ascend-a2", "run-002").speedup, 0.7
            )
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                history.append(
                    candidate, second_envelope, second_result, "ascend-a2", "run-001"
                )

            entries = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            entries[0]["key"] = "0" * 64
            path.write_text(
                "\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "line 1"):
                OfficialEvaluationHistory(path)

    def test_list_observations_filters_candidate_and_environment_without_writing(self) -> None:
        candidate = self._candidate()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "official.jsonl"
            history = OfficialEvaluationHistory(path)
            for observation_id, speedup, environment in (
                ("run-001", 0.6, "ascend-a2"),
                ("run-002", 0.7, "ascend-a2"),
                ("run-003", 0.8, "ascend-a3"),
            ):
                envelope = self._envelope(candidate, speedup=speedup)
                result = adapt_bound_official_evaluation(candidate, envelope)
                history.append(
                    candidate, envelope, result, environment, observation_id
                )
            other_candidate = replace(candidate, id="candidate-2")
            other_envelope = self._envelope(other_candidate, speedup=0.9)
            history.append(
                other_candidate,
                other_envelope,
                adapt_bound_official_evaluation(other_candidate, other_envelope),
                "ascend-a2",
                "run-004",
            )
            original = path.read_bytes()

            observations = history.list_observations(candidate, "ascend-a2")

            self.assertEqual(
                [result.metadata["observation_id"] for result in observations],
                ["run-001", "run-002"],
            )
            self.assertEqual([result.speedup for result in observations], [0.6, 0.7])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
