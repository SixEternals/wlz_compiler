"""Focused tests for fail-closed candidate correctness admission."""

import json
import unittest

from wlz_optimizer.correctness import decide_candidate_correctness
from wlz_optimizer.schemas import CorrectnessCaseResult, CorrectnessErrorSummary


SIG_A = "a" * 64
SIG_B = "b" * 64
SIG_C = "c" * 64


def result(signature: str, status: str = "passed", **overrides) -> CorrectnessCaseResult:
    values = {
        "candidate_id": "candidate-1",
        "case_id": f"case-{signature[0]}",
        "case_signature": signature,
        "oracle_policy_id": "fp16-v1",
        "oracle_status": status,
    }
    if status == "failed":
        values["error_summary"] = CorrectnessErrorSummary(
            mismatch_kind="value", mismatch_count=1, compared_count=16
        )
    if status == "oracle_error":
        values["message"] = "reference failed"
    values.update(overrides)
    return CorrectnessCaseResult(**values)


class CandidateCorrectnessGateTests(unittest.TestCase):
    def test_all_expected_cases_pass_exactly_once(self) -> None:
        decision = decide_candidate_correctness(
            "candidate-1", [SIG_B, SIG_A], [result(SIG_A), result(SIG_B)]
        )

        self.assertTrue(decision.eligible_for_performance)
        self.assertEqual(decision.blocking_reasons, ())
        self.assertEqual(decision.expected_signatures, (SIG_A, SIG_B))
        self.assertEqual(decision.passed_signatures, (SIG_A, SIG_B))
        json.dumps(decision.to_dict(), sort_keys=True)

    def test_empty_or_missing_expected_cases_fail_closed(self) -> None:
        empty = decide_candidate_correctness("candidate-1", [], [])
        missing = decide_candidate_correctness("candidate-1", [SIG_A, SIG_B], [result(SIG_A)])

        self.assertEqual(empty.blocking_reasons, ("no_expected_cases",))
        self.assertEqual(missing.missing_signatures, (SIG_B,))
        self.assertFalse(missing.eligible_for_performance)

    def test_each_nonpassing_status_is_classified(self) -> None:
        decision = decide_candidate_correctness(
            "candidate-1",
            [SIG_A, SIG_B, SIG_C],
            [result(SIG_A, "failed"), result(SIG_B, "oracle_error"), result(SIG_C, "unknown")],
        )

        self.assertEqual(decision.failed_signatures, (SIG_A,))
        self.assertEqual(decision.oracle_error_signatures, (SIG_B,))
        self.assertEqual(decision.unknown_signatures, (SIG_C,))
        self.assertEqual(decision.blocking_reasons, ("failed_case", "oracle_error", "unknown_case"))

    def test_duplicate_and_unexpected_results_are_blocking(self) -> None:
        decision = decide_candidate_correctness(
            "candidate-1",
            [SIG_A],
            [result(SIG_A), result(SIG_A, "failed"), result(SIG_C)],
        )

        self.assertEqual(decision.duplicate_signatures, (SIG_A,))
        self.assertEqual(decision.failed_signatures, (SIG_A,))
        self.assertEqual(decision.unexpected_signatures, (SIG_C,))
        self.assertFalse(decision.eligible_for_performance)

    def test_foreign_candidate_result_is_not_ignored_or_used(self) -> None:
        decision = decide_candidate_correctness(
            "candidate-1",
            [SIG_A],
            [result(SIG_A, candidate_id="candidate-2")],
        )

        self.assertEqual(
            decision.blocking_reasons,
            ("foreign_candidate_result", "missing_case_result"),
        )
        self.assertEqual(decision.foreign_results[0].candidate_id, "candidate-2")
        self.assertEqual(decision.missing_signatures, (SIG_A,))

    def test_rejects_invalid_expected_contract_and_result_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            decide_candidate_correctness("candidate-1", [SIG_A, SIG_A], [])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            decide_candidate_correctness("candidate-1", ["invalid"], [])
        with self.assertRaisesRegex(TypeError, "CorrectnessCaseResult"):
            decide_candidate_correctness("candidate-1", [SIG_A], [object()])


if __name__ == "__main__":
    unittest.main()
