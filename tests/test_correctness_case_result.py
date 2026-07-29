"""Focused tests for per-case correctness result contracts."""

import json
import unittest

from wlz_optimizer.schemas import CorrectnessCaseResult, CorrectnessErrorSummary


CASE_SIGNATURE = "a" * 64


def summary(**overrides) -> CorrectnessErrorSummary:
    values = {
        "max_abs_error": 0.001,
        "max_rel_error": 0.002,
        "mismatch_count": 0,
        "compared_count": 4096,
    }
    values.update(overrides)
    return CorrectnessErrorSummary(**values)


def result(**overrides) -> CorrectnessCaseResult:
    values = {
        "candidate_id": "candidate-1",
        "case_id": "public-case-1",
        "case_signature": CASE_SIGNATURE,
        "oracle_policy_id": "fp16-v1",
        "oracle_status": "passed",
        "error_summary": summary(),
    }
    values.update(overrides)
    return CorrectnessCaseResult(**values)


class CorrectnessCaseResultTests(unittest.TestCase):
    def test_json_roundtrip_preserves_passed_error_metrics(self) -> None:
        original = result()
        restored = CorrectnessCaseResult.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )

        self.assertEqual(restored, original)
        self.assertEqual(restored.error_summary.max_abs_error, 0.001)

    def test_failed_result_records_structured_mismatch(self) -> None:
        mismatch = summary(
            mismatch_kind="shape",
            max_abs_error=None,
            max_rel_error=None,
            mismatch_count=1,
            compared_count=1,
            first_mismatch="output[0]: expected [32, 128], got [32, 64]",
        )
        failed = result(oracle_status="failed", error_summary=mismatch)

        self.assertEqual(failed.error_summary.mismatch_kind, "shape")
        self.assertEqual(failed.oracle_status, "failed")

    def test_unknown_preserves_missing_oracle_details(self) -> None:
        unknown = result(oracle_status="unknown", error_summary=None)

        self.assertIsNone(unknown.error_summary)
        self.assertIsNone(unknown.message)

    def test_rejects_invalid_identity_signature_and_status(self) -> None:
        invalid = (
            {"candidate_id": ""},
            {"case_id": ""},
            {"case_signature": "not-a-hash"},
            {"oracle_policy_id": ""},
            {"oracle_status": "success"},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                result(**values)

    def test_rejects_invalid_error_metrics_and_counts(self) -> None:
        invalid = (
            {"max_abs_error": -1.0},
            {"max_rel_error": float("inf")},
            {"mismatch_count": True},
            {"mismatch_count": 2, "compared_count": 1},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                summary(**values)

    def test_enforces_oracle_status_consistency(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an error_summary"):
            result(oracle_status="failed", error_summary=None)
        with self.assertRaisesRegex(ValueError, "requires a message"):
            result(oracle_status="oracle_error", error_summary=None)
        with self.assertRaisesRegex(ValueError, "must not contain"):
            result(oracle_status="unknown", error_summary=summary())
        with self.assertRaisesRegex(ValueError, "cannot report mismatches"):
            result(error_summary=summary(mismatch_count=1))


if __name__ == "__main__":
    unittest.main()
