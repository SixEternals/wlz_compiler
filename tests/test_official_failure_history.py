import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from wlz_optimizer.cache import (
    OfficialFailureHistory,
    make_official_failure_record,
    validate_official_failure_record,
)
from wlz_optimizer.official_adapter import BoundOfficialTaskFailure, OfficialTaskFailure


ENV = (
    "coursegrading:contest=1mTsU6jaSZ0:task=14955089:problem=3153461:"
    "assign=47585:observation=20260721-013859"
)


class OfficialFailureRecordTests(unittest.TestCase):
    def _failure(self) -> BoundOfficialTaskFailure:
        raw = "_op tc2 _op_v1: runtime error (first detail) (returncode=0)"
        task = OfficialTaskFailure(
            "_op", "tc2", "_op_v1", "runtime_error",
            "runtime error (first detail)", 0, raw,
        )
        return BoundOfficialTaskFailure("candidate-1", "_op", "a" * 64, task)

    @staticmethod
    def _resign(record: dict) -> dict:
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        encoded = json.dumps(
            body, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
        record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
        return record

    def test_record_roundtrip_golden_and_unknown(self) -> None:
        failure = self._failure()
        record = make_official_failure_record(
            failure, ENV, "20260721-013859", created_at="2026-07-21T01:48:11+08:00"
        )
        self.assertEqual(validate_official_failure_record(record), failure)
        self.assertEqual(
            record["record_sha256"],
            "4ab2b801f5b6b48e0740024f0622b1d1a0154153ce8c33d0f116e8ce9e64a398",
        )
        unknown_task = replace(
            failure.task_failure, failure_kind="unknown", detail="platform-specific failure",
            raw_line="_op tc2 _op_v1: platform-specific failure", returncode=None,
        )
        unknown = replace(failure, task_failure=unknown_task)
        self.assertEqual(
            validate_official_failure_record(
                make_official_failure_record(unknown, ENV, "20260721-013859")
            ),
            unknown,
        )

    def test_record_accepts_ranked_variants_and_rejects_out_of_range(self) -> None:
        failure = self._failure()
        for rank in (2, 5):
            with self.subTest(rank=rank):
                variant = f"_op_v{rank}"
                task = replace(
                    failure.task_failure,
                    candidate_variant=variant,
                    raw_line=f"_op tc2 {variant}: runtime error (first detail) (returncode=0)",
                )
                ranked = replace(failure, task_failure=task)
                self.assertEqual(
                    validate_official_failure_record(
                        make_official_failure_record(ranked, ENV, "20260721-013859")
                    ),
                    ranked,
                )

        for rank in (0, 6):
            with self.subTest(rank=rank), self.assertRaisesRegex(
                ValueError, "candidate variant mismatch"
            ):
                variant = f"_op_v{rank}"
                task = replace(
                    failure.task_failure,
                    candidate_variant=variant,
                    raw_line=f"_op tc2 {variant}: runtime error (first detail) (returncode=0)",
                )
                make_official_failure_record(
                    replace(failure, task_failure=task), ENV, "20260721-013859"
                )

    def test_record_rejects_corrupt_or_ambiguous_identity(self) -> None:
        failure = self._failure()
        record = make_official_failure_record(failure, ENV, "20260721-013859")
        with self.assertRaisesRegex(ValueError, "record hash mismatch"):
            validate_official_failure_record({**record, "candidate_id": "tampered"})
        inconsistent = replace(failure.task_failure, detail="runtime error (different)")
        with self.assertRaisesRegex(ValueError, "raw line mismatch"):
            make_official_failure_record(
                replace(failure, task_failure=inconsistent), ENV, "20260721-013859"
            )
        for version in (True, 1.0):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "schema"):
                validate_official_failure_record(
                    self._resign({**record, "schema_version": version})
                )
        for returncode in (False, 0.0):
            with self.subTest(returncode=returncode), self.assertRaisesRegex(ValueError, "returncode"):
                make_official_failure_record(
                    replace(failure, task_failure=replace(failure.task_failure, returncode=returncode)),
                    ENV, "20260721-013859",
                )
        for timestamp in ("", False, 0):
            with self.subTest(timestamp=timestamp), self.assertRaises((TypeError, ValueError)):
                make_official_failure_record(
                    failure, ENV, "20260721-013859", created_at=timestamp
                )
        with self.assertRaisesRegex(ValueError, "candidate ID"):
            make_official_failure_record(
                replace(failure, candidate_id=" "), ENV, "20260721-013859"
            )
        for field, value, message in (
            ("key", "0" * 64, "identity mismatch"),
            ("observation_id", "other", "observation identity mismatch"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                validate_official_failure_record(self._resign({**record, field: value}))
        variant = json.loads(json.dumps(record))
        variant["task_failure"].update(
            candidate_variant="_other_v1",
            raw_line="_op tc2 _other_v1: runtime error (first detail) (returncode=0)",
        )
        with self.assertRaisesRegex(ValueError, "candidate variant mismatch"):
            validate_official_failure_record(self._resign(variant))

    def test_jsonl_append_is_idempotent_and_conflicts_fail_closed(self) -> None:
        failure = self._failure()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.jsonl"
            history = OfficialFailureHistory(path)
            key = history.append(failure, ENV, "20260721-013859")
            self.assertEqual(history.append(failure, ENV, "20260721-013859"), key)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len(OfficialFailureHistory(path)), 1)

            changed_task = replace(
                failure.task_failure,
                detail="runtime error (second detail)",
                raw_line="_op tc2 _op_v1: runtime error (second detail) (returncode=0)",
            )
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                history.append(
                    replace(failure, task_failure=changed_task), ENV, "20260721-013859"
                )

    def test_jsonl_load_rejects_bad_or_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.jsonl"
            OfficialFailureHistory(path).append(
                self._failure(), ENV, "20260721-013859"
            )
            line = path.read_text(encoding="utf-8")
            cases = (
                (
                    line.replace(
                        '"schema_version": 1',
                        '"schema_version": 1, "schema_version": 1',
                    ),
                    "line 1.*Duplicate JSON key",
                ),
                (line + line, "line 2.*duplicate"),
                (line.rstrip("\n"), "line 1.*terminating newline"),
                (line.rstrip()[:-1] + "\n", "line 1"),
            )
            for content, message in cases:
                path.write_text(content, encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    OfficialFailureHistory(path)

    def test_jsonl_append_many_preflights_before_writing(self) -> None:
        failure = self._failure()
        old_tc2 = replace(
            failure,
            task_failure=replace(
                failure.task_failure,
                detail="runtime error (old detail)",
                raw_line="_op tc2 _op_v1: runtime error (old detail) (returncode=0)",
            ),
        )
        tc1 = replace(
            failure,
            task_failure=replace(
                failure.task_failure,
                test_case="tc1",
                raw_line="_op tc1 _op_v1: runtime error (first detail) (returncode=0)",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.jsonl"
            history = OfficialFailureHistory(path)
            history.append(old_tc2, ENV, "20260721-013859")
            original = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "Conflicting"):
                history.append_many([tc1, failure], ENV, "20260721-013859")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(history), 1)

            with self.assertRaisesRegex(ValueError, "Duplicate"):
                history.append_many([tc1, tc1], ENV, "20260721-013859")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                history.append_many([old_tc2, old_tc2], ENV, "20260721-013859")
            self.assertEqual(path.read_bytes(), original)

    def test_jsonl_append_many_noop_order_and_fsync_failure(self) -> None:
        failure = self._failure()
        tc1 = replace(
            failure,
            task_failure=replace(
                failure.task_failure,
                test_case="tc1",
                raw_line="_op tc1 _op_v1: runtime error (first detail) (returncode=0)",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "failures.jsonl"
            history = OfficialFailureHistory(path)
            self.assertEqual(history.append_many([], "invalid", "invalid"), [])
            self.assertFalse(path.exists())

            existing = history.append(failure, ENV, "20260721-013859")
            keys = history.append_many([failure, tc1], ENV, "20260721-013859")
            self.assertEqual(keys[0], existing)
            self.assertEqual(list(history.entries), keys)

            failed = OfficialFailureHistory(root / "fsync-failure.jsonl")
            with patch("wlz_optimizer.cache.os.fsync", side_effect=OSError("fsync")):
                with self.assertRaisesRegex(OSError, "fsync"):
                    failed.append_many([failure], ENV, "20260721-013859")
            self.assertEqual(len(failed), 0)


if __name__ == "__main__":
    unittest.main()
