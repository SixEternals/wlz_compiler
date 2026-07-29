"""JSONL evaluation cache."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.official_adapter import (
    EXACT_FAILURE_SIGNATURE_SCHEMA,
    OFFICIAL_EXECUTOR_KIND,
    OFFICIAL_FRAMEWORK_COMMIT,
    BoundOfficialTaskFailure,
    OfficialTaskFailure,
    adapt_bound_official_evaluation,
    make_exact_official_failure_signature,
    official_failure_observation_id,
    parse_official_task_failures,
)
from wlz_optimizer.schemas import Candidate, EvaluationResult


class EvaluationCache:
    """Append-only JSONL cache keyed by operator, hash, executor, and env."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = entry.get("key")
            if isinstance(key, str):
                self.entries[key] = entry

    @staticmethod
    def make_key(
        op_name: str,
        code_hash: str,
        executor_kind: str,
        env_fingerprint: str,
        mutation_kind: str = "",
        baseline_hash: str = "",
    ) -> str:
        raw = "\n".join(
            [op_name, code_hash, executor_kind, env_fingerprint, mutation_kind, baseline_hash]
        )
        return sha256_text(raw)

    def _candidate_key(
        self,
        candidate: Candidate,
        executor_kind: str,
        env_fingerprint: str,
        baseline_file: Optional[Path],
    ) -> str:
        return self.make_key(
            candidate.op_name,
            candidate.code_hash,
            executor_kind,
            env_fingerprint,
            candidate.mutation_kind,
            _baseline_hash(baseline_file),
        )

    def get(
        self,
        candidate: Candidate,
        executor_kind: str,
        env_fingerprint: str,
        baseline_file: Optional[Path] = None,
    ) -> Optional[EvaluationResult]:
        key = self._candidate_key(candidate, executor_kind, env_fingerprint, baseline_file)
        entry = self.entries.get(key)
        if not entry:
            return None

        cached_result = EvaluationResult.from_dict(entry["result"])
        metadata = dict(cached_result.metadata or {})
        metadata["cache_hit"] = True
        metadata["cache_key"] = key
        metadata["cached_candidate_id"] = cached_result.candidate_id
        return replace(cached_result, candidate_id=candidate.id, metadata=metadata)

    def put(
        self,
        candidate: Candidate,
        result: EvaluationResult,
        executor_kind: str,
        env_fingerprint: str,
        baseline_file: Optional[Path] = None,
    ) -> str:
        key = self._candidate_key(candidate, executor_kind, env_fingerprint, baseline_file)
        if key in self.entries:
            return key

        metadata = dict(result.metadata or {})
        metadata["cache_hit"] = False
        metadata["cache_key"] = key
        result = replace(result, metadata=metadata)
        entry = {
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "op_name": candidate.op_name,
            "code_hash": candidate.code_hash,
            "executor": executor_kind,
            "env_fingerprint": env_fingerprint,
            "candidate_id": candidate.id,
            "result": result.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        self.entries[key] = entry
        return key

    def __len__(self) -> int:
        return len(self.entries)


def _baseline_hash(baseline_file: Optional[Path]) -> str:
    if baseline_file is None or not baseline_file.is_file():
        return ""
    return sha256_text(baseline_file.read_text(encoding="utf-8"))


class OfficialEvaluationHistory:
    """Append-only, identity-bound history for offline official replay."""

    ARTIFACT_KIND = "official-evaluation-history-entry"
    DEFAULT_OBSERVATION_ID = "legacy-singleton"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            return
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                key = self.make_key(
                    entry["op_name"],
                    entry["candidate_id"],
                    entry["code_hash"],
                    entry["env_fingerprint"],
                    entry["observation_id"],
                )
                if (
                    entry.get("schema_version") != 2
                    or entry.get("artifact_kind") != self.ARTIFACT_KIND
                    or entry.get("executor") != OFFICIAL_EXECUTOR_KIND
                    or entry.get("key") != key
                    or key in self.entries
                ):
                    raise ValueError("invalid official history identity")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid official evaluation history at line {line_number}: {exc}"
                ) from exc
            self.entries[key] = entry

    @staticmethod
    def make_key(
        op_name: str,
        candidate_id: str,
        code_hash: str,
        env_fingerprint: str,
        observation_id: str = DEFAULT_OBSERVATION_ID,
    ) -> str:
        if not all(
            isinstance(value, str) and value
            for value in (
                op_name,
                candidate_id,
                code_hash,
                env_fingerprint,
                observation_id,
            )
        ):
            raise ValueError("Official history identity fields must be non-empty strings")
        return sha256_text(
            "\n".join(
                [
                    op_name,
                    candidate_id,
                    code_hash,
                    OFFICIAL_EXECUTOR_KIND,
                    env_fingerprint,
                    observation_id,
                ]
            )
        )

    def append(
        self,
        candidate: Candidate,
        envelope: Mapping[str, Any],
        result: EvaluationResult,
        env_fingerprint: str,
        observation_id: str = DEFAULT_OBSERVATION_ID,
    ) -> str:
        baseline_us = result.baseline_ms * 1000.0 if result.baseline_ms is not None else None
        expected = adapt_bound_official_evaluation(
            candidate, envelope, baseline_time_us=baseline_us
        )
        if result.to_dict() != expected.to_dict():
            raise ValueError("Official result does not match its bound raw envelope")
        key = self.make_key(
            candidate.op_name,
            candidate.id,
            candidate.code_hash,
            env_fingerprint,
            observation_id,
        )
        payload = {
            "schema_version": 2,
            "artifact_kind": self.ARTIFACT_KIND,
            "op_name": candidate.op_name,
            "candidate_id": candidate.id,
            "code_hash": candidate.code_hash,
            "executor": OFFICIAL_EXECUTOR_KIND,
            "env_fingerprint": env_fingerprint,
            "observation_id": observation_id,
            "raw_envelope": dict(envelope),
            "result": result.to_dict(),
        }
        existing = self.entries.get(key)
        if existing is not None:
            if all(existing.get(field) == value for field, value in payload.items()):
                return key
            raise ValueError("Conflicting official evaluation already exists")
        entry = {"key": key, "created_at": datetime.now(timezone.utc).isoformat(), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.entries[key] = entry
        return key

    def replay(
        self,
        candidate: Candidate,
        env_fingerprint: str,
        observation_id: str = DEFAULT_OBSERVATION_ID,
    ) -> Optional[EvaluationResult]:
        key = self.make_key(
            candidate.op_name,
            candidate.id,
            candidate.code_hash,
            env_fingerprint,
            observation_id,
        )
        entry = self.entries.get(key)
        if entry is None:
            return None
        stored = EvaluationResult.from_dict(entry["result"])
        baseline_us = stored.baseline_ms * 1000.0 if stored.baseline_ms is not None else None
        expected = adapt_bound_official_evaluation(
            candidate, entry["raw_envelope"], baseline_time_us=baseline_us
        )
        if stored.to_dict() != expected.to_dict():
            raise ValueError("Stored official evaluation no longer matches its raw envelope")
        metadata = {
            **stored.metadata,
            "history_replay": True,
            "history_key": key,
            "observation_id": observation_id,
        }
        return replace(stored, metadata=metadata)

    def list_observations(
        self, candidate: Candidate, env_fingerprint: str
    ) -> list[EvaluationResult]:
        """Return this candidate's validated observations in append order."""
        results = []
        for entry in self.entries.values():
            if (
                entry["op_name"] != candidate.op_name
                or entry["candidate_id"] != candidate.id
                or entry["code_hash"] != candidate.code_hash
                or entry["env_fingerprint"] != env_fingerprint
            ):
                continue
            result = self.replay(
                candidate, env_fingerprint, entry["observation_id"]
            )
            if result is None:  # pragma: no cover - guarded by the entry match above
                raise ValueError("Official history entry disappeared during query")
            results.append(result)
        return results


OFFICIAL_FAILURE_RECORD_KIND = "official-task-failure-history-entry"
_OFFICIAL_FAILURE_TASK_FIELDS = {
    "operator", "test_case", "candidate_variant", "failure_kind",
    "detail", "returncode", "raw_line",
}
_OFFICIAL_FAILURE_RECORD_FIELDS = {
    "schema_version", "artifact_kind", "signature_schema", "executor",
    "official_framework_commit", "key", "created_at", "record_sha256",
    "candidate_id", "operator", "candidate_code_hash", "env_fingerprint",
    "observation_id", "task_failure",
}


def make_official_failure_record(
    failure: BoundOfficialTaskFailure,
    env_fingerprint: str,
    observation_id: str,
    *,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one integrity-checked, JSON-serializable official failure record."""
    if official_failure_observation_id(env_fingerprint) != observation_id:
        raise ValueError("Official failure observation identity mismatch")
    task = failure.task_failure
    record = {
        "schema_version": 1,
        "artifact_kind": OFFICIAL_FAILURE_RECORD_KIND,
        "signature_schema": EXACT_FAILURE_SIGNATURE_SCHEMA,
        "executor": OFFICIAL_EXECUTOR_KIND,
        "official_framework_commit": OFFICIAL_FRAMEWORK_COMMIT,
        "key": make_exact_official_failure_signature(failure, env_fingerprint),
        "created_at": (
            created_at if created_at is not None else datetime.now(timezone.utc).isoformat()
        ),
        "candidate_id": failure.candidate_id,
        "operator": failure.operator,
        "candidate_code_hash": failure.candidate_code_hash,
        "env_fingerprint": env_fingerprint,
        "observation_id": observation_id,
        "task_failure": {
            field: getattr(task, field) for field in _OFFICIAL_FAILURE_TASK_FIELDS
        },
    }
    record["record_sha256"] = _record_sha256(record)
    validate_official_failure_record(record)
    return record


def validate_official_failure_record(entry: Any) -> BoundOfficialTaskFailure:
    """Validate one record and reconstruct its bound failure identity."""
    if not isinstance(entry, Mapping) or set(entry) != _OFFICIAL_FAILURE_RECORD_FIELDS:
        raise ValueError("invalid official failure record fields")
    if (
        type(entry["schema_version"]) is not int
        or entry["schema_version"] != 1
        or entry["artifact_kind"] != OFFICIAL_FAILURE_RECORD_KIND
        or entry["signature_schema"] != EXACT_FAILURE_SIGNATURE_SCHEMA
        or entry["executor"] != OFFICIAL_EXECUTOR_KIND
        or entry["official_framework_commit"] != OFFICIAL_FRAMEWORK_COMMIT
    ):
        raise ValueError("unsupported official failure record schema")
    created = datetime.fromisoformat(entry["created_at"])
    if created.tzinfo is None:
        raise ValueError("official failure record timestamp must include timezone")
    if entry["record_sha256"] != _record_sha256(entry):
        raise ValueError("official failure record hash mismatch")
    task_data = entry["task_failure"]
    if not isinstance(task_data, Mapping) or set(task_data) != _OFFICIAL_FAILURE_TASK_FIELDS:
        raise ValueError("invalid official task failure fields")
    if task_data["returncode"] is not None and type(task_data["returncode"]) is not int:
        raise ValueError("official task failure returncode must be an integer or None")
    task = OfficialTaskFailure(**task_data)
    parsed = parse_official_task_failures(f"=== 失败任务 ===\n{task.raw_line}\n")
    if parsed != [task]:
        raise ValueError("official task failure raw line mismatch")
    if (
        not isinstance(entry["candidate_id"], str)
        or not entry["candidate_id"]
        or entry["candidate_id"] != entry["candidate_id"].strip()
    ):
        raise ValueError("official failure candidate ID must be non-empty")
    allowed_variants = {f"{entry['operator']}_v{rank}" for rank in range(1, 6)}
    if task.candidate_variant not in allowed_variants:
        raise ValueError("official failure candidate variant mismatch")
    bound = BoundOfficialTaskFailure(
        entry["candidate_id"], entry["operator"], entry["candidate_code_hash"], task
    )
    if official_failure_observation_id(entry["env_fingerprint"]) != entry["observation_id"]:
        raise ValueError("official failure observation identity mismatch")
    if make_exact_official_failure_signature(bound, entry["env_fingerprint"]) != entry["key"]:
        raise ValueError("official failure record identity mismatch")
    return bound


class OfficialFailureHistory:
    """Single-writer, append-only JSONL history of official task failures."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            return
        content = self.path.read_text(encoding="utf-8")
        if content and not content.endswith("\n"):
            line_number = content.count("\n") + 1
            raise ValueError(
                f"Invalid official failure history at line {line_number}: "
                "missing terminating newline"
            )
        for line_number, line in enumerate(
            content.splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                entry = json.loads(line, object_pairs_hook=_unique_json_object)
                validate_official_failure_record(entry)
                if entry["key"] in self.entries:
                    raise ValueError("duplicate official failure history key")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid official failure history at line {line_number}: {exc}"
                ) from exc
            self.entries[entry["key"]] = entry

    @staticmethod
    def _semantic_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key not in {"created_at", "record_sha256"}
        }

    def append(
        self,
        failure: BoundOfficialTaskFailure,
        env_fingerprint: str,
        observation_id: str,
    ) -> str:
        return self.append_many(
            [failure], env_fingerprint, observation_id
        )[0]

    def append_many(
        self,
        failures: list[BoundOfficialTaskFailure],
        env_fingerprint: str,
        observation_id: str,
    ) -> list[str]:
        """Preflight and persist one failure batch without logical partial writes."""
        records = [
            make_official_failure_record(failure, env_fingerprint, observation_id)
            for failure in failures
        ]
        pending = {}
        seen = set()
        for record in records:
            key = record["key"]
            if key in seen:
                raise ValueError("Duplicate official task failure in append batch")
            seen.add(key)
            existing = self.entries.get(key)
            if existing is not None:
                if self._semantic_payload(existing) != self._semantic_payload(record):
                    raise ValueError("Conflicting official task failure already exists")
                continue
            pending[key] = record
        if not pending:
            return [record["key"] for record in records]
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("".join(
                json.dumps(
                    record, ensure_ascii=False, allow_nan=False, sort_keys=True
                )
                + "\n"
                for record in pending.values()
            ))
            handle.flush()
            os.fsync(handle.fileno())
        self.entries.update(pending)
        return [record["key"] for record in records]

    def __len__(self) -> int:
        return len(self.entries)


def _record_sha256(entry: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in entry.items() if key != "record_sha256"}
    canonical = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    return sha256_text(canonical)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in official failure history: {key}")
        result[key] = value
    return result
