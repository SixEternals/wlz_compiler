"""Pure adapters for the fixed official Triton agent result schemas."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List, Optional

from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.schemas import Candidate, EvaluationResult


OFFICIAL_FRAMEWORK_COMMIT = "ef8c3bbc7bae6bdfa2af61722f9da14fd8ea5781"
OFFICIAL_EXECUTOR_KIND = "official_triton_executor"
BOUND_EVALUATION_KIND = "bound-official-evaluation"
EXACT_FAILURE_SIGNATURE_SCHEMA = "official-task-failure-exact-v1"
_TASK_FAILURE_PATTERN = re.compile(
    r"^(?P<operator>\S+)\s+(?P<test_case>tc[0-9]+)\s+"
    r"(?P<candidate_variant>\S+):\s+(?P<detail>.+)$",
    re.DOTALL,
)
_RETURN_CODE_PATTERN = re.compile(
    r"^(?P<detail>.+?)\s+\(returncode=(?P<returncode>-?[0-9]+)\)$",
    re.DOTALL,
)
_OFFICIAL_FAILURE_ENV_PATTERN = re.compile(
    r"^coursegrading:contest=[A-Za-z0-9]+:task=[A-Za-z0-9]+:"
    r"problem=[0-9]+:assign=[0-9]+:observation=(?P<observation>[A-Za-z0-9._-]+)$"
)


@dataclass(frozen=True)
class OfficialTaskFailure:
    operator: str
    test_case: str
    candidate_variant: str
    failure_kind: str
    detail: str
    returncode: Optional[int]
    raw_line: str


@dataclass(frozen=True)
class BoundOfficialTaskFailure:
    candidate_id: str
    operator: str
    candidate_code_hash: str
    task_failure: OfficialTaskFailure


def make_exact_official_failure_signature(
    failure: BoundOfficialTaskFailure, env_fingerprint: str
) -> str:
    """Hash one exact candidate/case/environment failure observation."""
    if not isinstance(failure, BoundOfficialTaskFailure):
        raise TypeError("Failure signature requires a bound official task failure")
    official_failure_observation_id(env_fingerprint)
    if not failure.operator or failure.operator != failure.task_failure.operator:
        raise ValueError("Bound official failure operator mismatch")
    if re.fullmatch(r"tc[0-9]+", failure.task_failure.test_case) is None:
        raise ValueError("Official failure test case must use the tcN format")
    _sha256_field(failure.candidate_code_hash, "bound candidate")
    if failure.task_failure.failure_kind not in {
        "runtime_error",
        "accuracy_check_failed",
        "profiling_no_data",
        "unknown",
    }:
        raise ValueError("Unsupported official failure kind")
    payload = {
        "candidate_code_hash": failure.candidate_code_hash,
        "env_fingerprint": env_fingerprint,
        "executor": OFFICIAL_EXECUTOR_KIND,
        "failure_kind": failure.task_failure.failure_kind,
        "official_framework_commit": OFFICIAL_FRAMEWORK_COMMIT,
        "operator": failure.operator,
        "signature_schema": EXACT_FAILURE_SIGNATURE_SCHEMA,
        "test_case": failure.task_failure.test_case,
    }
    canonical = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    return sha256_text(canonical)


def official_failure_observation_id(env_fingerprint: str) -> str:
    """Return the observation bound into a validated CourseGrading fingerprint."""
    match = (
        _OFFICIAL_FAILURE_ENV_PATTERN.fullmatch(env_fingerprint)
        if isinstance(env_fingerprint, str)
        else None
    )
    if match is None:
        raise ValueError(
            "Official failure environment fingerprint must bind "
            "contest/task/problem/assign/observation"
        )
    return match.group("observation")


def parse_official_task_failures(raw_text: str) -> List[OfficialTaskFailure]:
    """Parse the explicit failure section without inferring hidden stages."""
    if not isinstance(raw_text, str):
        raise TypeError("Official result text must be a string")

    found_section = False
    records: List[str] = []
    pending: Optional[str] = None
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if line == "=== 失败任务 ===":
            found_section = True
            continue
        if not found_section:
            continue
        if line.startswith("===") and line.endswith("==="):
            break
        if not line:
            continue

        if _TASK_FAILURE_PATTERN.fullmatch(line) is not None:
            if pending is not None:
                records.append(pending)
            pending = line
            continue
        if pending is not None and "(log tail:" in pending:
            pending = f"{pending}\n{line}"
            continue
        raise ValueError(f"Invalid official task failure line: {line}")

    if pending is not None:
        records.append(pending)
    if not found_section:
        raise ValueError("Official result text has no failure section")

    failures: List[OfficialTaskFailure] = []
    for record in records:
        line = record.strip()
        match = _TASK_FAILURE_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"Invalid official task failure line: {line}")
        detail = match.group("detail")
        returncode = None
        returncode_match = _RETURN_CODE_PATTERN.fullmatch(detail)
        if returncode_match is not None:
            detail = returncode_match.group("detail")
            returncode = int(returncode_match.group("returncode"))

        if detail.startswith("runtime error"):
            failure_kind = "runtime_error"
        elif detail.startswith("accuracy check failed"):
            failure_kind = "accuracy_check_failed"
        elif detail.startswith("msprof no data"):
            failure_kind = "profiling_no_data"
        else:
            failure_kind = "unknown"
        failures.append(
            OfficialTaskFailure(
                operator=match.group("operator"),
                test_case=match.group("test_case"),
                candidate_variant=match.group("candidate_variant"),
                failure_kind=failure_kind,
                detail=detail,
                returncode=returncode,
                raw_line=line,
            )
        )

    return failures


def bind_official_task_failures(
    failures: List[OfficialTaskFailure],
    submission_manifest: Mapping[str, Any],
    expected_artifact_sha256: str,
    *,
    artifact_bytes: bytes,
    base_manifest: Optional[Mapping[str, Any]] = None,
) -> List[BoundOfficialTaskFailure]:
    """Bind platform variant labels to exact submitted candidate identities."""
    if not isinstance(submission_manifest, Mapping):
        raise TypeError("Submission manifest must be a mapping")
    expected_sha = _sha256_field(expected_artifact_sha256, "expected artifact")
    if _sha256_field(
        submission_manifest.get("artifact_sha256"), "submission manifest"
    ) != expected_sha:
        raise ValueError("Submission manifest does not match expected artifact SHA-256")
    base_link = submission_manifest.get("base_artifact")
    if base_link is not None:
        if (
            not isinstance(base_link, Mapping)
            or not isinstance(base_manifest, Mapping)
        ):
            raise ValueError("Overlay submission requires its base manifest")
        base_sha = _sha256_field(base_manifest.get("artifact_sha256"), "base manifest")
        link_sha = _sha256_field(base_link.get("sha256"), "overlay base link")
        if base_sha != link_sha:
            raise ValueError("Overlay base manifest hash mismatch")
    elif base_manifest is not None:
        raise ValueError("Base manifest supplied for a non-overlay submission")

    base_candidates = {}
    if base_manifest is not None:
        base_candidates = _submission_candidates(base_manifest, "selections")
    selections = _submission_candidates(submission_manifest, "selections")
    replacements = _submission_candidates(submission_manifest, "replacements")
    if base_manifest is not None:
        if selections or not set(replacements).issubset(base_candidates):
            raise ValueError("Overlay replacements must be a subset of base selections")
        candidates = {**base_candidates, **replacements}
        if any(variant != f"{operator}_v1" for operator, variant in candidates):
            raise ValueError("Multi-version overlay submissions are not supported")
    else:
        if replacements:
            raise ValueError("Non-overlay submission cannot contain replacements")
        candidates = selections
    if not candidates:
        raise ValueError("Submission manifest has no candidate selections")

    archive_entries = submission_manifest.get("archive_entries")
    if not isinstance(archive_entries, list) or not all(
        isinstance(entry, str) for entry in archive_entries
    ):
        raise ValueError("Submission manifest has invalid archive_entries")
    if len(archive_entries) != len(set(archive_entries)):
        raise ValueError("Submission manifest has duplicate archive_entries")
    archived = set(archive_entries)
    _verify_submission_archive(
        artifact_bytes, expected_sha, archived, candidates
    )

    bound = []
    for failure in failures:
        if not isinstance(failure, OfficialTaskFailure):
            raise TypeError("Failures must contain OfficialTaskFailure records")
        candidate = candidates.get((failure.operator, failure.candidate_variant))
        if candidate is None:
            if any(operator == failure.operator for operator, _ in candidates):
                raise ValueError(
                    f"Official candidate variant mismatch for {failure.operator}"
                )
            raise ValueError(f"No submitted candidate for operator {failure.operator}")
        expected_entry = f"output/{failure.operator}/{failure.candidate_variant}.py"
        if expected_entry not in archived:
            raise ValueError(
                f"Submitted candidate archive entry missing: {expected_entry}"
            )
        bound.append(
            BoundOfficialTaskFailure(
                candidate_id=candidate["candidate_id"],
                operator=failure.operator,
                candidate_code_hash=candidate["candidate_sha256"],
                task_failure=failure,
            )
        )
    return bound


def _verify_submission_archive(
    artifact_bytes: bytes,
    expected_sha: str,
    declared_entries: set[str],
    candidates: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    if not isinstance(artifact_bytes, bytes):
        raise TypeError("Submission artifact must be bytes")
    if hashlib.sha256(artifact_bytes).hexdigest() != expected_sha:
        raise ValueError("Submission artifact SHA-256 mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(artifact_bytes)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != declared_entries:
                raise ValueError("Submission ZIP entries do not match its manifest")
            counts = {}
            for operator, _ in candidates:
                counts[operator] = counts.get(operator, 0) + 1
            for operator in counts:
                best = f"output/{operator}/{operator}_best.py"
                v1 = f"output/{operator}/{operator}_v1.py"
                if best in declared_entries and archive.read(best) != archive.read(v1):
                    raise ValueError(f"Submitted candidate best source mismatch: {operator}")
            for (operator, variant), candidate in candidates.items():
                source = f"output/{operator}/{variant}.py"
                stats = f"output/{operator}/{operator}_stats.json"
                if hashlib.sha256(archive.read(source)).hexdigest() != candidate["candidate_sha256"]:
                    raise ValueError(f"Submitted candidate source hash mismatch: {variant}")
                summary = json.loads(
                    archive.read(stats), object_pairs_hook=_unique_json_object
                )
                top5 = summary.get("top5_summary") if isinstance(summary, Mapping) else None
                index = int(variant.rsplit("_v", 1)[1]) - 1
                if (
                    not isinstance(top5, list)
                    or len(top5) != counts[operator]
                    or not isinstance(top5[index], Mapping)
                    or top5[index].get("id") != candidate["candidate_id"]
                ):
                    raise ValueError(f"Submitted candidate stats identity mismatch: {variant}")
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid submission ZIP: {exc}") from exc


def _submission_candidates(
    manifest: Mapping[str, Any], field: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    records = manifest.get(field, [])
    if not isinstance(records, list):
        raise ValueError(f"Submission manifest field '{field}' must be a list")
    candidates = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"Submission manifest field '{field}' has invalid record")
        operator = record.get("operator")
        candidate_id = record.get("candidate_id")
        candidate_hash = record.get("candidate_sha256")
        if not isinstance(operator, str) or not operator:
            raise ValueError("Submitted candidate operator must be a non-empty string")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Submitted candidate ID must be a non-empty string")
        _sha256_field(candidate_hash, "submitted candidate")
        variant = record.get("candidate_variant", f"{operator}_v1")
        match = (
            re.fullmatch(re.escape(operator) + r"_v([1-5])", variant)
            if isinstance(variant, str)
            else None
        )
        if match is None:
            raise ValueError("Submitted candidate variant must match its operator and rank")
        key = (operator, variant)
        if key in candidates:
            raise ValueError(f"Duplicate submitted candidate variant: {variant}")
        candidates[key] = record
    by_operator = {}
    for operator, variant in candidates:
        by_operator.setdefault(operator, []).append(int(variant.rsplit("_v", 1)[1]))
    if any(sorted(indices) != list(range(1, len(indices) + 1)) for indices in by_operator.values()):
        raise ValueError("Submitted candidate variants must be contiguous from v1")
    return candidates


def _sha256_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} SHA-256 must be lowercase hexadecimal")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in candidate stats: {key}")
        result[key] = value
    return result


def adapt_bound_official_evaluation(
    candidate: Candidate,
    envelope: Mapping[str, Any],
    *,
    baseline_time_us: Optional[float] = None,
) -> EvaluationResult:
    """Import one offline official result only after exact candidate binding."""

    if not isinstance(envelope, Mapping):
        raise TypeError("Official evaluation envelope must be a mapping")
    expected = {
        "schema_version": 1,
        "artifact_kind": BOUND_EVALUATION_KIND,
        "operator": candidate.op_name,
        "candidate_id": candidate.id,
        "candidate_code_hash": candidate.code_hash,
    }
    if sha256_text(candidate.code) != candidate.code_hash:
        raise ValueError("Candidate code does not match candidate_code_hash")
    for field, value in expected.items():
        if envelope.get(field) != value:
            raise ValueError(f"Official evaluation envelope mismatch: {field}")
    if "evaluation" not in envelope:
        raise ValueError("Missing official evaluation envelope field: evaluation")

    result = adapt_official_evaluation(
        candidate.id,
        envelope["evaluation"],
        baseline_time_us=baseline_time_us,
    )
    result.metadata = {
        **result.metadata,
        "binding_verified": True,
        "bound_operator": candidate.op_name,
        "bound_candidate_code_hash": candidate.code_hash,
    }
    return result


def adapt_official_evaluation(
    candidate_id: str,
    raw_result: Any,
    *,
    baseline_time_us: Optional[float] = None,
) -> EvaluationResult:
    """Map the official executor dataclass or an equivalent mapping.

    The official schema has no separate compile or correctness fields. Those
    values therefore remain unknown even when its aggregate ``success`` flag is
    true. Failure-side zero metrics are treated as sentinels, not measurements.
    """

    success = _bool_field(raw_result, "success")
    execution_time_us = _number_field(raw_result, "execution_time")
    official_speedup = _number_field(raw_result, "speedup")
    official_fitness = _number_field(raw_result, "fitness")
    error = _field(raw_result, "error")
    if error is not None and not isinstance(error, str):
        raise TypeError("Official field 'error' must be a string or None")

    baseline_ms = None
    if baseline_time_us is not None:
        baseline_us = _finite_number(baseline_time_us, "baseline_time_us")
        if baseline_us <= 0:
            raise ValueError("baseline_time_us must be positive")
        baseline_ms = baseline_us / 1000.0

    if success and execution_time_us <= 0:
        raise ValueError("A successful official result must have positive execution_time")

    metadata = {
        "source_schema": "official_triton_agent.EvaluationResult",
        "official_framework_commit": OFFICIAL_FRAMEWORK_COMMIT,
        "official_execution_time_us": execution_time_us,
        "official_speedup": official_speedup,
        "official_fitness": official_fitness,
        "stage_detail_available": False,
    }

    return EvaluationResult(
        candidate_id=candidate_id,
        executor=OFFICIAL_EXECUTOR_KIND,
        status="official_evaluation_success" if success else "official_evaluation_failed",
        passed=success,
        correctness_ok=None,
        compile_ok=None,
        latency_ms=execution_time_us / 1000.0 if success else None,
        baseline_ms=baseline_ms,
        speedup=official_speedup if success else None,
        proxy_score=None,
        error_type=None if success else "official_evaluation_failed",
        error_message=None if success else (error or "Performance test failed"),
        metadata=metadata,
    )


def adapt_official_optimization_result(
    op_name: str,
    raw_result: Mapping[str, Any],
) -> List[Candidate]:
    """Convert the official ``optimize()`` Top-5 into local candidates."""

    if not isinstance(raw_result, Mapping):
        raise TypeError("Official optimization result must be a mapping")
    top5 = raw_result.get("top5_codes")
    if not isinstance(top5, list):
        raise TypeError("Official field 'top5_codes' must be a list")
    if len(top5) > 5:
        raise ValueError("Official optimization result contains more than five candidates")

    summary = {
        key: raw_result.get(key)
        for key in (
            "best_fitness",
            "speedup",
            "generations",
            "time_elapsed",
            "llm_stats",
        )
    }
    candidates: List[Candidate] = []
    for rank, item in enumerate(top5, start=1):
        if not isinstance(item, Mapping):
            raise TypeError("Each official Top-5 entry must be a mapping")
        code = _typed_item(item, "code", str)
        candidate_id = _typed_item(item, "id", str)
        generation = _typed_item(item, "generation", int)
        fitness = _finite_number(item.get("fitness"), "fitness")

        candidates.append(
            Candidate(
                id=candidate_id,
                op_name=op_name,
                code=code,
                code_hash=sha256_text(code),
                parent_ids=[],
                generation=generation,
                mutation_kind="official_unknown",
                model_used=None,
                prompt_id=None,
                status="official_top5",
                score=fitness,
                metadata={
                    "source_schema": "official_triton_agent.optimize.top5_codes",
                    "official_framework_commit": OFFICIAL_FRAMEWORK_COMMIT,
                    "official_rank": rank,
                    "official_run_summary": summary,
                    "provenance_complete": False,
                    "missing_provenance_fields": [
                        "parent_ids",
                        "mutation_kind",
                        "model_used",
                        "prompt_id",
                    ],
                },
            )
        )
    return candidates


def _field(raw_result: Any, name: str) -> Any:
    if isinstance(raw_result, Mapping):
        if name not in raw_result:
            raise ValueError(f"Missing official field: {name}")
        return raw_result[name]
    if not hasattr(raw_result, name):
        raise ValueError(f"Missing official field: {name}")
    return getattr(raw_result, name)


def _bool_field(raw_result: Any, name: str) -> bool:
    value = _field(raw_result, name)
    if not isinstance(value, bool):
        raise TypeError(f"Official field '{name}' must be bool")
    return value


def _number_field(raw_result: Any, name: str) -> float:
    return _finite_number(_field(raw_result, name), name)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Official field '{name}' must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Official field '{name}' must be finite")
    return number


def _typed_item(item: Mapping[str, Any], name: str, expected: type) -> Any:
    value = item.get(name)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise TypeError(f"Official Top-5 field '{name}' must be {expected.__name__}")
    return value
