"""Fail-closed correctness admission before expensive performance evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

from .schemas import CorrectnessCaseResult


@dataclass(frozen=True, order=True)
class ForeignCorrectnessResult:
    """A result that cannot be attributed to the candidate being decided."""

    candidate_id: str
    case_signature: str


@dataclass(frozen=True)
class CandidateCorrectnessDecision:
    """Complete, deterministic evidence for performance-test admission."""

    candidate_id: str
    eligible_for_performance: bool
    blocking_reasons: Tuple[str, ...]
    expected_signatures: Tuple[str, ...]
    passed_signatures: Tuple[str, ...]
    missing_signatures: Tuple[str, ...]
    failed_signatures: Tuple[str, ...]
    oracle_error_signatures: Tuple[str, ...]
    unknown_signatures: Tuple[str, ...]
    duplicate_signatures: Tuple[str, ...]
    unexpected_signatures: Tuple[str, ...]
    foreign_results: Tuple[ForeignCorrectnessResult, ...]
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def decide_candidate_correctness(
    candidate_id: str,
    expected_case_signatures: Sequence[str],
    results: Iterable[CorrectnessCaseResult],
) -> CandidateCorrectnessDecision:
    """Admit only a candidate with exactly one passing result per expected case."""

    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must not be empty")

    expected_list = list(expected_case_signatures)
    for signature in expected_list:
        _validate_signature(signature)
    if len(expected_list) != len(set(expected_list)):
        raise ValueError("expected_case_signatures must not contain duplicates")

    expected = tuple(sorted(expected_list))
    expected_set = set(expected)
    by_signature: Dict[str, list[CorrectnessCaseResult]] = {}
    foreign = []

    for result in results:
        if not isinstance(result, CorrectnessCaseResult):
            raise TypeError("results must contain only CorrectnessCaseResult values")
        if result.candidate_id != candidate_id:
            foreign.append(
                ForeignCorrectnessResult(result.candidate_id, result.case_signature)
            )
            continue
        by_signature.setdefault(result.case_signature, []).append(result)

    missing = tuple(signature for signature in expected if signature not in by_signature)
    duplicate = tuple(
        sorted(signature for signature, group in by_signature.items() if len(group) > 1)
    )
    unexpected = tuple(sorted(set(by_signature) - expected_set))

    statuses = {
        status: tuple(
            sorted(
                signature
                for signature, group in by_signature.items()
                if signature in expected_set
                and any(result.oracle_status == status for result in group)
            )
        )
        for status in ("passed", "failed", "oracle_error", "unknown")
    }

    reason_conditions = (
        ("no_expected_cases", not expected),
        ("foreign_candidate_result", bool(foreign)),
        ("missing_case_result", bool(missing)),
        ("failed_case", bool(statuses["failed"])),
        ("oracle_error", bool(statuses["oracle_error"])),
        ("unknown_case", bool(statuses["unknown"])),
        ("duplicate_case_result", bool(duplicate)),
        ("unexpected_case_result", bool(unexpected)),
    )
    blocking_reasons = tuple(reason for reason, blocked in reason_conditions if blocked)

    return CandidateCorrectnessDecision(
        candidate_id=candidate_id,
        eligible_for_performance=not blocking_reasons,
        blocking_reasons=blocking_reasons,
        expected_signatures=expected,
        passed_signatures=statuses["passed"],
        missing_signatures=missing,
        failed_signatures=statuses["failed"],
        oracle_error_signatures=statuses["oracle_error"],
        unknown_signatures=statuses["unknown"],
        duplicate_signatures=duplicate,
        unexpected_signatures=unexpected,
        foreign_results=tuple(sorted(foreign)),
    )


def _validate_signature(signature: str) -> None:
    if (
        not isinstance(signature, str)
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise ValueError("expected case signatures must be lowercase SHA-256 hex strings")
