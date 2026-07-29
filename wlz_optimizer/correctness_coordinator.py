"""Fail-closed coordination of isolated runners, oracle comparison, and gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from .candidate_runner import CandidateRunResult
from .correctness import CandidateCorrectnessDecision, decide_candidate_correctness
from .correctness_oracle import InvocationSnapshot, compare_oracle
from .correctness_references import ReferenceResult
from .input_materializer import (
    MaterializedInputs,
    TensorBackend,
    clone_inputs_for_run,
    materialize_inputs,
)
from .schemas import Candidate, CorrectnessCaseResult, CorrectnessErrorSummary, EvaluationCase


ReferenceCaseRunner = Callable[[EvaluationCase, MaterializedInputs], ReferenceResult]
CandidateCaseRunner = Callable[
    [Candidate, EvaluationCase, MaterializedInputs], CandidateRunResult
]
SnapshotDecoder = Callable[[object], InvocationSnapshot]


@dataclass(frozen=True)
class CandidateCorrectnessRun:
    """Per-case evidence plus the existing fail-closed gate decision."""

    results: Tuple[CorrectnessCaseResult, ...]
    decision: CandidateCorrectnessDecision


def evaluate_correctness_case(
    candidate: Candidate,
    case: EvaluationCase,
    backend: TensorBackend,
    reference_runner: ReferenceCaseRunner,
    candidate_runner: CandidateCaseRunner,
    snapshot_decoder: SnapshotDecoder,
) -> CorrectnessCaseResult:
    """Run one case with fresh reference/candidate storage and classify all failures."""

    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    if not isinstance(case, EvaluationCase):
        raise TypeError("case must be an EvaluationCase")

    try:
        pristine = materialize_inputs(case, backend)
        reference_inputs = clone_inputs_for_run(pristine, backend)
        candidate_inputs = clone_inputs_for_run(pristine, backend)
    except Exception as exc:
        return _oracle_error(candidate, case, "input_materialization", exc)

    try:
        reference_result = reference_runner(case, reference_inputs)
    except Exception as exc:
        return _oracle_error(candidate, case, "reference_runner", exc)
    if not isinstance(reference_result, ReferenceResult):
        return _oracle_error(
            candidate, case, "reference_runner", TypeError("runner returned wrong result type")
        )
    if reference_result.status != "completed":
        return _oracle_error(
            candidate,
            case,
            f"reference_{reference_result.status}",
            RuntimeError(reference_result.error_message or reference_result.status),
        )
    try:
        reference_snapshot = snapshot_decoder(reference_result.value)
        if not isinstance(reference_snapshot, InvocationSnapshot):
            raise TypeError("decoder did not return InvocationSnapshot")
    except Exception as exc:
        return _oracle_error(candidate, case, "reference_snapshot", exc)

    try:
        candidate_result = candidate_runner(candidate, case, candidate_inputs)
    except Exception as exc:
        return _candidate_failure(candidate, case, "candidate_runner", exc)
    if not isinstance(candidate_result, CandidateRunResult):
        return _candidate_failure(
            candidate, case, "candidate_runner", TypeError("runner returned wrong result type")
        )
    if candidate_result.candidate_id != candidate.id:
        return _candidate_failure(
            candidate, case, "candidate_identity", ValueError("candidate_id mismatch")
        )
    if candidate_result.status == "not_configured":
        return _unknown(
            candidate,
            case,
            f"candidate_not_configured: {candidate_result.error_message or candidate_result.phase}",
        )
    if candidate_result.status != "completed":
        return _candidate_failure(
            candidate,
            case,
            f"candidate_{candidate_result.status}",
            RuntimeError(candidate_result.error_message or candidate_result.phase),
        )
    try:
        candidate_snapshot = snapshot_decoder(candidate_result.value)
        if not isinstance(candidate_snapshot, InvocationSnapshot):
            raise TypeError("decoder did not return InvocationSnapshot")
    except Exception as exc:
        return _candidate_failure(candidate, case, "candidate_snapshot", exc)

    try:
        comparison = compare_oracle(
            case.oracle_policy,
            case.oracle_targets,
            candidate_snapshot,
            reference_snapshot,
        )
    except Exception as exc:
        return _oracle_error(candidate, case, "oracle_comparator", exc)
    return CorrectnessCaseResult(
        candidate_id=candidate.id,
        case_id=case.case_id,
        case_signature=case.signature(),
        oracle_policy_id=case.oracle_policy.policy_id,
        oracle_status=comparison.status,
        error_summary=comparison.error_summary,
        message=comparison.message,
    )


def evaluate_candidate_correctness(
    candidate: Candidate,
    cases: Sequence[EvaluationCase],
    backend: TensorBackend,
    reference_runner: ReferenceCaseRunner,
    candidate_runner: CandidateCaseRunner,
    snapshot_decoder: SnapshotDecoder,
) -> CandidateCorrectnessRun:
    """Evaluate every declared case and immediately apply the existing gate."""

    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    if not isinstance(cases, Sequence) or any(
        not isinstance(case, EvaluationCase) for case in cases
    ):
        raise TypeError("cases must be a sequence of EvaluationCase values")
    results = tuple(
        evaluate_correctness_case(
            candidate,
            case,
            backend,
            reference_runner,
            candidate_runner,
            snapshot_decoder,
        )
        for case in cases
    )
    decision = decide_candidate_correctness(
        candidate.id,
        [case.signature() for case in cases],
        results,
    )
    return CandidateCorrectnessRun(results, decision)


def _oracle_error(
    candidate: Candidate, case: EvaluationCase, stage: str, error: Exception
) -> CorrectnessCaseResult:
    return CorrectnessCaseResult(
        candidate.id,
        case.case_id,
        case.signature(),
        case.oracle_policy.policy_id,
        "oracle_error",
        message=_bounded(f"{stage}: {type(error).__name__}: {error}"),
    )


def _candidate_failure(
    candidate: Candidate, case: EvaluationCase, kind: str, error: Exception
) -> CorrectnessCaseResult:
    message = _bounded(f"{kind}: {type(error).__name__}: {error}")
    return CorrectnessCaseResult(
        candidate.id,
        case.case_id,
        case.signature(),
        case.oracle_policy.policy_id,
        "failed",
        error_summary=CorrectnessErrorSummary(
            mismatch_kind=kind,
            mismatch_count=1,
            compared_count=1,
            first_mismatch=message,
        ),
        message=message,
    )


def _unknown(
    candidate: Candidate, case: EvaluationCase, message: str
) -> CorrectnessCaseResult:
    return CorrectnessCaseResult(
        candidate.id,
        case.case_id,
        case.signature(),
        case.oracle_policy.policy_id,
        "unknown",
        message=_bounded(message),
    )


def _bounded(message: str, limit: int = 512) -> str:
    return message.encode("utf-8", errors="replace")[:limit].decode(
        "utf-8", errors="ignore"
    )
