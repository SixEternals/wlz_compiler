"""Read-only, bounded evidence projection for prompt construction."""

from __future__ import annotations

import ast
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Iterable, Optional

from wlz_optimizer.cache import validate_official_failure_record
from wlz_optimizer.official_adapter import OFFICIAL_EXECUTOR_KIND
from wlz_optimizer.schemas import Candidate, EvaluationResult, ShapeObservation


_FAILURE_CATEGORIES = {
    "syntax_fail",
    "import_fail",
    "signature_fail",
    "launch_contract_fail",
    "triton_semantic_fail",
    "runtime_error",
    "accuracy_check_failed",
    "timeout",
}
MAX_OBSERVED_SPEEDUPS = 8
MAX_FAILURE_CATEGORIES = 8
MAX_SHAPE_OBSERVATIONS = 32
MAX_SOURCE_ACCESS_COUNT = 99
MAX_TENSOR_RANK_BUCKET = 16
SANITIZATION_VERSION = "prompt-context-sanitization-v2"
_DTYPE_FAMILIES = ("bool", "float", "integer", "other", "unknown")
_SOURCE_ACCESS_KINDS = (
    "loads",
    "stores",
    "atomics",
    "block_pointers",
    "transposes",
)


def _coarse_failure_category(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return "other"
    return value if value in _FAILURE_CATEGORIES else "other"


def _finite_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _status(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError("evaluation status fields must be booleans or None")
    return value


def _dtype_family(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    lowered = value.lower()
    if "bool" in lowered:
        return "bool"
    if "float" in lowered or "bfloat" in lowered:
        return "float"
    if "int" in lowered or "index" in lowered:
        return "integer"
    return "other"


def _source_access_counts(source: str) -> tuple[tuple[str, int], ...]:
    counts = Counter({kind: 0 for kind in _SOURCE_ACCESS_KINDS})
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
        ):
            continue
        name = node.func.attr
        if name == "load":
            counts["loads"] += 1
        elif name == "store":
            counts["stores"] += 1
        elif isinstance(name, str) and name.startswith("atomic_"):
            counts["atomics"] += 1
        elif name in {"make_block_ptr", "advance"}:
            counts["block_pointers"] += 1
        elif name in {"trans", "permute"}:
            counts["transposes"] += 1
    return tuple(
        (kind, min(counts[kind], MAX_SOURCE_ACCESS_COUNT))
        for kind in _SOURCE_ACCESS_KINDS
    )


@dataclass(frozen=True)
class EvidenceView:
    """A parent-bound, non-persistent view over existing evaluation sources."""

    parent: Candidate
    env_fingerprint: Optional[str]
    evaluations: tuple[EvaluationResult, ...]
    official_failure_records: tuple[Mapping[str, Any], ...] = ()
    shape_observations: tuple[ShapeObservation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parent, Candidate):
            raise TypeError("parent must be a Candidate")
        if self.env_fingerprint is not None and (
            not isinstance(self.env_fingerprint, str) or not self.env_fingerprint.strip()
        ):
            raise ValueError("env_fingerprint must be a non-empty string or None")
        if not isinstance(self.evaluations, tuple) or not all(
            isinstance(result, EvaluationResult) for result in self.evaluations
        ):
            raise TypeError("evaluations must be a tuple of EvaluationResult values")
        if not isinstance(self.official_failure_records, tuple) or not all(
            isinstance(record, Mapping) for record in self.official_failure_records
        ):
            raise TypeError("official_failure_records must be a tuple of mappings")
        if not isinstance(self.shape_observations, tuple) or not all(
            isinstance(item, ShapeObservation) for item in self.shape_observations
        ):
            raise TypeError("shape_observations must be a tuple of ShapeObservation values")
        if len(self.shape_observations) > MAX_SHAPE_OBSERVATIONS:
            raise ValueError(
                f"shape_observations must contain at most {MAX_SHAPE_OBSERVATIONS} values"
            )


@dataclass(frozen=True)
class PromptContext:
    """Bounded, model-facing context with no raw diagnostics or identifiers."""

    parent_code_hash: str
    generation: int
    environment_bound: bool
    evaluation_count: int
    evaluation_pass_count: int
    compile_counts: tuple[int, int, int]
    correctness_counts: tuple[int, int, int]
    failure_category_counts: tuple[tuple[str, int], ...]
    observed_speedups: tuple[float, ...]
    shape_observation_count: int = 0
    tensor_rank_counts: tuple[tuple[int, int], ...] = ()
    dtype_family_counts: tuple[tuple[str, int], ...] = ()
    unknown_dimension_count: int = 0
    source_access_counts: tuple[tuple[str, int], ...] = ()
    official_performance_count: int = 0
    official_speedup_best: Optional[float] = None
    official_speedup_median: Optional[float] = None
    official_speedup_latest: Optional[float] = None
    official_latency_ms_best: Optional[float] = None
    sanitization_version: str = SANITIZATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptContextProjector:
    """Project EvidenceView into deterministic, bounded prompt data."""

    def project(self, evidence: EvidenceView) -> PromptContext:
        if not isinstance(evidence, EvidenceView):
            raise TypeError("evidence must be an EvidenceView")
        parent = evidence.parent
        if not isinstance(parent.code_hash, str) or not parent.code_hash.strip():
            raise ValueError("parent code_hash must be non-empty")
        if type(parent.generation) is not int or parent.generation < 0:
            raise ValueError("parent generation must be a non-negative integer")
        if any(result.candidate_id != parent.id for result in evidence.evaluations):
            raise ValueError("evaluation candidate_id does not match parent")
        if any(type(result.passed) is not bool for result in evidence.evaluations):
            raise ValueError("evaluation passed must be a boolean")
        if any(item.op_name != parent.op_name for item in evidence.shape_observations):
            raise ValueError("shape observation op_name does not match parent")
        compile_counts = self._counts(_status(r.compile_ok) for r in evidence.evaluations)
        correctness_counts = self._counts(
            _status(r.correctness_ok) for r in evidence.evaluations
        )
        failures = Counter()
        for record in evidence.official_failure_records:
            if (
                record.get("operator") != parent.op_name
                or record.get("candidate_id") != parent.id
                or record.get("candidate_code_hash") != parent.code_hash
                or record.get("env_fingerprint") != evidence.env_fingerprint
            ):
                continue
            failure = validate_official_failure_record(record)
            failures[_coarse_failure_category(failure.task_failure.failure_kind)] += 1
        failures.update(
            category
            for result in evidence.evaluations
            if not result.passed
            for category in (_coarse_failure_category(result.error_type),)
            if category is not None
        )
        categories = sorted(failures.items(), key=lambda item: (-item[1], item[0]))
        speeds = tuple(
            speed
            for result in evidence.evaluations
            if result.executor == OFFICIAL_EXECUTOR_KIND and result.passed
            for speed in (_finite_score(result.speedup),)
            if speed is not None
        )[-MAX_OBSERVED_SPEEDUPS:]
        official_passes = tuple(
            result
            for result in evidence.evaluations
            if result.executor == OFFICIAL_EXECUTOR_KIND and result.passed
        )
        all_speedups = tuple(
            value
            for result in official_passes
            for value in (_finite_score(result.speedup),)
            if value is not None
        )
        latencies = tuple(
            value
            for result in official_passes
            for value in (_finite_score(result.latency_ms),)
            if value is not None and value >= 0
        )
        ranks = Counter(
            min(len(shape), MAX_TENSOR_RANK_BUCKET)
            for observation in evidence.shape_observations
            for shape in observation.tensor_shapes.values()
        )
        dtype_families = Counter(
            _dtype_family(observation.tensor_dtypes.get(tensor_name))
            for observation in evidence.shape_observations
            for tensor_name in observation.tensor_shapes
        )
        return PromptContext(
            parent_code_hash=parent.code_hash,
            generation=parent.generation,
            environment_bound=evidence.env_fingerprint is not None,
            evaluation_count=len(evidence.evaluations),
            evaluation_pass_count=sum(r.passed for r in evidence.evaluations),
            compile_counts=compile_counts,
            correctness_counts=correctness_counts,
            failure_category_counts=tuple(categories[:MAX_FAILURE_CATEGORIES]),
            observed_speedups=speeds,
            shape_observation_count=len(evidence.shape_observations),
            tensor_rank_counts=tuple(sorted(ranks.items())),
            dtype_family_counts=tuple(
                (family, dtype_families[family])
                for family in _DTYPE_FAMILIES
                if dtype_families[family]
            ),
            unknown_dimension_count=sum(
                dimension is None
                for observation in evidence.shape_observations
                for shape in observation.tensor_shapes.values()
                for dimension in shape
            ),
            source_access_counts=_source_access_counts(parent.code),
            official_performance_count=len(official_passes),
            official_speedup_best=max(all_speedups) if all_speedups else None,
            official_speedup_median=median(all_speedups) if all_speedups else None,
            official_speedup_latest=all_speedups[-1] if all_speedups else None,
            official_latency_ms_best=min(latencies) if latencies else None,
            sanitization_version=SANITIZATION_VERSION,
        )

    @staticmethod
    def _counts(values: Iterable[Optional[bool]]) -> tuple[int, int, int]:
        values = tuple(values)
        return (
            sum(value is True for value in values),
            sum(value is False for value in values),
            sum(value is None for value in values),
        )
