"""Correctness-gated CUDA timing for local proxy measurements only."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from .correctness import CandidateCorrectnessDecision


@dataclass(frozen=True)
class CudaBenchmarkConfig:
    warmup_runs: int = 5
    measurement_runs: int = 20
    percentile: float = 95.0

    def __post_init__(self) -> None:
        if isinstance(self.warmup_runs, bool) or not isinstance(self.warmup_runs, int):
            raise ValueError("warmup_runs must be an integer")
        if isinstance(self.measurement_runs, bool) or not isinstance(self.measurement_runs, int):
            raise ValueError("measurement_runs must be an integer")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.measurement_runs < 1:
            raise ValueError("measurement_runs must be positive")
        if not isinstance(self.percentile, (int, float)) or isinstance(self.percentile, bool):
            raise ValueError("percentile must be numeric")
        if not math.isfinite(float(self.percentile)) or not 0.0 <= self.percentile <= 100.0:
            raise ValueError("percentile must be between 0 and 100")


@dataclass(frozen=True)
class LocalCudaBenchmarkResult:
    candidate_id: str
    case_signature: str
    executor: str
    status: str
    local_latency_ms: Optional[float] = None
    percentile_latency_ms: Optional[float] = None
    samples_ms: Tuple[float, ...] = ()
    environment_fingerprint: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class LocalCudaComparisonResult:
    candidate_id: str
    case_signature: str
    executor: str
    status: str
    baseline_local_latency_ms: Optional[float] = None
    candidate_local_latency_ms: Optional[float] = None
    baseline_percentile_latency_ms: Optional[float] = None
    candidate_percentile_latency_ms: Optional[float] = None
    candidate_over_baseline_latency_ratio: Optional[float] = None
    baseline_samples_ms: Tuple[float, ...] = ()
    candidate_samples_ms: Tuple[float, ...] = ()
    environment_fingerprint: Optional[str] = None
    message: Optional[str] = None


def benchmark_cuda_callable(
    candidate_id: str,
    case_signature: str,
    decision: CandidateCorrectnessDecision,
    run_once: Callable[[], Any],
    *,
    torch_module: Any = None,
    config: Optional[CudaBenchmarkConfig] = None,
    stream: Any = None,
) -> LocalCudaBenchmarkResult:
    """Measure one already-correct candidate with CUDA events.

    The result is a local proxy observation. It is not an Ascend latency or score.
    The caller owns input reset between repetitions when the case is stateful. If
    ``stream`` is supplied, ``run_once`` is launched in that stream's context.
    """

    if not isinstance(decision, CandidateCorrectnessDecision):
        raise TypeError("decision must be a CandidateCorrectnessDecision")
    if decision.candidate_id != candidate_id:
        raise ValueError("benchmark candidate_id does not match correctness decision")
    if (
        not isinstance(case_signature, str)
        or len(case_signature) != 64
        or any(char not in "0123456789abcdef" for char in case_signature)
    ):
        raise ValueError("case_signature must be a lowercase SHA-256 hex string")
    if case_signature not in decision.expected_signatures:
        raise ValueError("case_signature is not an expected correctness case")
    if not callable(run_once):
        raise TypeError("run_once must be callable")
    if config is not None and not isinstance(config, CudaBenchmarkConfig):
        raise TypeError("config must be a CudaBenchmarkConfig")
    benchmark_config = config or CudaBenchmarkConfig()
    if not decision.eligible_for_performance:
        return LocalCudaBenchmarkResult(
            candidate_id,
            case_signature,
            "local_cuda_proxy",
            "blocked",
            message="correctness gate did not admit this candidate",
        )
    if case_signature not in decision.passed_signatures:
        raise ValueError("case_signature is not a passed correctness case")

    try:
        cuda, timing_stream, stream_context, fingerprint = _cuda_setup(
            torch_module, stream
        )
    except Exception as exc:
        return LocalCudaBenchmarkResult(
            candidate_id,
            case_signature,
            "local_cuda_proxy",
            "not_configured",
            message=f"CUDA backend unavailable: {type(exc).__name__}: {exc}",
        )

    samples = []
    try:
        with stream_context:
            for _ in range(benchmark_config.warmup_runs):
                run_once()
            cuda.synchronize()
            for _ in range(benchmark_config.measurement_runs):
                samples.append(_timed_call(cuda, timing_stream, run_once))
    except Exception as exc:
        return LocalCudaBenchmarkResult(
            candidate_id,
            case_signature,
            "local_cuda_proxy",
            "failed",
            samples_ms=tuple(samples),
            environment_fingerprint=fingerprint,
            message=f"CUDA benchmark failed: {type(exc).__name__}: {exc}",
        )

    median = float(statistics.median(samples))
    percentile = _percentile(samples, float(benchmark_config.percentile))
    return LocalCudaBenchmarkResult(
        candidate_id,
        case_signature,
        "local_cuda_proxy",
        "completed",
        local_latency_ms=median,
        percentile_latency_ms=percentile,
        samples_ms=tuple(samples),
        environment_fingerprint=fingerprint,
    )


def compare_cuda_callables(
    candidate_id: str,
    case_signature: str,
    decision: CandidateCorrectnessDecision,
    baseline_once: Callable[[], Any],
    candidate_once: Callable[[], Any],
    *,
    torch_module: Any = None,
    config: Optional[CudaBenchmarkConfig] = None,
    stream: Any = None,
) -> LocalCudaComparisonResult:
    """Interleave local CUDA baseline/candidate timing to reduce drift bias."""

    if not isinstance(decision, CandidateCorrectnessDecision):
        raise TypeError("decision must be a CandidateCorrectnessDecision")
    if decision.candidate_id != candidate_id:
        raise ValueError("comparison candidate_id does not match correctness decision")
    if case_signature not in decision.expected_signatures:
        raise ValueError("case_signature is not an expected correctness case")
    if not callable(baseline_once) or not callable(candidate_once):
        raise TypeError("baseline_once and candidate_once must be callable")
    if config is not None and not isinstance(config, CudaBenchmarkConfig):
        raise TypeError("config must be a CudaBenchmarkConfig")
    benchmark_config = config or CudaBenchmarkConfig()
    if not decision.eligible_for_performance:
        return LocalCudaComparisonResult(
            candidate_id,
            case_signature,
            "local_cuda_proxy_interleaved",
            "blocked",
            message="correctness gate did not admit this candidate",
        )
    if case_signature not in decision.passed_signatures:
        raise ValueError("case_signature is not a passed correctness case")

    try:
        cuda, timing_stream, stream_context, fingerprint = _cuda_setup(
            torch_module, stream
        )
    except Exception as exc:
        return LocalCudaComparisonResult(
            candidate_id,
            case_signature,
            "local_cuda_proxy_interleaved",
            "not_configured",
            message=f"CUDA backend unavailable: {type(exc).__name__}: {exc}",
        )

    baseline_samples = []
    candidate_samples = []
    try:
        with stream_context:
            for index in range(benchmark_config.warmup_runs):
                first, second = (
                    (baseline_once, candidate_once)
                    if index % 2 == 0
                    else (candidate_once, baseline_once)
                )
                first()
                second()
            cuda.synchronize()
            for index in range(benchmark_config.measurement_runs):
                if index % 2 == 0:
                    baseline_samples.append(
                        _timed_call(cuda, timing_stream, baseline_once)
                    )
                    candidate_samples.append(
                        _timed_call(cuda, timing_stream, candidate_once)
                    )
                else:
                    candidate_samples.append(
                        _timed_call(cuda, timing_stream, candidate_once)
                    )
                    baseline_samples.append(
                        _timed_call(cuda, timing_stream, baseline_once)
                    )
    except Exception as exc:
        return LocalCudaComparisonResult(
            candidate_id,
            case_signature,
            "local_cuda_proxy_interleaved",
            "failed",
            baseline_samples_ms=tuple(baseline_samples),
            candidate_samples_ms=tuple(candidate_samples),
            environment_fingerprint=fingerprint,
            message=f"CUDA comparison failed: {type(exc).__name__}: {exc}",
        )

    baseline_median = float(statistics.median(baseline_samples))
    candidate_median = float(statistics.median(candidate_samples))
    if baseline_median <= 0.0:
        return LocalCudaComparisonResult(
            candidate_id,
            case_signature,
            "local_cuda_proxy_interleaved",
            "failed",
            baseline_samples_ms=tuple(baseline_samples),
            candidate_samples_ms=tuple(candidate_samples),
            environment_fingerprint=fingerprint,
            message="baseline CUDA median must be positive",
        )
    percentile = float(benchmark_config.percentile)
    return LocalCudaComparisonResult(
        candidate_id,
        case_signature,
        "local_cuda_proxy_interleaved",
        "completed",
        baseline_local_latency_ms=baseline_median,
        candidate_local_latency_ms=candidate_median,
        baseline_percentile_latency_ms=_percentile(baseline_samples, percentile),
        candidate_percentile_latency_ms=_percentile(candidate_samples, percentile),
        candidate_over_baseline_latency_ratio=candidate_median / baseline_median,
        baseline_samples_ms=tuple(baseline_samples),
        candidate_samples_ms=tuple(candidate_samples),
        environment_fingerprint=fingerprint,
    )


def _cuda_setup(torch_module: Any, stream: Any) -> tuple[Any, Any, Any, str]:
    torch = torch_module or __import__("torch")
    cuda = torch.cuda
    if not cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    timing_stream = stream if stream is not None else cuda.current_stream()
    stream_context = cuda.stream(timing_stream) if stream is not None else nullcontext()
    return cuda, timing_stream, stream_context, _environment_fingerprint(
        torch, timing_stream
    )


def _timed_call(cuda: Any, timing_stream: Any, run_once: Callable[[], Any]) -> float:
    start = cuda.Event(enable_timing=True)
    end = cuda.Event(enable_timing=True)
    start.record(timing_stream)
    run_once()
    end.record(timing_stream)
    end.synchronize()
    elapsed = float(start.elapsed_time(end))
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("CUDA event returned an invalid duration")
    return elapsed


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _environment_fingerprint(torch: Any, timing_stream: Any) -> str:
    cuda = torch.cuda
    stream_device = getattr(timing_stream, "device", None)
    stream_index = getattr(stream_device, "index", None)
    if stream_index is None:
        raise RuntimeError("timing stream has no concrete device index")
    index = int(stream_index)
    payload = {
        "executor": "local_cuda_proxy",
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
        "device_index": index,
        "device_name": str(cuda.get_device_name(index)),
        "capability": tuple(cuda.get_device_capability(index)),
        "schema": "cuda-proxy-v1",
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
