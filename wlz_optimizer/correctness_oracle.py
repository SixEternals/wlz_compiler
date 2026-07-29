"""Backend-neutral correctness comparison for captured invocation values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .schemas import CorrectnessErrorSummary, OraclePolicy, OracleTarget, ValueSelector


_MAX_MESSAGE_BYTES = 512
_SCALAR_TYPES = (type(None), bool, int, float, str)
_TENSOR_VALUE_TYPES = (bool, int, float)
SHAPE_ONLY_DTYPE = "shape_only"


@dataclass(frozen=True)
class TensorSnapshot:
    """A flattened tensor value captured by a backend adapter."""

    shape: Tuple[int, ...]
    dtype: str
    values: Tuple[Any, ...]
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.shape
        ):
            raise ValueError("tensor snapshot shape must contain non-negative integers")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("tensor snapshot dtype must not be empty")
        if self.device is not None and (
            not isinstance(self.device, str)
            or not self.device.isidentifier()
            or self.device != self.device.lower()
        ):
            raise ValueError("tensor snapshot device must be a normalized device type")
        if not isinstance(self.values, tuple) or any(
            not isinstance(value, _TENSOR_VALUE_TYPES) for value in self.values
        ):
            raise ValueError("tensor snapshot values must be numeric or boolean tuples")
        if self.dtype == SHAPE_ONLY_DTYPE:
            if self.values:
                raise ValueError("shape-only tensor snapshots must omit values")
            return
        expected = math.prod(self.shape) if self.shape else 1
        if len(self.values) != expected:
            raise ValueError("tensor snapshot value count must equal shape numel")


@dataclass(frozen=True)
class InvocationSnapshot:
    """Return value and post-call tensor state from one invocation."""

    return_value: Any
    tensors: Dict[str, TensorSnapshot]

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, dict):
            raise ValueError("invocation tensors must be a mapping")
        for name, value in self.tensors.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("invocation tensor names must not be empty")
            if not isinstance(value, TensorSnapshot):
                raise ValueError("invocation tensors must contain TensorSnapshot values")


def decode_invocation_snapshot(value: Any) -> InvocationSnapshot:
    """Decode the bounded JSON snapshot format used by isolated runners."""

    if not isinstance(value, dict) or set(value) != {"return", "tensors"}:
        raise ValueError("invocation snapshot must contain return and tensors")
    if not isinstance(value["tensors"], dict):
        raise ValueError("invocation snapshot tensors must be a mapping")
    tensors = {
        name: _decode_snapshot_value(encoded)
        for name, encoded in value["tensors"].items()
    }
    if any(not isinstance(tensor, TensorSnapshot) for tensor in tensors.values()):
        raise ValueError("invocation tensor entries must be tensor snapshots")
    return InvocationSnapshot(
        return_value=_decode_snapshot_value(value["return"]),
        tensors=tensors,
    )


def _decode_snapshot_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) in (
        {"shape", "dtype", "values"},
        {"shape", "dtype", "values", "device"},
    ):
        shape = value["shape"]
        values = value["values"]
        if not isinstance(shape, list) or not isinstance(values, list):
            raise ValueError("tensor snapshot shape and values must be lists")
        return TensorSnapshot(
            tuple(shape), value["dtype"], tuple(values), value.get("device")
        )
    if isinstance(value, list):
        return tuple(_decode_snapshot_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _decode_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, _SCALAR_TYPES):
        return value
    raise ValueError(f"unsupported snapshot value type: {type(value).__name__}")


@dataclass(frozen=True)
class OracleComparison:
    """One local comparison result, ready for D1 case-result wrapping."""

    status: str
    compared_targets: int
    error_summary: Optional[CorrectnessErrorSummary] = None
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "oracle_error"}:
            raise ValueError(f"unsupported oracle comparison status: {self.status}")
        if (
            isinstance(self.compared_targets, bool)
            or not isinstance(self.compared_targets, int)
            or self.compared_targets < 0
        ):
            raise ValueError("compared_targets must be a non-negative integer")
        if self.status == "failed" and self.error_summary is None:
            raise ValueError("failed oracle comparison requires an error summary")
        if self.status == "oracle_error" and not self.message:
            raise ValueError("oracle_error comparison requires a message")
        if self.status == "oracle_error" and self.error_summary is not None:
            raise ValueError("oracle_error comparison must not contain an error summary")
        if (
            self.status == "passed"
            and self.error_summary is not None
            and self.error_summary.mismatch_count not in {None, 0}
        ):
            raise ValueError("passed oracle comparison cannot report mismatches")


@dataclass
class _Stats:
    compared: int = 0
    mismatches: int = 0
    max_abs: Optional[float] = None
    max_rel: Optional[float] = None
    first_kind: Optional[str] = None
    first_message: Optional[str] = None

    def mismatch(self, kind: str, message: str) -> None:
        self.compared += 1
        self.mismatches += 1
        if self.first_kind is None:
            self.first_kind = kind
            self.first_message = _bounded(message)

    def value(self, matches: bool, kind: str, message: str, absolute: Optional[float], relative: Optional[float]) -> None:
        self.compared += 1
        if absolute is not None and math.isfinite(absolute):
            self.max_abs = absolute if self.max_abs is None else max(self.max_abs, absolute)
        if relative is not None and math.isfinite(relative):
            self.max_rel = relative if self.max_rel is None else max(self.max_rel, relative)
        if not matches:
            self.mismatches += 1
            if self.first_kind is None:
                self.first_kind = kind
                self.first_message = _bounded(message)

    def summary(self) -> CorrectnessErrorSummary:
        return CorrectnessErrorSummary(
            mismatch_kind=self.first_kind,
            max_abs_error=self.max_abs,
            max_rel_error=self.max_rel,
            mismatch_count=self.mismatches,
            compared_count=self.compared,
            first_mismatch=self.first_message,
        )


class _SelectionError(ValueError):
    pass


def compare_oracle(
    policy: OraclePolicy,
    targets: Sequence[OracleTarget],
    candidate: InvocationSnapshot,
    reference: InvocationSnapshot,
) -> OracleComparison:
    """Compare all explicit targets without importing a tensor framework."""

    if not isinstance(policy, OraclePolicy):
        raise TypeError("policy must be an OraclePolicy")
    if not targets or any(not isinstance(target, OracleTarget) for target in targets):
        raise ValueError("targets must contain at least one OracleTarget")
    if not isinstance(candidate, InvocationSnapshot) or not isinstance(
        reference, InvocationSnapshot
    ):
        raise TypeError("candidate and reference must be InvocationSnapshot values")

    stats = _Stats()
    compared_targets = 0
    for target in sorted(targets, key=lambda item: item.target_name):
        try:
            expected = _select(reference, target.reference)
        except _SelectionError as exc:
            return OracleComparison(
                "oracle_error",
                compared_targets,
                message=_bounded(f"reference target {target.target_name}: {exc}"),
            )
        compared_targets += 1
        try:
            actual = _select(candidate, target.candidate)
        except _SelectionError as exc:
            stats.mismatch(
                "structure", f"{target.kind}:{target.target_name}: candidate selector: {exc}"
            )
            continue
        _compare_value(
            actual,
            expected,
            policy,
            stats,
            f"{target.kind}:{target.target_name}",
        )

    summary = stats.summary()
    return OracleComparison(
        "passed" if stats.mismatches == 0 else "failed",
        compared_targets,
        error_summary=summary,
    )


def _select(snapshot: InvocationSnapshot, selector: ValueSelector) -> Any:
    if selector.source == "return":
        value = snapshot.return_value
    else:
        try:
            value = snapshot.tensors[selector.tensor_name]
        except KeyError as exc:
            raise _SelectionError(f"missing tensor {selector.tensor_name!r}") from exc
    for item in selector.path:
        if isinstance(item, int) and isinstance(value, (list, tuple)):
            if item >= len(value):
                raise _SelectionError(f"index {item} is out of range")
            value = value[item]
        elif isinstance(item, str) and isinstance(value, dict):
            if item not in value:
                raise _SelectionError(f"missing key {item!r}")
            value = value[item]
        elif (
            isinstance(item, dict)
            and set(item) == {"tensor_slice"}
            and isinstance(value, TensorSnapshot)
        ):
            value = _slice_tensor_last_axis(value, item["tensor_slice"])
        elif (
            isinstance(item, dict)
            and set(item) == {"tensor_flat_slice"}
            and isinstance(value, TensorSnapshot)
        ):
            value = _slice_tensor_flat(value, item["tensor_flat_slice"])
        else:
            raise _SelectionError(f"cannot apply path item {item!r}")
    return value


def _slice_tensor_last_axis(
    tensor: TensorSnapshot, spec: list[Any]
) -> TensorSnapshot:
    axis, start, stop = spec
    if not tensor.shape or axis != len(tensor.shape) - 1:
        raise _SelectionError("tensor_slice currently supports the last axis only")
    width = tensor.shape[-1]
    stop = width if stop is None else stop
    if start > width or stop > width:
        raise _SelectionError(
            f"tensor_slice [{start}:{stop}] exceeds last-axis size {width}"
        )
    shape = tensor.shape[:-1] + (stop - start,)
    if tensor.dtype == SHAPE_ONLY_DTYPE:
        return TensorSnapshot(shape, tensor.dtype, (), tensor.device)
    rows = math.prod(tensor.shape[:-1]) if tensor.shape[:-1] else 1
    values = tuple(
        value
        for row in range(rows)
        for value in tensor.values[row * width + start : row * width + stop]
    )
    return TensorSnapshot(shape, tensor.dtype, values, tensor.device)


def _slice_tensor_flat(tensor: TensorSnapshot, spec: list[int]) -> TensorSnapshot:
    start, stop = spec
    numel = math.prod(tensor.shape) if tensor.shape else 1
    if start > numel or stop > numel:
        raise _SelectionError(
            f"tensor_flat_slice [{start}:{stop}] exceeds tensor numel {numel}"
        )
    shape = (stop - start,)
    values = () if tensor.dtype == SHAPE_ONLY_DTYPE else tensor.values[start:stop]
    return TensorSnapshot(shape, tensor.dtype, values, tensor.device)


def _compare_value(
    actual: Any,
    expected: Any,
    policy: OraclePolicy,
    stats: _Stats,
    path: str,
) -> None:
    if isinstance(actual, TensorSnapshot) or isinstance(expected, TensorSnapshot):
        if not isinstance(actual, TensorSnapshot) or not isinstance(expected, TensorSnapshot):
            stats.mismatch("structure", f"{path}: tensor/non-tensor structure differs")
            return
        _compare_tensor(actual, expected, policy, stats, path)
        return
    if type(actual) is not type(expected):
        mismatch_kind = (
            "structure"
            if isinstance(actual, (dict, list, tuple))
            or isinstance(expected, (dict, list, tuple))
            else "dtype"
        )
        stats.mismatch(
            mismatch_kind,
            f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}",
        )
        return
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            stats.mismatch("structure", f"{path}: mapping keys differ")
            return
        for key in sorted(expected, key=lambda item: (type(item).__name__, repr(item))):
            _compare_value(actual[key], expected[key], policy, stats, f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            stats.mismatch("structure", f"{path}: expected length {len(expected)}, got {len(actual)}")
            return
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare_value(actual_item, expected_item, policy, stats, f"{path}[{index}]")
        return
    if policy.kind in {"shape", "metadata"}:
        stats.mismatch(
            "structure", f"{path}: {policy.kind} policy requires tensor leaves"
        )
        return
    if isinstance(expected, _SCALAR_TYPES):
        _compare_scalar(actual, expected, policy, stats, path, floating=isinstance(expected, float))
        return
    stats.mismatch("structure", f"{path}: unsupported value type {type(expected).__name__}")


def _compare_tensor(
    actual: TensorSnapshot,
    expected: TensorSnapshot,
    policy: OraclePolicy,
    stats: _Stats,
    path: str,
) -> None:
    if actual.shape != expected.shape:
        stats.mismatch("shape", f"{path}: expected shape {expected.shape}, got {actual.shape}")
        return
    if policy.kind == "shape":
        stats.value(True, "shape", "", None, None)
        return
    if actual.dtype != expected.dtype:
        stats.mismatch("dtype", f"{path}: expected dtype {expected.dtype}, got {actual.dtype}")
        return
    if policy.kind == "metadata":
        if expected.device is not None and actual.device != expected.device:
            stats.mismatch(
                "device",
                f"{path}: expected device {expected.device}, got {actual.device}",
            )
            return
        stats.value(True, "metadata", "", None, None)
        return
    floating = _is_floating_dtype(expected.dtype)
    for index, (actual_item, expected_item) in enumerate(zip(actual.values, expected.values)):
        _compare_scalar(actual_item, expected_item, policy, stats, f"{path}[{index}]", floating)


def _compare_scalar(
    actual: Any,
    expected: Any,
    policy: OraclePolicy,
    stats: _Stats,
    path: str,
    floating: bool,
) -> None:
    numeric = (
        not isinstance(actual, bool)
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and isinstance(expected, (int, float))
    )
    if floating and numeric:
        actual_float, expected_float = float(actual), float(expected)
        if math.isnan(actual_float) or math.isnan(expected_float):
            matches = policy.equal_nan and math.isnan(actual_float) and math.isnan(expected_float)
            stats.value(matches, "nan", f"{path}: NaN values differ", None, None)
            return
        if actual_float == expected_float:
            stats.value(True, "value", "", 0.0, 0.0 if expected_float else None)
            return
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            stats.value(False, "value", f"{path}: expected {expected!r}, got {actual!r}", None, None)
            return
        absolute = abs(actual_float - expected_float)
        relative = absolute / abs(expected_float) if expected_float != 0 else None
        matches = policy.kind == "allclose" and absolute <= policy.atol + policy.rtol * abs(expected_float)
        stats.value(matches, "value", f"{path}: expected {expected!r}, got {actual!r}", absolute, relative)
        return
    matches = type(actual) is type(expected) and actual == expected
    absolute = None
    relative = None
    if numeric and math.isfinite(float(actual)) and math.isfinite(float(expected)):
        absolute = abs(float(actual) - float(expected))
        relative = absolute / abs(float(expected)) if expected != 0 else None
    stats.value(matches, "value", f"{path}: expected {expected!r}, got {actual!r}", absolute, relative)


def _is_floating_dtype(dtype: str) -> bool:
    name = dtype.lower().rsplit(".", 1)[-1]
    return "float" in name or name in {"half", "double"}


def _bounded(message: str) -> str:
    return message.encode("utf-8", errors="replace")[:_MAX_MESSAGE_BYTES].decode(
        "utf-8", errors="ignore"
    )
