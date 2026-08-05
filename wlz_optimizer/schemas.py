"""Shared data contracts for candidates and executor results."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ShapeObservation:
    """One concrete or partially known input-shape observation.

    ``None`` is an explicit unknown dimension/value. The observation records
    evidence only; it does not infer ranges, divisibility, or hidden cases.
    """

    op_name: str
    case_id: str
    tensor_shapes: Dict[str, List[Optional[int]]]
    tensor_dtypes: Dict[str, Optional[str]] = field(default_factory=dict)
    scalars: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.op_name.strip():
            raise ValueError("op_name must not be empty")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.source.strip():
            raise ValueError("source must not be empty")

        for tensor_name, shape in self.tensor_shapes.items():
            if not tensor_name.strip():
                raise ValueError("tensor name must not be empty")
            if not isinstance(shape, list):
                raise ValueError(f"shape for {tensor_name!r} must be a list")
            for dimension in shape:
                if dimension is None:
                    continue
                if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                    raise ValueError(
                        f"shape dimensions for {tensor_name!r} must be non-negative integers or None"
                    )

        unknown_dtype_tensors = set(self.tensor_dtypes) - set(self.tensor_shapes)
        if unknown_dtype_tensors:
            names = ", ".join(sorted(unknown_dtype_tensors))
            raise ValueError(f"dtype provided for unknown tensor(s): {names}")
        for tensor_name, dtype in self.tensor_dtypes.items():
            if dtype is not None and (not isinstance(dtype, str) or not dtype.strip()):
                raise ValueError(f"dtype for {tensor_name!r} must be a non-empty string or None")

        for scalar_name, value in self.scalars.items():
            if not scalar_name.strip():
                raise ValueError("scalar name must not be empty")
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise ValueError(f"scalar {scalar_name!r} must be a JSON primitive or None")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"scalar {scalar_name!r} must be finite")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShapeObservation":
        return cls(
            op_name=data["op_name"],
            case_id=data["case_id"],
            tensor_shapes=data["tensor_shapes"],
            tensor_dtypes=data.get("tensor_dtypes", {}),
            scalars=data.get("scalars", {}),
            source=data.get("source", "unknown"),
            source_ref=data.get("source_ref"),
        )

    def signature(self) -> str:
        """Return a stable execution-input signature, excluding evidence provenance."""

        payload = {
            "op_name": self.op_name,
            "tensor_shapes": self.tensor_shapes,
            "tensor_dtypes": self.tensor_dtypes,
            "scalars": self.scalars,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class TensorInitializer:
    """Explicit, non-executing recipe for one tensor's initial values."""

    kind: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_kinds = {"randn", "zeros", "full", "randint", "literal"}
        if self.kind not in allowed_kinds:
            raise ValueError(f"unsupported tensor initializer kind: {self.kind!r}")
        if not isinstance(self.parameters, dict):
            raise ValueError("initializer parameters must be a mapping")

        expected_keys = {
            "randn": set(),
            "zeros": set(),
            "full": {"fill_value"},
            "randint": {"low", "high"},
            "literal": {"values"},
        }[self.kind]
        actual_keys = set(self.parameters)
        if actual_keys != expected_keys:
            raise ValueError(
                f"{self.kind} initializer parameters must be exactly "
                f"{sorted(expected_keys)!r}"
            )

        if self.kind == "full":
            fill_value = self.parameters["fill_value"]
            if (
                isinstance(fill_value, bool)
                or not isinstance(fill_value, (int, float))
                or (isinstance(fill_value, float) and not math.isfinite(fill_value))
            ):
                raise ValueError("full initializer fill_value must be a finite JSON number")
        elif self.kind == "randint":
            low = self.parameters["low"]
            high = self.parameters["high"]
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (low, high)):
                raise ValueError("randint initializer low/high must be integers")
            if low >= high:
                raise ValueError("randint initializer requires low < high")
        elif self.kind == "literal":
            values = self.parameters["values"]
            if not isinstance(values, list) or any(
                not isinstance(value, (bool, int, float))
                or (isinstance(value, float) and not math.isfinite(value))
                for value in values
            ):
                raise ValueError("literal initializer values must be finite JSON numbers or booleans")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorInitializer":
        if not isinstance(data, dict) or set(data) != {"kind", "parameters"}:
            raise ValueError("initializer must contain exactly kind and parameters")
        return cls(kind=data["kind"], parameters=data["parameters"])


@dataclass
class TensorInputContract:
    """Execution-relevant properties of one tensor input."""

    shape: List[int]
    dtype: str
    layout: str
    initializer: TensorInitializer
    strides: Optional[List[int]] = None
    mutable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.shape, list):
            raise ValueError("tensor shape must be a list")
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in self.shape
        ):
            raise ValueError("tensor dimensions must be non-negative integers")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise ValueError("tensor dtype must be a non-empty string")
        if self.layout not in {"contiguous", "strided"}:
            raise ValueError("tensor layout must be 'contiguous' or 'strided'")
        if not isinstance(self.initializer, TensorInitializer):
            raise ValueError("tensor initializer must be a TensorInitializer")
        if self.initializer.kind == "literal":
            if self.layout != "contiguous":
                raise ValueError("literal tensor inputs must use contiguous layout")
            expected = math.prod(self.shape) if self.shape else 1
            if len(self.initializer.parameters["values"]) != expected:
                raise ValueError("literal initializer value count must equal tensor shape numel")
        if self.layout == "contiguous" and self.strides is not None:
            raise ValueError("contiguous tensor inputs must omit strides")
        if self.layout == "strided" and (
            not isinstance(self.strides, list) or len(self.strides) != len(self.shape)
        ):
            raise ValueError("strided tensor inputs require one stride per dimension")
        if self.strides is not None and any(
            isinstance(stride, bool) or not isinstance(stride, int) or stride < 0
            for stride in self.strides
        ):
            raise ValueError("tensor strides must be non-negative integers")
        if not isinstance(self.mutable, bool):
            raise ValueError("tensor mutable must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorInputContract":
        return cls(
            shape=data["shape"],
            dtype=data["dtype"],
            layout=data["layout"],
            initializer=TensorInitializer.from_dict(data["initializer"]),
            strides=data.get("strides"),
            mutable=data.get("mutable", False),
        )


@dataclass
class TensorAliasGroup:
    """Tensor inputs that must share storage in one evaluation case."""

    tensor_names: List[str]

    def __post_init__(self) -> None:
        if not isinstance(self.tensor_names, list):
            raise ValueError("alias tensor_names must be a list")
        if len(self.tensor_names) < 2:
            raise ValueError("an alias group must contain at least two tensors")
        if any(not isinstance(name, str) or not name.strip() for name in self.tensor_names):
            raise ValueError("alias tensor names must be non-empty strings")
        if len(self.tensor_names) != len(set(self.tensor_names)):
            raise ValueError("alias tensor names must be unique within a group")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorAliasGroup":
        return cls(tensor_names=data["tensor_names"])


@dataclass
class ArgumentBinding:
    """Bind one entrypoint parameter to a declared tensor or scalar input."""

    parameter_name: str
    source_kind: str
    source_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_name, str) or not self.parameter_name.strip():
            raise ValueError("argument parameter_name must not be empty")
        if self.source_kind not in {"tensor", "scalar"}:
            raise ValueError("argument source_kind must be 'tensor' or 'scalar'")
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise ValueError("argument source_name must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArgumentBinding":
        return cls(
            parameter_name=data["parameter_name"],
            source_kind=data["source_kind"],
            source_name=data["source_name"],
        )


@dataclass
class ExecutionBinding:
    """Candidate wrapper entrypoint and its ordered input bindings."""

    entrypoint: str
    arguments: List[ArgumentBinding]

    def __post_init__(self) -> None:
        if not isinstance(self.entrypoint, str) or not self.entrypoint.strip():
            raise ValueError("execution entrypoint must not be empty")
        if not isinstance(self.arguments, list) or not self.arguments:
            raise ValueError("execution arguments must be a non-empty list")
        if any(not isinstance(item, ArgumentBinding) for item in self.arguments):
            raise ValueError("execution arguments must contain ArgumentBinding instances")
        parameter_names = [item.parameter_name for item in self.arguments]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("execution parameter names must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionBinding":
        return cls(
            entrypoint=data["entrypoint"],
            arguments=[ArgumentBinding.from_dict(item) for item in data["arguments"]],
        )


@dataclass
class ValueSelector:
    """Select a value from an invocation return or a post-call tensor input."""

    source: str
    tensor_name: Optional[str] = None
    path: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source not in {"return", "tensor"}:
            raise ValueError("selector source must be 'return' or 'tensor'")
        if not isinstance(self.path, list):
            raise ValueError("selector path must be a list")
        for item in self.path:
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                continue
            if isinstance(item, str) and item.strip():
                continue
            if isinstance(item, dict) and set(item) == {"tensor_slice"}:
                spec = item["tensor_slice"]
                if (
                    isinstance(spec, list)
                    and len(spec) == 3
                    and all(
                        isinstance(value, int) and not isinstance(value, bool) and value >= 0
                        for value in spec[:2]
                    )
                    and (
                        spec[2] is None
                        or (
                            isinstance(spec[2], int)
                            and not isinstance(spec[2], bool)
                            and spec[2] >= spec[1]
                        )
                    )
                ):
                    continue
            if isinstance(item, dict) and set(item) == {"tensor_flat_slice"}:
                spec = item["tensor_flat_slice"]
                if (
                    isinstance(spec, list)
                    and len(spec) == 2
                    and all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0
                        for value in spec
                    )
                    and spec[1] >= spec[0]
                ):
                    continue
            raise ValueError(
                "selector path items must be names, indexes, or "
                "tensor_slice/tensor_flat_slice specifications"
            )
        if self.source == "return" and self.tensor_name is not None:
            raise ValueError("return selectors must omit tensor_name")
        if self.source == "tensor" and (
            not isinstance(self.tensor_name, str) or not self.tensor_name.strip()
        ):
            raise ValueError("tensor selectors require tensor_name")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValueSelector":
        return cls(
            source=data["source"],
            tensor_name=data.get("tensor_name"),
            path=data.get("path", []),
        )


@dataclass
class OracleTarget:
    """Pair candidate/reference selectors for one explicit comparison target."""

    target_name: str
    kind: str
    candidate: ValueSelector
    reference: ValueSelector
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_name, str) or not self.target_name.strip():
            raise ValueError("oracle target_name must not be empty")
        if self.kind not in {"output", "side_effect"}:
            raise ValueError("oracle target kind must be 'output' or 'side_effect'")
        if not isinstance(self.candidate, ValueSelector) or not isinstance(
            self.reference, ValueSelector
        ):
            raise ValueError("oracle targets require candidate/reference selectors")
        if self.evidence not in {"public_assertion", "manifest_strengthening"}:
            raise ValueError(
                "oracle target evidence must be 'public_assertion' or 'manifest_strengthening'"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OracleTarget":
        return cls(
            target_name=data["target_name"],
            kind=data["kind"],
            candidate=ValueSelector.from_dict(data["candidate"]),
            reference=ValueSelector.from_dict(data["reference"]),
            evidence=data["evidence"],
        )


@dataclass
class OraclePolicy:
    """Versioned value, shape, metadata, or official segmented policy."""

    reference_id: str
    policy_id: str
    kind: str
    rtol: Optional[float] = None
    atol: Optional[float] = None
    equal_nan: bool = False
    dtype_family: Optional[str] = None
    accumulation_count: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference_id, str) or not self.reference_id.strip():
            raise ValueError("oracle reference_id must not be empty")
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise ValueError("oracle policy_id must not be empty")
        if self.kind not in {
            "exact",
            "allclose",
            "shape",
            "metadata",
            "official_segmented",
        }:
            raise ValueError(
                "oracle kind must be 'exact', 'allclose', 'shape', 'metadata', or "
                "'official_segmented'"
            )
        if not isinstance(self.equal_nan, bool):
            raise ValueError("oracle equal_nan must be a boolean")
        if self.kind in {"exact", "shape", "metadata"} and (
            self.rtol is not None or self.atol is not None
        ):
            raise ValueError(f"{self.kind} oracle policy must not define tolerances")
        if self.kind in {"shape", "metadata"} and self.equal_nan:
            raise ValueError(f"{self.kind} oracle policy must not enable equal_nan")
        if self.kind == "allclose":
            for name, value in (("rtol", self.rtol), ("atol", self.atol)):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"allclose oracle {name} must be finite and non-negative")
        if self.kind != "official_segmented" and (
            self.dtype_family is not None or self.accumulation_count is not None
        ):
            raise ValueError(
                f"{self.kind} oracle policy must not define official segmented fields"
            )
        if self.kind == "official_segmented":
            if self.rtol is not None or self.atol is not None:
                raise ValueError(
                    "official_segmented oracle policy must not define allclose tolerances"
                )
            if self.equal_nan:
                raise ValueError(
                    "official_segmented oracle policy must reject NaN reference values"
                )
            if self.dtype_family is not None and self.dtype_family not in {
                "fp16",
                "bf16",
                "fp32",
                "integer",
                "unknown",
            }:
                raise ValueError("unsupported official segmented dtype_family")
            if self.accumulation_count is not None and (
                isinstance(self.accumulation_count, bool)
                or not isinstance(self.accumulation_count, int)
                or self.accumulation_count <= 0
            ):
                raise ValueError(
                    "official segmented accumulation_count must be a positive integer or None"
                )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Keep legacy case signatures stable: segmented-only fields are not
        # part of older allclose/exact/shape/metadata policy payloads.
        if self.kind != "official_segmented":
            data.pop("dtype_family", None)
            data.pop("accumulation_count", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OraclePolicy":
        return cls(
            reference_id=data["reference_id"],
            policy_id=data["policy_id"],
            kind=data["kind"],
            rtol=data.get("rtol"),
            atol=data.get("atol"),
            equal_nan=data.get("equal_nan", False),
            dtype_family=data.get("dtype_family"),
            accumulation_count=data.get("accumulation_count"),
        )


@dataclass
class InputContract:
    """Tensor, scalar, layout, mutability, and alias inputs for one case."""

    tensors: Dict[str, TensorInputContract]
    scalars: Dict[str, Any] = field(default_factory=dict)
    alias_groups: List[TensorAliasGroup] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, dict):
            raise ValueError("tensors must be a mapping")
        if not isinstance(self.scalars, dict):
            raise ValueError("scalars must be a mapping")
        if not isinstance(self.alias_groups, list):
            raise ValueError("alias_groups must be a list")
        for name, tensor in self.tensors.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tensor input names must not be empty")
            if not isinstance(tensor, TensorInputContract):
                raise ValueError("tensor inputs must be TensorInputContract instances")
        for name, value in self.scalars.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("scalar input names must not be empty")
            if isinstance(value, (list, tuple)):
                value = tuple(value)
                self.scalars[name] = value
                items = value
            else:
                items = (value,)
            if any(
                item is not None and not isinstance(item, (bool, int, float, str))
                for item in items
            ):
                raise ValueError(
                    f"scalar {name!r} must be a JSON primitive or flat sequence"
                )
            if any(isinstance(item, float) and not math.isfinite(item) for item in items):
                raise ValueError(f"scalar {name!r} must contain only finite values")

        aliased_names = []
        for group in self.alias_groups:
            if not isinstance(group, TensorAliasGroup):
                raise ValueError("alias_groups must contain TensorAliasGroup instances")
            aliased_names.extend(group.tensor_names)
        unknown = set(aliased_names) - set(self.tensors)
        if unknown:
            raise ValueError(f"alias groups reference unknown tensors: {', '.join(sorted(unknown))}")
        if len(aliased_names) != len(set(aliased_names)):
            raise ValueError("a tensor cannot belong to multiple alias groups")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputContract":
        return cls(
            tensors={
                name: TensorInputContract.from_dict(tensor)
                for name, tensor in data.get("tensors", {}).items()
            },
            scalars=data.get("scalars", {}),
            alias_groups=[
                TensorAliasGroup.from_dict(group) for group in data.get("alias_groups", [])
            ],
        )

    def canonical_dict(self) -> Dict[str, Any]:
        data = self.to_dict()
        data["alias_groups"] = sorted(
            (sorted(group.tensor_names) for group in self.alias_groups),
            key=lambda names: tuple(names),
        )
        return data


@dataclass
class EvaluationCase:
    """One versioned execution input and oracle-policy contract."""

    op_name: str
    case_id: str
    inputs: InputContract
    seed: int
    execution: ExecutionBinding
    oracle_policy: OraclePolicy
    oracle_targets: List[OracleTarget]
    source: str = "unknown"
    source_ref: Optional[str] = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.op_name, str) or not self.op_name.strip():
            raise ValueError("op_name must not be empty")
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not isinstance(self.inputs, InputContract):
            raise ValueError("inputs must be an InputContract")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.execution, ExecutionBinding):
            raise ValueError("execution must be an ExecutionBinding")
        if not isinstance(self.oracle_policy, OraclePolicy):
            raise ValueError("oracle_policy must be an OraclePolicy")
        if not isinstance(self.oracle_targets, list) or not self.oracle_targets:
            raise ValueError("oracle_targets must be a non-empty list")
        if any(not isinstance(target, OracleTarget) for target in self.oracle_targets):
            raise ValueError("oracle_targets must contain OracleTarget instances")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must not be empty")
        if self.schema_version != 2:
            raise ValueError(f"unsupported evaluation case schema version: {self.schema_version}")

        available_sources = {("tensor", name) for name in self.inputs.tensors}
        available_sources.update(("scalar", name) for name in self.inputs.scalars)
        bound_sources = {
            (argument.source_kind, argument.source_name)
            for argument in self.execution.arguments
        }
        unknown_sources = bound_sources - available_sources
        if unknown_sources:
            raise ValueError(f"execution references unknown inputs: {sorted(unknown_sources)!r}")
        unbound_sources = available_sources - bound_sources
        if unbound_sources:
            raise ValueError(f"execution leaves inputs unbound: {sorted(unbound_sources)!r}")

        target_names = [target.target_name for target in self.oracle_targets]
        if len(target_names) != len(set(target_names)):
            raise ValueError("oracle target names must be unique")
        for target in self.oracle_targets:
            for selector in (target.candidate, target.reference):
                if selector.source == "tensor" and selector.tensor_name not in self.inputs.tensors:
                    raise ValueError(
                        f"oracle selector references unknown tensor: {selector.tensor_name}"
                    )
            if target.kind == "side_effect":
                tensor_name = target.candidate.tensor_name
                if target.candidate.source != "tensor" or not self.inputs.tensors[
                    tensor_name
                ].mutable:
                    raise ValueError(
                        "side-effect oracle targets require a mutable candidate tensor selector"
                    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationCase":
        return cls(
            op_name=data["op_name"],
            case_id=data["case_id"],
            inputs=InputContract.from_dict(data["inputs"]),
            seed=data.get("seed"),
            execution=ExecutionBinding.from_dict(data["execution"]),
            oracle_policy=OraclePolicy.from_dict(data["oracle_policy"]),
            oracle_targets=[
                OracleTarget.from_dict(target) for target in data["oracle_targets"]
            ],
            source=data.get("source", "unknown"),
            source_ref=data.get("source_ref"),
            schema_version=data.get("schema_version", 2),
        )

    def input_signature(self) -> str:
        """Hash concrete execution inputs, excluding oracle and provenance."""

        payload = {
            "signature_schema": "evaluation-input-v2",
            "op_name": self.op_name,
            "inputs": self.inputs.canonical_dict(),
            "seed": self.seed,
            "execution": self.execution.to_dict(),
        }
        return _stable_json_hash(payload)

    def signature(self) -> str:
        """Hash input and oracle semantics, excluding identity/provenance."""

        payload = {
            "signature_schema": "evaluation-case-v2",
            "input_signature": self.input_signature(),
            "oracle_policy": self.oracle_policy.to_dict(),
            "oracle_targets": [
                target.to_dict()
                for target in sorted(self.oracle_targets, key=lambda item: item.target_name)
            ],
        }
        return _stable_json_hash(payload)


def _stable_json_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CorrectnessErrorSummary:
    """Finite, structured mismatch evidence from one oracle comparison."""

    mismatch_kind: Optional[str] = None
    max_abs_error: Optional[float] = None
    max_rel_error: Optional[float] = None
    mismatch_count: Optional[int] = None
    compared_count: Optional[int] = None
    first_mismatch: Optional[str] = None

    def __post_init__(self) -> None:
        if all(value is None for value in asdict(self).values()):
            raise ValueError("correctness error summary must contain at least one value")
        for name in ("mismatch_kind", "first_mismatch"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        for name in ("max_abs_error", "max_rel_error"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative or None")
        for name in ("mismatch_count", "compared_count"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if (
            self.mismatch_count is not None
            and self.compared_count is not None
            and self.mismatch_count > self.compared_count
        ):
            raise ValueError("mismatch_count cannot exceed compared_count")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrectnessErrorSummary":
        return cls(
            mismatch_kind=data.get("mismatch_kind"),
            max_abs_error=data.get("max_abs_error"),
            max_rel_error=data.get("max_rel_error"),
            mismatch_count=data.get("mismatch_count"),
            compared_count=data.get("compared_count"),
            first_mismatch=data.get("first_mismatch"),
        )


@dataclass
class CorrectnessCaseResult:
    """Oracle result for one candidate and one complete EvaluationCase."""

    candidate_id: str
    case_id: str
    case_signature: str
    oracle_policy_id: str
    oracle_status: str
    error_summary: Optional[CorrectnessErrorSummary] = None
    message: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("candidate_id", "case_id", "oracle_policy_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if (
            not isinstance(self.case_signature, str)
            or len(self.case_signature) != 64
            or any(char not in "0123456789abcdef" for char in self.case_signature)
        ):
            raise ValueError("case_signature must be a lowercase SHA-256 hex string")
        allowed = {"passed", "failed", "oracle_error", "unknown"}
        if self.oracle_status not in allowed:
            raise ValueError(f"unsupported oracle_status: {self.oracle_status!r}")
        if self.error_summary is not None and not isinstance(
            self.error_summary, CorrectnessErrorSummary
        ):
            raise ValueError("error_summary must be a CorrectnessErrorSummary or None")
        if self.message is not None and (
            not isinstance(self.message, str) or not self.message.strip()
        ):
            raise ValueError("message must be a non-empty string or None")
        if self.oracle_status == "failed" and self.error_summary is None:
            raise ValueError("failed oracle result requires an error_summary")
        if self.oracle_status in {"oracle_error", "unknown"} and self.error_summary is not None:
            raise ValueError(f"{self.oracle_status} result must not contain an error_summary")
        if self.oracle_status == "oracle_error" and self.message is None:
            raise ValueError("oracle_error result requires a message")
        if (
            self.oracle_status == "passed"
            and self.error_summary is not None
            and self.error_summary.mismatch_count not in {None, 0}
        ):
            raise ValueError("passed oracle result cannot report mismatches")
        if self.schema_version != 1:
            raise ValueError(f"unsupported correctness result schema version: {self.schema_version}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrectnessCaseResult":
        summary = data.get("error_summary")
        return cls(
            candidate_id=data["candidate_id"],
            case_id=data["case_id"],
            case_signature=data["case_signature"],
            oracle_policy_id=data["oracle_policy_id"],
            oracle_status=data["oracle_status"],
            error_summary=(
                CorrectnessErrorSummary.from_dict(summary) if summary is not None else None
            ),
            message=data.get("message"),
            schema_version=data.get("schema_version", 1),
        )


@dataclass
class TensorAxisBinding:
    """Bind a symbolic dimension to one named tensor axis."""

    tensor_name: str
    axis: int

    def __post_init__(self) -> None:
        if not isinstance(self.tensor_name, str) or not self.tensor_name.strip():
            raise ValueError("tensor_name must not be empty")
        if isinstance(self.axis, bool) or not isinstance(self.axis, int) or self.axis < 0:
            raise ValueError("axis must be a non-negative integer")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorAxisBinding":
        return cls(tensor_name=data["tensor_name"], axis=data["axis"])


@dataclass
class SymbolicDimension:
    """An explicitly declared symbolic dimension and its tensor bindings."""

    name: str
    bindings: List[TensorAxisBinding]
    source: str = "user_declared"
    source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("symbolic dimension name must not be empty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("symbolic dimension source must not be empty")
        if not self.bindings:
            raise ValueError(f"symbolic dimension {self.name!r} must have at least one binding")
        keys = [(item.tensor_name, item.axis) for item in self.bindings]
        if len(keys) != len(set(keys)):
            raise ValueError(f"symbolic dimension {self.name!r} contains duplicate bindings")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolicDimension":
        return cls(
            name=data["name"],
            bindings=[TensorAxisBinding.from_dict(item) for item in data["bindings"]],
            source=data.get("source", "user_declared"),
            source_ref=data.get("source_ref"),
        )


@dataclass
class ShapeConstraint:
    """An explicit bound or divisibility constraint on one symbolic dimension."""

    kind: str
    dimension: str
    value: int
    source: str = "user_declared"
    source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        allowed = {"min_inclusive", "max_inclusive", "divisible_by"}
        if self.kind not in allowed:
            raise ValueError(f"unsupported shape constraint kind: {self.kind!r}")
        if not isinstance(self.dimension, str) or not self.dimension.strip():
            raise ValueError("constraint dimension must not be empty")
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise ValueError("constraint value must be an integer")
        if self.kind == "divisible_by" and self.value <= 0:
            raise ValueError("divisible_by value must be positive")
        if self.kind != "divisible_by" and self.value < 0:
            raise ValueError(f"{self.kind} value must be non-negative")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("constraint source must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShapeConstraint":
        return cls(
            kind=data["kind"],
            dimension=data["dimension"],
            value=data["value"],
            source=data.get("source", "user_declared"),
            source_ref=data.get("source_ref"),
        )


@dataclass
class ShapeContract:
    """Partial, evidence-backed symbolic shape declarations for one operator."""

    op_name: str
    symbolic_dimensions: List[SymbolicDimension]
    observation_signatures: List[str] = field(default_factory=list)
    evidence_scope: str = "partial"
    schema_version: int = 1
    constraints: List[ShapeConstraint] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.op_name, str) or not self.op_name.strip():
            raise ValueError("op_name must not be empty")
        if not isinstance(self.evidence_scope, str) or not self.evidence_scope.strip():
            raise ValueError("evidence_scope must not be empty")
        if self.schema_version != 1:
            raise ValueError(f"unsupported shape contract schema version: {self.schema_version}")

        names = [dimension.name for dimension in self.symbolic_dimensions]
        if len(names) != len(set(names)):
            raise ValueError("symbolic dimension names must be unique")

        bindings = [
            (binding.tensor_name, binding.axis)
            for dimension in self.symbolic_dimensions
            for binding in dimension.bindings
        ]
        if len(bindings) != len(set(bindings)):
            raise ValueError("a tensor axis cannot belong to multiple symbolic dimensions")

        constraint_keys = [(item.kind, item.dimension) for item in self.constraints]
        if len(constraint_keys) != len(set(constraint_keys)):
            raise ValueError("shape constraints must be unique by kind and dimension")
        unknown_dimensions = {item.dimension for item in self.constraints} - set(names)
        if unknown_dimensions:
            rendered = ", ".join(sorted(unknown_dimensions))
            raise ValueError(f"shape constraints reference unknown dimensions: {rendered}")
        bounds: Dict[str, Dict[str, int]] = {}
        for constraint in self.constraints:
            bounds.setdefault(constraint.dimension, {})[constraint.kind] = constraint.value
        for dimension, dimension_bounds in bounds.items():
            minimum = dimension_bounds.get("min_inclusive")
            maximum = dimension_bounds.get("max_inclusive")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"minimum exceeds maximum for symbolic dimension {dimension!r}")

        if len(self.observation_signatures) != len(set(self.observation_signatures)):
            raise ValueError("observation signatures must be unique")
        for signature in self.observation_signatures:
            if (
                not isinstance(signature, str)
                or len(signature) != 64
                or any(char not in "0123456789abcdef" for char in signature)
            ):
                raise ValueError("observation signatures must be lowercase SHA-256 hex strings")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShapeContract":
        return cls(
            op_name=data["op_name"],
            symbolic_dimensions=[
                SymbolicDimension.from_dict(item) for item in data.get("symbolic_dimensions", [])
            ],
            observation_signatures=data.get("observation_signatures", []),
            evidence_scope=data.get("evidence_scope", "partial"),
            schema_version=data.get("schema_version", 1),
            constraints=[ShapeConstraint.from_dict(item) for item in data.get("constraints", [])],
        )


LAUNCH_PROFILE_SOURCE_KINDS = frozenset(
    {
        "static_ast",
        "generator_config",
        "runtime_observation",
        "imported_manifest",
        "model_declared",
    }
)
TRUSTED_BOUNDARY_SOURCE_KINDS = frozenset(
    {"static_ast", "generator_config", "runtime_observation"}
)


@dataclass
class TileDimBinding:
    """Bind one launch parameter to a symbolic dimension and tile value."""

    parameter_name: str
    symbolic_dimension: str
    value: int
    source_kind: str
    source_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_name, str) or not self.parameter_name.strip():
            raise ValueError("parameter_name must be a non-empty string")
        if not isinstance(self.symbolic_dimension, str) or not self.symbolic_dimension.strip():
            raise ValueError("symbolic_dimension must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value <= 0:
            raise ValueError("tile value must be a positive integer")
        _validate_launch_profile_source(self.source_kind, self.source_ref)

    @property
    def trusted_for_boundary_proposals(self) -> bool:
        return self.source_kind in TRUSTED_BOUNDARY_SOURCE_KINDS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TileDimBinding":
        return cls(
            parameter_name=data["parameter_name"],
            symbolic_dimension=data["symbolic_dimension"],
            value=data["value"],
            source_kind=data["source_kind"],
            source_ref=data["source_ref"],
        )


@dataclass
class LaunchProfile:
    """Tile/block/launch metadata bound to one exact Candidate code hash."""

    candidate_code_hash: str
    source_kind: str
    source_ref: str
    tiles: List[TileDimBinding] = field(default_factory=list)
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None
    grid_rank: Optional[int] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_code_hash, str)
            or len(self.candidate_code_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in self.candidate_code_hash)
        ):
            raise ValueError("candidate_code_hash must be a lowercase SHA-256 hex string")
        _validate_launch_profile_source(self.source_kind, self.source_ref)
        if self.schema_version != 1:
            raise ValueError(f"unsupported launch profile schema version: {self.schema_version}")
        for tile in self.tiles:
            if not isinstance(tile, TileDimBinding):
                raise ValueError(f"each tile must be a TileDimBinding instance, got {type(tile)}")
        parameter_names = [tile.parameter_name for tile in self.tiles]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("tile parameter names must be unique")
        if self.num_warps is not None:
            if isinstance(self.num_warps, bool) or not isinstance(self.num_warps, int) or self.num_warps <= 0:
                raise ValueError("num_warps must be a positive integer or None")
        if self.num_stages is not None:
            if isinstance(self.num_stages, bool) or not isinstance(self.num_stages, int) or self.num_stages <= 0:
                raise ValueError("num_stages must be a positive integer or None")
        if self.grid_rank is not None and (
            isinstance(self.grid_rank, bool)
            or not isinstance(self.grid_rank, int)
            or not 1 <= self.grid_rank <= 3
        ):
            raise ValueError("grid_rank must be an integer from 1 to 3 or None")

    @property
    def trusted_for_boundary_proposals(self) -> bool:
        return self.source_kind in TRUSTED_BOUNDARY_SOURCE_KINDS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LaunchProfile":
        return cls(
            candidate_code_hash=data["candidate_code_hash"],
            source_kind=data["source_kind"],
            source_ref=data["source_ref"],
            tiles=[TileDimBinding.from_dict(t) for t in data.get("tiles", [])],
            num_warps=data.get("num_warps"),
            num_stages=data.get("num_stages"),
            grid_rank=data.get("grid_rank"),
            schema_version=data.get("schema_version", 1),
        )


def _validate_launch_profile_source(source_kind: str, source_ref: str) -> None:
    if source_kind not in LAUNCH_PROFILE_SOURCE_KINDS:
        allowed = ", ".join(sorted(LAUNCH_PROFILE_SOURCE_KINDS))
        raise ValueError(f"source_kind must be one of: {allowed}")
    if not isinstance(source_ref, str) or not source_ref.strip():
        raise ValueError("source_ref must be a non-empty traceable reference")


@dataclass
class Candidate:
    """A generated Triton candidate with provenance attached."""

    id: str
    op_name: str
    code: str
    code_hash: str
    parent_ids: List[str]
    generation: int
    mutation_kind: str
    model_used: Optional[str]
    prompt_id: Optional[str]
    status: str
    score: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    launch_profile: Optional[LaunchProfile] = None

    def __post_init__(self) -> None:
        if self.launch_profile is not None:
            if self.launch_profile.candidate_code_hash != self.code_hash:
                raise ValueError(
                    f"LaunchProfile candidate_code_hash "
                    f"{self.launch_profile.candidate_code_hash!r} does not match "
                    f"Candidate code_hash {self.code_hash!r}"
                )

    def to_dict(self, include_code: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_code:
            data.pop("code", None)
        return data


@dataclass
class EvalContext:
    """Information an executor may use while evaluating one candidate."""

    op_name: str
    input_dir: Path
    output_dir: Path
    required_functions: List[str] = field(default_factory=list)
    test_file: Optional[Path] = None
    baseline_file: Optional[Path] = None


@dataclass
class EvaluationResult:
    """Unified executor return value.

    Local/static executors must not fill real latency or speedup fields unless
    they performed a real timed execution. For the first local skeleton,
    proxy_score is only a sorting signal.
    """

    candidate_id: str
    executor: str
    status: str
    passed: bool
    correctness_ok: Optional[bool]
    compile_ok: Optional[bool]
    latency_ms: Optional[float]
    baseline_ms: Optional[float]
    speedup: Optional[float]
    proxy_score: Optional[float]
    error_type: Optional[str]
    error_message: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        fields = {
            "candidate_id",
            "executor",
            "status",
            "passed",
            "correctness_ok",
            "compile_ok",
            "latency_ms",
            "baseline_ms",
            "speedup",
            "proxy_score",
            "error_type",
            "error_message",
            "metadata",
        }
        return cls(**{key: data.get(key) for key in fields})


@dataclass
class OperatorInput:
    """Loaded source material for one operator directory."""

    op_name: str
    op_dir: Path
    baseline_file: Path
    test_file: Optional[Path]
    seeds: List[Dict[str, Any]]
    required_functions: List[str]


@dataclass
class CandidateEvaluation:
    """A candidate and its evaluation kept together for reporting."""

    candidate: Candidate
    result: EvaluationResult
    candidate_path: Optional[Path] = None
    top5_path: Optional[Path] = None

    def to_manifest_record(self, root: Path) -> Dict[str, Any]:
        candidate = self.candidate.to_dict(include_code=False)
        result = self.result.to_dict()
        record: Dict[str, Any] = {
            "candidate": candidate,
            "evaluation": result,
        }
        if self.candidate_path:
            record["candidate_path"] = str(self.candidate_path.relative_to(root))
        if self.top5_path:
            record["top5_path"] = str(self.top5_path.relative_to(root))
        return record
