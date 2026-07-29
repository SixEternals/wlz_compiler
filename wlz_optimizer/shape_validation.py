"""Evidence-only consistency checks for symbolic shape contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from wlz_optimizer.schemas import ShapeContract, ShapeObservation


@dataclass
class ShapeConsistencyIssue:
    code: str
    message: str
    symbol: Optional[str] = None
    tensor_name: Optional[str] = None
    axis: Optional[int] = None


@dataclass
class ConstraintConsistencyResult:
    kind: str
    dimension: str
    expected_value: int
    status: str
    observed_value: Optional[int] = None


@dataclass
class SymbolConsistencyResult:
    symbol: str
    status: str
    known_values: Dict[str, int] = field(default_factory=dict)
    unknown_bindings: List[str] = field(default_factory=list)
    constraints: List[ConstraintConsistencyResult] = field(default_factory=list)
    issues: List[ShapeConsistencyIssue] = field(default_factory=list)


@dataclass
class ShapeConsistencyResult:
    op_name: str
    case_id: str
    observation_signature: str
    status: str
    symbols: List[SymbolConsistencyResult] = field(default_factory=list)
    issues: List[ShapeConsistencyIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_shape_consistency(
    contract: ShapeContract,
    observation: ShapeObservation,
) -> ShapeConsistencyResult:
    """Compare explicit bindings with one observation without inferring unknowns."""

    issues: List[ShapeConsistencyIssue] = []
    if contract.op_name != observation.op_name:
        issues.append(
            ShapeConsistencyIssue(
                code="operator_mismatch",
                message=(
                    f"contract operator {contract.op_name!r} does not match "
                    f"observation operator {observation.op_name!r}"
                ),
            )
        )

    symbol_results: List[SymbolConsistencyResult] = []
    for dimension in contract.symbolic_dimensions:
        known_values: Dict[str, int] = {}
        unknown_bindings: List[str] = []
        symbol_issues: List[ShapeConsistencyIssue] = []

        for binding in dimension.bindings:
            label = f"{binding.tensor_name}[{binding.axis}]"
            shape = observation.tensor_shapes.get(binding.tensor_name)
            if shape is None:
                symbol_issues.append(
                    ShapeConsistencyIssue(
                        code="missing_tensor",
                        symbol=dimension.name,
                        tensor_name=binding.tensor_name,
                        axis=binding.axis,
                        message=f"tensor {binding.tensor_name!r} is absent from the observation",
                    )
                )
                continue
            if binding.axis >= len(shape):
                symbol_issues.append(
                    ShapeConsistencyIssue(
                        code="axis_out_of_range",
                        symbol=dimension.name,
                        tensor_name=binding.tensor_name,
                        axis=binding.axis,
                        message=f"axis {binding.axis} is outside tensor {binding.tensor_name!r} rank {len(shape)}",
                    )
                )
                continue

            value = shape[binding.axis]
            if value is None:
                unknown_bindings.append(label)
            else:
                known_values[label] = value

        distinct_values = set(known_values.values())
        if len(distinct_values) > 1:
            rendered = ", ".join(f"{name}={value}" for name, value in known_values.items())
            symbol_issues.append(
                ShapeConsistencyIssue(
                    code="value_mismatch",
                    symbol=dimension.name,
                    message=f"symbol {dimension.name!r} has conflicting observed values: {rendered}",
                )
            )

        constraint_results: List[ConstraintConsistencyResult] = []
        binding_incomplete = bool(symbol_issues or unknown_bindings)
        for constraint in (
            item for item in contract.constraints if item.dimension == dimension.name
        ):
            observed_value = next(iter(distinct_values)) if len(distinct_values) == 1 else None
            violation = observed_value is not None and _constraint_is_violated(
                constraint.kind,
                observed_value,
                constraint.value,
            )
            if violation:
                constraint_status = "inconsistent"
                symbol_issues.append(
                    ShapeConsistencyIssue(
                        code=f"{constraint.kind}_violation",
                        symbol=dimension.name,
                        message=(
                            f"symbol {dimension.name!r} observed value {observed_value} "
                            f"violates {constraint.kind}={constraint.value}"
                        ),
                    )
                )
            elif observed_value is None or binding_incomplete:
                constraint_status = "unknown"
            else:
                constraint_status = "consistent"
            constraint_results.append(
                ConstraintConsistencyResult(
                    kind=constraint.kind,
                    dimension=constraint.dimension,
                    expected_value=constraint.value,
                    observed_value=observed_value,
                    status=constraint_status,
                )
            )

        if symbol_issues:
            status = "inconsistent"
        elif unknown_bindings:
            status = "unknown"
        else:
            status = "consistent"
        symbol_results.append(
            SymbolConsistencyResult(
                symbol=dimension.name,
                status=status,
                known_values=known_values,
                unknown_bindings=unknown_bindings,
                constraints=constraint_results,
                issues=symbol_issues,
            )
        )
        issues.extend(symbol_issues)

    if issues:
        status = "inconsistent"
    elif any(item.status == "unknown" for item in symbol_results):
        status = "unknown"
    else:
        status = "consistent"

    return ShapeConsistencyResult(
        op_name=contract.op_name,
        case_id=observation.case_id,
        observation_signature=observation.signature(),
        status=status,
        symbols=symbol_results,
        issues=issues,
    )


def _constraint_is_violated(kind: str, observed: int, expected: int) -> bool:
    if kind == "min_inclusive":
        return observed < expected
    if kind == "max_inclusive":
        return observed > expected
    if kind == "divisible_by":
        return observed % expected != 0
    raise ValueError(f"unsupported shape constraint kind: {kind!r}")
