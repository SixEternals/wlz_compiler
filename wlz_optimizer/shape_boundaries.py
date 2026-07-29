"""Generate local-only boundary proposals from explicit shape constraints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from wlz_optimizer.schemas import ShapeConstraint, ShapeContract


@dataclass(frozen=True)
class BoundaryCaseProposal:
    """A suggested symbolic value, not an observed or official test case."""

    dimension: str
    value: int
    reasons: Tuple[str, ...]
    expected_contract_status: str
    source: str = field(default="derived_from_explicit_contract", init=False)
    official: bool = field(default=False, init=False)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def plan_boundary_cases(contract: ShapeContract) -> List[BoundaryCaseProposal]:
    """Propose V-1, V, and V+1 for every explicit constraint value."""

    constraints_by_dimension: Dict[str, List[ShapeConstraint]] = {}
    for constraint in contract.constraints:
        constraints_by_dimension.setdefault(constraint.dimension, []).append(constraint)

    proposals: List[BoundaryCaseProposal] = []
    for dimension in sorted(constraints_by_dimension):
        constraints = constraints_by_dimension[dimension]
        values = {
            constraint.value + offset
            for constraint in constraints
            for offset in (-1, 0, 1)
            if constraint.value + offset >= 0
        }
        for value in sorted(values):
            evaluations = [_evaluate_constraint(item, value) for item in constraints]
            proposals.append(
                BoundaryCaseProposal(
                    dimension=dimension,
                    value=value,
                    reasons=tuple(reason for _, reason in evaluations),
                    expected_contract_status=(
                        "satisfies" if all(passed for passed, _ in evaluations) else "violates"
                    ),
                )
            )
    return proposals


def _evaluate_constraint(constraint: ShapeConstraint, value: int) -> tuple[bool, str]:
    if constraint.kind == "min_inclusive":
        passed = value >= constraint.value
    elif constraint.kind == "max_inclusive":
        passed = value <= constraint.value
    elif constraint.kind == "divisible_by":
        passed = value % constraint.value == 0
    else:
        raise ValueError(f"unsupported constraint kind: {constraint.kind!r}")
    state = "satisfies" if passed else "violates"
    return passed, f"{constraint.kind}={constraint.value}: value={value} ({state})"
