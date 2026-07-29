import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.schemas import (
    ShapeConstraint,
    ShapeContract,
    SymbolicDimension,
    TensorAxisBinding,
)
from wlz_optimizer.shape_boundaries import plan_boundary_cases


def _contract(constraints):
    return ShapeContract(
        op_name="demo_op",
        symbolic_dimensions=[
            SymbolicDimension("K", [TensorAxisBinding("x", 1)]),
            SymbolicDimension("M", [TensorAxisBinding("x", 0)]),
        ],
        constraints=constraints,
    )


class BoundaryCasePlannerTests(unittest.TestCase):
    def test_empty_contract_has_no_proposals(self) -> None:
        self.assertEqual(plan_boundary_cases(_contract([])), [])

    def test_combined_constraints_generate_deduplicated_boundaries(self) -> None:
        contract = _contract(
            [
                ShapeConstraint("min_inclusive", "M", 16),
                ShapeConstraint("max_inclusive", "M", 64),
                ShapeConstraint("divisible_by", "M", 16),
            ]
        )

        proposals = plan_boundary_cases(contract)

        self.assertEqual([item.value for item in proposals], [15, 16, 17, 63, 64, 65])
        self.assertEqual(len({(item.dimension, item.value) for item in proposals}), 6)
        statuses = {item.value: item.expected_contract_status for item in proposals}
        self.assertEqual(statuses[16], "satisfies")
        self.assertEqual(statuses[15], "violates")
        self.assertEqual(statuses[17], "violates")
        self.assertEqual(statuses[64], "satisfies")
        self.assertTrue(all(len(item.reasons) == 3 for item in proposals))

    def test_zero_boundary_drops_negative_value(self) -> None:
        proposals = plan_boundary_cases(
            _contract([ShapeConstraint("min_inclusive", "M", 0)])
        )

        self.assertEqual([item.value for item in proposals], [0, 1])

    def test_cross_dimension_output_is_stably_sorted(self) -> None:
        proposals = plan_boundary_cases(
            _contract(
                [
                    ShapeConstraint("min_inclusive", "M", 8),
                    ShapeConstraint("divisible_by", "K", 4),
                ]
            )
        )

        pairs = [(item.dimension, item.value) for item in proposals]
        self.assertEqual(pairs, sorted(pairs))
        self.assertEqual(pairs, [("K", 3), ("K", 4), ("K", 5), ("M", 7), ("M", 8), ("M", 9)])

    def test_proposals_are_permanently_marked_non_official(self) -> None:
        proposal = plan_boundary_cases(
            _contract([ShapeConstraint("divisible_by", "K", 8)])
        )[0]

        self.assertEqual(proposal.source, "derived_from_explicit_contract")
        self.assertIs(proposal.official, False)
        self.assertEqual(proposal.to_dict()["official"], False)
        self.assertIsInstance(proposal.to_dict()["reasons"], list)
        with self.assertRaises(TypeError):
            type(proposal)(
                dimension="K",
                value=8,
                reasons=[],
                expected_contract_status="satisfies",
                official=True,
            )
        with self.assertRaises(FrozenInstanceError):
            proposal.official = True
        with self.assertRaises(FrozenInstanceError):
            proposal.reasons = ()

    def test_planning_does_not_modify_contract(self) -> None:
        contract = _contract([ShapeConstraint("min_inclusive", "M", 8)])
        before = contract.to_dict()

        plan_boundary_cases(contract)

        self.assertEqual(contract.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
