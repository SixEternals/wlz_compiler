import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.schemas import (
    ShapeContract,
    ShapeObservation,
    SymbolicDimension,
    TensorAxisBinding,
)
from wlz_optimizer.shape_validation import validate_shape_consistency


class ShapeObservationTests(unittest.TestCase):
    def _observation(self, **overrides) -> ShapeObservation:
        values = {
            "op_name": "demo_op",
            "case_id": "public-case-1",
            "tensor_shapes": {"x": [32, 128], "y": [32, None]},
            "tensor_dtypes": {"x": "torch.float16", "y": None},
            "scalars": {"group_size": 128, "causal": False, "limit": None},
            "source": "static_test_hint",
            "source_ref": "test_demo_op_1.py:20",
        }
        values.update(overrides)
        return ShapeObservation(**values)

    def test_json_round_trip_preserves_explicit_unknowns(self) -> None:
        original = self._observation()
        payload = json.loads(json.dumps(original.to_dict()))
        restored = ShapeObservation.from_dict(payload)

        self.assertEqual(restored.to_dict(), original.to_dict())
        self.assertIsNone(restored.tensor_shapes["y"][1])
        self.assertIsNone(restored.tensor_dtypes["y"])
        self.assertIsNone(restored.scalars["limit"])

    def test_signature_is_stable_across_mapping_and_provenance_order(self) -> None:
        first = self._observation()
        second = self._observation(
            case_id="runtime-case-9",
            tensor_shapes={"y": [32, None], "x": [32, 128]},
            tensor_dtypes={"y": None, "x": "torch.float16"},
            scalars={"limit": None, "causal": False, "group_size": 128},
            source="runtime_observation",
            source_ref="worker-3",
        )

        self.assertEqual(first.signature(), second.signature())

    def test_execution_relevant_changes_change_signature(self) -> None:
        baseline = self._observation().signature()

        self.assertNotEqual(
            baseline,
            self._observation(tensor_shapes={"x": [33, 128], "y": [33, None]}).signature(),
        )
        self.assertNotEqual(
            baseline,
            self._observation(
                tensor_dtypes={"x": "torch.float32", "y": None}
            ).signature(),
        )
        self.assertNotEqual(
            baseline,
            self._observation(
                scalars={"group_size": 64, "causal": False, "limit": None}
            ).signature(),
        )

    def test_zero_and_unknown_dimensions_are_allowed(self) -> None:
        observation = self._observation(
            tensor_shapes={"x": [0, None]},
            tensor_dtypes={"x": None},
        )

        self.assertEqual(observation.tensor_shapes["x"], [0, None])

    def test_invalid_dimensions_are_rejected(self) -> None:
        for invalid in (-1, True, 1.5, "N"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self._observation(
                        tensor_shapes={"x": [invalid, 128]},
                        tensor_dtypes={"x": "torch.float16"},
                    )

    def test_dtype_cannot_reference_an_unknown_tensor(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown tensor"):
            self._observation(
                tensor_shapes={"x": [32, 128]},
                tensor_dtypes={"missing": "torch.float16"},
            )

    def test_non_json_or_non_finite_scalars_are_rejected(self) -> None:
        for invalid in ([1], float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self._observation(scalars={"value": invalid})


class ShapeContractTests(unittest.TestCase):
    def _dimension(self, name="M", bindings=None) -> SymbolicDimension:
        return SymbolicDimension(
            name=name,
            bindings=bindings
            or [TensorAxisBinding("x", 0), TensorAxisBinding("y", 0)],
            source="user_declared",
            source_ref="design-note-1",
        )

    def test_json_round_trip_preserves_cross_tensor_bindings(self) -> None:
        contract = ShapeContract(
            op_name="demo_op",
            symbolic_dimensions=[
                self._dimension(),
                self._dimension("K", [TensorAxisBinding("x", 1)]),
            ],
            observation_signatures=["a" * 64],
        )

        restored = ShapeContract.from_dict(json.loads(json.dumps(contract.to_dict())))

        self.assertEqual(restored.to_dict(), contract.to_dict())
        self.assertEqual(restored.evidence_scope, "partial")
        self.assertEqual(
            restored.symbolic_dimensions[0].to_dict()["bindings"],
            [{"tensor_name": "x", "axis": 0}, {"tensor_name": "y", "axis": 0}],
        )

    def test_axis_must_be_a_non_negative_integer(self) -> None:
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    TensorAxisBinding("x", invalid)

    def test_symbolic_dimension_requires_unique_bindings(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate bindings"):
            self._dimension(
                bindings=[TensorAxisBinding("x", 0), TensorAxisBinding("x", 0)]
            )

    def test_contract_rejects_duplicate_names_and_cross_symbol_axis_conflicts(self) -> None:
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            ShapeContract("demo_op", [self._dimension(), self._dimension()])

        with self.assertRaisesRegex(ValueError, "multiple symbolic dimensions"):
            ShapeContract(
                "demo_op",
                [
                    self._dimension("M", [TensorAxisBinding("x", 0)]),
                    self._dimension("N", [TensorAxisBinding("x", 0)]),
                ],
            )

    def test_contract_rejects_invalid_or_duplicate_observation_signatures(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            ShapeContract("demo_op", [], observation_signatures=["not-a-hash"])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            ShapeContract("demo_op", [], observation_signatures=["a" * 64, "a" * 64])


class ShapeConsistencyTests(unittest.TestCase):
    def _contract(self, bindings=None) -> ShapeContract:
        return ShapeContract(
            op_name="demo_op",
            symbolic_dimensions=[
                SymbolicDimension(
                    "M",
                    bindings
                    or [TensorAxisBinding("x", 0), TensorAxisBinding("y", 0)],
                )
            ],
        )

    def _observation(self, shapes) -> ShapeObservation:
        return ShapeObservation(
            op_name="demo_op",
            case_id="case-1",
            tensor_shapes=shapes,
            source="static_test_hint",
        )

    def test_equal_known_values_are_consistent(self) -> None:
        result = validate_shape_consistency(
            self._contract(),
            self._observation({"x": [32, 128], "y": [32, 64]}),
        )

        self.assertEqual(result.status, "consistent")
        self.assertEqual(result.symbols[0].known_values, {"x[0]": 32, "y[0]": 32})
        self.assertEqual(result.issues, [])

    def test_none_remains_unknown_without_inference(self) -> None:
        observation = self._observation({"x": [32], "y": [None]})
        result = validate_shape_consistency(self._contract(), observation)

        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.symbols[0].known_values, {"x[0]": 32})
        self.assertEqual(result.symbols[0].unknown_bindings, ["y[0]"])
        self.assertIsNone(observation.tensor_shapes["y"][0])

    def test_conflicting_known_values_are_inconsistent(self) -> None:
        result = validate_shape_consistency(
            self._contract(),
            self._observation({"x": [32], "y": [64]}),
        )

        self.assertEqual(result.status, "inconsistent")
        self.assertEqual([issue.code for issue in result.issues], ["value_mismatch"])

    def test_missing_tensor_and_out_of_range_axis_are_inconsistent(self) -> None:
        missing = validate_shape_consistency(
            self._contract(),
            self._observation({"x": [32]}),
        )
        out_of_range = validate_shape_consistency(
            self._contract([TensorAxisBinding("x", 1)]),
            self._observation({"x": [32]}),
        )

        self.assertEqual([issue.code for issue in missing.issues], ["missing_tensor"])
        self.assertEqual([issue.code for issue in out_of_range.issues], ["axis_out_of_range"])

    def test_operator_mismatch_has_highest_priority(self) -> None:
        observation = self._observation({"x": [None], "y": [None]})
        observation.op_name = "other_op"

        result = validate_shape_consistency(self._contract(), observation)

        self.assertEqual(result.status, "inconsistent")
        self.assertEqual([issue.code for issue in result.issues], ["operator_mismatch"])


if __name__ == "__main__":
    unittest.main()
