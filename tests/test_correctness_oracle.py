"""Focused tests for the backend-neutral correctness oracle."""

import unittest

from wlz_optimizer.correctness_oracle import (
    InvocationSnapshot,
    OracleComparison,
    TensorSnapshot,
    compare_oracle,
    decode_invocation_snapshot,
)
from wlz_optimizer.schemas import (
    CorrectnessErrorSummary,
    OraclePolicy,
    OracleTarget,
    ValueSelector,
)


def tensor(values, shape=None, dtype="float32", device=None):
    values = tuple(values)
    return TensorSnapshot(
        shape=shape or (len(values),), dtype=dtype, values=values, device=device
    )


def policy(kind="allclose", equal_nan=False):
    return OraclePolicy(
        reference_id="reference.v1",
        policy_id=f"{kind}.v1",
        kind=kind,
        rtol=1e-3 if kind == "allclose" else None,
        atol=1e-4 if kind == "allclose" else None,
        equal_nan=equal_nan,
    )


def target(name="out", kind="output", candidate=None, reference=None):
    return OracleTarget(
        target_name=name,
        kind=kind,
        candidate=candidate or ValueSelector("return"),
        reference=reference or ValueSelector("return"),
        evidence="public_assertion",
    )


def snapshot(return_value=None, **tensors):
    return InvocationSnapshot(return_value=return_value, tensors=tensors)


class CorrectnessOracleTests(unittest.TestCase):
    def test_nested_output_and_side_effect_pass_with_metrics(self):
        targets = [
            target(candidate=ValueSelector("return", path=["out"]), reference=ValueSelector("return", path=["out"])),
            target(
                "state",
                "side_effect",
                candidate=ValueSelector("tensor", tensor_name="state"),
                reference=ValueSelector("return", path=["state"]),
            ),
        ]
        reference = snapshot({"out": tensor([1.0, 2.0]), "state": tensor([3.0])})
        candidate = snapshot({"out": tensor([1.0005, 2.0])}, state=tensor([3.0]))

        result = compare_oracle(policy(), targets, candidate, reference)

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.compared_targets, 2)
        self.assertEqual(result.error_summary.mismatch_count, 0)
        self.assertEqual(result.error_summary.compared_count, 3)
        self.assertAlmostEqual(result.error_summary.max_abs_error, 0.0005)

    def test_structure_shape_and_dtype_fail_before_value_comparison(self):
        failures = (
            ("structure", [tensor([1.0])], (tensor([1.0]),)),
            ("shape", tensor([1.0, 2.0], (2,)), tensor([1.0, 2.0], (1, 2))),
            ("dtype", tensor([1], dtype="int32"), tensor([1], dtype="int64")),
        )
        for mismatch_kind, actual, expected in failures:
            with self.subTest(mismatch_kind=mismatch_kind):
                result = compare_oracle(policy(), [target()], snapshot(actual), snapshot(expected))
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.error_summary.mismatch_kind, mismatch_kind)
                self.assertEqual(result.error_summary.mismatch_count, 1)

    def test_allclose_boundary_and_exact_policy(self):
        within = compare_oracle(
            policy(), [target()], snapshot(tensor([1.001])), snapshot(tensor([1.0]))
        )
        outside = compare_oracle(
            policy(), [target()], snapshot(tensor([1.002])), snapshot(tensor([1.0]))
        )
        exact = compare_oracle(
            policy("exact"), [target()], snapshot(tensor([1.00001])), snapshot(tensor([1.0]))
        )

        self.assertEqual(within.status, "passed")
        self.assertEqual(outside.status, "failed")
        self.assertEqual(exact.status, "failed")
        self.assertGreater(outside.error_summary.max_rel_error, 0)

    def test_shape_policy_ignores_dtype_and_values_but_rejects_shape_mismatch(self):
        same_shape = compare_oracle(
            policy("shape"),
            [target()],
            snapshot(tensor([9, 8], shape=(1, 2), dtype="int32")),
            snapshot(tensor([1.0, 2.0], shape=(1, 2), dtype="float32")),
        )
        wrong_shape = compare_oracle(
            policy("shape"),
            [target()],
            snapshot(tensor([9, 8], shape=(2,), dtype="int32")),
            snapshot(tensor([1.0, 2.0], shape=(1, 2), dtype="float32")),
        )

        self.assertEqual(same_shape.status, "passed")
        self.assertEqual(same_shape.error_summary.compared_count, 1)
        self.assertIsNone(same_shape.error_summary.max_abs_error)
        self.assertEqual(wrong_shape.status, "failed")
        self.assertEqual(wrong_shape.error_summary.mismatch_kind, "shape")

    def test_shape_policy_rejects_scalar_targets_and_numeric_options(self):
        scalar = compare_oracle(
            policy("shape"), [target()], snapshot(1), snapshot(2)
        )

        self.assertEqual(scalar.status, "failed")
        self.assertEqual(scalar.error_summary.mismatch_kind, "structure")
        with self.assertRaisesRegex(ValueError, "must not define tolerances"):
            OraclePolicy("reference", "shape.v1", "shape", rtol=0.0)
        with self.assertRaisesRegex(ValueError, "must not enable equal_nan"):
            OraclePolicy("reference", "shape.v1", "shape", equal_nan=True)

    def test_metadata_policy_compares_shape_dtype_and_optional_device_only(self):
        passed = compare_oracle(
            policy("metadata"),
            [target()],
            snapshot(tensor([9, 8], dtype="float32", device="npu")),
            snapshot(tensor([1, 2], dtype="float32", device="npu")),
        )
        device_optional = compare_oracle(
            policy("metadata"),
            [target()],
            snapshot(tensor([9, 8], dtype="float32", device="cuda")),
            snapshot(tensor([1, 2], dtype="float32")),
        )
        wrong_dtype = compare_oracle(
            policy("metadata"),
            [target()],
            snapshot(tensor([1, 2], dtype="float16", device="npu")),
            snapshot(tensor([1, 2], dtype="float32", device="npu")),
        )
        wrong_device = compare_oracle(
            policy("metadata"),
            [target()],
            snapshot(tensor([1, 2], dtype="float32", device="cuda")),
            snapshot(tensor([1, 2], dtype="float32", device="npu")),
        )
        exact_ignores_device = compare_oracle(
            policy("exact"),
            [target()],
            snapshot(tensor([1, 2], dtype="float32", device="cuda")),
            snapshot(tensor([1, 2], dtype="float32", device="npu")),
        )
        scalar = compare_oracle(
            policy("metadata"), [target()], snapshot(1), snapshot(1)
        )

        self.assertEqual(passed.status, "passed")
        self.assertEqual(passed.error_summary.compared_count, 1)
        self.assertEqual(device_optional.status, "passed")
        self.assertEqual(wrong_dtype.error_summary.mismatch_kind, "dtype")
        self.assertEqual(wrong_device.error_summary.mismatch_kind, "device")
        self.assertEqual(exact_ignores_device.status, "passed")
        self.assertEqual(scalar.error_summary.mismatch_kind, "structure")

    def test_metadata_policy_validation_and_snapshot_device_decoding(self):
        old = decode_invocation_snapshot(
            {
                "return": {"shape": [1], "dtype": "float32", "values": [1.0]},
                "tensors": {},
            }
        )
        new = decode_invocation_snapshot(
            {
                "return": {
                    "shape": [1],
                    "dtype": "float32",
                    "values": [1.0],
                    "device": "npu",
                },
                "tensors": {},
            }
        )

        self.assertIsNone(old.return_value.device)
        self.assertEqual(new.return_value.device, "npu")
        with self.assertRaisesRegex(ValueError, "normalized device type"):
            tensor([1], device="cuda:0")
        with self.assertRaisesRegex(ValueError, "must not define tolerances"):
            OraclePolicy("reference", "metadata.v1", "metadata", atol=0.0)
        with self.assertRaisesRegex(ValueError, "must not enable equal_nan"):
            OraclePolicy("reference", "metadata.v1", "metadata", equal_nan=True)

    def test_integer_tensor_stays_exact_under_allclose(self):
        result = compare_oracle(
            policy(),
            [target()],
            snapshot(tensor([1, 3], dtype="int32")),
            snapshot(tensor([1, 2], dtype="int32")),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_summary.mismatch_count, 1)
        self.assertEqual(result.error_summary.compared_count, 2)

    def test_nan_policy_and_infinity_do_not_emit_nonfinite_metrics(self):
        unequal_nan = compare_oracle(
            policy(equal_nan=False), [target()], snapshot(tensor([float("nan")])), snapshot(tensor([float("nan")]))
        )
        equal_nan = compare_oracle(
            policy(equal_nan=True), [target()], snapshot(tensor([float("nan")])), snapshot(tensor([float("nan")]))
        )
        infinite_mismatch = compare_oracle(
            policy(), [target()], snapshot(tensor([float("inf")])), snapshot(tensor([1.0]))
        )

        self.assertEqual(unequal_nan.error_summary.mismatch_kind, "nan")
        self.assertEqual(equal_nan.status, "passed")
        self.assertEqual(infinite_mismatch.status, "failed")
        self.assertIsNone(infinite_mismatch.error_summary.max_abs_error)

    def test_candidate_selector_failure_is_mismatch_reference_failure_is_oracle_error(self):
        selected = target(candidate=ValueSelector("return", path=["missing"]), reference=ValueSelector("return", path=["missing"]))
        candidate_failure = compare_oracle(
            policy(), [selected], snapshot({}), snapshot({"missing": tensor([1.0])})
        )
        reference_failure = compare_oracle(
            policy(), [selected], snapshot({"missing": tensor([1.0])}), snapshot({})
        )

        self.assertEqual(candidate_failure.status, "failed")
        self.assertEqual(candidate_failure.error_summary.mismatch_kind, "structure")
        self.assertEqual(reference_failure.status, "oracle_error")
        self.assertIsNone(reference_failure.error_summary)

    def test_last_axis_tensor_slice_compares_each_row_and_roundtrips(self):
        selector = ValueSelector(
            "return", path=[{"tensor_slice": [1, 2, 4]}]
        )
        restored = ValueSelector.from_dict(selector.to_dict())
        result = compare_oracle(
            policy("exact"),
            [target(candidate=restored)],
            snapshot(tensor(range(10), shape=(2, 5), dtype="int32")),
            snapshot(tensor([2, 3, 7, 8], shape=(2, 2), dtype="int32")),
        )
        open_ended = compare_oracle(
            policy("exact"),
            [
                target(
                    candidate=ValueSelector(
                        "return", path=[{"tensor_slice": [1, 3, None]}]
                    )
                )
            ],
            snapshot(tensor(range(10), shape=(2, 5), dtype="int32")),
            snapshot(tensor([3, 4, 8, 9], shape=(2, 2), dtype="int32")),
        )

        self.assertEqual(restored, selector)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.error_summary.compared_count, 4)
        self.assertEqual(open_ended.status, "passed")

    def test_tensor_slice_rejects_invalid_specs_non_last_axis_and_bounds(self):
        invalid_specs = (
            {"tensor_slice": [-1, 0, 1]},
            {"tensor_slice": [1, 3, 2]},
            {"tensor_slice": [1, 0, 1, 2]},
            {"tensor_slice": [True, 0, 1]},
        )
        for spec in invalid_specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                ValueSelector("return", path=[spec])

        actual = snapshot(tensor(range(10), shape=(2, 5), dtype="int32"))
        expected = snapshot(tensor([0, 1], shape=(2,), dtype="int32"))
        non_last = compare_oracle(
            policy("exact"),
            [target(candidate=ValueSelector("return", path=[{"tensor_slice": [0, 0, 1]}]))],
            actual,
            expected,
        )
        out_of_bounds = compare_oracle(
            policy("exact"),
            [target(candidate=ValueSelector("return", path=[{"tensor_slice": [1, 4, 6]}]))],
            actual,
            expected,
        )

        self.assertEqual(non_last.error_summary.mismatch_kind, "structure")
        self.assertIn("last axis", non_last.error_summary.first_mismatch)
        self.assertEqual(out_of_bounds.error_summary.mismatch_kind, "structure")
        self.assertIn("exceeds", out_of_bounds.error_summary.first_mismatch)

    def test_flat_tensor_slice_roundtrips_and_rejects_invalid_bounds(self):
        selector = ValueSelector(
            "return", path=[{"tensor_flat_slice": [3, 7]}]
        )
        result = compare_oracle(
            policy("exact"),
            [target(candidate=ValueSelector.from_dict(selector.to_dict()))],
            snapshot(tensor(range(12), shape=(3, 4), dtype="uint8")),
            snapshot(tensor([3, 4, 5, 6], shape=(4,), dtype="uint8")),
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.error_summary.compared_count, 4)
        for spec in ([-1, 1], [2, 1], [0, None], [0, 1, 2], [True, 1]):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                ValueSelector("return", path=[{"tensor_flat_slice": spec}])
        out_of_bounds = compare_oracle(
            policy("exact"),
            [
                target(
                    candidate=ValueSelector(
                        "return", path=[{"tensor_flat_slice": [11, 13]}]
                    )
                )
            ],
            snapshot(tensor(range(12), shape=(3, 4), dtype="uint8")),
            snapshot(tensor([11, 12], shape=(2,), dtype="uint8")),
        )
        self.assertEqual(out_of_bounds.error_summary.mismatch_kind, "structure")
        self.assertIn("tensor numel", out_of_bounds.error_summary.first_mismatch)

    def test_first_mismatch_is_bounded_and_side_effect_is_compared(self):
        long_name = "state-" + "x" * 1000
        side_effect = target(
            long_name,
            "side_effect",
            candidate=ValueSelector("tensor", tensor_name="state"),
            reference=ValueSelector("tensor", tensor_name="state"),
        )
        result = compare_oracle(
            policy("exact"), [side_effect], snapshot(state=tensor([2.0])), snapshot(state=tensor([1.0]))
        )

        self.assertEqual(result.status, "failed")
        self.assertLessEqual(len(result.error_summary.first_mismatch.encode()), 512)
        self.assertIn("side_effect", result.error_summary.first_mismatch)

    def test_snapshot_contract_rejects_bad_numel_and_tensor_mapping(self):
        with self.assertRaisesRegex(ValueError, "numel"):
            TensorSnapshot((2,), "float32", (1.0,))
        with self.assertRaisesRegex(ValueError, "numeric or boolean"):
            TensorSnapshot((1,), "float32", ("not-a-number",))
        with self.assertRaisesRegex(ValueError, "TensorSnapshot"):
            InvocationSnapshot(None, {"x": object()})

    def test_empty_tensor_and_comparison_result_contract(self):
        empty = tensor((), shape=(0,))
        result = compare_oracle(policy(), [target()], snapshot(empty), snapshot(empty))

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.error_summary.compared_count, 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            OracleComparison("passed", -1)
        with self.assertRaisesRegex(ValueError, "cannot report mismatches"):
            OracleComparison(
                "passed",
                1,
                CorrectnessErrorSummary(mismatch_count=1, compared_count=1),
            )


if __name__ == "__main__":
    unittest.main()
