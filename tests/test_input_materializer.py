"""Focused tests for backend-neutral input materialization."""

import importlib.util
import unittest
from dataclasses import replace

from wlz_optimizer.input_materializer import (
    UnsupportedInputContractError,
    clone_inputs_for_run,
    materialize_inputs,
)
from wlz_optimizer.torch_backend import TorchTensorBackend
from wlz_optimizer.schemas import (
    ArgumentBinding,
    EvaluationCase,
    ExecutionBinding,
    InputContract,
    OraclePolicy,
    OracleTarget,
    TensorAliasGroup,
    TensorInitializer,
    TensorInputContract,
    ValueSelector,
)


def tensor(initializer: TensorInitializer, *, layout: str = "contiguous"):
    return TensorInputContract(
        shape=[2, 3],
        dtype="torch.float32",
        layout=layout,
        strides=[3, 1] if layout == "strided" else None,
        initializer=initializer,
    )


def evaluation_case(seed: int = 17) -> EvaluationCase:
    inputs = InputContract(
        tensors={
            "normal": tensor(TensorInitializer("randn")),
            "zero": tensor(TensorInitializer("zeros")),
            "filled": tensor(TensorInitializer("full", {"fill_value": -1.5})),
            "integer": tensor(TensorInitializer("randint", {"low": -2, "high": 7})),
        },
        scalars={"scale": 0.25},
    )
    return EvaluationCase(
        op_name="demo",
        case_id="case-1",
        inputs=inputs,
        seed=seed,
        execution=ExecutionBinding(
            "run",
            [
                ArgumentBinding("filled", "tensor", "filled"),
                ArgumentBinding("scale", "scalar", "scale"),
                ArgumentBinding("normal", "tensor", "normal"),
                ArgumentBinding("normal_again", "tensor", "normal"),
                ArgumentBinding("integer", "tensor", "integer"),
                ArgumentBinding("zero", "tensor", "zero"),
            ],
        ),
        oracle_policy=OraclePolicy("reference", "policy-v1", "exact"),
        oracle_targets=[
            OracleTarget(
                "output",
                "output",
                ValueSelector("return"),
                ValueSelector("return"),
                "public_assertion",
            )
        ],
    )


class RecordingBackend:
    def __init__(self):
        self.calls = []
        self.current_seed = None

    def seed(self, seed):
        self.current_seed = seed
        self.calls.append(("seed", seed))

    def _record(self, kind, **kwargs):
        call = (kind, kwargs)
        self.calls.append(call)
        return {
            "seed": self.current_seed,
            "kind": kind,
            "kwargs": kwargs,
            "mutations": [],
        }

    def randn(self, **kwargs):
        return self._record("randn", **kwargs)

    def zeros(self, **kwargs):
        return self._record("zeros", **kwargs)

    def full(self, **kwargs):
        return self._record("full", **kwargs)

    def randint(self, **kwargs):
        return self._record("randint", **kwargs)

    def literal(self, **kwargs):
        return self._record("literal", **kwargs)

    def clone(self, value):
        self.calls.append(("clone", value))
        return {
            "seed": value["seed"],
            "kind": value["kind"],
            "kwargs": dict(value["kwargs"]),
            "mutations": list(value["mutations"]),
        }

    def as_strided(self, value, **kwargs):
        self.calls.append(("as_strided", kwargs))
        return {"storage": value, **kwargs}


class InputMaterializerTests(unittest.TestCase):
    def test_literal_values_are_stable_materialized_inputs(self):
        base = evaluation_case()
        literal = TensorInputContract(
            shape=[3],
            dtype="torch.int64",
            layout="contiguous",
            initializer=TensorInitializer("literal", {"values": [0, 64, 128]}),
        )
        inputs = replace(
            base.inputs,
            tensors={**base.inputs.tensors, "locations": literal},
        )
        case = replace(
            base,
            inputs=inputs,
            execution=replace(
                base.execution,
                arguments=[
                    *base.execution.arguments,
                    ArgumentBinding("locations", "tensor", "locations"),
                ],
            ),
        )
        changed = replace(
            case,
            inputs=replace(
                inputs,
                tensors={
                    **inputs.tensors,
                    "locations": replace(
                        literal,
                        initializer=TensorInitializer(
                            "literal", {"values": [0, 64, 129]}
                        ),
                    ),
                },
            ),
        )
        backend = RecordingBackend()

        materialized = materialize_inputs(case, backend)
        fresh = clone_inputs_for_run(materialized, backend)

        self.assertEqual(
            materialized.tensors["locations"]["kwargs"]["values"],
            (0, 64, 128),
        )
        self.assertEqual(fresh.tensors["locations"]["kwargs"]["values"], (0, 64, 128))
        self.assertNotEqual(case.input_signature(), changed.input_signature())
        with self.assertRaisesRegex(ValueError, "value count"):
            replace(literal, initializer=TensorInitializer("literal", {"values": [0, 64]}))
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            TensorInitializer("literal", {"values": [float("nan")]})
        with self.assertRaisesRegex(ValueError, "contiguous"):
            replace(literal, layout="strided", strides=[1])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_torch_backend_materializes_literal_dtype_and_values(self):
        value = TorchTensorBackend("cpu").literal(
            shape=(3,), dtype="torch.int64", values=(0, 64, 128)
        )

        self.assertEqual(str(value.dtype), "torch.int64")
        self.assertEqual(value.tolist(), [0, 64, 128])

    def test_dispatches_in_stable_first_occurrence_order(self):
        case = evaluation_case()
        backend = RecordingBackend()

        result = materialize_inputs(case, backend)

        self.assertEqual([call[0] for call in backend.calls], [
            "seed", "full", "randn", "randint", "zeros"
        ])
        self.assertEqual(list(result.tensors), ["filled", "normal", "integer", "zero"])
        self.assertEqual(result.scalars, {"scale": 0.25})
        self.assertEqual(result.order, (
            ("tensor", "filled"), ("scalar", "scale"), ("tensor", "normal"),
            ("tensor", "integer"), ("tensor", "zero"),
        ))
        self.assertEqual(backend.calls[1][1]["fill_value"], -1.5)
        self.assertEqual(backend.calls[3][1]["low"], -2)
        self.assertEqual(backend.calls[3][1]["high"], 7)
        self.assertTrue(all(call[1]["shape"] == (2, 3) for call in backend.calls[1:]))

    def test_seeds_once_per_materialization_and_exposes_identity(self):
        backend = RecordingBackend()
        first = materialize_inputs(evaluation_case(17), backend)
        second = materialize_inputs(evaluation_case(17), backend)
        third = materialize_inputs(evaluation_case(18), backend)

        self.assertEqual([call for call in backend.calls if call[0] == "seed"], [
            ("seed", 17), ("seed", 17), ("seed", 18)
        ])
        self.assertEqual(first.tensors, second.tensors)
        self.assertNotEqual(first.tensors, third.tensors)
        self.assertEqual(first.seed, 17)
        self.assertEqual(first.input_signature, evaluation_case(17).input_signature())

    def test_oracle_and_mapping_order_do_not_change_materialized_inputs(self):
        first_case = evaluation_case()
        reordered_case = replace(
            first_case,
            inputs=replace(
                first_case.inputs,
                tensors=dict(reversed(list(first_case.inputs.tensors.items()))),
            ),
            oracle_policy=replace(first_case.oracle_policy, policy_id="policy-v2"),
        )

        first = materialize_inputs(first_case, RecordingBackend())
        reordered = materialize_inputs(reordered_case, RecordingBackend())

        self.assertNotEqual(first_case.signature(), reordered_case.signature())
        self.assertEqual(first.input_signature, reordered.input_signature)
        self.assertEqual(first.tensors, reordered.tensors)
        self.assertEqual(first.order, reordered.order)

    def test_strided_alias_views_survive_fresh_run_cloning(self):
        base = evaluation_case()
        tensors = dict(base.inputs.tensors)
        tensors["normal"] = replace(tensors["normal"], mutable=True)
        tensors["zero"] = replace(
            tensors["normal"],
            layout="strided",
            strides=[1, 2],
            mutable=False,
        )
        inputs = replace(
            base.inputs,
            tensors=tensors,
            alias_groups=[TensorAliasGroup(["normal", "zero"])],
        )
        backend = RecordingBackend()

        pristine = materialize_inputs(replace(base, inputs=inputs), backend)
        first = clone_inputs_for_run(pristine, backend)
        second = clone_inputs_for_run(pristine, backend)

        pristine_storage = pristine.tensors["normal"]["storage"]
        first_storage = first.tensors["normal"]["storage"]
        second_storage = second.tensors["normal"]["storage"]
        self.assertIs(pristine_storage, pristine.tensors["zero"]["storage"])
        self.assertTrue(inputs.tensors["normal"].mutable)
        self.assertFalse(inputs.tensors["zero"].mutable)
        self.assertIs(first_storage, first.tensors["zero"]["storage"])
        self.assertIs(second_storage, second.tensors["zero"]["storage"])
        self.assertIsNot(first_storage, pristine_storage)
        self.assertIsNot(first_storage, second_storage)
        self.assertEqual(first.tensors["normal"]["strides"], (3, 1))
        self.assertEqual(first.tensors["zero"]["strides"], (1, 2))
        self.assertEqual(pristine_storage["kwargs"]["shape"], (6,))
        self.assertEqual(
            len([call for call in backend.calls if call[0] == "clone"]),
            2 * len(pristine.storages),
        )

        first_storage["mutations"].append("changed")
        self.assertEqual(pristine_storage["mutations"], [])
        self.assertEqual(second_storage["mutations"], [])
        self.assertEqual(first.input_signature, pristine.input_signature)
        self.assertEqual(first.order, pristine.order)

    def test_incompatible_alias_contract_fails_before_seeding(self):
        base = evaluation_case()
        inputs = replace(
            base.inputs,
            alias_groups=[TensorAliasGroup(["normal", "zero"])],
        )
        backend = RecordingBackend()

        with self.assertRaises(UnsupportedInputContractError):
            materialize_inputs(replace(base, inputs=inputs), backend)
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
