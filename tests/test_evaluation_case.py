"""Focused tests for EvaluationCase and InputContract."""

import json
import unittest
from dataclasses import replace

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


def tensor(**overrides) -> TensorInputContract:
    values = {
        "shape": [32, 128],
        "dtype": "torch.float16",
        "strides": [128, 1],
        "layout": "strided",
        "initializer": TensorInitializer("randn"),
        "mutable": True,
    }
    values.update(overrides)
    return TensorInputContract(**values)


def evaluation_case(**overrides) -> EvaluationCase:
    values = {
        "op_name": "demo_op",
        "case_id": "public-case-1",
        "inputs": InputContract(
            tensors={"x": tensor(), "x_view": tensor()},
            scalars={"scale": 0.5, "causal": False, "limit": None},
            alias_groups=[TensorAliasGroup(["x", "x_view"])],
        ),
        "seed": 7,
        "execution": ExecutionBinding(
            entrypoint="demo",
            arguments=[
                ArgumentBinding("x", "tensor", "x"),
                ArgumentBinding("x_view", "tensor", "x_view"),
                ArgumentBinding("scale", "scalar", "scale"),
                ArgumentBinding("causal", "scalar", "causal"),
                ArgumentBinding("limit", "scalar", "limit"),
            ],
        ),
        "oracle_policy": OraclePolicy(
            reference_id="demo-reference",
            policy_id="fp16-v1",
            kind="allclose",
            rtol=1e-3,
            atol=1e-3,
        ),
        "oracle_targets": [
            OracleTarget(
                target_name="output",
                kind="output",
                candidate=ValueSelector("return"),
                reference=ValueSelector("return"),
                evidence="public_assertion",
            ),
            OracleTarget(
                target_name="x",
                kind="side_effect",
                candidate=ValueSelector("tensor", tensor_name="x"),
                reference=ValueSelector("return", path=[1]),
                evidence="manifest_strengthening",
            ),
        ],
        "source": "public_test",
        "source_ref": "test_demo_op_1.py:20",
    }
    values.update(overrides)
    return EvaluationCase(**values)


class EvaluationCaseTests(unittest.TestCase):
    def test_side_effect_only_case_roundtrips_with_stable_signature(self) -> None:
        baseline = evaluation_case()
        side_effect_only = evaluation_case(
            oracle_targets=[baseline.oracle_targets[1]]
        )
        restored = EvaluationCase.from_dict(
            json.loads(json.dumps(side_effect_only.to_dict()))
        )

        self.assertEqual(restored, side_effect_only)
        self.assertEqual(restored.signature(), side_effect_only.signature())
        self.assertEqual([target.kind for target in restored.oracle_targets], ["side_effect"])
        self.assertNotEqual(side_effect_only.signature(), baseline.signature())

    def test_json_roundtrip_preserves_contract(self) -> None:
        initializers = (
            TensorInitializer("randn"),
            TensorInitializer("zeros"),
            TensorInitializer("full", {"fill_value": -1.5}),
            TensorInitializer("randint", {"low": -2, "high": 7}),
        )
        for initializer in initializers:
            baseline = evaluation_case()
            tensors = dict(baseline.inputs.tensors)
            tensors["x"] = replace(tensors["x"], initializer=initializer)
            original = evaluation_case(
                inputs=replace(baseline.inputs, tensors=tensors)
            )
            restored = EvaluationCase.from_dict(json.loads(json.dumps(original.to_dict())))
            with self.subTest(kind=initializer.kind):
                self.assertEqual(restored, original)

    def test_signature_ignores_identity_provenance_and_mapping_order(self) -> None:
        first = evaluation_case()
        reordered = InputContract(
            tensors={"x_view": tensor(), "x": tensor()},
            scalars={"limit": None, "causal": False, "scale": 0.5},
            alias_groups=[TensorAliasGroup(["x_view", "x"])],
        )
        second = evaluation_case(
            case_id="generated-case-9",
            inputs=reordered,
            execution=ExecutionBinding(
                entrypoint="demo",
                arguments=[
                    ArgumentBinding("x", "tensor", "x"),
                    ArgumentBinding("x_view", "tensor", "x_view"),
                    ArgumentBinding("scale", "scalar", "scale"),
                    ArgumentBinding("causal", "scalar", "causal"),
                    ArgumentBinding("limit", "scalar", "limit"),
                ],
            ),
            source="generated_proposal",
            source_ref="planner-run-2",
        )

        self.assertEqual(first.input_signature(), second.input_signature())
        self.assertEqual(first.signature(), second.signature())
        reordered_targets = evaluation_case(
            oracle_targets=list(reversed(first.oracle_targets))
        )
        self.assertEqual(first.signature(), reordered_targets.signature())

        ordered = TensorInitializer("randint", {"low": 0, "high": 8})
        reversed_order = TensorInitializer("randint", {"high": 8, "low": 0})
        ordered_inputs = dict(first.inputs.tensors)
        ordered_inputs["x"] = replace(ordered_inputs["x"], initializer=ordered)
        reversed_inputs = dict(first.inputs.tensors)
        reversed_inputs["x"] = replace(
            reversed_inputs["x"], initializer=reversed_order
        )
        first_init = evaluation_case(
            inputs=replace(first.inputs, tensors=ordered_inputs)
        )
        second_init = evaluation_case(
            case_id="other",
            inputs=replace(first.inputs, tensors=reversed_inputs),
            source="other",
            source_ref=None,
        )
        self.assertEqual(first_init.signature(), second_init.signature())

    def test_execution_or_oracle_changes_change_signature(self) -> None:
        baseline = evaluation_case()
        base_tensor = baseline.inputs.tensors["x"]

        changes = {
            "shape": replace(base_tensor, shape=[33, 128]),
            "dtype": replace(base_tensor, dtype="torch.float32"),
            "strides": replace(base_tensor, strides=[1, 32]),
            "layout": replace(base_tensor, layout="contiguous", strides=None),
            "initializer_kind": replace(
                base_tensor, initializer=TensorInitializer("zeros")
            ),
            "initializer_fill_value": replace(
                base_tensor,
                initializer=TensorInitializer("full", {"fill_value": 1.0}),
            ),
            "initializer_low": replace(
                base_tensor,
                initializer=TensorInitializer("randint", {"low": 0, "high": 8}),
            ),
            "initializer_high": replace(
                base_tensor,
                initializer=TensorInitializer("randint", {"low": 0, "high": 9}),
            ),
        }
        for name, changed_tensor in changes.items():
            tensors = dict(baseline.inputs.tensors)
            tensors["x"] = changed_tensor
            changed = evaluation_case(inputs=replace(baseline.inputs, tensors=tensors))
            with self.subTest(field=name):
                self.assertNotEqual(baseline.input_signature(), changed.input_signature())
                self.assertNotEqual(baseline.signature(), changed.signature())

        cases = {
            "alias": evaluation_case(inputs=replace(baseline.inputs, alias_groups=[])),
            "mutable": evaluation_case(
                inputs=replace(
                    baseline.inputs,
                    tensors={
                        **baseline.inputs.tensors,
                        "x_view": replace(
                            baseline.inputs.tensors["x_view"], mutable=False
                        ),
                    },
                )
            ),
            "seed": evaluation_case(seed=8),
            "scalar": evaluation_case(
                inputs=replace(
                    baseline.inputs,
                    scalars={"scale": 1.0, "causal": False, "limit": None},
                )
            ),
        }
        for name, changed in cases.items():
            with self.subTest(field=name):
                self.assertNotEqual(baseline.input_signature(), changed.input_signature())
                self.assertNotEqual(baseline.signature(), changed.signature())

        oracle_changed = evaluation_case(
            oracle_policy=replace(baseline.oracle_policy, policy_id="fp16-v2")
        )
        self.assertEqual(baseline.input_signature(), oracle_changed.input_signature())
        self.assertNotEqual(baseline.signature(), oracle_changed.signature())

        execution_changed = evaluation_case(
            execution=replace(baseline.execution, entrypoint="demo_v2")
        )
        self.assertNotEqual(baseline.input_signature(), execution_changed.input_signature())
        target_changed = evaluation_case(
            oracle_targets=[
                replace(
                    baseline.oracle_targets[0],
                    candidate=ValueSelector("return", path=[0]),
                ),
                baseline.oracle_targets[1],
            ]
        )
        self.assertEqual(baseline.input_signature(), target_changed.input_signature())
        self.assertNotEqual(baseline.signature(), target_changed.signature())

    def test_rejects_invalid_tensor_contracts(self) -> None:
        invalid = (
            {"shape": [-1, 2]},
            {"shape": [None, 2]},
            {"dtype": ""},
            {"strides": [1]},
            {"strides": [True, 1]},
            {"strides": [-1, 1]},
            {"layout": "contiguous"},
            {"mutable": 1},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                tensor(**values)

    def test_rejects_invalid_initializer_combinations(self) -> None:
        invalid = (
            ("uniform", {}),
            ("randn", {"fill_value": 0}),
            ("zeros", {"low": 0}),
            ("full", {}),
            ("full", {"fill_value": True}),
            ("full", {"fill_value": float("inf")}),
            ("full", {"fill_value": 0, "high": 1}),
            ("randint", {"low": 0}),
            ("randint", {"low": True, "high": 2}),
            ("randint", {"low": 0, "high": 1.5}),
            ("randint", {"low": 2, "high": 2}),
            ("randint", {"low": 3, "high": 2}),
            ("randint", {"low": 0, "high": 2, "extra": 1}),
        )
        for kind, parameters in invalid:
            with self.subTest(kind=kind, parameters=parameters), self.assertRaises(ValueError):
                TensorInitializer(kind, parameters)

        with self.assertRaises(ValueError):
            tensor(initializer={"kind": "zeros", "parameters": {}})
        with self.assertRaises(ValueError):
            TensorInitializer.from_dict({"kind": "zeros"})

    def test_rejects_invalid_alias_contracts(self) -> None:
        with self.assertRaises(ValueError):
            TensorAliasGroup("xy")
        with self.assertRaises(ValueError):
            TensorAliasGroup(["x"])
        with self.assertRaisesRegex(ValueError, "unknown tensors"):
            InputContract(
                tensors={"x": tensor()},
                alias_groups=[TensorAliasGroup(["x", "missing"])],
            )
        with self.assertRaisesRegex(ValueError, "multiple alias groups"):
            InputContract(
                tensors={"x": tensor(), "y": tensor(), "z": tensor()},
                alias_groups=[TensorAliasGroup(["x", "y"]), TensorAliasGroup(["x", "z"])],
            )

    def test_rejects_invalid_execution_and_oracle_targets(self) -> None:
        baseline = evaluation_case()
        with self.assertRaisesRegex(ValueError, "unknown inputs"):
            evaluation_case(
                execution=replace(
                    baseline.execution,
                    arguments=[
                        *baseline.execution.arguments,
                        ArgumentBinding("missing", "tensor", "missing"),
                    ],
                )
            )
        with self.assertRaisesRegex(ValueError, "leaves inputs unbound"):
            evaluation_case(
                execution=replace(
                    baseline.execution,
                    arguments=baseline.execution.arguments[:-1],
                )
            )
        with self.assertRaisesRegex(ValueError, "parameter names must be unique"):
            ExecutionBinding(
                "demo",
                [
                    ArgumentBinding("x", "tensor", "x"),
                    ArgumentBinding("x", "tensor", "x_view"),
                ],
            )
        with self.assertRaisesRegex(ValueError, "unknown tensor"):
            evaluation_case(
                oracle_targets=[
                    replace(
                        baseline.oracle_targets[0],
                        candidate=ValueSelector("tensor", tensor_name="missing"),
                    )
                ]
            )
        with self.assertRaisesRegex(ValueError, "mutable candidate tensor"):
            immutable_inputs = replace(
                baseline.inputs,
                tensors={
                    **baseline.inputs.tensors,
                    "x": replace(baseline.inputs.tensors["x"], mutable=False),
                },
            )
            evaluation_case(inputs=immutable_inputs)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            evaluation_case(oracle_targets=[])

    def test_rejects_invalid_seed_policy_and_scalars(self) -> None:
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                evaluation_case(seed=seed)
        with self.assertRaises(ValueError):
            OraclePolicy("", "policy-v1", "exact")
        with self.assertRaises(ValueError):
            OraclePolicy("reference", "policy-v1", "exact", rtol=0.0)
        with self.assertRaises(ValueError):
            OraclePolicy("reference", "policy-v1", "allclose", rtol=1e-3)
        with self.assertRaises(ValueError):
            InputContract(tensors={}, scalars={"bad": float("nan")})
        with self.assertRaisesRegex(ValueError, "unsupported evaluation case schema"):
            evaluation_case(schema_version=1)

    def test_from_dict_defaults_to_current_schema_version(self) -> None:
        data = json.loads(json.dumps(evaluation_case().to_dict()))
        data.pop("schema_version")
        self.assertEqual(EvaluationCase.from_dict(data).schema_version, 2)

    def test_flat_sequence_parameter_roundtrips_with_stable_signature(self) -> None:
        baseline = evaluation_case()
        inputs = replace(
            baseline.inputs,
            scalars={"scale": 0.5, "causal": False, "limit": [0.0, 10.0]},
        )
        original = evaluation_case(inputs=inputs)
        restored = EvaluationCase.from_dict(
            json.loads(json.dumps(original.to_dict()))
        )

        self.assertEqual(original.inputs.scalars["limit"], (0.0, 10.0))
        self.assertEqual(restored, original)
        self.assertEqual(restored.input_signature(), original.input_signature())
        self.assertNotEqual(original.input_signature(), baseline.input_signature())

    def test_rejects_nested_or_nonfinite_sequence_parameters(self) -> None:
        invalid = (
            [[0.0, 10.0]],
            [(0.0, 10.0)],
            [{"min": 0.0}],
            [float("inf")],
            [float("nan")],
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                InputContract(tensors={}, scalars={"limit": value})


if __name__ == "__main__":
    unittest.main()
