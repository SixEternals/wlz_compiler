"""Focused tests for the evidence-only public case catalog."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from wlz_optimizer.case_catalog import (
    ACT_QUANT_CASE_PATH,
    CHUNK_CUMSUM_CASE_PATH,
    COUNT_EXPERT_CASE_PATH,
    IDENTITY_SIGNATURE_KIND,
    PER_GROUP_TRANSPOSE_CASE_PATH,
    PublicCaseCatalog,
    PublicCaseRecord,
    QUANTIZE_K_CACHE_CASE_PATH,
    RMS_NORM_CASE_PATH,
    SET_K_AND_S_CASE_PATH,
    SELECTIVE_SCAN_CASE_PATH,
    build_public_case_catalog,
    materialize_act_quant_public_cases,
    materialize_chunk_cumsum_public_case,
    materialize_count_expert_basic_public_case,
    materialize_per_group_transpose_public_case,
    materialize_quantize_k_cache_public_cases,
    materialize_rms_norm_public_case,
    materialize_set_k_and_s_public_case,
    materialize_selective_scan_public_case,
)
from wlz_optimizer.schemas import (
    InputContract,
    TensorAliasGroup,
    TensorInitializer,
    TensorInputContract,
)
from wlz_optimizer.state_reset import plan_state_reset


class PublicCaseCatalogTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "datasets"
        for operator, body in (
            ("op_b", "def test_b():\n    assert True\n"),
            ("op_a", "if __name__ == '__main__':\n    print('ok')\n"),
        ):
            op_dir = dataset / operator
            op_dir.mkdir(parents=True)
            (op_dir / f"test_{operator}_1.py").write_text(body, encoding="utf-8")
        return dataset

    def test_build_is_deterministic_and_source_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp))
            first = build_public_case_catalog(dataset)
            second = build_public_case_catalog(dataset)

            self.assertEqual(first, second)
            self.assertEqual([item.operator_name for item in first.records], ["op_a", "op_b"])
            record = first.records[0]
            source = (dataset / record.source_path).read_bytes()
            self.assertEqual(record.source_sha256, hashlib.sha256(source).hexdigest())
            self.assertEqual(record.identity_signature_kind, IDENTITY_SIGNATURE_KIND)
            self.assertEqual(len(record.identity_signature), 64)

    def test_roundtrip_preserves_unmaterialized_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = build_public_case_catalog(self._dataset(Path(tmp)))
            restored = PublicCaseCatalog.from_dict(
                json.loads(json.dumps(catalog.to_dict()))
            )

            self.assertEqual(restored, catalog)
            self.assertTrue(
                all(record.materialization_status == "unmaterialized" for record in catalog.records)
            )
            self.assertTrue(
                all(not record.evaluation_case_signatures for record in catalog.records)
            )

    def test_multiple_signatures_flatten_and_legacy_json_remains_readable(self) -> None:
        base = {
            "operator_name": "op_a",
            "case_id": "op_a/test_op_a.py",
            "source_path": "op_a/test_op_a.py",
            "source_sha256": "a" * 64,
            "identity_signature": "b" * 64,
            "materialization_status": "materialized_explicit_manifest",
            "materialization_reasons": (),
        }
        record = PublicCaseRecord(
            **base,
            evaluation_case_signatures=("d" * 64, "c" * 64),
        )
        catalog = PublicCaseCatalog(dataset_root="/tmp/datasets", records=(record,))

        self.assertEqual(
            catalog.expected_case_signatures_for("op_a"),
            ("c" * 64, "d" * 64),
        )
        restored = PublicCaseCatalog.from_dict(
            json.loads(json.dumps(catalog.to_dict()))
        )
        self.assertEqual(restored, catalog)
        self.assertEqual(restored.schema_version, 2)

        legacy = PublicCaseRecord.from_dict(
            {
                **base,
                "materialization_reasons": [],
                "evaluation_case_signature": "e" * 64,
            }
        )
        self.assertEqual(legacy.evaluation_case_signatures, ("e" * 64,))
        self.assertEqual(legacy.evaluation_case_signature, "e" * 64)

    def test_identity_signature_changes_with_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp))
            before = build_public_case_catalog(dataset).records[0].identity_signature
            path = dataset / "op_a" / "test_op_a_1.py"
            path.write_text("if __name__ == '__main__':\n    print('changed')\n", encoding="utf-8")
            after = build_public_case_catalog(dataset).records[0].identity_signature

            self.assertNotEqual(before, after)

    def test_unmaterialized_records_cannot_feed_correctness_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog = build_public_case_catalog(self._dataset(Path(tmp)))

            with self.assertRaisesRegex(ValueError, "signatures unavailable"):
                catalog.expected_case_signatures_for("op_a")
            with self.assertRaises(KeyError):
                catalog.expected_case_signatures_for("missing")

    def test_invalid_python_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp))
            (dataset / "op_a" / "test_op_a_1.py").write_text("def broken(:\n", encoding="utf-8")

            with self.assertRaises(SyntaxError):
                build_public_case_catalog(dataset)

    def test_reviewed_rms_norm_script_materializes_complete_case(self) -> None:
        dataset = Path("work/official_triton_agent/datasets")
        case = materialize_rms_norm_public_case(dataset)
        catalog = build_public_case_catalog(dataset)

        self.assertEqual(case.seed, 0)
        self.assertEqual(case.inputs.tensors["input_tensor"].shape, [32, 128, 512])
        self.assertEqual(case.inputs.tensors["input_tensor"].layout, "contiguous")
        self.assertFalse(case.inputs.tensors["input_tensor"].mutable)
        self.assertEqual(case.inputs.tensors["input_tensor"].initializer.kind, "randn")
        self.assertEqual(case.inputs.tensors["weight_tensor"].initializer.kind, "randn")
        self.assertEqual(case.execution.entrypoint, "rms_norm")
        self.assertEqual(
            [argument.parameter_name for argument in case.execution.arguments],
            ["input", "weight", "eps"],
        )
        self.assertEqual(case.oracle_targets[0].candidate.source, "return")
        self.assertEqual(case.inputs.alias_groups, [])
        self.assertEqual(case.oracle_policy.rtol, 1e-4)
        self.assertEqual(case.oracle_policy.atol, 1e-5)
        self.assertEqual(
            case.signature(),
            "a8eb9bf2d7f45f4037f39b3cb49ea096ad7bb93f42d3d6e4eb54b718911d7bca",
        )
        self.assertEqual(
            catalog.expected_case_signatures_for("_rms_norm_kernel"),
            (case.signature(),),
        )

    def test_act_quant_script_materializes_two_distinct_invocations(self) -> None:
        dataset = Path("work/official_triton_agent/datasets")
        default_case, round_case = materialize_act_quant_public_cases(dataset)
        catalog = build_public_case_catalog(dataset)

        self.assertEqual(default_case.seed, 0)
        self.assertEqual(default_case.inputs.tensors["x"].shape, [2, 4, 256])
        self.assertEqual(default_case.inputs.tensors["x"].dtype, "torch.float32")
        self.assertEqual(default_case.inputs.tensors["x"].layout, "contiguous")
        self.assertEqual(default_case.inputs.tensors["x"].initializer.kind, "randn")
        self.assertEqual(default_case.inputs.scalars, {"block_size": 128})
        self.assertEqual(
            [item.parameter_name for item in default_case.execution.arguments],
            ["x", "block_size"],
        )
        self.assertEqual(default_case.oracle_policy.kind, "allclose")
        self.assertEqual(default_case.oracle_policy.rtol, 1e-2)
        self.assertEqual(default_case.oracle_policy.atol, 1e-2)

        self.assertEqual(
            round_case.inputs.scalars,
            {"block_size": 128, "scale_fmt": "round"},
        )
        self.assertEqual(
            [item.parameter_name for item in round_case.execution.arguments],
            ["x", "block_size", "scale_fmt"],
        )
        self.assertEqual(round_case.oracle_policy.kind, "shape")
        self.assertIsNone(round_case.oracle_policy.rtol)
        self.assertEqual(
            [target.candidate.path for target in round_case.oracle_targets],
            [[0], [1]],
        )
        self.assertTrue(default_case.source_ref.endswith("#default"))
        self.assertTrue(round_case.source_ref.endswith("#round"))

        expected = tuple(sorted((default_case.signature(), round_case.signature())))
        self.assertNotEqual(default_case.signature(), round_case.signature())
        self.assertEqual(
            default_case.input_signature(),
            "7f417c15cf1005e211c8450fcfff435ad09176e7bfede6ddfa2f5ac60fd48444",
        )
        self.assertEqual(
            default_case.signature(),
            "44142609123e230d2e042943938f219dc86ea18b0b30cc462eded126964b4fd5",
        )
        self.assertEqual(
            round_case.input_signature(),
            "f3e6d3fdd6e430ebab05f5cecd5498b21aaf743c8fa22fbbc07bbde0a2d6ad37",
        )
        self.assertEqual(
            round_case.signature(),
            "43c60fd0c717e7150baaf0bad738f8b41e0a6497116e3dfd0b7f2ab5edf12cd6",
        )
        self.assertEqual(
            catalog.expected_case_signatures_for("_act_quant_kernel"), expected
        )
        record = catalog.records_for("_act_quant_kernel")[0]
        self.assertEqual(record.case_id, ACT_QUANT_CASE_PATH)
        self.assertEqual(record.evaluation_case_signatures, expected)
        self.assertIsNone(record.evaluation_case_signature)

    def test_act_quant_materialization_rejects_source_change(self) -> None:
        source_root = Path("work/official_triton_agent/datasets")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "datasets"
            target = dataset / ACT_QUANT_CASE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(source_root / ACT_QUANT_CASE_PATH, target)
            materialize_act_quant_public_cases(dataset)
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize_act_quant_public_cases(dataset)

    def test_quantize_k_cache_materializes_rope_and_metadata_cases(self) -> None:
        dataset = Path("work/official_triton_agent/datasets")
        rope_case, metadata_case = materialize_quantize_k_cache_public_cases(dataset)
        catalog = build_public_case_catalog(dataset)

        self.assertEqual(rope_case.seed, 0)
        self.assertEqual(rope_case.inputs.tensors["k_nope"].shape, [4, 512])
        self.assertEqual(rope_case.inputs.tensors["k_rope"].shape, [4, 64])
        self.assertEqual(
            rope_case.inputs.tensors["k_nope"].dtype, "torch.bfloat16"
        )
        self.assertEqual(rope_case.inputs.scalars, {"group_size": 128})
        self.assertEqual(
            [item.parameter_name for item in rope_case.execution.arguments],
            ["k_nope", "k_rope", "group_size"],
        )
        self.assertEqual(rope_case.oracle_policy.kind, "allclose")
        self.assertEqual(rope_case.oracle_policy.rtol, 1e-5)
        self.assertEqual(rope_case.oracle_policy.atol, 1e-3)
        self.assertEqual(
            rope_case.oracle_targets[0].candidate.path,
            [{"tensor_slice": [1, 528, None]}],
        )
        self.assertEqual(metadata_case.oracle_policy.kind, "metadata")
        self.assertTrue(metadata_case.source_ref.endswith("#metadata"))

        expected = tuple(sorted((rope_case.signature(), metadata_case.signature())))
        self.assertNotEqual(rope_case.signature(), metadata_case.signature())
        self.assertEqual(rope_case.input_signature(), metadata_case.input_signature())
        self.assertEqual(
            rope_case.input_signature(),
            "71665b7fa3a499546ca612bfab8e4588c1f38cad64db12bb594cd3500beae655",
        )
        self.assertEqual(
            rope_case.signature(),
            "7f81693cf460544472b2c03d33d557d54bd2672cc1e87891656a03508c68d9e4",
        )
        self.assertEqual(
            metadata_case.signature(),
            "461360b14f555fac3d847162fb100dc2e8ca46fc92c3ae3ab88ec3f576f79f5d",
        )
        self.assertEqual(
            catalog.expected_case_signatures_for("_quantize_k_cache_fast_kernel"),
            expected,
        )
        record = catalog.records_for("_quantize_k_cache_fast_kernel")[0]
        self.assertEqual(record.case_id, QUANTIZE_K_CACHE_CASE_PATH)
        self.assertEqual(record.evaluation_case_signatures, expected)

    def test_quantize_k_cache_materialization_rejects_source_change(self) -> None:
        source_root = Path("work/official_triton_agent/datasets")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "datasets"
            target = dataset / QUANTIZE_K_CACHE_CASE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(source_root / QUANTIZE_K_CACHE_CASE_PATH, target)
            materialize_quantize_k_cache_public_cases(dataset)
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize_quantize_k_cache_public_cases(dataset)

    def test_count_expert_materializes_only_basic_no_map_case(self) -> None:
        dataset = Path("work/official_triton_agent/datasets")
        case = materialize_count_expert_basic_public_case(dataset)
        catalog = build_public_case_catalog(dataset)

        self.assertEqual(case.case_id, f"{COUNT_EXPERT_CASE_PATH}::basic-no-map")
        self.assertEqual(case.inputs.tensors["topk_ids"].shape, [8])
        self.assertEqual(case.inputs.tensors["topk_ids"].dtype, "torch.int32")
        self.assertEqual(
            case.inputs.tensors["topk_ids"].initializer.parameters["values"],
            [0, 1, 2, 0, 1, 3, 2, 0],
        )
        self.assertEqual(
            case.inputs.scalars,
            {"num_local_experts": 4, "expert_map": None},
        )
        self.assertEqual(case.execution.entrypoint, "count_expert_num_tokens")
        self.assertEqual(
            [argument.parameter_name for argument in case.execution.arguments],
            ["topk_ids", "num_local_experts", "expert_map"],
        )
        self.assertEqual(case.oracle_policy.kind, "exact")
        self.assertEqual(case.oracle_targets[0].target_name, "expert_num_tokens")
        self.assertTrue(case.source_ref.endswith("#basic-no-map"))
        self.assertEqual(
            case.input_signature(),
            "da2505dd0f6be715a209d2f6b763c19ea5e73bb029aab72ec08caa507ec79176",
        )
        self.assertEqual(
            case.signature(),
            "8e20825a7bb44ad0b45af82fe8d790b59a7d558a520b3dcbd14b9bd69ac279ea",
        )
        record = catalog.records_for("_count_expert_num_tokens")[0]
        self.assertEqual(record.case_id, COUNT_EXPERT_CASE_PATH)
        self.assertEqual(record.evaluation_case_signatures, (case.signature(),))
        self.assertEqual(
            record.materialization_status,
            "materialized_explicit_manifest",
        )

    def test_count_expert_materialization_rejects_source_change(self) -> None:
        source_root = Path("work/official_triton_agent/datasets")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "datasets"
            target = dataset / COUNT_EXPERT_CASE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(source_root / COUNT_EXPERT_CASE_PATH, target)
            materialize_count_expert_basic_public_case(dataset)
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize_count_expert_basic_public_case(dataset)
            record = build_public_case_catalog(dataset).records_for(
                "_count_expert_num_tokens"
            )[0]
            self.assertEqual(record.evaluation_case_signatures, ())
            self.assertEqual(record.materialization_reasons, ("source_sha256_mismatch",))

    def test_set_k_and_s_materializes_first_token_raw_byte_side_effects(self) -> None:
        dataset = Path("work/official_triton_agent/datasets")
        case = materialize_set_k_and_s_public_case(dataset)
        catalog = build_public_case_catalog(dataset)

        self.assertEqual(case.seed, 0)
        self.assertEqual(case.inputs.tensors["buf"].shape, [4, 8448])
        self.assertEqual(case.inputs.tensors["buf"].dtype, "torch.uint8")
        self.assertTrue(case.inputs.tensors["buf"].mutable)
        self.assertEqual(
            case.inputs.tensors["loc"].initializer.parameters["values"],
            [0, 64, 128],
        )
        self.assertFalse(case.inputs.tensors["loc"].mutable)
        self.assertEqual(case.inputs.tensors["index_k"].shape, [3, 128])
        self.assertEqual(case.inputs.tensors["index_k_scale"].shape, [3, 1])
        self.assertEqual(case.inputs.scalars, {"page_size": 64})
        self.assertEqual(case.execution.entrypoint, "_set_k_and_s_triton")
        self.assertEqual(
            [item.parameter_name for item in case.execution.arguments],
            ["buf", "loc", "index_k", "index_k_scale", "page_size"],
        )
        self.assertEqual(case.oracle_policy.kind, "exact")
        targets = {target.target_name: target for target in case.oracle_targets}
        self.assertEqual(set(targets), {"k_bytes", "scale_bytes"})
        self.assertEqual(
            targets["k_bytes"].candidate.path,
            [{"tensor_flat_slice": [0, 256]}],
        )
        self.assertEqual(targets["k_bytes"].reference.path, [0])
        self.assertEqual(
            targets["scale_bytes"].candidate.path,
            [{"tensor_flat_slice": [8192, 8196]}],
        )
        self.assertEqual(targets["scale_bytes"].reference.path, [1])
        self.assertTrue(
            all(target.evidence == "manifest_strengthening" for target in targets.values())
        )
        self.assertTrue(case.source_ref.endswith("#first-token-raw-byte-strengthening"))
        self.assertEqual(
            case.input_signature(),
            "666e03b5d488beb319d9849dfbb880904768878473064c4b780638297e23156b",
        )
        self.assertEqual(
            case.signature(),
            "8c4a6dac4f071a361413ffa4e59fbf8503932542003ee6acdfa137dd2158f15d",
        )
        self.assertEqual(
            catalog.expected_case_signatures_for("_set_k_and_s_triton_kernel"),
            (case.signature(),),
        )
        record = catalog.records_for("_set_k_and_s_triton_kernel")[0]
        self.assertEqual(record.case_id, SET_K_AND_S_CASE_PATH)
        self.assertEqual(record.evaluation_case_signatures, (case.signature(),))
        self.assertEqual(
            record.materialization_status,
            "materialized_explicit_manifest",
        )

    def test_set_k_and_s_materialization_rejects_source_change(self) -> None:
        source_root = Path("work/official_triton_agent/datasets")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "datasets"
            target = dataset / SET_K_AND_S_CASE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(source_root / SET_K_AND_S_CASE_PATH, target)
            materialize_set_k_and_s_public_case(dataset)
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize_set_k_and_s_public_case(dataset)
            record = build_public_case_catalog(dataset).records_for(
                "_set_k_and_s_triton_kernel"
            )[0]
            self.assertEqual(record.evaluation_case_signatures, ())
            self.assertEqual(record.materialization_reasons, ("source_sha256_mismatch",))

    def test_per_group_transpose_materializes_semantic_strengthening(self) -> None:
        dataset = Path("work/official_triton_agent/datasets")
        case = materialize_per_group_transpose_public_case(dataset)
        catalog = build_public_case_catalog(dataset)

        self.assertEqual(case.seed, 0)
        self.assertEqual(case.inputs.tensors["a"].shape, [32, 16])
        self.assertEqual(case.inputs.tensors["a"].dtype, "torch.float32")
        self.assertEqual(case.inputs.tensors["a"].layout, "contiguous")
        self.assertEqual(case.inputs.tensors["a"].initializer.kind, "randn")
        self.assertEqual(case.inputs.tensors["expert_offsets"].shape, [3])
        self.assertEqual(case.inputs.tensors["expert_offsets"].dtype, "torch.int32")
        self.assertEqual(
            case.inputs.tensors["expert_offsets"].initializer.parameters["values"],
            [0, 16, 32],
        )
        self.assertEqual(case.inputs.alias_groups, [])
        self.assertEqual(case.inputs.scalars, {"M_ALIGNMENT": 1})
        self.assertEqual(case.execution.entrypoint, "per_group_transpose")
        self.assertEqual(
            [item.parameter_name for item in case.execution.arguments],
            ["a", "expert_offsets", "M_ALIGNMENT"],
        )
        self.assertEqual(case.oracle_policy.kind, "exact")
        self.assertEqual(case.oracle_targets[0].evidence, "manifest_strengthening")
        self.assertTrue(case.source_ref.endswith("#blockwise-transpose-strengthening"))
        self.assertEqual(
            case.input_signature(),
            "b4a5d515c18a90ab3ca4091312061f841004c2b85dc314e05c4a66db9a140c3a",
        )
        self.assertEqual(
            case.signature(),
            "6e9274c24d29b015d404467a2ee2cae8b31bbee7492d15061e1752507824a188",
        )
        self.assertEqual(
            catalog.expected_case_signatures_for("_per_group_transpose"),
            (case.signature(),),
        )

    def test_per_group_transpose_materialization_rejects_source_change(self) -> None:
        source_root = Path("work/official_triton_agent/datasets")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "datasets"
            target = dataset / PER_GROUP_TRANSPOSE_CASE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(source_root / PER_GROUP_TRANSPOSE_CASE_PATH, target)
            materialize_per_group_transpose_public_case(dataset)
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize_per_group_transpose_public_case(dataset)
            record = build_public_case_catalog(dataset).records_for(
                "_per_group_transpose"
            )[0]
            self.assertEqual(record.evaluation_case_signatures, ())
            self.assertEqual(record.materialization_reasons, ("source_sha256_mismatch",))

    def test_rms_norm_materialization_rejects_source_change(self) -> None:
        source_root = Path("work/official_triton_agent/datasets")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "datasets"
            target = dataset / RMS_NORM_CASE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(source_root / RMS_NORM_CASE_PATH, target)
            materialize_rms_norm_public_case(dataset)
            target.write_text(target.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize_rms_norm_public_case(dataset)

    def test_selective_scan_materializes_mutable_state_contract(self) -> None:
        dataset = Path("work/official_triton_agent/datasets")
        case = materialize_selective_scan_public_case(dataset)
        catalog = build_public_case_catalog(dataset)

        self.assertEqual(case.seed, 0)
        self.assertEqual(case.inputs.tensors["state"].shape, [2, 4, 64, 16])
        self.assertTrue(case.inputs.tensors["state"].mutable)
        self.assertTrue(case.inputs.tensors["out"].mutable)
        self.assertEqual(case.inputs.tensors["state"].initializer.kind, "randn")
        self.assertEqual(case.inputs.tensors["out"].initializer.kind, "zeros")
        self.assertEqual(case.execution.entrypoint, "selective_state_update")
        targets = {target.target_name: target for target in case.oracle_targets}
        self.assertEqual(targets["out"].candidate.tensor_name, "out")
        self.assertEqual(targets["out"].reference.path, [0])
        self.assertEqual(targets["out"].evidence, "public_assertion")
        self.assertEqual(targets["state"].kind, "side_effect")
        self.assertEqual(targets["state"].candidate.tensor_name, "state")
        self.assertEqual(targets["state"].reference.path, [1])
        self.assertEqual(targets["state"].evidence, "manifest_strengthening")
        self.assertFalse(case.inputs.tensors["x"].mutable)
        self.assertEqual(case.inputs.alias_groups, [])
        self.assertTrue(
            case.oracle_policy.reference_id.endswith("::SelectiveScanUpdateReference")
        )
        self.assertEqual(case.oracle_policy.rtol, 1e-4)
        self.assertEqual(case.oracle_policy.atol, 1e-4)
        self.assertEqual(
            case.signature(),
            "faf039e75001e59dc9e05428b037a605758f3befaf99f237b0da332db53ac36b",
        )
        self.assertEqual(
            catalog.expected_case_signatures_for("_selective_scan_update_kernel"),
            (case.signature(),),
        )
        record = catalog.records_for("_selective_scan_update_kernel")[0]
        self.assertEqual(record.evaluation_case_signatures, (case.signature(),))

    def test_selective_scan_no_alias_resets_mutable_inputs_only(self) -> None:
        case = materialize_selective_scan_public_case(
            Path("work/official_triton_agent/datasets")
        )

        plan = plan_state_reset(case.inputs)

        self.assertEqual(
            [(group.tensor_names, group.preserve_alias) for group in plan.reset_groups],
            [(('out',), False), (('state',), False)],
        )
        self.assertNotIn("state", plan.untouched_tensors)
        self.assertNotIn("out", plan.untouched_tensors)
        self.assertIn("x", plan.untouched_tensors)

    def test_mutable_alias_restores_whole_group_and_preserves_alias(self) -> None:
        tensor = lambda mutable=False: TensorInputContract(
            shape=[4],
            dtype="torch.float32",
            layout="contiguous",
            initializer=TensorInitializer("zeros"),
            mutable=mutable,
        )
        inputs = InputContract(
            tensors={
                "state": tensor(mutable=True),
                "state_view": tensor(),
                "x": tensor(),
            },
            alias_groups=[TensorAliasGroup(["state", "state_view"])],
        )

        plan = plan_state_reset(inputs)

        self.assertEqual(len(plan.reset_groups), 1)
        self.assertEqual(plan.reset_groups[0].tensor_names, ("state", "state_view"))
        self.assertTrue(plan.reset_groups[0].preserve_alias)
        self.assertEqual(plan.untouched_tensors, ("x",))

    def test_selective_scan_materialization_rejects_source_change(self) -> None:
        source_root = Path("work/official_triton_agent/datasets")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "datasets"
            target = dataset / SELECTIVE_SCAN_CASE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(source_root / SELECTIVE_SCAN_CASE_PATH, target)
            materialize_selective_scan_public_case(dataset)
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize_selective_scan_public_case(dataset)

    def test_chunk_cumsum_materializes_complete_public_invocation(self) -> None:
        dataset = Path("work/official_triton_agent/datasets")
        case = materialize_chunk_cumsum_public_case(dataset)
        catalog = build_public_case_catalog(dataset)

        self.assertEqual(case.seed, 0)
        self.assertEqual(case.inputs.tensors["dt"].shape, [2, 16, 4])
        self.assertEqual(case.inputs.tensors["A"].shape, [4])
        self.assertEqual(case.inputs.tensors["dt_bias"].shape, [4])
        self.assertTrue(
            all(
                tensor.dtype == "torch.float32"
                and tensor.layout == "contiguous"
                and tensor.initializer.kind == "randn"
                for tensor in case.inputs.tensors.values()
            )
        )
        self.assertEqual(
            case.inputs.scalars,
            {"chunk_size": 8, "dt_softplus": True, "dt_limit": (0.0, 10.0)},
        )
        self.assertEqual(case.execution.entrypoint, "_chunk_cumsum_fwd")
        self.assertEqual(
            [argument.parameter_name for argument in case.execution.arguments],
            ["dt", "A", "chunk_size", "dt_bias", "dt_softplus", "dt_limit"],
        )
        targets = {target.target_name: target for target in case.oracle_targets}
        self.assertEqual(set(targets), {"dA_cumsum", "dt_out"})
        self.assertEqual(targets["dA_cumsum"].candidate.path, [0])
        self.assertEqual(targets["dA_cumsum"].reference.path, [0])
        self.assertEqual(targets["dt_out"].candidate.path, [1])
        self.assertEqual(targets["dt_out"].reference.path, [1])
        self.assertTrue(
            all(target.evidence == "public_assertion" for target in targets.values())
        )
        self.assertEqual(case.oracle_policy.kind, "allclose")
        self.assertEqual(case.oracle_policy.rtol, 1e-4)
        self.assertEqual(case.oracle_policy.atol, 1e-4)
        self.assertEqual(
            case.input_signature(),
            "217689af989aec7e62c9544afd7ed380ab84bae9b9586f1121d852e8fb1a4863",
        )
        self.assertEqual(
            case.signature(),
            "c91c339645ef60b98c5aecee231826ea19f1cac5986de6dd057d614ea1677616",
        )
        self.assertEqual(
            catalog.expected_case_signatures_for("_chunk_cumsum_fwd_kernel"),
            (case.signature(),),
        )
        materialized = [
            record for record in catalog.records if record.evaluation_case_signatures
        ]
        self.assertEqual(len({record.operator_name for record in materialized}), 9)
        self.assertEqual(
            sum(len(record.evaluation_case_signatures) for record in materialized),
            11,
        )

    def test_chunk_cumsum_materialization_rejects_source_change(self) -> None:
        source_root = Path("work/official_triton_agent/datasets")
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "datasets"
            target = dataset / CHUNK_CUMSUM_CASE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(source_root / CHUNK_CUMSUM_CASE_PATH, target)
            materialize_chunk_cumsum_public_case(dataset)
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                materialize_chunk_cumsum_public_case(dataset)
            record = build_public_case_catalog(dataset).records_for(
                "_chunk_cumsum_fwd_kernel"
            )[0]
            self.assertEqual(record.evaluation_case_signatures, ())
            self.assertEqual(record.materialization_reasons, ("source_sha256_mismatch",))


if __name__ == "__main__":
    unittest.main()
