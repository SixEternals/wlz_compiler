"""Focused tests for the trusted reference registry."""

import importlib.util
import math
import struct
import unittest
from pathlib import Path

from wlz_optimizer.candidate_runner import CandidateRunResult, run_pack_seq_candidate
from wlz_optimizer.case_catalog import (
    ACT_QUANT_DEFAULT_REFERENCE_ID,
    ACT_QUANT_ROUND_REFERENCE_ID,
    CHUNK_CUMSUM_REFERENCE_ID,
    COUNT_EXPERT_REFERENCE_ID,
    PACK_SEQ_REFERENCE_ID,
    PER_GROUP_TRANSPOSE_REFERENCE_ID,
    QUANTIZE_K_CACHE_METADATA_REFERENCE_ID,
    QUANTIZE_K_CACHE_ROPE_REFERENCE_ID,
    SET_K_AND_S_REFERENCE_ID,
    materialize_act_quant_public_cases,
    materialize_chunk_cumsum_public_case,
    materialize_count_expert_basic_public_case,
    materialize_pack_seq_public_case,
    materialize_per_group_transpose_public_case,
    materialize_quantize_k_cache_public_cases,
    materialize_set_k_and_s_public_case,
)
from wlz_optimizer.correctness_coordinator import evaluate_candidate_correctness
from wlz_optimizer.correctness_oracle import (
    InvocationSnapshot,
    TensorSnapshot,
    decode_invocation_snapshot,
)
from wlz_optimizer.correctness_references import (
    ReferenceRequest,
    registered_reference_ids,
    run_case_reference,
    run_reference,
)
from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.input_materializer import MaterializedInputs
from wlz_optimizer.schemas import Candidate
from wlz_optimizer.torch_backend import TorchTensorBackend

try:
    import torch as _torch
except ImportError:
    _torch = None

_CUDA_AVAILABLE = _torch is not None and _torch.cuda.is_available()


class CorrectnessReferenceTests(unittest.TestCase):
    def request(self, reference_id: str, payload=None, timeout=2.0) -> ReferenceRequest:
        return ReferenceRequest(
            reference_id=reference_id,
            payload=payload or {},
            timeout_seconds=timeout,
            max_output_bytes=4096,
        )

    def test_registered_reference_runs_in_removed_temporary_cwd(self) -> None:
        result = run_reference(self.request("protocol.echo.v1", {"value": 7}))

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.value["payload"], {"value": 7})
        self.assertIn("wlz-correctness-worker-", result.value["working_directory"])
        self.assertFalse(Path(result.value["working_directory"]).exists())
        self.assertEqual(result.worker_result.returncode, 0)

    def test_snapshot_decoder_accepts_protocol_and_rejects_malformed_values(self) -> None:
        snapshot = decode_invocation_snapshot(
            {
                "return": {
                    "shape": [1],
                    "dtype": "float32",
                    "values": [1.0],
                },
                "tensors": {},
            }
        )

        self.assertEqual(snapshot.return_value.shape, (1,))
        with self.assertRaisesRegex(ValueError, "return and tensors"):
            decode_invocation_snapshot({"return": None})
        with self.assertRaisesRegex(ValueError, "must be lists"):
            decode_invocation_snapshot(
                {
                    "return": {"shape": (1,), "dtype": "float32", "values": [1.0]},
                    "tensors": {},
                }
            )
        with self.assertRaisesRegex(ValueError, "must omit values"):
            TensorSnapshot((1,), "shape_only", (0,))

    def test_unknown_id_and_non_json_payload_are_rejected_before_worker(self) -> None:
        with self.assertRaisesRegex(KeyError, "not registered"):
            self.request("os.system", {"command": "echo unsafe"})
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            self.request("protocol.echo.v1", {"bad": object()})
        self.assertEqual(
            registered_reference_ids(),
            (
                ACT_QUANT_DEFAULT_REFERENCE_ID,
                ACT_QUANT_ROUND_REFERENCE_ID,
                CHUNK_CUMSUM_REFERENCE_ID,
                COUNT_EXPERT_REFERENCE_ID,
                PACK_SEQ_REFERENCE_ID,
                PER_GROUP_TRANSPOSE_REFERENCE_ID,
                QUANTIZE_K_CACHE_ROPE_REFERENCE_ID,
                QUANTIZE_K_CACHE_METADATA_REFERENCE_ID,
                SET_K_AND_S_REFERENCE_ID,
                "protocol.echo.v1",
                "protocol.raise.v1",
                "protocol.sleep.v1",
            ),
        )

    def test_chunk_cumsum_materialized_bridge_is_deterministic(self) -> None:
        class TensorStub:
            dtype = "torch.float32"

            def __init__(self, shape, values):
                self.shape = shape
                self.values = values

            def detach(self):
                return self

            cpu = detach
            contiguous = detach

            def reshape(self, _size):
                return self

            def tolist(self):
                return self.values

        case = materialize_chunk_cumsum_public_case(
            Path("work/official_triton_agent/datasets")
        )
        inputs = MaterializedInputs(
            tensors={
                "dt": TensorStub((2, 16, 4), [0.0] * 128),
                "A": TensorStub((4,), [1.0, 2.0, 3.0, 4.0]),
                "dt_bias": TensorStub((4,), [0.0] * 4),
            },
            scalars=dict(case.inputs.scalars),
            order=(),
            seed=case.seed,
            input_signature=case.input_signature(),
            storages={},
            tensor_storage_keys={},
            view_specs={},
        )

        first = run_case_reference(case, inputs)
        second = run_case_reference(case, inputs)

        self.assertEqual(first.status, "completed")
        self.assertEqual(first.value, second.value)
        d_a_cumsum, dt_out = decode_invocation_snapshot(first.value).return_value
        self.assertEqual(d_a_cumsum.shape, (2, 4, 2, 8))
        self.assertEqual(dt_out.shape, (2, 4, 2, 8))
        self.assertEqual(d_a_cumsum.dtype, "torch.float32")
        self.assertEqual(len(d_a_cumsum.values), 128)
        self.assertAlmostEqual(dt_out.values[0], math.log(2.0))
        self.assertAlmostEqual(d_a_cumsum.values[7], 8 * math.log(2.0))
        self.assertAlmostEqual(d_a_cumsum.values[23], 16 * math.log(2.0))

    def test_chunk_cumsum_reference_rejects_wrong_public_shape(self) -> None:
        def tensor(shape, values):
            return {"shape": shape, "dtype": "torch.float32", "values": values}

        result = run_reference(
            ReferenceRequest(
                CHUNK_CUMSUM_REFERENCE_ID,
                {
                    "dt": tensor([2, 8, 4], [0.0] * 64),
                    "A": tensor([4], [1.0] * 4),
                    "dt_bias": tensor([4], [0.0] * 4),
                    "chunk_size": 8,
                    "dt_softplus": True,
                    "dt_limit": [0.0, 10.0],
                },
                timeout_seconds=2.0,
                max_output_bytes=4096,
            )
        )

        self.assertEqual(result.status, "reference_error")
        self.assertIn("dt payload is invalid", result.error_message)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_act_quant_default_reference_returns_coordinator_snapshot(self) -> None:
        values = [0.0] * 128 + [1.0] * (2 * 4 * 256 - 128)
        result = run_reference(
            ReferenceRequest(
                reference_id=ACT_QUANT_DEFAULT_REFERENCE_ID,
                payload={
                    "x": {
                        "shape": [2, 4, 256],
                        "dtype": "torch.float32",
                        "values": values,
                    },
                    "block_size": 128,
                },
                timeout_seconds=5.0,
                max_output_bytes=128 * 1024,
            )
        )

        self.assertEqual(result.status, "completed")
        encoded_y, encoded_scale = result.value["return"]
        snapshot = InvocationSnapshot(
            return_value=(
                TensorSnapshot(
                    tuple(encoded_y["shape"]),
                    encoded_y["dtype"],
                    tuple(encoded_y["values"]),
                ),
                TensorSnapshot(
                    tuple(encoded_scale["shape"]),
                    encoded_scale["dtype"],
                    tuple(encoded_scale["values"]),
                ),
            ),
            tensors={},
        )
        y, scale = snapshot.return_value
        self.assertEqual(y.shape, (2, 4, 256))
        self.assertEqual(y.dtype, "torch.float16")
        self.assertTrue(all(value == 0.0 for value in y.values[:128]))
        self.assertTrue(all(value == 448.0 for value in y.values[128:]))
        self.assertEqual(scale.shape, (2, 4, 2))
        self.assertEqual(scale.dtype, "torch.float32")
        self.assertAlmostEqual(scale.values[0], 1e-4 / 448.0, places=10)
        self.assertAlmostEqual(scale.values[1], 1.0 / 448.0, places=7)

    def test_act_quant_reference_rejects_non_public_shape(self) -> None:
        result = run_reference(
            self.request(
                ACT_QUANT_DEFAULT_REFERENCE_ID,
                {
                    "x": {"shape": [1, 256], "dtype": "torch.float32", "values": []},
                    "block_size": 128,
                },
            )
        )

        self.assertEqual(result.status, "reference_error")
        self.assertEqual(result.error_type, "ValueError")
        self.assertIn("public shape", result.error_message)

    def test_act_quant_round_reference_returns_shapes_without_torch(self) -> None:
        result = run_reference(
            self.request(
                ACT_QUANT_ROUND_REFERENCE_ID,
                {
                    "x_shape": [2, 4, 256],
                    "block_size": 128,
                    "scale_fmt": "round",
                },
            )
        )

        self.assertEqual(result.status, "completed")
        snapshot = decode_invocation_snapshot(result.value)
        y, scale = snapshot.return_value
        self.assertEqual(y.shape, (2, 4, 256))
        self.assertEqual(scale.shape, (2, 4, 2))
        self.assertEqual(y.dtype, "shape_only")
        self.assertEqual(y.values, ())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_materialized_bridge_passes_coordinator_with_fresh_inputs(self) -> None:
        case = materialize_act_quant_public_cases(
            Path("work/official_triton_agent/datasets")
        )[0]
        code = "def act_quant(x, block_size=128): return x"
        candidate = Candidate(
            id="act-quant-bridge-smoke",
            op_name="_act_quant_kernel",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=[],
            generation=0,
            mutation_kind="bridge-smoke",
            model_used=None,
            prompt_id=None,
            status="generated",
            score=None,
        )
        data_ptrs = {}

        def reference_runner(item_case, inputs):
            data_ptrs["reference"] = inputs.tensors["x"].data_ptr()
            return run_case_reference(item_case, inputs)

        def candidate_runner(item, item_case, inputs):
            data_ptrs["candidate"] = inputs.tensors["x"].data_ptr()
            result = run_case_reference(item_case, inputs)
            return CandidateRunResult(
                item.id, "completed", "runtime", value=result.value
            )

        run = evaluate_candidate_correctness(
            candidate,
            [case],
            TorchTensorBackend("cpu"),
            reference_runner,
            candidate_runner,
            decode_invocation_snapshot,
        )

        self.assertNotEqual(data_ptrs["reference"], data_ptrs["candidate"])
        self.assertEqual(run.results[0].oracle_status, "passed")
        self.assertEqual(run.results[0].error_summary.compared_count, 2064)
        self.assertTrue(run.decision.eligible_for_performance)

    def test_quantize_k_cache_references_return_public_rope_and_metadata(self) -> None:
        rope_values = [float(index) for index in range(4 * 64)]
        rope = run_reference(
            ReferenceRequest(
                reference_id=QUANTIZE_K_CACHE_ROPE_REFERENCE_ID,
                payload={
                    "k_rope": {
                        "shape": [4, 64],
                        "dtype": "torch.bfloat16",
                        "values": rope_values,
                    }
                },
                timeout_seconds=2.0,
                max_output_bytes=128 * 1024,
            )
        )
        metadata = run_reference(
            ReferenceRequest(
                reference_id=QUANTIZE_K_CACHE_METADATA_REFERENCE_ID,
                payload={
                    "k_nope_shape": [4, 512],
                    "k_rope_shape": [4, 64],
                    "group_size": 128,
                },
                timeout_seconds=2.0,
                max_output_bytes=128 * 1024,
            )
        )

        self.assertEqual(rope.status, "completed")
        self.assertEqual(
            decode_invocation_snapshot(rope.value).return_value.values,
            tuple(rope_values),
        )
        self.assertEqual(metadata.status, "completed")
        output = decode_invocation_snapshot(metadata.value).return_value
        self.assertEqual(output.shape, (4, 592))
        self.assertEqual(output.dtype, "torch.bfloat16")
        self.assertIsNone(output.device)
        self.assertEqual(len(output.values), 4 * 592)
        self.assertFalse(metadata.worker_result.stdout.truncated)
        self.assertLess(metadata.worker_result.stdout.total_bytes, 128 * 1024)

    def test_count_expert_reference_is_exact_and_rejects_case_drift(self) -> None:
        payload = {
            "topk_ids": {
                "shape": [8],
                "dtype": "torch.int32",
                "values": [0, 1, 2, 0, 1, 3, 2, 0],
            },
            "num_local_experts": 4,
            "expert_map": None,
        }
        result = run_reference(self.request(COUNT_EXPERT_REFERENCE_ID, payload))
        drifted = run_reference(
            self.request(
                COUNT_EXPERT_REFERENCE_ID,
                {
                    **payload,
                    "topk_ids": {**payload["topk_ids"], "values": [0] * 8},
                },
            )
        )

        self.assertEqual(result.status, "completed")
        output = decode_invocation_snapshot(result.value).return_value
        self.assertEqual(output.shape, (4,))
        self.assertEqual(output.dtype, "torch.int32")
        self.assertEqual(output.values, (3, 2, 2, 1))
        self.assertEqual(drifted.status, "reference_error")

        class TensorStub:
            shape = (8,)
            dtype = "torch.int32"

            def detach(self):
                return self

            cpu = contiguous = detach

            def reshape(self, *shape):
                return self

            def tolist(self):
                return payload["topk_ids"]["values"]

        case = materialize_count_expert_basic_public_case(
            Path("work/official_triton_agent/datasets")
        )
        inputs = MaterializedInputs(
            tensors={"topk_ids": TensorStub()},
            scalars={"num_local_experts": 4, "expert_map": None},
            order=(
                ("tensor", "topk_ids"),
                ("scalar", "num_local_experts"),
                ("scalar", "expert_map"),
            ),
            seed=0,
            input_signature=case.input_signature(),
            storages={},
            tensor_storage_keys={},
            view_specs={},
        )
        bridged = run_case_reference(case, inputs)
        self.assertEqual(bridged.status, "completed")
        self.assertEqual(
            decode_invocation_snapshot(bridged.value).return_value.values,
            (3, 2, 2, 1),
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_quantize_k_cache_bridges_pass_with_candidate_test_double(self) -> None:
        cases = materialize_quantize_k_cache_public_cases(
            Path("work/official_triton_agent/datasets")
        )
        code = "def _quantize_k_cache_fast(k_nope, k_rope, group_size): return k_rope"
        candidate = Candidate(
            id="quantize-k-cache-bridge-smoke",
            op_name="_quantize_k_cache_fast_kernel",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=[],
            generation=0,
            mutation_kind="bridge-smoke",
            model_used=None,
            prompt_id=None,
            status="generated",
            score=None,
        )

        def candidate_runner(item, item_case, inputs):
            rope = inputs.tensors["k_rope"].detach().cpu().contiguous()
            values = []
            for row in rope.tolist():
                values.extend([0.0] * 528)
                values.extend(row)
            return CandidateRunResult(
                item.id,
                "completed",
                "runtime",
                value={
                    "return": {
                        "shape": [4, 592],
                        "dtype": "torch.bfloat16",
                        "values": values,
                        "device": "npu",
                    },
                    "tensors": {},
                },
            )

        run = evaluate_candidate_correctness(
            candidate,
            cases,
            TorchTensorBackend("cpu"),
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )

        self.assertEqual(
            [result.oracle_status for result in run.results],
            ["passed", "passed"],
        )
        self.assertEqual(run.results[0].error_summary.compared_count, 256)
        self.assertEqual(run.results[1].error_summary.compared_count, 1)
        self.assertTrue(run.decision.eligible_for_performance)

    def test_per_group_transpose_reference_uses_blockwise_flattening(self) -> None:
        payload = {
            "a": {
                "shape": [32, 16],
                "dtype": "torch.float32",
                "values": [float(index) for index in range(32 * 16)],
            },
            "expert_offsets": {
                "shape": [3],
                "dtype": "torch.int32",
                "values": [0, 16, 32],
            },
            "M_ALIGNMENT": 1,
        }
        result = run_reference(
            self.request(PER_GROUP_TRANSPOSE_REFERENCE_ID, payload)
        )

        invalid_payloads = (
            {**payload, "M_ALIGNMENT": True},
            {
                **payload,
                "a": {**payload["a"], "shape": [32.0, 16]},
            },
            {
                **payload,
                "expert_offsets": {
                    **payload["expert_offsets"],
                    "shape": [3.0],
                },
            },
            {
                **payload,
                "expert_offsets": {
                    **payload["expert_offsets"],
                    "values": [False, 16, 32],
                },
            },
        )

        self.assertEqual(result.status, "completed")
        for invalid_payload in invalid_payloads:
            rejected = run_reference(
                self.request(PER_GROUP_TRANSPOSE_REFERENCE_ID, invalid_payload)
            )
            self.assertEqual(rejected.status, "reference_error")
        output = decode_invocation_snapshot(result.value).return_value
        self.assertEqual(output.values[:18], tuple(range(0, 256, 16)) + (1, 17))
        self.assertEqual(output.values[256:274], tuple(range(256, 512, 16)) + (257, 273))
        self.assertEqual(output.shape, (32, 16))

    @unittest.skipUnless(_CUDA_AVAILABLE, "pack-seq isolated smoke requires CUDA")
    def test_pack_seq_public_case_passes_baseline_and_rejects_wrong_candidate(self):
        dataset = Path("work/official_triton_agent/datasets")
        case = materialize_pack_seq_public_case(dataset)

        def item(candidate_id, code):
            return Candidate(
                candidate_id,
                "_pack_seq_kernel",
                code,
                sha256_text(code),
                [],
                0,
                "correctness-smoke",
                None,
                None,
                "generated",
                None,
            )

        baseline = item(
            "pack-seq-baseline",
            (dataset / "_pack_seq_kernel/_pack_seq_kernel.py").read_text(encoding="utf-8"),
        )
        wrong = item(
            "pack-seq-wrong",
            "import torch\n"
            "def pack_seq_triton(x, lengths, pad_value=0.0, block_t=32, block_d=32):\n"
            "    return torch.zeros((3, 4, 4), device=x.device, dtype=x.dtype)\n",
        )
        runner = lambda candidate, item_case, inputs: run_pack_seq_candidate(
            candidate, item_case, inputs
        )
        backend = TorchTensorBackend("cuda")
        baseline_run = evaluate_candidate_correctness(
            baseline, [case], backend, run_case_reference, runner, decode_invocation_snapshot
        )
        wrong_run = evaluate_candidate_correctness(
            wrong, [case], backend, run_case_reference, runner, decode_invocation_snapshot
        )

        self.assertEqual(case.oracle_policy.atol, 1e-5)
        self.assertEqual(baseline_run.results[0].oracle_status, "passed")
        self.assertTrue(baseline_run.decision.eligible_for_performance)
        self.assertEqual(wrong_run.results[0].oracle_status, "failed")
        self.assertFalse(wrong_run.decision.eligible_for_performance)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_per_group_transpose_bridge_is_exact_and_fail_closed(self) -> None:
        case = materialize_per_group_transpose_public_case(
            Path("work/official_triton_agent/datasets")
        )
        code = "def per_group_transpose(a, expert_offsets, M_ALIGNMENT=1): return a"
        candidate = Candidate(
            id="per-group-transpose-reference-smoke",
            op_name="_per_group_transpose",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=[],
            generation=0,
            mutation_kind="reference-smoke",
            model_used=None,
            prompt_id=None,
            status="generated",
            score=None,
        )
        corrupt_first_value = False

        def candidate_runner(item, item_case, inputs):
            a = inputs.tensors["a"].detach().cpu()
            output = inputs.tensors["a"].new_empty(a.shape)
            output.reshape(-1)[:256] = a[:16].T.contiguous().reshape(-1)
            output.reshape(-1)[256:] = a[16:].T.contiguous().reshape(-1)
            if corrupt_first_value:
                output.reshape(-1)[0] += 1.0
            return CandidateRunResult(
                item.id,
                "completed",
                "runtime",
                value={
                    "return": {
                        "shape": [32, 16],
                        "dtype": "torch.float32",
                        "values": output.reshape(-1).tolist(),
                    },
                    "tensors": {},
                },
            )

        run = evaluate_candidate_correctness(
            candidate,
            [case],
            TorchTensorBackend("cpu"),
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )
        self.assertEqual(run.results[0].oracle_status, "passed")
        self.assertEqual(run.results[0].error_summary.compared_count, 512)
        self.assertTrue(run.decision.eligible_for_performance)

        corrupt_first_value = True
        failed_run = evaluate_candidate_correctness(
            candidate,
            [case],
            TorchTensorBackend("cpu"),
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )
        self.assertEqual(failed_run.results[0].oracle_status, "failed")
        self.assertFalse(failed_run.decision.eligible_for_performance)

    def test_set_k_and_s_reference_packs_first_token_raw_bytes(self) -> None:
        k_values = [0.0] * 128
        k_values[0:2] = [1.0, -2.0]
        result = run_reference(
            self.request(
                SET_K_AND_S_REFERENCE_ID,
                {
                    "loc": {
                        "shape": [3],
                        "dtype": "torch.int64",
                        "values": [0, 64, 128],
                    },
                    "index_k_first": {
                        "shape": [128],
                        "dtype": "torch.float16",
                        "values": k_values,
                    },
                    "scale_first": {
                        "shape": [1],
                        "dtype": "torch.float32",
                        "values": [1.25],
                    },
                    "page_size": 64,
                },
            )
        )

        self.assertEqual(result.status, "completed")
        k_bytes, scale_bytes = decode_invocation_snapshot(result.value).return_value
        self.assertEqual(k_bytes.values[:4], (0x00, 0x3C, 0x00, 0xC0))
        self.assertEqual(scale_bytes.values, (0x00, 0x00, 0xA0, 0x3F))
        self.assertEqual(k_bytes.dtype, "torch.uint8")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_set_k_and_s_reference_bridge_passes_first_token_side_effect_gate(self) -> None:
        case = materialize_set_k_and_s_public_case(
            Path("work/official_triton_agent/datasets")
        )
        code = "def _set_k_and_s_triton(buf, loc, index_k, index_k_scale, page_size): pass"
        candidate = Candidate(
            id="set-k-and-s-reference-smoke",
            op_name="_set_k_and_s_triton_kernel",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=[],
            generation=0,
            mutation_kind="reference-smoke",
            model_used=None,
            prompt_id=None,
            status="generated",
            score=None,
        )
        corrupt_first_byte = False

        def candidate_runner(item, item_case, inputs):
            buf = inputs.tensors["buf"].reshape(-1)
            k_bytes = b"".join(
                struct.pack("<e", float(value))
                for value in inputs.tensors["index_k"][0].tolist()
            )
            scale_bytes = struct.pack(
                "<f", float(inputs.tensors["index_k_scale"][0, 0])
            )
            encoded_k = list(k_bytes)
            if corrupt_first_byte:
                encoded_k[0] ^= 1
            buf[0:256] = inputs.tensors["buf"].new_tensor(encoded_k)
            buf[8192:8196] = inputs.tensors["buf"].new_tensor(list(scale_bytes))
            return CandidateRunResult(
                item.id,
                "completed",
                "runtime",
                value={
                    "return": None,
                    "tensors": {
                        "buf": {
                            "shape": [4, 8448],
                            "dtype": "torch.uint8",
                            "values": buf.tolist(),
                        }
                    },
                },
            )

        run = evaluate_candidate_correctness(
            candidate,
            [case],
            TorchTensorBackend("cpu"),
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )

        self.assertEqual(run.results[0].oracle_status, "passed")
        self.assertEqual(run.results[0].error_summary.compared_count, 260)
        self.assertTrue(run.decision.eligible_for_performance)

        corrupt_first_byte = True
        failed_run = evaluate_candidate_correctness(
            candidate,
            [case],
            TorchTensorBackend("cpu"),
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )
        self.assertEqual(failed_run.results[0].oracle_status, "failed")
        self.assertFalse(failed_run.decision.eligible_for_performance)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_round_bridge_passes_only_matching_shapes(self) -> None:
        case = materialize_act_quant_public_cases(
            Path("work/official_triton_agent/datasets")
        )[1]
        code = "def act_quant(x, block_size=128, scale_fmt=None): return x"
        candidate = Candidate(
            id="act-quant-round-bridge-smoke",
            op_name="_act_quant_kernel",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=[],
            generation=0,
            mutation_kind="bridge-smoke",
            model_used=None,
            prompt_id=None,
            status="generated",
            score=None,
        )

        def candidate_runner(item, item_case, inputs):
            value = {
                "return": [
                    {
                        "shape": [2, 4, 256],
                        "dtype": "torch.int32",
                        "values": [9] * (2 * 4 * 256),
                    },
                    {
                        "shape": [2, 4, 2],
                        "dtype": "torch.float64",
                        "values": [3.0] * (2 * 4 * 2),
                    },
                ],
                "tensors": {},
            }
            return CandidateRunResult(item.id, "completed", "runtime", value=value)

        run = evaluate_candidate_correctness(
            candidate,
            [case],
            TorchTensorBackend("cpu"),
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )

        self.assertEqual(run.results[0].oracle_status, "passed")
        self.assertEqual(run.results[0].error_summary.compared_count, 2)
        self.assertTrue(run.decision.eligible_for_performance)

    def test_reference_exception_is_structured(self) -> None:
        result = run_reference(
            self.request("protocol.raise.v1", {"message": "expected failure"})
        )

        self.assertEqual(result.status, "reference_error")
        self.assertEqual(result.error_type, "ValueError")
        self.assertEqual(result.error_message, "expected failure")
        self.assertEqual(result.worker_result.status, "completed")

    def test_reference_timeout_does_not_break_next_reference(self) -> None:
        timed_out = run_reference(
            self.request("protocol.sleep.v1", {"seconds": 10}, timeout=0.1)
        )
        healthy = run_reference(self.request("protocol.echo.v1", {"ok": True}))

        self.assertEqual(timed_out.status, "timeout")
        self.assertEqual(healthy.status, "completed")
        self.assertEqual(healthy.value["payload"], {"ok": True})


if __name__ == "__main__":
    unittest.main()
