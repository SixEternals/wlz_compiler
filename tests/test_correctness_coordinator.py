"""Focused tests for D1 correctness coordination and fail-closed gating."""

import base64
import copy
import importlib.util
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from wlz_optimizer.candidate_runner import (
    ACT_QUANT_ADAPTER_ID,
    CandidateRunRequest,
    CandidateRunResult,
    COUNT_EXPERT_ADAPTER_ID,
    PER_GROUP_TRANSPOSE_ADAPTER_ID,
    QUANTIZE_K_CACHE_ADAPTER_ID,
    SET_K_AND_S_ADAPTER_ID,
    run_act_quant_candidate,
    run_candidate,
    run_count_expert_candidate,
    run_per_group_transpose_candidate,
    run_quantize_k_cache_candidate,
    run_set_k_and_s_candidate,
)
from wlz_optimizer.case_catalog import (
    materialize_act_quant_public_cases,
    materialize_count_expert_basic_public_case,
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
    run_case_reference,
    run_reference,
)
from wlz_optimizer.schemas import (
    ArgumentBinding,
    Candidate,
    EvaluationCase,
    ExecutionBinding,
    InputContract,
    OraclePolicy,
    OracleTarget,
    TensorInitializer,
    TensorInputContract,
    ValueSelector,
)
from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.input_materializer import materialize_inputs
from wlz_optimizer.local_cuda_benchmark import (
    CudaBenchmarkConfig,
    compare_cuda_callables,
)
from wlz_optimizer.torch_backend import TorchTensorBackend


CANDIDATE_CODE = "def run(payload): return payload['snapshot']"

try:
    import torch as _torch
except ImportError:
    _torch = None

_CUDA_AVAILABLE = _torch is not None and _torch.cuda.is_available()


def candidate(code=CANDIDATE_CODE):
    return Candidate(
        id="candidate-1",
        op_name="demo",
        code=code,
        code_hash=sha256_text(code),
        parent_ids=[],
        generation=1,
        mutation_kind="smoke",
        model_used=None,
        prompt_id=None,
        status="generated",
        score=None,
    )


def evaluation_case(case_id="case-1"):
    return EvaluationCase(
        op_name="demo",
        case_id=case_id,
        inputs=InputContract(
            tensors={
                "state": TensorInputContract(
                    shape=[2],
                    dtype="float32",
                    layout="contiguous",
                    initializer=TensorInitializer("zeros"),
                    mutable=True,
                )
            }
        ),
        seed=7,
        execution=ExecutionBinding(
            "run", [ArgumentBinding("state", "tensor", "state")]
        ),
        oracle_policy=OraclePolicy(
            "protocol.echo.v1", "allclose.v1", "allclose", rtol=1e-3, atol=1e-4
        ),
        oracle_targets=[
            OracleTarget(
                "out",
                "output",
                ValueSelector("return"),
                ValueSelector("return"),
                "public_assertion",
            ),
            OracleTarget(
                "state",
                "side_effect",
                ValueSelector("tensor", tensor_name="state"),
                ValueSelector("tensor", tensor_name="state"),
                "manifest_strengthening",
            ),
        ],
    )


def encoded_snapshot(out=(1.0, 2.0), state=(3.0, 4.0)):
    def encoded(values):
        return {"shape": [len(values)], "dtype": "float32", "values": list(values)}

    return {"return": encoded(out), "tensors": {"state": encoded(state)}}


def decode_snapshot(value):
    def decode(data):
        return TensorSnapshot(tuple(data["shape"]), data["dtype"], tuple(data["values"]))

    return InvocationSnapshot(
        return_value=decode(value["return"]),
        tensors={name: decode(data) for name, data in value["tensors"].items()},
    )


class RecordingBackend:
    def __init__(self):
        self.seed_value = None
        self.next_id = 0

    def seed(self, seed):
        self.seed_value = seed

    def _value(self, shape, dtype, fill=0):
        self.next_id += 1
        return {
            "id": self.next_id,
            "shape": shape,
            "dtype": dtype,
            "values": [fill] * (shape[0] if shape else 1),
            "mutations": [],
        }

    def zeros(self, *, shape, dtype):
        return self._value(shape, dtype)

    def randn(self, *, shape, dtype):
        return self._value(shape, dtype, self.seed_value)

    def full(self, *, shape, dtype, fill_value):
        return self._value(shape, dtype, fill_value)

    def randint(self, *, shape, dtype, low, high):
        return self._value(shape, dtype, low)

    def clone(self, value):
        cloned = copy.deepcopy(value)
        self.next_id += 1
        cloned["id"] = self.next_id
        return cloned

    def as_strided(self, value, *, shape, strides):
        return value


def isolated_runners(candidate_snapshot, reference_snapshot=None, observations=None):
    reference_snapshot = reference_snapshot or encoded_snapshot()
    observations = observations if observations is not None else {}

    def reference_runner(case, inputs):
        storage = inputs.storages["tensor:state"]
        observations["reference_id"] = storage["id"]
        storage["mutations"].append("reference-mutated")
        result = run_reference(
            ReferenceRequest(
                "protocol.echo.v1",
                {"snapshot": reference_snapshot},
                timeout_seconds=2.0,
            )
        )
        return replace(result, value=result.value["payload"]["snapshot"])

    def candidate_runner(item, case, inputs):
        storage = inputs.storages["tensor:state"]
        observations["candidate_id"] = storage["id"]
        observations["candidate_mutations"] = list(storage["mutations"])
        return run_candidate(
            CandidateRunRequest(
                item.id,
                item.code,
                item.code_hash,
                case.execution.entrypoint,
                {"snapshot": candidate_snapshot},
                timeout_seconds=2.0,
            )
        )

    return reference_runner, candidate_runner


class CorrectnessCoordinatorTests(unittest.TestCase):
    def test_real_c2_c3_smoke_passes_gate_and_preserves_fresh_state(self):
        observations = {}
        runners = isolated_runners(encoded_snapshot(), observations=observations)

        run = evaluate_candidate_correctness(
            candidate(), [evaluation_case()], RecordingBackend(), *runners, decode_snapshot
        )

        self.assertEqual(run.results[0].oracle_status, "passed")
        self.assertTrue(run.decision.eligible_for_performance)
        self.assertNotEqual(observations["reference_id"], observations["candidate_id"])
        self.assertEqual(observations["candidate_mutations"], [])

    def test_numeric_mismatch_blocks_gate_with_c4_summary(self):
        runners = isolated_runners(encoded_snapshot(out=(9.0, 2.0)))

        run = evaluate_candidate_correctness(
            candidate(), [evaluation_case()], RecordingBackend(), *runners, decode_snapshot
        )

        result = run.results[0]
        self.assertEqual(result.oracle_status, "failed")
        self.assertEqual(result.error_summary.mismatch_kind, "value")
        self.assertEqual(run.decision.blocking_reasons, ("failed_case",))

    def test_reference_failure_is_oracle_error_and_skips_candidate(self):
        calls = []

        def reference_runner(case, inputs):
            return run_reference(
                ReferenceRequest(
                    "protocol.raise.v1",
                    {"message": "reference failed"},
                    timeout_seconds=2.0,
                )
            )

        def candidate_runner(*args):
            calls.append("candidate")
            raise AssertionError("candidate must not run")

        run = evaluate_candidate_correctness(
            candidate(),
            [evaluation_case()],
            RecordingBackend(),
            reference_runner,
            candidate_runner,
            decode_snapshot,
        )

        self.assertEqual(run.results[0].oracle_status, "oracle_error")
        self.assertEqual(run.decision.blocking_reasons, ("oracle_error",))
        self.assertEqual(calls, [])

    def test_candidate_runtime_failure_is_failed_and_blocks_gate(self):
        failing = candidate("def run(payload): raise RuntimeError('candidate failed')")
        runners = isolated_runners(encoded_snapshot())

        run = evaluate_candidate_correctness(
            failing, [evaluation_case()], RecordingBackend(), *runners, decode_snapshot
        )

        result = run.results[0]
        self.assertEqual(result.oracle_status, "failed")
        self.assertEqual(result.error_summary.mismatch_kind, "candidate_runtime_error")
        self.assertFalse(run.decision.eligible_for_performance)

    def test_unconfigured_act_quant_backend_is_unknown_and_blocks_gate(self):
        case = materialize_act_quant_public_cases(
            Path("work/official_triton_agent/datasets")
        )[0]
        item = replace(candidate(), op_name="_act_quant_kernel")
        calls = []

        class UnavailableDevice:
            @staticmethod
            def is_available():
                return False

        unavailable_torch = type(
            "UnavailableTorch",
            (),
            {"npu": UnavailableDevice(), "cuda": UnavailableDevice()},
        )()

        def reference_runner(item_case, inputs):
            result = run_reference(
                ReferenceRequest(
                    "protocol.echo.v1",
                    {
                        "snapshot": {
                            "return": [
                                {"shape": [2, 4, 256], "dtype": "shape_only", "values": []},
                                {"shape": [2, 4, 2], "dtype": "shape_only", "values": []},
                            ],
                            "tensors": {},
                        }
                    },
                    timeout_seconds=2.0,
                )
            )
            return replace(result, value=result.value["payload"]["snapshot"])

        def candidate_runner(item_candidate, item_case, inputs):
            return run_act_quant_candidate(
                item_candidate,
                item_case,
                inputs,
                torch_module=unavailable_torch,
                backend_executor=lambda *args: calls.append(args),
            )

        run = evaluate_candidate_correctness(
            item,
            [case],
            RecordingBackend(),
            reference_runner,
            candidate_runner,
            decode_invocation_snapshot,
        )

        self.assertEqual(calls, [])
        self.assertEqual(run.results[0].oracle_status, "unknown")
        self.assertIn("candidate_not_configured", run.results[0].message)
        self.assertEqual(run.decision.blocking_reasons, ("unknown_case",))
        self.assertFalse(run.decision.eligible_for_performance)

    def test_act_quant_bridge_uses_isolated_adapter_or_explicit_executor(self):
        case = materialize_act_quant_public_cases(
            Path("work/official_triton_agent/datasets")
        )[0]
        item = replace(candidate(), op_name="_act_quant_kernel")
        inputs = materialize_inputs(case, RecordingBackend())

        class Device:
            def __init__(self, available):
                self.available = available

            def is_available(self):
                return self.available

        fake_torch = type(
            "CudaTorch", (), {"npu": Device(False), "cuda": Device(True)}
        )()
        missing = run_act_quant_candidate(item, case, inputs, torch_module=fake_torch)
        devices = []
        completed = run_act_quant_candidate(
            item,
            case,
            inputs,
            torch_module=fake_torch,
            backend_executor=lambda candidate, item_case, item_inputs, device: (
                devices.append(device) or {"return": None, "tensors": {}}
            ),
        )

        self.assertEqual(missing.status, "not_configured")
        self.assertEqual(missing.error_type, "CandidateBackendInputsUnavailable")
        self.assertEqual(completed.status, "completed")
        self.assertEqual(devices, ["cuda"])

    @unittest.skipUnless(_torch is not None, "Torch is not installed")
    def test_act_quant_worker_protocol_is_bounded(self):
        values = [0.0] * (2 * 4 * 256)
        payload = {
            "device": "cpu",
            "x": {
                "shape": [2, 4, 256],
                "dtype": "torch.float32",
                "values": values,
            },
            "block_size": 128,
            "scale_fmt": None,
        }
        code = (
            "import torch\n"
            "def act_quant(x, block_size=128, scale_fmt=None):\n"
            " return x.to(torch.float16), torch.zeros((2, 4, 2), dtype=torch.float32)\n"
        )
        completed = run_candidate(
            CandidateRunRequest(
                "act-quant-worker",
                code,
                sha256_text(code),
                "act_quant",
                payload,
                timeout_seconds=10.0,
                max_output_bytes=128 * 1024,
                adapter_id=ACT_QUANT_ADAPTER_ID,
            )
        )
        malformed = run_candidate(
            CandidateRunRequest(
                "act-quant-malformed",
                code,
                sha256_text(code),
                "act_quant",
                {**payload, "block_size": 64},
                timeout_seconds=10.0,
                max_output_bytes=128 * 1024,
                adapter_id=ACT_QUANT_ADAPTER_ID,
            )
        )

        self.assertEqual((completed.status, completed.phase), ("completed", "runtime"))
        self.assertEqual(completed.value["return"][0]["shape"], [2, 4, 256])
        self.assertEqual(completed.value["return"][1]["shape"], [2, 4, 2])
        self.assertEqual((malformed.status, malformed.phase), ("worker_error", "protocol"))
        self.assertFalse(Path(completed.worker_result.working_directory).exists())

    @unittest.skipUnless(_CUDA_AVAILABLE, "act-quant isolated smoke requires CUDA")
    def test_act_quant_cuda_parent_passes_and_wrong_candidate_fails(self):
        datasets = Path(
            os.environ.get(
                "WLZ_CORRECTNESS_DATASETS_DIR",
                "work/official_triton_agent/datasets",
            )
        )
        configured_path = os.environ.get("WLZ_ACT_QUANT_CANDIDATE")
        candidate_path = Path(configured_path) if configured_path else (
            datasets / "_act_quant_kernel/_act_quant_kernel_1.py"
        )
        code = candidate_path.resolve().read_text(encoding="utf-8")
        cases = materialize_act_quant_public_cases(datasets)

        def evaluate(item_code, item_id):
            item = replace(
                candidate(item_code), id=item_id, op_name="_act_quant_kernel"
            )
            return evaluate_candidate_correctness(
                item,
                cases,
                TorchTensorBackend("cuda"),
                run_case_reference,
                lambda value, case, inputs: run_act_quant_candidate(
                    value, case, inputs
                ),
                decode_invocation_snapshot,
            )

        admitted = evaluate(code, candidate_path.stem)
        self.assertEqual(
            [result.oracle_status for result in admitted.results],
            ["passed", "passed"],
        )
        self.assertTrue(admitted.decision.eligible_for_performance)
        self.assertEqual(admitted.decision.blocking_reasons, ())

        if configured_path is None:
            wrong_code = (
                "import torch\n"
                "def act_quant(x, block_size=128, scale_fmt=None):\n"
                " return torch.zeros_like(x, dtype=torch.float16), "
                "torch.zeros((*x.shape[:-1], x.shape[-1] // block_size), "
                "dtype=torch.float32, device=x.device)\n"
            )
            rejected = evaluate(wrong_code, "known-wrong-act-quant")
            self.assertEqual(
                [result.oracle_status for result in rejected.results],
                ["failed", "passed"],
            )
            self.assertFalse(rejected.decision.eligible_for_performance)
            self.assertEqual(rejected.decision.blocking_reasons, ("failed_case",))

        print("B2_LOCAL_ACT_QUANT_MATRIX")

    @unittest.skipUnless(_torch is not None, "Torch is not installed")
    def test_count_expert_worker_protocol_is_bounded(self):
        payload = {
            "device": "cpu",
            "topk_ids": {
                "shape": [8],
                "dtype": "torch.int32",
                "values": [0, 1, 2, 0, 1, 3, 2, 0],
            },
            "num_local_experts": 4,
            "expert_map": None,
        }
        code = (
            "import torch\n"
            "def count_expert_num_tokens(topk_ids, num_local_experts, expert_map):\n"
            " return torch.bincount(topk_ids, minlength=num_local_experts).to(torch.int32)\n"
        )
        completed = run_candidate(
            CandidateRunRequest(
                "count-expert-worker", code, sha256_text(code),
                "count_expert_num_tokens", payload,
                timeout_seconds=5.0,
                max_output_bytes=128 * 1024,
                adapter_id=COUNT_EXPERT_ADAPTER_ID,
            )
        )
        malformed = run_candidate(
            CandidateRunRequest(
                "count-expert-malformed", code, sha256_text(code),
                "count_expert_num_tokens", {**payload, "expert_map": [0, 1, 2, 3]},
                timeout_seconds=5.0,
                max_output_bytes=128 * 1024,
                adapter_id=COUNT_EXPERT_ADAPTER_ID,
            )
        )
        timeout_code = (
            "import time\n"
            "def count_expert_num_tokens(a, b, c): time.sleep(10)\n"
        )
        timed_out = run_candidate(
            CandidateRunRequest(
                "count-expert-timeout", timeout_code, sha256_text(timeout_code),
                "count_expert_num_tokens", payload,
                timeout_seconds=0.1,
                max_output_bytes=128 * 1024,
                adapter_id=COUNT_EXPERT_ADAPTER_ID,
            )
        )

        self.assertEqual((completed.status, completed.phase), ("completed", "runtime"))
        self.assertEqual(completed.value["return"]["values"], [3, 2, 2, 1])
        self.assertEqual((malformed.status, malformed.phase), ("worker_error", "protocol"))
        self.assertEqual((timed_out.status, timed_out.phase), ("timeout", "candidate_process"))
        self.assertFalse(Path(completed.worker_result.working_directory).exists())
        self.assertFalse(Path(timed_out.worker_result.working_directory).exists())

    @unittest.skipUnless(_CUDA_AVAILABLE, "count-expert isolated smoke requires CUDA")
    def test_count_expert_cuda_parent_passes_and_wrong_candidate_fails(self):
        datasets = Path(
            os.environ.get(
                "WLZ_CORRECTNESS_DATASETS_DIR",
                "work/official_triton_agent/datasets",
            )
        )
        configured_path = os.environ.get("WLZ_COUNT_EXPERT_CANDIDATE")
        parent_path = datasets / (
            "_count_expert_num_tokens/_count_expert_num_tokens.py"
        )
        candidate_path = Path(configured_path) if configured_path else parent_path
        source = candidate_path.resolve().read_text(encoding="utf-8")
        case = materialize_count_expert_basic_public_case(datasets)

        def evaluate(item_code, item_id):
            item = replace(
                candidate(item_code), id=item_id, op_name="_count_expert_num_tokens"
            )
            return evaluate_candidate_correctness(
                item,
                [case],
                TorchTensorBackend("cuda"),
                run_case_reference,
                lambda value, item_case, inputs: run_count_expert_candidate(
                    value, item_case, inputs
                ),
                decode_invocation_snapshot,
            )

        admitted = evaluate(source, candidate_path.stem)
        self.assertEqual(admitted.results[0].oracle_status, "passed")
        self.assertTrue(admitted.decision.eligible_for_performance)

        if configured_path is None:
            wrong_source = source.replace(
                "tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))",
                "tl.store(expert_num_tokens_ptr + curr_expert, 0)",
                1,
            )
            self.assertNotEqual(wrong_source, source)
            rejected = evaluate(wrong_source, "known-wrong-count-expert")
            self.assertEqual(rejected.results[0].oracle_status, "failed")
            self.assertFalse(rejected.decision.eligible_for_performance)

        print("B2_LOCAL_COUNT_EXPERT_MATRIX")

    @unittest.skipUnless(_torch is not None, "Torch is not installed")
    def test_set_k_and_s_worker_protocol_is_bounded(self):
        payload = {
            "device": "cpu",
            "buf": {
                "shape": [4, 8448],
                "dtype": "torch.uint8",
                "data_base64": base64.urlsafe_b64encode(bytes(4 * 8448)).decode(
                    "ascii"
                ),
            },
            "loc": {
                "shape": [3],
                "dtype": "torch.int64",
                "values": [0, 64, 128],
            },
            "index_k": {
                "shape": [3, 128],
                "dtype": "torch.float16",
                "values": [float(index % 17) for index in range(3 * 128)],
            },
            "index_k_scale": {
                "shape": [3, 1],
                "dtype": "torch.float32",
                "values": [1.0, 2.0, 3.0],
            },
            "page_size": 64,
        }
        code = (
            "import torch\n"
            "def _set_k_and_s_triton(buf, loc, index_k, index_k_scale, page_size):\n"
            " data = buf.view(torch.float16).view(-1)\n"
            " scales = buf.view(torch.float32).view(-1)\n"
            " for token in range(loc.numel()):\n"
            "  page = int(loc[token]) // page_size\n"
            "  offset = int(loc[token]) % page_size\n"
            "  data[page * (8448 // 2) + offset * 128:"
            "page * (8448 // 2) + (offset + 1) * 128] = index_k[token]\n"
            "  scales[page * (8448 // 4) + 8192 // 4 + offset] = "
            "index_k_scale[token, 0]\n"
        )
        completed = run_candidate(
            CandidateRunRequest(
                "set-k-and-s-worker",
                code,
                sha256_text(code),
                "_set_k_and_s_triton",
                payload,
                timeout_seconds=10.0,
                max_output_bytes=128 * 1024,
                adapter_id=SET_K_AND_S_ADAPTER_ID,
            )
        )
        malformed = run_candidate(
            CandidateRunRequest(
                "set-k-and-s-malformed",
                code,
                sha256_text(code),
                "_set_k_and_s_triton",
                {**payload, "page_size": 32},
                timeout_seconds=10.0,
                max_output_bytes=128 * 1024,
                adapter_id=SET_K_AND_S_ADAPTER_ID,
            )
        )

        self.assertEqual(
            (completed.status, completed.phase),
            ("completed", "runtime"),
            completed,
        )
        self.assertEqual(completed.value["tensors"]["buf"]["shape"], [4, 8448])
        self.assertEqual(completed.value["tensors"]["buf"]["dtype"], "torch.uint8")
        self.assertEqual((malformed.status, malformed.phase), ("worker_error", "protocol"))
        self.assertFalse(Path(completed.worker_result.working_directory).exists())

    @unittest.skipUnless(_CUDA_AVAILABLE, "set-k-and-s isolated smoke requires CUDA")
    def test_set_k_and_s_cuda_parent_is_unsafe_and_corrected_control_passes(self):
        datasets = Path(
            os.environ.get(
                "WLZ_CORRECTNESS_DATASETS_DIR",
                "work/official_triton_agent/datasets",
            )
        )
        configured_path = os.environ.get("WLZ_SET_K_AND_S_CANDIDATE")
        parent_path = datasets / (
            "_set_k_and_s_triton_kernel/_set_k_and_s_triton_kernel.py"
        )
        candidate_path = Path(configured_path) if configured_path else parent_path
        source = candidate_path.resolve().read_text(encoding="utf-8")
        case = materialize_set_k_and_s_public_case(datasets)

        def evaluate(item_code, item_id):
            item = replace(
                candidate(item_code),
                id=item_id,
                op_name="_set_k_and_s_triton_kernel",
            )
            return evaluate_candidate_correctness(
                item,
                [case],
                TorchTensorBackend("cuda"),
                run_case_reference,
                lambda value, item_case, inputs: run_set_k_and_s_candidate(
                    value, item_case, inputs
                ),
                decode_invocation_snapshot,
            )

        configured = evaluate(source, candidate_path.stem)
        if configured_path is not None:
            self.assertEqual(configured.results[0].oracle_status, "passed")
            self.assertTrue(configured.decision.eligible_for_performance)
        else:
            self.assertEqual(configured.results[0].oracle_status, "failed")
            self.assertFalse(configured.decision.eligible_for_performance)

            corrected_source = source.replace(
                "loc_page_index * BUF_NUMEL_PER_PAGE\n",
                "loc_page_index * (BUF_NUMEL_PER_PAGE // 2)\n",
                1,
            )
            self.assertNotEqual(corrected_source, source)
            corrected = evaluate(corrected_source, "corrected-set-k-and-s-control")
            self.assertEqual(
                corrected.results[0].oracle_status,
                "passed",
                corrected.results[0],
            )
            self.assertTrue(corrected.decision.eligible_for_performance)

            no_op_code = (
                "def _set_k_and_s_triton(buf, loc, index_k, index_k_scale, page_size):\n"
                " pass\n"
            )
            rejected = evaluate(no_op_code, "known-wrong-set-k-and-s")
            self.assertEqual(rejected.results[0].oracle_status, "failed")
            self.assertFalse(rejected.decision.eligible_for_performance)

        print("B2_LOCAL_SET_K_AND_S_MATRIX")

    @unittest.skipUnless(_torch is not None, "Torch is not installed")
    def test_per_group_transpose_dispatch_is_explicit_and_fail_closed(self):
        case = materialize_per_group_transpose_public_case(
            Path("work/official_triton_agent/datasets")
        )
        item = replace(candidate(), op_name="_per_group_transpose")
        inputs = materialize_inputs(case, TorchTensorBackend("cpu"))

        class Device:
            def __init__(self, available):
                self.available = available

            def is_available(self):
                return self.available

        unavailable = type(
            "UnavailableTorch", (), {"npu": Device(False), "cuda": Device(False)}
        )()
        cuda_torch = type(
            "CudaTorch", (), {"npu": Device(False), "cuda": Device(True)}
        )()
        npu_torch = type(
            "NpuTorch", (), {"npu": Device(True), "cuda": Device(True)}
        )()
        devices = []
        snapshot = {"return": None, "tensors": {}}

        no_device = run_per_group_transpose_candidate(
            item, case, inputs, torch_module=unavailable
        )
        with patch("wlz_optimizer.candidate_runner.run_candidate") as isolated:
            isolated.return_value = CandidateRunResult(
                item.id, "completed", "runtime", value=snapshot
            )
            no_executor = run_per_group_transpose_candidate(
                item, case, inputs, torch_module=cuda_torch
            )
            request = isolated.call_args.args[0]
        cuda = run_per_group_transpose_candidate(
            item,
            case,
            inputs,
            torch_module=cuda_torch,
            backend_executor=lambda *args: devices.append(args[3]) or snapshot,
        )
        npu = run_per_group_transpose_candidate(
            item,
            case,
            inputs,
            torch_module=npu_torch,
            backend_executor=lambda *args: devices.append(args[3]) or snapshot,
        )
        oversized = run_per_group_transpose_candidate(
            item,
            case,
            inputs,
            torch_module=cuda_torch,
            backend_executor=lambda *args: {"payload": "x" * (128 * 1024)},
        )
        nonfinite = run_per_group_transpose_candidate(
            item,
            case,
            inputs,
            torch_module=cuda_torch,
            backend_executor=lambda *args: {"value": float("nan")},
        )

        def fail_executor(*args):
            raise RuntimeError("executor failed")

        runtime_error = run_per_group_transpose_candidate(
            item,
            case,
            inputs,
            torch_module=cuda_torch,
            backend_executor=fail_executor,
        )

        def candidate_runner(item_candidate, item_case, item_inputs):
            return run_per_group_transpose_candidate(
                item_candidate,
                item_case,
                item_inputs,
                torch_module=unavailable,
            )

        run = evaluate_candidate_correctness(
            item,
            [case],
            TorchTensorBackend("cpu"),
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )

        self.assertEqual(
            (no_device.status, no_device.phase, no_device.error_type),
            ("not_configured", "backend", "CandidateBackendNotConfigured"),
        )
        self.assertEqual(
            (no_executor.status, no_executor.phase, no_executor.error_type),
            ("completed", "runtime", None),
        )
        self.assertEqual(request.adapter_id, PER_GROUP_TRANSPOSE_ADAPTER_ID)
        self.assertEqual(request.max_output_bytes, 128 * 1024)
        self.assertEqual(request.timeout_seconds, 20.0)
        self.assertEqual((cuda.status, npu.status), ("completed", "completed"))
        self.assertEqual(devices, ["cuda", "npu"])
        self.assertEqual(oversized.error_type, "CandidateOutputTooLarge")
        self.assertEqual(nonfinite.phase, "protocol")
        self.assertEqual((runtime_error.status, runtime_error.phase), ("runtime_error", "runtime"))
        self.assertEqual(run.results[0].oracle_status, "unknown")
        self.assertEqual(run.decision.blocking_reasons, ("unknown_case",))
        self.assertFalse(run.decision.eligible_for_performance)

    @unittest.skipUnless(_CUDA_AVAILABLE, "per-group isolated smoke requires CUDA")
    def test_per_group_transpose_isolated_gpu_baseline_and_wrong_candidate(self):
        dataset = Path("work/official_triton_agent/datasets")
        case = materialize_per_group_transpose_public_case(dataset)
        source_path = (
            dataset / "_per_group_transpose/_per_group_transpose.py"
        )
        baseline_code = source_path.read_text(encoding="utf-8")
        baseline = replace(
            candidate(baseline_code),
            id="per-group-transpose-gpu-baseline",
            op_name="_per_group_transpose",
        )
        wrong_code = (
            "import torch\n"
            "def per_group_transpose(a, expert_offsets, M_ALIGNMENT=1):\n"
            "    return a\n"
        )
        wrong = replace(
            candidate(wrong_code),
            id="per-group-transpose-gpu-wrong",
            op_name="_per_group_transpose",
        )
        backend = TorchTensorBackend("cuda")

        def candidate_runner(item, item_case, item_inputs):
            return run_per_group_transpose_candidate(item, item_case, item_inputs)

        baseline_run = evaluate_candidate_correctness(
            baseline,
            [case],
            backend,
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )
        wrong_run = evaluate_candidate_correctness(
            wrong,
            [case],
            backend,
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )

        self.assertEqual(baseline_run.results[0].oracle_status, "passed")
        self.assertTrue(baseline_run.decision.eligible_for_performance)
        self.assertEqual(wrong_run.results[0].oracle_status, "failed")
        self.assertFalse(wrong_run.decision.eligible_for_performance)

    @unittest.skipUnless(_CUDA_AVAILABLE, "per-group timing requires CUDA")
    def test_per_group_transpose_correctness_gate_precedes_interleaved_timing(self):
        dataset = Path("work/official_triton_agent/datasets")
        case = materialize_per_group_transpose_public_case(dataset)
        baseline_path = dataset / "_per_group_transpose/_per_group_transpose.py"
        control_path = dataset / "_per_group_transpose/_per_group_transpose_1.py"
        control_code = control_path.read_text(encoding="utf-8")
        control = replace(
            candidate(control_code),
            id="per-group-transpose-timing-control",
            op_name="_per_group_transpose",
        )
        backend = TorchTensorBackend("cuda")
        correctness = evaluate_candidate_correctness(
            control,
            [case],
            backend,
            run_case_reference,
            lambda item, item_case, item_inputs: run_per_group_transpose_candidate(
                item, item_case, item_inputs
            ),
            decode_invocation_snapshot,
        )

        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        baseline_module = load_module(
            "_wlz_per_group_transpose_timing_baseline", baseline_path
        )
        control_module = load_module(
            "_wlz_per_group_transpose_timing_control", control_path
        )
        inputs = materialize_inputs(case, backend)

        def baseline_once():
            return baseline_module.per_group_transpose(
                inputs.tensors["a"],
                inputs.tensors["expert_offsets"],
                inputs.scalars["M_ALIGNMENT"],
            )

        def control_once():
            return control_module.per_group_transpose(
                inputs.tensors["a"],
                inputs.tensors["expert_offsets"],
                inputs.scalars["M_ALIGNMENT"],
            )

        comparison = compare_cuda_callables(
            control.id,
            case.signature(),
            correctness.decision,
            baseline_once,
            control_once,
            torch_module=_torch,
            config=CudaBenchmarkConfig(warmup_runs=2, measurement_runs=5),
        )

        self.assertTrue(correctness.decision.eligible_for_performance)
        self.assertEqual(comparison.status, "completed")
        self.assertEqual(comparison.executor, "local_cuda_proxy_interleaved")
        self.assertEqual(len(comparison.baseline_samples_ms), 5)
        self.assertEqual(len(comparison.candidate_samples_ms), 5)
        self.assertGreater(comparison.baseline_local_latency_ms, 0.0)
        self.assertGreater(comparison.candidate_local_latency_ms, 0.0)
        self.assertGreater(comparison.candidate_over_baseline_latency_ratio, 0.0)
        self.assertEqual(len(comparison.environment_fingerprint), 16)

    @unittest.skipUnless(_torch is not None, "Torch is not installed")
    def test_per_group_transpose_worker_cpu_protocol_is_bounded(self):
        payload = {
            "device": "cpu",
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
        code = (
            "import torch\n"
            "def per_group_transpose(a, expert_offsets, M_ALIGNMENT):\n"
            "    return a\n"
        )
        completed = run_candidate(
            CandidateRunRequest(
                "per-group-cpu-adapter",
                code,
                sha256_text(code),
                "per_group_transpose",
                payload,
                timeout_seconds=5.0,
                max_output_bytes=128 * 1024,
                adapter_id=PER_GROUP_TRANSPOSE_ADAPTER_ID,
            )
        )
        malformed = run_candidate(
            CandidateRunRequest(
                "per-group-cpu-malformed",
                code,
                sha256_text(code),
                "per_group_transpose",
                {**payload, "M_ALIGNMENT": True},
                timeout_seconds=5.0,
                max_output_bytes=128 * 1024,
                adapter_id=PER_GROUP_TRANSPOSE_ADAPTER_ID,
            )
        )
        malformed_offsets = run_candidate(
            CandidateRunRequest(
                "per-group-cpu-malformed-offsets",
                code,
                sha256_text(code),
                "per_group_transpose",
                {
                    **payload,
                    "expert_offsets": {
                        **payload["expert_offsets"],
                        "values": [0.5, 16, 32],
                    },
                },
                timeout_seconds=5.0,
                max_output_bytes=128 * 1024,
                adapter_id=PER_GROUP_TRANSPOSE_ADAPTER_ID,
            )
        )
        wrong_device_code = (
            "import torch\n"
            "def per_group_transpose(a, expert_offsets, M_ALIGNMENT):\n"
            "    return torch.empty_like(a, device='meta')\n"
        )
        wrong_device = run_candidate(
            CandidateRunRequest(
                "per-group-cpu-wrong-device",
                wrong_device_code,
                sha256_text(wrong_device_code),
                "per_group_transpose",
                payload,
                timeout_seconds=5.0,
                max_output_bytes=128 * 1024,
                adapter_id=PER_GROUP_TRANSPOSE_ADAPTER_ID,
            )
        )
        timeout_code = "import time\ndef per_group_transpose(a, b, c): time.sleep(10)"
        timed_out = run_candidate(
            CandidateRunRequest(
                "per-group-cpu-timeout",
                timeout_code,
                sha256_text(timeout_code),
                "per_group_transpose",
                payload,
                timeout_seconds=0.1,
                max_output_bytes=128 * 1024,
                adapter_id=PER_GROUP_TRANSPOSE_ADAPTER_ID,
            )
        )

        self.assertEqual((completed.status, completed.phase), ("completed", "runtime"))
        self.assertEqual(completed.value["return"]["shape"], [32, 16])
        self.assertEqual(completed.value["return"]["device"], "cpu")
        self.assertFalse(Path(completed.worker_result.working_directory).exists())
        self.assertEqual((malformed.status, malformed.phase), ("worker_error", "protocol"))
        self.assertEqual(
            (malformed_offsets.status, malformed_offsets.phase),
            ("worker_error", "protocol"),
        )
        self.assertEqual(
            (wrong_device.status, wrong_device.phase), ("worker_error", "protocol")
        )
        self.assertEqual((timed_out.status, timed_out.phase), ("timeout", "candidate_process"))
        self.assertFalse(Path(timed_out.worker_result.working_directory).exists())

    def test_quantize_k_cache_bridge_bounds_snapshot_and_preserves_device(self):
        case = materialize_quantize_k_cache_public_cases(
            Path("work/official_triton_agent/datasets")
        )[0]
        item = replace(candidate(), op_name="_quantize_k_cache_fast_kernel")
        inputs = materialize_inputs(case, RecordingBackend())

        class Device:
            def __init__(self, available):
                self.available = available

            def is_available(self):
                return self.available

        fake_torch = type(
            "CudaTorch", (), {"npu": Device(False), "cuda": Device(True)}
        )()
        npu_torch = type(
            "NpuTorch", (), {"npu": Device(True), "cuda": Device(True)}
        )()
        calls = []
        snapshot = {
            "return": {
                "shape": [4, 592],
                "dtype": "torch.bfloat16",
                "values": [0.0] * (4 * 592),
                "device": "cuda",
            },
            "tensors": {},
        }

        result = run_quantize_k_cache_candidate(
            item,
            case,
            inputs,
            torch_module=fake_torch,
            backend_executor=lambda *args: calls.append(args) or snapshot,
        )
        missing = run_quantize_k_cache_candidate(
            item, case, inputs, torch_module=fake_torch
        )
        npu_calls = []
        run_quantize_k_cache_candidate(
            item,
            case,
            inputs,
            torch_module=npu_torch,
            backend_executor=lambda *args: npu_calls.append(args) or snapshot,
        )

        class SerializableTensor:
            dtype = "torch.bfloat16"

            def __init__(self, shape):
                self.shape = shape

            def detach(self):
                return self

            cpu = contiguous = detach

            def reshape(self, *shape):
                return self

            def tolist(self):
                return [0.0] * (self.shape[0] * self.shape[1])

        inputs.tensors["k_nope"] = SerializableTensor((4, 512))
        inputs.tensors["k_rope"] = SerializableTensor((4, 64))
        with patch("wlz_optimizer.candidate_runner.run_candidate") as isolated:
            isolated.return_value = CandidateRunResult(
                item.id, "completed", "runtime", value=snapshot
            )
            run_quantize_k_cache_candidate(
                item, case, inputs, torch_module=fake_torch, timeout_seconds=7.0
            )
        request = isolated.call_args.args[0]
        oversized = run_quantize_k_cache_candidate(
            item,
            case,
            inputs,
            torch_module=fake_torch,
            backend_executor=lambda *args: {"payload": "x" * (128 * 1024)},
        )
        nonfinite = run_quantize_k_cache_candidate(
            item,
            case,
            inputs,
            torch_module=fake_torch,
            backend_executor=lambda *args: {"value": float("nan")},
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(calls[0][3], "cuda")
        self.assertEqual(missing.status, "not_configured")
        self.assertEqual(missing.error_type, "CandidateBackendInputsUnavailable")
        self.assertEqual(npu_calls[0][3], "npu")
        self.assertEqual(result.value["return"]["device"], "cuda")
        self.assertEqual(request.adapter_id, QUANTIZE_K_CACHE_ADAPTER_ID)
        self.assertEqual(request.max_output_bytes, 128 * 1024)
        self.assertEqual(request.timeout_seconds, 7.0)
        self.assertEqual((oversized.status, oversized.phase), ("worker_error", "protocol"))
        self.assertEqual(oversized.error_type, "CandidateOutputTooLarge")
        self.assertEqual((nonfinite.status, nonfinite.phase), ("worker_error", "protocol"))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_quantize_k_cache_worker_is_bounded_and_kills_timeout(self):
        payload = {
            "device": "cpu",
            "k_nope": {
                "shape": [4, 512],
                "dtype": "torch.bfloat16",
                "values": [0.0] * (4 * 512),
            },
            "k_rope": {
                "shape": [4, 64],
                "dtype": "torch.bfloat16",
                "values": [1.0] * (4 * 64),
            },
            "group_size": 128,
        }
        code = (
            "import torch\n"
            "def run(k_nope, k_rope, group_size):\n"
            " return torch.cat((torch.zeros((4, 528), dtype=k_rope.dtype), k_rope), 1)\n"
        )
        completed = run_candidate(
            CandidateRunRequest(
                "quantize-worker-smoke",
                code,
                sha256_text(code),
                "run",
                payload,
                timeout_seconds=10.0,
                max_output_bytes=128 * 1024,
                adapter_id=QUANTIZE_K_CACHE_ADAPTER_ID,
            )
        )
        timeout_code = "import time\ndef run(a, b, c): time.sleep(10)"
        timed_out = run_candidate(
            CandidateRunRequest(
                "quantize-worker-timeout",
                timeout_code,
                sha256_text(timeout_code),
                "run",
                payload,
                timeout_seconds=0.1,
                max_output_bytes=128 * 1024,
                adapter_id=QUANTIZE_K_CACHE_ADAPTER_ID,
            )
        )
        bad_payload = dict(payload, group_size=64)
        malformed = run_candidate(
            CandidateRunRequest(
                "quantize-worker-bad-payload",
                code,
                sha256_text(code),
                "run",
                bad_payload,
                timeout_seconds=2.0,
                max_output_bytes=128 * 1024,
                adapter_id=QUANTIZE_K_CACHE_ADAPTER_ID,
            )
        )

        self.assertEqual((completed.status, completed.phase), ("completed", "runtime"))
        self.assertEqual(completed.value["return"]["shape"], [4, 592])
        self.assertEqual(completed.value["return"]["device"], "cpu")
        self.assertFalse(completed.worker_result.stdout.truncated)
        self.assertFalse(Path(completed.worker_result.working_directory).exists())
        self.assertEqual((timed_out.status, timed_out.phase), ("timeout", "candidate_process"))
        self.assertFalse(Path(timed_out.worker_result.working_directory).exists())
        self.assertEqual((malformed.status, malformed.phase), ("worker_error", "protocol"))
        with self.assertRaisesRegex(ValueError, "not registered"):
            CandidateRunRequest(
                "bad-adapter", code, sha256_text(code), "run", payload, 1.0,
                adapter_id="arbitrary.module:function",
            )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
    def test_quantize_worker_wrong_rope_blocks_gate(self):
        cases = materialize_quantize_k_cache_public_cases(
            Path("work/official_triton_agent/datasets")
        )
        code = (
            "import torch\n"
            "def _quantize_k_cache_fast(k_nope, k_rope, group_size):\n"
            " return torch.zeros((4, 592), dtype=k_rope.dtype)\n"
        )
        item = replace(
            candidate(code),
            op_name="_quantize_k_cache_fast_kernel",
        )
        worker_results = []

        def encoded(tensor):
            tensor = tensor.detach().cpu().contiguous()
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "values": tensor.reshape(-1).tolist(),
            }

        def candidate_runner(item_candidate, item_case, inputs):
            result = run_candidate(
                CandidateRunRequest(
                    item_candidate.id,
                    item_candidate.code,
                    item_candidate.code_hash,
                    item_case.execution.entrypoint,
                    {
                        "device": "cpu",
                        "k_nope": encoded(inputs.tensors["k_nope"]),
                        "k_rope": encoded(inputs.tensors["k_rope"]),
                        "group_size": inputs.scalars["group_size"],
                    },
                    timeout_seconds=10.0,
                    max_output_bytes=128 * 1024,
                    adapter_id=QUANTIZE_K_CACHE_ADAPTER_ID,
                )
            )
            worker_results.append(result)
            return result

        run = evaluate_candidate_correctness(
            item,
            cases,
            TorchTensorBackend("cpu"),
            run_case_reference,
            candidate_runner,
            decode_invocation_snapshot,
        )

        self.assertEqual(
            [result.oracle_status for result in run.results],
            ["failed", "passed"],
        )
        self.assertEqual(run.results[0].error_summary.compared_count, 256)
        self.assertEqual(run.decision.blocking_reasons, ("failed_case",))
        self.assertFalse(run.decision.eligible_for_performance)
        self.assertEqual(len(worker_results), 2)
        for result in worker_results:
            self.assertEqual(result.status, "completed")
            self.assertFalse(result.worker_result.stdout.truncated)
            self.assertFalse(Path(result.worker_result.working_directory).exists())

    @unittest.skipUnless(_CUDA_AVAILABLE, "quantize isolated smoke requires CUDA")
    def test_quantize_cuda_parent_passes_and_wrong_candidate_fails(self):
        datasets = Path(
            os.environ.get(
                "WLZ_CORRECTNESS_DATASETS_DIR",
                "work/official_triton_agent/datasets",
            )
        )
        configured_path = os.environ.get("WLZ_QUANTIZE_K_CACHE_CANDIDATE")
        candidate_path = Path(configured_path) if configured_path else (
            datasets
            / "_quantize_k_cache_fast_kernel/_quantize_k_cache_fast_kernel_1.py"
        )
        code = candidate_path.resolve().read_text(encoding="utf-8")
        cases = materialize_quantize_k_cache_public_cases(datasets)

        def evaluate(item_code, item_id):
            item = replace(
                candidate(item_code),
                id=item_id,
                op_name="_quantize_k_cache_fast_kernel",
            )
            return evaluate_candidate_correctness(
                item,
                cases,
                TorchTensorBackend("cuda"),
                run_case_reference,
                lambda value, case, inputs: run_quantize_k_cache_candidate(
                    value, case, inputs
                ),
                decode_invocation_snapshot,
            )

        admitted = evaluate(code, candidate_path.stem)
        self.assertEqual(
            [result.oracle_status for result in admitted.results],
            ["passed", "passed"],
        )
        self.assertTrue(admitted.decision.eligible_for_performance)
        self.assertEqual(admitted.decision.blocking_reasons, ())

        if configured_path is None:
            wrong_code = (
                "import torch\n"
                "def _quantize_k_cache_fast(k_nope, k_rope, group_size=128):\n"
                " return torch.zeros((4, 592), dtype=torch.bfloat16, "
                "device=k_nope.device)\n"
            )
            rejected = evaluate(wrong_code, "known-wrong-quantize")
            self.assertEqual(
                [result.oracle_status for result in rejected.results],
                ["failed", "passed"],
            )
            self.assertFalse(rejected.decision.eligible_for_performance)
            self.assertEqual(rejected.decision.blocking_reasons, ("failed_case",))

        print("B2_LOCAL_QUANTIZE_K_CACHE_MATRIX")

    def test_materialization_and_decode_errors_fail_closed(self):
        runners = isolated_runners({"invalid": "snapshot"})
        candidate_decode = evaluate_candidate_correctness(
            candidate(), [evaluation_case()], RecordingBackend(), *runners, decode_snapshot
        )

        class FailingBackend(RecordingBackend):
            def zeros(self, *, shape, dtype):
                raise RuntimeError("allocation failed")

        materialization = evaluate_candidate_correctness(
            candidate(), [evaluation_case()], FailingBackend(), *runners, decode_snapshot
        )

        self.assertEqual(candidate_decode.results[0].oracle_status, "failed")
        self.assertEqual(candidate_decode.results[0].error_summary.mismatch_kind, "candidate_snapshot")
        self.assertEqual(materialization.results[0].oracle_status, "oracle_error")
        self.assertFalse(materialization.decision.eligible_for_performance)

        with self.assertRaisesRegex(TypeError, "Candidate"):
            evaluate_candidate_correctness(
                object(), [], RecordingBackend(), *runners, decode_snapshot
            )


if __name__ == "__main__":
    unittest.main()
