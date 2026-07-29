"""Staged candidate execution through the isolated C1 worker protocol."""

from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import keyword
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

if __package__:
    from .correctness_worker import WorkerRequest, WorkerResult, run_worker
    from .hash_utils import sha256_text
    from .input_materializer import MaterializedInputs
    from .schemas import Candidate, EvaluationCase
else:  # Invoked by absolute path from the worker's temporary cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wlz_optimizer.correctness_worker import WorkerRequest, WorkerResult, run_worker
    from wlz_optimizer.hash_utils import sha256_text
    from wlz_optimizer.input_materializer import MaterializedInputs
    from wlz_optimizer.schemas import Candidate, EvaluationCase


COMPILE_HOOK_NAME = "__wlz_compile_hook__"
MIN_PROTOCOL_BYTES = 2048
MAX_CANDIDATE_SNAPSHOT_BYTES = 128 * 1024
ACT_QUANT_ADAPTER_ID = "act_quant_snapshot.v1"
COUNT_EXPERT_ADAPTER_ID = "count_expert_snapshot.v1"
QUANTIZE_K_CACHE_ADAPTER_ID = "quantize_k_cache_snapshot.v1"
PER_GROUP_TRANSPOSE_ADAPTER_ID = "per_group_transpose_snapshot.v1"
PACK_SEQ_ADAPTER_ID = "pack_seq_snapshot.v1"
SET_K_AND_S_ADAPTER_ID = "set_k_and_s_snapshot.v1"
_ADAPTER_IDS = {
    ACT_QUANT_ADAPTER_ID,
    COUNT_EXPERT_ADAPTER_ID,
    QUANTIZE_K_CACHE_ADAPTER_ID,
    PER_GROUP_TRANSPOSE_ADAPTER_ID,
    PACK_SEQ_ADAPTER_ID,
    SET_K_AND_S_ADAPTER_ID,
}
_STATUS_PHASES = {
    "hash_mismatch": {"preflight"},
    "python_source_error": {"python_source_compile"},
    "import_error": {"module_import"},
    "entrypoint_error": {"entrypoint_resolution"},
    "local_compile_hook_error": {"local_compile_hook"},
    "runtime_error": {"runtime"},
    "timeout": {"candidate_process"},
    "imported": {"module_import"},
    "completed": {"runtime"},
    "worker_error": {"protocol", "worker"},
    "not_configured": {"backend"},
}


class _AdapterProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateRunRequest:
    candidate_id: str
    code: str
    code_hash: str
    entrypoint: str
    payload: Any
    timeout_seconds: float
    max_output_bytes: int = 64 * 1024
    stop_after_import: bool = False
    adapter_id: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id.strip()
            or len(self.candidate_id.encode("utf-8")) > 256
        ):
            raise ValueError("candidate_id must be a non-empty string of at most 256 bytes")
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("candidate code must be a non-empty string")
        if not _is_sha256(self.code_hash):
            raise ValueError("candidate code_hash must be lowercase SHA-256")
        if (
            not isinstance(self.entrypoint, str)
            or not self.entrypoint.isidentifier()
            or keyword.iskeyword(self.entrypoint)
        ):
            raise ValueError("candidate entrypoint must be a Python identifier")
        try:
            json.dumps(self.payload, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate payload must be finite JSON data") from exc
        WorkerRequest(
            argv=(sys.executable,),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
        if self.max_output_bytes < MIN_PROTOCOL_BYTES:
            raise ValueError(f"candidate max_output_bytes must be at least {MIN_PROTOCOL_BYTES}")
        if not isinstance(self.stop_after_import, bool):
            raise ValueError("candidate stop_after_import must be a boolean")
        if self.adapter_id is not None and self.adapter_id not in _ADAPTER_IDS:
            raise ValueError(f"candidate adapter_id is not registered: {self.adapter_id}")


@dataclass(frozen=True)
class CandidateRunResult:
    candidate_id: str
    status: str
    phase: str
    value: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    worker_result: Optional[WorkerResult] = None

    def __post_init__(self) -> None:
        if self.status not in _STATUS_PHASES or self.phase not in _STATUS_PHASES[self.status]:
            raise ValueError(f"unsupported candidate status/phase: {self.status}/{self.phase}")


BackendExecutor = Callable[
    [Candidate, EvaluationCase, MaterializedInputs, str], Any
]
ActQuantBackendExecutor = BackendExecutor
PerGroupTransposeBackendExecutor = BackendExecutor
QuantizeKCacheBackendExecutor = BackendExecutor
SetKAndSBackendExecutor = BackendExecutor


def _run_injected_backend(
    candidate: Candidate,
    case: EvaluationCase,
    inputs: MaterializedInputs,
    *,
    torch_module: Any,
    backend_executor: Optional[BackendExecutor],
    executor_name: str,
    bounded_snapshot: bool = False,
) -> CandidateRunResult:
    try:
        torch = torch_module or importlib.import_module("torch")
        device = _available_torch_device(torch)
    except Exception as exc:
        return CandidateRunResult(
            candidate.id,
            "not_configured",
            "backend",
            error_type=type(exc).__name__,
            error_message=f"Torch candidate backend unavailable: {exc}",
        )
    if device is None:
        return CandidateRunResult(
            candidate.id,
            "not_configured",
            "backend",
            error_type="CandidateBackendNotConfigured",
            error_message="Neither an Ascend NPU nor a compatible CUDA device is available.",
        )
    if backend_executor is None:
        return CandidateRunResult(
            candidate.id,
            "not_configured",
            "backend",
            error_type="CandidateBackendExecutorMissing",
            error_message=f"No {executor_name} candidate executor is configured for {device}.",
        )
    try:
        value = backend_executor(candidate, case, inputs, device)
    except Exception as exc:
        return CandidateRunResult(
            candidate.id,
            "runtime_error",
            "runtime",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return (
        _bounded_snapshot_result(candidate.id, value)
        if bounded_snapshot
        else CandidateRunResult(candidate.id, "completed", "runtime", value=value)
    )


def run_act_quant_candidate(
    candidate: Candidate,
    case: EvaluationCase,
    inputs: MaterializedInputs,
    *,
    torch_module: Any = None,
    backend_executor: Optional[ActQuantBackendExecutor] = None,
    timeout_seconds: float = 20.0,
) -> CandidateRunResult:
    """Run act-quant in an isolated fixed adapter or an injected test backend."""

    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    if (
        not isinstance(case, EvaluationCase)
        or case.op_name != "_act_quant_kernel"
        or candidate.op_name != case.op_name
    ):
        raise ValueError("candidate and case must target act-quant")
    if not isinstance(inputs, MaterializedInputs):
        raise TypeError("inputs must be MaterializedInputs")
    if inputs.input_signature != case.input_signature():
        raise ValueError("materialized inputs do not match the evaluation case")

    if backend_executor is None:
        try:
            torch = torch_module or importlib.import_module("torch")
            device = _available_torch_device(torch)
        except Exception as exc:
            return CandidateRunResult(
                candidate.id,
                "not_configured",
                "backend",
                error_type=type(exc).__name__,
                error_message=f"Torch candidate backend unavailable: {exc}",
            )
        if device is None:
            return CandidateRunResult(
                candidate.id,
                "not_configured",
                "backend",
                error_type="CandidateBackendNotConfigured",
                error_message="Neither an Ascend NPU nor a compatible CUDA device is available.",
            )
        try:
            payload = {
                "device": device,
                "x": _tensor_payload(inputs.tensors["x"]),
                "block_size": inputs.scalars["block_size"],
                "scale_fmt": inputs.scalars.get("scale_fmt"),
            }
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return CandidateRunResult(
                candidate.id,
                "not_configured",
                "backend",
                error_type="CandidateBackendInputsUnavailable",
                error_message=f"Cannot serialize act-quant inputs: {exc}",
            )
        return run_candidate(
            CandidateRunRequest(
                candidate.id,
                candidate.code,
                candidate.code_hash,
                case.execution.entrypoint,
                payload,
                timeout_seconds=timeout_seconds,
                max_output_bytes=MAX_CANDIDATE_SNAPSHOT_BYTES,
                adapter_id=ACT_QUANT_ADAPTER_ID,
            )
        )
    return _run_injected_backend(
        candidate,
        case,
        inputs,
        torch_module=torch_module,
        backend_executor=backend_executor,
        executor_name="act-quant",
    )


def run_count_expert_candidate(
    candidate: Candidate,
    case: EvaluationCase,
    inputs: MaterializedInputs,
    *,
    torch_module: Any = None,
    timeout_seconds: float = 20.0,
) -> CandidateRunResult:
    """Run the basic/no-map count-expert case in an isolated fixed adapter."""

    if (
        not isinstance(candidate, Candidate)
        or not isinstance(case, EvaluationCase)
        or candidate.op_name != "_count_expert_num_tokens"
        or case.op_name != candidate.op_name
    ):
        raise ValueError("candidate and case must target count-expert")
    if not isinstance(inputs, MaterializedInputs):
        raise TypeError("inputs must be MaterializedInputs")
    if inputs.input_signature != case.input_signature():
        raise ValueError("materialized inputs do not match the evaluation case")
    try:
        torch = torch_module or importlib.import_module("torch")
        device = _available_torch_device(torch)
    except Exception as exc:
        return CandidateRunResult(
            candidate.id, "not_configured", "backend",
            error_type=type(exc).__name__,
            error_message=f"Torch candidate backend unavailable: {exc}",
        )
    if device is None:
        return CandidateRunResult(
            candidate.id, "not_configured", "backend",
            error_type="CandidateBackendNotConfigured",
            error_message="Neither an Ascend NPU nor a compatible CUDA device is available.",
        )
    try:
        payload = {
            "device": device,
            "topk_ids": _tensor_payload(inputs.tensors["topk_ids"]),
            "num_local_experts": inputs.scalars["num_local_experts"],
            "expert_map": inputs.scalars["expert_map"],
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return CandidateRunResult(
            candidate.id, "not_configured", "backend",
            error_type="CandidateBackendInputsUnavailable",
            error_message=f"Cannot serialize count-expert inputs: {exc}",
        )
    return run_candidate(
        CandidateRunRequest(
            candidate.id,
            candidate.code,
            candidate.code_hash,
            case.execution.entrypoint,
            payload,
            timeout_seconds=timeout_seconds,
            max_output_bytes=MAX_CANDIDATE_SNAPSHOT_BYTES,
            adapter_id=COUNT_EXPERT_ADAPTER_ID,
        )
    )


def run_per_group_transpose_candidate(
    candidate: Candidate,
    case: EvaluationCase,
    inputs: MaterializedInputs,
    *,
    torch_module: Any = None,
    backend_executor: Optional[PerGroupTransposeBackendExecutor] = None,
    timeout_seconds: float = 20.0,
) -> CandidateRunResult:
    """Dispatch only to an explicitly available NPU or CUDA executor."""

    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    if (
        not isinstance(case, EvaluationCase)
        or case.op_name != "_per_group_transpose"
        or candidate.op_name != case.op_name
    ):
        raise ValueError("candidate and case must target per-group-transpose")
    if not isinstance(inputs, MaterializedInputs):
        raise TypeError("inputs must be MaterializedInputs")
    if inputs.input_signature != case.input_signature():
        raise ValueError("materialized inputs do not match the evaluation case")
    if backend_executor is None:
        try:
            torch = torch_module or importlib.import_module("torch")
            device = _available_torch_device(torch)
        except Exception as exc:
            return CandidateRunResult(
                candidate.id,
                "not_configured",
                "backend",
                error_type=type(exc).__name__,
                error_message=f"Torch candidate backend unavailable: {exc}",
            )
        if device is None:
            return CandidateRunResult(
                candidate.id,
                "not_configured",
                "backend",
                error_type="CandidateBackendNotConfigured",
                error_message="Neither an Ascend NPU nor a compatible CUDA device is available.",
            )
        try:
            payload = {
                "device": device,
                "a": _tensor_payload(inputs.tensors["a"]),
                "expert_offsets": _tensor_payload(inputs.tensors["expert_offsets"]),
                "M_ALIGNMENT": inputs.scalars["M_ALIGNMENT"],
            }
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return CandidateRunResult(
                candidate.id,
                "not_configured",
                "backend",
                error_type="CandidateBackendInputsUnavailable",
                error_message=f"Cannot serialize per-group-transpose inputs: {exc}",
            )
        return run_candidate(
            CandidateRunRequest(
                candidate.id,
                candidate.code,
                candidate.code_hash,
                case.execution.entrypoint,
                payload,
                timeout_seconds=timeout_seconds,
                max_output_bytes=MAX_CANDIDATE_SNAPSHOT_BYTES,
                adapter_id=PER_GROUP_TRANSPOSE_ADAPTER_ID,
            )
        )
    return _run_injected_backend(
        candidate,
        case,
        inputs,
        torch_module=torch_module,
        backend_executor=backend_executor,
        executor_name="per-group-transpose",
        bounded_snapshot=True,
    )


def run_pack_seq_candidate(
    candidate: Candidate,
    case: EvaluationCase,
    inputs: MaterializedInputs,
    *,
    torch_module: Any = None,
    timeout_seconds: float = 20.0,
) -> CandidateRunResult:
    """Run the fixed public pack-seq case in the isolated candidate worker."""

    if (
        not isinstance(candidate, Candidate)
        or not isinstance(case, EvaluationCase)
        or candidate.op_name != "_pack_seq_kernel"
        or case.op_name != candidate.op_name
    ):
        raise ValueError("candidate and case must target pack-seq")
    if not isinstance(inputs, MaterializedInputs):
        raise TypeError("inputs must be MaterializedInputs")
    if inputs.input_signature != case.input_signature():
        raise ValueError("materialized inputs do not match the evaluation case")
    try:
        torch = torch_module or importlib.import_module("torch")
        device = _available_torch_device(torch)
    except Exception as exc:
        return CandidateRunResult(
            candidate.id, "not_configured", "backend",
            error_type=type(exc).__name__, error_message=f"Torch backend unavailable: {exc}",
        )
    if device is None:
        return CandidateRunResult(
            candidate.id, "not_configured", "backend",
            error_type="CandidateBackendNotConfigured",
            error_message="Neither an Ascend NPU nor a compatible CUDA device is available.",
        )
    try:
        x = inputs.tensors["x"].detach().cpu().contiguous()
        raw = bytes(x.view(torch.uint8).reshape(-1).tolist())
        payload = {
            "device": device,
            "x": {
                "shape": [int(size) for size in x.shape],
                "dtype": str(x.dtype),
                "data_base64": base64.urlsafe_b64encode(raw).decode("ascii"),
            },
            "lengths": _tensor_payload(inputs.tensors["lengths"]),
            "pad_value": inputs.scalars["pad_value"],
            "block_t": inputs.scalars["block_t"],
            "block_d": inputs.scalars["block_d"],
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return CandidateRunResult(
            candidate.id, "not_configured", "backend",
            error_type="CandidateBackendInputsUnavailable",
            error_message=f"Cannot serialize pack-seq inputs: {exc}",
        )
    return run_candidate(
        CandidateRunRequest(
            candidate.id,
            candidate.code,
            candidate.code_hash,
            case.execution.entrypoint,
            payload,
            timeout_seconds=timeout_seconds,
            max_output_bytes=MAX_CANDIDATE_SNAPSHOT_BYTES,
            adapter_id=PACK_SEQ_ADAPTER_ID,
        )
    )


def run_quantize_k_cache_candidate(
    candidate: Candidate,
    case: EvaluationCase,
    inputs: MaterializedInputs,
    *,
    torch_module: Any = None,
    backend_executor: Optional[QuantizeKCacheBackendExecutor] = None,
    timeout_seconds: float = 20.0,
) -> CandidateRunResult:
    """Run through an explicit backend and reject unbounded snapshots."""

    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    if (
        not isinstance(case, EvaluationCase)
        or case.op_name != "_quantize_k_cache_fast_kernel"
        or candidate.op_name != case.op_name
    ):
        raise ValueError("candidate and case must target quantize-k-cache")
    if not isinstance(inputs, MaterializedInputs):
        raise TypeError("inputs must be MaterializedInputs")
    if inputs.input_signature != case.input_signature():
        raise ValueError("materialized inputs do not match the evaluation case")
    try:
        torch = torch_module or importlib.import_module("torch")
        device = _available_torch_device(torch)
    except Exception as exc:
        return CandidateRunResult(
            candidate.id,
            "not_configured",
            "backend",
            error_type=type(exc).__name__,
            error_message=f"Torch candidate backend unavailable: {exc}",
        )
    if device is None:
        return CandidateRunResult(
            candidate.id,
            "not_configured",
            "backend",
            error_type="CandidateBackendNotConfigured",
            error_message="Neither an Ascend NPU nor a compatible CUDA device is available.",
        )
    if backend_executor is None:
        try:
            payload = {
                "device": device,
                "k_nope": _tensor_payload(inputs.tensors["k_nope"]),
                "k_rope": _tensor_payload(inputs.tensors["k_rope"]),
                "group_size": inputs.scalars["group_size"],
            }
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return CandidateRunResult(
                candidate.id,
                "not_configured",
                "backend",
                error_type="CandidateBackendInputsUnavailable",
                error_message=f"Cannot serialize quantize-k-cache inputs: {exc}",
            )
        return run_candidate(
            CandidateRunRequest(
                candidate.id,
                candidate.code,
                candidate.code_hash,
                case.execution.entrypoint,
                payload,
                timeout_seconds=timeout_seconds,
                max_output_bytes=MAX_CANDIDATE_SNAPSHOT_BYTES,
                adapter_id=QUANTIZE_K_CACHE_ADAPTER_ID,
            )
        )
    try:
        value = backend_executor(candidate, case, inputs, device)
    except Exception as exc:
        return CandidateRunResult(
            candidate.id,
            "runtime_error",
            "runtime",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return _bounded_snapshot_result(candidate.id, value)


def run_set_k_and_s_candidate(
    candidate: Candidate,
    case: EvaluationCase,
    inputs: MaterializedInputs,
    *,
    torch_module: Any = None,
    backend_executor: Optional[SetKAndSBackendExecutor] = None,
    timeout_seconds: float = 20.0,
) -> CandidateRunResult:
    """Run set-k-and-s with canaries around the public buffer allocation."""

    if not isinstance(candidate, Candidate):
        raise TypeError("candidate must be a Candidate")
    if (
        not isinstance(case, EvaluationCase)
        or case.op_name != "_set_k_and_s_triton_kernel"
        or candidate.op_name != case.op_name
    ):
        raise ValueError("candidate and case must target set-k-and-s")
    if not isinstance(inputs, MaterializedInputs):
        raise TypeError("inputs must be MaterializedInputs")
    if inputs.input_signature != case.input_signature():
        raise ValueError("materialized inputs do not match the evaluation case")
    if backend_executor is not None:
        return _run_injected_backend(
            candidate,
            case,
            inputs,
            torch_module=torch_module,
            backend_executor=backend_executor,
            executor_name="set-k-and-s",
            bounded_snapshot=True,
        )
    try:
        torch = torch_module or importlib.import_module("torch")
        device = _available_torch_device(torch)
    except Exception as exc:
        return CandidateRunResult(
            candidate.id,
            "not_configured",
            "backend",
            error_type=type(exc).__name__,
            error_message=f"Torch candidate backend unavailable: {exc}",
        )
    if device is None:
        return CandidateRunResult(
            candidate.id,
            "not_configured",
            "backend",
            error_type="CandidateBackendNotConfigured",
            error_message="Neither an Ascend NPU nor a compatible CUDA device is available.",
        )
    try:
        encoded_buf = inputs.tensors["buf"].detach().cpu().contiguous()
        payload = {
            "device": device,
            "buf": {
                "shape": [int(size) for size in encoded_buf.shape],
                "dtype": str(encoded_buf.dtype),
                "data_base64": base64.urlsafe_b64encode(
                    bytes(encoded_buf.reshape(-1).tolist())
                ).decode("ascii"),
            },
            "loc": _tensor_payload(inputs.tensors["loc"]),
            "index_k": _tensor_payload(inputs.tensors["index_k"]),
            "index_k_scale": _tensor_payload(inputs.tensors["index_k_scale"]),
            "page_size": inputs.scalars["page_size"],
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return CandidateRunResult(
            candidate.id,
            "not_configured",
            "backend",
            error_type="CandidateBackendInputsUnavailable",
            error_message=f"Cannot serialize set-k-and-s inputs: {exc}",
        )
    return run_candidate(
        CandidateRunRequest(
            candidate.id,
            candidate.code,
            candidate.code_hash,
            case.execution.entrypoint,
            payload,
            timeout_seconds=timeout_seconds,
            max_output_bytes=MAX_CANDIDATE_SNAPSHOT_BYTES,
            adapter_id=SET_K_AND_S_ADAPTER_ID,
        )
    )


def _bounded_snapshot_result(candidate_id: str, value: Any) -> CandidateRunResult:
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return CandidateRunResult(
            candidate_id,
            "worker_error",
            "protocol",
            error_type=type(exc).__name__,
            error_message="candidate snapshot must be finite JSON data",
        )
    if len(encoded) > MAX_CANDIDATE_SNAPSHOT_BYTES:
        return CandidateRunResult(
            candidate_id,
            "worker_error",
            "protocol",
            error_type="CandidateOutputTooLarge",
            error_message="candidate snapshot exceeds 128 KiB",
        )
    return CandidateRunResult(candidate_id, "completed", "runtime", value=value)


def _tensor_payload(tensor: Any) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    return {
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype),
        "values": tensor.reshape(-1).tolist(),
    }


def _available_torch_device(torch: Any) -> Optional[str]:
    npu = getattr(torch, "npu", None)
    if npu is not None and callable(getattr(npu, "is_available", None)):
        if npu.is_available():
            return "npu"
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        if cuda.is_available():
            return "cuda"
    return None


def run_candidate(request: CandidateRunRequest) -> CandidateRunResult:
    """Validate identity, then run fixed file-import phases in one worker."""

    if not isinstance(request, CandidateRunRequest):
        raise TypeError("request must be a CandidateRunRequest")
    if sha256_text(request.code) != request.code_hash:
        return CandidateRunResult(
            request.candidate_id, "hash_mismatch", "preflight",
            error_type="CodeHashMismatch",
            error_message="candidate source does not match code_hash",
        )

    worker = run_worker(
        WorkerRequest(
            argv=(
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                _encode(request.candidate_id),
                request.code_hash,
                request.entrypoint,
                _encode(request.code),
                _encode(request.payload),
                _encode(request.stop_after_import),
                _encode(request.adapter_id),
            ),
            timeout_seconds=request.timeout_seconds,
            max_output_bytes=request.max_output_bytes,
        )
    )
    if worker.status == "timeout":
        return CandidateRunResult(
            request.candidate_id, "timeout", "candidate_process", worker_result=worker
        )
    if worker.status != "completed" or worker.returncode != 0:
        return _worker_error(request.candidate_id, worker, "worker")
    if worker.stdout.truncated:
        return _worker_error(request.candidate_id, worker, "protocol")
    try:
        message = json.loads(worker.stdout.text)
    except (json.JSONDecodeError, TypeError) as exc:
        return _worker_error(request.candidate_id, worker, "protocol", str(exc), type(exc).__name__)
    if not isinstance(message, dict) or message.get("candidate_id") != request.candidate_id:
        return _worker_error(request.candidate_id, worker, "protocol")
    try:
        return CandidateRunResult(
            candidate_id=request.candidate_id,
            status=message.get("status"),
            phase=message.get("phase"),
            value=message.get("value"),
            error_type=message.get("error_type"),
            error_message=message.get("error_message"),
            worker_result=worker,
        )
    except ValueError:
        return _worker_error(request.candidate_id, worker, "protocol")


def _worker_main(
    candidate_data: str,
    code_hash: str,
    entrypoint_name: str,
    code_data: str,
    payload_data: str,
    stop_after_import_data: str,
    adapter_id_data: str,
) -> int:
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        candidate_id = _decode(candidate_data)
        source = _decode(code_data)
        payload = _decode(payload_data)
        stop_after_import = _decode(stop_after_import_data)
        adapter_id = _decode(adapter_id_data)
    except Exception as exc:
        return _write_error(protocol_fd, "unknown", "worker_error", "protocol", exc)

    candidate_path = Path.cwd() / "candidate.py"
    try:
        candidate_path.write_bytes(source.encode("utf-8"))
        written_source = candidate_path.read_bytes().decode("utf-8")
    except Exception as exc:
        return _write_error(protocol_fd, candidate_id, "worker_error", "worker", exc)
    if sha256_text(written_source) != code_hash:
        return _write_error(
            protocol_fd, candidate_id, "hash_mismatch", "preflight",
            ValueError("candidate.py hash mismatch after write"),
        )
    try:
        compile(written_source, str(candidate_path), "exec")
    except BaseException as exc:
        return _write_error(
            protocol_fd, candidate_id, "python_source_error", "python_source_compile", exc
        )

    try:
        spec = importlib.util.spec_from_file_location("_wlz_candidate", candidate_path)
        if spec is None or spec.loader is None:
            raise ImportError("unable to create fixed candidate module spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except BaseException as exc:
        return _write_error(protocol_fd, candidate_id, "import_error", "module_import", exc)

    if stop_after_import:
        return _write_message(protocol_fd, {
            "candidate_id": candidate_id,
            "status": "imported",
            "phase": "module_import",
            "value": None,
        })

    entrypoint = module.__dict__.get(entrypoint_name)
    if not callable(entrypoint):
        return _write_error(
            protocol_fd, candidate_id, "entrypoint_error", "entrypoint_resolution",
            TypeError(f"entrypoint is missing or not callable: {entrypoint_name}"),
        )
    hook = module.__dict__.get(COMPILE_HOOK_NAME)
    if hook is not None:
        if not callable(hook):
            return _write_error(
                protocol_fd, candidate_id, "local_compile_hook_error", "local_compile_hook",
                TypeError("local compile hook is not callable"),
            )
        try:
            hook(_decode(_encode(payload)))
        except BaseException as exc:
            return _write_error(
                protocol_fd, candidate_id, "local_compile_hook_error", "local_compile_hook", exc
            )
    try:
        value = (
            entrypoint(payload)
            if adapter_id is None
            else _run_adapter(adapter_id, entrypoint, payload)
        )
    except _AdapterProtocolError as exc:
        return _write_error(protocol_fd, candidate_id, "worker_error", "protocol", exc)
    except BaseException as exc:
        return _write_error(protocol_fd, candidate_id, "runtime_error", "runtime", exc)
    return _write_message(
        protocol_fd,
        {"candidate_id": candidate_id, "status": "completed", "phase": "runtime", "value": value},
    )


def _run_adapter(adapter_id: str, entrypoint: Callable[..., Any], payload: Any) -> Any:
    if adapter_id == ACT_QUANT_ADAPTER_ID:
        return _run_act_quant_adapter(entrypoint, payload)
    if adapter_id == COUNT_EXPERT_ADAPTER_ID:
        return _run_count_expert_adapter(entrypoint, payload)
    if adapter_id == PER_GROUP_TRANSPOSE_ADAPTER_ID:
        return _run_per_group_transpose_adapter(entrypoint, payload)
    if adapter_id == PACK_SEQ_ADAPTER_ID:
        return _run_pack_seq_adapter(entrypoint, payload)
    if adapter_id == SET_K_AND_S_ADAPTER_ID:
        return _run_set_k_and_s_adapter(entrypoint, payload)
    if adapter_id != QUANTIZE_K_CACHE_ADAPTER_ID:
        raise _AdapterProtocolError(f"candidate adapter is not registered: {adapter_id}")
    if not isinstance(payload, dict) or set(payload) != {
        "device",
        "k_nope",
        "k_rope",
        "group_size",
    }:
        raise _AdapterProtocolError("quantize-k-cache adapter payload is invalid")
    if payload["device"] not in {"cpu", "cuda", "npu"} or payload["group_size"] != 128:
        raise _AdapterProtocolError("quantize-k-cache device or group_size is invalid")
    torch = importlib.import_module("torch")

    def materialize(name: str, shape: list[int]) -> Any:
        encoded = payload[name]
        if (
            not isinstance(encoded, dict)
            or set(encoded) != {"shape", "dtype", "values"}
            or encoded["shape"] != shape
            or encoded["dtype"] != "torch.bfloat16"
            or not isinstance(encoded["values"], list)
            or len(encoded["values"]) != shape[0] * shape[1]
        ):
            raise _AdapterProtocolError(f"quantize-k-cache {name} payload is invalid")
        return torch.tensor(
            encoded["values"], dtype=torch.bfloat16, device=payload["device"]
        ).reshape(shape)

    output = entrypoint(
        materialize("k_nope", [4, 512]),
        materialize("k_rope", [4, 64]),
        payload["group_size"],
    )
    if output.numel() > 4096:
        raise _AdapterProtocolError(
            "quantize-k-cache output exceeds the snapshot numel limit"
        )
    device = output.device.type
    synchronize = getattr(getattr(torch, device, None), "synchronize", None)
    if callable(synchronize):
        synchronize()
    return {
        "return": {**_tensor_payload(output), "device": device},
        "tensors": {},
    }


def _run_act_quant_adapter(entrypoint: Callable[..., Any], payload: Any) -> Any:
    expected = {"device", "x", "block_size", "scale_fmt"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise _AdapterProtocolError("act-quant adapter payload is invalid")
    if (
        payload["device"] not in {"cpu", "cuda", "npu"}
        or payload["block_size"] != 128
        or payload["scale_fmt"] not in {None, "round"}
    ):
        raise _AdapterProtocolError("act-quant device or scalars are invalid")
    encoded = payload["x"]
    if (
        not isinstance(encoded, dict)
        or set(encoded) != {"shape", "dtype", "values"}
        or encoded["shape"] != [2, 4, 256]
        or encoded["dtype"] != "torch.float32"
        or not isinstance(encoded["values"], list)
        or len(encoded["values"]) != 2 * 4 * 256
    ):
        raise _AdapterProtocolError("act-quant x payload is invalid")
    torch = importlib.import_module("torch")
    x = torch.tensor(
        encoded["values"], dtype=torch.float32, device=payload["device"]
    ).reshape(2, 4, 256)
    output = entrypoint(x, payload["block_size"], payload["scale_fmt"])
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise _AdapterProtocolError("act-quant output must contain y and scale")
    if any(not hasattr(value, "numel") for value in output):
        raise _AdapterProtocolError("act-quant outputs must be tensors")
    if sum(int(value.numel()) for value in output) > 4096:
        raise _AdapterProtocolError("act-quant output exceeds the snapshot numel limit")
    if any(value.device.type != payload["device"] for value in output):
        raise _AdapterProtocolError("act-quant output device is invalid")
    synchronize = getattr(getattr(torch, payload["device"], None), "synchronize", None)
    if callable(synchronize):
        synchronize()
    return {
        "return": [_tensor_payload(value) for value in output],
        "tensors": {},
    }


def _run_count_expert_adapter(entrypoint: Callable[..., Any], payload: Any) -> Any:
    expected = {"device", "topk_ids", "num_local_experts", "expert_map"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise _AdapterProtocolError("count-expert adapter payload is invalid")
    if (
        payload["device"] not in {"cpu", "cuda", "npu"}
        or payload["num_local_experts"] != 4
        or payload["expert_map"] is not None
    ):
        raise _AdapterProtocolError("count-expert device or scalars are invalid")
    encoded = payload["topk_ids"]
    expected_ids = [0, 1, 2, 0, 1, 3, 2, 0]
    if (
        not isinstance(encoded, dict)
        or set(encoded) != {"shape", "dtype", "values"}
        or encoded["shape"] != [8]
        or encoded["dtype"] != "torch.int32"
        or encoded["values"] != expected_ids
        or any(type(value) is not int for value in encoded["values"])
    ):
        raise _AdapterProtocolError("count-expert topk_ids payload is invalid")
    torch = importlib.import_module("torch")
    topk_ids = torch.tensor(
        expected_ids, dtype=torch.int32, device=payload["device"]
    )
    output = entrypoint(topk_ids, payload["num_local_experts"], None)
    if (
        not hasattr(output, "numel")
        or list(output.shape) != [4]
        or output.dtype != torch.int32
        or output.device.type != payload["device"]
    ):
        raise _AdapterProtocolError("count-expert output contract is invalid")
    synchronize = getattr(getattr(torch, payload["device"], None), "synchronize", None)
    if callable(synchronize):
        synchronize()
    return {
        "return": {**_tensor_payload(output), "device": payload["device"]},
        "tensors": {},
    }


def _run_set_k_and_s_adapter(entrypoint: Callable[..., Any], payload: Any) -> Any:
    expected = {
        "device",
        "buf",
        "loc",
        "index_k",
        "index_k_scale",
        "page_size",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise _AdapterProtocolError("set-k-and-s adapter payload is invalid")
    if payload["device"] not in {"cpu", "cuda", "npu"} or payload["page_size"] != 64:
        raise _AdapterProtocolError("set-k-and-s device or page_size is invalid")
    torch = importlib.import_module("torch")

    def materialize(name: str, shape: list[int], dtype_name: str) -> Any:
        encoded = payload[name]
        if (
            not isinstance(encoded, dict)
            or set(encoded) != {"shape", "dtype", "values"}
            or encoded["shape"] != shape
            or encoded["dtype"] != f"torch.{dtype_name}"
            or not isinstance(encoded["values"], list)
            or len(encoded["values"]) != math.prod(shape)
        ):
            raise _AdapterProtocolError(f"set-k-and-s {name} payload is invalid")
        return torch.tensor(
            encoded["values"],
            dtype=getattr(torch, dtype_name),
            device=payload["device"],
        ).reshape(shape)

    encoded_buf = payload["buf"]
    buf_numel = 4 * 8448
    if (
        not isinstance(encoded_buf, dict)
        or set(encoded_buf) != {"shape", "dtype", "data_base64"}
        or encoded_buf["shape"] != [4, 8448]
        or encoded_buf["dtype"] != "torch.uint8"
        or not isinstance(encoded_buf["data_base64"], str)
    ):
        raise _AdapterProtocolError("set-k-and-s buf payload is invalid")
    try:
        raw_buf = base64.b64decode(
            encoded_buf["data_base64"], altchars=b"-_", validate=True
        )
    except (ValueError, TypeError) as exc:
        raise _AdapterProtocolError("set-k-and-s buf bytes are invalid") from exc
    if len(raw_buf) != buf_numel:
        raise _AdapterProtocolError("set-k-and-s buf byte length is invalid")
    guard_size = 512
    guard_value = 0xA5
    storage = torch.full(
        (buf_numel + 2 * guard_size,),
        guard_value,
        dtype=torch.uint8,
        device=payload["device"],
    )
    buf = storage[guard_size : guard_size + buf_numel].view(4, 8448)
    decoded_buf = torch.frombuffer(bytearray(raw_buf), dtype=torch.uint8).reshape(4, 8448)
    buf.copy_(decoded_buf.to(payload["device"]))
    entrypoint(
        buf,
        materialize("loc", [3], "int64"),
        materialize("index_k", [3, 128], "float16"),
        materialize("index_k_scale", [3, 1], "float32"),
        payload["page_size"],
    )
    synchronize = getattr(getattr(torch, payload["device"], None), "synchronize", None)
    if callable(synchronize):
        synchronize()
    prefix_ok = bool(torch.all(storage[:guard_size] == guard_value).item())
    suffix_ok = bool(torch.all(storage[-guard_size:] == guard_value).item())
    if not prefix_ok or not suffix_ok:
        raise _AdapterProtocolError("set-k-and-s candidate modified an allocation guard")
    return {"return": None, "tensors": {"buf": _tensor_payload(buf)}}


def _run_pack_seq_adapter(entrypoint: Callable[..., Any], payload: Any) -> Any:
    expected = {"device", "x", "lengths", "pad_value", "block_t", "block_d"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise _AdapterProtocolError("pack-seq adapter payload is invalid")
    if (
        payload["device"] not in {"cuda", "npu"}
        or payload["pad_value"] != 0.0
        or payload["block_t"] != 32
        or payload["block_d"] != 32
    ):
        raise _AdapterProtocolError("pack-seq device or scalars are invalid")
    encoded = payload["x"]
    if (
        not isinstance(encoded, dict)
        or set(encoded) != {"shape", "dtype", "data_base64"}
        or encoded["shape"] != [4096, 4]
        or encoded["dtype"] != "torch.float32"
        or not isinstance(encoded["data_base64"], str)
    ):
        raise _AdapterProtocolError("pack-seq x payload is invalid")
    try:
        raw = base64.b64decode(encoded["data_base64"], altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise _AdapterProtocolError("pack-seq x bytes are invalid") from exc
    if len(raw) != 4096 * 4 * 4:
        raise _AdapterProtocolError("pack-seq x byte length is invalid")
    lengths = payload["lengths"]
    if (
        not isinstance(lengths, dict)
        or lengths.get("shape") != [3]
        or lengths.get("dtype") != "torch.int32"
        or lengths.get("values") != [3, 4, 3]
    ):
        raise _AdapterProtocolError("pack-seq lengths payload is invalid")
    torch = importlib.import_module("torch")
    x = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(4096, 4)
    x = x.to(payload["device"])
    lengths_tensor = torch.tensor([3, 4, 3], dtype=torch.int32, device=payload["device"])
    output = entrypoint(x, lengths_tensor, 0.0, 32, 32)
    if not hasattr(output, "numel") or output.numel() > 4096:
        raise _AdapterProtocolError("pack-seq output exceeds snapshot numel limit")
    device = getattr(getattr(output, "device", None), "type", None)
    if device != payload["device"]:
        raise _AdapterProtocolError("pack-seq output device is invalid")
    synchronize = getattr(getattr(torch, device, None), "synchronize", None)
    if callable(synchronize):
        synchronize()
    return {"return": {**_tensor_payload(output), "device": device}, "tensors": {}}


def _run_per_group_transpose_adapter(
    entrypoint: Callable[..., Any], payload: Any
) -> Any:
    if not isinstance(payload, dict) or set(payload) != {
        "device",
        "a",
        "expert_offsets",
        "M_ALIGNMENT",
    }:
        raise _AdapterProtocolError("per-group-transpose adapter payload is invalid")
    if (
        not isinstance(payload["device"], str)
        or payload["device"] not in {"cpu", "cuda", "npu"}
        or type(payload["M_ALIGNMENT"]) is not int
        or payload["M_ALIGNMENT"] != 1
    ):
        raise _AdapterProtocolError("per-group-transpose device or alignment is invalid")
    torch = importlib.import_module("torch")

    def materialize(
        name: str,
        shape: list[int],
        dtype: str,
        exact_values: Optional[list[int]] = None,
    ) -> Any:
        encoded = payload[name]
        if (
            not isinstance(encoded, dict)
            or set(encoded) != {"shape", "dtype", "values"}
            or not isinstance(encoded["shape"], list)
            or any(type(size) is not int for size in encoded["shape"])
            or encoded["shape"] != shape
            or encoded["dtype"] != dtype
            or not isinstance(encoded["values"], list)
            or len(encoded["values"]) != math.prod(shape)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in encoded["values"]
            )
            or (
                exact_values is not None
                and (
                    any(type(value) is not int for value in encoded["values"])
                    or encoded["values"] != exact_values
                )
            )
        ):
            raise _AdapterProtocolError(f"per-group-transpose {name} payload is invalid")
        return torch.tensor(
            encoded["values"],
            dtype=getattr(torch, dtype.rsplit(".", 1)[-1]),
            device=payload["device"],
        ).reshape(shape)

    a = materialize("a", [32, 16], "torch.float32")
    expert_offsets = materialize(
        "expert_offsets", [3], "torch.int32", exact_values=[0, 16, 32]
    )
    output = entrypoint(a, expert_offsets, payload["M_ALIGNMENT"])
    if not hasattr(output, "numel") or output.numel() > 4096:
        raise _AdapterProtocolError("per-group-transpose output exceeds snapshot numel limit")
    device = getattr(getattr(output, "device", None), "type", None)
    if device != payload["device"]:
        raise _AdapterProtocolError("per-group-transpose output device is invalid")
    synchronize = getattr(getattr(torch, device, None), "synchronize", None)
    if callable(synchronize):
        synchronize()
    return {
        "return": {**_tensor_payload(output), "device": device},
        "tensors": {},
    }


def _encode(value: Any) -> str:
    data = json.dumps(value, allow_nan=False, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(data).decode("ascii")


def _decode(value: str) -> Any:
    return json.loads(base64.urlsafe_b64decode(value).decode("utf-8"))


def _write_error(
    fd: int, candidate_id: str, status: str, phase: str, error: BaseException
) -> int:
    return _write_message(fd, {
        "candidate_id": candidate_id,
        "status": status,
        "phase": phase,
        "error_type": type(error).__name__,
        "error_message": _bounded_message(str(error)),
    })


def _write_message(fd: int, message: dict[str, Any]) -> int:
    try:
        encoded = json.dumps(message, allow_nan=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        encoded = json.dumps({
            "candidate_id": message.get("candidate_id", "unknown"),
            "status": "worker_error",
            "phase": "protocol",
            "error_type": type(exc).__name__,
            "error_message": "candidate result is not finite JSON data",
        }, sort_keys=True).encode("utf-8")
    try:
        while encoded:
            encoded = encoded[os.write(fd, encoded):]
        return 0
    finally:
        os.close(fd)


def _bounded_message(message: str, limit: int = 512) -> str:
    return message.encode("utf-8", errors="replace")[:limit].decode("utf-8", errors="ignore")


def _worker_error(
    candidate_id: str,
    worker: WorkerResult,
    phase: str,
    message: str = "candidate worker did not return a complete protocol result",
    error_type: str = "WorkerProtocolError",
) -> CandidateRunResult:
    return CandidateRunResult(
        candidate_id, "worker_error", phase,
        error_type=error_type, error_message=_bounded_message(message), worker_result=worker,
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


if __name__ == "__main__":
    if len(sys.argv) != 9 or sys.argv[1] != "--worker":
        raise SystemExit(2)
    raise SystemExit(_worker_main(*sys.argv[2:]))
