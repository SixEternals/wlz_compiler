"""Trusted reference registry executed through the isolated worker protocol."""

from __future__ import annotations

import base64
import contextlib
import importlib
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

if __package__:
    from .case_catalog import (
        ACT_QUANT_DEFAULT_REFERENCE_ID,
        ACT_QUANT_ROUND_REFERENCE_ID,
        CHUNK_CUMSUM_REFERENCE_ID,
        COUNT_EXPERT_REFERENCE_ID,
        PACK_SEQ_REFERENCE_ID,
        PER_GROUP_TRANSPOSE_REFERENCE_ID,
        QUANTIZE_K_CACHE_METADATA_REFERENCE_ID,
        QUANTIZE_K_CACHE_ROPE_REFERENCE_ID,
        SET_K_AND_S_REFERENCE_ID,
    )
    from .correctness_worker import WorkerRequest, WorkerResult, run_worker
    from .input_materializer import MaterializedInputs
    from .schemas import EvaluationCase
else:  # Executed by absolute path from the worker's temporary cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    )
    from wlz_optimizer.correctness_worker import WorkerRequest, WorkerResult, run_worker
    from wlz_optimizer.input_materializer import MaterializedInputs
    from wlz_optimizer.schemas import EvaluationCase


ReferenceFunction = Callable[[Dict[str, Any]], Any]


def _protocol_echo(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"payload": payload, "working_directory": os.getcwd()}


def _protocol_raise(payload: Dict[str, Any]) -> None:
    raise ValueError(str(payload.get("message", "reference failure")))


def _protocol_sleep(payload: Dict[str, Any]) -> str:
    time.sleep(float(payload["seconds"]))
    return "finished"


def _act_quant_default(payload: Dict[str, Any]) -> Dict[str, Any]:
    if set(payload) != {"x", "block_size"} or payload["block_size"] != 128:
        raise ValueError("act-quant reference requires x and block_size=128")
    x_data = payload["x"]
    if not isinstance(x_data, dict) or set(x_data) != {"shape", "dtype", "values"}:
        raise ValueError("act-quant x must contain shape, dtype, and values")
    if x_data["shape"] != [2, 4, 256] or x_data["dtype"] != "torch.float32":
        raise ValueError("act-quant x must have public shape [2, 4, 256] and torch.float32")
    values = x_data["values"]
    if (
        not isinstance(values, list)
        or len(values) != 2 * 4 * 256
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise ValueError("act-quant x values must be 2048 finite JSON numbers")

    torch = importlib.import_module("torch")
    x = torch.tensor(values, dtype=torch.float32).reshape(2, 4, 256)
    x_reshaped = x.view(-1, 128)
    amax = torch.max(torch.abs(x_reshaped), dim=1, keepdim=True).values
    scale = torch.clamp(amax, min=1e-4) / 448.0
    y = torch.clamp(x_reshaped / scale, min=-448.0, max=448.0)
    y = y.view_as(x).half()
    scale = scale.view(2, 4, 2).squeeze(-1)

    def encoded(tensor: Any) -> Dict[str, Any]:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "values": tensor.reshape(-1).tolist(),
        }

    return {"return": [encoded(y), encoded(scale)], "tensors": {}}


def _act_quant_round_shapes(payload: Dict[str, Any]) -> Dict[str, Any]:
    expected = {"x_shape": [2, 4, 256], "block_size": 128, "scale_fmt": "round"}
    if payload != expected:
        raise ValueError("act-quant round reference requires the public shape contract")

    def encoded(shape: list[int]) -> Dict[str, Any]:
        return {
            "shape": shape,
            "dtype": "shape_only",
            "values": [],
        }

    return {
        "return": [encoded([2, 4, 256]), encoded([2, 4, 2])],
        "tensors": {},
    }


def _chunk_cumsum(payload: Dict[str, Any]) -> Dict[str, Any]:
    expected = {"dt", "A", "dt_bias", "chunk_size", "dt_softplus", "dt_limit"}
    if set(payload) != expected:
        raise ValueError("chunk-cumsum reference payload is invalid")
    if (
        type(payload["chunk_size"]) is not int
        or payload["chunk_size"] != 8
        or payload["dt_softplus"] is not True
        or payload["dt_limit"] != [0.0, 10.0]
    ):
        raise ValueError("chunk-cumsum parameters must match the public case")

    def tensor_values(name: str, shape: list[int]) -> list[float]:
        encoded = payload[name]
        if (
            not isinstance(encoded, dict)
            or set(encoded) != {"shape", "dtype", "values"}
            or encoded["shape"] != shape
            or encoded["dtype"] != "torch.float32"
            or not isinstance(encoded["values"], list)
            or len(encoded["values"]) != math.prod(shape)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in encoded["values"]
            )
        ):
            raise ValueError(f"chunk-cumsum {name} payload is invalid")
        return [float(value) for value in encoded["values"]]

    dt = tensor_values("dt", [2, 16, 4])
    a = tensor_values("A", [4])
    bias = tensor_values("dt_bias", [4])
    d_a_cumsum = []
    dt_out = []
    for batch in range(2):
        for head in range(4):
            for chunk in range(2):
                running = 0.0
                for offset in range(8):
                    sequence = chunk * 8 + offset
                    value = dt[(batch * 16 + sequence) * 4 + head] + bias[head]
                    if value <= 20.0:
                        value = math.log1p(math.exp(value))
                    value = min(10.0, max(0.0, value))
                    running += value * a[head]
                    dt_out.append(value)
                    d_a_cumsum.append(running)

    def snapshot(values: list[float]) -> Dict[str, Any]:
        return {
            "shape": [2, 4, 2, 8],
            "dtype": "torch.float32",
            "values": values,
        }

    return {
        "return": [snapshot(d_a_cumsum), snapshot(dt_out)],
        "tensors": {},
    }


def _quantize_k_cache_rope(payload: Dict[str, Any]) -> Dict[str, Any]:
    if set(payload) != {"k_rope"}:
        raise ValueError("quantize-k-cache rope reference requires k_rope")
    k_rope = payload["k_rope"]
    if not isinstance(k_rope, dict) or set(k_rope) != {"shape", "dtype", "values"}:
        raise ValueError("quantize-k-cache k_rope must contain shape, dtype, and values")
    if k_rope["shape"] != [4, 64] or k_rope["dtype"] != "torch.bfloat16":
        raise ValueError("quantize-k-cache k_rope must have public shape and dtype")
    values = k_rope["values"]
    if (
        not isinstance(values, list)
        or len(values) != 4 * 64
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise ValueError("quantize-k-cache k_rope values must be 256 finite JSON numbers")
    return {"return": k_rope, "tensors": {}}


def _count_expert_basic_no_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    expected_ids = [0, 1, 2, 0, 1, 3, 2, 0]
    if set(payload) != {"topk_ids", "num_local_experts", "expert_map"}:
        raise ValueError("count-expert reference payload is invalid")
    if payload["num_local_experts"] != 4 or payload["expert_map"] is not None:
        raise ValueError("count-expert reference requires the basic no-map scalars")
    topk_ids = payload["topk_ids"]
    if (
        not isinstance(topk_ids, dict)
        or set(topk_ids) != {"shape", "dtype", "values"}
        or topk_ids["shape"] != [8]
        or topk_ids["dtype"] != "torch.int32"
        or topk_ids["values"] != expected_ids
        or any(type(value) is not int for value in topk_ids["values"])
    ):
        raise ValueError("count-expert topk_ids must match the basic public case")
    counts = [0] * 4
    for expert_id in topk_ids["values"]:
        if expert_id >= 0:
            counts[expert_id] += 1
    return {
        "return": {
            "shape": [4],
            "dtype": "torch.int32",
            "values": counts,
        },
        "tensors": {},
    }


def _quantize_k_cache_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    expected = {
        "k_nope_shape": [4, 512],
        "k_rope_shape": [4, 64],
        "group_size": 128,
    }
    if payload != expected:
        raise ValueError("quantize-k-cache metadata reference requires the public contract")
    return {
        "return": {
            "shape": [4, 592],
            "dtype": "torch.bfloat16",
            "values": [0.0] * (4 * 592),
            # None so the oracle skips the device comparison: the candidate may
            # run on local CUDA snapshots or on the official NPU environment.
            "device": None,
        },
        "tensors": {},
    }


def _set_k_and_s_first_token_bytes(payload: Dict[str, Any]) -> Dict[str, Any]:
    if set(payload) != {"loc", "index_k_first", "scale_first", "page_size"}:
        raise ValueError("set-k-and-s reference payload is invalid")
    if payload["page_size"] != 64:
        raise ValueError("set-k-and-s page_size must be 64")

    def values(name: str, shape: list[int], dtype: str) -> list[Any]:
        encoded = payload[name]
        if (
            not isinstance(encoded, dict)
            or set(encoded) != {"shape", "dtype", "values"}
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
        ):
            raise ValueError(f"set-k-and-s {name} payload is invalid")
        return encoded["values"]

    loc = values("loc", [3], "torch.int64")
    if loc != [0, 64, 128]:
        raise ValueError("set-k-and-s loc must match the visible public case")
    k_bytes = b"".join(
        struct.pack("<e", float(value))
        for value in values("index_k_first", [128], "torch.float16")
    )
    scale_bytes = struct.pack(
        "<f", float(values("scale_first", [1], "torch.float32")[0])
    )

    def snapshot(data: bytes) -> Dict[str, Any]:
        return {
            "shape": [len(data)],
            "dtype": "torch.uint8",
            "values": list(data),
        }

    return {
        "return": [snapshot(k_bytes), snapshot(scale_bytes)],
        "tensors": {},
    }


def _per_group_transpose(payload: Dict[str, Any]) -> Dict[str, Any]:
    if set(payload) != {"a", "expert_offsets", "M_ALIGNMENT"}:
        raise ValueError("per-group-transpose reference payload is invalid")
    if type(payload["M_ALIGNMENT"]) is not int or payload["M_ALIGNMENT"] != 1:
        raise ValueError("per-group-transpose M_ALIGNMENT must be 1")

    a = payload["a"]
    if (
        not isinstance(a, dict)
        or set(a) != {"shape", "dtype", "values"}
        or not isinstance(a["shape"], list)
        or any(type(size) is not int for size in a["shape"])
        or a["shape"] != [32, 16]
        or a["dtype"] != "torch.float32"
        or not isinstance(a["values"], list)
        or len(a["values"]) != 32 * 16
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in a["values"]
        )
    ):
        raise ValueError("per-group-transpose a payload is invalid")
    offsets = payload["expert_offsets"]
    if (
        not isinstance(offsets, dict)
        or set(offsets) != {"shape", "dtype", "values"}
        or not isinstance(offsets["shape"], list)
        or any(type(size) is not int for size in offsets["shape"])
        or offsets["shape"] != [3]
        or offsets["dtype"] != "torch.int32"
        or not isinstance(offsets["values"], list)
        or any(type(value) is not int for value in offsets["values"])
        or offsets["values"] != [0, 16, 32]
    ):
        raise ValueError("per-group-transpose expert_offsets must match the public case")

    values = a["values"]
    width = 16
    output = [
        values[row * width + column]
        for start, stop in zip(offsets["values"], offsets["values"][1:])
        for column in range(width)
        for row in range(start, stop)
    ]
    return {
        "return": {
            "shape": [32, 16],
            "dtype": "torch.float32",
            "values": output,
        },
        "tensors": {},
    }


def _pack_seq(payload: Dict[str, Any]) -> Dict[str, Any]:
    if set(payload) != {"x_prefix", "lengths", "pad_value", "block_t", "block_d"}:
        raise ValueError("pack-seq reference payload is invalid")
    if (
        payload["lengths"] != [3, 4, 3]
        or payload["pad_value"] != 0.0
        or payload["block_t"] != 32
        or payload["block_d"] != 32
    ):
        raise ValueError("pack-seq reference scalars do not match the public case")
    x = payload["x_prefix"]
    if (
        not isinstance(x, dict)
        or set(x) != {"shape", "dtype", "values"}
        or x["shape"] != [10, 4]
        or x["dtype"] != "torch.float32"
        or not isinstance(x["values"], list)
        or len(x["values"]) != 40
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in x["values"]
        )
    ):
        raise ValueError("pack-seq x prefix is invalid")
    output = [0.0] * (3 * 4 * 4)
    source_row = 0
    for batch, length in enumerate(payload["lengths"]):
        for row in range(length):
            source = (source_row + row) * 4
            target = (batch * 4 + row) * 4
            output[target : target + 4] = x["values"][source : source + 4]
        source_row += length
    return {
        "return": {"shape": [3, 4, 4], "dtype": "torch.float32", "values": output},
        "tensors": {},
    }


_REFERENCE_REGISTRY: Dict[str, ReferenceFunction] = {
    ACT_QUANT_DEFAULT_REFERENCE_ID: _act_quant_default,
    ACT_QUANT_ROUND_REFERENCE_ID: _act_quant_round_shapes,
    CHUNK_CUMSUM_REFERENCE_ID: _chunk_cumsum,
    COUNT_EXPERT_REFERENCE_ID: _count_expert_basic_no_map,
    PACK_SEQ_REFERENCE_ID: _pack_seq,
    PER_GROUP_TRANSPOSE_REFERENCE_ID: _per_group_transpose,
    QUANTIZE_K_CACHE_ROPE_REFERENCE_ID: _quantize_k_cache_rope,
    QUANTIZE_K_CACHE_METADATA_REFERENCE_ID: _quantize_k_cache_metadata,
    SET_K_AND_S_REFERENCE_ID: _set_k_and_s_first_token_bytes,
    "protocol.echo.v1": _protocol_echo,
    "protocol.raise.v1": _protocol_raise,
    "protocol.sleep.v1": _protocol_sleep,
}


def registered_reference_ids() -> Tuple[str, ...]:
    return tuple(sorted(_REFERENCE_REGISTRY))


def run_case_reference(
    case: EvaluationCase, inputs: MaterializedInputs
) -> "ReferenceResult":
    """Bridge one materialized case to its registered isolated reference."""

    if not isinstance(case, EvaluationCase):
        raise TypeError("case must be an EvaluationCase")
    if not isinstance(inputs, MaterializedInputs):
        raise TypeError("inputs must be MaterializedInputs")
    if inputs.input_signature != case.input_signature():
        raise ValueError("materialized inputs do not match the evaluation case")
    reference_id = case.oracle_policy.reference_id
    if reference_id not in {
        ACT_QUANT_DEFAULT_REFERENCE_ID,
        ACT_QUANT_ROUND_REFERENCE_ID,
        CHUNK_CUMSUM_REFERENCE_ID,
        COUNT_EXPERT_REFERENCE_ID,
        PACK_SEQ_REFERENCE_ID,
        PER_GROUP_TRANSPOSE_REFERENCE_ID,
        QUANTIZE_K_CACHE_ROPE_REFERENCE_ID,
        QUANTIZE_K_CACHE_METADATA_REFERENCE_ID,
        SET_K_AND_S_REFERENCE_ID,
    }:
        raise KeyError(
            f"no materialized reference bridge: {reference_id}"
        )
    try:
        if reference_id == ACT_QUANT_DEFAULT_REFERENCE_ID:
            x = inputs.tensors["x"]
            x = x.detach().cpu().contiguous()
            payload = {
                "x": {
                    "shape": [int(size) for size in x.shape],
                    "dtype": str(x.dtype),
                    "values": x.reshape(-1).tolist(),
                },
                "block_size": inputs.scalars["block_size"],
            }
        elif reference_id == ACT_QUANT_ROUND_REFERENCE_ID:
            x = inputs.tensors["x"]
            payload = {
                "x_shape": [int(size) for size in x.shape],
                "block_size": inputs.scalars["block_size"],
                "scale_fmt": inputs.scalars["scale_fmt"],
            }
        elif reference_id == CHUNK_CUMSUM_REFERENCE_ID:
            payload = {
                name: {
                    "shape": [int(size) for size in inputs.tensors[name].shape],
                    "dtype": str(inputs.tensors[name].dtype),
                    "values": inputs.tensors[name]
                    .detach()
                    .cpu()
                    .contiguous()
                    .reshape(-1)
                    .tolist(),
                }
                for name in ("dt", "A", "dt_bias")
            }
            payload.update(
                chunk_size=inputs.scalars["chunk_size"],
                dt_softplus=inputs.scalars["dt_softplus"],
                dt_limit=inputs.scalars["dt_limit"],
            )
        elif reference_id == COUNT_EXPERT_REFERENCE_ID:
            topk_ids = inputs.tensors["topk_ids"].detach().cpu().contiguous()
            payload = {
                "topk_ids": {
                    "shape": [int(size) for size in topk_ids.shape],
                    "dtype": str(topk_ids.dtype),
                    "values": topk_ids.reshape(-1).tolist(),
                },
                "num_local_experts": inputs.scalars["num_local_experts"],
                "expert_map": inputs.scalars["expert_map"],
            }
        elif reference_id == PER_GROUP_TRANSPOSE_REFERENCE_ID:
            a = inputs.tensors["a"].detach().cpu().contiguous()
            offsets = inputs.tensors["expert_offsets"].detach().cpu().contiguous()
            payload = {
                "a": {
                    "shape": [int(size) for size in a.shape],
                    "dtype": str(a.dtype),
                    "values": a.reshape(-1).tolist(),
                },
                "expert_offsets": {
                    "shape": [int(size) for size in offsets.shape],
                    "dtype": str(offsets.dtype),
                    "values": offsets.reshape(-1).tolist(),
                },
                "M_ALIGNMENT": inputs.scalars["M_ALIGNMENT"],
            }
        elif reference_id == PACK_SEQ_REFERENCE_ID:
            x = inputs.tensors["x"][:10].detach().cpu().contiguous()
            lengths = inputs.tensors["lengths"].detach().cpu().contiguous()
            payload = {
                "x_prefix": {
                    "shape": [int(size) for size in x.shape],
                    "dtype": str(x.dtype),
                    "values": x.reshape(-1).tolist(),
                },
                "lengths": lengths.reshape(-1).tolist(),
                "pad_value": inputs.scalars["pad_value"],
                "block_t": inputs.scalars["block_t"],
                "block_d": inputs.scalars["block_d"],
            }
        elif reference_id == QUANTIZE_K_CACHE_ROPE_REFERENCE_ID:
            k_rope = inputs.tensors["k_rope"].detach().cpu().contiguous()
            payload = {
                "k_rope": {
                    "shape": [int(size) for size in k_rope.shape],
                    "dtype": str(k_rope.dtype),
                    "values": k_rope.reshape(-1).tolist(),
                }
            }
        elif reference_id == QUANTIZE_K_CACHE_METADATA_REFERENCE_ID:
            payload = {
                "k_nope_shape": [int(size) for size in inputs.tensors["k_nope"].shape],
                "k_rope_shape": [int(size) for size in inputs.tensors["k_rope"].shape],
                "group_size": inputs.scalars["group_size"],
            }
        elif reference_id == SET_K_AND_S_REFERENCE_ID:
            loc = inputs.tensors["loc"].detach().cpu().contiguous()
            index_k = inputs.tensors["index_k"].detach().cpu().contiguous()
            scale = inputs.tensors["index_k_scale"].detach().cpu().contiguous()
            payload = {
                "loc": {
                    "shape": [int(size) for size in loc.shape],
                    "dtype": str(loc.dtype),
                    "values": loc.reshape(-1).tolist(),
                },
                "index_k_first": {
                    "shape": [int(index_k.shape[1])],
                    "dtype": str(index_k.dtype),
                    "values": index_k[0].reshape(-1).tolist(),
                },
                "scale_first": {
                    "shape": [1],
                    "dtype": str(scale.dtype),
                    "values": scale[0].reshape(-1).tolist(),
                },
                "page_size": inputs.scalars["page_size"],
            }
        else:  # Guarded by the explicit bridge allowlist above.
            raise AssertionError(f"unhandled materialized reference bridge: {reference_id}")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot serialize materialized reference inputs: {exc}") from exc
    return run_reference(
        ReferenceRequest(
            reference_id=reference_id,
            payload=payload,
            timeout_seconds=10.0,
            max_output_bytes=128 * 1024,
        )
    )


@dataclass(frozen=True)
class ReferenceRequest:
    reference_id: str
    payload: Dict[str, Any]
    timeout_seconds: float
    max_output_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if self.reference_id not in _REFERENCE_REGISTRY:
            raise KeyError(f"reference_id is not registered: {self.reference_id}")
        if not isinstance(self.payload, dict):
            raise ValueError("reference payload must be a mapping")
        try:
            json.dumps(self.payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("reference payload must be finite JSON data") from exc
        WorkerRequest(
            argv=(sys.executable,),
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )


@dataclass(frozen=True)
class ReferenceResult:
    status: str
    value: Any
    error_type: Optional[str]
    error_message: Optional[str]
    worker_result: WorkerResult

    def __post_init__(self) -> None:
        if self.status not in {"completed", "reference_error", "timeout", "worker_error"}:
            raise ValueError(f"unsupported reference status: {self.status}")


def run_reference(request: ReferenceRequest) -> ReferenceResult:
    """Run a pre-registered reference ID without accepting import paths or code."""

    if not isinstance(request, ReferenceRequest):
        raise TypeError("request must be a ReferenceRequest")
    payload = base64.urlsafe_b64encode(
        json.dumps(request.payload, allow_nan=False, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    worker = run_worker(
        WorkerRequest(
            argv=(
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                request.reference_id,
                payload,
            ),
            timeout_seconds=request.timeout_seconds,
            max_output_bytes=request.max_output_bytes,
        )
    )
    if worker.status == "timeout":
        return ReferenceResult("timeout", None, None, None, worker)
    if worker.status != "completed" or worker.returncode != 0 or worker.stdout.truncated:
        return ReferenceResult(
            "worker_error",
            None,
            "WorkerProtocolError",
            "reference worker did not return a complete protocol result",
            worker,
        )
    try:
        message = json.loads(worker.stdout.text)
    except (json.JSONDecodeError, TypeError) as exc:
        return ReferenceResult("worker_error", None, type(exc).__name__, str(exc), worker)
    if message.get("status") == "completed":
        return ReferenceResult("completed", message.get("value"), None, None, worker)
    if message.get("status") == "reference_error":
        return ReferenceResult(
            "reference_error",
            None,
            message.get("error_type"),
            message.get("error_message"),
            worker,
        )
    return ReferenceResult(
        "worker_error", None, "WorkerProtocolError", "unknown worker message", worker
    )


def _worker_main(reference_id: str, encoded_payload: str) -> int:
    if reference_id not in _REFERENCE_REGISTRY:
        return _write_message(
            {
                "status": "reference_error",
                "error_type": "UnknownReference",
                "error_message": "reference ID is not registered",
            }
        )
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload).decode("utf-8"))
        with contextlib.redirect_stdout(sys.stderr):
            value = _REFERENCE_REGISTRY[reference_id](payload)
        return _write_message({"status": "completed", "value": value})
    except Exception as exc:
        return _write_message(
            {
                "status": "reference_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )


def _write_message(message: Dict[str, Any]) -> int:
    try:
        sys.stdout.write(json.dumps(message, allow_nan=False, sort_keys=True))
        return 0
    except (TypeError, ValueError) as exc:
        fallback = {
            "status": "reference_error",
            "error_type": type(exc).__name__,
            "error_message": "reference result is not finite JSON data",
        }
        sys.stdout.write(json.dumps(fallback, sort_keys=True))
        return 0


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit(2)
    raise SystemExit(_worker_main(sys.argv[2], sys.argv[3]))
