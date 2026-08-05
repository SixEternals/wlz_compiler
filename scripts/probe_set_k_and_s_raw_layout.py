#!/usr/bin/env python3
"""Diagnose the raw-byte Set K And S layout without launching a kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "work/official_triton_agent/datasets/_set_k_and_s_triton_kernel/_set_k_and_s_triton_kernel.py"
PAGE_SIZE = 64
K_CONTRACT_BYTES = 128
K_STORAGE_BYTES = 256  # index_k is float16[128], despite the 128B layout comment.
SCALE_BYTES = 4
SCALE_OFFSET_BYTES = PAGE_SIZE * K_CONTRACT_BYTES
PAGE_BYTES = SCALE_OFFSET_BYTES + PAGE_SIZE * SCALE_BYTES
SCENARIOS = (("all_pages", (0, 64, 128, 192)), ("page_boundaries", (63, 64, 127, 128)), ("permuted_pages", (192, 0, 128, 64)), ("static_conflicts", (0, 32, 255)))
CONTROL_SCENARIOS = SCENARIOS[:3] + (("isolated_tail", (255,)),)
CONTROL_PREFIX_BYTES = 4096
CONTROL_SUFFIX_BYTES = 65536
CONTROL_SENTINEL = 0xA5
_SOURCE_FRAGMENTS = ("buf_fp16 = buf.view(torch.float16)", "buf_fp32 = buf.view(torch.float32)", "BUF_NUMEL_PER_PAGE=buf_numel_per_page")
_CONTROL_MARKER = "WLZ_SET_K_RAW_CONTROL="
_CONTROL_WORKER = r'''
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch_npu

payload = json.loads(sys.stdin.read())
source = Path(payload["source"])
locs = payload["locs"]
num_pages = payload["num_pages"]
page_bytes = payload["page_bytes"]
prefix = payload["prefix_bytes"]
suffix = payload["suffix_bytes"]
sentinel = payload["sentinel"]
if num_pages != 4 or page_bytes != 8448 or not locs:
    raise ValueError("unsupported raw control shape")
if any(type(loc) is not int or not 0 <= loc < num_pages * 64 for loc in locs):
    raise ValueError("invalid raw control loc")

spec = importlib.util.spec_from_file_location("wlz_set_k_raw_control", source)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load Set K source")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

logical_bytes = num_pages * page_bytes
storage = torch.full(
    (prefix + logical_bytes + suffix,), sentinel, dtype=torch.uint8, device="npu"
)
buf = storage[prefix : prefix + logical_bytes].view(num_pages, page_bytes)
loc = torch.tensor(locs, dtype=torch.int64, device="npu")
index_k = torch.stack(
    [
        torch.full((128,), float(index + 1), dtype=torch.float16, device="npu")
        for index in range(len(locs))
    ]
)
scale = torch.tensor(
    [[10.0 + index] for index in range(len(locs))],
    dtype=torch.float32,
    device="npu",
)
module._set_k_and_s_triton(buf, loc, index_k, scale, 64)
torch.npu.synchronize()

def raw_bytes(tensor):
    host = tensor.detach().cpu().contiguous()
    return bytes(host.view(torch.uint8).reshape(-1).tolist())

observed = bytes(storage.detach().cpu().tolist())
expected = bytearray([sentinel]) * len(observed)
k_matches = []
scale_matches = []
k_intervals = []
scale_intervals = []
for index, value in enumerate(locs):
    page, token = divmod(value, 64)
    k_start = page * page_bytes * 2 + token * 256
    scale_start = page * page_bytes + 8192 + token * 4
    k_value = raw_bytes(index_k[index])
    scale_value = raw_bytes(scale[index])
    k_intervals.append([k_start, k_start + len(k_value)])
    scale_intervals.append([scale_start, scale_start + len(scale_value)])
    expected[prefix + k_start : prefix + k_start + len(k_value)] = k_value
    expected[prefix + scale_start : prefix + scale_start + len(scale_value)] = scale_value
    k_matches.append(observed[prefix + k_start : prefix + k_start + len(k_value)] == k_value)
    scale_matches.append(
        observed[prefix + scale_start : prefix + scale_start + len(scale_value)] == scale_value
    )

suffix_start = prefix + logical_bytes
result = {
    "device_name": torch.npu.get_device_name(0),
    "k_raw_intervals": k_intervals,
    "k_writes_match_modeled_offsets": all(k_matches),
    "locs": locs,
    "prefix_guard_unchanged": all(value == sentinel for value in observed[:prefix]),
    "python_executable": sys.executable,
    "scale_raw_intervals": scale_intervals,
    "scale_writes_match_modeled_offsets": all(scale_matches),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "storage_sha256": hashlib.sha256(observed).hexdigest(),
    "suffix_guard_changed": any(value != sentinel for value in observed[suffix_start:]),
    "unexpected_changed_byte_count": sum(
        actual != expected_value for actual, expected_value in zip(observed, expected)
    ),
    "writes_beyond_logical_buffer": any(end > logical_bytes for _, end in k_intervals),
}
result["status"] = "passed" if (
    result["k_writes_match_modeled_offsets"]
    and result["scale_writes_match_modeled_offsets"]
    and result["prefix_guard_unchanged"]
    and result["suffix_guard_changed"]
    and result["writes_beyond_logical_buffer"]
    and result["unexpected_changed_byte_count"] == 0
) else "failed"
print("WLZ_SET_K_RAW_CONTROL=" + json.dumps(result, sort_keys=True))
'''


def _interval(start: int, width: int) -> tuple[int, int]:
    return start, start + width


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def token_address(mode: str, loc: int, *, num_pages: int = 4) -> dict:
    """Return one modeled kernel write in raw uint8 byte intervals."""
    if mode not in {"baseline", "fp16_row_stride_control"}:
        raise ValueError(f"unsupported mode: {mode}")
    if isinstance(loc, bool) or not isinstance(loc, int) or not 0 <= loc < num_pages * PAGE_SIZE:
        raise ValueError("loc is outside the modeled buffer")
    if isinstance(num_pages, bool) or not isinstance(num_pages, int) or num_pages < 1:
        raise ValueError("num_pages must be a positive integer")

    page, token = divmod(loc, PAGE_SIZE)
    page_base = page * PAGE_BYTES
    expected = _interval(page_base + token * K_CONTRACT_BYTES, K_CONTRACT_BYTES)
    page_stride = PAGE_BYTES * 2 if mode == "baseline" else PAGE_BYTES
    k = _interval(page * page_stride + token * K_STORAGE_BYTES, K_STORAGE_BYTES)
    scale = _interval(page_base + SCALE_OFFSET_BYTES + token * SCALE_BYTES, SCALE_BYTES)
    scale_region = _interval(page_base + SCALE_OFFSET_BYTES, PAGE_SIZE * SCALE_BYTES)
    total_bytes = num_pages * PAGE_BYTES
    k_in_buffer = k[1] <= total_bytes
    scale_in_buffer = scale[1] <= total_bytes
    k_page_matches = page_base <= k[0] and k[1] <= page_base + PAGE_BYTES
    issues = []
    if not k_in_buffer:
        issues.append("k_out_of_buffer")
    if not scale_in_buffer:
        issues.append("scale_out_of_buffer")
    if not k_page_matches:
        issues.append("k_page_mismatch")
    if k != expected:
        issues.append("k_interval_contract_mismatch")
    if _overlaps(k, scale_region):
        issues.append("k_overlaps_scale_region")
    return {
        "loc": loc,
        "page_index": page,
        "token_offset_in_page": token,
        "expected_k_raw_interval": list(expected),
        "k_raw_interval": list(k),
        "scale_raw_interval": list(scale),
        "scale_region_raw_interval": list(scale_region),
        "k_in_buffer": k_in_buffer,
        "scale_in_buffer": scale_in_buffer,
        "k_page_matches": k_page_matches,
        "k_overlaps_scale_region": "k_overlaps_scale_region" in issues,
        "issues": issues,
    }


def inspect_locs(mode: str, locs: tuple[int, ...], *, num_pages: int = 4) -> dict:
    addresses = [token_address(mode, loc, num_pages=num_pages) for loc in locs]
    has = lambda issue: any(issue in item["issues"] for item in addresses)
    return {
        "mode": mode,
        "addresses": addresses,
        "summary": {
            "all_k_in_buffer": all(item["k_in_buffer"] for item in addresses),
            "all_scale_in_buffer": all(item["scale_in_buffer"] for item in addresses),
            "any_k_page_mismatch": has("k_page_mismatch"),
            "any_k_contract_mismatch": has("k_interval_contract_mismatch"),
            "any_k_scale_overlap": has("k_overlaps_scale_region"),
            "any_out_of_buffer": has("k_out_of_buffer") or has("scale_out_of_buffer"),
        },
    }


def _validate_control_result(result: object, source: Path, locs: tuple[int, ...]) -> dict:
    if not isinstance(result, dict):
        raise ValueError("baseline control worker did not return an object")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    required_true = (
        "k_writes_match_modeled_offsets",
        "scale_writes_match_modeled_offsets",
        "prefix_guard_unchanged",
        "suffix_guard_changed",
        "writes_beyond_logical_buffer",
    )
    if (
        result.get("status") != "passed"
        or result.get("source_sha256") != expected_hash
        or result.get("locs") != list(locs)
        or not isinstance(result.get("device_name"), str)
        or not result["device_name"].startswith("Ascend910B4")
        or result.get("unexpected_changed_byte_count") != 0
        or any(result.get(field) is not True for field in required_true)
    ):
        raise ValueError("baseline raw-byte control did not match the modeled writes")
    return {
        "status": "passed",
        "device_name": result["device_name"],
        "source_sha256": expected_hash,
        "locs": list(locs),
        "k_writes_match_modeled_offsets": True,
        "k_raw_intervals": result.get("k_raw_intervals"),
        "scale_writes_match_modeled_offsets": True,
        "scale_raw_intervals": result.get("scale_raw_intervals"),
        "storage_sha256": result.get("storage_sha256"),
        "writes_beyond_logical_buffer": True,
        "prefix_guard_unchanged": True,
        "suffix_guard_changed": True,
        "unexpected_changed_byte_count": 0,
    }


def run_baseline_control(source: Path, python: Path, *, timeout: float = 60.0) -> dict:
    """Run only non-overlapping baseline writes in an extended allocation."""
    source = source.resolve()
    python = python.resolve()
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not python.is_file():
        raise FileNotFoundError(f"Ascend Python is missing: {python}")
    source_text = source.read_text(encoding="utf-8")
    if not all(fragment in source_text for fragment in _SOURCE_FRAGMENTS):
        raise ValueError("source no longer matches the modeled Set K And S layout")
    controls = []
    for name, locs in CONTROL_SCENARIOS:
        payload = {
            "source": str(source),
            "locs": list(locs),
            "num_pages": 4,
            "page_bytes": PAGE_BYTES,
            "prefix_bytes": CONTROL_PREFIX_BYTES,
            "suffix_bytes": CONTROL_SUFFIX_BYTES,
            "sentinel": CONTROL_SENTINEL,
        }
        try:
            completed = subprocess.run(
                [str(python), "-c", _CONTROL_WORKER],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"baseline raw-byte control timed out for {name}") from exc
        marker_line = next(
            (
                line
                for line in reversed(completed.stdout.splitlines())
                if line.startswith(_CONTROL_MARKER)
            ),
            None,
        )
        if completed.returncode != 0 or marker_line is None:
            raise RuntimeError(
                f"baseline raw-byte control failed for {name}: rc={completed.returncode}: "
                f"{completed.stderr[-512:]}"
            )
        try:
            result = json.loads(marker_line[len(_CONTROL_MARKER) :])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"baseline raw-byte control emitted invalid JSON for {name}") from exc
        worker_python = result.get("python_executable") if isinstance(result, dict) else None
        if (
            not isinstance(worker_python, str)
            or Path(worker_python).resolve() != python
        ):
            raise RuntimeError(f"baseline raw-byte control used an unexpected interpreter for {name}")
        summary = _validate_control_result(result, source, locs)
        summary["name"] = name
        summary["python_executable"] = worker_python
        summary["worker_stdout_sha256"] = hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest()
        summary["worker_stderr_sha256"] = hashlib.sha256(
            completed.stderr.encode("utf-8")
        ).hexdigest()
        controls.append(summary)
    return {
        "status": "passed",
        "evidence_scope": "local_ascend_910b4_raw_byte_baseline_control_not_official",
        "execution": "isolated_extended_suffix_guard_baseline_only",
        "excluded_from_dynamic_execution": ["static_conflicts"],
        "scenarios": controls,
    }


def build_report(
    source: Path = DEFAULT_SOURCE,
    *,
    num_pages: int = 4,
    baseline_control: dict | None = None,
) -> dict:
    if isinstance(num_pages, bool) or not isinstance(num_pages, int) or num_pages < 1:
        raise ValueError("num_pages must be a positive integer")
    source = source.resolve()
    text = source.read_text(encoding="utf-8")
    if not all(fragment in text for fragment in _SOURCE_FRAGMENTS):
        raise ValueError("source no longer matches the modeled Set K And S layout")
    report = {
        "schema_version": 1,
        "artifact_kind": "set-k-and-s-raw-layout-diagnostic",
        "evidence_scope": "static_layout_diagnostic_not_performance_or_official",
        "semantic_conclusion": "unknown",
        "candidate_admission": "not_applicable",
        "source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "layout": {
            "raw_address_unit": "uint8_bytes",
            "page_size": PAGE_SIZE,
            "page_bytes": PAGE_BYTES,
            "k_contract_bytes_per_token": K_CONTRACT_BYTES,
            "k_storage_bytes_per_token": K_STORAGE_BYTES,
            "scale_bytes_per_token": SCALE_BYTES,
            "scale_offset_bytes_in_page": SCALE_OFFSET_BYTES,
            "num_pages": num_pages,
        },
        "scenarios": [
            {
                "name": name,
                "locs": list(locs),
                "execution": "static_address_model_only",
                "baseline": inspect_locs("baseline", locs, num_pages=num_pages),
                "fp16_row_stride_control": inspect_locs("fp16_row_stride_control", locs, num_pages=num_pages),
            }
            for name, locs in SCENARIOS
        ],
        "limitations": ["No Triton kernel was launched; this is not correctness or performance evidence.", "The //2 control corrects only the page stride and still stores 256 bytes per token."],
    }
    if baseline_control is not None:
        report["baseline_control"] = baseline_control
        report["limitations"][0] = (
            "The isolated baseline control observes raw writes only; it is not "
            "candidate correctness, performance, or official evidence."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--num-pages", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-control-python", type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    build_report(args.source, num_pages=args.num_pages)
    baseline_control = None
    if args.baseline_control_python is not None:
        if args.num_pages != 4:
            raise ValueError("baseline raw-byte control requires --num-pages 4")
        baseline_control = run_baseline_control(
            args.source, args.baseline_control_python, timeout=args.timeout
        )
    report = build_report(
        args.source,
        num_pages=args.num_pages,
        baseline_control=baseline_control,
    )
    rendered = json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
