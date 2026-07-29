#!/usr/bin/env python3
"""Package one real Agent candidate using the organizer save_results layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.output_contract import validate_output_contract


_B2_ADMISSION_POLICY_IDS = {
    "_act_quant_kernel": "local-act-quant-public-cuda-v1",
    "_count_expert_num_tokens": "local-count-expert-basic-no-map-cuda-v1",
    "_quantize_k_cache_fast_kernel": "local-quantize-k-cache-public-cuda-v1",
    "_set_k_and_s_triton_kernel": "local-set-k-and-s-public-cuda-guard-v1",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _has_b2_packaging_admission(kernel: str, candidate: dict, manifest: dict) -> bool:
    policy_id = _B2_ADMISSION_POLICY_IDS.get(kernel)
    if policy_id is None:
        return True
    import_evaluation = manifest.get("import_evaluation")
    correctness = manifest.get("correctness_evaluation")
    results = correctness.get("results") if isinstance(correctness, dict) else None
    llm_stats = manifest.get("llm_stats")
    calls = llm_stats.get("calls") if isinstance(llm_stats, dict) else None
    call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
    usage = call.get("usage") if isinstance(call, dict) else None
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return (
        candidate.get("status") == "static_pass"
        and manifest.get("rejection_error") is None
        and isinstance(import_evaluation, dict)
        and import_evaluation.get("status") == "imported"
        and import_evaluation.get("phase") == "module_import"
        and isinstance(correctness, dict)
        and correctness.get("admission_policy_id") == policy_id
        and correctness.get("status") == "passed"
        and correctness.get("eligible_for_performance") is True
        and correctness.get("blocking_reasons") == []
        and correctness.get("evidence_scope")
        == "local_cuda_proxy_only_not_ascend_or_official"
        and isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        and results[0].get("returncode") == 0
        and results[0].get("matrix_completed") is True
        and isinstance(llm_stats, dict)
        and llm_stats.get("call_count") == 1
        and isinstance(call, dict)
        and call.get("status") == "succeeded"
        and call.get("model") == candidate.get("model_used")
        and call.get("prompt_sha256") == candidate.get("prompt_id")
        and type(total_tokens) is int
        and total_tokens > 0
    )


def build_agent_smoke(
    datasets_dir: Path,
    kernel: str,
    candidate_path: Path | list[Path],
    candidate_manifest_path: Path | list[Path],
    output_zip: Path,
) -> dict:
    if output_zip.suffix.lower() != ".zip":
        raise ValueError("Output artifact must use the .zip suffix")
    sidecar_path = output_zip.with_suffix(".manifest.json")
    if output_zip.exists() or sidecar_path.exists():
        raise FileExistsError("Refusing to overwrite an existing artifact or manifest")

    candidate_paths = [candidate_path] if isinstance(candidate_path, Path) else candidate_path
    manifest_paths = (
        [candidate_manifest_path]
        if isinstance(candidate_manifest_path, Path)
        else candidate_manifest_path
    )
    if not 1 <= len(candidate_paths) <= 5 or len(candidate_paths) != len(manifest_paths):
        raise ValueError("Candidate and manifest paths must contain between 1 and 5 pairs")

    candidates = []
    for source, manifest_path in zip(candidate_paths, manifest_paths):
        candidate_bytes = source.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = manifest.get("candidate") or {}
        if not isinstance(candidate.get("id"), str) or not candidate["id"]:
            raise ValueError("Candidate manifest requires a non-empty candidate ID")
        if candidate.get("op_name") != kernel:
            raise ValueError("Candidate manifest operator does not match requested kernel")
        if candidate.get("code_hash") != _sha256(candidate_bytes):
            raise ValueError("Candidate bytes do not match candidate manifest hash")
        if not manifest.get("static_evaluation", {}).get("passed"):
            raise ValueError("Candidate manifest does not record a passing static evaluation")
        if not _has_b2_packaging_admission(kernel, candidate, manifest):
            raise ValueError(
                "B2 candidate lacks valid import/correctness/usage admission evidence"
            )
        candidates.append((candidate, manifest, manifest_path, candidate_bytes))
    if len({item[0]["id"] for item in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique")
    if len({item[0]["code_hash"] for item in candidates}) != len(candidates):
        raise ValueError("Candidate code hashes must be unique")

    output_root = Path("output") / kernel
    best_relative = output_root / f"{kernel}_best.py"
    stats_relative = output_root / f"{kernel}_stats.json"
    first_candidate, first_manifest, first_manifest_path, first_bytes = candidates[0]
    stats = {
        "schema_version": 1,
        "evaluation_status": "not_evaluated_on_ascend",
        "best_fitness": None,
        "speedup": None,
        "generations": max(item[0].get("generation", 0) for item in candidates),
        "time_elapsed": None,
        "llm_stats": (
            first_manifest.get("llm_stats", {})
            if len(candidates) == 1
            else {"candidate_calls": [item[1].get("llm_stats", {}) for item in candidates]}
        ),
        "top5_summary": [
            {
                "id": candidate.get("id"),
                "fitness": None,
                "generation": candidate.get("generation"),
            }
            for candidate, _, _, _ in candidates
        ],
    }
    stats_bytes = (json.dumps(stats, indent=2, sort_keys=True) + "\n").encode("utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        entries = {best_relative: first_bytes, stats_relative: stats_bytes}
        for index, (_, _, _, candidate_bytes) in enumerate(candidates, start=1):
            entries[output_root / f"{kernel}_v{index}.py"] = candidate_bytes
        for relative, data in entries.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        report = validate_output_contract(
            staging / "output", datasets_dir, naming="reference-v", kernel=kernel
        )
        if not report["valid"]:
            raise ValueError(f"Generated output failed contract: {report['errors']}")

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", compresslevel=9) as archive:
            for relative, data in sorted(entries.items(), key=lambda item: str(item[0])):
                info = zipfile.ZipInfo(str(relative), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)

    artifact_bytes = output_zip.read_bytes()
    sidecar = {
        "schema_version": 1,
        "artifact_kind": "official-real-agent-smoke",
        "scoring_intent": "candidate-scheduling-and-compile-smoke",
        "kernel": kernel,
        "layout": "organizer-save-results-v1",
        "candidate_count": len(candidates),
        "candidate_id": first_candidate.get("id"),
        "candidate_sha256": _sha256(first_bytes),
        "candidate_manifest_path": _display_path(first_manifest_path),
        "selections": [
            {
                "operator": kernel,
                "candidate_variant": f"{kernel}_v{index}",
                "candidate_id": candidate.get("id"),
                "candidate_sha256": _sha256(candidate_bytes),
                "candidate_manifest_path": _display_path(manifest_path),
            }
            for index, (candidate, _, manifest_path, candidate_bytes) in enumerate(
                candidates, start=1
            )
        ],
        "archive_entries": [str(path) for path in sorted(entries, key=str)],
        "artifact_path": output_zip.name,
        "artifact_size": len(artifact_bytes),
        "artifact_sha256": _sha256(artifact_bytes),
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--candidate-manifest", action="append", required=True)
    parser.add_argument("--output-zip", required=True)
    args = parser.parse_args()
    result = build_agent_smoke(
        Path(args.datasets_dir),
        args.kernel,
        [Path(path) for path in args.candidate],
        [Path(path) for path in args.candidate_manifest],
        Path(args.output_zip),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
