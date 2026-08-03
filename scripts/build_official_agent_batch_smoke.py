#!/usr/bin/env python3
"""Package one locally validated real-Agent candidate for every official operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.io_utils import discover_operators
from wlz_optimizer.output_contract import validate_output_contract


_B2_ADMISSION_POLICY_IDS = {
    "_act_quant_kernel": "local-act-quant-public-cuda-v1",
    "_count_expert_num_tokens": "local-count-expert-basic-no-map-cuda-v1",
    "_quantize_k_cache_fast_kernel": "local-quantize-k-cache-public-cuda-v1",
    "_set_k_and_s_triton_kernel": "local-set-k-and-s-public-cuda-guard-v1",
}

# Keep aligned with the operator gates in generate_official_candidates_batch.py.
_CORRECTNESS_ADMISSION_OPERATORS = frozenset(
    {
        "_per_group_transpose",
        "_selective_scan_update_kernel",
        *_B2_ADMISSION_POLICY_IDS,
    }
)
_CANDIDATE_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_ARTIFACT_BYTES = 20_971_520


def _verify_archive(output_zip: Path, expected_entries: list[str]) -> None:
    if output_zip.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Artifact exceeds platform limit of {MAX_ARTIFACT_BYTES} bytes")
    with zipfile.ZipFile(output_zip) as archive:
        names = archive.namelist()
        bad_entry = archive.testzip()
    if bad_entry is not None:
        raise ValueError(f"Artifact ZIP integrity check failed at {bad_entry}")
    if names != expected_entries:
        raise ValueError("Artifact ZIP entries do not match the staged output")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe artifact ZIP entry: {name}")


def _has_packaging_admission(operator: str, manifest: dict, candidate_id: str) -> bool:
    evaluation = manifest.get("correctness_evaluation")
    if evaluation is None:
        return operator not in _CORRECTNESS_ADMISSION_OPERATORS
    admitted = (
        isinstance(evaluation, dict)
        and evaluation.get("status") == "passed"
        and evaluation.get("eligible_for_performance") is True
        and evaluation.get("blocking_reasons") == []
    )
    if admitted and operator in _B2_ADMISSION_POLICY_IDS:
        results = evaluation.get("results")
        candidate = manifest.get("candidate")
        llm_stats = manifest.get("llm_stats")
        calls = llm_stats.get("calls") if isinstance(llm_stats, dict) else None
        call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
        usage = call.get("usage") if isinstance(call, dict) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        admitted = (
            isinstance(candidate, dict)
            and evaluation.get("admission_policy_id")
            == _B2_ADMISSION_POLICY_IDS[operator]
            and evaluation.get("evidence_scope")
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
    if not admitted or "decision" not in evaluation:
        return admitted
    decision = evaluation["decision"]
    return (
        isinstance(decision, dict)
        and decision.get("candidate_id") == candidate_id
        and decision.get("eligible_for_performance") is True
        and decision.get("blocking_reasons") == []
    )


def _load_selection_lock(path: Path, operators: list[str]) -> dict[str, tuple[str, str]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read selection manifest {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("selections"), list):
        raise ValueError("Selection manifest must contain a selections list")

    expected = set(operators)
    locked: dict[str, tuple[str, str]] = {}
    for index, item in enumerate(document["selections"]):
        if not isinstance(item, dict):
            raise ValueError(f"Selection at index {index} must be an object")
        operator = item.get("operator")
        candidate_id = item.get("candidate_id")
        candidate_hash = item.get("candidate_sha256")
        if not isinstance(operator, str) or operator not in expected:
            raise ValueError(f"Selection contains unknown operator: {operator!r}")
        if operator in locked:
            raise ValueError(f"Selection contains duplicate operator: {operator}")
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(candidate_id):
            raise ValueError(f"Invalid candidate_id for {operator}")
        if not isinstance(candidate_hash, str) or not _SHA256_RE.fullmatch(candidate_hash):
            raise ValueError(f"Invalid candidate_sha256 for {operator}")
        locked[operator] = (candidate_id, candidate_hash)

    missing = sorted(expected - locked.keys())
    if missing:
        raise ValueError(f"Selection is missing operators: {', '.join(missing)}")
    return locked


def _select_candidate(
    candidate_root: Path,
    operator: str,
    expected_id: str,
    expected_hash: str,
    *,
    require_current_admission: bool = True,
) -> tuple[Path, Path, dict]:
    manifest_path = candidate_root / operator / f"{expected_id}.manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = manifest["candidate"]
        if not isinstance(candidate, dict):
            raise TypeError("candidate must be an object")
        candidate_path = manifest_path.with_name(f"{expected_id}.py")
        code = candidate_path.read_text(encoding="utf-8")
        import_evaluation = manifest.get("import_evaluation")
        static_evaluation = manifest.get("static_evaluation")
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot load locked candidate {expected_id} for {operator}: {exc}"
        ) from exc
    if candidate.get("id") != expected_id or candidate.get("code_hash") != expected_hash:
        raise ValueError(f"Locked candidate identity mismatch for {operator}")
    if (
        candidate.get("op_name") != operator
        or candidate.get("status") != "static_pass"
        or sha256_text(code) != expected_hash
        or manifest.get("rejection_error") is not None
        or not isinstance(static_evaluation, dict)
        or static_evaluation.get("passed") is not True
    ):
        raise ValueError(f"Locked candidate lacks valid static evidence for {operator}")
    if require_current_admission and (
        not isinstance(import_evaluation, dict)
        or import_evaluation.get("status") != "imported"
        or import_evaluation.get("phase") != "module_import"
        or not _has_packaging_admission(operator, manifest, expected_id)
    ):
        raise ValueError(
            f"Locked candidate lacks valid static/import/admission evidence for {operator}"
        )
    return candidate_path, manifest_path, manifest


def _stats_bytes(candidate: dict, manifest: dict, manifest_path: Path) -> bytes:
    parent_ids = candidate.get("parent_ids", [])
    if not isinstance(parent_ids, list):
        parent_ids = []
    stats = {
        "schema_version": 1,
        "evaluation_status": "not_evaluated_on_ascend",
        "best_fitness": None,
        "speedup": None,
        "generations": candidate.get("generation"),
        "time_elapsed": None,
        "llm_stats": manifest.get("llm_stats", {}),
        "top5_summary": [
            {
                "id": candidate.get("id"),
                "code_hash": candidate.get("code_hash"),
                "parent_ids": [item for item in parent_ids if isinstance(item, str)],
                "fitness": None,
                "generation": candidate.get("generation"),
                "mutation_kind": candidate.get("mutation_kind"),
                "model_used": candidate.get("model_used"),
                "prompt_id": candidate.get("prompt_id"),
                "status": candidate.get("status"),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
        ],
    }
    return (json.dumps(stats, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _load_historical_source(
    path: Path,
    operators: list[str],
    selection_lock: dict[str, tuple[str, str]],
) -> dict:
    source_lock = _load_selection_lock(path, operators)
    if source_lock != selection_lock:
        raise ValueError("Historical source selections do not match selection manifest")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        artifact_name = document["artifact_path"]
        expected_hash = document["artifact_sha256"]
        expected_entries = document["archive_entries"]
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read historical source manifest {path}: {exc}") from exc
    if (
        not isinstance(artifact_name, str)
        or Path(artifact_name).name != artifact_name
        or not isinstance(expected_hash, str)
        or not _SHA256_RE.fullmatch(expected_hash)
        or document.get("operator_count") != len(operators)
        or document.get("candidate_count") != len(selection_lock)
        or document.get("scoring_intent")
        != "21-operator-functional-and-performance-smoke"
        or document.get("layout") != "organizer-save-results-v1"
        or not isinstance(expected_entries, list)
        or not all(isinstance(entry, str) for entry in expected_entries)
    ):
        raise ValueError("Historical source manifest identity is invalid")
    artifact_path = path.parent / artifact_name
    try:
        artifact_bytes = artifact_path.read_bytes()
        with zipfile.ZipFile(artifact_path) as archive:
            bad_entry = archive.testzip()
            actual_entries = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Cannot verify historical source artifact: {exc}") from exc
    if (
        bad_entry is not None
        or actual_entries != expected_entries
        or hashlib.sha256(artifact_bytes).hexdigest() != expected_hash
    ):
        raise ValueError("Historical source artifact integrity check failed")
    return {
        "manifest_path": _display_path(path),
        "artifact_path": _display_path(artifact_path),
        "artifact_sha256": expected_hash,
    }


def build_batch_smoke(
    datasets_dir: Path,
    candidate_root: Path,
    output_zip: Path,
    selection_manifest: Path,
    *,
    historical_source_manifest: Path | None = None,
) -> dict:
    if output_zip.suffix.lower() != ".zip":
        raise ValueError("Output artifact must use the .zip suffix")
    sidecar_path = output_zip.with_suffix(".manifest.json")
    if output_zip.exists() or sidecar_path.exists():
        raise FileExistsError("Refusing to overwrite an existing artifact or manifest")

    operators = discover_operators(datasets_dir)
    selection_lock = _load_selection_lock(selection_manifest, operators)
    historical_source = (
        _load_historical_source(
            historical_source_manifest, operators, selection_lock
        )
        if historical_source_manifest is not None
        else None
    )
    entries: dict[Path, bytes] = {}
    selections = []
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for operator in operators:
            expected_id, expected_hash = selection_lock[operator]
            candidate_path, manifest_path, manifest = _select_candidate(
                candidate_root,
                operator,
                expected_id,
                expected_hash,
                require_current_admission=historical_source is None,
            )
            candidate = manifest["candidate"]
            candidate_bytes = candidate_path.read_bytes()
            output_root = Path("output") / operator
            relative_entries = {
                output_root / f"{operator}_best.py": candidate_bytes,
                output_root / f"{operator}_v1.py": candidate_bytes,
                output_root / f"{operator}_stats.json": _stats_bytes(
                    candidate, manifest, manifest_path
                ),
            }
            for relative, data in relative_entries.items():
                entries[relative] = data
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            selections.append(
                {
                    "operator": operator,
                    "candidate_id": candidate["id"],
                    "candidate_sha256": candidate["code_hash"],
                    "candidate_manifest_path": _display_path(manifest_path),
                    "admission_level": (
                        "historical_exact_identity_only_current_admission_not_claimed"
                        if historical_source is not None
                        else (
                            "local_correctness_admitted"
                            if isinstance(manifest.get("correctness_evaluation"), dict)
                            else "static_import_only"
                        )
                    ),
                }
            )

        report = validate_output_contract(
            staging / "output", datasets_dir, naming="reference-v"
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

    expected_entries = [str(path) for path in sorted(entries, key=str)]
    try:
        _verify_archive(output_zip, expected_entries)
    except (OSError, zipfile.BadZipFile, ValueError):
        output_zip.unlink(missing_ok=True)
        raise
    artifact_bytes = output_zip.read_bytes()
    sidecar = {
        "schema_version": 1,
        "artifact_kind": (
            "historical-selection-rebuild"
            if historical_source is not None
            else "official-real-agent-batch-smoke"
        ),
        "scoring_intent": (
            "historical-composition-reproduction-not-for-submission"
            if historical_source is not None
            else "mixed-local-admission-smoke-not-for-official-scoring"
        ),
        "official_scoring_ready": False,
        "layout": "organizer-save-results-v1",
        "operator_count": len(operators),
        "candidate_count": len(selections),
        "selections": selections,
        "archive_entries": expected_entries,
        "artifact_path": output_zip.name,
        "artifact_size": len(artifact_bytes),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }
    if historical_source is not None:
        sidecar["historical_source"] = historical_source
        sidecar["current_admission_claimed"] = False
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument(
        "--historical-source-manifest",
        help="Bind an exact historical composition; output is marked not for submission",
    )
    args = parser.parse_args()
    result = build_batch_smoke(
        Path(args.datasets_dir),
        Path(args.candidate_root),
        Path(args.output_zip),
        Path(args.selection_manifest),
        historical_source_manifest=(
            Path(args.historical_source_manifest)
            if args.historical_source_manifest
            else None
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
