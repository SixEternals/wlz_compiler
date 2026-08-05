#!/usr/bin/env python3
"""Package one locally validated real-Agent candidate for every official operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from scripts.materialize_local_candidate import _preflight


_B2_ADMISSION_POLICY_IDS = {
    "_act_quant_kernel": "local-act-quant-public-cuda-v1",
    "_count_expert_num_tokens": "local-count-expert-basic-no-map-cuda-v1",
    "_quantize_k_cache_fast_kernel": "local-quantize-k-cache-public-cuda-v1",
    "_set_k_and_s_triton_kernel": "local-set-k-and-s-public-cuda-guard-v1",
}
_LOCAL_ASCEND_ADMISSION_POLICY_ID = "local_ascend_910b4_manual_evidence_v1"
_LOCAL_ASCEND_EVIDENCE_SCOPE_PREFIX = "local_ascend_910b4_"
_CANDIDATE_ONLY_FUNCTIONAL_POLICIES = {
    "_chunk_cumsum_fwd_kernel": "local_ascend_910b4_candidate_only_functional_v1",
    "_set_k_and_s_triton_kernel": (
        "local_ascend_910b4_set_k_fp8_candidate_only_functional_v1"
    ),
}
# Keep the original names for focused tests and callers of the first policy.
_CANDIDATE_ONLY_FUNCTIONAL_OPERATOR = "_chunk_cumsum_fwd_kernel"
_CANDIDATE_ONLY_FUNCTIONAL_POLICY_ID = _CANDIDATE_ONLY_FUNCTIONAL_POLICIES[
    _CANDIDATE_ONLY_FUNCTIONAL_OPERATOR
]
_SET_K_CANDIDATE_ONLY_FUNCTIONAL_OPERATOR = "_set_k_and_s_triton_kernel"
_SET_K_CANDIDATE_ONLY_FUNCTIONAL_POLICY_ID = _CANDIDATE_ONLY_FUNCTIONAL_POLICIES[
    _SET_K_CANDIDATE_ONLY_FUNCTIONAL_OPERATOR
]
_CANDIDATE_ONLY_FUNCTIONAL_STATUS = "candidate_only_functional_passed"
_CANDIDATE_ONLY_HOLDOUT_STATUS = "candidate_only_passed"

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


def _matches_evidence_artifact(
    result: dict, evidence_roots: tuple[Path, ...]
) -> bool:
    artifact_path = result.get("artifact_path")
    artifact_sha256 = result.get("artifact_sha256")
    if (
        not isinstance(artifact_path, str)
        or not artifact_path
        or not isinstance(artifact_sha256, str)
        or not _SHA256_RE.fullmatch(artifact_sha256)
    ):
        return False
    relative = PurePosixPath(artifact_path)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    for root in evidence_roots:
        base = root.resolve()
        candidate = (base / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            continue
        if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == artifact_sha256:
            return True
    return False


def _has_local_ascend_admission(
    evaluation: dict | None,
    candidate_id: str,
    evidence_roots: tuple[Path, ...],
) -> bool:
    """Validate manually captured 910B4 evidence without requiring LLM usage."""
    if not isinstance(evaluation, dict):
        return False
    if evaluation.get("admission_policy_id") != _LOCAL_ASCEND_ADMISSION_POLICY_ID:
        return False
    scope = evaluation.get("evidence_scope")
    if not isinstance(scope, str) or not scope.startswith(_LOCAL_ASCEND_EVIDENCE_SCOPE_PREFIX):
        return False
    results = evaluation.get("results")
    if not isinstance(results, list) or not results:
        return False
    for result in results:
        if (
            not isinstance(result, dict)
            or result.get("status") != "passed"
            or not _matches_evidence_artifact(result, evidence_roots)
        ):
            return False
    decision = evaluation.get("decision")
    return (
        isinstance(decision, dict)
        and decision.get("candidate_id") == candidate_id
        and decision.get("eligible_for_performance") is True
        and decision.get("blocking_reasons") == []
    )


def _has_candidate_only_functional_admission(
    operator: str,
    manifest: dict,
    candidate_id: str,
    evidence_roots: tuple[Path, ...],
) -> bool:
    """Admit an explicit reference-passing fix when the parent is broken.

    This policy is intentionally limited to explicitly named, locally evidenced
    functional repairs. It keeps the failed baseline control visible in the
    manifest while requiring separately hashed candidate-passing artifacts.
    """
    policy_id = _CANDIDATE_ONLY_FUNCTIONAL_POLICIES.get(operator)
    if policy_id is None:
        return False
    evaluation = manifest.get("correctness_evaluation")
    if not isinstance(evaluation, dict) or (
        evaluation.get("admission_policy_id")
        != policy_id
        or evaluation.get("status") != _CANDIDATE_ONLY_FUNCTIONAL_STATUS
        or evaluation.get("eligible_for_performance") is not True
        or evaluation.get("blocking_reasons") != []
        or not isinstance(evaluation.get("evidence_scope"), str)
        or not evaluation["evidence_scope"].startswith(
            _LOCAL_ASCEND_EVIDENCE_SCOPE_PREFIX
        )
    ):
        return False
    decision = evaluation.get("decision")
    if not isinstance(decision, dict) or (
        decision.get("candidate_id") != candidate_id
        or decision.get("eligible_for_performance") is not True
        or decision.get("blocking_reasons") != []
    ):
        return False
    results = evaluation.get("results")
    if not isinstance(results, list) or not results:
        return False
    candidate_passed = False
    baseline_failed = False
    for result in results:
        if not isinstance(result, dict) or not _matches_evidence_artifact(
            result, evidence_roots
        ):
            return False
        if (
            result.get("candidate_status") != "passed"
            or (
                operator == _SET_K_CANDIDATE_ONLY_FUNCTIONAL_OPERATOR
                and result.get("baseline_status") != "failed"
            )
        ):
            return False
        candidate_passed |= result.get("candidate_status") == "passed"
        baseline_failed |= result.get("baseline_status") == "failed"
    if not candidate_passed or not baseline_failed:
        return False
    holdout = manifest.get("holdout_evaluation")
    if _holdout_admission_error(manifest) is not None or not isinstance(holdout, dict):
        return False
    if (
        holdout.get("candidate_id") != candidate_id
        or holdout.get("candidate_status") != "passed"
        or holdout.get("baseline_status") != "failed"
        or not _matches_evidence_artifact(
            {
                "artifact_path": holdout.get("correctness_artifact_path"),
                "artifact_sha256": holdout.get("correctness_artifact_sha256"),
            },
            evidence_roots,
        )
    ):
        return False
    return True


def _has_packaging_admission(
    operator: str,
    manifest: dict,
    candidate_id: str,
    evidence_roots: tuple[Path, ...],
) -> bool:
    if _has_candidate_only_functional_admission(
        operator, manifest, candidate_id, evidence_roots
    ):
        return True
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
        if _has_local_ascend_admission(evaluation, candidate_id, evidence_roots):
            return True
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


def _holdout_admission_error(manifest: dict) -> str | None:
    evidence = manifest.get("holdout_evaluation")
    if not isinstance(evidence, dict):
        return "holdout_required"
    status = evidence.get("status")
    if status == _CANDIDATE_ONLY_HOLDOUT_STATUS:
        if (
            evidence.get("candidate_status") != "passed"
            or evidence.get("baseline_status") != "failed"
        ):
            return "candidate_only_metadata_incomplete"
    elif status != "passed":
        return "holdout_required"
    if evidence.get("split") != "holdout":
        return "holdout_required"
    signature = evidence.get("case_signature")
    if not isinstance(signature, str) or not signature.strip():
        return "holdout_required"
    count = evidence.get("case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return "holdout_required"
    if type(evidence.get("used_for_search")) is not bool or type(
        evidence.get("used_in_prompt")
    ) is not bool:
        return "holdout_metadata_incomplete"
    if evidence.get("used_for_search") is not False:
        return "holdout_search_contamination"
    if evidence.get("used_in_prompt") is True:
        return "holdout_prompt_contamination"
    return None


def _selection_slot(item: dict, operator: str, index: int) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"Selection slot {operator}#{index} must be an object")
    candidate_id = item.get("candidate_id")
    candidate_hash = item.get("candidate_sha256")
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise ValueError(f"Invalid candidate_id for {operator} slot {index}")
    if not isinstance(candidate_hash, str) or not _SHA256_RE.fullmatch(candidate_hash):
        raise ValueError(f"Invalid candidate_sha256 for {operator} slot {index}")
    modification_type = item.get("modification_type", item.get("mutation_kind"))
    if modification_type is not None and (
        not isinstance(modification_type, str) or not modification_type.strip()
    ):
        raise ValueError(f"Invalid modification_type for {operator} slot {index}")
    local_ratio = item.get("local_ratio")
    if local_ratio is not None and (
        isinstance(local_ratio, bool)
        or not isinstance(local_ratio, (int, float))
        or not math.isfinite(local_ratio)
    ):
        raise ValueError(f"Invalid local_ratio for {operator} slot {index}")
    return {
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_hash,
        "modification_type": modification_type,
        "local_ratio": local_ratio,
    }


def _load_selection_lock(path: Path, operators: list[str]) -> dict[str, list[dict]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read selection manifest {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("selections"), list):
        raise ValueError("Selection manifest must contain a selections list")

    expected = set(operators)
    locked: dict[str, list[dict]] = {}
    for index, item in enumerate(document["selections"]):
        if not isinstance(item, dict):
            raise ValueError(f"Selection at index {index} must be an object")
        operator = item.get("operator")
        if not isinstance(operator, str) or operator not in expected:
            raise ValueError(f"Selection contains unknown operator: {operator!r}")
        if operator in locked:
            raise ValueError(f"Selection contains duplicate operator: {operator}")
        raw_slots = item.get("candidates")
        if raw_slots is None:
            raw_slots = [item]
        if not isinstance(raw_slots, list) or not 1 <= len(raw_slots) <= 5:
            raise ValueError(f"Selection for {operator} must contain 1 to 5 candidates")
        slots = [_selection_slot(slot, operator, slot_index)
                 for slot_index, slot in enumerate(raw_slots, start=1)]
        if len({slot["candidate_id"] for slot in slots}) != len(slots):
            raise ValueError(f"Selection contains duplicate candidate_id for {operator}")
        if len({slot["candidate_sha256"] for slot in slots}) != len(slots):
            raise ValueError(f"Selection contains duplicate candidate_sha256 for {operator}")
        locked[operator] = slots

    missing = sorted(expected - locked.keys())
    if missing:
        raise ValueError(f"Selection is missing operators: {', '.join(missing)}")
    return locked


def _selection_slot_count(selection_lock: dict[str, list[dict]]) -> int:
    return sum(len(slots) for slots in selection_lock.values())


def _selection_identity(selection_lock: dict[str, list[dict]]) -> dict[str, list[tuple[str, str]]]:
    return {
        operator: [
            (slot["candidate_id"], slot["candidate_sha256"])
            for slot in slots
        ]
        for operator, slots in selection_lock.items()
    }


def _select_candidate(
    candidate_root: Path,
    datasets_dir: Path,
    operator: str,
    expected_id: str,
    expected_hash: str,
    *,
    evidence_roots: tuple[Path, ...],
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
    # Re-run the same test-aware interface preflight used when materializing a
    # candidate.  Manifest fields are provenance, not an authorization to skip
    # this deterministic boundary check.
    try:
        contract_preflight = _preflight(
            operator,
            datasets_dir / operator / f"{operator}.py",
            candidate_path.read_bytes(),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(
            f"Locked candidate failed interface contract preflight for {operator}: {exc}"
        ) from exc
    if require_current_admission and (
        not isinstance(import_evaluation, dict)
        or import_evaluation.get("status") != "imported"
        or import_evaluation.get("phase") != "module_import"
        or not _has_packaging_admission(
            operator, manifest, expected_id, evidence_roots
        )
    ):
        raise ValueError(
            f"Locked candidate lacks valid static/import/admission evidence for {operator}"
        )
    if require_current_admission:
        holdout_error = _holdout_admission_error(manifest)
        if holdout_error is not None:
            raise ValueError(
                f"Locked candidate lacks holdout admission for {operator}: {holdout_error}"
            )
    return candidate_path, manifest_path, manifest, contract_preflight


def _candidate_stats_entry(
    candidate: dict,
    manifest: dict,
    manifest_path: Path,
    slot: int,
    modification_type: str | None = None,
    local_ratio=None,
) -> dict:
    parent_ids = candidate.get("parent_ids", [])
    if not isinstance(parent_ids, list):
        parent_ids = []
    return {
        "id": candidate.get("id"),
        "code_hash": candidate.get("code_hash"),
        "parent_ids": [item for item in parent_ids if isinstance(item, str)],
        "fitness": None,
        "generation": candidate.get("generation"),
        "mutation_kind": candidate.get("mutation_kind"),
        "modification_type": modification_type,
        "local_ratio": local_ratio,
        "model_used": candidate.get("model_used"),
        "prompt_id": candidate.get("prompt_id"),
        "status": candidate.get("status"),
        "slot": slot,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def _stats_bytes(records: list[dict]) -> bytes:
    first = records[0]
    stats = {
        "schema_version": 1,
        "evaluation_status": "not_evaluated_on_ascend",
        "best_fitness": None,
        "speedup": None,
        "generations": max(
            (record["candidate"].get("generation", 0) or 0 for record in records),
            default=0,
        ),
        "time_elapsed": None,
        "llm_stats": (
            first["manifest"].get("llm_stats", {})
            if len(records) == 1
            else {
                "candidate_calls": [record["manifest"].get("llm_stats", {}) for record in records]
            }
        ),
        "top5_summary": [
            record["stats_entry"] for record in records
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
    selection_lock: dict[str, list[dict]],
) -> dict:
    source_lock = _load_selection_lock(path, operators)
    if _selection_identity(source_lock) != _selection_identity(selection_lock):
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
        or document.get("candidate_count") != _selection_slot_count(selection_lock)
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
    evidence_roots = (selection_manifest.parent, ROOT)
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
            baseline_path = datasets_dir / operator / f"{operator}.py"
            baseline_hash = sha256_text(baseline_path.read_text(encoding="utf-8"))
            records = []
            for slot_index, slot in enumerate(selection_lock[operator], start=1):
                expected_id = slot["candidate_id"]
                expected_hash = slot["candidate_sha256"]
                if expected_hash == baseline_hash:
                    raise ValueError(
                        f"Selection includes baseline candidate for {operator} slot {slot_index}"
                    )
                candidate_path, manifest_path, manifest, contract_preflight = _select_candidate(
                    candidate_root,
                    datasets_dir,
                    operator,
                    expected_id,
                    expected_hash,
                    evidence_roots=evidence_roots,
                    require_current_admission=historical_source is None,
                )
                candidate = manifest["candidate"]
                performance_evaluation = manifest.get("local_performance_evaluation")
                if performance_evaluation is not None and not isinstance(
                    performance_evaluation, dict
                ):
                    raise ValueError(
                        f"Invalid local_performance_evaluation for {operator} slot {slot_index}"
                    )
                selection_intent = (
                    performance_evaluation.get("intended_slot")
                    if performance_evaluation is not None
                    else None
                )
                if (
                    selection_intent == "v3_functional_fallback_only"
                    and slot_index != 3
                ):
                    raise ValueError(
                        f"Functional fallback candidate for {operator} must use slot 3"
                    )
                candidate_bytes = candidate_path.read_bytes()
                modification_type = slot.get("modification_type") or candidate.get(
                    "mutation_kind"
                )
                records.append(
                    {
                        "candidate": candidate,
                        "manifest": manifest,
                        "candidate_path": candidate_path,
                        "manifest_path": manifest_path,
                        "candidate_bytes": candidate_bytes,
                        "contract_preflight": contract_preflight,
                        "modification_type": modification_type,
                        "local_ratio": slot.get("local_ratio"),
                        "stats_entry": _candidate_stats_entry(
                            candidate,
                            manifest,
                            manifest_path,
                            slot_index,
                            modification_type,
                            slot.get("local_ratio"),
                        ),
                    }
                )
            if len(records) >= 2:
                first_type = records[0]["modification_type"]
                second_type = records[1]["modification_type"]
                if not isinstance(first_type, str) or not first_type.strip():
                    raise ValueError(f"Missing modification type for {operator} slot 1")
                if not isinstance(second_type, str) or not second_type.strip():
                    raise ValueError(f"Missing modification type for {operator} slot 2")
                if first_type == second_type:
                    raise ValueError(
                        f"v1 and v2 must use different modification types for {operator}"
                    )
            output_root = Path("output") / operator
            relative_entries = {
                output_root / f"{operator}_best.py": records[0]["candidate_bytes"],
                output_root / f"{operator}_stats.json": _stats_bytes(records),
            }
            relative_entries.update(
                {
                    output_root / f"{operator}_v{slot_index}.py": record["candidate_bytes"]
                    for slot_index, record in enumerate(records, start=1)
                }
            )
            for relative, data in relative_entries.items():
                entries[relative] = data
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            selections.append(
                {
                    "operator": operator,
                    # Keep the v1 fields for consumers of the old singleton schema.
                    "candidate_id": records[0]["candidate"]["id"],
                    "candidate_sha256": records[0]["candidate"]["code_hash"],
                    "candidate_manifest_path": _display_path(records[0]["manifest_path"]),
                    "interface_contract_preflight": records[0]["contract_preflight"],
                    "candidates": [
                        {
                            "candidate_variant": f"{operator}_v{slot_index}",
                            "candidate_id": record["candidate"]["id"],
                            "candidate_sha256": record["candidate"]["code_hash"],
                            "candidate_manifest_path": _display_path(record["manifest_path"]),
                            "interface_contract_preflight": record["contract_preflight"],
                            "modification_type": record["modification_type"],
                            "local_ratio": record["local_ratio"],
                        }
                        for slot_index, record in enumerate(records, start=1)
                    ],
                    "admission_level": (
                        "historical_exact_identity_only_current_admission_not_claimed"
                        if historical_source is not None
                        else (
                            "local_ascend_910b4_candidate_only_functional"
                            if _has_candidate_only_functional_admission(
                                records[0]["candidate"].get("op_name"),
                                records[0]["manifest"],
                                records[0]["candidate"]["id"],
                                evidence_roots,
                            )
                            else (
                                "local_ascend_910b4_manual_evidence"
                                if _has_local_ascend_admission(
                                    records[0]["manifest"].get("correctness_evaluation", {}),
                                    records[0]["candidate"]["id"],
                                    evidence_roots,
                                )
                                else (
                                    "local_correctness_admitted"
                                    if isinstance(
                                        records[0]["manifest"].get("correctness_evaluation"),
                                        dict,
                                    )
                                    else "static_import_only"
                                )
                            )
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
        "candidate_count": _selection_slot_count(selection_lock),
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
