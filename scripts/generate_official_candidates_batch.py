#!/usr/bin/env python3
"""Generate missing official candidates with an explicit, resumable call budget."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_official_candidate import _load_official_modules, generate_candidate
from wlz_optimizer.budget import BudgetController, BudgetLimits
from wlz_optimizer.executors import validate_static_structure
from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.io_utils import discover_operators


_SELECTIVE_CUDA_TEST = (
    "tests.test_torch_triton_local_smoke.TorchTritonLocalSmokeTests."
    "test_official_selective_scan_stateful_fresh_rerun"
)
_SELECTIVE_CUDA_COMPLETION_MARKER = "D2_LOCAL_SELECTIVE_SCAN_MATRIX"
_QUANTIZE_CUDA_TEST = (
    "tests.test_correctness_coordinator.CorrectnessCoordinatorTests."
    "test_quantize_cuda_parent_passes_and_wrong_candidate_fails"
)
_QUANTIZE_CUDA_COMPLETION_MARKER = "B2_LOCAL_QUANTIZE_K_CACHE_MATRIX"
_ACT_QUANT_CUDA_TEST = (
    "tests.test_correctness_coordinator.CorrectnessCoordinatorTests."
    "test_act_quant_cuda_parent_passes_and_wrong_candidate_fails"
)
_ACT_QUANT_CUDA_COMPLETION_MARKER = "B2_LOCAL_ACT_QUANT_MATRIX"
_COUNT_EXPERT_CUDA_TEST = (
    "tests.test_correctness_coordinator.CorrectnessCoordinatorTests."
    "test_count_expert_cuda_parent_passes_and_wrong_candidate_fails"
)
_COUNT_EXPERT_CUDA_COMPLETION_MARKER = "B2_LOCAL_COUNT_EXPERT_MATRIX"
_SET_K_AND_S_CUDA_TEST = (
    "tests.test_correctness_coordinator.CorrectnessCoordinatorTests."
    "test_set_k_and_s_cuda_parent_is_unsafe_and_corrected_control_passes"
)
_SET_K_AND_S_CUDA_COMPLETION_MARKER = "B2_LOCAL_SET_K_AND_S_MATRIX"

# Bump the matching policy id whenever a gate or its public case matrix changes.
_ADMISSION_POLICY_IDS = {
    "_act_quant_kernel": "local-act-quant-public-cuda-v1",
    "_count_expert_num_tokens": "local-count-expert-basic-no-map-cuda-v1",
    "_per_group_transpose": "local-per-group-public-case-v1",
    "_quantize_k_cache_fast_kernel": "local-quantize-k-cache-public-cuda-v1",
    "_selective_scan_update_kernel": "local-selective-cuda-dstate-matrix-v1",
    "_set_k_and_s_triton_kernel": "local-set-k-and-s-public-cuda-guard-v1",
}


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _find_static_pass_manifest(output_dir: Path, operator: str) -> Optional[Path]:
    candidate_dir = output_dir / operator
    for manifest_path in sorted(candidate_dir.glob("*.manifest.json")):
        if _is_static_pass_manifest(manifest_path, operator):
            return manifest_path
    return None


def _is_static_pass_manifest(manifest_path: Path, operator: str) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = manifest["candidate"]
        candidate_path = manifest_path.with_name(f"{candidate['id']}.py")
        code = candidate_path.read_text(encoding="utf-8")
        import_evaluation = manifest.get("import_evaluation")
        static_evaluation = manifest.get("static_evaluation")
    except (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return (
        candidate.get("op_name") == operator
        and candidate.get("status") == "static_pass"
        and manifest.get("rejection_error") is None
        and isinstance(static_evaluation, dict)
        and static_evaluation.get("passed") is True
        and isinstance(import_evaluation, dict)
        and import_evaluation.get("status") == "imported"
        and import_evaluation.get("phase") == "module_import"
        and candidate.get("code_hash") == sha256_text(code)
    )


def _checkpoint_entry(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "code_hash": sha256_text(path.read_text(encoding="utf-8")),
    }


def _candidate_artifacts(output_dir: Path, operator: str) -> dict[str, str]:
    candidate_dir = output_dir / operator
    if not candidate_dir.is_dir():
        return {}
    return {
        path.name: sha256_text(path.read_text(encoding="utf-8"))
        for path in sorted(candidate_dir.iterdir())
        if path.is_file()
        and (path.suffix == ".py" or path.name.endswith(".manifest.json"))
    }


def _budget_attempt_state(budget: Optional[BudgetController]) -> Optional[dict]:
    if budget is None:
        return None
    snapshot = budget.snapshot()
    return {
        "used_tokens": snapshot.used_tokens,
        "remaining_tokens": snapshot.remaining_tokens,
        "elapsed_seconds": snapshot.elapsed_seconds,
        "remaining_seconds": snapshot.remaining_seconds,
        "stop_reason": snapshot.stop_reason,
        "in_flight_calls": snapshot.in_flight_calls,
    }


def _validate_budget_history(state: dict, records: list, limits: dict) -> None:
    token_limit = limits.get("token_limit")
    wall_time_seconds = limits.get("wall_time_seconds")
    if isinstance(token_limit, bool) or not isinstance(token_limit, int):
        raise ValueError("checkpoint budget token limit is invalid")
    if isinstance(wall_time_seconds, bool) or not isinstance(
        wall_time_seconds, (int, float)
    ):
        raise ValueError("checkpoint budget wall limit is invalid")

    previous_used = 0
    previous_elapsed = 0.0
    previous_stop_reason = None
    denial_reasons = []
    for index, record in enumerate(records):
        before = record.get("budget_before")
        after = record.get("budget_after")
        for label, snapshot in (("before", before), ("after", after)):
            if not isinstance(snapshot, dict):
                raise ValueError(f"checkpoint record {index} budget_{label} is missing")
            used = snapshot.get("used_tokens")
            remaining = snapshot.get("remaining_tokens")
            elapsed = snapshot.get("elapsed_seconds")
            stop_reason = snapshot.get("stop_reason")
            if (
                isinstance(used, bool)
                or not isinstance(used, int)
                or used < 0
                or isinstance(remaining, bool)
                or not isinstance(remaining, int)
                or remaining != max(0, token_limit - used)
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or elapsed < previous_elapsed
                or snapshot.get("in_flight_calls") != 0
                or stop_reason
                not in {
                    None,
                    "token_limit",
                    "wall_time_limit",
                    "unknown_in_flight_call",
                }
                or (
                    previous_stop_reason is not None
                    and stop_reason != previous_stop_reason
                )
            ):
                raise ValueError(f"checkpoint record {index} budget_{label} is invalid")
            previous_elapsed = float(elapsed)
            previous_stop_reason = stop_reason
        if before["used_tokens"] != previous_used:
            raise ValueError("checkpoint budget usage history is discontinuous")
        if after["used_tokens"] < before["used_tokens"]:
            raise ValueError("checkpoint budget usage moved backwards")
        previous_used = after["used_tokens"]

        llm = record.get("llm")
        if isinstance(llm, dict) and llm.get("error_type") == "budget_denied":
            message = record.get("error_message")
            prefix = "LLM budget denied request:"
            denial_reasons.append(
                message[len(prefix):].strip()
                if isinstance(message, str) and message.startswith(prefix)
                else "budget_denied"
            )

    checkpoint = state.get("budget_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint budget state is missing")
    checkpoint_elapsed = checkpoint.get("elapsed_seconds", -1)
    checkpoint_stop_reason = checkpoint.get("stop_reason")
    final_wall_expiry = (
        previous_stop_reason is None
        and checkpoint_stop_reason == "wall_time_limit"
        and isinstance(checkpoint_elapsed, (int, float))
        and not isinstance(checkpoint_elapsed, bool)
        and checkpoint_elapsed >= wall_time_seconds
    )
    if (
        checkpoint.get("used_tokens") != previous_used
        or not isinstance(checkpoint_elapsed, (int, float))
        or isinstance(checkpoint_elapsed, bool)
        or checkpoint_elapsed < previous_elapsed
        or checkpoint.get("in_flight_calls") != 0
        or checkpoint.get("reserved_tokens") != 0
        or checkpoint.get("reserved_seconds") != 0.0
        or checkpoint.get("remaining_tokens") != max(0, token_limit - previous_used)
        or (
            checkpoint_stop_reason != previous_stop_reason
            and not final_wall_expiry
        )
    ):
        raise ValueError("checkpoint budget state does not match attempt history")

    budget_denial_reason = state.get("budget_denial_reason")
    if len(denial_reasons) > 1 or budget_denial_reason != (
        denial_reasons[0] if denial_reasons else None
    ):
        raise ValueError("checkpoint budget denial does not match attempt history")


def _llm_attempt_evidence(
    manifest: Optional[dict], error: Exception | None = None
) -> Optional[dict]:
    stats = manifest.get("llm_stats") if isinstance(manifest, dict) else None
    if not isinstance(stats, dict) and error is not None:
        stats = getattr(error, "_wlz_llm_stats", None)
    calls = stats.get("calls") if isinstance(stats, dict) else None
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return None
    call = calls[0]
    candidate = manifest.get("candidate", {}) if isinstance(manifest, dict) else {}
    usage = call.get("usage")
    return {
        "model": candidate.get("model_used") or call.get("model"),
        "prompt_sha256": candidate.get("prompt_id") or call.get("prompt_sha256"),
        "request_fingerprint": call.get("request_fingerprint"),
        "status": call.get("status"),
        "error_type": call.get("error_type"),
        "usage": usage if isinstance(usage, dict) else None,
    }


def _new_attempt_manifest(
    output_dir: Path, operator: str, artifacts_before: dict[str, str]
) -> tuple[Optional[Path], Optional[dict], list[str]]:
    artifacts_after = _candidate_artifacts(output_dir, operator)
    new_names = sorted(set(artifacts_after) - set(artifacts_before))
    manifest_names = [name for name in new_names if name.endswith(".manifest.json")]
    if len(manifest_names) != 1:
        return None, None, new_names
    manifest_path = output_dir / operator / manifest_names[0]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return manifest_path, None, new_names
    return manifest_path, manifest, new_names


@contextmanager
def _checkpoint_writer_lock(checkpoint_path: Optional[Path]):
    if checkpoint_path is None:
        yield
        return
    lock_path = checkpoint_path.resolve().with_name(f".{checkpoint_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("mutation checkpoint is already in use") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _restore_mutation_checkpoint(
    checkpoint_path: Path,
    operator: str,
    work_dir: Path,
    datasets_dir: Path,
    parent: Path,
    output_dir: Path,
    random_seed: int,
    beam_width: int,
    admission_mode: str,
    admission_policy_id: str,
    retry_in_flight: bool,
    budget_controller: Optional[BudgetController],
) -> dict:
    try:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 2,
            "artifact_kind": "official-mutation-loop-report",
            "operator": operator,
            "beam_width": beam_width,
            "base_random_seed": random_seed,
            "admission_mode": admission_mode,
            "admission_policy_id": admission_policy_id,
            "work_dir": str(work_dir.resolve()),
            "datasets_dir": str(datasets_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "initial_parent": _checkpoint_entry(parent),
        }
        if budget_controller is not None:
            expected["budget_limits"] = dict(
                budget_controller.snapshot().limits
            )
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint identity does not match this run")
        if budget_controller is None and (
            "budget_limits" in state or "budget_checkpoint" in state
        ):
            raise ValueError("checkpoint requires an explicit budget controller")
        records = state["records"]
        next_ordinal = state["next_ordinal"]
        if not isinstance(records, list) or next_ordinal != len(records):
            raise ValueError("checkpoint history is inconsistent")
        for ordinal, record in enumerate(records):
            attempts = record.get("physical_attempts_maybe_started")
            retried = record.get("retried_in_flight")
            if (
                record.get("ordinal") != ordinal
                or record.get("random_seed") != random_seed + ordinal
                or type(attempts) is not int
                or attempts < 1
                or retried is not (attempts > 1)
                or record.get("prior_token_usage")
                != ("unknown_may_have_been_consumed" if retried else None)
            ):
                raise ValueError("checkpoint ordinal or seed history is inconsistent")
        budget_denial_reason = state.get("budget_denial_reason")
        if budget_denial_reason is not None and (
            not isinstance(budget_denial_reason, str) or not budget_denial_reason
        ):
            raise ValueError("checkpoint budget denial reason is invalid")

        candidate_root = (output_dir / operator).resolve()

        def saved_path(value: str) -> Path:
            path = Path(value)
            return (path if path.is_absolute() else ROOT / path).resolve(strict=True)

        def restore_entry(entry: dict) -> tuple[Path, str, str, int]:
            path = Path(entry["path"])
            if not path.is_absolute():
                raise ValueError("checkpoint paths must be absolute")
            path = path.resolve(strict=True)
            if path != parent and (path.parent != candidate_root or path.suffix != ".py"):
                raise ValueError("checkpoint candidate path is outside the operator output")
            actual_hash = sha256_text(path.read_text(encoding="utf-8"))
            if entry.get("code_hash") != actual_hash:
                raise ValueError("checkpoint candidate hash mismatch")
            candidate_id = f"seed-{actual_hash[:12]}"
            generation = 0
            if path != parent:
                manifest_path = path.with_suffix(".manifest.json")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                candidate = manifest.get("candidate")
                if not isinstance(candidate, dict) or not (
                    candidate.get("id") == path.stem
                    and candidate.get("op_name") == operator
                    and candidate.get("code_hash") == actual_hash
                    and isinstance(candidate.get("generation"), int)
                ):
                    raise ValueError("checkpoint candidate manifest identity mismatch")
                candidate_id = candidate["id"]
                generation = candidate["generation"]
                if not _is_static_pass_manifest(manifest_path, operator):
                    raise ValueError("checkpoint candidate lost static/import admission")
                if admission_mode == "correctness":
                    evaluation = manifest.get("correctness_evaluation")
                    if not isinstance(evaluation, dict) or not (
                        evaluation.get("status") == "passed"
                        and evaluation.get("eligible_for_performance") is True
                        and evaluation.get("blocking_reasons") == []
                    ):
                        raise ValueError("checkpoint candidate lost correctness admission")
                    if "decision" in evaluation:
                        decision = evaluation["decision"]
                        if not isinstance(decision, dict) or not (
                            decision.get("candidate_id") == path.stem
                            and decision.get("eligible_for_performance") is True
                            and decision.get("blocking_reasons") == []
                        ):
                            raise ValueError("checkpoint correctness decision is inconsistent")
            return path, actual_hash, candidate_id, generation

        restored_frontier = [restore_entry(entry) for entry in state["frontier"]]
        if not 1 <= len(restored_frontier) <= beam_width:
            raise ValueError("checkpoint frontier width is invalid")
        initial_hash = expected["initial_parent"]["code_hash"]
        replay_frontier = [(parent, initial_hash, f"seed-{initial_hash[:12]}", 0)]
        replay_seen = {expected["initial_parent"]["code_hash"]}
        replay_last = parent
        replay_cursor = 0
        non_admitted_statuses = {
            "generation_failed",
            "duplicate_code_hash",
            "correctness_failed",
            "correctness_unknown",
            "correctness_oracle_error",
        }
        for record in records:
            source_index = replay_cursor % len(replay_frontier)
            if saved_path(record["source_parent_path"]) != replay_frontier[source_index][0]:
                raise ValueError("checkpoint source parent history is inconsistent")
            action = record.get("frontier_action")
            if action == "none":
                if record.get("status") not in non_admitted_statuses:
                    raise ValueError("checkpoint rejection history is inconsistent")
            elif action in {"append", "replace"}:
                expected_status = (
                    "correctness_pass"
                    if admission_mode == "correctness"
                    else "static_import_pass"
                )
                if record.get("status") != expected_status:
                    raise ValueError("checkpoint admission status is inconsistent")
                candidate_path = saved_path(record["candidate_path"])
                candidate = restore_entry(
                    {"path": str(candidate_path), "code_hash": record["candidate_code_hash"]}
                )
                manifest = json.loads(
                    candidate_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
                )
                replay_parent = replay_frontier[source_index]
                if not (
                    record.get("candidate_id") == candidate_path.stem
                    and saved_path(record["manifest_path"])
                    == candidate_path.with_suffix(".manifest.json").resolve()
                    and record.get("correctness_status")
                    == ("passed" if admission_mode == "correctness" else None)
                    and manifest.get("random_seed") == record["random_seed"]
                    and saved_path(manifest["parent_path"]) == replay_parent[0]
                    and manifest.get("parent_sha256") == replay_parent[1]
                    and manifest["candidate"].get("parent_ids") == [replay_parent[2]]
                    and manifest["candidate"].get("generation") == replay_parent[3] + 1
                ):
                    raise ValueError("checkpoint candidate provenance is inconsistent")
                if beam_width > 1 and candidate[1] in replay_seen:
                    raise ValueError("checkpoint history admits a duplicate code hash")
                if action == "append":
                    if len(replay_frontier) >= beam_width:
                        raise ValueError("checkpoint append exceeds frontier width")
                    replay_frontier.append(candidate)
                else:
                    if len(replay_frontier) != beam_width:
                        raise ValueError("checkpoint replace occurred before frontier was full")
                    replay_frontier[source_index] = candidate
                replay_seen.add(candidate[1])
                replay_last = candidate[0]
            else:
                raise ValueError("checkpoint frontier action is invalid")
            replay_cursor = (source_index + 1) % len(replay_frontier)

        seen_hashes = state["seen_code_hashes"]
        if (
            not isinstance(seen_hashes, list)
            or len(seen_hashes) != len(set(seen_hashes))
            or set(seen_hashes) != replay_seen
            or restored_frontier != replay_frontier
        ):
            raise ValueError("checkpoint frontier or admitted hash history is inconsistent")
        last_entry = restore_entry(state["last_admitted_parent"])
        last_admitted_parent, last_hash = last_entry[:2]
        if last_admitted_parent != replay_last or last_hash not in replay_seen:
            raise ValueError("checkpoint last admitted parent is inconsistent")
        next_parent_index = state["next_parent_index"]
        if (
            not isinstance(next_parent_index, int)
            or next_parent_index != replay_cursor
            or not 0 <= next_parent_index < len(restored_frontier)
        ):
            raise ValueError("checkpoint parent cursor is invalid")
        in_flight = state.get("in_flight")
        if in_flight is None:
            if retry_in_flight:
                raise ValueError("retry-in-flight requires an unresolved attempt")
        else:
            expected_source = _checkpoint_entry(restored_frontier[next_parent_index][0])
            expected_fields = {
                "ordinal": next_ordinal,
                "random_seed": random_seed + next_ordinal,
                "source_parent": expected_source,
            }
            attempts = in_flight.get("physical_attempts_maybe_started")
            artifacts_before = in_flight.get("artifacts_before")
            if (
                any(in_flight.get(key) != value for key, value in expected_fields.items())
                or not isinstance(attempts, int)
                or attempts < 1
                or not isinstance(in_flight.get("explicit_retry"), bool)
                or in_flight["explicit_retry"] is not (attempts > 1)
                or not isinstance(artifacts_before, dict)
                or not all(
                    isinstance(name, str) and isinstance(code_hash, str)
                    for name, code_hash in artifacts_before.items()
                )
            ):
                raise ValueError("checkpoint in-flight attempt is inconsistent")
            if artifacts_before != _candidate_artifacts(output_dir, operator):
                raise ValueError("checkpoint in-flight attempt has ambiguous new artifacts")
            if not retry_in_flight:
                raise ValueError("checkpoint has an unresolved in-flight attempt")
            if budget_controller is not None:
                raise ValueError(
                    "budgeted checkpoint has an unresolved in-flight attempt"
                )
        if budget_controller is not None:
            budget_checkpoint = state.get("budget_checkpoint")
            _validate_budget_history(
                state,
                records,
                dict(budget_controller.snapshot().limits),
            )
            budget_controller.restore(budget_checkpoint)
    except (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"Invalid mutation checkpoint: {exc}") from exc
    return {
        "frontier": [entry[0] for entry in restored_frontier],
        "seen_hashes": set(seen_hashes),
        "last_admitted_parent": last_admitted_parent,
        "next_parent_index": next_parent_index,
        "next_ordinal": next_ordinal,
        "records": records,
        "in_flight": in_flight,
        "budget_denial_reason": budget_denial_reason,
    }


def _run_mutation_loop(
    work_dir: Path,
    datasets_dir: Path,
    output_dir: Path,
    operator: str,
    parent_path: Path,
    random_seed: int,
    max_new_calls: int,
    generate_fn: Callable[..., dict] = generate_candidate,
    correctness_gate: Optional[Callable[[Path], dict]] = None,
    beam_width: int = 1,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
    retry_in_flight: bool = False,
    parent_policy: Optional[str] = None,
    budget_controller: Optional[BudgetController] = None,
) -> dict:
    """Expand a bounded admitted frontier without local performance ranking."""

    if max_new_calls < 0:
        raise ValueError("max_new_calls must be non-negative")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if beam_width > 1 and correctness_gate is None:
        raise ValueError("beam_width > 1 requires a correctness gate")
    if retry_in_flight and not resume:
        raise ValueError("retry_in_flight requires resume")
    if retry_in_flight and max_new_calls < 1:
        raise ValueError("retry_in_flight requires at least one new call")
    parent = parent_path.resolve()
    if parent.parent != (datasets_dir / operator).resolve():
        raise ValueError("initial mutation parent must be an operator dataset seed")
    admission_mode = "correctness" if correctness_gate is not None else "static_import"
    admission_policy_id = _ADMISSION_POLICY_IDS.get(
        operator, f"{operator}:{admission_mode}:v1"
    )
    checkpoint_path = checkpoint_path.resolve() if checkpoint_path is not None else None
    if resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise ValueError("resume requires an existing mutation checkpoint")
        restored = _restore_mutation_checkpoint(
            checkpoint_path,
            operator,
            work_dir,
            datasets_dir,
            parent,
            output_dir,
            random_seed,
            beam_width,
            admission_mode,
            admission_policy_id,
            retry_in_flight,
            budget_controller,
        )
        frontier = restored["frontier"]
        seen_hashes = restored["seen_hashes"]
        last_admitted_parent = restored["last_admitted_parent"]
        next_parent_index = restored["next_parent_index"]
        next_ordinal = restored["next_ordinal"]
        records = restored["records"]
        in_flight = restored["in_flight"]
        budget_denial_reason = restored["budget_denial_reason"]
        retry_pending = in_flight is not None
    else:
        if checkpoint_path is not None and checkpoint_path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
        frontier = [parent]
        last_admitted_parent = parent
        seen_hashes = {_checkpoint_entry(parent)["code_hash"]}
        next_parent_index = 0
        next_ordinal = 0
        records = []
        in_flight = None
        budget_denial_reason = None
        retry_pending = False
    first_new_record = len(records)
    physical_attempts_started_this_run = 0

    def build_report() -> dict:
        possible_physical_attempts = sum(
            record.get("physical_attempts_maybe_started", 1) for record in records
        )
        if in_flight is not None:
            possible_physical_attempts += in_flight["physical_attempts_maybe_started"]
        counts = {}
        for record in records:
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        last_run_counts = {}
        for record in records[first_new_record:]:
            last_run_counts[record["status"]] = last_run_counts.get(record["status"], 0) + 1
        report = {
            "schema_version": 2,
            "artifact_kind": "official-mutation-loop-report",
            "operator": operator,
            "beam_width": beam_width,
            "admission_mode": admission_mode,
            "admission_policy_id": admission_policy_id,
            "work_dir": str(work_dir.resolve()),
            "datasets_dir": str(datasets_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "base_random_seed": random_seed,
            "initial_parent": _checkpoint_entry(parent),
            "next_ordinal": next_ordinal,
            "next_parent_index": next_parent_index,
            "in_flight": in_flight,
            "frontier": [_checkpoint_entry(path) for path in frontier],
            "last_admitted_parent": _checkpoint_entry(last_admitted_parent),
            "seen_code_hashes": sorted(seen_hashes),
            "max_new_calls": max_new_calls,
            "calls_started": possible_physical_attempts,
            "calls_started_this_run": physical_attempts_started_this_run,
            "logical_calls_completed": len(records),
            "status_counts": counts,
            "last_run_status_counts": last_run_counts,
            "records": records,
            "final_admitted_parent_path": _display_path(last_admitted_parent),
            "final_admitted_parent_paths": [_display_path(path) for path in frontier],
        }
        if parent_policy is not None:
            report["parent_policy"] = parent_policy
        if budget_controller is not None:
            report["budget_limits"] = dict(budget_controller.snapshot().limits)
            report["budget_checkpoint"] = dict(budget_controller.snapshot())
            report["budget_denial_reason"] = budget_denial_reason
        return report

    if checkpoint_path is not None:
        _write_json_atomic(checkpoint_path, build_report())
    for _ in range(max_new_calls):
        if budget_denial_reason is not None or (
            budget_controller is not None
            and budget_controller.snapshot().stop_reason is not None
        ):
            break
        retrying_attempt = retry_pending
        ordinal = next_ordinal
        attempt_seed = random_seed + ordinal
        source_parent_index = next_parent_index % len(frontier)
        source_parent = frontier[source_parent_index]
        if retrying_attempt:
            in_flight = dict(in_flight)
            in_flight["physical_attempts_maybe_started"] += 1
            in_flight["explicit_retry"] = True
        else:
            in_flight = {
                "ordinal": ordinal,
                "random_seed": attempt_seed,
                "source_parent": _checkpoint_entry(source_parent),
                "physical_attempts_maybe_started": 1,
                "explicit_retry": False,
                "artifacts_before": _candidate_artifacts(output_dir, operator),
            }
        physical_attempts_started_this_run += 1
        if checkpoint_path is not None:
            _write_json_atomic(checkpoint_path, build_report())
        attempt_provenance = {
            "retried_in_flight": retrying_attempt,
            "physical_attempts_maybe_started": in_flight[
                "physical_attempts_maybe_started"
            ],
            "prior_token_usage": (
                "unknown_may_have_been_consumed" if retrying_attempt else None
            ),
        }
        if budget_controller is not None:
            attempt_provenance["budget_before"] = _budget_attempt_state(
                budget_controller
            )
        records_before_attempt = len(records)
        try:
            generate_kwargs = (
                {"budget_controller": budget_controller}
                if budget_controller is not None
                else {}
            )
            manifest = generate_fn(
                work_dir,
                datasets_dir,
                operator,
                source_parent,
                output_dir,
                attempt_seed,
                **generate_kwargs,
            )
            candidate_id = manifest["candidate"]["id"]
            manifest_path = output_dir / operator / f"{candidate_id}.manifest.json"
            if not _is_static_pass_manifest(manifest_path, operator):
                raise ValueError("Generator returned without a valid static/import-pass child")
            candidate_path = manifest_path.with_name(f"{candidate_id}.py").resolve()
            candidate_hash = sha256_text(candidate_path.read_text(encoding="utf-8"))
            llm_evidence = _llm_attempt_evidence(manifest)
            correctness_evaluation = None
            if correctness_gate is not None:
                try:
                    correctness_evaluation = correctness_gate(manifest_path)
                    if not isinstance(correctness_evaluation, dict):
                        raise TypeError("correctness gate must return a mapping")
                except Exception as exc:
                    _reject_manifest(
                        manifest_path,
                        f"correctness gate error: {type(exc).__name__}: {str(exc)[:400]}",
                    )
                    raise
                _persist_correctness_evaluation(
                    manifest_path, correctness_evaluation
                )
                correctness_status = correctness_evaluation.get("status", "unknown")
                eligible = correctness_evaluation.get("eligible_for_performance") is True
                valid_pass = (
                    correctness_status == "passed"
                    and eligible
                    and correctness_evaluation.get("blocking_reasons") == []
                )
                valid_rejection = (
                    correctness_status in {"failed", "unknown", "oracle_error"}
                    and not eligible
                )
                if not valid_pass and not valid_rejection:
                    _reject_manifest(manifest_path, "invalid correctness gate result")
                    raise ValueError("correctness gate returned inconsistent admission fields")
                if valid_rejection:
                    records.append(
                        {
                            "ordinal": ordinal,
                            "random_seed": attempt_seed,
                            "status": f"correctness_{correctness_status}",
                            "candidate_id": candidate_id,
                            "candidate_code_hash": candidate_hash,
                            "source_parent_path": _display_path(source_parent),
                            "candidate_path": _display_path(candidate_path),
                            "manifest_path": _display_path(manifest_path),
                            "correctness_status": correctness_status,
                            "frontier_action": "none",
                            "llm": llm_evidence,
                            "budget_after": _budget_attempt_state(budget_controller),
                            **attempt_provenance,
                        }
                    )
                    continue
            if beam_width > 1 and candidate_hash in seen_hashes:
                _reject_manifest(manifest_path, "duplicate admitted code hash")
                records.append(
                    {
                        "ordinal": ordinal,
                        "random_seed": attempt_seed,
                        "status": "duplicate_code_hash",
                        "candidate_id": candidate_id,
                        "candidate_code_hash": candidate_hash,
                        "source_parent_path": _display_path(source_parent),
                        "candidate_path": _display_path(candidate_path),
                        "manifest_path": _display_path(manifest_path),
                        "correctness_status": (
                            correctness_evaluation.get("status")
                            if correctness_evaluation is not None
                            else None
                        ),
                        "frontier_action": "none",
                        "llm": llm_evidence,
                        "budget_after": _budget_attempt_state(budget_controller),
                        **attempt_provenance,
                    }
                )
                continue
            if len(frontier) < beam_width:
                frontier.append(candidate_path)
                frontier_action = "append"
            else:
                frontier[source_parent_index] = candidate_path
                frontier_action = "replace"
            seen_hashes.add(candidate_hash)
            last_admitted_parent = candidate_path
            records.append(
                {
                    "ordinal": ordinal,
                    "random_seed": attempt_seed,
                    "status": (
                        "correctness_pass"
                        if correctness_evaluation is not None
                        else "static_import_pass"
                    ),
                    "candidate_id": candidate_id,
                    "candidate_code_hash": candidate_hash,
                    "source_parent_path": _display_path(source_parent),
                    "candidate_path": _display_path(candidate_path),
                    "manifest_path": _display_path(manifest_path),
                    "correctness_status": (
                        correctness_evaluation.get("status")
                        if correctness_evaluation is not None
                        else None
                    ),
                    "frontier_action": frontier_action,
                    "llm": llm_evidence,
                    "budget_after": _budget_attempt_state(budget_controller),
                    **attempt_provenance,
                }
            )
        except Exception as exc:
            created_manifest_path, created_manifest, new_artifacts = (
                _new_attempt_manifest(
                    output_dir,
                    operator,
                    in_flight["artifacts_before"],
                )
            )
            created_candidate = (
                created_manifest.get("candidate", {})
                if isinstance(created_manifest, dict)
                else {}
            )
            llm_evidence = _llm_attempt_evidence(created_manifest, exc)
            if (
                budget_controller is not None
                and isinstance(llm_evidence, dict)
                and llm_evidence.get("error_type") == "budget_denied"
            ):
                message = str(exc)
                prefix = "LLM budget denied request:"
                budget_denial_reason = (
                    message[len(prefix):].strip()
                    if message.startswith(prefix)
                    else "budget_denied"
                )
            records.append(
                {
                    "ordinal": ordinal,
                    "random_seed": attempt_seed,
                    "status": "generation_failed",
                    "source_parent_path": _display_path(source_parent),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                    "frontier_action": "none",
                    "candidate_id": created_candidate.get("id"),
                    "candidate_code_hash": created_candidate.get("code_hash"),
                    "manifest_path": (
                        _display_path(created_manifest_path)
                        if created_manifest_path is not None
                        else None
                    ),
                    "new_artifacts": new_artifacts,
                    "llm": llm_evidence,
                    "budget_after": _budget_attempt_state(budget_controller),
                    **attempt_provenance,
                }
            )
        finally:
            if len(records) == records_before_attempt + 1:
                next_parent_index = (source_parent_index + 1) % len(frontier)
                next_ordinal = ordinal + 1
                in_flight = None
                retry_pending = False
                if checkpoint_path is not None:
                    _write_json_atomic(checkpoint_path, build_report())
    return build_report()


def run_mutation_loop(
    work_dir: Path,
    datasets_dir: Path,
    output_dir: Path,
    operator: str,
    parent_path: Path,
    random_seed: int,
    max_new_calls: int,
    generate_fn: Callable[..., dict] = generate_candidate,
    correctness_gate: Optional[Callable[[Path], dict]] = None,
    beam_width: int = 1,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
    retry_in_flight: bool = False,
    parent_policy: Optional[str] = None,
    budget_controller: Optional[BudgetController] = None,
) -> dict:
    """Run one mutation loop while excluding concurrent checkpoint writers."""

    with _checkpoint_writer_lock(checkpoint_path):
        return _run_mutation_loop(
            work_dir,
            datasets_dir,
            output_dir,
            operator,
            parent_path,
            random_seed,
            max_new_calls,
            generate_fn,
            correctness_gate,
            beam_width,
            checkpoint_path,
            resume,
            retry_in_flight,
            parent_policy,
            budget_controller,
        )


def _persist_correctness_evaluation(
    manifest_path: Path, correctness_evaluation: dict
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["correctness_evaluation"] = correctness_evaluation
    if correctness_evaluation.get("eligible_for_performance") is not True:
        status = correctness_evaluation.get("status", "unknown")
        candidate = dict(manifest["candidate"])
        candidate["status"] = "rejected"
        manifest["candidate"] = candidate
        message = correctness_evaluation.get("message") or ",".join(
            correctness_evaluation.get("blocking_reasons", [])
        )
        manifest["rejection_error"] = f"correctness_{status}: {message}"[:500]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reject_manifest(manifest_path: Path, reason: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = dict(manifest["candidate"])
    candidate["status"] = "rejected"
    manifest["candidate"] = candidate
    manifest["rejection_error"] = reason[:500]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _local_per_group_correctness_gate(
    manifest_path: Path, datasets_dir: Path
) -> dict:
    """Run the existing local CUDA/NPU correctness coordinator for one candidate."""

    from wlz_optimizer.candidate_runner import (
        run_per_group_transpose_candidate,
    )
    from wlz_optimizer.case_catalog import materialize_per_group_transpose_public_case
    from wlz_optimizer.correctness_coordinator import evaluate_candidate_correctness
    from wlz_optimizer.correctness_oracle import decode_invocation_snapshot
    from wlz_optimizer.correctness_references import run_case_reference
    from wlz_optimizer.hash_utils import sha256_text
    from wlz_optimizer.schemas import Candidate
    from wlz_optimizer.torch_backend import TorchTensorBackend

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_data = manifest["candidate"]
    candidate_path = manifest_path.with_name(f"{candidate_data['id']}.py")
    code = candidate_path.read_text(encoding="utf-8")
    if sha256_text(code) != candidate_data.get("code_hash"):
        return {
            "status": "unknown",
            "eligible_for_performance": False,
            "blocking_reasons": ["candidate_hash_mismatch"],
            "results": [],
        }
    try:
        candidate = Candidate(
            id=candidate_data["id"],
            op_name=candidate_data["op_name"],
            code=code,
            code_hash=candidate_data["code_hash"],
            parent_ids=list(candidate_data.get("parent_ids", [])),
            generation=int(candidate_data.get("generation", 0)),
            mutation_kind=candidate_data.get("mutation_kind", "mutation"),
            model_used=candidate_data.get("model_used"),
            prompt_id=candidate_data.get("prompt_id"),
            status=candidate_data.get("status", "static_pass"),
            score=None,
            metadata=dict(candidate_data.get("metadata", {})),
        )
        case = materialize_per_group_transpose_public_case(datasets_dir)
        run = evaluate_candidate_correctness(
            candidate,
            [case],
            TorchTensorBackend("cpu"),
            run_case_reference,
            lambda item, item_case, item_inputs: run_per_group_transpose_candidate(
                item, item_case, item_inputs
            ),
            decode_invocation_snapshot,
        )
        status = "passed"
        if not run.decision.eligible_for_performance:
            if "failed_case" in run.decision.blocking_reasons:
                status = "failed"
            elif "oracle_error" in run.decision.blocking_reasons:
                status = "oracle_error"
            else:
                status = "unknown"
        return {
            "status": status,
            "eligible_for_performance": run.decision.eligible_for_performance,
            "blocking_reasons": list(run.decision.blocking_reasons),
            "results": [result.to_dict() for result in run.results],
            "decision": run.decision.to_dict(),
        }
    except Exception as exc:
        return {
            "status": "oracle_error",
            "eligible_for_performance": False,
            "blocking_reasons": ["correctness_gate_error"],
            "results": [],
            "message": f"{type(exc).__name__}: {str(exc)[:500]}",
        }


def _local_cuda_unittest_gate(
    manifest_path: Path,
    *,
    operator: str,
    candidate_env_var: str,
    test_name: str,
    completion_marker: str,
    datasets_dir: Optional[Path] = None,
) -> dict:
    admission_policy_id = _ADMISSION_POLICY_IDS[operator]
    if not _is_static_pass_manifest(manifest_path, operator):
        return {
            "status": "unknown",
            "eligible_for_performance": False,
            "blocking_reasons": ["invalid_candidate_manifest"],
            "results": [],
            "admission_policy_id": admission_policy_id,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_id = manifest["candidate"]["id"]
    candidate_path = manifest_path.with_name(f"{candidate_id}.py").resolve()
    python_executable = Path(
        os.environ.get("WLZ_TRITON_PYTHON", sys.executable)
    ).resolve()
    if not python_executable.is_file():
        return {
            "status": "unknown",
            "eligible_for_performance": False,
            "blocking_reasons": ["cuda_python_not_configured"],
            "results": [],
            "admission_policy_id": admission_policy_id,
        }
    environment = os.environ.copy()
    environment[candidate_env_var] = str(candidate_path)
    if datasets_dir is not None:
        environment["WLZ_CORRECTNESS_DATASETS_DIR"] = str(datasets_dir.resolve())
    command = [str(python_executable), "-m", "unittest", test_name, "-v"]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "oracle_error",
            "eligible_for_performance": False,
            "blocking_reasons": ["correctness_gate_error"],
            "results": [],
            "message": f"{type(exc).__name__}: {str(exc)[:500]}",
            "admission_policy_id": admission_policy_id,
        }
    matrix_completed = completion_marker in completed.stdout
    passed = completed.returncode == 0 and matrix_completed
    return {
        "status": "passed" if passed else "failed",
        "eligible_for_performance": passed,
        "blocking_reasons": [] if passed else ["failed_case"],
        "admission_policy_id": admission_policy_id,
        "evidence_scope": "local_cuda_proxy_only_not_ascend_or_official",
        "results": [
            {
                "test": test_name,
                "returncode": completed.returncode,
                "matrix_completed": matrix_completed,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        ],
    }


def _local_selective_correctness_gate(manifest_path: Path) -> dict:
    """Run the bounded local CUDA matrix without treating it as Ascend evidence."""

    return _local_cuda_unittest_gate(
        manifest_path,
        operator="_selective_scan_update_kernel",
        candidate_env_var="WLZ_SELECTIVE_CANDIDATE",
        test_name=_SELECTIVE_CUDA_TEST,
        completion_marker=_SELECTIVE_CUDA_COMPLETION_MARKER,
    )


def _local_quantize_k_cache_correctness_gate(
    manifest_path: Path, datasets_dir: Path
) -> dict:
    """Run the two public quantize assertions on an isolated local CUDA worker."""

    return _local_cuda_unittest_gate(
        manifest_path,
        operator="_quantize_k_cache_fast_kernel",
        candidate_env_var="WLZ_QUANTIZE_K_CACHE_CANDIDATE",
        test_name=_QUANTIZE_CUDA_TEST,
        completion_marker=_QUANTIZE_CUDA_COMPLETION_MARKER,
        datasets_dir=datasets_dir,
    )


def _local_act_quant_correctness_gate(
    manifest_path: Path, datasets_dir: Path
) -> dict:
    """Run both public act-quant invocations on an isolated local CUDA worker."""

    return _local_cuda_unittest_gate(
        manifest_path,
        operator="_act_quant_kernel",
        candidate_env_var="WLZ_ACT_QUANT_CANDIDATE",
        test_name=_ACT_QUANT_CUDA_TEST,
        completion_marker=_ACT_QUANT_CUDA_COMPLETION_MARKER,
        datasets_dir=datasets_dir,
    )


def _local_count_expert_correctness_gate(
    manifest_path: Path, datasets_dir: Path
) -> dict:
    """Run the basic/no-map count-expert assertion on isolated local CUDA."""

    return _local_cuda_unittest_gate(
        manifest_path,
        operator="_count_expert_num_tokens",
        candidate_env_var="WLZ_COUNT_EXPERT_CANDIDATE",
        test_name=_COUNT_EXPERT_CUDA_TEST,
        completion_marker=_COUNT_EXPERT_CUDA_COMPLETION_MARKER,
        datasets_dir=datasets_dir,
    )


def _local_set_k_and_s_correctness_gate(
    manifest_path: Path, datasets_dir: Path
) -> dict:
    """Run the public set-k-and-s case with allocation guards on local CUDA."""

    return _local_cuda_unittest_gate(
        manifest_path,
        operator="_set_k_and_s_triton_kernel",
        candidate_env_var="WLZ_SET_K_AND_S_CANDIDATE",
        test_name=_SET_K_AND_S_CUDA_TEST,
        completion_marker=_SET_K_AND_S_CUDA_COMPLETION_MARKER,
        datasets_dir=datasets_dir,
    )


def _correctness_gate_for(operator: str, datasets_dir: Path):
    if operator == "_act_quant_kernel":
        return lambda manifest_path: _local_act_quant_correctness_gate(
            manifest_path, datasets_dir
        )
    if operator == "_count_expert_num_tokens":
        return lambda manifest_path: _local_count_expert_correctness_gate(
            manifest_path, datasets_dir
        )
    if operator == "_per_group_transpose":
        return lambda manifest_path: _local_per_group_correctness_gate(
            manifest_path, datasets_dir
        )
    if operator == "_selective_scan_update_kernel":
        return _local_selective_correctness_gate
    if operator == "_quantize_k_cache_fast_kernel":
        return lambda manifest_path: _local_quantize_k_cache_correctness_gate(
            manifest_path, datasets_dir
        )
    if operator == "_set_k_and_s_triton_kernel":
        return lambda manifest_path: _local_set_k_and_s_correctness_gate(
            manifest_path, datasets_dir
        )
    return None


def _select_parent(datasets_dir: Path, operator: str, contract_module) -> tuple[Path, str]:
    operator_dir = datasets_dir / operator
    baseline = operator_dir / f"{operator}.py"
    variant = operator_dir / f"{operator}_1.py"
    if variant.is_file():
        error = contract_module.interface_contract_error(
            baseline.read_text(encoding="utf-8"),
            variant.read_text(encoding="utf-8"),
        )
        target_device_ok = validate_static_structure(
            variant.read_text(encoding="utf-8"), []
        )["target_device_ok"]
        if error is None and target_device_ok:
            return variant, "compatible_variant"
        if error is None and not target_device_ok:
            return baseline, "baseline_after_target_device_preflight"
    return baseline, "baseline"


def run_batch(
    work_dir: Path,
    datasets_dir: Path,
    output_dir: Path,
    random_seed: int,
    max_new_calls: int,
    kernels: Optional[list[str]] = None,
    dry_run: bool = False,
    generate_fn: Callable[..., dict] = generate_candidate,
    contract_module=None,
) -> dict:
    if max_new_calls < 0:
        raise ValueError("max_new_calls must be non-negative")
    operators = discover_operators(datasets_dir)
    if kernels:
        unknown = sorted(set(kernels) - set(operators))
        if unknown:
            raise ValueError(f"Unknown kernel(s): {', '.join(unknown)}")
        selected = set(kernels)
        operators = [operator for operator in operators if operator in selected]
    if contract_module is None:
        contract_module = _load_official_modules(work_dir.resolve())[2]

    records = []
    calls_started = 0
    for operator in operators:
        existing = _find_static_pass_manifest(output_dir, operator)
        if existing is not None:
            records.append(
                {
                    "operator": operator,
                    "status": "skipped_static_pass",
                    "manifest_path": _display_path(existing),
                }
            )
            continue

        parent, parent_policy = _select_parent(datasets_dir, operator, contract_module)
        base_record = {
            "operator": operator,
            "parent_path": _display_path(parent),
            "parent_policy": parent_policy,
            "random_seed": random_seed,
        }
        if dry_run:
            records.append({**base_record, "status": "planned"})
            continue
        if calls_started >= max_new_calls:
            records.append({**base_record, "status": "deferred_call_budget"})
            continue

        calls_started += 1
        try:
            manifest = generate_fn(
                work_dir,
                datasets_dir,
                operator,
                parent,
                output_dir,
                random_seed,
            )
        except Exception as exc:
            records.append(
                {
                    **base_record,
                    "status": "generation_failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )
            continue
        persisted = _find_static_pass_manifest(output_dir, operator)
        if persisted is None:
            records.append(
                {
                    **base_record,
                    "status": "generation_failed",
                    "error_type": "PersistenceError",
                    "error_message": "Generator returned without a valid persisted candidate",
                }
            )
            continue
        records.append(
            {
                **base_record,
                "status": "generated_static_pass",
                "candidate_id": manifest["candidate"]["id"],
                "manifest_path": _display_path(persisted),
            }
        )

    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return {
        "schema_version": 1,
        "artifact_kind": "official-candidate-batch-report",
        "dry_run": dry_run,
        "max_new_calls": max_new_calls,
        "calls_started": calls_started,
        "operator_count": len(operators),
        "status_counts": counts,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-seed", type=int, default=5)
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--remaining-token-budget", type=int)
    parser.add_argument("--remaining-wall-seconds", type=float)
    parser.add_argument("--kernel", action="append", dest="kernels")
    parser.add_argument(
        "--mutation-loop",
        action="store_true",
        help="Chain accepted mutations for exactly one --kernel",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-in-flight",
        action="store_true",
        help="Explicitly retry one unresolved attempt; prior model quota may be consumed",
    )
    args = parser.parse_args()
    if args.max_new_calls < 0:
        parser.error("--max-new-calls must be non-negative")
    if (args.remaining_token_budget is None) != (
        args.remaining_wall_seconds is None
    ):
        parser.error(
            "--remaining-token-budget and --remaining-wall-seconds must be supplied together"
        )
    if args.remaining_token_budget is not None and not args.mutation_loop:
        parser.error("fresh budget options require --mutation-loop")
    budget_controller = None
    if args.remaining_token_budget is not None:
        try:
            budget_controller = BudgetController(
                BudgetLimits(
                    args.remaining_token_budget,
                    args.remaining_wall_seconds,
                )
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
    if args.mutation_loop:
        if budget_controller is None:
            parser.error(
                "--mutation-loop requires --remaining-token-budget and "
                "--remaining-wall-seconds"
            )
        if not args.kernels or len(args.kernels) != 1:
            parser.error("--mutation-loop requires exactly one --kernel")
        if args.dry_run:
            parser.error("--mutation-loop cannot be combined with --dry-run")
        if args.resume and not args.report:
            parser.error("--resume requires --report")
        if args.retry_in_flight and not args.resume:
            parser.error("--retry-in-flight requires --resume")
        if args.retry_in_flight and args.max_new_calls < 1:
            parser.error("--retry-in-flight requires --max-new-calls >= 1")
        if args.resume and not Path(args.report).is_file():
            parser.error("--resume requires an existing --report checkpoint")
        if not args.resume and args.report and Path(args.report).exists():
            parser.error("fresh mutation loop refuses to overwrite --report")
        datasets_dir = Path(args.datasets_dir)
        contract_module = _load_official_modules(Path(args.work_dir).resolve())[2]
        operator = args.kernels[0]
        parent, parent_policy = _select_parent(datasets_dir, operator, contract_module)
        correctness_gate = _correctness_gate_for(operator, datasets_dir)
        report = run_mutation_loop(
            Path(args.work_dir),
            datasets_dir,
            Path(args.output_dir),
            operator,
            parent,
            args.random_seed,
            args.max_new_calls,
            correctness_gate=correctness_gate,
            beam_width=2 if correctness_gate is not None else 1,
            checkpoint_path=Path(args.report) if args.report else None,
            resume=args.resume,
            retry_in_flight=args.retry_in_flight,
            parent_policy=parent_policy,
            budget_controller=budget_controller,
        )
        rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        print(rendered, end="")
        return 1 if report["last_run_status_counts"].get("generation_failed") else 0
    if args.resume or args.retry_in_flight:
        parser.error("--resume/--retry-in-flight require --mutation-loop")
    report = run_batch(
        Path(args.work_dir),
        Path(args.datasets_dir),
        Path(args.output_dir),
        args.random_seed,
        args.max_new_calls,
        args.kernels,
        args.dry_run,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.report:
        report_path = Path(args.report)
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite report: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if report["status_counts"].get("generation_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
