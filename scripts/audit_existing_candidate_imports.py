#!/usr/bin/env python3
"""Audit existing candidate manifests with the isolated import-only runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.candidate_runner import CandidateRunRequest, run_candidate
from wlz_optimizer.executors import LocalExecutor
from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.io_utils import discover_operators, load_operator_input
from wlz_optimizer.schemas import Candidate, EvalContext


def _record_error(manifest_path: Path, error_type: str, message: str) -> Dict[str, Any]:
    return {
        "manifest_path": str(manifest_path),
        "status": "manifest_error",
        "error_type": error_type,
        "error_message": message[:512],
    }


def _recovery_recommendation(record: Dict[str, Any]) -> str:
    if record["status"] == "manifest_error":
        return "requires_manifest_repair"
    if record.get("static_passed") is False:
        return "requires_candidate_repair"
    if record["status"] == "imported":
        return "eligible_for_manifest_backfill_review"
    if record.get("error_type") == "ModuleNotFoundError":
        return "requires_target_environment_recheck"
    return "requires_manual_review"


def _backfill_import_pass(record: Dict[str, Any], python_executable: str) -> None:
    """Recover only dependency-missing import failures after a passing recheck."""

    if (
        record.get("status") != "imported"
        or record.get("phase") != "module_import"
        or record.get("static_passed") is not True
    ):
        raise ValueError("audit record is not an import/static pass")
    manifest_path = Path(record["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = manifest.get("candidate") or {}
    previous = manifest.get("import_evaluation") or {}
    if (
        candidate.get("status") != "rejected"
        or candidate.get("code_hash") != record.get("code_hash")
        or manifest.get("static_evaluation", {}).get("passed") is not True
        or previous.get("status") != "import_error"
        or previous.get("phase") != "module_import"
        or previous.get("error_type") != "ModuleNotFoundError"
        or not str(manifest.get("rejection_error", "")).startswith(
            "Generated candidate failed import gate:"
        )
    ):
        raise ValueError("manifest is not an eligible dependency-missing import rejection")

    manifest.setdefault("import_evaluation_history", []).append(previous)
    manifest["import_evaluation"] = {
        "status": "imported",
        "phase": "module_import",
        "error_type": None,
        "error_message": None,
        "python_executable": python_executable,
    }
    candidate["status"] = "static_pass"
    manifest["candidate"] = candidate
    manifest["rejection_error"] = None
    manifest.setdefault("recovery_history", []).append(
        {
            "kind": "target_environment_import_recheck",
            "code_hash": record["code_hash"],
            "python_executable": python_executable,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_candidates(
    candidate_root: Path,
    datasets_dir: Path,
    timeout_seconds: float = 20.0,
    backfill_import_pass: bool = False,
) -> Dict[str, Any]:
    """Audit imports and optionally recover narrowly eligible dependency failures."""

    candidate_root = candidate_root.resolve()
    datasets_dir = datasets_dir.resolve()
    records: List[Dict[str, Any]] = []
    for operator in discover_operators(datasets_dir):
        operator_dir = candidate_root / operator
        try:
            operator_input = load_operator_input(datasets_dir, operator)
        except Exception as exc:
            records.append(_record_error(operator_dir, type(exc).__name__, str(exc)))
            continue
        for manifest_path in sorted(operator_dir.glob("*.manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                candidate = manifest["candidate"]
                candidate_id = candidate["id"]
                code_path = manifest_path.with_name(f"{candidate_id}.py")
                code = code_path.read_text(encoding="utf-8")
                manifest_hash = candidate["code_hash"]
            except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                records.append(_record_error(manifest_path, type(exc).__name__, str(exc)))
                continue

            actual_hash = sha256_text(code)
            base = {
                "operator": operator,
                "candidate_id": candidate_id,
                "manifest_path": str(manifest_path),
                "code_path": str(code_path),
                "manifest_code_hash": manifest_hash,
                "code_hash": actual_hash,
            }
            if manifest_hash != actual_hash:
                records.append({
                    **base,
                    "status": "manifest_error",
                    "error_type": "CodeHashMismatch",
                    "error_message": "manifest candidate.code_hash does not match source",
                })
                continue

            static_result = LocalExecutor().evaluate(
                Candidate(
                    id=candidate_id,
                    op_name=operator,
                    code=code,
                    code_hash=actual_hash,
                    parent_ids=candidate.get("parent_ids", []),
                    generation=candidate.get("generation", 0),
                    mutation_kind=candidate.get("mutation_kind", "historical_audit"),
                    model_used=candidate.get("model_used"),
                    prompt_id=candidate.get("prompt_id"),
                    status=candidate.get("status", "historical"),
                    score=candidate.get("score"),
                    metadata=candidate.get("metadata", {}),
                ),
                EvalContext(
                    op_name=operator,
                    input_dir=datasets_dir,
                    output_dir=candidate_root,
                    required_functions=operator_input.required_functions,
                    test_file=operator_input.test_file,
                    baseline_file=operator_input.baseline_file,
                ),
            )
            try:
                result = run_candidate(
                    CandidateRunRequest(
                        candidate_id=candidate_id,
                        code=code,
                        code_hash=actual_hash,
                        entrypoint=(operator_input.required_functions or [operator])[0],
                        payload={},
                        timeout_seconds=timeout_seconds,
                        stop_after_import=True,
                    )
                )
                records.append({
                    **base,
                    "status": result.status,
                    "phase": result.phase,
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                    "static_passed": static_result.passed,
                    "static_error_type": static_result.error_type,
                    "target_device_ok": static_result.metadata.get("target_device_ok"),
                    "target_device_errors": static_result.metadata.get("target_device_errors", []),
                    "triton_semantic_errors": static_result.metadata.get(
                        "triton_semantic_errors", []
                    ),
                })
            except Exception as exc:
                records.append(_record_error(manifest_path, type(exc).__name__, str(exc)))

    status_counts: Dict[str, int] = {}
    recovery_counts: Dict[str, int] = {}
    for record in records:
        status = record["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        recommendation = _recovery_recommendation(record)
        record["recovery_recommendation"] = recommendation
        recovery_counts[recommendation] = recovery_counts.get(recommendation, 0) + 1
        if backfill_import_pass and recommendation == "eligible_for_manifest_backfill_review":
            try:
                _backfill_import_pass(record, sys.executable)
                record["manifest_backfilled"] = True
            except Exception as exc:
                record["manifest_backfilled"] = False
                record["backfill_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    backfilled_count = sum(record.get("manifest_backfilled") is True for record in records)
    backfill_error_count = sum(record.get("manifest_backfilled") is False for record in records)
    return {
        "schema_version": 1,
        "artifact_kind": "existing-candidate-import-audit",
        "candidate_root": str(candidate_root),
        "datasets_dir": str(datasets_dir),
        "python_executable": sys.executable,
        "timeout_seconds": timeout_seconds,
        "backfill_import_pass": backfill_import_pass,
        "backfilled_candidate_count": backfilled_count,
        "backfill_error_count": backfill_error_count,
        "candidate_count": len(records),
        "status_counts": status_counts,
        "recovery_counts": recovery_counts,
        "recoverable_operator_count": len({
            record["operator"]
            for record in records
            if record.get("recovery_recommendation")
            == "eligible_for_manifest_backfill_review"
        }),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--backfill-import-pass", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = audit_candidates(
        Path(args.candidate_root),
        Path(args.datasets_dir),
        args.timeout_seconds,
        args.backfill_import_pass,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.report:
        report_path = Path(args.report)
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite report: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if report["backfill_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
