#!/usr/bin/env python3
"""Attach verified local evidence to a candidate without overwriting inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.candidate_runner import CandidateRunRequest, run_candidate

ADMISSION_POLICY = "local_ascend_910b4_manual_evidence_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence must be inside repository: {path}") from exc


def _output_reference(path: Path) -> str:
    try:
        return _relative(path)
    except ValueError:
        return str(path.resolve())


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return value


def _identity(document: dict, candidate: dict, baseline_sha256: str, path: Path) -> None:
    record = document.get("candidate")
    baseline = document.get("baseline")
    if not isinstance(record, dict) or not isinstance(baseline, dict):
        raise ValueError(f"evidence lacks candidate/baseline identity: {path}")
    if record.get("id") != candidate.get("id") or record.get("source_sha256") != candidate.get("code_hash"):
        raise ValueError(f"candidate identity mismatch in evidence: {path}")
    if baseline.get("source_sha256") != baseline_sha256:
        raise ValueError(f"baseline identity mismatch in evidence: {path}")
    if document.get("operator") != candidate.get("op_name"):
        raise ValueError(f"operator identity mismatch in evidence: {path}")
    if document.get("correctness_status") != "passed" or document.get("process_status") != "passed":
        raise ValueError(f"correctness evidence is not passed: {path}")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"correctness evidence has no cases: {path}")
    for case in cases:
        if not isinstance(case, dict) or case.get("status") != "passed":
            raise ValueError(f"correctness case is not passed: {path}")
        if not isinstance(case.get("test_path"), str) or not isinstance(case.get("test_sha256"), str):
            raise ValueError(f"correctness case lacks test identity: {path}")
        baseline_run = case.get("baseline_run")
        if not isinstance(baseline_run, dict) or baseline_run.get("status") != "passed":
            raise ValueError(f"baseline case run is not passed: {path}")
        candidate_run = case.get("candidate_run")
        if not isinstance(candidate_run, dict) or candidate_run.get("status") != "passed":
            raise ValueError(f"candidate case run is not passed: {path}")


def _result(path: Path, document: dict, split: str, oracle_policy: str) -> dict:
    case = document["cases"][0]
    return {
        "artifact_path": _relative(path),
        "artifact_sha256": _sha256(path),
        "oracle_policy": oracle_policy,
        "split": split,
        "status": "passed",
        "test_sha256": case.get("test_sha256"),
    }


def _holdout(
    path: Path,
    document: dict,
    candidate: dict,
    baseline_sha256: str,
    paired_path: Path | None,
    evidence_manifest_path: Path | None,
    oracle_kind: str,
    oracle_policy_id: str,
) -> dict:
    case = document["cases"][0]
    result = {
        "baseline_sha256": document["baseline"]["source_sha256"],
        "candidate_id": candidate["id"],
        "candidate_sha256": candidate["code_hash"],
        "case_count": len(document["cases"]),
        "case_signature": case.get("test_sha256"),
        "correctness_artifact_path": _relative(path),
        "correctness_artifact_sha256": _sha256(path),
        "environment_fingerprint_sha256": document.get("environment", {}).get("fingerprint_sha256"),
        "oracle_policy": {"kind": oracle_kind, "policy_id": oracle_policy_id},
        "split": "holdout",
        "status": "passed",
        "test_path": case.get("test_path"),
        "test_sha256": case.get("test_sha256"),
        "used_for_search": False,
        "used_in_prompt": False,
    }
    if evidence_manifest_path is not None:
        result["evidence_manifest_path"] = _relative(evidence_manifest_path)
        result["evidence_manifest_sha256"] = _sha256(evidence_manifest_path)
    if paired_path is not None:
        paired = _load(paired_path)
        paired_candidate = paired.get("candidate")
        paired_baseline = paired.get("baseline")
        if (
            not isinstance(paired_candidate, dict)
            or not isinstance(paired_baseline, dict)
            or paired.get("operator") != candidate["op_name"]
            or paired_candidate.get("source_sha256") != candidate["code_hash"]
            or paired_baseline.get("source_sha256") != baseline_sha256
        ):
            raise ValueError(f"candidate identity mismatch in paired evidence: {paired_path}")
        result["performance_artifact_path"] = _relative(paired_path)
        result["performance_artifact_sha256"] = _sha256(paired_path)
    return result


def _derived_import_evaluation(
    document: dict,
    candidate: dict,
    evidence_path: Path,
) -> dict | None:
    """Derive import evidence from a passed candidate correctness run.

    The Ascend correctness worker must import the candidate before it can run
    the test. Reusing that fact keeps this assembler usable when the default
    development interpreter lacks torch, while retaining the interpreter and
    hashed evidence artifact for auditability.
    """
    python_executables: set[str] = set()
    observed = 0
    for case in document.get("cases", []):
        if not isinstance(case, dict):
            continue
        run = case.get("candidate_run")
        if not isinstance(run, dict):
            continue
        if (
            run.get("status") != "passed"
            or run.get("role") != "candidate"
            or run.get("returncode") != 0
            or run.get("prepared_source_sha256") != candidate.get("code_hash")
        ):
            continue
        executable = run.get("python_executable")
        if isinstance(executable, str) and executable:
            python_executables.add(executable)
        observed += 1
    if observed == 0 or len(python_executables) != 1:
        return None
    executable = next(iter(python_executables))
    return {
        "error_message": None,
        "error_type": None,
        "evidence_artifact_path": _relative(evidence_path),
        "evidence_artifact_sha256": _sha256(evidence_path),
        "evidence_source": "candidate_correctness_run",
        "observed_case_count": observed,
        "phase": "module_import",
        "python_executable": executable,
        "status": "imported",
        "worker_returncode": 0,
    }


def assemble(
    candidate_manifest_path: Path,
    visible_correctness_path: Path,
    holdout_correctness_path: Path,
    output_dir: Path,
    *,
    holdout_paired_path: Path | None = None,
    holdout_evidence_manifest_path: Path | None = None,
    oracle_policy: str = "unknown",
    oracle_kind: str = "unknown",
    oracle_policy_id: str = "unknown",
    evidence_scope: str | None = None,
    import_timeout: float = 20.0,
) -> Path:
    source_manifest = _load(candidate_manifest_path)
    candidate = source_manifest.get("candidate")
    if not isinstance(candidate, dict) or not all(
        isinstance(candidate.get(key), str) for key in ("id", "op_name", "code_hash")
    ):
        raise ValueError("candidate manifest lacks stable identity")
    if candidate.get("status") != "static_pass" or source_manifest.get("rejection_error") is not None:
        raise ValueError("candidate manifest is not a static pass")
    source_path = candidate_manifest_path.with_name(f"{candidate['id']}.py")
    code = source_path.read_text(encoding="utf-8")
    if hashlib.sha256(code.encode("utf-8")).hexdigest() != candidate["code_hash"]:
        raise ValueError("candidate source hash does not match manifest")

    visible = _load(visible_correctness_path)
    holdout = _load(holdout_correctness_path)
    if visible_correctness_path.resolve() == holdout_correctness_path.resolve():
        raise ValueError("visible and holdout evidence must be distinct")
    baseline_sha256 = source_manifest.get("parent_sha256")
    if not isinstance(baseline_sha256, str):
        raise ValueError("candidate manifest lacks parent_sha256")
    _identity(visible, candidate, baseline_sha256, visible_correctness_path)
    _identity(holdout, candidate, baseline_sha256, holdout_correctness_path)

    imported_evaluation = source_manifest.get("import_evaluation")
    if not (
        isinstance(imported_evaluation, dict)
        and imported_evaluation.get("status") == "imported"
        and imported_evaluation.get("phase") == "module_import"
    ):
        visible_import = _derived_import_evaluation(
            visible, candidate, visible_correctness_path
        )
        holdout_import = _derived_import_evaluation(
            holdout, candidate, holdout_correctness_path
        )
        if (
            visible_import is not None
            and holdout_import is not None
            and visible_import["python_executable"]
            != holdout_import["python_executable"]
        ):
            raise ValueError("candidate import evidence uses mixed interpreters")
        imported_evaluation = visible_import or holdout_import
        if imported_evaluation is None:
            imported = run_candidate(
                CandidateRunRequest(
                    candidate_id=candidate["id"],
                    code=code,
                    code_hash=candidate["code_hash"],
                    entrypoint=candidate["op_name"],
                    payload={},
                    timeout_seconds=import_timeout,
                    stop_after_import=True,
                )
            )
            if (imported.status, imported.phase) != ("imported", "module_import"):
                raise ValueError(
                    f"candidate import failed: {imported.status}/{imported.phase}: {imported.error_message}"
                )
            imported_evaluation = {
                "error_message": imported.error_message,
                "error_type": imported.error_type,
                "phase": imported.phase,
                "status": imported.status,
            }

    output_operator_dir = output_dir / candidate["op_name"]
    output_manifest_path = output_operator_dir / candidate_manifest_path.name
    output_candidate_path = output_operator_dir / source_path.name
    if output_manifest_path.exists() or output_candidate_path.exists():
        raise FileExistsError(f"refusing to overwrite assembled candidate: {output_manifest_path}")
    output_operator_dir.mkdir(parents=True, exist_ok=True)
    output_candidate_path.write_text(code, encoding="utf-8")

    assembled = dict(source_manifest)
    assembled["candidate_path"] = _output_reference(output_candidate_path)
    assembled["import_evaluation"] = imported_evaluation
    scope = evidence_scope or "local_ascend_910b4_visible_and_holdout_not_official"
    if not scope.startswith("local_ascend_910b4_"):
        raise ValueError("evidence_scope must identify local Ascend 910B4 evidence")
    assembled["correctness_evaluation"] = {
        "admission_policy_id": ADMISSION_POLICY,
        "blocking_reasons": [],
        "case_count": len(visible["cases"]) + len(holdout["cases"]),
        "decision": {
            "blocking_reasons": [],
            "candidate_id": candidate["id"],
            "eligible_for_performance": True,
        },
        "eligible_for_performance": True,
        "evidence_scope": scope,
        "results": [
            _result(visible_correctness_path, visible, "search_visible", oracle_policy),
            _result(holdout_correctness_path, holdout, "holdout", oracle_policy),
        ],
        "status": "passed",
    }
    assembled["holdout_evaluation"] = _holdout(
        holdout_correctness_path,
        holdout,
        candidate,
        baseline_sha256,
        holdout_paired_path,
        holdout_evidence_manifest_path,
        oracle_kind,
        oracle_policy_id,
    )
    output_manifest_path.write_text(
        json.dumps(assembled, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--visible-correctness", required=True, type=Path)
    parser.add_argument("--holdout-correctness", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-paired", type=Path)
    parser.add_argument("--holdout-evidence-manifest", type=Path)
    parser.add_argument("--oracle-policy", default="unknown")
    parser.add_argument("--oracle-kind", default="unknown")
    parser.add_argument("--oracle-policy-id", default="unknown")
    parser.add_argument("--evidence-scope")
    parser.add_argument("--import-timeout", type=float, default=20.0)
    args = parser.parse_args()
    path = assemble(
        args.candidate_manifest,
        args.visible_correctness,
        args.holdout_correctness,
        args.output_dir,
        holdout_paired_path=args.holdout_paired,
        holdout_evidence_manifest_path=args.holdout_evidence_manifest,
        oracle_policy=args.oracle_policy,
        oracle_kind=args.oracle_kind,
        oracle_policy_id=args.oracle_policy_id,
        evidence_scope=args.evidence_scope,
        import_timeout=args.import_timeout,
    )
    print(json.dumps({"manifest_path": _relative(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
