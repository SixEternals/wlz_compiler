#!/usr/bin/env python3
"""Register an existing source variant as a locally auditable candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_WORK = ROOT / "work" / "official_triton_agent"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repository root: {path}") from exc


def _interface_error(
    baseline: str,
    candidate: str,
    test_code: Optional[str] = None,
):
    """Reuse the official runtime contract without duplicating its AST rules."""
    work_path = str(OFFICIAL_WORK)
    inserted = work_path not in sys.path
    if inserted:
        sys.path.insert(0, work_path)
    try:
        from contract_executor import interface_contract_error
        return interface_contract_error(
            baseline,
            candidate,
            test_code,
            enforce_semantic_change=True,
        )
    finally:
        if inserted:
            sys.path.remove(work_path)


def _preflight(operator: str, baseline: Path, candidate: bytes) -> dict:
    try:
        baseline_text = baseline.read_text(encoding="utf-8")
        candidate_text = candidate.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"static preflight could not read source: {type(exc).__name__}"
        ) from exc

    test_paths = sorted(baseline.parent.glob(f"test_{operator}_*.py"))
    checked = []
    if not test_paths:
        error = _interface_error(baseline_text, candidate_text)
        if error is not None:
            raise ValueError(f"static preflight failed: {error}")
    for test_path in test_paths:
        try:
            test_text = test_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(
                f"static preflight could not read test {test_path.name}: "
                f"{type(exc).__name__}"
            ) from exc
        error = _interface_error(baseline_text, candidate_text, test_text)
        if error is not None:
            raise ValueError(
                f"static preflight failed for {test_path.name}: {error}"
            )
        checked.append(_relative(test_path))
    return {
        "status": "passed",
        "executor": "official-contract-preflight",
        "test_files": checked,
        "case_count": len(checked),
    }


def materialize(
    operator: str,
    baseline: Path,
    source: Path,
    output_dir: Path,
    mutation_kind: str = "existing_variant_import",
    replace_old: str | None = None,
    replace_new: str | None = None,
) -> tuple[Path, Path]:
    baseline = baseline.resolve()
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not baseline.is_file() or not source.is_file():
        raise FileNotFoundError("baseline and source must be files")
    baseline_sha = _sha256(baseline)
    source_bytes = source.read_bytes()
    candidate_bytes = source_bytes
    metadata = {
        "source_kind": "existing_dataset_variant",
        "source_path": _relative(source),
    }
    if replace_old is not None:
        if replace_old == "":
            raise ValueError("replace_old must not be empty")
        source_text = source_bytes.decode("utf-8")
        if source_text.count(replace_old) != 1:
            raise ValueError("replace_old must occur exactly once")
        candidate_bytes = source_text.replace(replace_old, replace_new or "", 1).encode("utf-8")
        metadata = {
            "source_kind": "exact_text_rewrite",
            "source_path": _relative(source),
            "replace_old_sha256": hashlib.sha256(replace_old.encode("utf-8")).hexdigest(),
            "replace_new_sha256": hashlib.sha256((replace_new or "").encode("utf-8")).hexdigest(),
        }
    source_sha = hashlib.sha256(candidate_bytes).hexdigest()
    if source_sha == baseline_sha:
        raise ValueError("source variant must not be byte-identical to baseline")
    preflight = _preflight(operator, baseline, candidate_bytes)
    candidate_id = f"localv-{source_sha[:12]}"
    candidate_dir = output_dir / operator
    candidate_path = candidate_dir / f"{candidate_id}.py"
    manifest_path = candidate_dir / f"{candidate_id}.manifest.json"
    if candidate_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite candidate {candidate_id}")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes(candidate_bytes)
    candidate_sha = _sha256(candidate_path)
    if candidate_sha != source_sha:
        raise OSError("candidate copy changed source bytes")
    manifest = {
        "schema_version": 1,
        "artifact_kind": "real-agent-candidate",
        "candidate": {
            "id": candidate_id,
            "op_name": operator,
            "code_hash": candidate_sha,
            "parent_ids": [f"seed-{baseline_sha[:12]}"],
            "generation": 1,
            "mutation_kind": mutation_kind,
            "model_used": "manual-local-qualification",
            "prompt_id": None,
            "status": "static_pass",
            "score": None,
            "metadata": {
                **metadata,
            },
        },
        "candidate_path": _relative(candidate_path),
        "parent_path": _relative(baseline),
        "parent_sha256": baseline_sha,
        "seed_sha256": sorted({baseline_sha, source_sha}),
        "static_evaluation": {
            "executor": preflight["executor"],
            "status": "interface_contract_pass",
            "passed": True,
            "test_files": preflight["test_files"],
            "case_count": preflight["case_count"],
            "compile_ok": None,
            "correctness_ok": None,
            "latency_ms": None,
            "proxy_score": None,
        },
        "import_evaluation": {
            "status": "not_run",
            "phase": "not_run",
            "error_type": None,
            "error_message": None,
        },
        "rejection_error": None,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return candidate_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mutation-kind", default="existing_variant_import")
    parser.add_argument("--replace-old")
    parser.add_argument("--replace-new")
    args = parser.parse_args()
    if (args.replace_old is None) != (args.replace_new is None):
        parser.error("--replace-old and --replace-new must be supplied together")
    candidate, manifest = materialize(
        args.operator,
        Path(args.baseline),
        Path(args.source),
        Path(args.output_dir),
        args.mutation_kind,
        args.replace_old,
        args.replace_new,
    )
    print(f"[materialize] candidate={_relative(candidate)}")
    print(f"[materialize] manifest={_relative(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
