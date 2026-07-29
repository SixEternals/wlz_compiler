"""Output writers for candidates, top5, manifest, and run report."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from wlz_optimizer.evolutionary_algorithm import top_k_per_operator
from wlz_optimizer.schemas import CandidateEvaluation


def write_run_outputs(
    output_dir: Path,
    evaluations: List[CandidateEvaluation],
    run_config: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    top5_dir = output_dir / "top5"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    top5_dir.mkdir(parents=True, exist_ok=True)

    with_paths = _write_candidate_files(output_dir, candidates_dir, evaluations)
    top_by_op = top_k_per_operator(with_paths, top_k)
    with_top_paths = _write_top5_files(output_dir, top5_dir, with_paths, top_by_op)

    manifest = _build_manifest(output_dir, with_top_paths, run_config, top_by_op)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    _write_top5_summary(top5_dir, top_by_op, output_dir)
    _write_report(output_dir / "run_report.md", with_top_paths, run_config, top_by_op)
    return manifest


def _write_candidate_files(
    root: Path,
    candidates_dir: Path,
    evaluations: List[CandidateEvaluation],
) -> List[CandidateEvaluation]:
    out: List[CandidateEvaluation] = []
    for item in evaluations:
        op_dir = candidates_dir / item.candidate.op_name
        op_dir.mkdir(parents=True, exist_ok=True)
        path = op_dir / f"{item.candidate.id}.py"
        path.write_text(item.candidate.code, encoding="utf-8")
        out.append(replace(item, candidate_path=path))
    return out


def _write_top5_files(
    root: Path,
    top5_dir: Path,
    evaluations: List[CandidateEvaluation],
    top_by_op: Dict[str, List[CandidateEvaluation]],
) -> List[CandidateEvaluation]:
    top_paths: Dict[str, Path] = {}

    for op_name, items in top_by_op.items():
        op_dir = top5_dir / op_name
        op_dir.mkdir(parents=True, exist_ok=True)
        stats: List[Dict[str, Any]] = []
        for rank, item in enumerate(items, start=1):
            path = op_dir / f"{op_name}_{rank}.py"
            path.write_text(item.candidate.code, encoding="utf-8")
            top_paths[item.candidate.id] = path
            stats.append(
                {
                    "rank": rank,
                    "candidate_id": item.candidate.id,
                    "code_hash": item.candidate.code_hash,
                    "proxy_score": item.result.proxy_score,
                    "status": item.result.status,
                    "passed": item.result.passed,
                    "mutation_kind": item.candidate.mutation_kind,
                    "generation": item.candidate.generation,
                    "top5_path": str(path.relative_to(root)),
                }
            )

        (op_dir / f"{op_name}_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    return [
        replace(item, top5_path=top_paths.get(item.candidate.id))
        for item in evaluations
    ]


def _build_manifest(
    output_dir: Path,
    evaluations: List[CandidateEvaluation],
    run_config: Dict[str, Any],
    top_by_op: Dict[str, List[CandidateEvaluation]],
) -> Dict[str, Any]:
    operators = sorted({item.candidate.op_name for item in evaluations})
    status_counts = Counter(item.result.status for item in evaluations)
    error_counts = Counter(item.result.error_type or "none" for item in evaluations)
    cache_hits = sum(1 for item in evaluations if item.result.metadata.get("cache_hit"))

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "summary": {
            "operator_count": len(operators),
            "candidate_count": len(evaluations),
            "cache_hits": cache_hits,
            "status_counts": dict(status_counts),
            "error_counts": dict(error_counts),
        },
        "operators": operators,
        "top5": {
            op_name: [
                {
                    "rank": idx + 1,
                    "candidate_id": item.candidate.id,
                    "code_hash": item.candidate.code_hash,
                    "proxy_score": item.result.proxy_score,
                    "path": str((output_dir / "top5" / op_name / f"{op_name}_{idx + 1}.py").relative_to(output_dir)),
                }
                for idx, item in enumerate(items)
            ]
            for op_name, items in top_by_op.items()
        },
        "candidates": [item.to_manifest_record(output_dir) for item in evaluations],
    }


def _write_top5_summary(
    top5_dir: Path,
    top_by_op: Dict[str, List[CandidateEvaluation]],
    output_dir: Path,
) -> None:
    summary = {
        op_name: [
            {
                "rank": rank,
                "candidate_id": item.candidate.id,
                "proxy_score": item.result.proxy_score,
                "status": item.result.status,
                "path": str((top5_dir / op_name / f"{op_name}_{rank}.py").relative_to(output_dir)),
            }
            for rank, item in enumerate(items, start=1)
        ]
        for op_name, items in top_by_op.items()
    }
    (top5_dir / "top5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    evaluations: List[CandidateEvaluation],
    run_config: Dict[str, Any],
    top_by_op: Dict[str, List[CandidateEvaluation]],
) -> None:
    status_counts = Counter(item.result.status for item in evaluations)
    error_counts = Counter(item.result.error_type or "none" for item in evaluations)
    cache_hits = sum(1 for item in evaluations if item.result.metadata.get("cache_hit"))

    lines = [
        "# Local Mock Run Report",
        "",
        "This run used static local validation and proxy scores only. It did not measure Ascend latency and did not compute real speedup.",
        "",
        "## Config",
        "",
    ]
    for key in sorted(run_config):
        lines.append(f"- {key}: `{run_config[key]}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Operators: {len(top_by_op)}",
            f"- Candidate evaluations: {len(evaluations)}",
            f"- Cache hits: {cache_hits}",
            f"- Status counts: {dict(status_counts)}",
            f"- Error counts: {dict(error_counts)}",
            "",
            "## Top5 By Operator",
            "",
            "| Operator | Rank | Candidate | Proxy Score | Status | Mutation | Generation |",
            "| --- | ---: | --- | ---: | --- | --- | ---: |",
        ]
    )

    for op_name in sorted(top_by_op):
        for rank, item in enumerate(top_by_op[op_name], start=1):
            score = item.result.proxy_score if item.result.proxy_score is not None else 0.0
            lines.append(
                "| {op} | {rank} | `{cid}` | {score:.6f} | {status} | {mut} | {gen} |".format(
                    op=op_name,
                    rank=rank,
                    cid=item.candidate.id,
                    score=score,
                    status=item.result.status,
                    mut=item.candidate.mutation_kind,
                    gen=item.candidate.generation,
                )
            )

    lines.extend(
        [
            "",
            "## Failure Taxonomy",
            "",
            "The schema reserves these failure categories for later real executors: syntax_fail, import_fail, target_device_fail, signature_fail, launch_contract_fail, triton_semantic_fail, compile_fail, correctness_fail, timeout, runtime_error.",
            "LocalExecutor currently emits syntax_fail, import_fail, target_device_fail, signature_fail, launch_contract_fail, and triton_semantic_fail from static checks.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
