#!/usr/bin/env python3
"""Build a fail-closed matrix from local correctness and profile evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.io_utils import discover_operators, find_test_files
from scripts.generate_official_candidate import _structure_hash


_DECLARED_NEUTRAL_MUTATION_KINDS = {
    "comment_only_baseline_probe",
    "existing_variant_comment_only",
    "existing_variant_format_only",
    "existing_variant_identifier_only",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _ref(root: Path, value: object, digest: object) -> Path | None:
    if not isinstance(value, str) or not isinstance(digest, str) or Path(value).is_absolute():
        return None
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path if path.is_file() and _sha256(path) == digest else None


def _fingerprint_ok(environment: object, paired: bool = False) -> bool:
    if not isinstance(environment, dict) or not isinstance(environment.get("fingerprint_sha256"), str):
        return False
    facts = {key: value for key, value in environment.items() if key != "fingerprint_sha256"}
    required = (
        {"python_executable", "python_version", "machine", "torch", "torch_npu", "triton", "npu_available", "device_name"}
        if paired
        else {"cann_version", "device_name", "machine", "python_executable", "python_version", "torch", "torch_npu", "triton"}
    )
    if set(facts) != required or not str(facts.get("device_name", "")).startswith("Ascend910B4"):
        return False
    if paired and facts.get("npu_available") is not True:
        return False
    payload = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest() == environment["fingerprint_sha256"]


def _manifest_ok(root: Path, path: Path, operator: str, candidate_id: object, source_sha: object) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        record = data.get("candidate", {})
        return (
            isinstance(record, dict)
            and record.get("id") == candidate_id
            and record.get("op_name") == operator
            and record.get("code_hash") == source_sha
            and record.get("status") == "static_pass"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        return False


def _candidate_change_class(
    baseline_path: Path, candidate_path: Path | None, manifest_path: Path | None
) -> str:
    if candidate_path is None or manifest_path is None:
        return "unverified"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutation_kind = manifest["candidate"].get("mutation_kind")
        baseline_code = baseline_path.read_text(encoding="utf-8")
        candidate_code = candidate_path.read_text(encoding="utf-8")
        if mutation_kind in _DECLARED_NEUTRAL_MUTATION_KINDS:
            return "declared_neutral"
        if _structure_hash(baseline_code) == _structure_hash(candidate_code):
            return "ast_equivalent"
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError, SyntaxError):
        return "unverified"
    return "substantive"


def _rank_candidates(evaluations: list[dict]) -> tuple[list[dict], list[dict]]:
    qualified = sorted(
        (item for item in evaluations if item["qualified"]),
        key=lambda item: item["candidate_over_baseline_ratio"],
    )
    substantive = [
        item for item in qualified if item.get("change_class") == "substantive"
    ]
    return qualified, substantive


def _has_explicit_seed(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "manual_seed"
        for node in ast.walk(tree)
    )


def _run_metadata_ok(root: Path, run: object, role: str, source_sha: object, test_sha: object) -> bool:
    if not isinstance(run, dict):
        return False
    return (
        run.get("role") == role
        and run.get("status") == "passed"
        and run.get("returncode") == 0
        and run.get("source_sha256") == source_sha
        and run.get("test_sha256") == test_sha
        and isinstance(run.get("duration_us"), (int, float))
        and not isinstance(run.get("duration_us"), bool)
        and math.isfinite(float(run["duration_us"]))
        and float(run["duration_us"]) > 0
        and isinstance(run.get("device_id"), int)
        and isinstance(run.get("frequency_mhz"), (int, float))
        and _ref(root, run.get("csv_path"), run.get("csv_sha256")) is not None
        and _ref(root, run.get("log_path"), run.get("log_sha256")) is not None
    )


def _paired_runs_ok(root: Path, case: dict, baseline_sha: str, candidate_sha: str) -> tuple[bool, float | None]:
    baseline_runs = case.get("baseline_runs")
    candidate_runs = case.get("candidate_runs")
    if (
        case.get("sequence") != ["baseline", "candidate", "candidate", "baseline"]
        or case.get("status") not in {"passed", "failed"}
        or not isinstance(baseline_runs, list)
        or not isinstance(candidate_runs, list)
        or len(baseline_runs) != 2
        or len(candidate_runs) != 2
    ):
        return False, None
    test_sha = case.get("test_sha256")
    runs = [
        (run, "baseline", baseline_sha) for run in baseline_runs
    ] + [(run, "candidate", candidate_sha) for run in candidate_runs]
    if not all(_run_metadata_ok(root, run, role, source_sha, test_sha) for run, role, source_sha in runs):
        return False, None
    all_runs = baseline_runs + candidate_runs
    if len({run["device_id"] for run in all_runs}) != 1 or len({run["frequency_mhz"] for run in all_runs}) != 1:
        return False, None
    baseline_values = [float(run["duration_us"]) for run in baseline_runs]
    candidate_values = [float(run["duration_us"]) for run in candidate_runs]
    ratio = statistics.median(candidate_values) / statistics.median(baseline_values)
    recorded = case.get("candidate_over_baseline_ratio")
    if not isinstance(recorded, (int, float)) or not math.isclose(ratio, float(recorded), rel_tol=1e-9, abs_tol=1e-9):
        return False, None
    return True, ratio


def _inspect_paired(
    root: Path,
    path: Path,
    operator: str,
    baseline_path: Path,
    visible: dict[Path, str],
    ratio_limit: float,
    correctness_dir: Path,
) -> dict:
    blockers: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        data = {}
        blockers.append("invalid_json")
    if not isinstance(data, dict):
        data = {}
        blockers.append("invalid_json_shape")
    candidate = data.get("candidate") if isinstance(data.get("candidate"), dict) else {}
    baseline = data.get("baseline") if isinstance(data.get("baseline"), dict) else {}
    candidate_id = candidate.get("id")
    candidate_path = _ref(root, candidate.get("source_path"), candidate.get("source_sha256"))
    baseline_ref = _ref(root, baseline.get("source_path"), baseline.get("source_sha256"))
    manifest_ref = _ref(root, candidate.get("manifest_path"), candidate.get("manifest_sha256"))
    identity_ok = (
        data.get("schema_version") == 1
        and data.get("artifact_kind") == "local-ascend-paired-benchmark"
        and data.get("evidence_scope") == "local_ascend_910b4_currently_visible_cases_not_official"
        and data.get("operator") == operator
        and isinstance(candidate_id, str)
        and path.name == f"{candidate_id}.paired.json"
        and candidate_path is not None
        and baseline_ref == baseline_path.resolve()
        and manifest_ref is not None
        and _fingerprint_ok(data.get("environment"), paired=True)
    )
    if not identity_ok:
        blockers.append("paired_contract_or_identity_unverified")
    if candidate_path is not None and _sha256(candidate_path) == _sha256(baseline_path):
        blockers.append("candidate_identity_not_nonbaseline")
    if manifest_ref is None or not _manifest_ok(root, manifest_ref, operator, candidate_id, candidate.get("source_sha256")):
        blockers.append("candidate_manifest_binding_mismatch")
    change_class = _candidate_change_class(
        baseline_path, candidate_path, manifest_ref
    )

    correctness_ref = _ref(root, data.get("correctness_artifact_path"), data.get("correctness_artifact_sha256"))
    correctness_ok = False
    if correctness_ref is None or correctness_ref.parent.parent != correctness_dir.resolve():
        blockers.append("correctness_artifact_unverified")
    else:
        try:
            correctness = json.loads(correctness_ref.read_text(encoding="utf-8"))
            cases = correctness.get("cases", [])
            correctness_tests = {(case["test_path"], case["test_sha256"]) for case in cases}
            visible_tests = {(_relative(root, test), digest) for test, digest in visible.items()}
            correctness_ok = (
                correctness.get("artifact_kind") == "local-ascend-correctness-evaluation"
                and correctness.get("operator") == operator
                and correctness.get("qualified_for_local_performance") is True
                and correctness.get("correctness_status") == "passed"
                and correctness.get("candidate", {}).get("source_sha256") == candidate.get("source_sha256")
                and correctness.get("baseline", {}).get("source_sha256") == baseline.get("source_sha256")
                and correctness.get("environment", {}).get("fingerprint_sha256") == data.get("environment", {}).get("fingerprint_sha256")
                and correctness_tests == visible_tests
                and all(case.get("status") == "passed" for case in cases)
            )
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, KeyError, TypeError):
            correctness_ok = False
    if not correctness_ok:
        blockers.append("correctness_artifact_not_passed")

    cases = data.get("cases") if isinstance(data.get("cases"), list) else []
    ratios: list[float] = []
    if len(cases) != len(visible) or data.get("sequence") != ["baseline", "candidate", "candidate", "baseline"]:
        blockers.append("paired_case_matrix_incomplete")
    for case in cases:
        if not isinstance(case, dict):
            blockers.append("paired_case_invalid")
            continue
        test = _ref(root, case.get("test_path"), case.get("test_sha256"))
        if test is None or visible.get(test) != case.get("test_sha256"):
            blockers.append("paired_test_binding_mismatch")
        if test is not None and not _has_explicit_seed(test):
            seed = data.get("input_seed")
            case_runs = case.get("baseline_runs", []) + case.get("candidate_runs", [])
            seed_ok = (
                isinstance(seed, int)
                and not isinstance(seed, bool)
                and seed >= 0
                and case.get("input_seed") == seed
                and len(case_runs) == 4
                and all(isinstance(run, dict) and run.get("input_seed") == seed for run in case_runs)
            )
            if not seed_ok:
                blockers.append("paired_input_seed_unverified")
        valid, ratio = _paired_runs_ok(root, case, baseline.get("source_sha256"), candidate.get("source_sha256"))
        if not valid or ratio is None:
            blockers.append("paired_run_evidence_unverified")
            continue
        ratios.append(ratio)
        expected_case_status = "passed" if ratio <= ratio_limit else "failed"
        if case.get("status") != expected_case_status:
            blockers.append("paired_case_status_mismatch")
        if ratio > ratio_limit:
            blockers.append("performance_ratio_above_limit")
    if not ratios or not isinstance(data.get("worst_case_ratio"), (int, float)) or not math.isclose(
        max(ratios), float(data["worst_case_ratio"]), rel_tol=1e-9, abs_tol=1e-9
    ):
        blockers.append("paired_worst_ratio_unverified")
    expected_qualified = len(ratios) == len(visible) and all(ratio <= ratio_limit for ratio in ratios)
    if data.get("qualified_for_local_performance") is not expected_qualified:
        blockers.append("paired_qualification_flag_mismatch")
    return {
        "candidate_id": candidate_id,
        "evidence_path": _relative(root, path),
        "correctness_status": "passed" if correctness_ok and not any(reason.startswith("correctness") for reason in blockers) else "unknown",
        "performance_status": "within_limit" if ratios and correctness_ok and not any(reason in {"performance_ratio_above_limit", "paired_run_evidence_unverified", "paired_worst_ratio_unverified"} for reason in blockers) else "regressed" if "performance_ratio_above_limit" in blockers else "unknown",
        "candidate_over_baseline_ratio": max(ratios) if ratios else None,
        "change_class": change_class,
        "blocking_reasons": sorted(set(blockers)),
        "qualified": not blockers,
    }


def _inspect_legacy(root: Path, path: Path, operator: str, baseline_path: Path, visible: dict[Path, str], ratio_limit: float) -> dict:
    """Consume the one historical v1 sidecar without treating it as the new producer schema."""
    blockers: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        data = {}
        blockers.append("invalid_json")
    candidate = data.get("candidate", {}) if isinstance(data, dict) else {}
    sources = data.get("sources", {}) if isinstance(data, dict) else {}
    correctness = data.get("correctness", {}) if isinstance(data, dict) else {}
    profiler = data.get("profiler", {}) if isinstance(data, dict) else {}
    candidate_id = candidate.get("id") if isinstance(candidate, dict) else None
    candidate_path = _ref(root, candidate.get("source_path"), candidate.get("source_sha256")) if isinstance(candidate, dict) else None
    manifest_path = _ref(root, candidate.get("manifest_path"), candidate.get("manifest_sha256")) if isinstance(candidate, dict) else None
    test = _ref(root, sources.get("test_path"), correctness.get("test_sha256")) if isinstance(sources, dict) and isinstance(correctness, dict) else None
    identity_ok = (
        data.get("schema_version") == 1
        and data.get("artifact_kind") == "local-ascend-candidate-evaluation"
        and data.get("operator") == operator
        and data.get("evidence_scope") == "local-ascend-910b4-msprof-not-official-evaluation"
        and _fingerprint_ok(data.get("environment"))
        and isinstance(candidate_id, str)
        and path.name == f"{candidate_id}.ascend-evaluation.json"
        and candidate_path is not None
        and _sha256(candidate_path) != _sha256(baseline_path)
        and manifest_path is not None
        and _manifest_ok(root, manifest_path, operator, candidate_id, candidate.get("source_sha256"))
        and _ref(root, sources.get("baseline_path"), sources.get("baseline_sha256")) == baseline_path.resolve()
        and len(visible) == 1
        and test in visible
        and correctness.get("status") == "local_public_case_1_passed"
        and correctness.get("completion_marker") == "All tests passed!"
        and correctness.get("candidate_public_test_run_count", 0) >= 2
    )
    if not identity_ok:
        blockers.append("legacy_contract_or_identity_unverified")
    change_class = _candidate_change_class(
        baseline_path, candidate_path, manifest_path
    )
    baseline_runs = profiler.get("baseline_runs", []) if isinstance(profiler, dict) else []
    candidate_runs = profiler.get("candidate_runs", []) if isinstance(profiler, dict) else []

    marker = correctness.get("completion_marker") if isinstance(correctness, dict) else None
    required_log_markers = (
        marker,
        "Profiling running finished. All task success.",
    )
    candidate_logs_ok = isinstance(marker, str) and len(candidate_runs) == 2
    if candidate_logs_ok:
        for run in candidate_runs:
            log_path = _ref(root, run.get("log_path"), run.get("log_sha256")) if isinstance(run, dict) else None
            try:
                log_text = log_path.read_text(encoding="utf-8") if log_path is not None else ""
            except (OSError, UnicodeError):
                log_text = ""
            operator_marker = re.search(
                rf"Op Name:[ \t]*{re.escape(operator)}(?:[ \t]*\r?\n|$)", log_text
            )
            if any(item not in log_text for item in required_log_markers) or operator_marker is None:
                candidate_logs_ok = False
                break
    if not candidate_logs_ok:
        blockers.append("legacy_correctness_log_unverified")

    def values(runs: object) -> list[float]:
        if not isinstance(runs, list) or len(runs) != 2:
            return []
        values = []
        for run in runs:
            if (
                not isinstance(run, dict)
                or _ref(root, run.get("csv_path"), run.get("csv_sha256")) is None
                or _ref(root, run.get("log_path"), run.get("log_sha256")) is None
                or not isinstance(run.get("duration_us"), (int, float))
            ):
                return []
            values.append(float(run["duration_us"]))
        return values
    base_values, cand_values = values(baseline_runs), values(candidate_runs)
    ratio = statistics.median(cand_values) / statistics.median(base_values) if len(base_values) == len(cand_values) == 2 else None
    if ratio is None or not math.isclose(ratio, float(profiler.get("candidate_over_baseline_ratio")), rel_tol=1e-9, abs_tol=1e-9):
        blockers.append("legacy_ratio_unverified")
    elif ratio > ratio_limit:
        blockers.append("performance_ratio_above_limit")
    return {
        "candidate_id": candidate_id,
        "evidence_path": _relative(root, path),
        "correctness_status": "passed" if identity_ok and candidate_logs_ok else "unknown",
        "performance_status": "within_limit" if ratio is not None and ratio <= ratio_limit and identity_ok and candidate_logs_ok else "regressed" if "performance_ratio_above_limit" in blockers else "unknown",
        "candidate_over_baseline_ratio": ratio,
        "change_class": change_class,
        "blocking_reasons": sorted(set(blockers)),
        "qualified": not blockers,
    }


def build_matrix(root: Path, dataset_dir: Path, candidate_dir: Path, ratio_limit: float, correctness_dir: Path | None = None, paired_dir: Path | None = None) -> dict:
    if isinstance(ratio_limit, bool) or not math.isfinite(ratio_limit) or ratio_limit <= 0:
        raise ValueError("ratio_limit must be a positive finite number")
    correctness_dir = (correctness_dir or root / "output" / "local-correctness").resolve()
    paired_dir = (paired_dir or root / "output" / "local-paired").resolve()
    rows = []
    for operator in discover_operators(dataset_dir):
        op_dir = dataset_dir / operator
        baseline_path = op_dir / f"{operator}.py"
        baseline_sha = _sha256(baseline_path)
        visible = {test.resolve(): _sha256(test) for test in find_test_files(op_dir, operator)}
        sources = sorted((candidate_dir / operator).glob("*.py"))
        evaluations = [
            _inspect_legacy(root, path, operator, baseline_path, visible, ratio_limit)
            for path in sorted((candidate_dir / operator).glob("*.ascend-evaluation.json"))
        ] + [
            _inspect_paired(root, path, operator, baseline_path, visible, ratio_limit, correctness_dir)
            for path in sorted((paired_dir / operator).glob("*.paired.json"))
        ]
        qualified, substantive = _rank_candidates(evaluations)
        blockers = []
        if not any(_sha256(source) != baseline_sha for source in sources):
            blockers.append("missing_nonbaseline_candidate")
        if not evaluations:
            blockers.append("missing_local_evaluation")
        elif not qualified:
            blockers.extend(reason for item in evaluations for reason in item["blocking_reasons"])
        rows.append(
            {
                "operator": operator,
                "baseline_sha256": baseline_sha,
                "visible_case_count": len(visible),
                "visible_case_sha256": sorted(visible.values()),
                "candidate_source_count": len(sources),
                "nonbaseline_candidate_count": sum(_sha256(source) != baseline_sha for source in sources),
                "compile_status": "unknown",
                "process_status": "unknown",
                "correctness_status": "passed" if any(item["correctness_status"] == "passed" for item in evaluations) else "unknown",
                "performance_status": "within_limit" if qualified else "regressed" if any(item["performance_status"] == "regressed" for item in evaluations) else "unknown",
                "qualification_status": "qualified" if qualified else "incomplete",
                "selection_status": (
                    "substantive_candidate_ready"
                    if substantive
                    else "neutral_only"
                    if qualified
                    else "incomplete"
                ),
                "best_candidate_id": (
                    substantive[0]["candidate_id"] if substantive else None
                ),
                "best_qualification_candidate_id": (
                    qualified[0]["candidate_id"] if qualified else None
                ),
                "blocking_reasons": sorted(set(blockers)),
                "evaluations": evaluations,
            }
        )
    qualified_count = sum(row["qualification_status"] == "qualified" for row in rows)
    return {
        "schema_version": 3,
        "evidence_scope": "local_ascend_910b4_currently_visible_cases_not_official",
        "limitations": [
            "Only currently checked-out tests are covered; organizer case 2/3 remain unknown.",
            "Compile and process are not promoted to official functional-pass facts.",
            "910B4 evidence is not an official A2/A3 evaluation.",
            "Selection excludes AST-equivalent and explicitly neutral candidates.",
        ],
        "candidate_over_baseline_ratio_limit": ratio_limit,
        "summary": {
            "operator_count": len(rows),
            "qualified_count": qualified_count,
            "incomplete_count": len(rows) - qualified_count,
            "with_nonbaseline_candidate_count": sum(row["nonbaseline_candidate_count"] > 0 for row in rows),
            "with_local_evaluation_count": sum(bool(row["evaluations"]) for row in rows),
            "substantive_selected_count": sum(
                row["selection_status"] == "substantive_candidate_ready"
                for row in rows
            ),
            "neutral_only_count": sum(
                row["selection_status"] == "neutral_only" for row in rows
            ),
        },
        "operators": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--correctness-dir", default=None)
    parser.add_argument("--paired-dir", default=None)
    parser.add_argument("--ratio-limit", type=float, default=1.03)
    args = parser.parse_args()
    report = build_matrix(
        ROOT, Path(args.dataset_dir), Path(args.candidate_dir), args.ratio_limit,
        Path(args.correctness_dir) if args.correctness_dir else None,
        Path(args.paired_dir) if args.paired_dir else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[qualification] qualified={report['summary']['qualified_count']}/{report['summary']['operator_count']}")
    print(f"[qualification] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
