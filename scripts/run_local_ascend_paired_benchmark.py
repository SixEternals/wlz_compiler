#!/usr/bin/env python3
"""Run a fail-closed, serial ABBA msprof benchmark on local Ascend."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_local_ascend_correctness import DEFAULT_EVIDENCE_SCOPE
from scripts.run_local_ascend_correctness import SUPPORTED_EVIDENCE_SCOPES
from scripts.run_local_ascend_correctness import probe_environment
from scripts.run_local_ascend_correctness import _prepare_compatibility_shims


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path is outside repository root: {path}") from exc


def _ref(root: Path, value: object, digest: object) -> Path | None:
    if not isinstance(value, str) or not isinstance(digest, str):
        return None
    path = Path(value)
    if path.is_absolute():
        return None
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved if resolved.is_file() and _sha256(resolved) == digest else None


def _parse_profile(csv_path: Path, operator: str) -> tuple[float, int, float, str] | None:
    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = ("Op Name", "Task Duration(us)", "Device Id", "Current Freq")
            if reader.fieldnames is None or any(
                reader.fieldnames.count(name) != 1 for name in required
            ):
                return None
            allowed_names = {operator, f"{operator}_mix_aic", f"{operator}_mix_aiv"}
            rows = [row for row in reader if row.get("Op Name") in allowed_names]
            if len(rows) != 1:
                return None
            duration = float(rows[0]["Task Duration(us)"])
            device = int(rows[0]["Device Id"])
            frequency = float(rows[0]["Current Freq"])
            if (
                duration > 0
                and frequency > 0
                and device >= 0
                and math.isfinite(duration)
                and math.isfinite(frequency)
            ):
                return duration, device, frequency, rows[0]["Op Name"]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return None


def _find_profile(run_dir: Path, operator: str) -> tuple[Path, tuple[float, int, float, str]] | None:
    profiles = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("OPPROF_"))
    if len(profiles) != 1:
        return None
    csv_path = profiles[0] / "OpBasicInfo.csv"
    observation = _parse_profile(csv_path, operator) if csv_path.is_file() else None
    return (csv_path, observation) if observation is not None else None


def _run_msprof(
    root: Path,
    operator: str,
    role: str,
    index: int,
    source: Path,
    test: Path,
    python: Path,
    msprof: str,
    profile_dir: Path,
    timeout: float,
    input_seed: int,
) -> dict:
    source_bytes = source.read_bytes()
    test_bytes = test.read_bytes()
    run_name = f"{index:02d}-{role}"
    run_dir = (profile_dir / run_name).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"profile run directory already contains evidence: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "get_prof.log"
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"wlz-profile-{operator}-{role}-") as tmp:
        work = Path(tmp)
        source_tmp = work / f"{operator}.py"
        source_alias = work / f"{operator}_1.py"
        test_tmp = work / test.name
        runner_tmp = work / "_wlz_seeded_runner.py"
        source_tmp.write_bytes(source_bytes)
        source_alias.write_bytes(source_bytes)
        test_tmp.write_bytes(test_bytes)
        compatibility_shims = _prepare_compatibility_shims(work, source_bytes, test_bytes)
        runner_tmp.write_text(
            "import runpy\n"
            "import torch\n"
            f"torch.manual_seed({input_seed})\n"
            f"runpy.run_path({str(test_tmp)!r}, run_name='__main__')\n",
            encoding="utf-8",
        )
        command = [
            msprof,
            "op",
            f"--output={run_dir}",
            f"--application={python} {runner_tmp}",
            f"--kernel-name={operator}",
            "--aic-metrics=MemoryDetail,Occupancy,PipeUtilization,Roofline",
        ]
        try:
            run_env = os.environ.copy()
            run_env["WLZ_PAIRED_ROLE"] = role
            run_env["WLZ_INPUT_SEED"] = str(input_seed)
            with log_path.open("wb") as log:
                completed = subprocess.run(
                    command,
                    cwd=work,
                    env=run_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            returncode = completed.returncode
            process_status = "passed" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            returncode = None
            process_status = "timeout"
        except OSError as exc:
            log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            returncode = None
            process_status = "failed"
    parsed = _find_profile(run_dir, operator) if process_status == "passed" else None
    result = {
        "index": index,
        "role": role,
        "status": "passed" if parsed is not None else "profile_unverified" if process_status == "passed" else process_status,
        "returncode": returncode,
        "started_at": started,
        "elapsed_seconds": time.monotonic() - start,
        "python_executable": str(python.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "test_sha256": hashlib.sha256(test_bytes).hexdigest(),
        "input_seed": input_seed,
        "compatibility_shims": compatibility_shims,
        "log_path": _relative(root, log_path),
        "log_sha256": _sha256(log_path),
    }
    if parsed is not None:
        csv_path, (duration, device, frequency, profile_op_name) = parsed
        result.update(
            {
                "csv_path": _relative(root, csv_path),
                "csv_sha256": _sha256(csv_path),
                "duration_us": duration,
                "device_id": device,
                "frequency_mhz": frequency,
                "profile_op_name": profile_op_name,
            }
        )
    return result


def _validate_correctness(root: Path, artifact_path: Path) -> tuple[dict, Path, Path, Path, str]:
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("artifact_kind") != "local-ascend-correctness-evaluation"
        or data.get("qualified_for_local_performance") is not True
        or data.get("correctness_status") != "passed"
    ):
        raise ValueError("correctness artifact does not admit performance")
    candidate = data.get("candidate", {})
    baseline = data.get("baseline", {})
    cases = data.get("cases")
    if not isinstance(candidate, dict) or not isinstance(baseline, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("correctness artifact is incomplete")
    source = _ref(root, candidate.get("source_path"), candidate.get("source_sha256"))
    baseline_path = _ref(root, baseline.get("source_path"), baseline.get("source_sha256"))
    manifest = _ref(root, candidate.get("manifest_path"), candidate.get("manifest_sha256"))
    if source is None or baseline_path is None or manifest is None:
        raise ValueError("correctness artifact source references are not verifiable")
    if _sha256(source) == _sha256(baseline_path):
        raise ValueError("correctness artifact candidate is baseline")
    operator = data.get("operator")
    if not isinstance(operator, str) or not operator:
        raise ValueError("correctness artifact operator is missing")
    for case in cases:
        test = _ref(root, case.get("test_path"), case.get("test_sha256")) if isinstance(case, dict) else None
        if test is None or case.get("status") != "passed":
            raise ValueError("all correctness cases must be passed and hash-bound")
    return data, source, baseline_path, manifest, operator


def run_paired_benchmark(
    root: Path,
    correctness_artifact: Path,
    python: Path,
    output: Path,
    msprof: str = "msprof",
    pairs: int = 2,
    timeout: float = 300.0,
    ratio_limit: float = 1.03,
    environment: dict | None = None,
    input_seed: int = 0,
) -> dict:
    if (
        pairs != 2
        or timeout <= 0
        or ratio_limit <= 0
        or isinstance(input_seed, bool)
        or not isinstance(input_seed, int)
        or input_seed < 0
    ):
        raise ValueError("schema v1 requires pairs=2, positive timeout and ratio limit")
    correctness, candidate, baseline, manifest, operator = _validate_correctness(
        root, correctness_artifact
    )
    evidence_scope = correctness.get("evidence_scope", DEFAULT_EVIDENCE_SCOPE)
    if evidence_scope not in SUPPORTED_EVIDENCE_SCOPES:
        raise ValueError("correctness artifact has unsupported local Ascend evidence scope")
    environment = environment or probe_environment(python)
    if environment.get("fingerprint_sha256") != correctness.get("environment", {}).get("fingerprint_sha256"):
        raise ValueError("correctness and benchmark environments differ")
    cases = correctness["cases"]
    profile_root = output.with_suffix("")
    profile_root.mkdir(parents=True, exist_ok=True)
    case_results = []
    sequence = ("baseline", "candidate", "candidate", "baseline")
    for case_index, case in enumerate(cases, 1):
        test = _ref(root, case["test_path"], case["test_sha256"])
        assert test is not None
        runs = []
        for index, role in enumerate(sequence, 1):
            source = baseline if role == "baseline" else candidate
            runs.append(
                _run_msprof(
                    root, operator, role, index, source, test, python, msprof,
                    profile_root / f"case-{case_index}", timeout, input_seed,
                )
            )
            if runs[-1]["status"] != "passed":
                break
        baseline_runs = [run for run in runs if run["role"] == "baseline" and run["status"] == "passed"]
        candidate_runs = [run for run in runs if run["role"] == "candidate" and run["status"] == "passed"]
        ratio = None
        complete_runs = all(
            run.get("status") == "passed"
            and isinstance(run.get("duration_us"), (int, float))
            and isinstance(run.get("device_id"), int)
            and isinstance(run.get("frequency_mhz"), (int, float))
            for run in runs
        )
        if len(runs) == 4 and len(baseline_runs) == len(candidate_runs) == 2 and complete_runs:
            if len({run["device_id"] for run in runs}) == 1 and len({run["frequency_mhz"] for run in runs}) == 1:
                ratio = statistics.median(run["duration_us"] for run in candidate_runs) / statistics.median(
                    run["duration_us"] for run in baseline_runs
                )
        case_results.append(
            {
                "case_index": case_index,
                "test_path": case["test_path"],
                "test_sha256": case["test_sha256"],
                "input_seed": input_seed,
                "sequence": [run["role"] for run in runs],
                "baseline_runs": baseline_runs,
                "candidate_runs": candidate_runs,
                "candidate_over_baseline_ratio": ratio,
                "status": "passed" if ratio is not None and ratio <= ratio_limit else "failed",
            }
        )
        if case_results[-1]["status"] != "passed":
            break
    ratios = [case["candidate_over_baseline_ratio"] for case in case_results if case["candidate_over_baseline_ratio"] is not None]
    qualified = len(case_results) == len(cases) and len(ratios) == len(cases) and all(r <= ratio_limit for r in ratios)
    report = {
        "schema_version": 1,
        "artifact_kind": "local-ascend-paired-benchmark",
        "evidence_scope": evidence_scope,
        "operator": operator,
        "correctness_artifact_path": _relative(root, correctness_artifact),
        "correctness_artifact_sha256": _sha256(correctness_artifact),
        "candidate": {
            "id": candidate.stem,
            "source_path": _relative(root, candidate),
            "source_sha256": _sha256(candidate),
            "manifest_path": _relative(root, manifest),
            "manifest_sha256": _sha256(manifest),
        },
        "baseline": {"source_path": _relative(root, baseline), "source_sha256": _sha256(baseline)},
        "environment": environment,
        "sequence": list(sequence),
        "pairs": pairs,
        "candidate_over_baseline_ratio_limit": ratio_limit,
        "input_seed": input_seed,
        "cases": case_results,
        "qualified_for_local_performance": qualified,
        "worst_case_ratio": max(ratios) if ratios else None,
        "limitations": [
            "Only currently checked-out tests are covered; organizer case 2/3 remain unknown.",
            "This is local Ascend 910B4 evidence, not official A2/A3 performance.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-artifact", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--msprof", default="msprof")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--ratio-limit", type=float, default=1.03)
    parser.add_argument("--input-seed", type=int, default=0)
    args = parser.parse_args()
    report = run_paired_benchmark(
        ROOT, Path(args.correctness_artifact), Path(args.python), Path(args.output),
        args.msprof, 2, args.timeout, args.ratio_limit, input_seed=args.input_seed,
    )
    print(
        f"[ascend-paired] operator={report['operator']} "
        f"qualified={report['qualified_for_local_performance']} "
        f"worst_ratio={report['worst_case_ratio']}"
    )
    return 0 if report["qualified_for_local_performance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
