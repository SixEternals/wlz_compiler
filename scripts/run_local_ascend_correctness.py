#!/usr/bin/env python3
"""Run baseline and candidate against all visible tests on local Ascend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.io_utils import find_test_files


DEFAULT_EVIDENCE_SCOPE = "local_ascend_910b4_currently_visible_cases_not_official"
DERIVED_SHAPE_MATRIX_EVIDENCE_SCOPE = (
    "local_ascend_910b4_derived_shape_matrix_not_official"
)
SUPPORTED_EVIDENCE_SCOPES = frozenset(
    {
        DEFAULT_EVIDENCE_SCOPE,
        DERIVED_SHAPE_MATRIX_EVIDENCE_SCOPE,
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"evidence path is outside repository root: {path}") from exc


def _select_tests(
    root: Path,
    dataset_dir: Path,
    operator: str,
    explicit_tests: tuple[Path, ...] | None,
) -> tuple[list[Path], str]:
    if explicit_tests is None:
        tests = find_test_files(dataset_dir / operator, operator)
        selection_kind = "dataset_visible"
    else:
        if not explicit_tests:
            raise ValueError("explicit test selection must not be empty")
        root_resolved = root.resolve()
        tests = []
        seen = set()
        for test in explicit_tests:
            resolved = test.resolve()
            try:
                resolved.relative_to(root_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"explicit test is outside repository root: {test}"
                ) from exc
            if not resolved.is_file():
                raise FileNotFoundError(f"explicit test is not a file: {test}")
            if resolved in seen:
                raise ValueError(f"explicit test is duplicated: {test}")
            seen.add(resolved)
            tests.append(resolved)
        selection_kind = "explicit"
    if not tests:
        raise FileNotFoundError("operator visible tests are missing")
    return tests, selection_kind


def _fingerprint(facts: dict) -> dict:
    payload = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**facts, "fingerprint_sha256": hashlib.sha256(payload).hexdigest()}


def _valid_environment(environment: dict) -> bool:
    if not isinstance(environment, dict):
        return False
    digest = environment.get("fingerprint_sha256")
    facts = {key: value for key, value in environment.items() if key != "fingerprint_sha256"}
    return (
        isinstance(digest, str)
        and _fingerprint(facts)["fingerprint_sha256"] == digest
        and facts.get("npu_available") is True
        and str(facts.get("device_name", "")).startswith("Ascend910B4")
    )


def _prepare_compatibility_shims(work: Path, source_bytes: bytes, test_bytes: bytes) -> list[str]:
    marker = b"vllm.triton_utils"
    if marker not in source_bytes and marker not in test_bytes:
        return []
    package = work / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "triton_utils.py").write_text(
        "import triton\nimport triton.language as tl\n", encoding="utf-8"
    )
    return ["vllm.triton_utils"]


def probe_environment(python: Path, timeout: float = 30.0) -> dict:
    code = """
import json, platform, sys
import torch, torch_npu, triton
facts = {
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "machine": platform.machine(),
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "triton": triton.__version__,
    "npu_available": bool(torch.npu.is_available()),
    "device_name": torch.npu.get_device_name(0) if torch.npu.is_available() else None,
}
print("WLZ_ENV_JSON=" + json.dumps(facts, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True, timeout=timeout
    )
    marker = "WLZ_ENV_JSON="
    line = next((item for item in completed.stdout.splitlines() if item.startswith(marker)), None)
    if completed.returncode != 0 or line is None:
        raise RuntimeError(
            f"Ascend environment probe failed: rc={completed.returncode}: {completed.stderr[-512:]}"
        )
    facts = json.loads(line[len(marker):])
    if (
        not isinstance(facts, dict)
        or facts.get("npu_available") is not True
        or not str(facts.get("device_name", "")).startswith("Ascend910B4")
        or Path(facts.get("python_executable", "")).resolve() != python.resolve()
    ):
        raise RuntimeError("target interpreter is not the expected local Ascend 910B4 environment")
    return _fingerprint(facts)


def _validate_candidate(
    root: Path, operator: str, baseline: Path, candidate: Path, manifest_path: Path
) -> tuple[str, dict]:
    if _sha256(candidate) == _sha256(baseline):
        raise ValueError("candidate must not be byte-identical to baseline")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest.get("candidate") if isinstance(manifest, dict) else None
    candidate_id = candidate.stem
    if (
        not isinstance(record, dict)
        or record.get("id") != candidate_id
        or record.get("op_name") != operator
        or record.get("code_hash") != _sha256(candidate)
        or record.get("status") != "static_pass"
        or (root / Path(manifest.get("candidate_path", ""))).resolve()
        != candidate.resolve()
    ):
        raise ValueError("candidate manifest does not bind the requested source")
    return candidate_id, manifest


def _run_case(
    root: Path,
    operator: str,
    role: str,
    source: Path,
    test: Path,
    python: Path,
    log_dir: Path,
    timeout: float,
) -> dict:
    source_bytes = source.read_bytes()
    test_bytes = test.read_bytes()
    run_key = hashlib.sha256(
        role.encode() + source_bytes + test_bytes
    ).hexdigest()[:16]
    stdout_path = log_dir / f"{test.stem}.{role}.{run_key}.stdout.log"
    stderr_path = log_dir / f"{test.stem}.{role}.{run_key}.stderr.log"
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    returncode = None
    status = "failed"
    with tempfile.TemporaryDirectory(prefix=f"wlz-{operator}-{role}-") as tmp:
        work = Path(tmp)
        prepared_source = work / f"{operator}.py"
        prepared_alias = work / f"{operator}_1.py"
        prepared_test = work / test.name
        prepared_source.write_bytes(source_bytes)
        prepared_alias.write_bytes(source_bytes)
        prepared_test.write_bytes(test_bytes)
        compatibility_shims = _prepare_compatibility_shims(work, source_bytes, test_bytes)
        try:
            completed = subprocess.run(
                [str(python), prepared_test.name],
                cwd=work,
                capture_output=True,
                timeout=timeout,
            )
            stdout, stderr = completed.stdout, completed.stderr
            returncode = completed.returncode
            status = "passed" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
            status = "timeout"
        prepared_source_sha256 = _sha256(prepared_source)
        prepared_test_sha256 = _sha256(prepared_test)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    return {
        "run_id": run_key,
        "role": role,
        "status": status,
        "returncode": returncode,
        "started_at": started,
        "elapsed_seconds": time.monotonic() - start,
        "python_executable": str(python.resolve()),
        "prepared_source_sha256": prepared_source_sha256,
        "prepared_test_sha256": prepared_test_sha256,
        "compatibility_shims": compatibility_shims,
        "stdout_path": _relative(root, stdout_path),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_path": _relative(root, stderr_path),
        "stderr_sha256": _sha256(stderr_path),
    }


def run_correctness(
    root: Path,
    dataset_dir: Path,
    operator: str,
    candidate: Path,
    manifest: Path,
    python: Path,
    output: Path,
    timeout: float,
    environment: dict | None = None,
    evidence_scope: str = DEFAULT_EVIDENCE_SCOPE,
    explicit_tests: tuple[Path, ...] | None = None,
) -> dict:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if evidence_scope not in SUPPORTED_EVIDENCE_SCOPES:
        raise ValueError("unsupported local Ascend evidence scope")
    op_dir = dataset_dir / operator
    baseline = op_dir / f"{operator}.py"
    tests, test_selection_kind = _select_tests(
        root, dataset_dir, operator, explicit_tests
    )
    if not baseline.is_file() or not tests:
        raise FileNotFoundError("operator baseline or visible tests are missing")
    candidate_id, _ = _validate_candidate(root, operator, baseline, candidate, manifest)
    environment = environment or probe_environment(python)
    if not _valid_environment(environment):
        raise ValueError("environment does not identify an available Ascend 910B4")

    log_dir = output.with_suffix("")
    log_dir.mkdir(parents=True, exist_ok=True)
    case_results = []
    for test in tests:
        baseline_run = _run_case(
            root, operator, "baseline", baseline, test, python, log_dir, timeout
        )
        candidate_run = _run_case(
            root, operator, "candidate", candidate, test, python, log_dir, timeout
        )
        status = (
            "baseline_failed" if baseline_run["status"] != "passed"
            else "candidate_failed" if candidate_run["status"] != "passed"
            else "passed"
        )
        case_results.append(
            {
                "test_path": _relative(root, test),
                "test_sha256": _sha256(test),
                "baseline_run": baseline_run,
                "candidate_run": candidate_run,
                "status": status,
            }
        )
    qualified = all(item["status"] == "passed" for item in case_results)
    baseline_failed = any(item["status"] == "baseline_failed" for item in case_results)
    report = {
        "schema_version": 1,
        "artifact_kind": "local-ascend-correctness-evaluation",
        "evidence_scope": evidence_scope,
        "operator": operator,
        "candidate": {
            "id": candidate_id,
            "source_path": _relative(root, candidate),
            "source_sha256": _sha256(candidate),
            "manifest_path": _relative(root, manifest),
            "manifest_sha256": _sha256(manifest),
        },
        "baseline": {
            "source_path": _relative(root, baseline),
            "source_sha256": _sha256(baseline),
        },
        "environment": environment,
        "compile_status": "unknown",
        "process_status": "passed" if qualified else "failed",
        "correctness_status": (
            "passed" if qualified else "unknown" if baseline_failed else "failed"
        ),
        "case_count": len(tests),
        "visible_case_count": len(tests) if test_selection_kind == "dataset_visible" else 0,
        "test_selection": {
            "kind": test_selection_kind,
            "paths": [_relative(root, test) for test in tests],
        },
        "qualified_for_local_performance": qualified,
        "cases": case_results,
        "limitations": [
            (
                "Only explicitly supplied local tests are covered; organizer cases remain unknown."
                if test_selection_kind == "explicit"
                else "Only currently checked-out tests are covered; organizer case 2/3 remain unknown."
            ),
            "A passing direct test process does not provide a separately observed compile phase.",
            "This Ascend 910B4 result is not an official A2/A3 evaluation.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--test",
        action="append",
        type=Path,
        dest="explicit_tests",
        help="Run only this repository-local test file; repeat for multiple tests.",
    )
    parser.add_argument(
        "--evidence-scope",
        choices=sorted(SUPPORTED_EVIDENCE_SCOPES),
        default=DEFAULT_EVIDENCE_SCOPE,
    )
    args = parser.parse_args()
    report = run_correctness(
        ROOT,
        Path(args.dataset_dir),
        args.operator,
        Path(args.candidate),
        Path(args.manifest),
        Path(args.python),
        Path(args.output),
        args.timeout,
        evidence_scope=args.evidence_scope,
        explicit_tests=(tuple(args.explicit_tests) if args.explicit_tests else None),
    )
    print(
        f"[ascend-correctness] operator={args.operator} "
        f"status={report['correctness_status']} cases={report['visible_case_count']}"
    )
    return 0 if report["qualified_for_local_performance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
