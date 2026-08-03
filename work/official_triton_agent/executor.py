#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton Executor Module - Encapsulates Evaluation Interface (msprof version, JSON baseline)

[Note] This file is provided by the organizers, students do not need to modify it
"""

import os
import re
import json
import subprocess
import csv
import hashlib
import importlib.metadata
import io
import math
import platform
import sys
import tempfile
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from config import EAConfig


MAX_PROFILE_CSV_BYTES = 4 * 1024 * 1024
PROFILE_PARSER_RULE = "op-basic-info:first-exact-op-name:task-duration-us:v1"


# Icon definitions
ICONS = {
    'rocket': '🚀',
    'timer': '⏱️',
    'check': '✅',
    'cross': '❌',
    'gear': '⚙️',
    'chart': '📊',
    'sparkle': '✨',
    'warning': '⚠️',
    'bulb': '💡',
    'stopwatch': '⏱️',
    'target': '🎯',
    'zap': '⚡',
    'trophy': '🏆',
    'microscope': '🔬',
    'repeat': '🔄',
    'save': '💾'
}


@dataclass
class EvaluationResult:
    """Evaluation result"""
    success: bool
    execution_time: float  # Unit: microseconds (us)
    speedup: float
    fitness: float
    error: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


class TritonExecutor:
    """
    Triton Code Execution and Performance Evaluator (encapsulated, students do not need to modify)

    Provides unified interface:
    1. Read baseline time from JSON
    2. Run msprof performance test
    3. Calculate speedup and fitness score
    """

    def __init__(self, 
                 baseline_time: float,
                 test_code_path: str,
                 config: EAConfig,
                 kernel_name: str = "kernel",
                 work_dir: Optional[Path] = None):
        """
        Initialize executor

        Args:
            baseline_time: Baseline execution time (microseconds), read directly from JSON
            test_code_path: Test code file path
            config: Configuration object
            kernel_name: Operator name
            work_dir: Working directory
        """
        self.baseline_time = baseline_time  # Use the passed-in baseline directly
        self.test_code_path = Path(test_code_path)
        self.config = config
        self.kernel_name = kernel_name
        self.work_dir = work_dir or Path(".")
        self.performance_dir = self.work_dir / "performance"
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        self._last_profile_observation: Dict[str, Any] = {}

        print(f"\n[{ICONS['rocket']}] [Executor] Initializing Triton executor...")
        print(f"       └─ Operator name: {kernel_name}")
        print(f"       └─ Working directory: {self.work_dir}")
        print(f"       └─ Baseline time: {baseline_time:.2f}us (read from JSON)")
        print(f"       └─ Test file: {self.test_code_path}")

    def _find_latest_opprof_dir(self, result_dir: Path) -> Optional[Path]:
        """Find the latest OPPROF_* directory"""
        if not result_dir.exists():
            return None

        opprof_dirs = [
            d for d in result_dir.iterdir() 
            if d.is_dir() and d.name.startswith("OPPROF_")
        ]

        if not opprof_dirs:
            return None

        if len(opprof_dirs) != 1:
            return None
        return opprof_dirs[0]

    @property
    def last_profile_observation(self) -> Dict[str, Any]:
        """Return a JSON-safe copy of the most recent successful parse."""
        return json.loads(json.dumps(self._last_profile_observation))

    @staticmethod
    def _toolchain_fingerprint() -> Dict[str, Any]:
        packages = {}
        for package in ("torch", "torch-npu", "triton"):
            try:
                packages[package] = importlib.metadata.version(package)
            except Exception:
                # Package metadata is optional and must never invalidate timing.
                packages[package] = None
        facts = {
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "packages": packages,
        }
        payload = json.dumps(
            facts, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return {"facts": facts, "sha256": hashlib.sha256(payload).hexdigest()}

    def _relative_audit_path(self, path: Path) -> Optional[str]:
        try:
            relative = path.resolve(strict=True).relative_to(
                self.work_dir.resolve(strict=True)
            )
        except (OSError, RuntimeError, ValueError):
            return None
        if not relative.parts or ".." in relative.parts:
            return None
        return relative.as_posix()

    def _read_profile_observation(self, result_dir: Path) -> Optional[Dict[str, Any]]:
        """Read one bounded CSV snapshot and derive both timing and evidence."""
        opprof_dir = self._find_latest_opprof_dir(result_dir)
        if not opprof_dir:
            return None

        csv_path = opprof_dir / "OpBasicInfo.csv"
        csv_relative = self._relative_audit_path(csv_path)
        if csv_relative is None:
            return None
        try:
            if csv_path.stat().st_size > MAX_PROFILE_CSV_BYTES:
                return None
            csv_bytes = csv_path.read_bytes()
            csv_text = csv_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        reader = csv.DictReader(io.StringIO(csv_text))
        fieldnames = reader.fieldnames or []
        if (
            fieldnames.count("Op Name") != 1
            or fieldnames.count("Task Duration(us)") != 1
        ):
            return None
        for row_index, row in enumerate(reader, 1):
            if row.get("Op Name") != self.kernel_name:
                continue
            try:
                duration = float(row["Task Duration(us)"])
            except (TypeError, ValueError, KeyError):
                return None
            if not math.isfinite(duration) or duration <= 0:
                return None
            return {
                "schema_version": 1,
                "kind": "msprof-op-observation",
                "path_base": "executor_work_dir",
                "run_directory_id": result_dir.name,
                "csv_path": csv_relative,
                "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                "parser_rule": PROFILE_PARSER_RULE,
                "parse_status": "parsed",
                "kernel_name": self.kernel_name,
                "target_row_index": row_index,
                "execution_time_us": duration,
                "toolchain_fingerprint": self._toolchain_fingerprint(),
            }
        return None

    def _parse_op_basic_info(self, result_dir: Path) -> Optional[float]:
        """
        Parse the first exact target-kernel Task Duration from OpBasicInfo.csv.

        The organizer baseline contains one duration per test case and was
        produced with first-record semantics. Keep that comparison contract,
        but never score an unrelated op or silently skip a corrupt target row.
        Return execution time (microseconds), return None on failure
        """
        observation = self._read_profile_observation(result_dir)
        self._last_profile_observation = observation or {}
        return observation["execution_time_us"] if observation is not None else None

    def _run_msprof(self, test_file: Path, timeout: int = 300) -> Optional[float]:
        """
        Run msprof op command to get performance data

        Returns:
            Execution time (microseconds), return None on failure
        """
        kernel_performance_dir = self.performance_dir / self.kernel_name
        kernel_performance_dir.mkdir(parents=True, exist_ok=True)
        result_dir = Path(tempfile.mkdtemp(prefix="run-", dir=kernel_performance_dir))
        self._last_profile_observation = {}

        test_script_abs = str(test_file.resolve())

        # Keep each run's output isolated and avoid shell parsing of paths.
        cmd = [
            "msprof",
            "op",
            f"--output={result_dir}",
            f"--application={sys.executable} {test_script_abs}",
            f"--kernel-name={self.kernel_name}",
            "--aic-metrics=MemoryDetail,Occupancy,PipeUtilization,Roofline",
        ]

        print(" ".join(cmd))

        log_file = result_dir / "get_prof.log"

        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    shell=False,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout
                )

            if result.returncode != 0:
                print(f"msprof fail")
                return None

            # Parse result
            return self._parse_op_basic_info(result_dir)

        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None

    def evaluate(self, code: str, timeout: int = 1200) -> EvaluationResult:
        """
        Evaluate a single operator code - [Core Interface]

        Complete workflow:
        1. Create test environment (modify import)
        2. Run msprof performance test
        3. Calculate speedup and fitness
        """
        print(f"\n[{ICONS['microscope']}] [Executor] Starting evaluation of optimized code...")

        # Step 1: Create temporary environment
        print(f"[{ICONS['gear']}] [Executor] Step 1/3: Preparing test environment...")
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.kernel_name}_"))

        # Write code
        kernel_file = temp_dir / f"{self.kernel_name}.py"
        with open(kernel_file, 'w', encoding='utf-8') as f:
            f.write(code)

        # Copy test file and modify import
        with open(self.test_code_path, 'r', encoding='utf-8') as f:
            test_content = f.read()

        # Modify import: from kernel import ... -> from {kernel_name} import ...
        modified_test = re.sub(
            r'^from\s+kernel\s+import',
            f'from {self.kernel_name} import',
            test_content,
            flags=re.MULTILINE
        )
        modified_test = re.sub(
            r'^import\s+kernel\b',
            f'import {self.kernel_name}',
            modified_test,
            flags=re.MULTILINE
        )

        test_file = temp_dir / f"test_{self.kernel_name}.py"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(modified_test)

        print(f"       └─ Code length: {len(code)} characters")

        # Step 2: Run performance test
        print(f"[{ICONS['gear']}] [Executor] Step 2/3: Running performance test...")
        current_time = self._run_msprof(test_file, timeout=timeout)

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        if current_time is None:
            print(f"[{ICONS['cross']}] [Executor] Performance test failed!")
            return EvaluationResult(
                success=False,
                execution_time=0.0,
                speedup=0.0,
                fitness=0.0,
                error="Performance test failed"
            )

        # Step 3: Calculate speedup
        print(f"[{ICONS['check']}] [Executor] Step 3/3: Calculating speedup...")
        print(f"       └─ Baseline time: {self.baseline_time:.2f}us")
        print(f"       └─ Optimized time: {current_time:.2f}us")

        if current_time > 0 and self.baseline_time > 0:
            # Calculation formula: speedup = baseline/current - 1
            raw_speedup = self.baseline_time / current_time - 1
            speedup = max(raw_speedup, 0.0)
            # Cap upper limit at 2.0 (corresponding to 200 points)
            fitness = min(speedup, 2.0)

            print(f"       └─ Raw speedup: {raw_speedup:.4f}")

            if raw_speedup < 0:
                print(f"[{ICONS['warning']}]       └─ Warning: Optimized slower than baseline, speedup set to 0")
            elif raw_speedup > 2.0:
                print(f"[{ICONS['trophy']}]       └─ Speedup exceeds upper limit (2.0), calculated as 2.0")
            else:
                print(f"[{ICONS['zap']}]       └─ Speedup valid, counted in score")

            print(f"[{ICONS['target']}] [Executor] Evaluation complete!")
            print(f"       └─ Final speedup: {speedup:.4f}")
            print(f"       └─ Fitness score: {fitness:.4f} (max 2.0)")

            # Convert to competition score
            competition_score = fitness * 100
            print(f"       └─ Competition score: {competition_score:.1f}/200 points")

        else:
            print(f"[{ICONS['warning']}] [Executor] Execution time invalid, score set to 0")
            speedup = 0.0
            fitness = 0.0

        return EvaluationResult(
            success=True,
            execution_time=current_time,
            speedup=speedup,
            fitness=fitness,
            error=None,
            evidence={"profile": self.last_profile_observation},
        )
