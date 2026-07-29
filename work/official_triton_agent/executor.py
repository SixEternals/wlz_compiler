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
import tempfile
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from config import EAConfig


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

        return max(opprof_dirs, key=lambda d: d.stat().st_mtime)

    def _parse_op_basic_info(self, result_dir: Path) -> Optional[float]:
        """
        Parse OpBasicInfo.csv to get Task Duration
        Return execution time (microseconds), return None on failure
        """
        opprof_dir = self._find_latest_opprof_dir(result_dir)
        if not opprof_dir:
            return None

        csv_path = opprof_dir / "OpBasicInfo.csv"
        if not csv_path.exists():
            return None

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        duration = float(row.get('Task Duration(us)', 0))
                        if duration > 0:
                            return duration
                    except (ValueError, KeyError):
                        continue
        except Exception:
            pass

        return None

    def _run_msprof(self, test_file: Path, timeout: int = 300) -> Optional[float]:
        """
        Run msprof op command to get performance data

        Returns:
            Execution time (microseconds), return None on failure
        """
        result_dir = self.performance_dir / self.kernel_name
        result_dir.mkdir(parents=True, exist_ok=True)

        test_script_abs = str(test_file.resolve())

        # Build msprof command
        cmd_str = (
            f'msprof op '
            f'--output={result_dir} '
            f'--application="python3 {test_script_abs}" '
            f'--kernel-name="{self.kernel_name}" '
            f'--aic-metrics=MemoryDetail,Occupancy,PipeUtilization,Roofline'
        )

        print(cmd_str)

        log_file = result_dir / "get_prof.log"

        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd_str,
                    shell=True,
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
            error=None
        )