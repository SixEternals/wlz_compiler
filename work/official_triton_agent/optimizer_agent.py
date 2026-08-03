#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton Optimization Agent Main Class - Unified Interface

[Note] This file is provided by the organizers, students do not need to modify it
This class is responsible for coordinating components and providing a standard calling interface for the evaluation system
"""

import os
import time
import json
import re
import hashlib
import math
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import EAConfig
from llm_interface import LLMInterface
from contract_executor import ContractCheckingExecutor, MultiCaseContractExecutor
from executor import TritonExecutor
from genetic_operators import GeneticOperators
from evolutionary_algorithm import EvolutionaryAlgorithm
from wlz_optimizer.budget import BudgetController, BudgetLimits


def get_baseline_from_json(baseline_json_path: str, kernel_name: str, test_case_id: int = 1) -> Optional[float]:
    """
    Read baseline time for the specified kernel and test case from the baseline JSON file

    Args:
        baseline_json_path: Baseline JSON file path
        kernel_name: Kernel name
        test_case_id: Test case number (1, 2, 3)

    Returns:
        Baseline time (us), returns None if not found
    """
    try:
        with open(baseline_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        grouped_data = data.get('grouped_by_kernel', {})

        if kernel_name not in grouped_data:
            print(f"[Baseline] Warning: Kernel not found in JSON: {kernel_name}")
            return None

        kernel_baseline = grouped_data[kernel_name]

        # Find the baseline corresponding to test_case_id
        for item in kernel_baseline:
            test_file = item.get('test_file', '')
            # Extract test case number: test_kernel_name_1.py -> 1
            match = re.search(r'_([0-9]+)\.py$', test_file)
            if match:
                current_id = int(match.group(1))
                if current_id == test_case_id:
                    return item.get('task_duration_us')

        print(f"[Baseline] Warning: Cannot find baseline for kernel {kernel_name} test_case {test_case_id}")
        return None

    except Exception as e:
        print(f"[Baseline] Error: Failed to read JSON: {e}")
        return None


class TritonOptimizerAgent:
    """
    Triton Optimization Agent Main Class (Unified Interface)

    Students complete the optimization logic by implementing EvolutionaryAlgorithm and GeneticOperators
    This class is responsible for coordinating components and providing a standard calling interface for the evaluation system

    Evaluation interfaces:
    - setup(): Initialize execution environment
    - optimize(): Execute optimization (standard interface)
    - get_results(): Get optimization results (return up to 5 versions)
    """

    def __init__(self, config: Optional[EAConfig] = None):
        """
        Initialize Agent

        Args:
            config: Configuration object, if None use default configuration
        """
        self.config = config or EAConfig()
        self.budget = BudgetController(BudgetLimits(
            token_limit=getattr(self.config, "max_total_tokens", 200_000),
            wall_time_seconds=20 * 60,
        ))
        self._budget_run_started = False
        self.config.budget_controller = self.budget
        self.llm = LLMInterface(self.config)
        self.executor: Optional[TritonExecutor] = None
        self.ea: Optional[EvolutionaryAlgorithm] = None

        # Result records
        self.optimization_history: List[Dict] = []

    def setup(self, 
              baseline_code: str, 
              test_code: Optional[str] = None,
              kernel_name: str = "kernel",
              work_dir: Optional[str] = None,
              test_case_id: int = 1,
              test_cases: Optional[List[tuple[int, str]]] = None):
        """
        Initialize execution environment

        Args:
            baseline_code: Baseline Triton code (used to get code structure, not for performance measurement)
            test_code: Test code (import statements will be modified)
            kernel_name: Operator name
            work_dir: Working directory path
            test_case_id: Which test case baseline to use (1, 2, 3)
            test_cases: Explicit ``(case id, test path)`` pairs. Mutually
                exclusive with ``test_code``.
        """
        print(f"[Agent] Initializing execution environment...")
        print(f"       └─ Operator: {kernel_name}")
        self.config.budget_controller = self.budget
        self.llm.budget_controller = self.budget
        self.llm.total_tokens_used = self.budget.snapshot().used_tokens

        if test_cases is not None and test_code is not None:
            raise ValueError("Provide test_code or test_cases, not both")
        explicit_cases = test_cases is not None
        selected_cases = test_cases or (
            [(test_case_id, test_code)] if test_code is not None else []
        )
        if not selected_cases:
            raise ValueError("At least one test case is required")

        work_dir_path = Path(work_dir) if work_dir else Path(".")
        executors = []
        for case_id, case_test in selected_cases:
            baseline_time = get_baseline_from_json(
                self.config.baseline_json, kernel_name, case_id
            )
            if baseline_time is None:
                raise RuntimeError(
                    f"Cannot read baseline for kernel {kernel_name} test case {case_id}"
                )
            print(f"       └─ Test case {case_id} baseline: {baseline_time:.2f}μs")
            if os.path.isfile(case_test):
                test_code_path = case_test
            else:
                filename = (
                    f"test_{kernel_name}_{case_id}.py"
                    if explicit_cases
                    else f"test_{kernel_name}.py"
                )
                test_code_path = work_dir_path / filename
                test_code_path.write_text(case_test, encoding="utf-8")
            executors.append((case_id, ContractCheckingExecutor(
                baseline_time=baseline_time,
                test_code_path=str(test_code_path),
                config=self.config,
                kernel_name=kernel_name,
                work_dir=work_dir_path,
                baseline_code=baseline_code,
            )))

        self.executor = MultiCaseContractExecutor(executors)

        genetic_ops = GeneticOperators(self.llm, self.config)
        self.ea = EvolutionaryAlgorithm(genetic_ops, self.executor, self.config)

    def optimize(self, seed_codes: List[str], max_time: int = 600) -> Dict[str, Any]:
        """
        Execute optimization - [Core interface, called by evaluation system]

        Workflow:
        1. Run evolutionary algorithm
        2. Collect optimization results
        3. Return dictionary containing code, fitness, and statistics

        Args:
            seed_codes: Initial seed code list (must include baseline_code)
            max_time: Maximum running time (seconds), controlled by evaluation system

        Returns:
            result: Dictionary containing the following fields:
                - best_code: Best code
                - best_fitness: Best fitness
                - speedup: Speedup ratio
                - generations: Actual evolution generations
                - time_elapsed: Time consumed
                - llm_stats: LLM call statistics
                - top5_codes: Top 5 best codes (for final submission)
        """
        if self.ea is None:
            raise RuntimeError("Please call setup() first to initialize")

        budget = getattr(self, "budget", None)
        if budget is not None:
            budget_state = budget.snapshot()
            if not getattr(self, "_budget_run_started", False):
                wall_limit = budget.limits.wall_time_seconds
                remaining = (
                    min(float(max_time), wall_limit) if max_time else wall_limit
                )
                budget_state["elapsed_seconds"] = wall_limit - remaining
                if budget_state.stop_reason == "wall_time_limit":
                    budget_state["stop_reason"] = None
                budget.restore(budget_state)
                self._budget_run_started = True
            self.config.budget_controller = budget
            self.llm.budget_controller = budget
            self.llm.total_tokens_used = budget.snapshot().used_tokens

        print(f"\n[Agent] Starting optimization, max time: {max_time}s")
        start_time = time.time()
        self._baseline_code = seed_codes[0] if seed_codes else None
        # Run evolutionary algorithm (max_time enforced as wall clock budget)
        best = self.ea.run(seed_codes, deadline_seconds=max_time)

        # Collect results
        elapsed = time.time() - start_time

        # Get top5 individuals (for final submission, at most 5)
        top5_individuals = self._get_top_k(5)
        top5_codes = [
            {
                'code': ind.code,
                **self._candidate_provenance(ind),
            }
            for ind in top5_individuals
        ]
        output_best = top5_individuals[0] if top5_individuals else best

        result = {
            'best_code': output_best.code,
            'best_fitness': output_best.fitness,
            'speedup': output_best.metadata.get('speedup', 0),
            'generations': self.ea.generation,
            'time_elapsed': elapsed,
            'llm_stats': self.llm.get_stats(),
            'top5_codes': top5_codes,
            'best_provenance': self._candidate_provenance(output_best),
        }

        self.optimization_history.append(result)

        print(f"\n[Agent] Optimization completed:")
        print(f"  - Best fitness: {result['best_fitness']:.4f}")
        print(f"  - Speedup: {result['speedup']:.4f}")
        print(f"  - Evolution generations: {result['generations']}")
        print(f"  - Time elapsed: {result['time_elapsed']:.2f}s")
        print(f"  - LLM call count: {result['llm_stats']['call_count']}")

        return result

    def _get_top_k(self, k: int) -> List[Any]:
        """
        Get top-k individuals

        Args:
            k: Number of individuals to return

        Returns:
            List[Individual]: Top k individuals sorted by fitness,
            deduplicated by code content and excluding the baseline code
        """
        baseline_key = self._baseline_code.strip() if getattr(self, '_baseline_code', None) else None
        seen = set()
        top = []
        for ind in sorted(self.ea.population, key=lambda x: x.fitness, reverse=True):
            metadata = getattr(ind, 'metadata', {})
            if not isinstance(metadata, dict) or metadata.get('success') is not True:
                continue
            key = ind.code.strip()
            if key == baseline_key or key in seen:
                continue
            seen.add(key)
            top.append(ind)
            if len(top) == k:
                break
        return top

    @staticmethod
    def _safe_profile_observation(
        value: Any, execution_time_us: Any
    ) -> Optional[Dict[str, Any]]:
        fields = {
            'schema_version', 'kind', 'path_base', 'run_directory_id',
            'csv_path', 'csv_sha256', 'parser_rule', 'parse_status',
            'kernel_name', 'target_row_index', 'execution_time_us',
            'toolchain_fingerprint',
        }
        if not isinstance(value, dict) or set(value) != fields:
            return None

        def sha256(item):
            return (
                isinstance(item, str) and len(item) == 64
                and item == item.lower()
                and all(char in '0123456789abcdef' for char in item)
            )

        def text(item, limit=255):
            return (
                isinstance(item, str) and 1 <= len(item) <= limit
                and '\x00' not in item and '\n' not in item and '\r' not in item
            )

        run_id = value.get('run_directory_id')
        csv_path = value.get('csv_path')
        path_parts = csv_path.split('/') if isinstance(csv_path, str) else []
        profile_time = value.get('execution_time_us')
        if (
            value.get('schema_version') != 1
            or type(value.get('schema_version')) is not int
            or value.get('kind') != 'msprof-op-observation'
            or value.get('path_base') != 'executor_work_dir'
            or not text(run_id, 128) or not run_id.startswith('run-')
            or '/' in run_id or '\\' in run_id or run_id in {'.', '..'}
            or not text(csv_path, 1024) or '\\' in csv_path
            or csv_path.startswith('/') or any(part in {'', '.', '..'} for part in path_parts)
            or len(path_parts) < 3 or path_parts[-3] != run_id
            or not path_parts[-2].startswith('OPPROF_')
            or path_parts[-1] != 'OpBasicInfo.csv'
            or not sha256(value.get('csv_sha256'))
            or value.get('parser_rule')
                != 'op-basic-info:first-exact-op-name:task-duration-us:v1'
            or value.get('parse_status') != 'parsed'
            or not text(value.get('kernel_name'))
            or type(value.get('target_row_index')) is not int
            or value.get('target_row_index') < 1
            or not isinstance(profile_time, (int, float))
            or isinstance(profile_time, bool) or not math.isfinite(profile_time)
            or profile_time <= 0
            or not math.isclose(profile_time, execution_time_us, rel_tol=1e-12)
        ):
            return None

        fingerprint = value.get('toolchain_fingerprint')
        if not isinstance(fingerprint, dict) or set(fingerprint) != {'facts', 'sha256'}:
            return None
        facts = fingerprint.get('facts')
        fact_fields = {'python_version', 'machine', 'system', 'release', 'packages'}
        if not isinstance(facts, dict) or set(facts) != fact_fields:
            return None
        packages = facts.get('packages')
        if (
            not isinstance(packages, dict)
            or set(packages) != {'torch', 'torch-npu', 'triton'}
            or any(value is not None and not text(value, 128) for value in packages.values())
            or any(not text(facts[key], 255) for key in fact_fields - {'packages'})
        ):
            return None
        payload = json.dumps(
            facts, sort_keys=True, separators=(',', ':'), ensure_ascii=True
        ).encode('utf-8')
        if (
            not sha256(fingerprint.get('sha256'))
            or hashlib.sha256(payload).hexdigest() != fingerprint.get('sha256')
        ):
            return None
        return json.loads(json.dumps(value))

    @staticmethod
    def _safe_v2_evaluation_evidence(
        value: Any, metadata: Dict[str, Any], fitness: Any
    ) -> Optional[Dict[str, Any]]:
        aggregation = {
            'execution_time_us': 'sum-completed-case-time-v1',
            'speedup': 'minimum-successful-case-speedup-v1',
            'fitness': 'minimum-successful-case-fitness-v1',
        }
        top_keys = {
            'schema_version', 'kind', 'official_aggregate',
            'aggregation', 'case_results',
        }
        if (
            not isinstance(value, dict) or set(value) != top_keys
            or type(value.get('schema_version')) is not int
            or value.get('schema_version') != 2
            or value.get('kind') != 'multi-case-real-evaluation'
            or value.get('official_aggregate') is not False
            or value.get('aggregation') != aggregation
        ):
            return None
        cases = value.get('case_results')
        if not isinstance(cases, list) or not 1 <= len(cases) <= 3:
            return None

        case_fields = (
            'case_id', 'status', 'success', 'baseline_time_us',
            'execution_time_us', 'speedup', 'fitness', 'test_file',
            'test_sha256', 'baseline_code_sha256', 'profile',
        )
        case_keys = set(case_fields)

        def number(item, *, positive=False):
            return (
                isinstance(item, (int, float)) and not isinstance(item, bool)
                and math.isfinite(item) and (item > 0 if positive else item >= 0)
            )

        def sha256(item):
            return (
                isinstance(item, str) and len(item) == 64 and item == item.lower()
                and all(char in '0123456789abcdef' for char in item)
            )

        normalized = []
        baseline_hash = None
        for expected_id, item in enumerate(cases, 1):
            if not isinstance(item, dict) or set(item) != case_keys:
                return None
            status = item.get('status')
            test_file = item.get('test_file')
            code_hash = item.get('baseline_code_sha256')
            if (
                type(item.get('case_id')) is not int or item.get('case_id') != expected_id
                or status not in {'passed', 'failed', 'not_run_budget_exhausted'}
                or item.get('success') is not (status == 'passed')
                or not number(item.get('baseline_time_us'), positive=True)
                or not isinstance(test_file, str) or not 1 <= len(test_file) <= 255
                or '/' in test_file or '\\' in test_file or not test_file.endswith('.py')
                or not sha256(item.get('test_sha256')) or not sha256(code_hash)
                or (baseline_hash is not None and code_hash != baseline_hash)
            ):
                return None
            baseline_hash = code_hash
            metrics = (
                item.get('execution_time_us'), item.get('speedup'), item.get('fitness')
            )
            profile = None
            if status == 'passed':
                if not all(number(metric, positive=index == 0) for index, metric in enumerate(metrics)):
                    return None
                expected_speedup = max(item['baseline_time_us'] / metrics[0] - 1.0, 0.0)
                expected_fitness = min(expected_speedup, 2.0)
                if (
                    not math.isclose(metrics[1], expected_speedup, rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(metrics[2], expected_fitness, rel_tol=1e-12, abs_tol=1e-12)
                ):
                    return None
                profile = TritonOptimizerAgent._safe_profile_observation(
                    item.get('profile'), metrics[0]
                )
                if profile is None:
                    return None
            elif status == 'failed':
                if item.get('profile') is not None or not all(number(metric) and metric == 0 for metric in metrics):
                    return None
            elif item.get('profile') is not None or any(metric is not None for metric in metrics):
                return None
            normalized.append({key: item[key] for key in case_fields if key != 'profile'} | {'profile': profile})

        statuses = [item['status'] for item in normalized]
        if any(status != 'passed' for status in statuses[:-1]):
            return None
        expected_time = sum(item['execution_time_us'] or 0.0 for item in normalized)
        all_passed = all(status == 'passed' for status in statuses)
        expected_speedup = min(item['speedup'] for item in normalized) if all_passed else 0.0
        expected_fitness = min(item['fitness'] for item in normalized) if all_passed else 0.0
        actuals = (metadata.get('execution_time'), metadata.get('speedup'), fitness)
        if (
            metadata.get('success') is not all_passed
            or any(
                not number(actual)
                or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                for actual, expected in zip(actuals, (expected_time, expected_speedup, expected_fitness))
            )
        ):
            return None
        return {
            'schema_version': 2,
            'kind': 'multi-case-real-evaluation',
            'official_aggregate': False,
            'aggregation': dict(aggregation),
            'case_results': normalized,
        }

    @staticmethod
    def _safe_evaluation_evidence(
        value: Any, metadata: Dict[str, Any], fitness: Any
    ) -> Optional[Dict[str, Any]]:
        """Return validated multi-case evidence without unknown/raw fields."""
        if isinstance(value, dict) and value.get('schema_version') == 2:
            return TritonOptimizerAgent._safe_v2_evaluation_evidence(
                value, metadata, fitness
            )
        aggregation = {
            'execution_time_us': 'sum-completed-case-time-v1',
            'speedup': 'minimum-successful-case-speedup-v1',
            'fitness': 'minimum-successful-case-fitness-v1',
        }
        top_keys = {
            'schema_version', 'kind', 'official_aggregate',
            'aggregation', 'case_results',
        }
        if (
            not isinstance(value, dict)
            or set(value) != top_keys
            or type(value.get('schema_version')) is not int
            or value.get('schema_version') != 1
            or value.get('kind') != 'multi-case-real-evaluation'
            or value.get('official_aggregate') is not False
            or value.get('aggregation') != aggregation
        ):
            return None
        cases = value.get('case_results')
        if not isinstance(cases, list) or not 2 <= len(cases) <= 3:
            return None

        case_fields = (
            'case_id', 'status', 'success', 'baseline_time_us',
            'execution_time_us', 'speedup', 'fitness', 'test_file',
            'test_sha256', 'baseline_code_sha256',
        )
        case_keys = set(case_fields)

        def number(item, *, minimum=0.0, positive=False):
            return (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(item)
                and (item > minimum if positive else item >= minimum)
            )

        def sha256(item):
            return (
                isinstance(item, str)
                and len(item) == 64
                and item == item.lower()
                and all(char in '0123456789abcdef' for char in item)
            )

        normalized = []
        for expected_id, item in enumerate(cases, 1):
            if not isinstance(item, dict) or set(item) != case_keys:
                return None
            status = item.get('status')
            success = item.get('success')
            test_file = item.get('test_file')
            if (
                type(item.get('case_id')) is not int
                or item.get('case_id') != expected_id
                or status not in {'passed', 'failed', 'not_run_budget_exhausted'}
                or not isinstance(success, bool)
                or success is not (status == 'passed')
                or not number(item.get('baseline_time_us'), positive=True)
                or not isinstance(test_file, str)
                or not 1 <= len(test_file) <= 255
                or '/' in test_file or '\\' in test_file
                or not test_file.endswith('.py')
                or not sha256(item.get('test_sha256'))
                or not sha256(item.get('baseline_code_sha256'))
            ):
                return None
            metrics = (
                item.get('execution_time_us'), item.get('speedup'), item.get('fitness')
            )
            if status == 'not_run_budget_exhausted':
                if any(metric is not None for metric in metrics):
                    return None
            elif (
                not number(metrics[0], positive=status == 'passed')
                or not number(metrics[1])
                or not number(metrics[2])
                or metrics[2] > 2.0
            ):
                return None
            normalized.append({key: item[key] for key in case_fields})

        statuses = [item['status'] for item in normalized]
        if any(status != 'passed' for status in statuses[:-1]):
            return None
        if all(status == 'passed' for status in statuses):
            expected_time = sum(item['execution_time_us'] for item in normalized)
            expected_speedup = min(item['speedup'] for item in normalized)
            expected_fitness = min(item['fitness'] for item in normalized)
            actuals = (metadata.get('execution_time'), metadata.get('speedup'), fitness)
            expecteds = (expected_time, expected_speedup, expected_fitness)
            if metadata.get('success') is not True or any(
                not number(actual) or not math.isclose(actual, expected, rel_tol=1e-12)
                for actual, expected in zip(actuals, expecteds)
            ):
                return None

        return {
            'schema_version': 1,
            'kind': 'multi-case-real-evaluation',
            'official_aggregate': False,
            'aggregation': dict(aggregation),
            'case_results': normalized,
        }

    @staticmethod
    def _candidate_provenance(individual: Any) -> Dict[str, Any]:
        """Return bounded, source-verifiable provenance without raw logs."""
        code = getattr(individual, 'code', '')
        metadata = getattr(individual, 'metadata', {})
        metadata = metadata if isinstance(metadata, dict) else {}
        parent_ids = []
        direct_parent = metadata.get('parent')
        if isinstance(direct_parent, str) and direct_parent:
            parent_ids.append(direct_parent)
        parents = metadata.get('parents', [])
        if not isinstance(parents, (list, tuple)):
            parents = []
        for parent in parents:
            if isinstance(parent, str) and parent and parent not in parent_ids:
                parent_ids.append(parent)
        raw_lineage = metadata.get('lineage', [])
        if not isinstance(raw_lineage, (list, tuple)):
            raw_lineage = []
        lineage = [
            ancestor for ancestor in raw_lineage
            if isinstance(ancestor, str) and ancestor
        ]
        context = metadata.get('prompt_context')
        failure_categories = []
        if isinstance(context, dict):
            for item in context.get('failure_category_counts', []):
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], int)
                ):
                    failure_categories.append([item[0], item[1]])
        fitness = getattr(individual, 'fitness', 0.0)
        provenance = {
            'id': getattr(individual, 'id', None),
            'code_hash': hashlib.sha256(code.encode('utf-8')).hexdigest(),
            'fitness': fitness,
            'generation': getattr(individual, 'generation', 0),
            'parent_ids': parent_ids,
            'lineage': lineage,
            'mutation_kind': metadata.get('mutation_type') or metadata.get('operation') or 'unknown',
            'model_used': getattr(individual, 'model_used', None),
            'prompt_id': metadata.get('prompt_id') if isinstance(metadata.get('prompt_id'), str) else None,
            'failure_category_counts': failure_categories,
            'evaluation_status': {
                'evaluated': bool(metadata.get('evaluated', False)),
                'success': metadata.get('success') if isinstance(metadata.get('success'), bool) else None,
                'speedup': metadata.get('speedup') if isinstance(metadata.get('speedup'), (int, float)) else None,
                'execution_time_us': (
                    metadata.get('execution_time')
                    if isinstance(metadata.get('execution_time'), (int, float))
                    else None
                ),
            },
        }
        evaluation_evidence = TritonOptimizerAgent._safe_evaluation_evidence(
            metadata.get('evaluation_evidence'), metadata, fitness
        )
        if evaluation_evidence is not None:
            provenance['evaluation_evidence'] = evaluation_evidence
        return provenance

    def save_results(self, output_dir: str, kernel_name: str) -> None:
        """
        Save optimization results to file

        Args:
            output_dir: Output directory
            kernel_name: Operator name
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.optimization_history:
            print("[Agent] No optimization results to save")
            return

        latest_result = self.optimization_history[-1]
        if not latest_result['top5_codes']:
            raise RuntimeError(
                "No non-baseline candidate is available for organizer output"
            )

        # Save best code
        best_code_path = output_path / f"{kernel_name}_best.py"
        with open(best_code_path, 'w') as f:
            f.write(latest_result['best_code'])
        print(f"[Agent] Best code saved: {best_code_path}")

        # Save top5 codes
        for i, code_info in enumerate(latest_result['top5_codes']):
            code_path = output_path / f"{kernel_name}_v{i+1}.py"
            with open(code_path, 'w') as f:
                f.write(code_info['code'])

        # Save statistics
        stats_path = output_path / f"{kernel_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'best_fitness': latest_result['best_fitness'],
                'speedup': latest_result['speedup'],
                'generations': latest_result['generations'],
                'time_elapsed': latest_result['time_elapsed'],
                'llm_stats': latest_result['llm_stats'],
                'best_candidate': latest_result.get('best_provenance'),
                'top5_summary': [
                    {key: value for key, value in c.items() if key != 'code'}
                    for c in latest_result['top5_codes']
                ]
            }, f, indent=2)

        print(f"[Agent] Statistics saved: {stats_path}")
