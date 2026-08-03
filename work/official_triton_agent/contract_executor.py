"""Cheap interface checks before invoking the organizer-provided executor."""

from __future__ import annotations

import ast
import hashlib
import time
from typing import Dict, Optional, Sequence

from executor import EvaluationResult, TritonExecutor


_PROFILE_FIELDS = {
    "schema_version", "kind", "path_base", "run_directory_id",
    "csv_path", "csv_sha256", "parser_rule", "parse_status",
    "kernel_name", "target_row_index", "execution_time_us",
    "toolchain_fingerprint",
}
_FINGERPRINT_FIELDS = {"facts", "sha256"}
_FACT_FIELDS = {"python_version", "machine", "system", "release", "packages"}
_PACKAGE_FIELDS = {"torch", "torch-npu", "triton"}


def _profile_evidence(result) -> Optional[dict]:
    """Copy only the profiler observation schema, never arbitrary result data."""
    evidence = getattr(result, "evidence", None)
    profile = evidence.get("profile") if isinstance(evidence, dict) else None
    if not isinstance(profile, dict) or set(profile) != _PROFILE_FIELDS:
        return None
    fingerprint = profile.get("toolchain_fingerprint")
    if not isinstance(fingerprint, dict) or set(fingerprint) != _FINGERPRINT_FIELDS:
        return None
    facts = fingerprint.get("facts")
    if not isinstance(facts, dict) or set(facts) != _FACT_FIELDS:
        return None
    packages = facts.get("packages")
    if not isinstance(packages, dict) or set(packages) != _PACKAGE_FIELDS:
        return None
    return {
        key: profile[key]
        for key in _PROFILE_FIELDS - {"toolchain_fingerprint"}
    } | {
        "toolchain_fingerprint": {
            "facts": {
                key: (dict(packages) if key == "packages" else facts[key])
                for key in _FACT_FIELDS
            },
            "sha256": fingerprint["sha256"],
        }
    }


class ContractCheckingExecutor(TritonExecutor):
    """Reject interface drift before an expensive ``msprof`` evaluation."""

    def __init__(self, *args, baseline_code: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_code = baseline_code

    def evaluate(self, code: str, timeout: int = 1200) -> EvaluationResult:
        error = interface_contract_error(self.baseline_code, code)
        if error is not None:
            print(f"[Executor] Static interface preflight failed: {error}")
            return EvaluationResult(
                success=False,
                execution_time=0.0,
                speedup=0.0,
                fitness=0.0,
                error=f"Static interface preflight failed: {error}",
            )
        return super().evaluate(code, timeout=timeout)


class MultiCaseContractExecutor:
    """Evaluate every supplied case and rank by the weakest real result."""

    def __init__(
        self,
        cases: Sequence[tuple[int, ContractCheckingExecutor]],
    ) -> None:
        ordered = tuple(sorted(cases, key=lambda item: item[0]))
        case_ids = [case_id for case_id, _ in ordered]
        if not ordered or any(case_id <= 0 for case_id in case_ids):
            raise ValueError("At least one positive test case id is required")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Test case ids must be unique")
        if case_ids != list(range(1, case_ids[-1] + 1)):
            raise ValueError(f"Test case ids must be contiguous from 1: {case_ids}")
        self.cases = ordered
        self.case_ids = tuple(case_ids)

    def evaluate(self, code: str, timeout: int = 1200) -> EvaluationResult:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        results = []
        case_results = []
        for case_id, executor in self.cases:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                case_results.append(
                    self._case_evidence(case_id, executor, None, "not_run_budget_exhausted")
                )
                return EvaluationResult(
                    False,
                    sum(result.execution_time for result in results),
                    0.0,
                    0.0,
                    f"Test case {case_id} not run: wall-clock budget exhausted",
                    self._evaluation_evidence(case_results),
                )
            result = executor.evaluate(code, timeout=remaining)
            results.append(result)
            case_results.append(
                self._case_evidence(
                    case_id, executor, result, "passed" if result.success else "failed"
                )
            )
            if not result.success:
                return EvaluationResult(
                    False,
                    sum(item.execution_time for item in results),
                    0.0,
                    0.0,
                    f"Test case {case_id} failed: {result.error or 'unknown error'}",
                    self._evaluation_evidence(case_results),
                )

        return EvaluationResult(
            True,
            sum(result.execution_time for result in results),
            min(result.speedup for result in results),
            min(result.fitness for result in results),
            None,
            self._evaluation_evidence(case_results),
        )

    @staticmethod
    def _case_evidence(case_id, executor, result, status: str) -> dict:
        test_path = executor.test_code_path
        try:
            test_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
        except OSError:
            test_sha256 = None
        return {
            "case_id": case_id,
            "status": status,
            "success": result.success if result is not None else False,
            "baseline_time_us": float(executor.baseline_time),
            "execution_time_us": result.execution_time if result is not None else None,
            "speedup": result.speedup if result is not None else None,
            "fitness": result.fitness if result is not None else None,
            "test_file": test_path.name,
            "test_sha256": test_sha256,
            "baseline_code_sha256": hashlib.sha256(
                executor.baseline_code.encode("utf-8")
            ).hexdigest(),
            "profile": _profile_evidence(result) if result is not None else None,
        }

    @staticmethod
    def _evaluation_evidence(case_results) -> dict:
        return {
            "schema_version": 2,
            "kind": "multi-case-real-evaluation",
            "official_aggregate": False,
            "aggregation": {
                "execution_time_us": "sum-completed-case-time-v1",
                "speedup": "minimum-successful-case-speedup-v1",
                "fitness": "minimum-successful-case-fitness-v1",
            },
            "case_results": case_results,
        }


def interface_contract_error(baseline_code: str, candidate_code: str) -> Optional[str]:
    """Return a concise mismatch reason without importing either module."""

    try:
        baseline_tree = ast.parse(baseline_code)
    except SyntaxError as exc:
        return f"baseline syntax error at line {exc.lineno}: {exc.msg}"
    try:
        candidate_tree = ast.parse(candidate_code)
    except SyntaxError as exc:
        return f"candidate syntax error at line {exc.lineno}: {exc.msg}"

    baseline_functions = _function_contracts(baseline_tree)
    candidate_functions = _function_contracts(candidate_tree)
    for name, expected in baseline_functions.items():
        actual = candidate_functions.get(name)
        if actual is None:
            return f"missing baseline function: {name}"
        if actual["signature"] != expected["signature"]:
            return f"signature differs from baseline: {name}"
        if actual["decorators"] != expected["decorators"]:
            return f"decorators differ from baseline: {name}"
    return None


def _function_contracts(tree: ast.Module) -> Dict[str, Dict[str, object]]:
    return {
        node.name: {
            "signature": _signature_contract(node),
            "decorators": [_ast_contract(dec) for dec in node.decorator_list],
        }
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _signature_contract(node: ast.AST) -> Dict[str, object]:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    default_start = len(positional) - len(args.defaults)
    parameters = []

    for index, arg in enumerate(positional):
        kind = "positional_only" if index < len(args.posonlyargs) else "positional_or_keyword"
        default = None if index < default_start else args.defaults[index - default_start]
        parameters.append(
            _parameter(arg, kind, required=index < default_start, default=default)
        )
    if args.vararg:
        parameters.append(
            _parameter(args.vararg, "var_positional", required=False, default=None)
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(
            _parameter(arg, "keyword_only", required=default is None, default=default)
        )
    if args.kwarg:
        parameters.append(
            _parameter(args.kwarg, "var_keyword", required=False, default=None)
        )

    return {
        "async": isinstance(node, ast.AsyncFunctionDef),
        "parameters": parameters,
        "returns": _ast_contract(node.returns),
    }


def _parameter(
    arg: ast.arg,
    kind: str,
    required: bool,
    default: Optional[ast.expr],
) -> Dict[str, object]:
    return {
        "name": arg.arg,
        "kind": kind,
        "required": required,
        "annotation": _ast_contract(arg.annotation),
        "default": _ast_contract(default),
    }


def _ast_contract(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    return ast.dump(node, annotate_fields=True, include_attributes=False)
