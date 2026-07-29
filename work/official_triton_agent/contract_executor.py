"""Cheap interface checks before invoking the organizer-provided executor."""

from __future__ import annotations

import ast
from typing import Dict, Optional

from executor import EvaluationResult, TritonExecutor


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
            "decorators": [_qualified_name(dec) for dec in node.decorator_list],
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
        parameters.append(_parameter(arg, kind, required=index < default_start))
    if args.vararg:
        parameters.append(_parameter(args.vararg, "var_positional", required=False))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(_parameter(arg, "keyword_only", required=default is None))
    if args.kwarg:
        parameters.append(_parameter(args.kwarg, "var_keyword", required=False))

    return {"async": isinstance(node, ast.AsyncFunctionDef), "parameters": parameters}


def _parameter(arg: ast.arg, kind: str, required: bool) -> Dict[str, object]:
    annotation = _qualified_name(arg.annotation) if arg.annotation is not None else ""
    return {
        "name": arg.arg,
        "kind": kind,
        "required": required,
        "tl_constexpr": annotation in {"constexpr", "tl.constexpr", "triton.language.constexpr"},
    }


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _qualified_name(node.func)
    return ""
