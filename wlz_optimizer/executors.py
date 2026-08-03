"""Executor interfaces and local static/proxy executor."""

from __future__ import annotations

import ast
import importlib.util
import platform
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.schemas import Candidate, EvalContext, EvaluationResult


class Executor(Protocol):
    kind: str

    def evaluate(self, candidate: Candidate, context: EvalContext) -> EvaluationResult:
        ...

    def env_fingerprint(self) -> str:
        ...


class LocalExecutor:
    """Local AST/static validator with a deterministic proxy score.

    This executor does not claim Ascend compile, correctness, latency, or
    speedup. It only returns static checks plus proxy_score for local sorting.
    """

    kind = "local_static_proxy"

    def env_fingerprint(self) -> str:
        triton_available = importlib.util.find_spec("triton") is not None
        raw = "|".join(
            [
                self.kind,
                sys.version.split()[0],
                platform.system(),
                platform.machine(),
                f"triton_available={triton_available}",
                "schema=v4",
                "import_rules=v3",
                "signature_rules=v1",
                "launch_contract_rules=v1",
                "triton_semantic_rules=v2",
                "target_device_rules=v1",
                "unused_triton_local_rules=v1",
            ]
        )
        return sha256_text(raw)[:16]

    def evaluate(self, candidate: Candidate, context: EvalContext) -> EvaluationResult:
        expected_function_contracts = _load_expected_function_contracts(
            context.baseline_file,
            context.required_functions,
        )
        baseline_code = _read_baseline_code(context.baseline_file)
        checks = validate_static_structure(
            candidate.code,
            context.required_functions,
            expected_function_contracts,
            baseline_code=baseline_code,
        )
        passed = (
            checks["syntax_ok"]
            and checks["imports_ok"]
            and checks["signature_ok"]
            and checks["launch_contract_ok"]
            and checks["triton_semantics_ok"]
            and checks["target_device_ok"]
            and checks["unused_triton_locals_ok"]
        )

        error_type: Optional[str] = None
        error_message: Optional[str] = None
        if not checks["syntax_ok"]:
            error_type = "syntax_fail"
            error_message = checks.get("syntax_error") or "Python syntax check failed"
        elif not checks["imports_ok"]:
            error_type = "import_fail"
            error_message = "Missing required Triton imports"
        elif not checks["target_device_ok"]:
            error_type = "target_device_fail"
            errors = checks.get("target_device_errors", [])
            error_message = "; ".join(
                f"{item['code']} at line {item['line']}"
                for item in errors
                if isinstance(item, dict)
            ) or "Candidate hard-codes a non-Ascend device context"
        elif not checks["signature_ok"]:
            error_type = "signature_fail"
            mismatched = ", ".join(checks.get("signature_mismatches", {}))
            missing = ", ".join(checks.get("missing_functions", []))
            if mismatched:
                error_message = f"Function signature differs from baseline: {mismatched}"
            elif missing:
                error_message = f"Missing expected function(s): {missing}"
            else:
                error_message = "No function definition found"
        elif not checks["launch_contract_ok"]:
            error_type = "launch_contract_fail"
            mismatched = ", ".join(checks.get("decorator_mismatches", {}))
            error_message = f"Function decorators differ from baseline: {mismatched}"
        elif not checks["triton_semantics_ok"]:
            error_type = "triton_semantic_fail"
            errors = checks.get("triton_semantic_errors", [])
            error_message = "; ".join(
                f"{item['code']} at {item['function']}:{item['line']}"
                for item in errors
                if isinstance(item, dict)
            ) or "Known-invalid Triton compile-time structure"
        elif not checks["unused_triton_locals_ok"]:
            error_type = "new_unused_triton_local"
            errors = checks.get("new_unused_triton_locals", [])
            error_message = "; ".join(
                f"{item['name']} at {item['function']}:{item['line']}"
                for item in errors
                if isinstance(item, dict)
            ) or "Candidate adds an unread Triton kernel local"

        proxy_score = compute_proxy_score(candidate, checks) if passed else 0.0
        metadata = dict(checks)
        metadata.update(
            {
                "proxy_score_note": "Static local sorting signal only; not Ascend speedup.",
                "failure_taxonomy": [
                    "syntax_fail",
                    "import_fail",
                    "target_device_fail",
                    "signature_fail",
                    "launch_contract_fail",
                    "triton_semantic_fail",
                    "new_unused_triton_local",
                    "compile_fail",
                    "correctness_fail",
                    "timeout",
                    "runtime_error",
                ],
            }
        )

        return EvaluationResult(
            candidate_id=candidate.id,
            executor=self.kind,
            status="local_static_pass" if passed else "local_static_fail",
            passed=passed,
            correctness_ok=None,
            compile_ok=None,
            latency_ms=None,
            baseline_ms=None,
            speedup=None,
            proxy_score=proxy_score,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata,
        )


class MockExecutor(LocalExecutor):
    """Interface-compatible mock executor using the same static/proxy checks."""

    kind = "mock_static_proxy"


def validate_static_structure(
    code: str,
    required_functions: List[str],
    expected_function_contracts: Optional[Dict[str, object]] = None,
    *,
    baseline_code: Optional[str] = None,
) -> Dict[str, object]:
    checks: Dict[str, object] = {
        "syntax_ok": False,
        "imports_ok": False,
        "signature_ok": False,
        "launch_contract_ok": False,
        "triton_semantics_ok": False,
        "target_device_ok": False,
        "has_triton_jit": False,
        "has_function_def": False,
        "required_functions": list(required_functions),
        "missing_functions": [],
        "defined_functions": [],
        "function_signatures": {},
        "signature_mismatches": {},
        "function_decorators": {},
        "decorator_mismatches": {},
        "triton_semantic_errors": [],
        "target_device_errors": [],
        "unused_triton_locals_ok": True,
        "new_unused_triton_locals": [],
    }

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        checks["syntax_error"] = f"SyntaxError: {exc.msg} at line {exc.lineno}, column {exc.offset}"
        return checks

    checks["syntax_ok"] = True
    imports = _collect_imports(tree)
    checks["imports"] = sorted(imports)
    checks["imports_ok"] = _has_triton_imports(imports)

    function_nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    functions = list(function_nodes)
    checks["defined_functions"] = functions
    checks["has_function_def"] = bool(functions)
    checks["function_signatures"] = {
        name: _signature_contract(node)
        for name, node in function_nodes.items()
        if not required_functions or name in required_functions
    }
    checks["function_decorators"] = {
        name: [_decorator_name(dec) for dec in node.decorator_list]
        for name, node in function_nodes.items()
        if not required_functions or name in required_functions
    }
    aliases = _import_aliases(tree)
    checks["has_triton_jit"] = any(
        _is_triton_jit(_canonical_name(dec, aliases))
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for dec in node.decorator_list
    )
    semantic_errors = _triton_semantic_errors(tree, aliases)
    checks["triton_semantic_errors"] = semantic_errors
    checks["triton_semantics_ok"] = not semantic_errors
    target_device_errors = _target_device_errors(tree)
    checks["target_device_errors"] = target_device_errors
    checks["target_device_ok"] = not target_device_errors
    new_unused_locals = _new_unused_triton_locals(tree, aliases, baseline_code)
    checks["new_unused_triton_locals"] = new_unused_locals
    checks["unused_triton_locals_ok"] = not new_unused_locals

    if required_functions:
        missing = [name for name in required_functions if name not in functions]
        checks["missing_functions"] = missing
        mismatches = {}
        decorator_mismatches = {}
        if expected_function_contracts:
            actual_signatures = checks["function_signatures"]
            actual_decorators = checks["function_decorators"]
            for name in required_functions:
                expected = expected_function_contracts.get(name)
                if not isinstance(expected, dict):
                    continue
                expected_signature = expected.get("signature")
                actual_signature = actual_signatures.get(name)
                if (
                    expected_signature is not None
                    and actual_signature is not None
                    and actual_signature != expected_signature
                ):
                    mismatches[name] = {
                        "expected": expected_signature,
                        "actual": actual_signature,
                    }
                expected_decorators = expected.get("decorators")
                actual_function_decorators = actual_decorators.get(name)
                if (
                    expected_decorators is not None
                    and actual_function_decorators is not None
                    and actual_function_decorators != expected_decorators
                ):
                    decorator_mismatches[name] = {
                        "expected": expected_decorators,
                        "actual": actual_function_decorators,
                    }
        checks["signature_mismatches"] = mismatches
        checks["decorator_mismatches"] = decorator_mismatches
        checks["signature_ok"] = not missing and not mismatches
        checks["launch_contract_ok"] = not decorator_mismatches
    else:
        checks["signature_ok"] = bool(functions)
        checks["launch_contract_ok"] = True

    checks["feature_counts"] = _feature_counts(code)
    return checks


def _target_device_errors(tree: ast.Module) -> List[Dict[str, object]]:
    errors: List[Dict[str, object]] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if _decorator_name(call.func) == "torch.cuda.device":
            errors.append(
                {
                    "code": "hardcoded_cuda_device_context",
                    "line": getattr(call, "lineno", 0),
                    "expression": ast.unparse(call),
                }
            )
    return errors


def _triton_semantic_errors(
    tree: ast.Module, aliases: Dict[str, str]
) -> List[Dict[str, object]]:
    """Detect a narrow set of runtime-dependent compile-time constructs."""

    errors: List[Dict[str, object]] = []
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = {_canonical_name(item, aliases) for item in function.decorator_list}
        if not any(_is_triton_jit(name) for name in decorators):
            continue

        arguments = (
            list(function.args.posonlyargs)
            + list(function.args.args)
            + list(function.args.kwonlyargs)
        )
        runtime_names = {
            argument.arg
            for argument in arguments
            if _canonical_name(argument.annotation, aliases) not in {
                "constexpr",
                "tl.constexpr",
                "triton.language.constexpr",
            }
        }
        if function.args.vararg:
            runtime_names.add(function.args.vararg.arg)
        if function.args.kwarg:
            runtime_names.add(function.args.kwarg.arg)

        assignments = [
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        ]
        changed = True
        while changed:
            changed = False
            for assignment in assignments:
                value = assignment.value
                targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
                if value is None or not _depends_on_runtime(value, runtime_names, aliases):
                    continue
                for target in targets:
                    for name in _assigned_names(target):
                        if name not in runtime_names:
                            runtime_names.add(name)
                            changed = True

        for call in ast.walk(function):
            if not isinstance(call, ast.Call):
                continue
            qualified = _canonical_name(call.func, aliases)
            if qualified not in {"tl.arange", "triton.language.arange"}:
                continue
            runtime_symbols = sorted(
                {name.id for arg in call.args for name in ast.walk(arg) if isinstance(name, ast.Name)}
                & runtime_names
            )
            if not runtime_symbols:
                continue
            errors.append(
                {
                    "code": "dynamic_tl_arange",
                    "function": function.name,
                    "line": getattr(call, "lineno", 0),
                    "expression": ast.unparse(call),
                    "runtime_symbols": runtime_symbols,
                }
            )
    return errors


def _depends_on_runtime(node: ast.AST, runtime_names: set[str], aliases: Dict[str, str]) -> bool:
    if any(isinstance(item, ast.Name) and item.id in runtime_names for item in ast.walk(node)):
        return True
    runtime_calls = {
        "tl.load",
        "triton.language.load",
        "tl.program_id",
        "triton.language.program_id",
        "tl.num_programs",
        "triton.language.num_programs",
    }
    return any(
        isinstance(item, ast.Call) and _canonical_name(item.func, aliases) in runtime_calls
        for item in ast.walk(node)
    )


def _assigned_names(node: ast.AST) -> List[str]:
    return [
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    ]


def _read_baseline_code(baseline_file: Optional[Path]) -> Optional[str]:
    if baseline_file is None or not baseline_file.is_file():
        return None
    try:
        return baseline_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _new_unused_triton_locals(
    candidate_tree: ast.Module,
    candidate_aliases: Dict[str, str],
    baseline_code: Optional[str],
) -> List[Dict[str, object]]:
    """Return candidate-only unread assignments in top-level Triton kernels.

    This is deliberately a baseline-relative check: pre-existing unread locals
    are tolerated, while a newly introduced unread assignment is rejected.
    When the baseline is unavailable or unparsable, the check is disabled.
    """

    if baseline_code is None:
        return []
    try:
        baseline_tree = ast.parse(baseline_code)
    except SyntaxError:
        return []

    baseline_aliases = _import_aliases(baseline_tree)
    baseline_unused = _unused_triton_assignment_keys(baseline_tree, baseline_aliases)
    candidate_unused = _unused_triton_assignment_records(candidate_tree, candidate_aliases)
    remaining = dict(baseline_unused)
    new_locals: List[Dict[str, object]] = []
    for record in candidate_unused:
        key = record["key"]
        count = remaining.get(key, 0)
        if count:
            remaining[key] = count - 1
        else:
            new_locals.append({name: value for name, value in record.items() if name != "key"})
    return new_locals


def _unused_triton_assignment_keys(tree: ast.Module, aliases: Dict[str, str]) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    for record in _unused_triton_assignment_records(tree, aliases):
        key = record["key"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def _unused_triton_assignment_records(
    tree: ast.Module, aliases: Dict[str, str]
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_is_triton_jit(_canonical_name(dec, aliases)) for dec in function.decorator_list):
            continue

        loaded_names = {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for assignment in ast.walk(function):
            if isinstance(assignment, ast.AugAssign):
                loaded_names.update(_assigned_names(assignment.target))
                continue
            if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                continue
            if isinstance(assignment, ast.AnnAssign) and assignment.value is None:
                continue
            targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            target_names = [name for target in targets for name in _assigned_names(target)]
            for name in target_names:
                if name in loaded_names:
                    continue
                key = (
                    function.name,
                    name,
                    ast.dump(assignment, annotate_fields=True, include_attributes=False),
                )
                records.append(
                    {
                        "code": "new_unused_triton_local",
                        "function": function.name,
                        "name": name,
                        "line": getattr(assignment, "lineno", 0),
                        "expression": ast.unparse(assignment),
                        "key": key,
                    }
                )
    return records


def _load_expected_function_contracts(
    baseline_file: Optional[Path],
    required_functions: List[str],
) -> Optional[Dict[str, object]]:
    if baseline_file is None or not baseline_file.is_file():
        return None
    try:
        tree = ast.parse(baseline_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None
    return {
        node.name: {
            "signature": _signature_contract(node),
            "decorators": [_decorator_name(dec) for dec in node.decorator_list],
        }
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (not required_functions or node.name in required_functions)
    }


def _signature_contract(node: ast.AST) -> Dict[str, object]:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    default_start = len(positional) - len(args.defaults)

    parameters = []
    for index, arg in enumerate(positional):
        parameters.append(
            _parameter_contract(
                arg,
                "positional_only" if index < len(args.posonlyargs) else "positional_or_keyword",
                required=index < default_start,
            )
        )
    if args.vararg:
        parameters.append(_parameter_contract(args.vararg, "var_positional", required=False))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(_parameter_contract(arg, "keyword_only", required=default is None))
    if args.kwarg:
        parameters.append(_parameter_contract(args.kwarg, "var_keyword", required=False))

    return {"async": isinstance(node, ast.AsyncFunctionDef), "parameters": parameters}


def _parameter_contract(arg: ast.arg, kind: str, required: bool) -> Dict[str, object]:
    annotation = _decorator_name(arg.annotation) if arg.annotation is not None else ""
    return {
        "name": arg.arg,
        "kind": kind,
        "required": required,
        "tl_constexpr": annotation in {"constexpr", "tl.constexpr", "triton.language.constexpr"},
    }


def _collect_imports(tree: ast.AST) -> set:
    """Collect imported module paths (``from a import b`` also adds ``a.b``)."""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
                for alias in node.names:
                    imports.add(f"{node.module}.{alias.name}")
    return imports


# Modules known to re-export ``triton`` and ``tl`` under their official names.
_TRITON_REEXPORT_MODULES = {"vllm.triton_utils"}


def _has_triton_imports(imports: set) -> bool:
    has_triton = any(name == "triton" or name.startswith("triton.") for name in imports)
    has_tl = any(
        name == "triton.language" or name.startswith("triton.language.") for name in imports
    )
    for module in _TRITON_REEXPORT_MODULES:
        has_triton = has_triton or f"{module}.triton" in imports
        has_tl = has_tl or f"{module}.tl" in imports
    return has_triton and has_tl


def _import_aliases(tree: ast.AST) -> Dict[str, str]:
    """Map imported names to canonical dotted paths (``tr`` -> ``triton``)."""
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                aliases[bound] = alias.name if alias.asname else bound
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            for alias in node.names:
                target = f"{node.module}.{alias.name}"
                if node.module in _TRITON_REEXPORT_MODULES and alias.name == "tl":
                    target = "triton.language"
                elif node.module in _TRITON_REEXPORT_MODULES and alias.name == "triton":
                    target = "triton"
                aliases[alias.asname or alias.name] = target
    return aliases


def _canonical_name(node: Optional[ast.AST], aliases: Dict[str, str]) -> str:
    """Resolve a dotted expression through the module's import aliases."""
    name = _decorator_name(node) if node is not None else ""
    head, _, rest = name.partition(".")
    mapped = aliases.get(head)
    if mapped is None:
        return name
    return f"{mapped}.{rest}" if rest else mapped


def _is_triton_jit(name: str) -> bool:
    return name == "triton.jit" or name.endswith(".triton.jit")


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _decorator_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _feature_counts(code: str) -> Dict[str, int]:
    return {
        "tl_load": len(re.findall(r"\btl\.load\b", code)),
        "tl_store": len(re.findall(r"\btl\.store\b", code)),
        "tl_arange": len(re.findall(r"\btl\.arange\b", code)),
        "tl_constexpr": len(re.findall(r"\btl\.constexpr\b", code)),
        "block_size_mentions": len(re.findall(r"\bBLOCK[_A-Z0-9]*\b", code)),
        "num_warps_mentions": len(re.findall(r"\bnum_warps\b", code)),
        "mask_mentions": len(re.findall(r"\bmask\b", code)),
        "mutation_markers": len(re.findall(r"wlz-mutation:", code)),
    }


def compute_proxy_score(candidate: Candidate, checks: Dict[str, object]) -> float:
    features = checks.get("feature_counts") or {}
    if not isinstance(features, dict):
        features = {}

    score = 0.08
    if checks.get("has_triton_jit"):
        score += 0.03
    score += min(int(features.get("tl_constexpr", 0)) * 0.006, 0.03)
    score += min(int(features.get("tl_arange", 0)) * 0.006, 0.024)
    score += min(int(features.get("tl_load", 0)) * 0.004, 0.04)
    score += min(int(features.get("tl_store", 0)) * 0.004, 0.024)
    score += min(int(features.get("block_size_mentions", 0)) * 0.003, 0.03)
    score += min(int(features.get("mask_mentions", 0)) * 0.002, 0.02)

    mutation_bonus = {
        "baseline": 0.0,
        "seed_variant": 0.006,
        "block_size_hint": 0.012,
        "num_warps_hint": 0.01,
        "masking_hint": 0.011,
        "constexpr_hint": 0.009,
        "elite_mutation": 0.014,
    }.get(candidate.mutation_kind, 0.005)
    score += mutation_bonus

    jitter = int(candidate.code_hash[:8], 16) / 0xFFFFFFFF * 0.025
    return round(min(score + jitter, 1.0), 6)
