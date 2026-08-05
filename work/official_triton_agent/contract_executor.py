"""Cheap interface checks before invoking the organizer-provided executor."""

from __future__ import annotations

import ast
import copy
import hashlib
import time
from collections import Counter
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
_SAFE_WRAPPER_CALLS = {
    "abs", "bool", "enumerate", "float", "int", "len", "list", "max",
    "min", "range", "round", "tuple", "zip", "math.ceil", "math.floor",
    "triton.Config", "triton.cdiv", "triton.next_power_of_2",
}
_SAFE_METADATA_METHODS = {"dim", "numel", "size", "stride", "view"}
_TUNABLE_LAUNCH_KEYWORDS = {"maxnreg", "num_ctas", "num_stages", "num_warps"}


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
        try:
            test_code = self.test_code_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            error = f"test source unavailable: {type(exc).__name__}"
        else:
            error = (
                None
                if code == self.baseline_code
                else interface_contract_error(
                    self.baseline_code,
                    code,
                    test_code,
                    enforce_semantic_change=True,
                )
            )
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


def interface_contract_error(
    baseline_code: str,
    candidate_code: str,
    test_code: Optional[str] = None,
    *,
    enforce_semantic_change: bool = False,
    visible_shape_values: Optional[set[int]] = None,
) -> Optional[str]:
    """Return a concise mismatch reason without importing either module."""

    try:
        baseline_tree = ast.parse(baseline_code)
    except SyntaxError as exc:
        return f"baseline syntax error at line {exc.lineno}: {exc.msg}"
    try:
        candidate_tree = ast.parse(candidate_code)
    except SyntaxError as exc:
        return f"candidate syntax error at line {exc.lineno}: {exc.msg}"

    test_tree = None
    if test_code is not None:
        try:
            test_tree = ast.parse(test_code)
        except SyntaxError as exc:
            return f"test syntax error at line {exc.lineno}: {exc.msg}"

    baseline_functions = _function_contracts(baseline_tree, include_jit=False)
    candidate_functions = _function_contracts(candidate_tree)
    for name, expected in baseline_functions.items():
        actual = candidate_functions.get(name)
        if actual is None:
            return f"missing baseline function: {name}"
        if actual["signature"] != expected["signature"]:
            return f"signature differs from baseline: {name}"
        if actual["decorators"] != expected["decorators"]:
            return f"decorators differ from baseline: {name}"
    if enforce_semantic_change and _normalized_ast(baseline_tree) == _normalized_ast(candidate_tree):
        return "no_semantic_change"
    # Arm the shape-fingerprint gate from the observed test shapes when the
    # caller did not pass an explicit set. An explicit set (including an empty
    # one) is honored as-is so callers can opt out deliberately.
    if visible_shape_values is None and test_tree is not None:
        visible_shape_values = visible_shape_values_from_test(test_tree)
    shape_error = _shape_fingerprint_error(
        baseline_tree,
        candidate_tree,
        visible_shape_values or set(),
    )
    if shape_error is not None:
        return shape_error
    jit_error = _jit_contract_error(baseline_tree, candidate_tree, test_tree)
    if jit_error is not None:
        return jit_error
    return _wrapper_timing_surface_error(baseline_tree, candidate_tree)


def _jit_contract_error(
    baseline_tree: ast.Module,
    candidate_tree: ast.Module,
    test_tree: Optional[ast.Module],
) -> Optional[str]:
    baseline_jit = _jit_nodes(baseline_tree)
    candidate_jit = _jit_nodes(candidate_tree)
    external = set(baseline_jit) - _internal_jit_names(baseline_tree, baseline_jit)
    if test_tree is not None:
        external.update(_test_referenced_jit_names(test_tree, set(baseline_jit)))

    for name, expected_node in baseline_jit.items():
        actual_node = candidate_jit.get(name)
        if actual_node is None:
            return f"missing baseline Triton JIT function: {name}"
        expected_decorators = [_ast_contract(item) for item in expected_node.decorator_list]
        actual_decorators = [_ast_contract(item) for item in actual_node.decorator_list]
        if actual_decorators != expected_decorators:
            return f"Triton JIT decorators differ from baseline: {name}"
        expected = (
            _signature_contract(expected_node)
            if name in external
            else _jit_runtime_contract(expected_node)
        )
        actual = (
            _signature_contract(actual_node)
            if name in external
            else _jit_runtime_contract(actual_node)
        )
        if actual != expected:
            detail = "external signature" if name in external else "runtime signature"
            return f"Triton JIT {detail} differs from baseline: {name}"
    return None


def _internal_jit_names(tree: ast.Module, jit_nodes) -> set[str]:
    internal = set()
    for node, _ in _walk_wrapper(tree):
        if isinstance(node, ast.Call):
            target = _launch_target(node.func, jit_nodes)
            if target is not None:
                internal.add(target)
    pending = list(internal)
    while pending:
        current = pending.pop()
        for call in (item for item in ast.walk(jit_nodes[current]) if isinstance(item, ast.Call)):
            helper = _dotted_name(call.func)
            if helper in jit_nodes and helper not in internal:
                internal.add(helper)
                pending.append(helper)
    return internal


def _test_referenced_jit_names(tree: ast.Module, jit_names: set[str]) -> set[str]:
    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                return set(jit_names)
            referenced.update(alias.name for alias in node.names if alias.name in jit_names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in jit_names:
                referenced.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in jit_names:
            referenced.add(node.attr)
    return referenced


def _jit_runtime_contract(node: ast.AST) -> Dict[str, object]:
    signature = _signature_contract(node)
    constexpr = _constexpr_parameters(node)
    runtime = []
    for position, parameter in enumerate(signature["parameters"]):
        if parameter["name"] in constexpr:
            continue
        runtime.append((
            position,
            {key: value for key, value in parameter.items() if key != "name"},
        ))
    return {
        "async": signature["async"],
        "runtime_parameters": runtime,
        "returns": signature["returns"],
    }


def _wrapper_timing_surface_error(
    baseline_tree: ast.Module,
    candidate_tree: ast.Module,
) -> Optional[str]:
    """Reject work moved outside the target-kernel profiler boundary."""

    baseline_imports, baseline_calls, expected_launches = _wrapper_surface(baseline_tree)
    candidate_imports, candidate_calls, actual_launches = _wrapper_surface(candidate_tree)
    for kind, expected, actual in (
        ("import", baseline_imports, candidate_imports),
        ("call", baseline_calls, candidate_calls),
    ):
        error = _counter_contract_error(kind, expected, actual)
        if error is not None:
            return error
    if len(expected_launches) != len(actual_launches):
        return (
            "wrapper timing surface changes Triton launch count: "
            f"expected {len(expected_launches)}, got {len(actual_launches)}"
        )
    for expected, actual in zip(expected_launches, actual_launches):
        for index, detail in enumerate(("target", "control", "bindings")):
            if actual[index] != expected[index]:
                return f"wrapper timing surface changes launch {detail}: {expected[0]}"
    return None


def _counter_contract_error(kind: str, expected, actual) -> Optional[str]:
    expected_counts = Counter(contract for _, contract in expected)
    actual_counts = Counter(contract for _, contract in actual)
    added = actual_counts - expected_counts
    removed = expected_counts - actual_counts
    if added:
        contract = next(iter(added))
        label = next(label for label, item in actual if item == contract)
        removed_labels = {label for label, item in expected if item in removed}
        action = "changes" if label in removed_labels else "adds"
        return f"wrapper timing surface {action} {kind}: {label}"
    if removed:
        contract = next(iter(removed))
        label = next(label for label, item in expected if item == contract)
        return f"wrapper timing surface removes {kind}: {label}"
    return None


def _wrapper_surface(tree: ast.Module) -> tuple[list, list, list]:
    """Collect executable host-side events while skipping Triton kernel bodies."""

    jit_nodes = _jit_nodes(tree)
    imports, calls, launches = [], [], []
    for node, control in _walk_wrapper(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            label = node.module or "." if isinstance(node, ast.ImportFrom) else ",".join(
                alias.asname or alias.name for alias in node.names
            )
            imports.append((label, _ast_contract(node)))
            continue
        if not isinstance(node, ast.Call):
            continue
        target = _launch_target(node.func, jit_nodes)
        if target is None:
            callee = _dotted_name(node.func) or "<dynamic>"
            if not _safe_wrapper_call(callee):
                calls.append((callee, _ast_contract(node)))
            continue
        runtime = _runtime_launch_contract(node, jit_nodes[target])
        launches.append((target, control, runtime))
    return imports, calls, launches


def _walk_wrapper(node: ast.AST, control=()):
    """Yield wrapper nodes with the control path that contains each node."""

    yield node, control
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if _jit_decorated(node):
            return
        children = (("body", statement) for statement in node.body)
    else:
        children = (
            (field, child)
            for field, value in ast.iter_fields(node)
            for child in (value if isinstance(value, list) else [value])
            if isinstance(child, ast.AST)
        )
    for field, child in children:
        child_control = control
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While)) and field in {"body", "orelse"}:
            child_control += (_control_contract(node, field),)
        yield from _walk_wrapper(child, child_control)


def _control_contract(node: ast.AST, branch: str) -> str:
    if isinstance(node, (ast.For, ast.AsyncFor)):
        header = f"{_ast_contract(node.target)}:{_ast_contract(node.iter)}"
    else:
        header = _ast_contract(node.test)
    return f"{type(node).__name__}:{header}:{branch}"


def _runtime_launch_contract(call: ast.Call, function: ast.AST) -> tuple:
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    positional = [*function.args.posonlyargs, *function.args.args]
    constexpr = _constexpr_parameters(function)
    runtime_names = [
        argument.arg
        for argument in (*positional, *function.args.kwonlyargs)
        if argument.arg not in constexpr
    ]
    slots = {name: position for position, name in enumerate(runtime_names)}
    bindings = []
    for position, argument in enumerate(call.args):
        if position >= len(positional):
            key = ("extra-positional", position)
        elif positional[position].arg in constexpr:
            continue
        else:
            key = ("slot", slots[positional[position].arg])
        bindings.append((key, _ast_contract(argument)))
    for keyword in call.keywords:
        name = keyword.arg or "**"
        if name in _TUNABLE_LAUNCH_KEYWORDS or name in constexpr:
            continue
        key = ("slot", slots[name]) if name in slots else ("unknown", name)
        bindings.append((key, _ast_contract(keyword.value)))
    return tuple(sorted(bindings, key=lambda item: repr(item[0])))


def _jit_nodes(tree: ast.Module) -> Dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _jit_decorated(node)
    }


def _constexpr_parameters(node: ast.AST) -> set[str]:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return {
        argument.arg
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if (_dotted_name(argument.annotation) or "").endswith(".constexpr")
    }


def _jit_decorated(node: ast.AST) -> bool:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    decorators = (
        item.func if isinstance(item, ast.Call) else item
        for item in node.decorator_list
    )
    return any(_dotted_name(item) in {"jit", "triton.jit"} for item in decorators)


def _launch_target(node: ast.AST, jit_functions) -> Optional[str]:
    if not isinstance(node, ast.Subscript):
        return None
    target = _dotted_name(node.value)
    return target if target in jit_functions else None


def _safe_wrapper_call(callee: str) -> bool:
    return (
        callee in _SAFE_WRAPPER_CALLS
        or callee.rsplit(".", 1)[-1] in _SAFE_METADATA_METHODS
    )


def _dotted_name(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _dotted_name(node.value)
        return f"{owner}.{node.attr}" if owner else f"<expr>.{node.attr}"
    return None


def _function_contracts(
    tree: ast.Module,
    *,
    include_jit: bool = True,
) -> Dict[str, Dict[str, object]]:
    return {
        node.name: {
            "signature": _signature_contract(node),
            "decorators": [_ast_contract(dec) for dec in node.decorator_list],
        }
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (include_jit or not _jit_decorated(node))
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


class _SemanticNormalizer(ast.NodeTransformer):
    """Remove presentation-only differences while retaining executable values."""

    def __init__(self) -> None:
        self._names: Dict[str, str] = {}

    def visit_Name(self, node: ast.Name):  # noqa: N802 - AST visitor API
        replacement = self._names.setdefault(node.id, f"local_{len(self._names)}")
        node.id = replacement
        return node

    def visit_arg(self, node: ast.arg):  # noqa: N802 - AST visitor API
        replacement = self._names.setdefault(node.arg, f"local_{len(self._names)}")
        node.arg = replacement
        return self.generic_visit(node)

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and _is_docstring(body[0]):
                node.body = body[1:]
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            node.value = "<string>"
        return node


def _is_docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _normalized_ast(tree: ast.AST) -> str:
    normalized = _SemanticNormalizer().visit(copy.deepcopy(tree))
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def _shape_literal_compare(node: ast.AST, values: set[int]) -> bool:
    if not isinstance(node, ast.Compare) or not values:
        return False
    if not any(isinstance(op, ast.Eq) for op in node.ops):
        return False
    operands = [node.left, *node.comparators]
    return any(
        isinstance(item, ast.Constant)
        and isinstance(item.value, int)
        and not isinstance(item.value, bool)
        and item.value in values
        for item in operands
    )


# Tensor constructors and shape methods whose integer arguments name concrete
# dimensions a candidate could hard-code against. The first group takes size as
# bare varargs (and optionally a tuple/list); the second passes size only as a
# tuple/list alongside non-shape scalars (fill value, RNG bounds), so for those
# only tuple/list elements are treated as dimensions.
_VARARG_SHAPE_CALLS = frozenset({
    "randn", "rand", "zeros", "ones", "empty",
    "new_zeros", "new_ones", "new_empty",
    "reshape", "view", "expand", "repeat", "broadcast_to",
})
_TUPLE_ONLY_SHAPE_CALLS = frozenset({"full", "randint"})


def _int_constant(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def _tuple_ints(node: ast.AST) -> list[int]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return []
    return [value for value in (_int_constant(item) for item in node.elts) if value is not None]


def visible_shape_values_from_test(test_tree: ast.AST) -> set[int]:
    """Collect the concrete tensor dimensions a test exercises.

    These are the observed shapes a candidate could fingerprint. Only integer
    arguments to tensor size constructors and shape methods are collected; 0 and
    1 are dropped as too common to signal a benchmark fingerprint. The set only
    arms the shape gate, which rejects a *new* equality comparison against one of
    these values, so over-collection at worst rejects an extra candidate rather
    than letting a fingerprint through.
    """
    values: set[int] = set()
    for node in ast.walk(test_tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        leaf = name.rsplit(".", 1)[-1] if name else None
        if leaf in _VARARG_SHAPE_CALLS:
            for argument in node.args:
                direct = _int_constant(argument)
                if direct is not None:
                    values.add(direct)
                else:
                    values.update(_tuple_ints(argument))
        elif leaf in _TUPLE_ONLY_SHAPE_CALLS:
            for argument in node.args:
                values.update(_tuple_ints(argument))
    return {value for value in values if value >= 2}


def _shape_fingerprint_error(
    baseline_tree: ast.AST,
    candidate_tree: ast.AST,
    shape_values: set[int],
) -> Optional[str]:
    baseline_comparisons = {
        _ast_contract(node)
        for node in ast.walk(baseline_tree)
        if _shape_literal_compare(node, shape_values)
    }
    for node in ast.walk(candidate_tree):
        if _shape_literal_compare(node, shape_values) and _ast_contract(node) not in baseline_comparisons:
            return "shape_fingerprint"
    return None


def holdout_gate_error(evidence: object) -> Optional[str]:
    """Require one independent correctness result without accepting prompt leakage."""

    if not isinstance(evidence, dict):
        return "holdout_required"
    if evidence.get("status") != "passed":
        return "holdout_required"
    if evidence.get("split") != "holdout":
        return "holdout_required"
    signature = evidence.get("case_signature")
    if not isinstance(signature, str) or not signature.strip():
        return "holdout_required"
    count = evidence.get("case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return "holdout_required"
    if evidence.get("used_for_search") is not False:
        return "holdout_search_contamination"
    if evidence.get("used_in_prompt") is True:
        return "holdout_prompt_contamination"
    return None
