"""Read-only dataset audit utilities for official-style operator folders."""

from __future__ import annotations

import ast
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from wlz_optimizer.executors import validate_static_structure
from wlz_optimizer.io_utils import discover_operators, find_test_files, load_operator_input
from wlz_optimizer.schemas import ShapeContract, ShapeObservation
from wlz_optimizer.shape_validation import validate_shape_consistency


def audit_dataset(
    input_dir: Path,
    kernel: Optional[str] = None,
    shape_contract: Optional[ShapeContract] = None,
) -> Dict[str, Any]:
    """Inspect an input dataset without generating or evaluating candidates."""

    operators = discover_operators(input_dir, kernel)
    if shape_contract and shape_contract.op_name not in operators:
        raise ValueError(
            f"shape contract operator {shape_contract.op_name!r} is not in the selected operators"
        )
    operator_reports = []

    for op_name in operators:
        operator_input = load_operator_input(input_dir, op_name)
        seed_reports = []
        for seed in operator_input.seeds:
            code = seed["code"]
            checks = validate_static_structure(code, operator_input.required_functions)
            seed_reports.append(
                {
                    "path": str(Path(seed["path"]).relative_to(input_dir)),
                    "kind": seed["kind"],
                    "syntax_ok": checks["syntax_ok"],
                    "imports_ok": checks["imports_ok"],
                    "signature_ok": checks["signature_ok"],
                    "has_triton_jit": checks["has_triton_jit"],
                    "defined_functions": checks.get("defined_functions", []),
                    "missing_functions": checks.get("missing_functions", []),
                    "imports": checks.get("imports", []),
                    "error": checks.get("syntax_error"),
                    "feature_counts": checks.get("feature_counts", {}),
                }
            )

        py_files = sorted(operator_input.op_dir.glob("*.py"))
        file_reports = [_audit_python_file(path, input_dir) for path in py_files]
        test_files = find_test_files(operator_input.op_dir, op_name)
        all_test_hints = [_extract_test_hints(path, input_dir) for path in test_files]
        test_hints = all_test_hints[0] if all_test_hints else _extract_test_hints(None, input_dir)
        shape_observations = []
        for path in test_files:
            shape_observations.extend(_extract_shape_observations(op_name, path, input_dir))

        operator_report = {
            "op_name": op_name,
            "op_dir": str(operator_input.op_dir.relative_to(input_dir)),
            "baseline_file": str(operator_input.baseline_file.relative_to(input_dir)),
            "test_file": str(operator_input.test_file.relative_to(input_dir)) if operator_input.test_file else None,
            "test_files": [str(path.relative_to(input_dir)) for path in test_files],
            "required_functions": operator_input.required_functions,
            "seed_count": len(operator_input.seeds),
            "python_file_count": len(py_files),
            "test_hints": test_hints,
            "all_test_hints": all_test_hints,
            "shape_observations": shape_observations,
            "seed_reports": seed_reports,
            "file_reports": file_reports,
        }
        if shape_contract and shape_contract.op_name == op_name:
            operator_report["shape_consistency"] = _shape_consistency_report(
                shape_contract,
                shape_observations,
            )
        operator_reports.append(operator_report)

    return _build_dataset_report(input_dir, operator_reports)


def write_audit_outputs(report: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    """Write JSON and Markdown audit outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_audit.json"
    md_path = output_dir / "dataset_audit.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")

    return {"json": json_path, "markdown": md_path}


def render_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 数据集静态审计报告",
        "",
        "这个报告只读取输入目录，不生成候选、不调用 executor、不写 mock 输出。",
        "",
        "## 总览",
        "",
        f"- 输入目录: `{report['input_dir']}`",
        f"- 算子数量: {summary['operator_count']}",
        f"- Python 文件数量: {summary['python_file_count']}",
        f"- Seed 文件数量: {summary['seed_count']}",
        f"- Seed 静态通过数量: {summary['seed_static_pass_count']}",
        f"- Seed 静态失败数量: {summary['seed_static_fail_count']}",
        f"- 语法失败数量: {summary['syntax_fail_count']}",
        f"- 导入失败数量: {summary['import_fail_count']}",
        f"- 签名失败数量: {summary['signature_fail_count']}",
        "",
        "## 算子明细",
        "",
        "| 算子 | Seeds | Required Functions | Seed Pass/Fail | Test File |",
        "| --- | ---: | --- | --- | --- |",
    ]

    for op in report["operators"]:
        seed_pass = sum(
            1
            for item in op["seed_reports"]
            if item["syntax_ok"] and item["imports_ok"] and item["signature_ok"]
        )
        seed_fail = op["seed_count"] - seed_pass
        funcs = ", ".join(op["required_functions"]) or "<any>"
        test_file = op["test_file"] or "<missing>"
        lines.append(
            f"| `{op['op_name']}` | {op['seed_count']} | `{funcs}` | {seed_pass}/{seed_fail} | `{test_file}` |"
        )

    lines.extend(["", "## Seed 问题清单", ""])
    has_issue = False
    for op in report["operators"]:
        for seed in op["seed_reports"]:
            issues = []
            if not seed["syntax_ok"]:
                issues.append("syntax_fail")
            if not seed["imports_ok"]:
                issues.append("import_fail")
            if not seed["signature_ok"]:
                issues.append("signature_fail")
            if not issues:
                continue
            has_issue = True
            lines.append(f"- `{seed['path']}`: {', '.join(issues)}")
            if seed.get("missing_functions"):
                lines.append(f"  - missing functions: `{', '.join(seed['missing_functions'])}`")
            if seed.get("error"):
                lines.append(f"  - error: `{seed['error']}`")

    if not has_issue:
        lines.append("- 未发现 seed 静态问题。")

    lines.extend(["", "## 导入风格统计", ""])
    for style, count in sorted(report["summary"]["import_style_counts"].items()):
        lines.append(f"- {style}: {count}")

    lines.extend(["", "## 测试输入静态提示", ""])
    for op in report["operators"]:
        hints = op.get("test_hints") or {}
        tensors = hints.get("tensor_creations") or []
        if not tensors:
            lines.append(f"- `{op['op_name']}`: 未提取到 tensor 创建调用。")
            continue
        lines.append(f"- `{op['op_name']}`:")
        for tensor in tensors[:5]:
            name = tensor.get("assigned_to") or "<unassigned>"
            shape = tensor.get("shape")
            dtype = tensor.get("dtype")
            device = tensor.get("device")
            lines.append(
                f"  - `{name}` via `{tensor['call']}` shape=`{shape}` dtype=`{dtype}` device=`{device}`"
            )

    lines.extend(["", "## Shape 契约一致性", ""])
    consistency_present = False
    for op in report["operators"]:
        consistency = op.get("shape_consistency")
        if not consistency:
            continue
        consistency_present = True
        counts = consistency["summary"]
        lines.extend(
            [
                f"### `{op['op_name']}`",
                "",
                (
                    f"总计 {counts['total']}，consistent={counts['consistent']}，"
                    f"unknown={counts['unknown']}，inconsistent={counts['inconsistent']}。"
                ),
                "",
                "| Observation | Status | Issues |",
                "| --- | --- | --- |",
            ]
        )
        for result in consistency["results"]:
            issue_codes = ", ".join(item["code"] for item in result["issues"]) or "-"
            lines.append(f"| `{result['case_id']}` | `{result['status']}` | `{issue_codes}` |")
        lines.append("")
    if not consistency_present:
        lines.append("- 未提供 ShapeContract，未执行一致性验证。")

    lines.append("")
    return "\n".join(lines)


def _build_dataset_report(input_dir: Path, operator_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    seed_static_pass_count = 0
    seed_static_fail_count = 0
    syntax_fail_count = 0
    import_fail_count = 0
    signature_fail_count = 0
    import_styles: Counter[str] = Counter()
    python_file_count = 0
    seed_count = 0
    test_file_count = 0
    shape_observation_count = 0
    shape_consistency_counts = Counter({"consistent": 0, "unknown": 0, "inconsistent": 0})

    for op in operator_reports:
        python_file_count += op["python_file_count"]
        seed_count += op["seed_count"]
        test_file_count += len(op.get("test_files", []))
        shape_observation_count += len(op.get("shape_observations", []))
        consistency = op.get("shape_consistency")
        if consistency:
            for status in shape_consistency_counts:
                shape_consistency_counts[status] += consistency["summary"][status]
        for seed in op["seed_reports"]:
            if seed["syntax_ok"] and seed["imports_ok"] and seed["signature_ok"]:
                seed_static_pass_count += 1
            else:
                seed_static_fail_count += 1
            if not seed["syntax_ok"]:
                syntax_fail_count += 1
            if not seed["imports_ok"]:
                import_fail_count += 1
            if not seed["signature_ok"]:
                signature_fail_count += 1
            import_styles[_classify_import_style(seed.get("imports", []))] += 1

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "summary": {
            "operator_count": len(operator_reports),
            "python_file_count": python_file_count,
            "seed_count": seed_count,
            "test_file_count": test_file_count,
            "shape_observation_count": shape_observation_count,
            "shape_consistency": {
                "total": sum(shape_consistency_counts.values()),
                **dict(shape_consistency_counts),
            },
            "seed_static_pass_count": seed_static_pass_count,
            "seed_static_fail_count": seed_static_fail_count,
            "syntax_fail_count": syntax_fail_count,
            "import_fail_count": import_fail_count,
            "signature_fail_count": signature_fail_count,
            "import_style_counts": dict(import_styles),
        },
        "operators": operator_reports,
    }


def _shape_consistency_report(
    contract: ShapeContract,
    observation_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    results = [
        validate_shape_consistency(contract, ShapeObservation.from_dict(record)).to_dict()
        for record in observation_records
    ]
    counts = Counter(result["status"] for result in results)
    return {
        "contract": contract.to_dict(),
        "summary": {
            "total": len(results),
            "consistent": counts["consistent"],
            "unknown": counts["unknown"],
            "inconsistent": counts["inconsistent"],
        },
        "results": results,
    }


def _audit_python_file(path: Path, input_dir: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "path": str(path.relative_to(input_dir)),
            "syntax_ok": False,
            "error": f"SyntaxError: {exc.msg} at line {exc.lineno}, column {exc.offset}",
            "imports": [],
            "defined_functions": [],
        }

    imports = []
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
                imports.extend(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)

    return {
        "path": str(path.relative_to(input_dir)),
        "syntax_ok": True,
        "error": None,
        "imports": sorted(set(imports)),
        "defined_functions": sorted(set(functions)),
    }


def _extract_test_hints(test_file: Optional[Path], input_dir: Path) -> Dict[str, Any]:
    if test_file is None:
        return {"path": None, "syntax_ok": None, "tensor_creations": []}

    text = test_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "path": str(test_file.relative_to(input_dir)),
            "syntax_ok": False,
            "error": f"SyntaxError: {exc.msg} at line {exc.lineno}, column {exc.offset}",
            "tensor_creations": [],
        }

    constants = _collect_simple_constants(tree)
    tensor_creations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if not _is_torch_tensor_creation(call_name):
            continue
        tensor_creations.append(
            {
                "assigned_to": _assigned_name(tree, node),
                "call": call_name,
                "shape": _extract_shape(call_name, node, constants),
                "dtype": _keyword_value(node, "dtype", constants),
                "device": _keyword_value(node, "device", constants),
                "lineno": getattr(node, "lineno", None),
            }
        )

    return {
        "path": str(test_file.relative_to(input_dir)),
        "syntax_ok": True,
        "tensor_creations": tensor_creations,
    }


def _extract_shape_observations(
    op_name: str,
    test_file: Path,
    input_dir: Path,
) -> List[Dict[str, Any]]:
    """Extract evidence-only observations, grouped by public test scope."""

    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    relative_path = str(test_file.relative_to(input_dir))
    observations: List[Dict[str, Any]] = []
    for qualname, function in _test_scopes(tree):
        constants = _collect_scope_constants(function)
        tensor_shapes: Dict[str, List[Optional[int]]] = {}
        tensor_dtypes: Dict[str, Optional[str]] = {}
        for node in _walk_scope(function):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func)
            if not _is_torch_tensor_creation(call_name):
                continue
            assigned_to = _assigned_name(function, node)
            normalized_shape = _normalize_observed_shape(_extract_shape(call_name, node, constants))
            if assigned_to is None or normalized_shape is None:
                continue
            tensor_name = assigned_to
            if tensor_name in tensor_shapes:
                tensor_name = f"{tensor_name}@L{getattr(node, 'lineno', 0)}"
            tensor_shapes[tensor_name] = normalized_shape
            assignment = _assignment_for_call(function, node)
            dtype = _keyword_value(node, "dtype", constants) if assignment and assignment.value is node else None
            tensor_dtypes[tensor_name] = dtype if isinstance(dtype, str) and dtype else None

        if not tensor_shapes:
            continue
        observation = ShapeObservation(
            op_name=op_name,
            case_id=f"{relative_path}::{qualname}",
            tensor_shapes=tensor_shapes,
            tensor_dtypes=tensor_dtypes,
            source="static_test_hint",
            source_ref=f"{relative_path}:{qualname}:{getattr(function, 'lineno', 1)}",
        )
        record = observation.to_dict()
        record["signature"] = observation.signature()
        observations.append(record)
    return observations


def _test_scopes(tree: ast.Module) -> List[tuple[str, ast.AST]]:
    scopes: List[tuple[str, ast.AST]] = [("<module>", tree)]

    def collect(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name.startswith("test_"):
                    scopes.append((f"{prefix}{child.name}", child))
                continue
            if isinstance(child, ast.ClassDef):
                collect(child, f"{prefix}{child.name}.")
            else:
                collect(child, prefix)

    collect(tree)
    return scopes


def _walk_scope(function: ast.AST):
    """Walk one test body without descending into nested definitions."""

    stack = list(reversed(getattr(function, "body", [])))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _collect_scope_constants(scope: ast.AST) -> Dict[str, Any]:
    constants: Dict[str, Any] = {}
    for node in _walk_scope(scope):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                value = _literalish(node.value, constants, allow_expr=False)
                if value is not None:
                    constants[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)):
                values = [_literalish(item, constants, allow_expr=False) for item in node.value.elts]
                for name_node, value in zip(target.elts, values):
                    if isinstance(name_node, ast.Name) and value is not None:
                        constants[name_node.id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _literalish(node.value, constants, allow_expr=False) if node.value else None
            if value is not None:
                constants[node.target.id] = value
    return constants


def _normalize_observed_shape(shape: Any) -> Optional[List[Optional[int]]]:
    if not isinstance(shape, list):
        return None
    normalized: List[Optional[int]] = []
    for dimension in shape:
        if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0:
            normalized.append(dimension)
        else:
            normalized.append(None)
    return normalized


def _collect_simple_constants(tree: ast.AST) -> Dict[str, Any]:
    constants: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name):
                value = _literalish(node.value, constants, allow_expr=False)
                if value is not None:
                    constants[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)):
                values = [_literalish(item, constants, allow_expr=False) for item in node.value.elts]
                for name_node, value in zip(target.elts, values):
                    if isinstance(name_node, ast.Name) and value is not None:
                        constants[name_node.id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _literalish(node.value, constants, allow_expr=False) if node.value else None
            if value is not None:
                constants[node.target.id] = value
    return constants


def _literalish(node: ast.AST, constants: Dict[str, Any], allow_expr: bool = True) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id, node.id)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_literalish(item, constants, allow_expr=allow_expr) for item in node.elts]
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    if isinstance(node, ast.Call) and _call_name(node.func) == "torch.device" and node.args:
        return _literalish(node.args[0], constants, allow_expr=allow_expr)
    if not allow_expr:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _is_torch_tensor_creation(call_name: str) -> bool:
    return call_name in {
        "torch.empty",
        "torch.empty_like",
        "torch.randn",
        "torch.randint",
        "torch.ones",
        "torch.ones_like",
        "torch.tensor",
        "torch.zeros",
        "torch.zeros_like",
    }


def _assigned_name(tree: ast.AST, call: ast.Call) -> Optional[str]:
    assignment = _assignment_for_call(tree, call)
    if assignment and len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name):
        return assignment.targets[0].id
    return None


def _assignment_for_call(tree: ast.AST, call: ast.Call) -> Optional[ast.Assign]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        contains_call = node.value is call or any(child is call for child in ast.walk(node.value))
        if not contains_call:
            continue
        return node
    return None


def _keyword_value(call: ast.Call, name: str, constants: Dict[str, Any]) -> Any:
    for keyword in call.keywords:
        if keyword.arg == name:
            return _literalish(keyword.value, constants)
    return None


def _extract_shape(call_name: str, call: ast.Call, constants: Dict[str, Any]) -> Any:
    if call_name in {"torch.empty_like", "torch.zeros_like", "torch.ones_like"}:
        return _literalish(call.args[0], constants) if call.args else None
    if call_name == "torch.randint":
        return _literalish(call.args[2], constants) if len(call.args) >= 3 else _keyword_value(call, "size", constants)
    if call_name == "torch.tensor":
        return _literal_shape(call.args[0]) if call.args else None
    if len(call.args) == 1:
        value = _literalish(call.args[0], constants)
        return value if isinstance(value, list) else [value]
    return [_literalish(arg, constants) for arg in call.args]


def _literal_shape(node: ast.AST) -> Optional[List[int]]:
    if isinstance(node, (ast.List, ast.Tuple)):
        if not node.elts:
            return [0]
        child = _literal_shape(node.elts[0])
        return [len(node.elts)] + (child or [])
    return []


def _classify_import_style(imports: List[str]) -> str:
    import_set = set(imports)
    if "vllm.triton_utils" in import_set:
        return "vllm.triton_utils"
    if "triton" in import_set and "triton.language" in import_set:
        return "standard_triton"
    if "triton" in import_set:
        return "partial_triton"
    return "missing_triton"
