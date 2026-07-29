"""Input discovery helpers for the Compiler2026-style dataset layout."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def discover_operators(input_dir: Path, kernel: Optional[str] = None) -> List[str]:
    if kernel:
        op_dir = input_dir / kernel
        if not op_dir.is_dir():
            raise FileNotFoundError(f"Operator directory not found: {op_dir}")
        if not (op_dir / f"{kernel}.py").is_file():
            raise FileNotFoundError(f"Operator baseline file not found: {op_dir / (kernel + '.py')}")
        return [kernel]

    operators: List[str] = []
    for item in sorted(input_dir.iterdir()):
        if item.is_dir() and (item / f"{item.name}.py").is_file():
            operators.append(item.name)
    if not operators:
        raise FileNotFoundError(f"No operator directories found under {input_dir}")
    return operators


def load_operator_input(input_dir: Path, op_name: str) -> "OperatorInput":
    from wlz_optimizer.schemas import OperatorInput

    op_dir = input_dir / op_name
    baseline_file = op_dir / f"{op_name}.py"
    if not baseline_file.is_file():
        raise FileNotFoundError(f"Missing baseline file: {baseline_file}")

    test_file = find_test_file(op_dir, op_name)
    seeds = load_seed_codes(op_dir, op_name)
    required_functions = extract_required_functions(baseline_file, test_file, op_name)

    return OperatorInput(
        op_name=op_name,
        op_dir=op_dir,
        baseline_file=baseline_file,
        test_file=test_file,
        seeds=seeds,
        required_functions=required_functions,
    )


def find_test_file(op_dir: Path, op_name: str) -> Optional[Path]:
    test_files = find_test_files(op_dir, op_name)
    return test_files[0] if test_files else None


def find_test_files(op_dir: Path, op_name: str) -> List[Path]:
    """Return every public test file for an operator in stable name order."""

    candidates = list(op_dir.glob(f"test_{op_name}_*.py"))
    unnumbered = op_dir / f"test_{op_name}.py"
    if unnumbered.is_file():
        candidates.append(unnumbered)
    return sorted({path for path in candidates if path.is_file()}, key=lambda path: path.name)


def load_seed_codes(op_dir: Path, op_name: str) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []

    main_file = op_dir / f"{op_name}.py"
    if main_file.is_file():
        seeds.append(_seed_record(main_file, "baseline"))

    for path in sorted(op_dir.glob(f"{op_name}_*.py")):
        if path.name.startswith(f"test_{op_name}"):
            continue
        seeds.append(_seed_record(path, "seed_variant"))

    variants_dir = op_dir / "variants"
    if variants_dir.is_dir():
        for path in sorted(variants_dir.glob("*.py")):
            seeds.append(_seed_record(path, "seed_variant"))

    if not seeds:
        raise FileNotFoundError(f"No seed code found in {op_dir}")
    return seeds


def _seed_record(path: Path, kind: str) -> Dict[str, Any]:
    return {
        "path": path,
        "kind": kind,
        "code": path.read_text(encoding="utf-8"),
    }


def extract_required_functions(
    baseline_file: Path,
    test_file: Optional[Path],
    op_name: str,
) -> List[str]:
    """Find functions the official test imports from the operator module."""

    if test_file and test_file.is_file():
        imported = _imported_names_from_test(test_file, op_name)
        if imported:
            return imported

    baseline_code = baseline_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(baseline_code)
    except SyntaxError:
        return []

    public = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    ]
    if public:
        return public

    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _imported_names_from_test(test_file: Path, op_name: str) -> List[str]:
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return []

    modules = {op_name, "kernel"}
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in modules:
            for alias in node.names:
                if alias.name != "*":
                    names.append(alias.name)
    return _dedupe(names)


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
