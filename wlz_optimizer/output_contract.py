"""Static checks for an official-style generated ``output/`` directory."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def validate_output_contract(
    output_dir: Path,
    datasets_dir: Path,
    naming: str = "numeric",
    kernel: str | None = None,
) -> Dict[str, Any]:
    """Validate discovery, naming, Python syntax, and optional JSON metadata."""

    if naming not in {"numeric", "reference-v"}:
        raise ValueError(f"Unsupported naming mode: {naming}")

    errors: List[Dict[str, str]] = []
    operators = _discover_operators(datasets_dir)
    if kernel is not None:
        if kernel not in operators:
            raise ValueError(f"Unknown kernel: {kernel}")
        operators = [kernel]
    if not output_dir.is_dir():
        errors.append(_issue("output_root_missing", output_dir))
        return _report(output_dir, operators, errors)

    for operator in operators:
        operator_dir = output_dir / operator
        if not operator_dir.is_dir():
            errors.append(_issue("operator_dir_missing", operator_dir))
            continue

        marker = "_v" if naming == "reference-v" else "_"
        alternate_marker = "_" if naming == "reference-v" else "_v"
        pattern = re.compile(rf"^{re.escape(operator)}{marker}(\d+)\.py$")
        alternate_pattern = re.compile(
            rf"^{re.escape(operator)}{alternate_marker}(\d+)\.py$"
        )
        candidates = []
        indices = []
        alternate_candidates = []
        for path in sorted(operator_dir.glob("*.py")):
            match = pattern.match(path.name)
            if match:
                candidates.append(path)
                indices.append(int(match.group(1)))
            elif alternate_pattern.match(path.name):
                alternate_candidates.append(path)

        if not candidates:
            errors.append(_issue("candidate_missing", operator_dir))
            continue
        if alternate_candidates:
            errors.append(_issue("candidate_alias_conflict", operator_dir))
        if len(candidates) > 5 or any(index < 1 or index > 5 for index in indices):
            errors.append(_issue("candidate_count_invalid", operator_dir))
        if indices != list(range(1, len(indices) + 1)):
            errors.append(_issue("candidate_numbering_invalid", operator_dir))

        for path in candidates:
            _check_python(path, errors)

        # The official organizer-save-results layout (reference-v naming)
        # requires the best/stats artifacts; numeric format probes do not.
        best_path = operator_dir / f"{operator}_best.py"
        if best_path.exists():
            _check_python(best_path, errors)
        elif naming == "reference-v":
            errors.append(_issue("best_candidate_missing", best_path))

        stats_path = operator_dir / f"{operator}_stats.json"
        if not stats_path.exists():
            if naming == "reference-v":
                errors.append(_issue("stats_json_missing", stats_path))
        else:
            try:
                json.loads(stats_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(_issue("stats_json_invalid", stats_path, str(exc)))

    return _report(output_dir, operators, errors)


def _discover_operators(datasets_dir: Path) -> List[str]:
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f"Datasets directory does not exist: {datasets_dir}")
    operators = [
        path.name
        for path in sorted(datasets_dir.iterdir())
        if path.is_dir() and (path / f"{path.name}.py").is_file()
    ]
    if not operators:
        raise FileNotFoundError(f"No operator directories found under {datasets_dir}")
    return operators


def _check_python(path: Path, errors: List[Dict[str, str]]) -> None:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as exc:
        errors.append(_issue("candidate_python_invalid", path, str(exc)))


def _issue(code: str, path: Path, detail: str = "") -> Dict[str, str]:
    return {"code": code, "path": str(path), "detail": detail}


def _report(output_dir: Path, operators: List[str], errors: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "valid": not errors,
        "output_dir": str(output_dir),
        "expected_operator_count": len(operators),
        "errors": errors,
    }
