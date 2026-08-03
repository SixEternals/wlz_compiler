#!/usr/bin/env python3
"""Build a deterministic, audited source ZIP for the official Agent smoke run."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
AGENT_SOURCE = ROOT / "work" / "official_triton_agent"
MAX_ARTIFACT_BYTES = 20_971_520
SOURCE_LAYOUT = {
    "Agent/config.py": AGENT_SOURCE / "config.py",
    "Agent/contract_executor.py": AGENT_SOURCE / "contract_executor.py",
    "Agent/evolutionary_algorithm.py": AGENT_SOURCE / "evolutionary_algorithm.py",
    "Agent/executor.py": AGENT_SOURCE / "executor.py",
    "Agent/genetic_operators.py": AGENT_SOURCE / "genetic_operators.py",
    "Agent/llm_interface.py": AGENT_SOURCE / "llm_interface.py",
    "Agent/main.py": AGENT_SOURCE / "main.py",
    "Agent/optimizer_agent.py": AGENT_SOURCE / "optimizer_agent.py",
    "Agent/README.md": AGENT_SOURCE / "readme.md",
    "Agent/README_English.md": AGENT_SOURCE / "README_English.md",
    "Agent/baseline/baseline.json": AGENT_SOURCE / "baseline" / "baseline.json",
    "Agent/wlz_optimizer/__init__.py": ROOT / "wlz_optimizer" / "__init__.py",
    "Agent/wlz_optimizer/budget.py": ROOT / "wlz_optimizer" / "budget.py",
}
SECRET_PATTERN = re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")
SMOKE_OVERRIDES = {
    "population_size": 2,
    "max_generations": 0,
    "max_total_tokens": 8_192,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_source(entry: str, data: bytes) -> None:
    path = PurePosixPath(entry)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "Agent":
        raise ValueError(f"Unsafe source entry: {entry}")
    forbidden = {".git", ".secrets", "__pycache__", "output", "datasets"}
    if forbidden.intersection(path.parts) or path.suffix == ".pyc":
        raise ValueError(f"Forbidden source entry: {entry}")
    if SECRET_PATTERN.search(data):
        raise ValueError(f"Potential API key found in source entry: {entry}")
    if path.suffix == ".py":
        compile(data, entry, "exec")


def _smoke_config(source: bytes) -> bytes:
    """Apply explicit smoke-only dataclass defaults without editing production."""
    tree = ast.parse(source, filename="Agent/config.py")
    updated = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "EAConfig":
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id in SMOKE_OVERRIDES
            ):
                statement.value = ast.Constant(SMOKE_OVERRIDES[statement.target.id])
                updated.add(statement.target.id)
    if updated != set(SMOKE_OVERRIDES):
        raise ValueError(f"Cannot apply smoke config overrides: {sorted(updated)}")
    ast.fix_missing_locations(tree)
    result = (ast.unparse(tree) + "\n").encode("utf-8")
    compile(result, "Agent/config.py", "exec")
    return result


def _verify_archive(path: Path, expected: list[str]) -> None:
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Artifact exceeds platform limit of {MAX_ARTIFACT_BYTES} bytes")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        bad_entry = archive.testzip()
    if bad_entry is not None:
        raise ValueError(f"ZIP integrity check failed at {bad_entry}")
    if names != expected:
        raise ValueError("ZIP entries do not match the fixed source layout")
    for name in names:
        candidate = PurePosixPath(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Unsafe ZIP entry: {name}")


def build_source_smoke(output_zip: Path) -> dict:
    if output_zip.suffix.lower() != ".zip":
        raise ValueError("Output artifact must use the .zip suffix")
    manifest_path = output_zip.with_suffix(".manifest.json")
    if output_zip.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an artifact or manifest")

    entries = {}
    for archive_name, source in SOURCE_LAYOUT.items():
        data = source.read_bytes()
        if archive_name == "Agent/config.py":
            data = _smoke_config(data)
        _validate_source(archive_name, data)
        entries[archive_name] = data
    expected = sorted(entries)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(output_zip, "w", compresslevel=9) as archive:
            for name in expected:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, entries[name])
        _verify_archive(output_zip, expected)
    except Exception:
        output_zip.unlink(missing_ok=True)
        raise

    artifact = output_zip.read_bytes()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "official-agent-source-smoke",
        "smoke_configuration": True,
        "effective_config": dict(SMOKE_OVERRIDES),
        "official_scoring_ready": False,
        "layout": "documented-Agent-root-v1",
        "expected_entrypoint": "Agent/main.py",
        "archive_entries": expected,
        "source_sha256": {name: _sha256(entries[name]) for name in expected},
        "artifact_path": output_zip.name,
        "artifact_size": len(artifact),
        "artifact_sha256": _sha256(artifact),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-zip", required=True)
    args = parser.parse_args()
    print(json.dumps(build_source_smoke(Path(args.output_zip)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
