#!/usr/bin/env python3
"""Build one traceable, format-only official output smoke ZIP."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.output_contract import validate_output_contract


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_smoke_zip(
    datasets_dir: Path,
    kernel: str,
    source: Path,
    output_zip: Path,
) -> dict:
    datasets_dir = datasets_dir.resolve()
    source = source.resolve()
    try:
        source_relative = source.relative_to(datasets_dir)
    except ValueError as exc:
        raise ValueError("Source must be inside the selected datasets directory") from exc
    if len(source_relative.parts) != 2 or source_relative.parts[0] != kernel:
        raise ValueError(f"Source must be directly inside datasets/{kernel}/")
    if source.suffix != ".py":
        raise ValueError("Source candidate must be a Python file")
    if output_zip.suffix.lower() != ".zip":
        raise ValueError("Output artifact must use the .zip suffix")
    manifest_path = output_zip.with_suffix(".manifest.json")
    if output_zip.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing artifact or manifest")

    source_bytes = source.read_bytes()
    ast.parse(source_bytes.decode("utf-8"), filename=str(source_relative))
    baseline = datasets_dir / kernel / f"{kernel}.py"
    baseline_bytes = baseline.read_bytes()
    if source_bytes == baseline_bytes:
        raise ValueError("Format smoke candidate must not be byte-identical to baseline")

    target_relative = Path("output") / kernel / f"{kernel}_1.py"
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        target = staging / target_relative
        target.parent.mkdir(parents=True)
        target.write_bytes(source_bytes)
        report = validate_output_contract(
            staging / "output", datasets_dir, naming="numeric", kernel=kernel
        )
        if not report["valid"]:
            raise ValueError(f"Generated output failed contract: {report['errors']}")

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        info = zipfile.ZipInfo(str(target_relative), date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        with zipfile.ZipFile(output_zip, "w", compresslevel=9) as archive:
            archive.writestr(info, source_bytes)

    with zipfile.ZipFile(output_zip) as archive:
        target_bytes = archive.read(str(target_relative))
    if target_bytes != source_bytes:
        raise RuntimeError("Archived candidate differs from its source")
    artifact_bytes = output_zip.read_bytes()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "official-output-format-smoke",
        "scoring_intent": "format-only",
        "kernel": kernel,
        "naming": "numeric",
        "source_path": str(source_relative),
        "source_sha256": sha256_bytes(source_bytes),
        "baseline_path": str(baseline.relative_to(datasets_dir)),
        "baseline_sha256": sha256_bytes(baseline_bytes),
        "target_path": str(target_relative),
        "target_sha256": sha256_bytes(target_bytes),
        "artifact_path": output_zip.name,
        "artifact_size": len(artifact_bytes),
        "artifact_sha256": sha256_bytes(artifact_bytes),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-zip", required=True)
    args = parser.parse_args()
    manifest = build_smoke_zip(
        Path(args.datasets_dir), args.kernel, Path(args.source), Path(args.output_zip)
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
