#!/usr/bin/env python3
"""Audit a Compiler2026-style dataset without touching mock outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.dataset_audit import audit_dataset, write_audit_outputs
from wlz_optimizer.schemas import ShapeContract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only static audit for Compiler2026-style Triton datasets."
    )
    parser.add_argument("--input-dir", required=True, help="Compiler2026-style datasets directory")
    parser.add_argument("--output-dir", required=True, help="Directory for audit report outputs")
    parser.add_argument("--kernel", default=None, help="Optional single operator name")
    parser.add_argument("--shape-contract", default=None, help="Optional ShapeContract JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    shape_contract = None
    if args.shape_contract:
        try:
            contract_data = json.loads(Path(args.shape_contract).read_text(encoding="utf-8"))
            shape_contract = ShapeContract.from_dict(contract_data)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"error: invalid shape contract: {exc}", file=sys.stderr)
            return 2

    try:
        report = audit_dataset(
            input_dir=input_dir,
            kernel=args.kernel,
            shape_contract=shape_contract,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = write_audit_outputs(report, output_dir=output_dir)
    summary = report["summary"]

    print(
        "[dataset-audit] operators={ops} seeds={seeds} seed_pass={passed} seed_fail={failed}".format(
            ops=summary["operator_count"],
            seeds=summary["seed_count"],
            passed=summary["seed_static_pass_count"],
            failed=summary["seed_static_fail_count"],
        )
    )
    print(f"[dataset-audit] wrote {paths['json']}")
    print(f"[dataset-audit] wrote {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
