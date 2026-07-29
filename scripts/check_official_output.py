#!/usr/bin/env python3
"""Check a generated output tree before building an official smoke ZIP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.output_contract import validate_output_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--kernel", help="Validate one format-probe operator only")
    parser.add_argument(
        "--naming",
        choices=("numeric", "reference-v"),
        default="numeric",
        help="Candidate naming policy; numeric means <kernel>_1.py.",
    )
    args = parser.parse_args()
    report = validate_output_contract(
        Path(args.output_dir), Path(args.datasets_dir), args.naming, args.kernel
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
