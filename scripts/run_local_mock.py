#!/usr/bin/env python3
"""Run the Python-first local mock optimization loop.

Example:
    python scripts/run_local_mock.py --input-dir doc/repos/Compiler2026-nwu/datasets --output-dir output/mock
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.cache import EvaluationCache
from wlz_optimizer.evolutionary_algorithm import EvolutionaryAlgorithm
from wlz_optimizer.executors import LocalExecutor
from wlz_optimizer.genetic_operators import GeneticOperators
from wlz_optimizer.io_utils import discover_operators, load_operator_input
from wlz_optimizer.report import write_run_outputs
from wlz_optimizer.schemas import CandidateEvaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local static/mock runner for Compiler2026 Triton EA skeleton."
    )
    parser.add_argument("--input-dir", required=True, help="Compiler2026-style datasets directory")
    parser.add_argument("--output-dir", required=True, help="Directory for mock run outputs")
    parser.add_argument("--kernel", default=None, help="Optional single operator name")
    parser.add_argument("--population-size", type=int, default=6, help="Candidates per generation")
    parser.add_argument("--generations", type=int, default=2, help="Mock evolution generations")
    parser.add_argument("--top-k", type=int, default=5, help="Top candidates per operator")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = EvaluationCache(output_dir / "cache.jsonl")
    executor = LocalExecutor()
    genetic_ops = GeneticOperators(model_name="stub-local")

    start = time.time()
    operators = discover_operators(input_dir, args.kernel)
    all_evaluations: List[CandidateEvaluation] = []

    print(f"[local-mock] input={input_dir}")
    print(f"[local-mock] output={output_dir}")
    print(f"[local-mock] operators={len(operators)} population={args.population_size} generations={args.generations}")

    for op_name in operators:
        operator_input = load_operator_input(input_dir, op_name)
        print(
            "[local-mock] operator={op} seeds={seeds} required_functions={funcs}".format(
                op=op_name,
                seeds=len(operator_input.seeds),
                funcs=",".join(operator_input.required_functions) or "<any>",
            )
        )
        ea = EvolutionaryAlgorithm(
            genetic_ops=genetic_ops,
            executor=executor,
            cache=cache,
            population_size=args.population_size,
            generations=args.generations,
            elite_count=min(2, args.population_size),
        )
        all_evaluations.extend(ea.run(operator_input, output_dir))

    elapsed = time.time() - start
    run_config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "kernel": args.kernel or "<all>",
        "population_size": args.population_size,
        "generations": args.generations,
        "top_k": args.top_k,
        "executor": executor.kind,
        "env_fingerprint": executor.env_fingerprint(),
        "elapsed_seconds": round(elapsed, 3),
    }
    manifest = write_run_outputs(output_dir, all_evaluations, run_config, top_k=args.top_k)

    summary = manifest["summary"]
    print(
        "[local-mock] done operators={ops} candidates={cands} cache_hits={hits}".format(
            ops=summary["operator_count"],
            cands=summary["candidate_count"],
            hits=summary["cache_hits"],
        )
    )
    print(f"[local-mock] wrote {output_dir / 'manifest.json'}")
    print(f"[local-mock] wrote {output_dir / 'run_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
