#!/usr/bin/env python3
"""Append, replay, or list identity-bound official evaluations offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.cache import OfficialEvaluationHistory
from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.official_adapter import adapt_bound_official_evaluation
from wlz_optimizer.schemas import Candidate


def _load_candidate(manifest_path: Path) -> Candidate:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = manifest["candidate"]
    candidate_path = manifest_path.with_name(f"{data['id']}.py")
    code = candidate_path.read_text(encoding="utf-8")
    if data.get("code_hash") != sha256_text(code):
        raise ValueError("Candidate source does not match manifest code_hash")
    return Candidate(
        id=data["id"],
        op_name=data["op_name"],
        code=code,
        code_hash=data["code_hash"],
        parent_ids=list(data.get("parent_ids", [])),
        generation=int(data.get("generation", 0)),
        mutation_kind=data.get("mutation_kind", "unknown"),
        model_used=data.get("model_used"),
        prompt_id=data.get("prompt_id"),
        status=data.get("status", "unknown"),
        score=data.get("score"),
        metadata=dict(data.get("metadata", {})),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("append", "replay", "list"))
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--env-fingerprint", required=True)
    parser.add_argument("--observation-id")
    parser.add_argument("--envelope", help="Required only for append mode")
    parser.add_argument("--baseline-time-us", type=float)
    args = parser.parse_args()

    if args.mode in ("append", "replay") and not args.observation_id:
        parser.error(f"{args.mode} mode requires --observation-id")
    if args.mode == "list" and (
        args.observation_id or args.envelope or args.baseline_time_us is not None
    ):
        parser.error(
            "list mode does not accept --observation-id, --envelope, or --baseline-time-us"
        )

    candidate = _load_candidate(Path(args.candidate_manifest))
    history = OfficialEvaluationHistory(Path(args.history))
    if args.mode == "list":
        observations = history.list_observations(candidate, args.env_fingerprint)
        print(
            json.dumps(
                {
                    "action": "listed",
                    "count": len(observations),
                    "observations": [result.to_dict() for result in observations],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "append":
        if not args.envelope:
            parser.error("append mode requires --envelope")
        envelope = json.loads(Path(args.envelope).read_text(encoding="utf-8"))
        if not isinstance(envelope, Mapping):
            raise TypeError("Official evaluation envelope must be a mapping")
        result = adapt_bound_official_evaluation(
            candidate, envelope, baseline_time_us=args.baseline_time_us
        )
        key = history.make_key(
            candidate.op_name,
            candidate.id,
            candidate.code_hash,
            args.env_fingerprint,
            args.observation_id,
        )
        action = "existing" if key in history.entries else "appended"
        history.append(
            candidate, envelope, result, args.env_fingerprint, args.observation_id
        )
    else:
        if args.envelope or args.baseline_time_us is not None:
            parser.error("replay mode does not accept --envelope or --baseline-time-us")
        result = history.replay(
            candidate, args.env_fingerprint, args.observation_id
        )
        if result is None:
            print(json.dumps({"action": "not_found"}, sort_keys=True))
            return 1
        key = result.metadata["history_key"]
        action = "replayed"

    print(
        json.dumps(
            {"action": action, "history_key": key, "result": result.to_dict()},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
