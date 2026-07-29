"""Safe, deterministic repair guidance derived from official failures."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from wlz_optimizer.budget import BudgetController
from wlz_optimizer.cache import OfficialFailureHistory


_ACTIONABLE_KINDS = ("runtime_error", "accuracy_check_failed")
REPAIR_POLICY_VERSION = "official-repair-policy-v1"


@dataclass(frozen=True)
class RepairDecision:
    """One deterministic decision; guidance remains transient and is never persisted."""

    allowed: bool
    reason: str
    guidance: Optional[str]
    estimated_total_tokens: int
    expected_seconds: float
    policy_version: str = REPAIR_POLICY_VERSION


def build_official_repair_guidance(
    history: OfficialFailureHistory,
    *,
    operator: str,
    candidate_code_hash: str,
    observation_id: str,
) -> Optional[str]:
    """Build coarse guidance without exposing raw official diagnostics."""
    if not isinstance(history, OfficialFailureHistory):
        raise TypeError("Official repair guidance requires OfficialFailureHistory")
    if not isinstance(operator, str) or not operator or operator != operator.strip():
        raise ValueError("Official repair operator must be non-empty")
    if not isinstance(candidate_code_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", candidate_code_hash
    ) is None:
        raise ValueError("Official repair candidate code hash must be lowercase SHA-256")
    if (
        not isinstance(observation_id, str)
        or not observation_id
        or observation_id != observation_id.strip()
    ):
        raise ValueError("Official repair observation ID must be non-empty")

    environments = set()
    cases = {kind: set() for kind in _ACTIONABLE_KINDS}
    for record in history.entries.values():
        if not (
            record["operator"] == operator
            and record["candidate_code_hash"] == candidate_code_hash
            and record["observation_id"] == observation_id
        ):
            continue
        environments.add(record["env_fingerprint"])
        task = record["task_failure"]
        if task["failure_kind"] in cases:
            cases[task["failure_kind"]].add(task["test_case"])
    if len(environments) > 1:
        raise ValueError("Official repair observation is ambiguous across environments")

    rows = []
    for kind in _ACTIONABLE_KINDS:
        if cases[kind]:
            rows.append(f"- {kind}: {len(cases[kind])} observed case(s)")
    if not rows:
        return None
    guidance = "\n".join([
        "Official evaluation feedback for this exact parent code:",
        *rows,
        "Repair only these observed failure categories while preserving the function "
        "interface and unrelated behavior.",
    ])
    if "\x00" in guidance or len(guidance) > 4096:  # pragma: no cover - fixed template bound
        raise ValueError("Generated official repair guidance is invalid")
    return guidance


def decide_official_repair(
    history: OfficialFailureHistory,
    *,
    operator: str,
    candidate_code_hash: str,
    observation_id: str,
    prior_repair_attempts: int,
    budget: BudgetController,
    estimated_total_tokens: int,
    expected_seconds: float,
) -> RepairDecision:
    """Allow one exact, actionable repair only when the existing budget can fund it."""

    if (
        isinstance(prior_repair_attempts, bool)
        or not isinstance(prior_repair_attempts, int)
        or prior_repair_attempts < 0
    ):
        raise ValueError("prior_repair_attempts must be a non-negative integer")
    if not isinstance(budget, BudgetController):
        raise TypeError("budget must be a BudgetController")
    if (
        isinstance(estimated_total_tokens, bool)
        or not isinstance(estimated_total_tokens, int)
        or estimated_total_tokens <= 0
    ):
        raise ValueError("estimated_total_tokens must be a positive integer")
    if (
        isinstance(expected_seconds, bool)
        or not isinstance(expected_seconds, (int, float))
        or not math.isfinite(float(expected_seconds))
        or expected_seconds <= 0
    ):
        raise ValueError("expected_seconds must be positive")
    if prior_repair_attempts >= 1:
        return RepairDecision(
            False,
            "repair_attempt_limit",
            None,
            estimated_total_tokens,
            float(expected_seconds),
        )
    guidance = build_official_repair_guidance(
        history,
        operator=operator,
        candidate_code_hash=candidate_code_hash,
        observation_id=observation_id,
    )
    if guidance is None:
        return RepairDecision(
            False,
            "no_actionable_exact_evidence",
            None,
            estimated_total_tokens,
            float(expected_seconds),
        )
    budget_decision = budget.check_start(estimated_total_tokens, expected_seconds)
    if not budget_decision.allowed:
        return RepairDecision(
            False,
            f"budget:{budget_decision.reason}",
            None,
            estimated_total_tokens,
            float(expected_seconds),
        )
    return RepairDecision(
        True,
        "exact_actionable_evidence",
        guidance,
        estimated_total_tokens,
        float(expected_seconds),
    )
