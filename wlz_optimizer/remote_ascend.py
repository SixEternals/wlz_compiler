"""Remote Ascend executor placeholder.

The remote environment is not defined yet, so this module intentionally avoids
SSH, paths, CANN versions, soc_version, or login assumptions.
"""

from __future__ import annotations

from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.schemas import Candidate, EvalContext, EvaluationResult


class RemoteAscendExecutor:
    """Interface-compatible placeholder for future real Ascend execution."""

    kind = "remote_ascend_mock"

    def env_fingerprint(self) -> str:
        return sha256_text(f"{self.kind}|not_configured|schema=v1")[:16]

    def evaluate(self, candidate: Candidate, context: EvalContext) -> EvaluationResult:
        return EvaluationResult(
            candidate_id=candidate.id,
            executor=self.kind,
            status="not_configured",
            passed=False,
            correctness_ok=None,
            compile_ok=None,
            latency_ms=None,
            baseline_ms=None,
            speedup=None,
            proxy_score=None,
            error_type="remote_not_configured",
            error_message="Remote Ascend execution is intentionally not configured in the first skeleton.",
            metadata={
                "no_ssh_assumptions": True,
                "no_cann_assumptions": True,
                "no_soc_version_assumptions": True,
            },
        )
