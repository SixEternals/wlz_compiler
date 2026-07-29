"""Deterministic first-pass candidate generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from wlz_optimizer.hash_utils import sha256_text, short_hash
from wlz_optimizer.schemas import Candidate, OperatorInput


class LlmClient(Protocol):
    def generate(self, prompt: str, purpose: str) -> str:
        ...


class StubLlmClient:
    """No-op LLM client reserved for later integration."""

    model_name = "stub-local"

    def generate(self, prompt: str, purpose: str) -> str:
        raise RuntimeError("StubLlmClient does not generate code in the local mock skeleton")


class GeneticOperators:
    """Generate safe mock variants while preserving provenance."""

    mutation_cycle = [
        "block_size_hint",
        "num_warps_hint",
        "masking_hint",
        "constexpr_hint",
        "elite_mutation",
    ]

    def __init__(self, model_name: str = "stub-local") -> None:
        self.model_name = model_name

    def initial_population(self, operator_input: OperatorInput, population_size: int) -> List[Candidate]:
        candidates: List[Candidate] = []

        for idx, seed in enumerate(operator_input.seeds):
            if len(candidates) >= population_size:
                break
            kind = "baseline" if idx == 0 else "seed_variant"
            candidates.append(
                self._candidate(
                    op_name=operator_input.op_name,
                    code=seed["code"],
                    parent_ids=[],
                    generation=0,
                    mutation_kind=kind,
                    ordinal=len(candidates),
                    prompt_id=None,
                    metadata={
                        "provenance": {
                            "source": kind,
                            "source_path": str(seed["path"]),
                        }
                    },
                )
            )

        seed_code = operator_input.seeds[0]["code"]
        while len(candidates) < population_size:
            ordinal = len(candidates)
            mutation_kind = self.mutation_cycle[(ordinal - 1) % len(self.mutation_cycle)]
            code = self._annotate(seed_code, operator_input.op_name, mutation_kind, 0, ordinal)
            candidates.append(
                self._candidate(
                    op_name=operator_input.op_name,
                    code=code,
                    parent_ids=[],
                    generation=0,
                    mutation_kind=mutation_kind,
                    ordinal=ordinal,
                    prompt_id=f"mock-init-{mutation_kind}",
                    metadata={
                        "provenance": {
                            "source": "deterministic_mock_mutation",
                            "strategy": mutation_kind,
                            "seed_path": str(operator_input.seeds[0]["path"]),
                        }
                    },
                )
            )

        return candidates

    def mutate(self, parent: Candidate, generation: int, ordinal: int) -> Candidate:
        mutation_kind = self.mutation_cycle[(generation + ordinal) % len(self.mutation_cycle)]
        code = self._annotate(parent.code, parent.op_name, mutation_kind, generation, ordinal)
        return self._candidate(
            op_name=parent.op_name,
            code=code,
            parent_ids=[parent.id],
            generation=generation,
            mutation_kind=mutation_kind,
            ordinal=ordinal,
            prompt_id=f"mock-gen{generation}-{mutation_kind}",
            metadata={
                "provenance": {
                    "source": "deterministic_mock_mutation",
                    "parent_ids": [parent.id],
                    "parent_hash": parent.code_hash,
                    "strategy": mutation_kind,
                    "generation": generation,
                }
            },
        )

    def _candidate(
        self,
        op_name: str,
        code: str,
        parent_ids: List[str],
        generation: int,
        mutation_kind: str,
        ordinal: int,
        prompt_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> Candidate:
        code_hash = sha256_text(code)
        ident_src = "|".join([op_name, str(generation), str(ordinal), mutation_kind, code_hash])
        candidate_id = f"{op_name}-g{generation:02d}-{ordinal:03d}-{short_hash(ident_src, 8)}"
        return Candidate(
            id=candidate_id,
            op_name=op_name,
            code=code,
            code_hash=code_hash,
            parent_ids=parent_ids,
            generation=generation,
            mutation_kind=mutation_kind,
            model_used=self.model_name,
            prompt_id=prompt_id,
            status="created",
            score=None,
            metadata=metadata,
        )

    def _annotate(
        self,
        code: str,
        op_name: str,
        mutation_kind: str,
        generation: int,
        ordinal: int,
    ) -> str:
        hint = self._hint_text(mutation_kind, generation, ordinal)
        header = [
            f"# wlz-mutation: {mutation_kind}",
            f"# wlz-op: {op_name}",
            f"# wlz-generation: {generation}",
            f"# wlz-ordinal: {ordinal}",
            f"# wlz-hint: {hint}",
            "",
        ]
        return "\n".join(header) + code.rstrip() + "\n"

    @staticmethod
    def _hint_text(mutation_kind: str, generation: int, ordinal: int) -> str:
        if mutation_kind == "block_size_hint":
            value = [16, 32, 64, 128][(generation + ordinal) % 4]
            return f"try BLOCK_SIZE multiple {value} when real executor is available"
        if mutation_kind == "num_warps_hint":
            value = [1, 2, 4, 8][(generation + ordinal) % 4]
            return f"try conservative num_warps={value}"
        if mutation_kind == "masking_hint":
            return "audit mask and boundary handling"
        if mutation_kind == "constexpr_hint":
            return "prefer tl.constexpr for compile-time constants"
        return "preserve elite structure and make one small safe change"
