"""Minimal evolutionary loop for the local mock skeleton."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from wlz_optimizer.cache import EvaluationCache
from wlz_optimizer.executors import Executor
from wlz_optimizer.genetic_operators import GeneticOperators
from wlz_optimizer.schemas import Candidate, CandidateEvaluation, EvalContext, OperatorInput


class EvolutionaryAlgorithm:
    def __init__(
        self,
        genetic_ops: GeneticOperators,
        executor: Executor,
        cache: EvaluationCache,
        population_size: int,
        generations: int,
        elite_count: int = 2,
    ) -> None:
        self.genetic_ops = genetic_ops
        self.executor = executor
        self.cache = cache
        self.population_size = max(1, population_size)
        self.generations = max(0, generations)
        self.elite_count = max(1, min(elite_count, self.population_size))
        self.history: List[CandidateEvaluation] = []

    def run(self, operator_input: OperatorInput, output_dir) -> List[CandidateEvaluation]:
        context = EvalContext(
            op_name=operator_input.op_name,
            input_dir=operator_input.op_dir,
            output_dir=output_dir,
            required_functions=operator_input.required_functions,
            test_file=operator_input.test_file,
            baseline_file=operator_input.baseline_file,
        )

        population = self.genetic_ops.initial_population(operator_input, self.population_size)
        evaluated = self._evaluate_population(population, context)

        for generation in range(1, self.generations + 1):
            parents = self._top_unique(evaluated, self.elite_count)
            next_population: List[Candidate] = [item.candidate for item in parents]
            ordinal = len(next_population)
            while len(next_population) < self.population_size:
                parent = parents[(ordinal - self.elite_count) % len(parents)].candidate
                next_population.append(self.genetic_ops.mutate(parent, generation, ordinal))
                ordinal += 1
            evaluated = self._evaluate_population(next_population, context)

        return list(self.history)

    def _evaluate_population(
        self,
        population: List[Candidate],
        context: EvalContext,
    ) -> List[CandidateEvaluation]:
        out: List[CandidateEvaluation] = []
        executor_kind = self.executor.kind
        fingerprint = self.executor.env_fingerprint()

        baseline_file = context.baseline_file
        for candidate in population:
            cached = self.cache.get(candidate, executor_kind, fingerprint, baseline_file)
            if cached is None:
                result = self.executor.evaluate(candidate, context)
                self.cache.put(candidate, result, executor_kind, fingerprint, baseline_file)
                # Re-fetch so manifest metadata contains the cache key and hit=false.
                result = self.cache.get(candidate, executor_kind, fingerprint, baseline_file) or result
                if result.metadata is not None:
                    result.metadata["cache_hit"] = False
            else:
                result = cached

            status = "passed" if result.passed else "failed"
            score = result.proxy_score
            candidate = replace(candidate, status=status, score=score)
            evaluation = CandidateEvaluation(candidate=candidate, result=result)
            self.history.append(evaluation)
            out.append(evaluation)

        return out

    def _top_unique(
        self,
        evaluated: List[CandidateEvaluation],
        limit: int,
    ) -> List[CandidateEvaluation]:
        seen_hashes = set()
        top: List[CandidateEvaluation] = []
        for item in sort_evaluations(evaluated):
            if item.candidate.code_hash in seen_hashes:
                continue
            seen_hashes.add(item.candidate.code_hash)
            top.append(item)
            if len(top) >= limit:
                break
        if not top and evaluated:
            top.append(evaluated[0])
        return top


def sort_evaluations(evaluations: List[CandidateEvaluation]) -> List[CandidateEvaluation]:
    return sorted(
        evaluations,
        key=lambda item: (
            item.result.proxy_score if item.result.proxy_score is not None else -1.0,
            1 if item.result.passed else 0,
            item.candidate.generation,
            item.candidate.code_hash,
        ),
        reverse=True,
    )


def top_k_unique(evaluations: List[CandidateEvaluation], k: int) -> List[CandidateEvaluation]:
    seen_by_op: Dict[str, set] = {}
    out: List[CandidateEvaluation] = []
    for item in sort_evaluations(evaluations):
        seen = seen_by_op.setdefault(item.candidate.op_name, set())
        if item.candidate.code_hash in seen:
            continue
        seen.add(item.candidate.code_hash)
        out.append(item)
        if len(out) >= k:
            break
    return out


def top_k_per_operator(
    evaluations: List[CandidateEvaluation],
    k: int,
) -> Dict[str, List[CandidateEvaluation]]:
    grouped: Dict[str, List[CandidateEvaluation]] = {}
    for item in evaluations:
        grouped.setdefault(item.candidate.op_name, []).append(item)

    result: Dict[str, List[CandidateEvaluation]] = {}
    for op_name, items in grouped.items():
        seen = set()
        selected: List[CandidateEvaluation] = []
        for item in sort_evaluations(items):
            if item.candidate.code_hash in seen:
                continue
            seen.add(item.candidate.code_hash)
            selected.append(item)
            if len(selected) >= k:
                break
        result[op_name] = selected
    return result
