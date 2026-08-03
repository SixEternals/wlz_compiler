#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolutionary Algorithm Main Logic Module - [Student Implementation Area 2]

Implements complete EA workflow:
1. Population Initialization
2. Selection Mechanism
3. Evolution Loop
4. Population Update

Reference EvoPrompt algorithm workflow:
- Use Roulette Wheel Selection
- Preserve elite individuals (Elitism)
- Iterative optimization until termination conditions are met
"""

import random
import time
from math import isfinite
from typing import List, Tuple, Optional
import numpy as np

from config import EAConfig
from genetic_operators import GeneticOperators, Individual, _ancestor_ids
from executor import TritonExecutor, EvaluationResult


class EvolutionaryAlgorithm:
    """
    [Student Implementation Area] Evolutionary Algorithm Main Logic

    TODO: Students need to implement the following methods:
    - initialize_population(): Population initialization
    - select_parents(): Parent selection mechanism
    - evolve_generation(): Single generation evolution process
    - run(): Complete EA process control
    """

    def __init__(self, 
                 genetic_ops: GeneticOperators,
                 executor: TritonExecutor,
                 config: EAConfig):
        """
        Initialize evolutionary algorithm

        Args:
            genetic_ops: Genetic operators instance
            executor: Triton executor instance
            config: Configuration object
        """
        self.genetic_ops = genetic_ops
        self.executor = executor
        self.config = config

        # Population state
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self._deadline: Optional[float] = None

    def _budget_exhausted(self) -> bool:
        """Competition budget: stop when wall clock deadline or token budget is hit."""
        llm = self.genetic_ops.llm
        budget = getattr(llm, "budget_controller", None)
        if budget is not None:
            snapshot = budget.snapshot()
            reason = snapshot.stop_reason or getattr(
                llm, "budget_denial_reason", None
            )
            if reason is not None:
                print(f"[EA] Shared budget stopped ({reason}), stopping")
                return True
            return False

        if self._deadline is not None and time.time() >= self._deadline:
            print("[EA] Time budget exhausted, stopping")
            return True
        if llm.total_tokens_used >= self.config.max_total_tokens:
            print("[EA] Token budget exhausted, stopping")
            return True
        return False

    def _evaluation_timeout(self) -> float:
        """Bound one executor call by the remaining wall-clock budget."""
        timeout = float(self.config.timeout_seconds)
        budget = getattr(self.genetic_ops.llm, "budget_controller", None)
        if budget is not None:
            remaining = float(budget.snapshot().remaining_seconds)
        elif self._deadline is not None:
            remaining = self._deadline - time.time()
        else:
            return timeout
        return max(0.0, min(timeout, remaining))

    @staticmethod
    def _failure_category(result: EvaluationResult) -> Optional[str]:
        """Map executor text to a bounded category without exposing raw logs."""
        if result.success:
            return None
        text = str(result.error or "").lower()
        markers = (
            (("syntax",), "syntax_fail"),
            (("import",), "import_fail"),
            (("signature",), "signature_fail"),
            (("launch contract",), "launch_contract_fail"),
            (("triton", "semantic"), "triton_semantic_fail"),
            (("accuracy", "correctness"), "accuracy_check_failed"),
            (("timeout", "timed out"), "timeout"),
            (("runtime",), "runtime_error"),
        )
        for needles, category in markers:
            if any(needle in text for needle in needles):
                return category
        return "other"

    @classmethod
    def _prompt_context_for_result(cls, previous, result: EvaluationResult) -> dict:
        """Extend only the typed evaluation facts consumed by mutation prompts."""
        source = dict(previous) if isinstance(previous, dict) else {}

        def count(name: str) -> int:
            value = source.get(name, 0)
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

        evaluation_count = count("evaluation_count")

        def triplet(name: str) -> list[int]:
            value = source.get(name, (0, 0, evaluation_count))
            if (
                isinstance(value, (list, tuple))
                and len(value) == 3
                and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value)
                and sum(value) == evaluation_count
            ):
                return list(value)
            return [0, 0, evaluation_count]

        compile_counts = triplet("compile_counts")
        correctness_counts = triplet("correctness_counts")
        category = cls._failure_category(result)
        if result.success:
            compile_counts[0] += 1
            correctness_counts[0] += 1
        else:
            compile_failed = category in {
                "syntax_fail",
                "import_fail",
                "signature_fail",
                "launch_contract_fail",
                "triton_semantic_fail",
            }
            correctness_failed = category == "accuracy_check_failed"
            compile_counts[1 if compile_failed else 0] += 1
            correctness_counts[1 if correctness_failed else 2] += 1

        failures = {}
        raw_failures = source.get("failure_category_counts", ())
        if isinstance(raw_failures, (list, tuple)):
            for item in raw_failures:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and item[0] in {
                        "syntax_fail",
                        "import_fail",
                        "signature_fail",
                        "launch_contract_fail",
                        "triton_semantic_fail",
                        "runtime_error",
                        "accuracy_check_failed",
                        "timeout",
                        "other",
                    }
                    and isinstance(item[1], int)
                    and not isinstance(item[1], bool)
                    and item[1] > 0
                ):
                    failures[item[0]] = item[1]
        if category is not None:
            failures[category] = failures.get(category, 0) + 1

        speedups = []
        raw_speedups = source.get("observed_speedups", ())
        if isinstance(raw_speedups, (list, tuple)):
            for value in raw_speedups[-7:]:
                if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
                    speedups.append(float(value))
        speedup = getattr(result, "speedup", None)
        if isinstance(speedup, (int, float)) and not isinstance(speedup, bool) and isfinite(speedup):
            speedups.append(float(speedup))

        return {
            "sanitization_version": "prompt-context-sanitization-v2",
            "evaluation_count": evaluation_count + 1,
            "evaluation_pass_count": count("evaluation_pass_count") + int(result.success),
            "compile_counts": compile_counts,
            "correctness_counts": correctness_counts,
            "failure_category_counts": sorted(failures.items()),
            "observed_speedups": speedups[-8:],
        }

    def initialize_population(self, seed_codes: List[str]) -> None:
        """
        [Student Implementation] Initialize population

        Implementation requirements:
        1. Use provided high-quality seed codes as initial individuals
        2. Use genetic operators to generate diverse variants to fill remaining population
        3. Evaluate fitness of all initial individuals

        Strategy suggestions:
        - Seed codes usually contain baseline and a few manually optimized versions
        - Can increase population diversity by mutating seed codes
        - Initial population size is determined by config.population_size

        Args:
            seed_codes: Initial seed code list (at least contains baseline)
        """
        # === Student implementation start ===

        print(f"[EA] Initializing population, target size: {self.config.population_size}")

        # Step 1: Add seed codes as initial individuals
        for i, code in enumerate(seed_codes):
            if i >= self.config.population_size:
                break
            ind = Individual(code=code, generation=0)
            self.population.append(ind)

        # Step 2: Use mutation to generate diverse variants to fill remaining
        while len(self.population) < self.config.population_size:
            if self._budget_exhausted():
                break
            # Randomly select seed for mutation
            seed_code = random.choice(seed_codes)
            temp_ind = Individual(code=seed_code, generation=0)

            # Use mutation operator to generate new individual
            try:
                new_ind = self.genetic_ops.mutate(temp_ind)
            except Exception as e:
                print(f"[EA] Warning: mutation failed during init ({e}), using seed instead")
                new_ind = temp_ind
            self.population.append(new_ind)

        # Step 3: Evaluate initial population
        self._evaluate_population()

        # Update best individual
        self.best_individual = max(self.population, key=lambda x: x.fitness)
        print(f"[EA] Initial best fitness: {self.best_individual.fitness:.4f}")

        # === Student implementation end ===

    def select_parents(self) -> Tuple[Individual, Individual]:
        """
        [Student Implementation] Selection: Select two parents based on fitness

        Implementation requirements:
        1. Implement roulette wheel selection (Roulette Wheel Selection)
        2. Or implement tournament selection (Tournament Selection)
        3. Ensure moderate selection pressure, preserving excellent individuals while maintaining diversity

        Roulette wheel selection formula:
        - Calculate total fitness: total = sum(fitness)
        - Selection probability of individual i: p_i = fitness_i / total
        - Randomly select two parents according to probability distribution (without replacement)

        Returns:
            (parent1, parent2): Two selected parent individuals
        """
        # === Student implementation start ===

        # Rank-based selection: probability comes from fitness *rank*, not raw
        # fitness. This keeps every weight strictly positive (so replace=False
        # never crashes when only one individual beat the baseline) and curbs a
        # single strong individual from dominating selection (early convergence).
        n = len(self.population)
        if n <= 2:
            # Not enough individuals to draw two distinct parents; reuse what we have.
            return self.population[0], self.population[-1]

        # Best individual gets weight n, worst gets weight 1 (linear ranking).
        ranked = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        weights = np.arange(n, 0, -1, dtype=float)
        probabilities = weights / weights.sum()

        i, j = np.random.choice(n, size=2, replace=False, p=probabilities)
        return ranked[i], ranked[j]

        # === Student implementation end ===

    def _breed_one(self, parents: Tuple[Individual, Individual]) -> Individual:
        """Create one independent child from a selected parent pair."""
        parent1, parent2 = parents
        better = parent1 if parent1.fitness >= parent2.fitness else parent2

        def clone_better() -> Individual:
            metadata = {
                'parent': better.id,
                'operation': 'clone',
                'lineage': _ancestor_ids(better),
            }
            if isinstance(better.metadata.get('prompt_context'), dict):
                metadata['prompt_context'] = dict(better.metadata['prompt_context'])
            return Individual(
                code=better.code,
                metadata=metadata,
                model_used=better.model_used,
            )

        if random.random() < self.config.crossover_rate:
            try:
                child = self.genetic_ops.crossover(parent1, parent2)
            except Exception as e:
                print(f"[EA] Warning: crossover failed ({e}), cloning better parent")
                child = clone_better()
        else:
            child = clone_better()

        if self._budget_exhausted():
            child.generation = self.generation + 1
            return child

        if random.random() < self.config.mutation_rate:
            try:
                child = self.genetic_ops.mutate(child)
            except Exception as e:
                print(f"[EA] Warning: mutation failed ({e}), keeping unmutated child")

        child.generation = self.generation + 1
        return child

    def evolve_generation(self) -> None:
        """
        [Student Implementation] Evolve one generation: Execute complete evolution steps

        Implementation requirements (reference EvoPrompt GA version):
        1. Elite preservation: Directly retain top-k best individuals
        2. Generate new individuals: Through selection, crossover, mutation fill remaining positions
        3. Evaluate new individuals: Calculate fitness
        4. Population update: Select next generation population ((mu+lambda) strategy)

        Standard process:
        - Generate N new individuals (N = population_size)
        - Merge with current population (total 2N)
        - Evaluate all individuals
        - Select top-N to form the new generation

        Tips:
        - Crossover probability: config.crossover_rate
        - Mutation probability: config.mutation_rate
        - Elite ratio: config.elite_ratio
        """
        # === Student implementation start ===

        old_population = list(self.population)
        offspring = []
        print("[EA] (μ+λ) selection: merging parents and offspring, keeping top-N by fitness")

        while len(offspring) < self.config.population_size:
            if self._budget_exhausted():
                break
            offspring.append(self._breed_one(self.select_parents()))

        # _evaluate_population walks self.population, so expose both groups while
        # evaluating and then perform true (mu+lambda) survivor selection.
        self.population = old_population + offspring
        self._evaluate_population()

        best_by_code = {}
        for ind in self.population:
            key = ind.code.strip()
            if key not in best_by_code or ind.fitness > best_by_code[key].fitness:
                best_by_code[key] = ind
        self.population = sorted(
            best_by_code.values(), key=lambda ind: ind.fitness, reverse=True
        )[:self.config.population_size]

        self.generation += 1
        current_best = max(self.population, key=lambda x: x.fitness)

        # Update historical best
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best
            print(f"[EA] Found better individual, fitness: {current_best.fitness:.4f}")

        # Print population statistics
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        print(f"[EA] Gen {self.generation}: Best={current_best.fitness:.4f}, Avg={avg_fitness:.4f}")

        # === Student implementation end ===

    def _evaluate_population(self) -> None:
        """
        Evaluate all un-evaluated individuals in the current population

        [Auxiliary method] Students can call as needed
        """
        for ind in self.population:
            if ind.fitness == 0 and not ind.metadata.get('evaluated', False):
                if self._budget_exhausted():
                    break
                timeout = self._evaluation_timeout()
                if timeout <= 0:
                    break
                result: EvaluationResult = self.executor.evaluate(
                    ind.code, timeout=timeout
                )
                evaluation_evidence = getattr(result, 'evidence', {})
                if not isinstance(evaluation_evidence, dict):
                    evaluation_evidence = {}
                ind.fitness = result.fitness
                ind.metadata.update({
                    'evaluated': True,
                    'success': result.success,
                    'speedup': result.speedup,
                    'execution_time': result.execution_time,
                    'error': result.error,
                    'evaluation_evidence': dict(evaluation_evidence),
                    'prompt_context': self._prompt_context_for_result(
                        ind.metadata.get('prompt_context'), result
                    ),
                })

    def run(self, seed_codes: List[str], deadline_seconds: Optional[float] = None) -> Individual:
        """
        [Student Implementation] Run complete evolutionary algorithm

        Implementation requirements:
        1. Call initialize_population() to initialize population
        2. Loop calling evolve_generation() for evolution
        3. Check termination conditions (reach max generations or find satisfactory solution)
        4. Return the best individual found

        Termination conditions:
        - Reach config.max_generations
        - Or find a near-optimal solution (fitness >= 1.9, close to upper limit 2.0)

        Args:
            seed_codes: Initial seed codes

        Returns:
            best_individual: Best individual found
        """
        # === Student implementation start ===

        # Budget: wall clock deadline (competition rule, first-hit wins with token budget)
        self._deadline = time.time() + deadline_seconds if deadline_seconds else None

        # Step 1: Initialize population
        self.initialize_population(seed_codes)

        # Step 2: Evolution loop
        while self.generation < self.config.max_generations and not self._budget_exhausted():
            print(
                f"\n[EA] ===== Generation {self.generation + 1}/"
                f"{self.config.max_generations} ====="
            )
            self.evolve_generation()

            # Early stop check: if a near-optimal solution is found
            if self.best_individual and self.best_individual.fitness >= 1.9:
                print(f"[EA] Reached performance limit, early stop (fitness={self.best_individual.fitness:.4f})")
                break

        # Step 3: Return best individual
        print(f"\n[EA] Evolution complete, best fitness: {self.best_individual.fitness:.4f}")
        return self.best_individual

        # === Student implementation end ===
