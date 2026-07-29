#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module - Evolutionary Algorithm Parameter Configuration

Students configure all parameters by modifying this file, including:
- Evolutionary algorithm parameters (population size, generations, etc.)
- Available large model list (supports model evolution)
- Baseline JSON file path
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EAConfig:
    """
    Evolutionary Algorithm Configuration Parameters

    Modification instructions:
    1. Directly modify the default values below
    2. Some parameters (population_size, max_generations, debug) can be overridden via command line
    3. Large model list llm_models can only be modified in this file, not supported via command line
    """

    # ==================== Evolutionary Algorithm Parameters ====================
    # Production defaults. Evolution is budget-driven (20 min / 200k tokens per
    # operator); max_generations is only a safety cap to prevent an unbounded loop
    # if the budget somehow never trips. For a smoke run pass
    # --population-size 2 --max-generations 0 on the command line.
    population_size: int = 10         # Candidates alive per generation
    max_generations: int = 50         # Safety cap; real stop is the budget wall

    # Genetic operation parameters
    crossover_rate: float = 0.8          # Crossover probability
    mutation_rate: float = 0.3          # Mutation probability
    elite_ratio: float = 0.2             # Elite preservation ratio

    # ==================== Large Model Configuration (Key) ====================
    # Students freely add/remove/modify this list to implement model evolution
    # The first model will be used as the initial default model
    llm_models: List[str] = field(default_factory=lambda: [
        os.getenv("ENGINE", "deepseek-v4-pro")
    ])

    # LLM generation parameters
    llm_temperature: float = 0.7         # Generation temperature
    max_llm_tokens: int = 4096           # Max tokens per call
    max_total_tokens: int = 200_000      # Total token budget per operator (competition rule)

    # Model evolution parameters
    model_switch_prob: float = 0.2         # Probability of switching model during mutation

    # ==================== Evaluation Parameters ====================
    baseline_json: str = "./baseline/baseline.json"  # Baseline data JSON file path
    timeout_seconds: int = 300             # Compilation/execution timeout
    max_iterations: int = 5                # Max fix attempts per operator

    # ==================== Debug Parameters ====================
    debug: bool = False                    # Whether to enable debug mode

    # ==================== API Configuration ====================
    # Injected by evaluation system, students do not need to modify
    api_url: Optional[str] = None
    api_key: Optional[str] = None

    def print_config(self):
        """Print current configuration"""
        print(f"[Config] Evolutionary Algorithm Configuration:")
        print(f"         Population size: {self.population_size}")
        print(f"         Max generations: {self.max_generations}")
        print(f"         Crossover rate: {self.crossover_rate}")
        print(f"         Mutation rate: {self.mutation_rate}")
        print(f"         Elite ratio: {self.elite_ratio}")
        print(f"[Config] Large Model Configuration:")
        print(f"         Available models: {', '.join(self.llm_models)}")
        print(f"         Default model: {self.llm_models[0] if self.llm_models else 'None'}")
        print(f"         Model switch probability: {self.model_switch_prob}")
        print(f"[Config] Baseline Configuration:")
        print(f"         Baseline JSON: {self.baseline_json}")
        print(f"[Config] LLM Parameters:")
        print(f"         Temperature: {self.llm_temperature}")
        print(f"         Max tokens: {self.max_llm_tokens}")
