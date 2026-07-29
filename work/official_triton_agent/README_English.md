# Triton Auto-Optimization System Based on Evolutionary Algorithm - Participant Guide

## Project Overview

This project is the **2026 National College Student Computer System Capability Competition - Compiler System Challenge** framework, requiring the implementation of a **Triton operator auto-optimization system based on Evolutionary Algorithm (EA)**.

### Core Task
Design an evolutionary algorithm within the given Agent framework to automatically optimize Triton operators, **maximizing computational performance (speedup ratio)** while ensuring functional correctness.

### Competition Scoring Criteria
| Metric | Weight | Description |
|--------|--------|-------------|
| **Functional Test** | 30% | Code compiles and runs correctly, passes test cases |
| **Performance Test** | 70% | Speedup = max(baseline/current - 1, 0), capped at 2.0 (corresponding to 200 points) |

---

## What You Need to Do?

### Core Work: Implement Two Python Files

#### 1. `genetic_operators.py` - Genetic Operators (Key Focus)
Implement **Crossover** and **Mutation** operations, using LLM as the intelligent evolution engine:

```python
class GeneticOperators:
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Crossover: Let LLM fuse the strengths of two parent codes

        Tips:
        - Provide fitness information of both parents in the Prompt
        - Let LLM select better optimization strategies for fusion
        - Support model evolution: can randomly select the model used by parents
        """
        # Your implementation: Build Prompt -> Call LLM -> Return offspring

    def mutate(self, individual: Individual) -> Individual:
        """
        Mutation: Let LLM randomly mutate the code

        Strategy options:
        - param_tuning: Adjust block size, num_stages, etc.
        - strategy_change: Change memory access patterns or parallel strategies
        - local_rewrite: Rewrite specific parts for better performance

        Model evolution: Certain probability of switching to different LLMs for mutation
        """
        # Your implementation: Select mutation type -> Build Prompt -> Call LLM -> Return mutated individual
```

#### 2. `evolutionary_algorithm.py` - Evolutionary Algorithm Main Logic (Key Focus)
Implement a complete EA workflow:

```python
class EvolutionaryAlgorithm:
    def initialize_population(self, seed_codes: List[str]) -> None:
        """
        Population Initialization: Generate initial population from seed codes

        Strategy:
        - Round 1: Seed code + multiple mutations to generate diversity
        - If organizers provide multiple seeds (ops_1.py ~ ops_10.py), load all of them
        """

    def select_parents(self) -> Tuple[Individual, Individual]:
        """
        Selection: Select two parents based on fitness

        Recommended: Roulette Wheel Selection
        - Higher fitness, higher probability of being selected
        - Implementation: Calculate probability distribution -> Random selection based on probability (without replacement)
        """

    def evolve_generation(self) -> None:
        """
        Evolve One Generation: Complete one-generation evolution workflow

        Standard workflow (referencing EvoPrompt paper):
        1. Elite preservation: Directly retain top-k best individuals
        2. Generate new individuals: Selection -> Crossover -> Mutation
        3. Evaluation: Calculate fitness (speedup ratio)
        4. Update: Select next-generation population ((mu+lambda) strategy)
        """

    def run(self, seed_codes: List[str]) -> Individual:
        """
        Run Complete Evolution: Loop multiple generations until convergence

        Termination conditions:
        - Reach max generations (config.max_generations)
        - Or find a satisfactory solution (fitness >= 1.9, close to the upper limit of 2.0)
        """
```

---

## Project Structure

```
Agent/
|-- config.py                    # Configuration file (student-adjustable parameters)
|-- executor.py                  # Triton executor (provided by organizers)
|-- llm_interface.py             # LLM interface (provided by organizers)
|-- genetic_operators.py         # Genetic operators (student implementation)
|-- evolutionary_algorithm.py    # Evolutionary algorithm main logic (student implementation)
|-- optimizer_agent.py           # Unified interface (provided by organizers)
|-- main.py                      # Command-line entry (provided by organizers)
|-- README.md                    # This document
```

### File Division

| File | Function | Who Modifies | Difficulty |
|------|----------|--------------|------------|
| `config.py` | Configuration parameters (population size, model list, etc.) | Student-adjustable parameters | Low |
| `executor.py` | Compile Triton, measure execution time, calculate speedup | **Provided by organizers** | - |
| `llm_interface.py` | Call LLM API, token statistics | **Provided by organizers** | - |
| `genetic_operators.py` | Crossover, mutation operations | **Student implementation** | High |
| `evolutionary_algorithm.py` | Population initialization, selection, evolution loop | **Student implementation** | High |
| `optimizer_agent.py` | Coordinate components, provide unified interface | **Provided by organizers** | - |
| `main.py` | Command-line entry, batch processing operators | **Provided by organizers** | - |

---

## Runtime Logic and Call Relationships

### Overall Architecture

```
+------------------------------------------------------------------+
|                         main.py (Entry)                           |
|  - Parse command-line arguments                                  |
|  - Traverse operator files (support ops.py / ops_1.py ~ ops_10.py)|
|  - Call TritonOptimizerAgent                                     |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                   optimizer_agent.py (Coordinator)               |
|  - setup(): Initialize executor + genetic_ops + ea               |
|  - optimize(): Run complete evolution workflow                   |
|  - save_results(): Save results (best code + Top5 versions)      |
+------------------------------------------------------------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
+------------------+  +------------------+  +------------------+
|  executor.py     |  |genetic_operators |  |evolutionary_    |
|  (Evaluator)     |  |   .py (Genetic   |  |algorithm.py     |
|                  |  |    Operators)    |  | (EA Main Logic)  |
|  - Measure       |  |                  |  |                  |
|    baseline time |  |                  |  |  - Population init|
|  - Compile and   |  |  - crossover()   |  |  - Selection     |
|    run code      |  |    [Student Impl]|  |  - Evolution loop|
|  - Calculate     |  |  - mutate()      |  |  - Population    |
|    speedup       |  |    [Student Impl]|  |    update        |
+------------------+  +------------------+  +------------------+
                              ^                      |
                              |                      |
                              +----------------------+
                              |
                              v
+------------------------------------------------------------------+
|                   llm_interface.py (LLM Interface)               |
|  - generate(): Call LLM to generate code                         |
|  - Support model evolution (switching between multiple models)   |
|  - Print call purpose (mutation/crossover/model switch)          |
+------------------------------------------------------------------+
                              ^
                              |
                              |
+------------------------------------------------------------------+
|                      config.py (Configuration)                   |
|  - EAConfig: Population size, generations, crossover rate, etc.  |
|  - llm_models: Available LLM list (support model evolution)     |
+------------------------------------------------------------------+
```

### Core Data Flow

```
1. Initialization Phase
   main.py -> optimizer_agent.setup()
                |-- executor.__init__() -> Measure baseline time
                |-- GeneticOperators.__init__()
                |-- EvolutionaryAlgorithm.__init__()

2. Evolution Loop (per generation)
   EA.run() -> evolve_generation()
                 |-- select_parents() -> Roulette wheel selection
                 |-- genetic_ops.crossover() -> LLM fuses two parents
                 |       |-- llm.generate(purpose='crossover')
                 |-- genetic_ops.mutate() -> LLM mutation (may switch model)
                 |       |-- llm.generate(purpose='mutate')
                 |-- executor.evaluate() -> Timing + Calculate speedup

3. Result Output
   optimizer_agent -> save_results()
                         |-- Best code (best_code.py)
                         |-- Top5 codes (v1~v5.py)
                         |-- Statistics (stats.json)
```

---

## Quick Start

### 1. Environment Setup

```bash
# Download and install CANN packages:
Ascend-cann-kernels-910b_8.3.RC1.alpha003_linux-aarch64.run
Ascend-cann-nnal_8.3.RC1.alpha003_linux-aarch64.run
Ascend-cann-toolkit_8.3.RC1.alpha003_linux-aarch64.run
Source the corresponding set_env.sh as required

# Download and install npuir:
ascendnpu-ir_1.0.0_linux-aarch64.run
Source the corresponding set_env.sh as required

# Extract and create conda environment:
TriTrans.tar.gz

# Set API key (SiliconFlow or other platform)
export API_KEY="your-api-key"
export API_URL="https://api.siliconflow.cn/v1"
```

### 2. Modify Configuration (config.py)

```python
# config.py - Student focus modification area

@dataclass
class EAConfig:
    # Evolutionary algorithm parameters
    population_size: int = 10          # Population size
    max_generations: int = 20          # Max evolution generations
    crossover_rate: float = 0.8        # Crossover probability
    mutation_rate: float = 0.3         # Mutation probability
    elite_ratio: float = 0.2           # Elite preservation ratio

    # ==================== Key: LLM Configuration ====================
    # Students freely add/remove/modify this list to implement model evolution
    # The first model will be used as the initial default model
    llm_models: List[str] = field(default_factory=lambda: [
        'deepseek-v3',
        # 'deepseek-v3.1',
        # 'deepseek-r1',
        # 'qwen3',
        # 'qwen3-coder',
        # 'kimi-k2-instruct',
        # 'glm-4.5'
    ])

    # Model evolution parameters
    model_switch_prob: float = 0.2       # Probability of switching model during mutation (20%)
```

### 3. Prepare Data

Input folder structure (supports multiple ways):

```
datasets/
|-- test_op/
    |-- test_op.py              # Main code (required)
    |-- test_op_1.py            # Variant 1 (optional)
    |-- test_op_2.py            # Variant 2 (optional)
    |-- ...                     # More variants (optional)
    |-- variants/               # Variant subfolder (optional)
        |-- variant_a.py
        |-- variant_b.py
```

### 4. Run Optimization

```bash
# Optimize all operators in the folder
python main.py --input-dir ./datasets/test --output-dir ./output

# Optimize a specific operator
python main.py --input-dir ./datasets/test --output-dir ./output --kernel test_op

# Use custom evolution parameters (override config values)
python main.py --input-dir ./datasets/test --output-dir ./output \
               --population-size 15 --max-generations 30

# Enable debug mode
python main.py --input-dir ./datasets/test --output-dir ./output --debug
```

---

## Key Design Points

### 1. How to Handle Crossover in the First Round?

**Problem**: Initially there is only one seed code, but 10 individuals need to be generated. How to crossover?

**Solution**:
- **First round**: Use **mutation** entirely to generate diversity
  ```
  Initial: [baseline]
  Fill: mutate(baseline) -> Individual 2
        mutate(baseline) -> Individual 3
        ...
        mutate(baseline) -> Individual 10
  ```
- **From second round**: With fitness differences, **crossover** can be used
  ```
  Select: Individual 3 (fitness=0.5) + Individual 7 (fitness=0.3)
  Crossover: crossover(Individual 3, Individual 7) -> Offspring 11
  ```

**You can also create the initial population through mutation yourself**

### 2. What is Model Evolution?

**Traditional EA**: Only evolve code (genes are code strings)

**This framework supports**: Simultaneously evolve **code + LLM**

```python
# Configure multiple models in config.py
llm_models = ['deepseek-v3', 'qwen3', 'kimi-k2-instruct']

# During mutation, there is a certain probability of switching models
if random.random() < 0.2:  # 20% probability
    new_model = random.choice(other_models)
    llm.switch_model(new_model)  # Switch model
    # Use the new model for mutation

# During crossover, can select the parent's model
selected_model = random.choice([parent1.model_used, parent2.model_used])
llm.switch_model(selected_model)
```

**Advantage**: Different models have different strengths, automatically select the most suitable model for the current task.

---

## Reference Resources

### EvoPrompt Paper Core Idea
> "Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers"
```
https://arxiv.org/html/2309.08532
```

- **Core innovation**: Use LLM as evolution operators, rather than traditional bit-wise operations
- **Key insight**: LLM understands code semantics, enabling intelligent crossover and mutation
- **Algorithm flow**: Initialization -> Roulette wheel selection -> LLM crossover -> LLM mutation -> Evaluation -> Update

### Speedup Calculation
```
speedup = max(baseline_time / current_time - 1, 0)

Examples:
- Baseline 100ms, optimized 50ms: speedup = 100/50 - 1 = 1.0 -> 100 points
- Baseline 100ms, optimized 25ms: speedup = 100/25 - 1 = 3.0 -> Capped at 2.0 -> 200 points (upper limit)
- Optimized slower than baseline: speedup = max(negative, 0) = 0 -> 0 points
```

---

## Important Notes

1. **Do not heavily modify organizer-provided files**: `executor.py`, `llm_interface.py`, `optimizer_agent.py`, `main.py`
2. **Ensure code is runnable**: Code generated by genetic operators must be valid Python + Triton
3. **Control token consumption**: Each LLM call consumes tokens, budget is limited, prompt design should be efficient
4. **Functional correctness first**: No matter how high the speedup ratio is, if functional tests fail, it is 0 points

---

## Learning Suggestions

1. **Run baseline first**: Without modifying any code, run once to see the output
2. **Understand data flow**: Track the complete flow of an operator from input to output
3. **Start simple**: First implement basic roulette wheel selection and simple mutation
4. **Iterate and optimize**: Observe log output, gradually improve prompts and strategies
5. **Reference EvoPrompt**: Prompt design in the paper is key

---

## Feedback

For technical questions, please contact the organizers through the competition platform.

**Wish you success in the competition!**
