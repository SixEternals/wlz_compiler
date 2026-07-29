# 基于进化算法的 Triton 自动优化系统 - 参赛指南

## 📋 项目简介

本项目是 **2026全国大学生计算机系统能力大赛 - 编译系统挑战赛** 的参赛框架，要求实现基于 **进化算法（Evolutionary Algorithm, EA）** 的 Triton 算子自动优化系统。

### 核心任务
在给定的 Agent 框架中设计进化算法，自动优化 Triton 算子，在确保功能正确的前提下，**最大化计算性能（加速比）**。

### 竞赛评分标准
| 指标 | 权重 | 说明 |
|------|------|------|
| **功能测试** | 30% | 代码正确编译运行，通过测试用例 |
| **性能测试** | 70% | 加速比 = max(baseline/current - 1, 0)，上限 2.0（对应 200 分）|

---

## 🎯 你需要做什么？

### 核心工作：实现两个 Python 文件

#### 1. `genetic_operators.py` - 遗传算子（⭐ 重点）
实现 **交叉（Crossover）** 和 **变异（Mutation）** 两个操作，使用 LLM 作为"智能进化引擎"：

```python
class GeneticOperators:
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        🧬 交叉操作：让 LLM 融合两个父代代码的优点
        
        提示：
        - 在 Prompt 中提供两个父代的适应度信息
        - 让 LLM 选择更好的优化策略进行融合
        - 支持模型进化：可随机选择父代使用的模型
        """
        # 你的实现：构建 Prompt → 调用 LLM → 返回子代
        
    def mutate(self, individual: Individual) -> Individual:
        """
        🦠 变异操作：让 LLM 对代码进行随机变异
        
        策略可选：
        - param_tuning: 调整 block size、num_stages 等参数
        - strategy_change: 改变内存访问模式或并行策略  
        - local_rewrite: 重写特定部分以获得更好性能
        
        模型进化：有一定概率切换不同的大模型进行变异
        """
        # 你的实现：选择变异类型 → 构建 Prompt → 调用 LLM → 返回变异个体
```

#### 2. `evolutionary_algorithm.py` - 进化算法主逻辑（⭐ 重点）
实现完整的 EA 流程：

```python
class EvolutionaryAlgorithm:
    def initialize_population(self, seed_codes: List[str]) -> None:
        """
        🌱 种群初始化：用种子代码生成初始种群
        
        策略：
        - 第一轮：种子代码 + 多次变异生成多样性
        - 如果主办方提供多个种子（ops_1.py ~ ops_10.py），全部加载
        """
        
    def select_parents(self) -> Tuple[Individual, Individual]:
        """
        👫 选择操作：根据适应度选择两个父代
        
        推荐：轮盘赌选择（Roulette Wheel Selection）
        - 适应度越高，被选中的概率越大
        - 实现：计算概率分布 → 根据概率随机选择（不放回）
        """
        
    def evolve_generation(self) -> None:
        """
        🔄 进化一代：完整的一代进化流程
        
        标准流程（参考 EvoPrompt 论文）：
        1. 精英保留：直接保留 top-k 最优个体
        2. 生成新个体：选择 → 交叉 → 变异
        3. 评估：计算适应度（加速比）
        4. 更新：选择下一代种群（(μ+λ) 策略）
        """
        
    def run(self, seed_codes: List[str]) -> Individual:
        """
        🏆 运行完整进化：循环多代直到收敛
        
        终止条件：
        - 达到最大代数（config.max_generations）
        - 或找到满意解（fitness >= 1.9，接近上限 2.0）
        """
```

---

## 📁 项目结构

```
Agent/
├── config.py                    # ⚙️ 配置文件（学生可调参数）
├── executor.py                  # 🔧 Triton 执行器（主办方提供）
├── llm_interface.py             # 🤖 LLM 接口（主办方提供）
├── genetic_operators.py         # 🧬 遗传算子（⭐ 学生实现）
├── evolutionary_algorithm.py    # 🔄 进化算法主逻辑（⭐ 学生实现）
├── optimizer_agent.py           # 🎯 统一接口（主办方提供）
├── main.py                      # 🚀 命令行入口（主办方提供）
└── README.md                    # 📖 本文档
```

### 文件分工

| 文件 | 功能 | 谁修改 | 难度 |
|------|------|--------|------|
| `config.py` | 配置参数（种群大小、模型列表等） | 学生可调参数 | ⭐ |
| `executor.py` | 编译 Triton、测量执行时间、计算加速比 | **主办方提供** | - |
| `llm_interface.py` | 调用大模型 API、Token 统计 | **主办方提供** | - |
| `genetic_operators.py` | 交叉、变异操作 | **学生实现** | ⭐⭐⭐ |
| `evolutionary_algorithm.py` | 种群初始化、选择、进化循环 | **学生实现** | ⭐⭐⭐ |
| `optimizer_agent.py` | 协调各组件，提供统一接口 | **主办方提供** | - |
| `main.py` | 命令行入口，批量处理算子 | **主办方提供** | - |

---

## 🔄 运行逻辑与调用关系

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (入口)                           │
│  - 解析命令行参数                                                 │
│  - 遍历算子文件（支持 ops.py / ops_1.py ~ ops_10.py）              │
│  - 调用 TritonOptimizerAgent                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   optimizer_agent.py (协调器)                     │
│  - setup(): 初始化 executor + genetic_ops + ea                   │
│  - optimize(): 运行完整进化流程                                   │
│  - save_results(): 保存结果（最优代码 + Top5 版本）                │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  executor.py    │  │genetic_operators │  │evolutionary_   │
│  (评测执行器)    │  │   .py (遗传算子)  │  │algorithm.py    │
│                 │  │                  │  │ (进化算法主逻辑) │
│ • 🕐 测量基线时间 │  │                  │  │                 │
│ • ⚙️ 编译运行代码 │  │ • 🧬 crossover() │  │ • 🌱 种群初始化  │
│ • 📊 计算加速比  │◄─┤   【学生实现】    │◄─┤ • 👫 选择机制   │
│   speedup=      │  │ • 🦠 mutate()    │  │ • 🔄 进化循环    │
│   max(b/c-1,0)  │  │   【学生实现】    │  │ • 💾 种群更新    │
└─────────────────┘  └──────────────────┘  └──────────────────┘
                              ▲                      │
                              │                      │
                              └──────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   llm_interface.py (LLM接口)                      │
│  - generate(): 调用大模型生成代码                                  │
│  - 支持模型进化（多个模型间切换）                                   │
│  - 打印调用目的（变异/交叉/模型切换）                               │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      config.py (配置)                            │
│  - EAConfig: 种群大小、代数、交叉率、变异率等                      │
│  - llm_models: 可用大模型列表（支持模型进化）                      │
└─────────────────────────────────────────────────────────────────┘
```

### 核心数据流

```
1️⃣ 初始化阶段
   main.py ──► optimizer_agent.setup()
                ├──► executor.__init__() ──► 🕐 测量基线时间
                ├──► GeneticOperators.__init__()
                └──► EvolutionaryAlgorithm.__init__()

2️⃣ 进化循环（每代）
   EA.run() ──► evolve_generation()
                 ├──► select_parents() ──► 👫 轮盘赌选择
                 ├──► genetic_ops.crossover() ──► 🧬 LLM融合两个父代
                 │       └──► llm.generate(purpose='crossover')
                 ├──► genetic_ops.mutate() ──► 🦠 LLM变异（可能切换模型）
                 │       └──► llm.generate(purpose='mutate')
                 └──► executor.evaluate() ──► 🕐 测速 + 📊 算加速比

3️⃣ 结果输出
   optimizer_agent ──► save_results()
                         ├── 🏆 最优代码 (best_code.py)
                         ├── 📁 Top5 代码 (v1~v5.py)
                         └── 📊 统计信息 (stats.json)
```

---

## 🚀 快速开始

### 1. 配置环境

```bash

#下载并安装CANN包：
Ascend-cann-kernels-910b_8.3.RC1.alpha003_linux-aarch64.run
Ascend-cann-nnal_8.3.RC1.alpha003_linux-aarch64.run
Ascend-cann-toolkit_8.3.RC1.alpha003_linux-aarch64.run

按要求source对应的set_env.sh

#下载并安装npuir：
ascendnpu-ir_1.0.0_linux-aarch64.run
按要求source对应的set_env.sh

#解压并创建conda环境：
TriTrans.tar.gz

# 设置 API 密钥（SiliconFlow 或其他平台）
export API_KEY="your-api-key"
export API_URL="https://api.siliconflow.cn/v1"

```

### 2. 修改配置（config.py）

```python
# config.py - 学生重点修改此处

@dataclass
class EAConfig:
    # 进化算法参数
    population_size: int = 10          # 种群大小
    max_generations: int = 20          # 最大进化代数
    crossover_rate: float = 0.8        # 交叉概率
    mutation_rate: float = 0.3         # 变异概率
    elite_ratio: float = 0.2           # 精英保留比例
    
    # ==================== 关键：大模型配置 ====================
    # 学生自由添加/删除/修改此列表，实现模型进化
    # 第一个模型将作为初始默认模型
    llm_models: List[str] = field(default_factory=lambda: [
        'deepseek-v3',
        # 'deepseek-v3.1',
        # 'deepseek-r1',
        # 'qwen3',
        # 'qwen3-coder',
        # 'kimi-k2-instruct',
        # 'glm-4.5'
    ])
    
    # 模型进化参数
    model_switch_prob: float = 0.2       # 变异时切换模型的概率（20%）
```

### 3. 准备数据

输入文件夹结构（支持多种方式）：

```
datasets/
└── test_op/
    ├── test_op.py              # 主代码（必须）
    ├── test_op_1.py            # 变体 1（可选）
    ├── test_op_2.py            # 变体 2（可选）
    ├── ...                     # 更多变体（可选）
    └── variants/               # 变体子文件夹（可选）
        ├── variant_a.py
        └── variant_b.py
```

### 4. 运行优化

```bash
# 优化整个文件夹的所有算子
python main.py --input-dir ./datasets/test --output-dir ./output

# 优化指定算子
python main.py --input-dir ./datasets/test --output-dir ./output --kernel test_op

# 使用自定义进化参数（覆盖 config 中的值）
python main.py --input-dir ./datasets/test --output-dir ./output \
               --population-size 15 --max-generations 30

# 启用调试模式
python main.py --input-dir ./datasets/test --output-dir ./output --debug
```

---

## 💡 关键设计要点

### 1. 第一轮交叉如何处理？

**问题**：初始只有一个种子代码，但需要生成 10 个个体，怎么交叉？

**解决方案**：
- **第一轮**：全部用 **变异** 生成多样性
  ```
  初始: [baseline]
  填充: mutate(baseline) → 个体 2
        mutate(baseline) → 个体 3
        ...
        mutate(baseline) → 个体 10
  ```
- **第二轮起**：有了适应度差异，可以 **交叉**
  ```
  选择: 个体 3 (fitness=0.5) + 个体 7 (fitness=0.3)
  交叉: crossover(个体 3, 个体 7) → 子代 11
  ```

**也可以通过变异自己创建初始种群**

### 2. 模型进化是什么？

**传统 EA**：只进化代码（基因是代码字符串）

**本框架支持**：同时进化 **代码 + 大模型**

```python
# 在 config.py 中配置多个模型
llm_models = ['deepseek-v3', 'qwen3', 'kimi-k2-instruct']

# 在变异时，有一定概率切换模型
if random.random() < 0.2:  # 20% 概率
    new_model = random.choice(other_models)
    llm.switch_model(new_model)  # 🔄 切换模型
    # 用新模型进行变异

# 在交叉时，可以选择父代的模型
selected_model = random.choice([parent1.model_used, parent2.model_used])
llm.switch_model(selected_model)
```

**优势**：不同模型有不同特长，自动选择最适合当前任务的模型。

---

## 📚 参考资源

### EvoPrompt 论文核心思想
> "Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt Optimizers"
```
https://arxiv.org/html/2309.08532
```

- **核心创新**：用 LLM 作为进化算子，而非传统 bit-wise 操作
- **关键洞察**：LLM 理解代码语义，能进行"智能"交叉和变异
- **算法流程**：初始化 → 轮盘赌选择 → LLM 交叉 → LLM 变异 → 评估 → 更新

### 加速比计算
```
speedup = max(baseline_time / current_time - 1, 0)

示例：
- 基线 100ms，优化后 50ms：speedup = 100/50 - 1 = 1.0 → 100 分
- 基线 100ms，优化后 25ms：speedup = 100/25 - 1 = 3.0 → 限制为 2.0 → 200 分（上限）
- 优化后比基线慢：speedup = max(负数, 0) = 0 → 0 分
```

---

## ⚠️ 注意事项

1. **不要大幅修改主办方提供的文件**：`executor.py`, `llm_interface.py`, `optimizer_agent.py`, `main.py`
2. **确保代码可运行**：遗传算子生成的代码必须是合法 Python + Triton
3. **控制 Token 消耗**：每次 LLM 调用消耗 Token，预算有限，Prompt 设计要高效
4. **功能正确优先**：加速比再高，功能测试不通过也是 0 分

---

## 🎓 学习建议

1. **先跑通 baseline**：不修改任何代码，先运行一遍看输出
2. **理解数据流**：跟踪一个算子从输入到输出的完整流程
3. **从简单开始**：先实现基本的轮盘赌选择和简单变异
4. **迭代优化**：观察日志输出，逐步改进 Prompt 和策略
5. **参考 EvoPrompt**：论文中的 Prompt 设计是关键

---

## 📞 问题反馈

如有技术问题，请通过竞赛平台联系主办方。

**祝你比赛顺利！🏆**