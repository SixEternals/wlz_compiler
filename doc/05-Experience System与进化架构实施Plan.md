# 05-Experience System 与进化架构实施 Plan

日期：2026-08-04

本文是给执行方（Codex）的实施计划，涵盖 Experience System 的分阶段落地路线。
架构方向经过多轮讨论已收敛，本文只写结论与可执行单元，不重复论证过程。

前置文档：

- 赛题与记分口径：[`2026编译系统设计赛大赛要求综合汇总.md`](2026编译系统设计赛大赛要求综合汇总.md)
- 当前算子状态：[`02-最近成果-本机910B4-21算子资格闭环.md`](02-最近成果-本机910B4-21算子资格闭环.md)
- EvoSci 论文导读：[`learn/01-EvoSci科学发现进化框架导读.md`](learn/01-EvoSci科学发现进化框架导读.md)
- 上一版执行 Plan：[`04-优化升级Plan-给执行方.md`](04-优化升级Plan-给执行方.md)

## 1. 一句话结论

经验不是文本记忆，而是搜索空间结构的改变；在改变搜索空间之前，必须先确认
选择压力（fitness）用的是官方口径。

## 2. 架构原则

三条，按优先级排列。经验进入系统前先自上而下问一遍。

1. **能编译成确定性规则的经验，写成 gate，不进检索层。**
   零 token、永久有效、不依赖 LLM 判断力。T3 的 `timing_surface_moved`
   就是 pack_seq 教训的编译版，这是范式。
2. **不能写成 gate 的，改采样分布，不进 prompt。**
   分布是硬的且免费；prompt 是软的、每次付费、且 LLM 可以忽略。
3. **改分布之前先确认 fitness 是官方口径，且保留探索下界。**
   前者防止学到错的东西，后者防止过早收敛。

## 3. 明确不做

| 不做 | 理由 |
| --- | --- |
| Optimization Knowledge Graph | 21 个算子、4 类可迁移策略的规模下，一跳查表与多跳图遍历返回同样结果，但要多付 schema、一致性、相似度阈值的成本。且论文未给出图谱更新的可复现 schema（导读 §4.2、§7.2）。 |
| Experience Agent | 检索是数据库查询 + ranking，不是智能任务，不需要 LLM。 |
| Planner / Strategy Agent | 与"同预算多生成候选"抢每算子 20 万 token。总计划 §7.1 要求等预算对照，该实验尚未做。 |
| Multi-Agent Evolution | 论文自身团队规模实验在 `team_size=5` 达峰，7 和 9 均下降（导读 §6.2），是"Agent 越多越好"的反证。 |

替代方案：JSONL archive + 复合 key
`(operator_family, memory_pattern, dtype, executor_kind, mutation_type)`。
若将来该 key 上的查询确实不够用再升级，届时应由数据提出要求，而非论文。

## 4. 规模前提（重要，此前讨论的错误假设）

前几轮讨论共享一个未经验证的前提："EA 每代淘汰大量候选，信息损失严重"。
实际情况不同。

`work/official_triton_agent/config.py:31-35`：

- `population_size = 10`、`max_generations = 50` 都是**安全上限，不是目标**
- 注释明确写 `real stop is the budget wall`，真正终止条件是 20 万 token 耗尽
- 注释举例为 `--population-size 2 --max-generations 0`

结论：单算子实际跑不了几代，`(mu+lambda)` 截断丢弃的候选是**十几个量级，
不是几百个**。这不改变"该做 archive"的结论，但改变三件事：

1. Phase 1 的收益理由从"抢救大量已付费评估"改为"**跨算子累积**"。
2. Phase 2.5 的在线学习不可行，改为离线人工权重（见 §8）。
3. Phase 1.5 优先级上升：样本少时每个样本的标注质量就是一切。

## 5. 执行顺序

顺序不可调整，理由见各阶段的依赖说明。

```text
Phase 0    Copy Page -> T6 四算子 -> T4 打包        <- 最高优先，确定收益
   |
   v
Phase 1    evaluated archive（验零侵入）
   |
   v
Phase 1.5  fitness 校准
   |
   v
Phase 2    typed KernelQuery
   |
   v
Phase 2.5  离线人工采样权重
```

**关键顺序约束：typed KernelQuery 必须排在 archive 之后，不能并行。**
因为 typed context 会改 prompt，从而改变 candidate sequence；而 Phase 1 的
验收标准是"archive 前后 candidate sequence 逐个一致"。两件事同时做，
零侵入就无法验证。

## 6. Phase 0：修功能分（当前，最高优先）

不变，沿用 [`04-优化升级Plan-给执行方.md`](04-优化升级Plan-给执行方.md)。

剩余工作：

| 项 | 状态 | 说明 |
| --- | --- | --- |
| Copy Page | 未修 | T5 里唯一需要控制流重写的。`range()` 接运行时标量与 `tl.arange` 上界非 constexpr 要一起解。建议单独一轮，不与 T6 混在同一提交。 |
| T6 四算子 | 未开始 | 依赖 T1 官方精度门（已完成）。 |
| Count Expert admission | 待接 | 需要本机 Ascend admission 分支，约束见 §6.1。 |
| T4 打包 | 未开始 | 依赖以上全部。打包后停下等批准，不得自动提交。 |

优先级理由：功能分是 0/100 二值，无部分分，且功能失败的算子性能分同样为 0。
8 个算子的 0 分是**确定的失血**；Experience System 的收益是**不确定的收入**。
先止血，再进补。

### 6.1 本机 Ascend admission 分支约束

现状：`scripts/build_official_agent_batch_smoke.py:85` 的策略 ID 写死
`local_cuda_proxy_only_not_ascend_or_official`，且要求 `llm_stats.call_count == 1`。
真实 Ascend 手工证据既无 CUDA policy 也无 LLM usage，必然被拒。

新增分支必须满足（比 CUDA 分支更严，不是更松）：

1. 独立的第三条策略 ID，如 `local_ascend_910b4_manual_evidence_v1`。
   不得放宽现有 CUDA 分支的条件。
2. 不得绕过 `_holdout_admission_error`，必须同时通过 holdout 校验。
3. 强制校验 artifact SHA、`evidence_scope` 前缀必须为 `local_ascend_910b4_`、
   holdout 三字段（`split` / `used_for_search` / `used_in_prompt`）齐全。

测试须证明三种坏输入被拒：缺 artifact SHA、`evidence_scope` 冒充官方、
holdout 字段缺失。

## 7. Phase 1：evaluated candidate archive

### 输入

EA 每次 evaluation 的结果。

### 漏点

`work/official_triton_agent/evolutionary_algorithm.py:369-380`：

```text
old_population + offspring
        |
        v
   全部 evaluation
        |
        v
sorted(...)[:population_size]   <- 截断，被淘汰个体直接丢弃
```

被截断的个体**已经付过完整评估代价**（编译 + correctness + benchmark），
然后被丢弃。这是最贵的一种遗忘。

### 输出

`archive/evaluated_candidates.jsonl`，append-only。每条记录字段：

```text
operator
candidate_hash
parent_hash
mutation_type

correctness_result
failure_category

fitness
raw_speedup
ratio
baseline_time          <- 原始值，必须存
current_time           <- 原始值，必须存

shape
dtype
hardware_fingerprint
executor_kind          <- 必须存
input_set_id           <- 必须存
seed
case_signature         <- 必须存

timestamp
```

### 字段必要性说明

三组字段容易被当成可选，实际缺一不可。

**`input_set_id` / `case_signature` / `executor_kind`。**
Selective Scan 的同一个 candidate 在 visible case 上 ratio `0.9747`、在 holdout
上 `1.05516`。缺这三个字段，两条记录长得一模一样，库里出现自相矛盾条目且
无法归因，**条件经验会退化成无条件经验**——正是要防的那种过拟合。
`executor_kind` 取值为
`local_ascend_910b4_manual` / `cuda_proxy` / `official_a2_a3`，
因为三种证据来源的 ratio 不可直接比较。

**`raw_speedup` 与 `ratio`（不能只存 `fitness`）。**
`work/official_triton_agent/executor.py:335-345`：

```python
raw_speedup = self.baseline_time / current_time - 1
speedup = max(raw_speedup, 0.0)
fitness = min(speedup, 2.0)
```

`max(raw, 0.0)` 把所有回归候选压成同一个 0，ratio `1.0069` 与 `1.53` 在 fitness 上
完全等价。这个压缩对官方记分是正确的，但对后续分析是致命的：dequant 那四个
探针（`1.096` / `1.042` / `1.064` / `1.043`）fitness 全为 0，只有 ratio 分布能
说明那个方向是**系统性失败而非随机噪声**。

**`baseline_time` / `current_time` 原始值。**
`raw_speedup` 已经是比值，跨算子比较时无法从比值反推该算子本身是 3 微秒还是
3 毫秒。而这个绝对量级决定 launch overhead 占比，正是解释"为什么 `num_warps`
在这个算子上没用"的关键变量。只存比值等于丢掉归因能力。

### 收益

停止丢弃已付费的评估数据。注意收益的真实来源是**跨算子累积**：单算子受
20 万 token 预算限制跑不了几代（见 §4），21 个算子横向攒起来才有样本量。
因此跨算子可比性字段是核心而非附属。

### 成本

单个提交量级。

### 风险

低。但有一条硬要求见下。

### 验收标准

两条都要满足：

1. 跑一轮 EA 后，能从 archive 回答：试过哪些方向、哪类 mutation 失败最多、
   哪类收益最高、哪些应禁止。
2. **加 archive 前后 candidate sequence 逐个一致。**
   这是硬要求，archive 不得改变搜索行为。

## 8. Phase 1.5：fitness 校准

优先级在 Phase 2 之前，且因 §4 的规模前提而**上升**：样本少时每个样本的
标注质量就是一切。fitness 算错，十几个样本全废；fitness 算对，十几个就够
支撑"降 `num_warps` 权重"这类粗粒度判断。

### 唯一落点

`work/official_triton_agent/executor.py:335-345`。

### 改动一：硬否决，不是加权求和

正确形式是先否决再排序：

```python
if not correctness_passed:      fitness = 0.0   # 硬否决
elif gate_error is not None:    fitness = 0.0   # T3 门禁，硬否决
else:                           fitness = f(worst_case_ratio) - risk_penalty
```

**不要**用 `performance + correctness + generalization - risk` 这种加权求和：
correctness 是**否决项**而非加分项（官方 0/100 二值），加权和会让"正确性失败
但性能极好"的候选拿到高分，正是要防的情况。

现状说明：correctness 硬否决**已经存在**于 `executor.py:330` 的 `speedup=0.0`
分支。补 T3 gate 否决时须与该分支**同构**，不要另开一套 fitness 路径。

### 改动二：worst case 聚合

现状 `current_time` 是单次测量，没有 worst-case 概念。所以"fitness 用
`worst_case_ratio`"**不是改一行 `max`**，需要先在上游把多 case 结果聚合：

```text
cases: [case1_ratio, case2_ratio, case3_ratio]
        |
        v
   worst_case_ratio
        |
        v
     fitness
```

这是本阶段真实工作量所在，不要按"改一行"估。

依据：Selective Scan `0.9747 -> 1.05516` 就是用平均或单 case 会犯的错。
State Passing 那轮报 worst case `0.92058` 而非更好看的 visible `0.8558`，
做法正确，应固化为规范。

用最差 shape 排序后，generalization 已内建在 fitness 里，**不需要单独一项**。

### 不要做

`historical_failure_penalty` 不要进 fitness，应进 mutation policy（§10）。
理由：fitness 判断"这个候选好不好"，历史失败判断"该往哪找"，混在一起会让
同一候选在不同轮次得到不同分数，破坏可复现性与 archive 的可比性。

## 9. Phase 2：typed KernelQuery

必须等 Phase 1 零侵入验证完成之后再做（见 §5 顺序约束）。

### 依据

这是 EvoSci 全篇证据最强的一条。导读 §6.2：加入 Problem Guidance 后
Novelty `+0.56`，而整个进化模块只带来 Overall `+0.044`（且 10 个主题里 4-5 个
下降、无显著性检验，见导读 §6.3）。**结构化上下文的收益是进化机制的十倍以上。**

### 输入

当前算子的静态与运行时特征。

### 输出

typed `KernelQuery`，替代自由文本历史：

```text
{
  operator_family,
  memory_pattern,
  shape,
  dtype,
  layout,
  executor_kind,
  hardware,
  launch_characteristic
}
```

目的是让模型明确"当前优化的问题属于什么类型"，而不是塞入大量历史经验文本。

### 收益

候选生成质量提升，不增加 Agent、不显著增加 token。

### 风险

会改变 candidate sequence，因此必须在 Phase 1 验收完成后进行。

## 10. Phase 2.5：离线人工采样权重

**注意：不是"mutation policy learning"（在线学习）。**

### 为什么不做在线学习

在线学习采样分布需要足够样本才能收敛。受 §4 的预算前提约束，每算子只有
十几次评估，**在这个样本量下加权分布与均匀分布的差别会被噪声淹没**，学不出来。

### 可行做法

人工从跨算子 archive 读出规律，手工写进采样权重。比让系统自学简单得多，
也诚实得多。

### 钩子已存在

`work/official_triton_agent/genetic_operators.py:238-242`：

```python
base_type, reason = self._base_choice(failure_category_counts)
if rng.random() < self.exploration_rate:      # OPERATOR_EXPLORATION_RATE = 0.2
    ...
    rng.choice(alternatives), "fixed_exploration", True
```

现在是 20% 概率**均匀**选 alternatives。要做的只是把均匀换成加权，
**不需要任何新架构**。

### 第一条权重规则（数据已足够支撑）

按 21 个算子的候选类型分组统计：

| 改动类型 | 样本 | ratio 区间 | 官方分数 | 结论 |
| --- | --- | --- | --- | --- |
| `num_warps` / `num_stages` 单变量 | 7 个候选 | 0.9397 - 1.0000 | 0 - 7 分，无一例外 | 期望收益接近零，**降权** |
| `BLOCK_SIZE` 类 | 多个 | 0.5814 - 1.079 | 0 - 72 分 | 期望收益高但**方差极大**，升权且必须逐 shape 验证 |

`BLOCK_SIZE` 方差的具体证据：log_softmax 的 2048 得 `0.5814`（72 分）、
state_passing 的 32 得 `0.8558`（17 分），但 RMS 的 512 是 `1.0069`（0 分）、
count 的 2048 是 `1.079`（回归）。所以规则是"值得试但必须逐 shape 验证"，
**不是**"BLOCK_SIZE 越大越好"。

这条经验本该在第三个算子之后就成立，实际被重新发现了至少十二次
（7 次并行度调参 + dequant 四次失败探针 + count 的 2048）。这些搜索预算是
真实损耗，也是本阶段的收益来源。

### 两条硬约束

1. **必须做等预算 A/B。** 对照组是"同样 token 用于多生成候选"。
   总计划 §7.1 的要求，不得跳过。
2. **降权不等于归零，保留 ≥5% 下界。**
   依据导读 §6.3：进化模块让主题内标准差从 `0.146` 涨到 `0.176`，
   探索更多但稳定性更低，且半数主题下降。把某类 mutation 概率压到 0，
   就永久失去发现"某个特定算子恰好吃这个"的可能。这是 EvoSci 的 Variation
   算子存在的理由（导读 §4.4）。

## 11. 已有的经验雏形（不要重复建设）

系统里已存在若干经验机制，Experience System 是**扩展它们**，不是从零开始。

| 已有组件 | 位置 | 本质 |
| --- | --- | --- |
| `OperatorPolicy._EVIDENCE_RULES` | `genetic_operators.py:203` | 失败类别 → mutation 策略映射 |
| Negative evidence 指令 | `genetic_operators.py:109` | 失败经验 → 禁止重复搜索 |
| T3 四门禁 | `contract_executor.py:675-704` | 历史失败 → 确定性规则 |
| candidate provenance / lineage | manifest | 可追溯反馈 |
| `evidence_scope` 分层 | manifest | 证据边界标注 |

`_EVIDENCE_RULES` 现有映射：`accuracy_check_failed → local_rewrite`、
`timeout → strategy_change`、`launch_contract_fail → param_tuning`。

T3 门禁按固定顺序短路，便于归因：
`no_semantic_change` → `shape_fingerprint` → `timing_surface_moved` → `holdout`。

## 12. 各阶段汇总

| Phase | 输入 | 输出 | 收益 | 成本 | 风险 |
| --- | --- | --- | --- | --- | --- |
| 0 | Copy Page + T6 四算子 | 8 个算子从官方 0 分转为可得分 | 最高且确定（功能分 0/100 二值） | 已在进行 | 不做即确定失血 |
| 1 | EA 评估结果 | `evaluated_candidates.jsonl` | 停止丢弃已付费数据；跨算子累积 | 单个提交 | 低；须验零侵入 |
| 1.5 | 官方 §6.2 + holdout | 硬否决 + worst case 排序 | 修量尺，后续结论才可信 | 中（多 case 聚合） | 不做则 Phase 2.5 会学到错的东西 |
| 2 | 算子静态/运行时特征 | typed `KernelQuery` | 证据最强的单项改进 | 小 | 改 candidate sequence，须在 Phase 1 后 |
| 2.5 | 跨算子 archive | 采样权重 | 减少重复无效搜索 | 中 | 须等预算 A/B；须保留 ≥5% 下界 |
| — | — | Knowledge Graph / Agent 扩张 | 未验证 | 高 | 不做，见 §3 |

## 13. 需要决策的事项

以下三项需项目方确认，执行方不得自行决定：

1. **T4 打包后的提交时机。** 打包完成后停下，逐 ZIP 报 SHA 并显式批准后
   才可提交。此前已有连续两轮盲提，本条为硬约束。
2. **Act Quant 候选的定性。** 本机 ratio `0.0258`（快约 38 倍）对一个量化
   kernel 属可疑量级。须先用 T3 门禁 2（`timing_surface_moved`）重验，
   再决定是否进 v1 槽位。在此之前不得当作既有资产计入收益预期。
3. **Phase 2.5 是否启动。** 取决于 Phase 1 跨算子 archive 的实际样本量。
   若样本不足以支撑等预算 A/B，本阶段应推迟而非降低标准。

## 14. 当前未做

- 未实现本文任何阶段的代码；本文是计划，不是完成声明。
- 未运行或复现 EvoSci；论文未提供代码入口。
- 未触发官方评测，未上传任何 artifact。
- Phase 2.5 的权重数值来自本机 910B4 证据，**不是官方 A2/A3 成绩**。

