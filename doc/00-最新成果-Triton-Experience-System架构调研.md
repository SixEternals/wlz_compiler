# Triton Experience System 架构调研

状态：**最新架构成果，分析完成，尚未实现**
日期：2026-08-03
范围：2026 Triton Kernel Optimization Agent，不讨论通用 AI Agent 平台

## 0. 执行结论

当前项目**已经具备加入 Experience System 的数据和流程基础，但尚不具备可直接检索的经验层**。
最合理的 MVP 不是向量数据库，也不是新增一组自由对话 Agent，而是：

```text
现有 Candidate / Evaluation / Failure / Manifest
                    |
                    v
       确定性 Experience Extractor
                    |
                    v
          append-only JSONL records
                    |
                    v
       确定性过滤 + 加权 Top-K 检索
                    |
                    v
       有界 typed prompt context
                    |
                    v
       现有 LLM mutation / search / executor
```

比赛主路径推荐 `Single LLM + Experience Retrieval`。只有证明“策略选择”而不是 correctness、
benchmark 或候选生成质量成为主要瓶颈后，才增加一次性的 Strategy Agent。Experience Retriever
应是确定性代码，Evaluator 必须继续由 compile、correctness test 和 profiler 承担。

对“先达到比赛合格线”而言，Experience System 长期值得投入，但当前只值得投入 Phase 1 的经验
采集薄层。Phase 2 检索、Phase 3 Strategy Agent 和完整 Multi-Agent 都不应排在正确性语义、
可信评测和稳定 mutation 之前。

## 1. 当前结构是否具备基础

### 1.1 可以直接复用的模块

这里必须区分三个层级：正式比赛 runtime（`work/official_triton_agent/`）、共享 schema/离线工具
（`wlz_optimizer/`、`scripts/`）和已保存证据。仓库里存在类或测试，不等于正式 runtime 已接入它。

| 现有能力 | 当前接入层级 | Experience System 中的复用方式 |
| --- | --- | --- |
| `Candidate`、丰富版 `EvaluationResult` | 共享 schema/离线工具；正式 runtime 使用自己的 `Individual` 和较窄 `EvaluationResult` | 作为经验 provenance/outcome 的目标契约；接入前需要显式 adapter，保持真实测量与 proxy 分离 |
| `ShapeObservation`、`EvaluationCase` | 共享 schema/离线 correctness 工具 | 生成 kernel query 的 shape/dtype 特征和 case scope |
| `LaunchProfile` | schema + 测试为主，尚非正式 runtime 的稳定采集输出 | 在真正有采集来源时提取 launch 配置变化和适用条件 |
| `EvaluationCache` | local mock/shared tooling | 继续只做评测复用，不升级成经验真相库 |
| `OfficialEvaluationHistory`、`OfficialFailureHistory` | 离线导入、replay、repair 工具；不在正式 runtime 搜索循环内 | 作为身份绑定的正/负向经验抽取源，不能假设每次正式运行都会自动写入 |
| `PromptContextProjector` | 共享模块且主要由测试覆盖；正式 runtime 有另一套较窄 lineage context | 复用其限长、脱敏思路，或用 adapter 统一两条边界 |
| `OperatorPolicy` / `MutationPlan` / Skill renderer | 正式 runtime | 让检索结果影响策略优先级和本次计划 |
| `BudgetController` | 正式 runtime 已直接复用 | 约束检索文本和未来 Strategy Agent 调用 |
| batch checkpoint / manifest | 开发脚本和离线 artifact，不是正式 EA 的完整 archive | 将结构迁入正式路径后，保存 experience IDs、采用策略和结果归因 |

主要代码入口：

- `wlz_optimizer/schemas.py`：`ShapeObservation`、`LaunchProfile`、`Candidate`、`EvaluationResult`。
- `wlz_optimizer/cache.py`：评测 cache、官方评测历史、官方失败历史。
- `wlz_optimizer/prompt_context.py`：当前候选证据的确定性投影。
- `work/official_triton_agent/genetic_operators.py`：真实 mutation policy、plan、Skill 和 renderer。
- `work/official_triton_agent/evolutionary_algorithm.py`：搜索循环和 lineage 内反馈。
- `scripts/generate_official_candidates_batch.py`：开发期完整 manifest、frontier 和 checkpoint。

正式 runtime 当前只直接导入共享 `BudgetController`，其 candidate/evaluation/context 与离线工具仍是
两套表示。Experience 接入时必须选择一条事实源并做显式 adapter；不能把 schema-only、离线 history
或开发 checkpoint 描述成已经在提交路径中生效。

### 1.2 必须新增的能力

当前缺口不是一个 `experience.jsonl` 文件名，而是以下语义：

1. **Kernel Pattern Extractor**：把当前 kernel 表达为 reduction、normalization、layout、
   scatter/gather、stateful/scan、elementwise 等 family，以及连续性、访问模式、shape bucket、dtype
   family 和 launch 特征。
2. **Semantic Change Extractor**：将父子代码差异转成可查询策略，例如
   `BLOCK_SIZE: 1024 -> 2048`、program-id 映射调整或访存/归约结构变化。
3. **Paired Outcome Builder**：只在 parent/child 使用同 executor、环境、case 口径和 baseline 时计算
   before/after；否则保留两个观察值但不计算提升。
4. **Experience Record**：保存 pattern、strategy、证据范围、结果、风险和置信度，而不是复制整份源码。
5. **Experience Store**：append-only、严格 schema 的 JSONL；索引可重建，不成为第二事实源。
6. **Experience Retriever**：硬过滤环境和适用条件，再做确定性加权排序、去重和多样性选择。
7. **Experience Prompt Projection**：只暴露少量 typed 摘要，不暴露 raw log、任意 metadata、完整历史
   源码或测试 case 身份。
8. **Usage Attribution**：记录检索了哪些 experience、LLM 实际采用哪个策略、结果如何，否则无法判断
   Experience System 是否有效。
9. **Full Evaluated Archive**：正式 EA 需要持久化全部已评测候选，不只保留 survivor 和 Top-5。

第 9 项是当前最大结构缺口。开发期 batch manifest 较完整，但正式 EA 每代截断 population，最终主要
保存 best/Top-5。被淘汰的失败候选和低收益 mutation 丢失后，经验库会严重偏向幸存者，无法学习
“什么在什么条件下失败”。

### 1.3 Lineage / mutation history 能否直接转化

不能直接转化，只能作为经验抽取的原始证据和 join key。

当前已有：

- direct parent、generation、mutation family、model、prompt 和候选 code hash；
- 部分完整 manifest、评测结果、失败记录和环境指纹；
- 当前 lineage 内的失败计数和性能摘要。

当前缺少：

- 统一的 parent code hash 和可长期解析的 parent source；
- 父子 semantic diff；
- kernel family、pattern 和策略适用条件；
- 同一 case、同一硬件、同一 executor 的 paired measurement；
- crossover 或多点改动的单因素归因；
- 多次重复、方差、全 case correctness coverage；
- 被淘汰候选的完整持久化；
- 跨 kernel 聚合后的 confidence。

`mutation_kind=param_tuning` 不能推出“增大 BLOCK_SIZE”，`generation=1` 也不能证明哪个具体改动带来
收益。crossover 或一次修改多个表面时，应将 `attribution_quality` 降为 `ambiguous`，不能强行生成
“成功策略”。

正确的转换流程应是：

```text
Candidate parent-child edge
        + parent/child source and hashes
        + matched EvaluationResult / observation
        + environment and case scope
                    |
                    v
             semantic diff
                    |
                    v
      quality gate + evidence classification
                    |
                    v
     ExperienceRecord(success / failure / neutral)
```

### 1.4 当前数据量与可用性

`output/real-agent-candidates/` 当前有 41 个 candidate manifest：38 个 `static_pass`、3 个
`rejected`，且全部是 generation 1。当前有 30 个新格式的本机 Ascend 配对
sidecar，另有 1 个历史格式 sidecar；它们合起来支撑当前 21 个公开算子的可见 case
qualification matrix。

可作为 schema 示例的真实记录是 `_log_softmax_kernel/857d2fef3baf`：

- baseline：`BLOCK_SIZE = min(1024, next_power_of_2(n_cols))`；
- candidate：上限改为 `2048`；
- 本机 910B4、公开 case 1、新格式配对中位数 ratio：`0.5803802535`；
- candidate 的该公开 case correctness 通过。

但证据仍只是本机 910B4 、单个可见 case，且不同于官方 A2/A3 评测。因此它只能
形成“本机、单公开 case、低到中置信度”的经验样本，不能写成“官方 A2/A3 约 1.72x 加速”或通用 reduction
规则。这个例子说明 Experience System 最重要的不是存数字，而是保存数字的证据边界。

## 2. 最小可行 Experience System

### 2.1 MVP 边界

MVP 只回答一个问题：

> 在固定比赛预算下，能否让当前 kernel 优先尝试历史上在相似条件下有效、且没有明显风险的少量
> 策略，从而提高有效候选比例或 best valid result？

首版不做：

- embedding、向量数据库、知识图谱或外部服务；
- LLM 自动总结全部历史；
- 自由文本长期记忆；
- 自动把任意一次正 speedup 提升为全局规则；
- 在线修改 prompt/Skill 本身；
- 让经验绕过 compile/correctness/performance 评测。

### 2.2 Experience Database

建议使用一个严格 schema 的 append-only JSONL。它是现有事实记录的**可重建派生层**，不是新的
权威测量源。最小字段分为八组。

| 字段组 | MVP 字段 | 说明 |
| --- | --- | --- |
| Identity | `experience_id`, `schema_version`, `created_at` | 稳定 ID 和版本 |
| Kernel pattern | `kernel_family`, `operation_tags`, `access_tags`, `shape_bucket`, `dtype_families` | 只保存可泛化模式；精确 case 信息留在 provenance |
| Strategy | `strategy_id`, `modification_type`, `target_surface`, `parameter_deltas`, `preconditions` | 例如 param tuning / BLOCK_SIZE / 1024->2048 |
| Lineage | `operator`, `parent_id/hash`, `child_id/hash`, `generation`, `mutation_kind` | 回到原始候选和源码的索引 |
| Evaluation | `executor`, `env_fingerprint`, `hardware`, `case_scope`, `compile_ok`, `correctness_ok`, `baseline_id`, `baseline_code_hash`, `before/after`, `delta` | 只有身份和口径严格配对才写 delta |
| Failure/risk | `outcome`, `failure_stage`, `failure_category`, `risk_tags` | success、failure、neutral/unknown 分开 |
| Evidence | `evidence_scope`, `source_refs`, `measurement_ids`, `raw_sample_refs` 或 `sample_summary`, `aggregation_rule`, `aggregation_version`, `repeat_count`, `variance`, `attribution_quality` | 可追溯原始测量及聚合方法；区分官方、本机真机、CUDA proxy、静态门 |
| Confidence | `confidence`, `confidence_reasons` | 规则计算，可解释，不由 LLM 自评分 |

`before/after` 应带明确 metric，例如 `latency_us`、`official_speedup` 或 `local_ratio`。本地 proxy 不得
写入 `official_speedup`。`hardware` 至少区分官方 A2/A3、本机 910B4、CUDA proxy 和 static-only；
未知就写 unknown，不猜测。经验记录不必内嵌大体积 raw samples，但必须保存稳定 measurement ID 和
可解析引用，使 `baseline -> candidate -> 聚合结果` 能被重建；只存 `delta`、`variance` 和次数不够。

建议同时保存一条轻量 `ExperienceUseRecord`：

```text
run_id / operator / query_fingerprint / retrieved_experience_ids
selected_strategy_ids / candidate_ids / outcome / token_cost / wall_time
```

它不是另一套候选历史，只负责回答“这条经验有没有被检索、采用并产生结果”。

### 2.3 经验质量门

Experience Extractor 应按以下规则 fail closed：

1. parent 或 child 源码/hash 缺失：不抽取策略经验，只保留原始评测。
2. compile/correctness 未明确通过：不得标为成功性能经验。
3. executor、环境、case scope、baseline identity/hash 或 aggregation rule 不同：不得计算 paired delta。
4. 一次修改多个独立策略或来自 crossover：允许保存 observation，但 attribution 为 ambiguous。
5. 只有静态门通过：只能形成 validity hint，不能形成 correctness/performance 成功经验。
6. 单 case 或单次测量：降低 confidence，并将覆盖范围写入 preconditions。
7. 相同策略在不同条件下有冲突：保留多条带条件记录，不覆盖旧记录。

confidence 首版用确定性规则即可，例如综合 evidence level、correctness coverage、paired measurement、
重复次数、方差和 attribution quality。不要让 LLM 返回一个没有可核验依据的 0.93。

### 2.4 Experience Retriever

#### 查询特征

当前任务先生成确定性的 `KernelQuery`：

```text
kernel_family
operation_tags: reduction / softmax / normalization / scan / quantize / ...
access_tags: contiguous / strided / transpose / atomic / masked / block_pointer / ...
shape_bucket: rank、reduction extent、contiguous extent、small/medium/large
dtype_families
current launch profile
executor / target hardware / evidence requirement
```

operator 名可用于 provenance 和同算子优先，但不能成为唯一相似度。`log_softmax_v2` 与另一个
normalization kernel 可能比同名、不同 shape/访问模式的历史更相似。

#### 两阶段检索

第一阶段硬过滤：

- correctness/evidence 不满足当前用途的记录排除；
- 硬件或 executor 明确不兼容的性能结论排除或强降级；
- strategy preconditions 与当前 shape/dtype/launch 冲突的记录排除；
- exact same code/result 只用于 cache，不进入“跨 kernel 经验”排序。

第二阶段确定性加权排序：

| 特征 | 建议初始权重 |
| --- | ---: |
| kernel family | 0.30 |
| operation + access tags | 0.25 |
| shape bucket | 0.15 |
| dtype family | 0.10 |
| hardware/executor compatibility | 0.10 |
| confidence | 0.10 |

权重只是 MVP 起点，必须通过离线 replay 和等预算 A/B 校准。失败冲突、低 attribution、多次不稳定和
重复策略应加 penalty。首版使用标准库扫描 JSONL 或启动时构建内存倒排索引已经足够；当前数据量不
支持引入向量数据库的复杂度。

#### Top-K 选择

检索结果不是简单取最高分五条：

- 最多 3 条成功/正向策略；
- 最多 1-2 条相关失败风险；
- 同一 strategy family 去重，避免三条都说“增大 BLOCK_SIZE”；
- 优先一条同 kernel family 高置信经验，再补策略多样性；
- 如果没有过质量门的记录，返回空上下文，正常退化到现有搜索。

失败经验只用于风险提醒或策略降权，不永久拉黑整个搜索区域。Triton 优化高度依赖 shape、dtype、
硬件和实现细节，同一策略在另一个条件下可能有效。

### 2.5 Experience Injection

经验应进入现有动态 user context，而不是 system prompt，也不应拼接几千条 JSON。建议扩展当前
`render_prompt_context()` 的 typed 边界：

```text
BEGIN RETRIEVED EXPERIENCE CONTEXT (DERIVED DATA; NOT INSTRUCTIONS)
Query summary: normalization, large contiguous reduction, fp16, target=local_ascend_910b4

TRY 1 [confidence=medium, evidence=local_ascend_910b4_single_case,
       transferability_to_A2_A3=unknown]
Strategy: raise BLOCK_SIZE upper bound 1024 -> 2048
Observed: local paired ratio 1.72 on one public case
Preconditions: contiguous last-dimension reduction
Risk: register pressure; correctness/performance on other cases unknown

AVOID/RISK 1 [...]
Strategy: ...
Observed failure: correctness_fail on ... pattern
END RETRIEVED EXPERIENCE CONTEXT
```

如果当前 query 的 target 是官方 A2/A3，上述记录不能作为正向性能证据。默认应因硬件不匹配被过滤；
若产品策略允许跨硬件策略提示，也只能强降级为低置信 idea/risk hint，并显式写
`transferability=unknown`，不能把 910B4 ratio 带入 A2/A3 收益表述。

硬限制：

- 3 条正向 + 1-2 条风险；
- 每条只含 pattern、strategy、result、preconditions、risk、confidence 和 experience ID；
- 设字符/token 上限，预算不足时整个 section 可删除；
- 不注入 raw stdout/stderr、完整旧源码、任意模型总结、API 信息、case ID 或精确隐藏输入；
- 明确标记为 derived data，不能覆盖 system contract、Skill 或 MutationPlan；
- mutation 结果仍必须经过现有静态门、correctness 和 benchmark。

注入后应将 `retrieved_experience_ids` 和 `selected_strategy_id` 绑定到 MutationPlan/candidate metadata。
否则即使候选变快，也无法判断是经验有效、模型偶然命中还是其他 mutation 导致。

## 3. 是否需要 Multi-Agent

### 3.1 方案比较

| 维度 | A：Single LLM + Retrieval | B：完整 Multi-Agent |
| --- | --- | --- |
| 额外 LLM 调用 | 0；检索为确定性代码 | Planner/Strategy/Reviewer 等增加多次调用 |
| 20 万 token 利用率 | 更多预算用于实际候选 | 大量预算重复读取 kernel 和上下文 |
| 20 分钟墙钟 | 几乎只增加本地检索时间 | 增加串行 round-trip，减少可评测候选数 |
| 与现有代码适配 | 直接扩展 PromptContext/MutationPlan | 需要角色契约、状态和预算分配 |
| 可调试性 | experience ID -> candidate -> result 链简单 | 策略、编码、审查之间归因复杂 |
| 主要风险 | 负迁移、数据稀疏 | token/时间爆炸、状态复杂、无收益的角色重复 |
| 工程成本 | 低到中 | 高 |

当前更适合方案 A。现有 EA 已承担 planner 的大部分职责：选择父代、选择 mutation family、维护种群、
停止和 Top-5；现有 executor 已承担 evaluator。再为每个 child 固定增加 Planner、Experience、
Strategy、Coder、Evaluator Agent，会重复职责并减少真实评测机会。

### 3.2 如果要增加 Agent，先只加 Strategy Agent

后期最小方案是：

```text
kernel analysis + deterministic retrieval
                  |
                  v
Strategy Agent（每 kernel 一次，或搜索停滞时一次）
                  |
                  v
typed StrategyPlan
                  |
                  v
现有 GeneticOperators Coder -> deterministic gates/executor
```

Strategy Agent 只输出经过 schema 校验的策略列表、优先级、适用条件和预算建议，不直接改代码。触发
条件建议是初始一次或连续若干有效候选无提升后的 stagnation；不能对每个 child 调用。

不建议新增：

- Experience Agent：检索、过滤和排序应可重放，不需要 LLM。
- Evaluator Agent：正确性和性能必须来自实际测试/profiler，不能由 LLM 判断。
- 常驻 Reviewer Agent：只有相同总预算 A/B 证明其收益高于再生成一个候选时才保留。

结论：比赛当前收益/成本比最高的是方案 A。完整 B 的收益上限目前未知、证据最弱，工程成本和预算
风险最高；只有同预算 A/B 胜过“多生成并验证候选”后，才能声称它有额外收益。

## 4. Multi-Agent 是否需要自建 Harness

需要自建，但不是从零实现通用 Agent 平台。当前 `TritonOptimizerAgent + EvolutionaryAlgorithm +
GeneticOperators + executor + BudgetController` 已经是比赛专用 Harness 主体。官方只提供 LLM API、
且运行时不依赖 Claude Code/Codex CLI 时，需要补的是窄领域角色编排，而不是 IDE Agent 能力。

最少组件如下：

| 组件 | 当前可复用 | Multi-Agent 需要补充 |
| --- | --- | --- |
| Agent orchestration | optimizer + EA 确定性循环 | 明确角色状态机、触发条件、fallback；不需通用 DAG |
| Context builder | 正式 renderer/窄 lineage context；离线 `PromptContextProjector` | 先统一或适配两套表示，再增加 KernelQuery、Top-K experience、每角色字段和长度上限 |
| Memory | 正式 population/lineage；离线 schema、cache/history/manifest | 选定事实源后增加 ExperienceRecord、full evaluated archive、use attribution |
| Tool calling | 当前代码直接调用 LLM/executor | MVP 仍由 orchestrator 调用；动态工具时才需 allowlist dispatcher |
| State management | 正式 population/lineage；开发脚本 batch checkpoint | 把所需恢复语义接入正式路径，增加 run/strategy/experience-use 状态和全候选 archive |
| Budget | 共享 BudgetController | 各角色 reservation、全角色合计、收尾预算和拒绝原因 |
| Evaluation loop | static gate、correctness、`msprof`、Top-5 | 分阶段结果写回策略；Evaluator 不由 LLM 代替 |
| LLM contract | mutation/crossover prompt | StrategyPlan JSON schema、校验、一次 fallback、调用 provenance |

若未来允许 Agent 动态调用工具，还要实现：工具白名单、参数 schema、超时、幂等 key、结果裁剪、审计
日志和预算 admission。MVP Strategy Agent 不需要这一层：让它输出 JSON，由 orchestrator 确定性调用
现有 coder/evaluator 即可。

明确不需要先做：自由规划、通用插件市场、长期对话压缩、通用向量服务、多角色群聊或让 LLM 自行
决定是否跳过 correctness。

## 5. 实际落地路线

工作量用小验收单元表示；每个单元都应能独立测试和回滚。

| 阶段 | 最小交付 | 工作量 | 预期收益 | 主要风险 / Go-No-Go |
| --- | --- | ---: | --- | --- |
| Phase 0：保持当前系统 | 冻结 Best-of-N、普通 EA、当前 mutation 和评测基线；记录统一 token/墙钟/有效候选指标 | 0-1 单元 | 获得 Experience 的对照组 | 没有基线就无法证明经验有效 |
| Phase 1：增加经验库 | full evaluated archive；Experience schema；从父子+评测派生 JSONL；不接 prompt | 2-4 单元 | 不改变搜索即可积累成功/失败原料，低风险 | 错误归因、第二事实源；必须可重建和 fail closed |
| Phase 2：增加经验检索 | KernelQuery；硬过滤+加权 Top-K；typed prompt section；usage attribution；离线 replay/A-B | 3-5 单元 | 减少随机试错，提高首批有效候选率 | 数据稀疏、跨 shape/硬件负迁移；无收益则关闭注入 |
| Phase 3：增加 Strategy Agent | 每 kernel/停滞时一次；StrategyPlan schema；共享预算；fallback 到现有 policy | 4-6 单元 | 对复杂 kernel 组合少量策略、提高搜索方向性 | 调用成本可能不如再生成一个候选；必须等预算 A/B |
| Phase 4：完整 Multi-Agent | 角色状态机、统一 checkpoint、严格 tool adapter、跨角色 attribution 和恢复 | 10+ 单元，按周计 | 可能改善复杂结构优化和解释性 | token/墙钟、状态爆炸、调试和归因成本；不适合当前合格线 |

### Phase 0 验收指标

- 相同模型、prompt、token、墙钟和真实评测次数；
- Top-5 至少一个完全正确候选的比例；
- static/import/compile/correctness 通过率；
- unique valid candidates、重复失败率；
- best valid result；
- 本地、官方硬件和 proxy 指标分栏报告。

### Phase 1 推荐拆分

1. 先让正式 EA 保存每个 evaluated candidate 和结果，不改变 selection。
2. 定义严格 ExperienceRecord，先只支持“单父代、单一参数变化、同口径评测”。
3. 离线 extractor 生成 JSONL；输入事实变化时可完整重建。
4. 增加数据质量报告：可配对、ambiguous、缺 parent、缺 correctness、环境冲突各多少条。

### Phase 2 推荐停止条件

- 高质量经验少于可形成跨 kernel 比较的规模时，不上 embedding，也不扩大 prompt。
- 同预算 A/B 中，retrieval 没有提高 unique valid、Top-5 功能率或 best valid result，则默认关闭。
- 经验对某 family 有益、对另一 family 有害时，采用 family gate，不全局启用。

### Phase 3/4 前置条件

- 经验样本有足够的跨 kernel 覆盖；
- 当前策略选择已被证明是主要瓶颈；
- Strategy Agent 的额外 token/时间有明确上限；
- 至少完成一次与“把相同预算用于多生成候选”的对照；
- 任意 Agent 失败都能退化到现有单 LLM 搜索并返回合法 Top-5。

## 6. 为先达到比赛合格线的优先级

官方功能门槛要求至少一个非原样候选成功编译并通过全部测试；只有功能通过的候选才进入性能评测。
因此当前优先级应是：

1. **Correctness / evaluation 语义**：明确 compile、correctness、process、profiler 各阶段，覆盖该算子
   全部 case，避免把 duration 或 process success 当功能通过。
2. **可信 benchmark 与环境绑定**：同 case、同 baseline、重复配对测量，记录设备、toolchain、方差；
   本机 910B4 与官方 A2/A3 分开。
3. **Mutation prompt 和廉价门**：提高非原样、唯一、静态/import/correctness 有效候选比例；优先
   结构化参数/局部 mutation，不扩建多 Agent。
4. **Search algorithm**：在相同预算下比较 Best-of-N、普通 EA、失败感知 EA；保证 Top-5 去重和
   策略多样性，避免种群塌缩。
5. **Experience Phase 1**：顺手保存完整 evaluated archive 和可重建经验，为后续积累数据。
6. **Experience Phase 2**：数据足够后做 deterministic retrieval 和小上下文 A/B。
7. **Strategy Agent / 完整 Multi-Agent**：只有前述瓶颈闭环且同预算实验支持时再做。

最终判断：

- **长期**：Experience System 值得做，因为 Triton 策略高度重复且 NPU 评测昂贵，成功和失败都应
  跨 run 复用。
- **当前合格线**：只投入 Phase 1 的低侵入采集薄层；不要让 Phase 2-4 延迟 correctness、benchmark、
  mutation 和基础搜索的闭环。
- **工程选择**：先 `Single LLM + deterministic retrieval`，后门控 Strategy Agent；不先做完整
  Planner -> Experience -> Strategy -> Coder -> Evaluator 链。

## 7. 架构决策记录

| 决策 | 当前结论 | 重新评估条件 |
| --- | --- | --- |
| Experience Store | append-only JSONL，派生且可重建 | 数据量或并发确实超过单机 JSONL 能力 |
| Retrieval | 规则过滤 + 加权 Top-K | 有足够样本证明语义 embedding 提升同预算结果 |
| Prompt 数量 | 最多 3 条正向 + 1-2 条风险 | token A/B 证明更长上下文有净收益 |
| Runtime 方案 | Single LLM + retrieval | Strategy Agent 同预算显著胜出 |
| Evaluator | 确定性测试和 profiler | 不变；LLM 不成为 correctness/performance oracle |
| 当前投入 | Phase 1 采集薄层 | 合格线基础闭环稳定且经验样本达到可检索规模 |

## 8. 证据入口

本报告基于 2026-08-03 当前工作树和以下可复核入口：

- `wlz_optimizer/schemas.py`
- `wlz_optimizer/cache.py`
- `wlz_optimizer/prompt_context.py`
- `wlz_optimizer/evolutionary_algorithm.py`
- `work/official_triton_agent/genetic_operators.py`
- `work/official_triton_agent/evolutionary_algorithm.py`
- `work/official_triton_agent/optimizer_agent.py`
- `scripts/generate_official_candidate.py`
- `scripts/generate_official_candidates_batch.py`
- `output/real-agent-candidates/_log_softmax_kernel/857d2fef3baf.py`
- `output/real-agent-candidates/_log_softmax_kernel/857d2fef3baf.ascend-evaluation.json`
- `doc/提示词工程与Skill系统设计方案.md`
- `doc/2026-Triton进化优化研究与实施总计划.md`

本轮只完成架构分析和文档收敛，没有实现 Experience Store、Retriever、Strategy Agent 或
Multi-Agent Harness。
