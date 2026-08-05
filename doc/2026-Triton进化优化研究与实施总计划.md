# 2026 Triton 进化优化研究与实施总计划

状态：当前权威研究与实施计划；文档入口以 `doc/README.md` 为准
更新时间：2026-08-03
适用项目：2026 华为毕昇杯编译系统挑战赛研究与工具
当前主赛题：**基于进化算法的 Triton 自动优化系统**

如果本文件与旧阶段报告冲突，以本文件为准；如果本文件与官方最新技术方案、任务正文、通知或
平台原始结果冲突，以官方最新材料为准。最新专项成果见
`doc/00-最新成果-Triton-Experience-System架构调研.md`。

## 1. 项目目标

本项目在官方给定 Agent 框架和接口内，用 LLM 作为语义 mutation/crossover 算子，用进化算法管理
候选、选择、反馈和预算，自动优化 Triton kernel。

研究主线是：

> 在每算子固定 token、墙钟和真实测量预算下，结构化失败反馈、质量多样性、预算调度和有证据
> 约束的跨 kernel 经验复用，能否提高有效候选比例和最终 Top-5 的 best valid result。

项目不以重写 Triton-Ascend、AscendNPU IR、runtime 或完整编译器为目标。IR/profile 只有在官方
接口稳定暴露、获取成本可接受且离线排序实验有收益时，才作为可删除增强项。

## 2. 官方硬约束

以下约束由 2026 年 4 月官方技术方案支持，除非赛方后续明确变更：

- 必须在给定 Agent 框架内实现进化算法，自动优化赛方给定 Triton 代码。
- 必须使用给定环境和接口，不允许参赛者手工修改候选。
- 每个算子最多 20 分钟或 20 万 token，任一先到即停止。
- Multi-Agent 的所有 Agent token 合计。
- 每个算子最多返回 5 个版本。
- 至少一个非原样候选必须成功编译并通过全部测试，功能项才得分。
- 只有功能完全通过的候选才进入 NPU 性能评测。
- 初赛公开 50 个 case；决赛追加 50 个隐藏 case。
- 初赛客观指标为 Passing Rate 30%、性能 70%。
- 官方目标平台为鲲鹏 920、Ascend A2/A3 和 openEuler。
- 比赛运行时模型必须来自官方 allowlist；当前项目默认模型族为 DeepSeek-V4。

Multi-Agent 被允许，不表示它在 20 万 token/20 分钟内天然更优。任何额外角色都必须与“把相同预算
用于多生成并验证一个候选”做对照。

## 3. 证据和表述规则

### 3.1 五级结果边界

文档、日志和答辩必须分别报告：

1. `upload accepted`：表单/传输成功。
2. `platform run completed`：平台结束排队或运行。
3. `official executor success`：平台显式确认执行器成功。
4. `functional pass`：显式确认编译和要求的全部正确性测试通过。
5. `performance result`：官方明确给出 latency、speedup 或 score。

低一级不能推出高一级。`运行结束`、HTTP 成功、进程返回 0、profiler 有 duration 或一个 aggregate
`success` 都不能单独证明 functional pass。

### 3.2 本地与官方环境

当前本机有 Ascend 910B4、CANN 和 `msprof` 开发环境。本机可产生真实设备测量，但 910B4 不是官方
A2/A3：

- 结果必须标为 `local_ascend_910b4` 或等价 evidence scope；
- 不能称官方 latency、官方 speedup 或官方排名；
- CUDA/static/mock 结果只能叫 `proxy_score`、`local_score` 或静态门结果；
- 跨 910B4 与 A2/A3 的候选排序稳定性保持 unknown，直到有配对证据。

### 3.3 未知项

没有证据时保持 unknown，不假设官方接口一定暴露：

- 独立 compile/correctness/performance 字段；
- pass 级编译失败位置；
- TTIR、Linalg IR、AscendNPU IR dump；
- UB、寄存器、workspace、occupancy 或稳定 profiler counter；
- 自由 SSH、远程命令、simulator 或 compiler option。

## 4. 当前架构

```text
Task / baseline / tests
          |
          v
Kernel contract + current evidence
          |
          v
OperatorPolicy + MutationPlan + bounded PromptContext
          |
          v
DeepSeek-V4 mutation / crossover
          |
          v
syntax -> interface -> import -> correctness -> performance
          |
          v
EvaluationResult + failure feedback + cache/history
          |
          v
selection / frontier / (mu+lambda) / Top-5
```

这是项目级逻辑图，合并展示正式比赛 runtime 与共享/离线开发工具，并不表示所有方框已在一条提交
路径中串联。正式 runtime 当前只直接导入共享 `BudgetController`，并自有 `Individual`、较窄的
`EvaluationResult` 和 lineage context；environment-aware history/cache、丰富 schema、batch checkpoint
主要位于共享库和开发脚本。

横切能力候选包括 `BudgetController`、Candidate provenance、environment fingerprint、checkpoint、
artifact validation 和安全打包；其中只有明确标为正式 runtime 的能力才能解释为比赛运行时已接入。

LLM 不是正确性或性能 oracle。它负责提出候选；确定性门、真实测试和 profiler 负责裁决。

## 5. 2026-08-03 当前状态

### 5.1 已具备（按接入层级）

正式比赛 runtime（`work/official_triton_agent/`）：

- LLM mutation/crossover、版本化 Prompt/Skill、`OperatorPolicy`、`MutationPlan` 和较窄的 lineage
  feedback context；
- `Individual` 中的 candidate ID、父代、generation、mutation、model 和部分 prompt provenance；
- 同一 `BudgetController` 约束 LLM token 与全流程墙钟，executor timeout 按剩余墙钟裁剪；
- rank selection、`(mu+lambda)` survivor、去重、实际 executor 调用和最多 Top-5 输出；
- runtime 自有的较窄 `EvaluationResult`，其 process/performance success 尚不能稳定替代独立
  compile/correctness 事实。

共享 schema、离线工具和开发 harness（`wlz_optimizer/`、`scripts/`）：

- 更丰富的 `Candidate`、`EvaluationResult`、`ShapeObservation`、`EvaluationCase`、`LaunchProfile` 数据
  契约；其中 `LaunchProfile` 目前主要是 schema/test 能力，不是正式 runtime 稳定输出；
- Python/Triton 静态契约、隔离 import、部分逐算子 correctness、环境感知 cache；
- 身份绑定的官方 evaluation/failure JSONL、离线 replay 和 repair 输入；
- `PromptContextProjector`、开发期 batch manifest、宽度 2 admitted frontier 和 checkpoint；
- 确定性 ZIP、hash、路径穿越和敏感信息门禁。

真实设备开发证据层另有统一 1 至 3 case 的 Ascend runner、`msprof` duration evidence 和本机 sidecar。
这些层之间尚无一个完整 adapter；不能把共享 schema、离线 history/cache 或 batch checkpoint 直接称为
正式比赛 runtime 的持久化能力。

本地统一验证：

```text
.venv/bin/python -m unittest discover -s tests -v
418 tests, OK, skipped=27
```

跳过项依赖额外环境，不能解释为对应 CUDA/Triton/NPU 路径通过。

### 5.2 已有真实设备开发证据

截至 2026-08-03，当前 checkout 的 21 个公开算子均完成本机 910B4 qualification matrix：每个算子
有非 baseline candidate、当前可见 correctness case 通过，以及同 device/frequency 的 seeded `B,C,C,B`
paired `msprof` evidence；candidate median / baseline median 不超过 `1.03`，矩阵为 `21/21`。

逐算子 candidate、ratio、source/test hash、失败 probe 和 raw sidecar 见
`doc/02-最近成果-本机910B4-21算子资格闭环.md`。这批结果只覆盖 `21 x 1` 当前可见 case，不能替代
官方 A2/A3、隐藏 case 或官方成绩。

其中 `_log_softmax_kernel` 候选 `857d2fef3baf` 的新格式本机 ratio 是 `0.5803802535`；最明显的当前策略
级提升来自 `_chunk_cumsum_fwd_kernel`（ratio `0.1568577236`）和 `_fwd_kernel_ep_gather`
（ratio `0.4806480915`）。`_silu_mul_fp8_quant_deep_gemm` 已从 comment-only candidate 替换为
`num_warps=2` 的真实 launch candidate（ratio `0.9948293690`）。当前 21 个 selected candidate 中
14 个有实质 kernel/launch 变化，7 个 neutral/equivalent candidate 只用于验证不退化，
不计入成功策略。

本机结果仍不是比赛性能结论；官方 case 2/3、A2/A3 和隐藏 case 均未知。

### 5.3 仍缺失

- 没有为所有 kernel、所有 case 稳定拆分 compile、correctness、process 和 profiler success；
- 没有在官方 A2/A3 上完成统一预算的 Best-of-N、普通 EA 和主方案消融；
- 没有质量多样性档案和按近期策略收益的执行预算分配；
- 没有稳定 IR/资源特征入口或 Top-K 排序实验；
- 正式 EA 没有持久化全部 evaluated candidate，被淘汰节点的失败和低收益信息可能丢失；
- 没有父子 semantic diff、Experience Record、跨 kernel Retriever 或使用效果归因；
- 没有证明 Strategy Agent 或完整 Multi-Agent 的单位预算收益。
- `21/21` 只是本机当前可见单 case qualification，不代表官方全 case functional pass 或最终性能通过。

因此当前已经超过本地 mock 骨架，但尚未验证完整研究假设，也不能声称可泛化的官方 Ascend 性能
提升。

## 6. 研究假设

### H1：结构化失败反馈

在固定预算下，规范化失败签名、失败 cache 和策略级统计，比直接把 raw log 塞回 prompt 更能降低
重复失败率，提高 compile/correctness 通过率。

### H2：质量多样性与预算调度

相比只按单一 fitness 保留精英，少量可解释行为维度和按策略近期产出分配预算，能降低种群塌缩并
提高 best valid result。

### H3：跨 kernel Experience Retrieval

从完整、有 provenance 的历史中派生 pattern + strategy + outcome，在相同模型、token、墙钟和真实
评测次数下，确定性检索少量相似经验能提高 unique valid candidate 比例或 best valid result。

H3 的前置是 full evaluated archive、semantic diff、同口径 paired outcome 和环境边界。lineage 或
`mutation_kind` 本身不等于经验。完整设计见 Experience 最新架构调研。

### H4：可选 IR/profile 排序

如果接口提供稳定、低成本且环境绑定的 IR/profile 特征，它们可能提高真实性能候选的 `Recall@K`。
数据不可得、开销过高、跨 shape 不稳定或离线消融无收益时，删除该增强项。

## 7. 实验设计

### 7.1 必须有的等预算基线

在相同模型、初始 prompt、token、墙钟、真实 NPU 测量次数和收尾预算下比较：

1. Best-of-N 独立采样。
2. 当前参考搜索流程。
3. 普通进化：精英选择 + mutation，不使用失败记忆。
4. 失败感知进化。
5. 失败感知 + 质量多样性 + 预算调度。
6. 单 LLM，关闭/开启 deterministic Experience Retrieval。
7. 若进入后期：Strategy Agent 与“同预算多生成候选”的配对实验。
8. 若接口允许：主方案关闭/开启 IR/profile 排序。

### 7.2 指标

- 功能：Top-5 至少一个非原样、全部 case 正确候选的比例。
- 漏斗：unique/non-noop、static/import/compile/correctness 通过率。
- 搜索：重复代码、重复失败、种群多样性、best valid result。
- 成本：token、墙钟、LLM 调用、compile、correctness 和真实测量次数。
- 性能：只用官方或明确标记硬件的 latency/speedup；报告 raw samples、方差和噪声。
- 检索：experience hit、strategy adoption、负迁移、token/unique-valid、按 family 收益。
- 泛化：公开/留出 shape、同 family 跨 kernel；隐藏 case 不进入反复调参外环。

### 7.3 消融顺序

```text
correctness/evaluation 口径
-> mutation 和 Best-of-N/普通 EA 基线
-> 失败复用
-> 质量多样性 / 预算调度
-> Experience 采集
-> Experience Retrieval
-> Strategy Agent
-> 可选 IR/profile
```

一次只改变一个可归因假设。没有相同预算基线，新增模块不能称为收益。

## 8. 工程边界

- Python 保持比赛主入口；不建立 Rust-first 旁路。
- 不自研 parser、通用 IR、runtime、backend 或通用 Agent 平台。
- Candidate/schema、executor、cache、ranking、genetic operators 和 EA 各自保持所有权。
- cache 继续做结果复用；Experience 是可重建派生层，不把 cache 改成第二事实源。
- 成功和失败都保存 provenance；缺失信息保持 unknown。
- raw stdout/stderr、候选注释和历史源码是 untrusted data，不进入 system instruction。
- Prompt 只注入 typed、限长、脱敏事实；接口冻结面由代码门独立验证。
- 不硬编码设备、CANN 路径、远程主机、用户名或凭据。
- 每轮一个可验收能力，不一次实现“完整 optimizer”或“完整 Multi-Agent”。

## 9. 当前实施顺序

| 顺序 | 阶段 | 当前状态 | 完成条件 |
| ---: | --- | --- | --- |
| 1 | A：correctness/evaluation 语义 | 进行中 | compile、correctness、process、profiler 分层；全 case fail closed |
| 2 | B：可信 benchmark 与环境绑定 | 部分完成 | 同 case paired repeats、设备/toolchain、raw samples、噪声边界 |
| 3 | C：mutation 与搜索基线 | 部分完成 | Best-of-N/普通 EA/失败感知 EA 在相同预算可比较 |
| 4 | D：Experience Phase 1 采集 | 未开始 | full evaluated archive + 可重建 JSONL，不改变搜索 |
| 5 | E：Experience Phase 2 检索 | 未开始 | deterministic Top-K + typed prompt + usage attribution + A/B |
| 6 | F：Strategy Agent | 暂缓 | 每 kernel/停滞时一次，且同预算显著优于多生成候选 |
| 7 | G：完整 Multi-Agent | No-Go 当前 | 只有窄 Strategy Agent 仍不足且证据支持时重新评估 |

Experience 的 Phase 0-4 工作量、收益和风险详见最新专项报告。当前为达到功能合格线，只允许 Phase 1
采集薄层与 A-C 并行；检索和 Multi-Agent 不得阻塞 A-C。

Prompt 继续使用当前纪律：确定性选择 mutation kind，一次 LLM 生成，代码门和真实 executor 评测；
repair、reviewer、planner 或多模型切换只有等预算 A/B 通过后才进入默认路径。

## 10. Experience 与 Multi-Agent 决策

Phase 2 的目标运行方案：

```text
Task -> deterministic Experience Retrieval -> bounded Prompt -> Single LLM
     -> static/correctness/performance -> selection
```

当前实际路径跳过 retrieval，Retriever 尚未实现。未来即使接入，经验库为空、质量门失败或预算不足时
也必须退化为现有搜索。

后期若增加 Agent，只先增加一个门控 Strategy Agent：每 kernel 一次或停滞时一次，输出 schema 化
`StrategyPlan`，继续由现有 GeneticOperators 编码、由真实 executor 评测。Experience Agent 不需要
LLM，Evaluator Agent 不能替代真实测试。

## 11. Go / No-Go

### Go

- correctness/evaluation 阶段事实可稳定获取并绑定具体 candidate/case/environment；
- mutation 在固定预算下能持续产生 unique valid candidate；
- 失败复用、多样性或检索在多个 family 的等预算实验中超过基线；
- 有真实 A2/A3 或官方结果可验证最终功能和性能。

### No-Go / 降级

- 阶段信息只有聚合布尔值：只做粗粒度 cache/feedback，不声称 compile-aware 或 correctness-aware。
- LLM mutation 通过率低：先收紧结构化参数/局部 mutation，不增加 Agent。
- Experience 高质量样本稀疏：只采集，不开启检索，不引入 embedding/vector DB。
- Retrieval 发生负迁移：按 family/hardware gate 或默认关闭。
- Strategy Agent 不优于同预算多生成候选：删除该角色。
- 没有 IR/profile：删除 H4，不影响主线。
- 没有官方 A2/A3：可报告工程与本机开发结果，不宣称比赛性能收益。

## 12. 下一验收单元

下一代码验收单元只做一件事：

> 审计并用聚焦测试固定官方 runtime 中 process/profiler success 与 compile/correctness 状态的边界，
> 确保没有显式 correctness 证据时保持 unknown，不改变搜索算法、不实现 Experience。

完成后再决定最小代码修复。Experience Phase 1 的第一个单元应另行定义为“正式 EA 保存全部 evaluated
candidate”，不能与 correctness 修复混在同一提交。

## 13. 资料和文档治理

- 当前阅读入口：`doc/README.md`。
- 最新架构成果：`doc/00-最新成果-Triton-Experience-System架构调研.md`。
- Prompt 专项：`doc/提示词工程与Skill系统设计方案.md`。
- 官方接口和运行派生证据按其固定日期/commit 阅读，不作为滚动状态。
- 原始官方 PDF、网页快照和下载资料位于外部 `/workspace/user_data/supply-doc/sources/`，本轮不改写、
  不移动、不删除。
- 过期 Rust-first、本地 mock 完成快照、冲刺/阶段/交接稿不保留在当前 `doc/`；需要追溯时查外部
  `supply-doc` Git 历史。

网络入口会变化。正式引用必须记录 URL、commit/版本和访问日期；平台操作必须以 fresh authenticated
page 和 `AGENTS.md` 中的当前固定任务身份为准。
