# 提示词工程与 Skill 系统设计方案

日期：2026-07-23
状态：P0、P1 实施及开发期裸 DeepSeek A/B 已闭合；尚无 Ascend 性能接受结论

## 1. 证据边界与竞赛规则

本方案区分四类证据，不能把参考实现中的 prompt 文案写成大赛条款：

1. **官方规则**：赛题技术方案、2026 总章程及其援引的违规认定说明。
2. **官方参考框架事实**：固定提交 `ef8c3bbc7bae6bdfa2af61722f9da14fd8ea5781`
   的源码和 README；它能证明当前接口形状，但不自动等于竞赛条款。
3. **项目工程规则**：为保护接口和降低失败率而增加的静态门、prompt 契约和保守冻结项。
4. **研究参考**：doge-code、SkillOpt-Lite 论文和 `skill-opt` 私有参考仓；只能产生
   待本项目 A/B 验证的假设，不是官方事实或已证明收益。

### 1.1 官方规则原文

赛题原文：
`doc/sources/official_compiler2026_text/2026年全国大学生计算机系统能力大赛编译系统设计赛-编译系统挑战赛-基于进化算法的Triton自动优化系统-技术方案.txt`

| 条款 | 原文短引 | 对提示词工程的约束 |
| --- | --- | --- |
| 第 1 条 | “在给定的 agent 框架中设计进化算法……实现对 triton 算子的自动优化” | LLM mutation/crossover 是合法的自动变换手段。 |
| 第 2 条 | “所使用的大模型必须在清单中”；“鼓励……新的工具，multi-agent 协作等方法” | Skill 和 multi-agent 原则上允许；比赛运行时模型必须在附录 2 清单内。 |
| 第 3 条 | 除规定和禁止事项外，“自行决定所采用的优化技术等细节” | Prompt 分层、路由和反馈不能越过其他硬约束。 |
| 第 4 条第 1 款 | “必须使用提供的环境以及接口，不可改变接口调用方式” | 冻结官方入口、参数和调用关系；输出 schema 能否扩展仍是 `unknown`。 |
| 第 4 条第 2、3 款 | “通过 agent 进行变换，搜索等步骤”；“不得手动修改 triton 代码” | 候选必须由 agent 自动生成；不得人工修补生成后的候选。 |
| 第 4 条第 4 款 | “如使用 multi-agent 设计，将计算所有 agent 消耗 token 之和” | analyzer、generator、reviewer、repairer 的 token 全部合计。 |
| 第 4 条 | “每个算子，最多运行 20 分钟或最多消耗 20 万 token（任一上限满足即停止）” | 必须共享统计墙钟和 token，不能只设置单次 `max_tokens`。 |
| 第 5 条 | “返回至多 5 种版本”；“与原算子完全相同，不得分” | 最终最多 5 个非原样候选；注释或诊断文本变化不算有效优化。 |
| 第 6 条 | “仅完全通过功能测试的代码会记入此测试” | LLM reviewer 的判断不能替代真实编译和全部正确性测试。 |
| 第 13 条 | 使用第三方源码“必须在设计文档和源代码的头部予以明确说明” | doge-code 只借鉴机制；如复制代码，必须核验许可证并披露。 |
| 第 14 条 | 未说明的代码借鉴或“修改的代码重复率在 20% 以上”将取消资格 | 采用更严格的 20% 口径，避免复制提示词框架源码。 |
| 附录 2 | “DeepSeek-V4 全系列, DeepSeek-V3.2”、Qwen3.5/3.6、Kimi-K2、GLM-4.6 | V4 明确允许；以该清单为准，参考框架中的旧模型列表已经过时。 |

2026 总章程的补充约束：

- 第 7.4 条：使用大模型辅助生成代码时，应在代码注释、工程文档和答辩中说明。
- 第 7.5 条：必须遵守《关于编译优化合理性与违规行为认定的说明》，反对针对特定测例的
  投机性、针对性优化。

总章程援引的违规认定说明进一步规定：

- 第二部分允许针对代码整体特性和处理器通用特性的优化，例如循环变换、指令重排和寄存器
  分配优化。
- 第三部分第（1）至（5）项禁止按函数名、特定字符串、输入数据、参数个数、精确输入值或
  其他特定输入模式激活专用路径；禁止硬编码结果、利用未定义行为，以及不能保证一般输入
  正确性的猜测性优化。

原文位置：

- `doc/sources/official_compiler2026_text/2026全国大学生计算机系统能力大赛编译系统设计赛-章程.txt`：第 7.4、7.5 条。
- `doc/sources/official_compiler2026_text/关于编译优化合理性及相关违规行为认定的说明.txt`：第二、三部分。

### 1.2 参考框架事实，不是竞赛条款

- `work/official_triton_agent/genetic_operators.py` 的生成 prompt 要求 `Output ONLY the code`。
  技术方案没有这句原文；本项目把它保留为候选生成阶段的输出协议。
- `work/official_triton_agent/readme.md` 只要求“不要大幅修改” `executor.py`、
  `llm_interface.py`、`optimizer_agent.py` 和 `main.py`，并非技术方案中的绝对禁止条款。
- 技术方案只明确“至多 5 种版本”，没有给出 Python 返回字段和候选文件名。官方参考框架
  当前返回字段与 `_best.py`、`_v1.py` 至 `_v5.py` 命名在获得澄清前保守冻结。
- 技术方案第 14 条的重复率阈值是 20%，总章程第 7.3 条是 50%；采用更严格的 20%，并保留
  向赛方确认的待办。

### 1.3 多次调用与历史反馈的准确结论

- 官方规则没有“每代或每次进化只能调用一次 LLM”的限制；第 4 条限制的是每算子累计
  20 分钟或 20 万 token，任一上限先到即停止。因此多次调用和 multi-agent **允许但不免费**，
  不能表述为“不受限制”。
- 参考框架 README 明确建议 crossover 携带父代 fitness，executor 也返回 fitness、耗时、
  speedup 和 error。这能证明参考框架预期复用历史评测结果，但 README 仍不是竞赛条款。
- 只使用官方接口真实暴露、与候选和环境绑定的 compile/runtime/fitness 数据。技术方案没有
  保证任意 profiler、IR 或隐藏测试细节可见；不可见字段保持 `unknown`。
- Evidence 可以保存 operator、case label 和原始日志作为 provenance；注入模型前必须投影为
  脱敏的失败类别、计数或摘要。禁止按函数字面名称、case ID、单个公开 case 的精确输入或
  其他测试指纹选择专用路径；这不等于禁止根据源码语义、通用计算模式和一般接口契约选择
  reduction、scan、layout、quantization 等通用优化策略。后者是工程合规解释，不冒充官方原文。

## 2. 提示词工程清单

### 2.1 本职责必须做

- 版本化 system message、任务模板和 `prompt_id`，使每个候选可追溯到父代、模型、模板和调用。
- 保持候选生成回复只有一份完整 Triton Python 源码；分析和审查的内部调用才允许结构化报告。
- 保持外部调用契约和可观察语义。decorator、wrapper 内部、grid 和 launch 默认冻结；只有
  当前 MutationPlan 明确开放且对应确定性门能够验证时才可修改。`local_rewrite` 默认不开放
  这些表面，不能用笼统的“接口不变”永久排除合法内部并行变换。
- 只注入真实可观测的静态或官方反馈；缺失的 IR、profile、编译阶段和正确性阶段写 `unknown`。
- 对所有运行时 agent 调用统一统计 token 和墙钟，并为生成合法 Top-5 预留预算。
- 用确定性静态门和真实 executor 验证模型输出，不相信模型对正确性或性能的自述。
- 记录 Codex/Claude、比赛运行时模型和 doge-code/skill-opt 借鉴范围，满足 AI 与第三方
  来源披露要求。

### 2.2 严格禁止做

- 禁止人工修改 LLM 已生成的 Triton 候选；修复必须产生新的、带 provenance 的 agent 候选。
- 禁止按函数字面名称、测试 case 标识、单个公开 case 的精确 shape/value 或特定字符串选择
  专用优化或激活候选分支；不禁止基于源码和一般算子语义的通用策略选择。
- 禁止硬编码测试结果、利用未定义行为、只对公开输入正确的猜测性优化。
- 禁止虚构 `speedup`、latency、IR、profiler、编译成功或正确性通过；本地 proxy 不得叫 speedup。
- 禁止修改测试脚本来适配候选，或让候选要求现有测试改变调用方式。
- 禁止把 reviewer 的 `accept` 当作功能通过，或把聚合 `success` 擅自拆成已证明的编译/正确性阶段。
- 禁止返回 baseline 原样、仅注释变化或仅诊断文本变化的候选冒充优化。
- 禁止照搬 doge-code/skill-opt 的大段源码、命令或提示词文本；如确需复制，先单独做
  许可证和披露验收。

### 2.3 未澄清前禁止修改

- `OptimizerAgent.setup()/optimize()/save_results()` 的签名、返回 key 和 Top-5 元素字段。
- `executor.py` 的调用方式、测试链、真实性能口径和环境路径。
- `_best.py`、`_v1.py` 至 `_v5.py`、`_stats.json` 的当前输出命名和布局。
- 官方入口 `main.py`、提交产物外层结构，以及官方尚未确认的额外返回/provenance 字段。

“不能修改输出”目前没有对应的技术方案原文。准确结论是“不可改变接口调用方式、最终至多
5 个版本”；schema 和命名仍未确认，所以工程上先冻结，而不是把冻结项伪装成竞赛条款。

## 3. 当前实现诊断

当前工作区的比赛式生成链是：

```text
GeneticOperators.mutate()/crossover()
-> mutation 显式 override、证据规则或 20% 随机探索
-> 拼接版本化 Skill、父代和可选 repair guidance
-> LLMInterface.generate(prompt, system_msg=静态 system v2)
-> 清理 Markdown 标记
-> 候选静态门与 executor
```

已存在、应直接复用的能力：

- `work/official_triton_agent/genetic_operators.py` 已有静态 system v2、三种版本化 mutation Skill、
  可选 repair overlay、共享 `INTERFACE_CONTRACT_RULES` 和代码清理。
- `work/official_triton_agent/contract_executor.py` 会在昂贵评测前用 baseline AST 检查语法、函数签名
  和装饰器。
- `wlz_optimizer/executors.py` 的本地静态工具还检查导入、Triton 语义和目标设备，但它不是官方
  executor 的返回 schema。
- `wlz_optimizer/repair_guidance.py` 已从已导入的官方失败历史生成有界、粗粒度 guidance；不应再
  新建一个平行历史库。
- `wlz_optimizer/stdlib_llm.py` 会保留 API `usage` 和 prompt SHA-256；官方
  `llm_interface.py` 仍用 `split()` 粗估 token，不能充当 20 万 token 的硬预算器。
- `work/prompt_skill_lab.py` 是 doge-code 机制的隔离原型，不在官方运行路径中；其 Skill 和
  renderer 不能与 `genetic_operators.py` 发展成两套长期事实源。
- `wlz_optimizer/budget.py` 已提供 monotonic deadline、逐调用 reserve/commit/release、usage fallback
  和 unknown in-flight fail-closed；`StdlibOpenAIClient` 已接入该账本，EA 主流程仍未改动。
- `wlz_optimizer/prompt_context.py` 已提供与父代、code hash 和环境绑定的只读投影；生产 Renderer
  现在只消费其白名单统计字段，并记录 sanitization version。一般 shape/rank、dtype、访存和性能摘要
  已接入；精确 shape contract 仍不直接进入 Prompt。

当前缺口：

- Budget Ledger 已接入 `StdlibOpenAIClient`；官方旧 `llm_interface.py` 仍用 `split()` 粗估 token，
  因而不能把旧接口的统计当成 20 万 token 的硬预算器。
- 当前 OperatorPolicy 有规则选择和 20% 随机探索，不能保证 checkpoint 重放；该调度属于 EA
  搜索策略，不由提示词工程继续扩张。
- PromptContext 的白名单统计已经进入 mutation/crossover 的 user prompt；raw log、case ID、环境指纹
  和未知 metadata 仍不可见。该接线没有证明 Ascend 性能收益。
- 官方路径已有唯一生产 Skill registry/纯 Renderer；`prompt_skill_lab.py` 仍只是隔离研究样品，
  不得被接成第二套 registry。
- `repair_guidance.py` 当前会输出 case label；即使 repair prompt 声明其仅作 provenance，仍应在
  PromptContext 层改为类别/计数，避免模型看到不必要的测例标识。
- ShapeObservation 已存在于本地审计层，但未接入比赛式生成。可以使用从源码、接口和一般输入
  契约确定的动态维、reduction 维、dtype、stride、尾块和整除关系；多个独立输入汇总出的区间
  或桶只在留出验证存在时谨慎使用。单个公开 case 的精确 shape/value 永不作为特化依据。
- 官方返回的失败阶段仍可能只有聚合状态，不能把历史统一标成 `compile_fail` 或
  `correctness_fail`。

## 4. 参考工程和研究机制

### 4.1 doge-code：运行时 prompt/Skill 装载

参考对象：`ref/doge-code`。这里只学习机制，不复制实现。

| 机制 | 源码证据 | 本项目采用方式 |
| --- | --- | --- |
| system prompt 有明确的 replace/append 优先级 | `src/utils/systemPrompt.ts:28-40,56-122` | 固定一个静态基础 prompt；任务上下文进入 user prompt，避免互相覆盖。 |
| 稳定 section 缓存，动态 section 明确标为会破坏缓存 | `src/constants/systemPromptSections.ts:16-37` | system message 只放稳定契约；候选、反馈和 shape contract 不放 system message。缓存不等于比赛 token 免费。 |
| Skill 列表只暴露名称/描述，完整内容调用时加载 | `src/skills/loadSkillsDir.ts:96-105,344-398` | 只注入当前 mutation type 对应的模板，不把所有 Skill few-shot 塞入每次请求。 |
| Skill discovery 有独立上下文预算 | `src/tools/SkillTool/prompt.ts:20-40` | Skill 路由元数据必须短小；首版用确定性路由，不再调用一个 LLM 做路由。 |
| Skill 元数据声明工具、模型、inline/fork 和 path 条件 | `src/types/command.ts:25-56` | 后续 Skill 规范显式声明输入、输出、允许反馈、预算和版本；默认 inline 单调用。 |
| 条件 Skill 只有命中路径后才激活 | `src/skills/loadSkillsDir.ts:771-802` | 采用渐进披露；不为每个算子预加载所有策略。不按函数字面名称或精确测例值路由，可按源码的一般计算语义路由。 |
| forked Skill 有独立上下文和预算 | `src/tools/SkillTool/SkillTool.ts:118-123` | 比赛规则要求所有 agent token 合计，因此默认不开 fork；只有消融证明收益才启用。 |
| 压缩后恢复 Skill 有单项和总预算 | `src/services/compact/compact.ts:122-130,1494-1524` | 反馈历史做条数、字符数和 token 上限；优先最近且与精确父代绑定的真实反馈。 |

doge-code 的目标是通用编码代理，拥有约 200K 甚至更大的会话上下文；本赛题是每算子 20 万
token 和 20 分钟的受限搜索。不能照搬其通用工具系统、长会话压缩、动态子代理或完整 Skill
发现机制。

`ref/doge-code` 不作为可分发的正常开源依赖。比赛仓库、提交包和公开附件不复制其源码、
长 prompt、命令或注释；这里只保留机制观察，并以本项目独立实现和公开资料作为可披露依据。
“竞赛允许借鉴”不等于“第三方许可证允许复制”。

### 4.2 skill-opt：开发期 prompt/Skill 版本优化

参考对象是私有仓库 `SixEternals/skill-opt` 的提交
`41d97998551631a5b05fec11b2a4f9a18d5a9edf`，以及其包含的 SkillOpt-Lite 论文
`arXiv:2607.03451v1`。该提交实际只有四份 `.claude/commands/*.md`、说明、setup 脚本
和论文；没有 README 所需的 `run.sh`、adapter、评测器、持久化状态机或测试。GitHub
也未识别到 license，因此本项目只借鉴机制，不复制文本或脚本。

| 候选机制 | 参考证据 | 本项目的保守采用方式 |
| --- | --- | --- |
| train 只产生改进信号，val 只决定接受/拒绝，test 最后只跑一次 | `.claude/commands/skillopt-loop.md:20-28,155-159,201-204`；论文第 2.2、3.2 节，PDF 第 5-9 页 | 用于开发期 prompt 版本 A/B；不把 val/test 内容反馈给模板。 |
| 失败先聚类，再读取代表样本，并用通过样本反证 | `.claude/commands/skillopt-improve.md:23-39` | 只从多个独立记录中提取通用规则；先去除 operator 标识、精确 shape/value 和秘密。 |
| 只修正重复出现的共性，一轮做最小可归因补丁 | `.claude/commands/skillopt-improve.md:41-54,80-85` | 一个 prompt 版本只检验一个假设；超出小验收单元则停止，不照搬其“4 处/40%”经验阈值。 |
| baseline/before/best/after 快照，val 回归时恢复 best | `.claude/commands/skillopt-loop.md:30-65,93-132,146-159` | 保留版本 ID、哈希、固定评测配置和决策原因；回退由确定性 controller/JJ 变更完成。 |
| accept/reject/flat 与连续无收益停止 | `.claude/commands/skillopt-loop.md:79-114,139-144` | 保留状态机形状；噪声区间必须根据本项目样本量和重复试验估计，不照搬 `±0.01/±0.05`。 |
| Skill 内容与 harness 修复分开归因 | `.claude/commands/harnessopt-improve.md:5-29,67-107` | 先判定是 prompt、生成编排、静态门还是 executor 缺陷；不用增加 prompt 规则掩盖代码问题。 |

这些都是**待本项目验证的实验纪律**。该参考仓的 gate、计分解析、回滚和 best tracking
都由模型按 Markdown 命令执行，不是独立 controller 的强制行为；其实验数字也不能证明对
Ascend Triton 有收益。不移植其 `rm -rf`、`git reset --hard`、强制移动 tag、固定死区或
“每个改动都加 feature flag”的编排。

## 5. 四层架构

### 第一层：稳定、分层的 system message

system message 只包含稳定内容，按以下顺序构造：

1. **角色与目标**：Ascend Triton 优化；正确性和通用性优先于性能。
2. **竞赛合规**：禁止测例特化、硬编码结果、未定义行为和人工式修补。
3. **接口契约**：外部调用方式和可观察语义不变；内部 decorator、wrapper、grid 和 launch
   默认冻结，只有当前 MutationPlan 明确授权且对应门禁可验证时才开放。
4. **输出协议**：生成阶段只返回一份完整源码，无解释、Markdown fence、测试或多个方案。

下一版 system message 的语义草案（尚未写入生产代码）：

```text
You generate one complete Triton Python candidate for Ascend NPU optimization.
Correctness, generality, and interface compatibility take priority over speed.
Never specialize for a function name, test-case identifier, exact observed input
shape/value, or other benchmark fingerprint. Never hardcode results or rely on
undefined behavior.
Preserve the externally visible calling contract and observable semantics. Existing
tests must call the candidate unchanged. Modify decorators, wrapper internals, launch
grid, tile, or launch settings only when the activated mutation plan explicitly allows
that surface and a deterministic gate can validate it; otherwise preserve it.
Return exactly one complete source file and nothing else: no explanation, Markdown,
tests, diff, or alternative version.
```

动态内容全部放在 user prompt：MutationPlan、父代源码、真实 fitness、已验证的粗粒度失败和允许的
一般语义/shape contract。公开 case 的精确 shape/value、合成 speedup 和未验证硬件结论不得注入。

### Contract Matrix：外部契约与内部自由度

Contract Matrix 是 system、MutationPromptSpec 和静态门共享的语义来源，不是第二套 Candidate
schema。当前门不能验证的表面继续冻结；不能仅凭模型承诺开放。

| 级别 | 内容 | 规则 |
| --- | --- | --- |
| 绝对冻结 | 官方可见函数名；外部参数名、顺序、kind、required/default 状态、影响功能语义的默认值和必要的 `tl.constexpr` 契约；测试调用方式；输出 schema/布局；外部语义 | 任何 Skill 都不得改变。 |
| 默认冻结、计划可开放 | 只编码 tile/launch 的默认值、decorator 可优化部分、wrapper 内部、grid 表达式和维度、`BLOCK_SIZE`、`num_warps`、`num_stages` 等 launch 参数 | 只有 MutationPlan 显式列入 `allowed_diff_surfaces` 且对应门可验证时修改。 |
| 原则上可优化 | kernel body、program-id 映射、通用 tile、访存顺序、mask、reduction、等价局部重写和合法内部并行变换 | 仍须通过一般输入 correctness 和目标环境评测。 |
| 永久禁止 | 修改测试、测试指纹分支、硬编码答案、未定义行为、伪造 measurement、改变外部调用契约、原样/no-op 冒充优化 | 不得由 MutationPlan 开放。 |

现有 `contract_executor.py` 只能确定性检查顶层函数语法、签名和 decorator，不能证明 wrapper/grid
语义兼容。因此在增加对应门之前，生产 prompt 仍必须保守冻结 wrapper/grid 变化。

Shape/语义上下文按三层处理：

- **允许**：从源码、接口和一般契约得到的动态维、reduction 维、dtype、stride、连续性、尾块和
  通用整除/对齐条件。
- **谨慎允许**：多个独立输入汇总出的区间、桶或一般关系；必须配套不相交留出验证，并继续
  支持区间外和非整除输入。
- **禁止特化**：单个公开 case 的精确 shape/value、case label、精确参数元组及其反推分支。

### 第二层：按需加载的 Skill

Skill 首先是**版本化 prompt 规范**，不是一开始就新增五个 Python 类。每个 Skill 至少声明：

```text
skill_id / version / purpose / allowed_inputs / forbidden_inputs
output_protocol / max_context_tokens / mutation_kind / validation_gates
```

首批候选 Skill 与现有 mutation type 一一对应：

- `param_tuning`：可调整 MutationPlan 开放的通用 tile/launch 配置，例如 `BLOCK_SIZE`、
  `num_warps`、`num_stages`；默认不改变算法结构或外部调用契约。
- `strategy_change`：改变通用内存访问、program-id 映射或并行策略；只有计划和门禁同时允许时
  才可调整内部 grid 结构，必须保留外部调用方式、算法语义和一般输入正确性。
- `local_rewrite`：局部等价改写；禁止修改接口、wrapper 和测试调用，默认不修改 tile/launch
  配置。
- `repair`：只接收与精确父代绑定的、实际观测到的粗粒度失败，并继承一个底层 mutation kind；
  它不是可以绕过 Contract Matrix 的无限变换类型。

每次生成前由确定性代码形成最小 MutationPlan：

```text
mutation_kind / parent_ids / allowed_diff_surfaces / frozen_surfaces
feedback_scope / max_change_scope / validation_gates / skill_id / skill_version
```

首版只需一个不可变值对象和现有调用者，不建立插件系统。静态门尚不能验证的 diff surface 不得
仅靠 prompt 声明为开放。

路由先由已有 `mutation_type` 确定，不额外消耗一次 LLM 调用。`shape_analyzer` 和
`correctness_reviewer` 只作为后续受控实验；前者不得使用测例指纹，后者不得代替真实验证。

### 第三层：门控的多步流水线

不把每次 mutation 固定扩成“分析 -> 生成 -> 审查 -> 修复”四次 LLM 调用。默认链路是：

```text
Step 0  确定性准备：提取 baseline 接口契约、选择 mutation type、裁剪真实反馈
Step 1  一次 LLM 生成：只输出完整候选源码
Step 2  确定性预检：去 fence、语法/签名/装饰器/设备检查、原样与 no-op 检查
Step 3  真实评测：只有通过廉价门的候选才进入 executor
Step 4  可选修复：仅有可信且可行动失败时调用一次，生成新 candidate id
```

独立 LLM analyzer/reviewer 只有在单调用基线稳定后做 A/B。内部分析/审查可输出 JSON；Step 1 和
Step 4 的候选生成仍必须只输出源码。

### 第四层：证据约束的反馈闭环

反馈记录复用现有 Candidate、官方历史和 repair guidance，至少绑定：

```text
candidate_id / code_hash / parent_ids / generation / mutation_kind
model / prompt_id / prompt_version / request_fingerprint / env_fingerprint / API usage
observed_status / raw_status / verified_failure_kind / official speedup or unknown
```

反馈注入规则：

- 只使用精确父代和相同环境的真实记录，限制条数、字符数和 token。
- operator 名只用于 provenance/检索；通用策略路由使用源码和计算语义，不使用字面名称。
- 只有独立阶段确实可见时才写 `compile_fail`、`correctness_fail`；否则保持聚合状态或 `unknown`。
- 只有真实 A2/A3 或官方输出才能写 speedup；合成 few-shot 不能伪装成成功历史。
- Few-shot 只从有 provenance 的已验证候选中选取，并去除 test case 标识和精确输入指纹。

父代源码、候选注释、stdout/stderr 和错误文本都视为不可信数据：

- system contract、SkillSpec 和 MutationPlan 是 trusted instruction；父代源码进入明确的
  `PARENT_SOURCE` 数据区，代码或注释中的自然语言不得覆盖上层指令。
- raw stdout/stderr、case ID 和精确输入只留在 Evidence/provenance，默认不对模型可见；repair
  只接收枚举化、限长的 `EXECUTOR_SUMMARY`。
- 投影记录 `sanitization_version`。不要求为每种来源扩张 Candidate schema；来源种类、信任级别
  和可见性可由 PromptContext 强类型与 Renderer 固定分区表达。
- 分隔符只能减少上下文混淆，不能证明消除 prompt injection；真实安全边界仍是结构化投影、
  不暴露 raw log 和确定性门。

`prompt_sha256` 只标识规范化 messages。调用前另计算不含秘密的 `request_fingerprint`，至少覆盖
messages hash、请求模型、temperature、top_p/seed（若存在）、completion 上限、stop、system/
Skill/repair/Renderer/Gateway 版本和脱敏 provider ID。响应 ID、API 返回的实际 model、usage、
起止时间、finish reason、timeout/retry 次数在响应后单独记录；API key 和原始 endpoint 不落盘。

本层包含两个不同时间尺度的闭环，不得混为一次超长运行：

```text
运行时内环（每算子）：prompt policy vN -> 生成候选 -> 确定性门/真实评测 -> 精确父代反馈
开发期外环（跨算子）：聚合脱敏轨迹 -> 最小 prompt 补丁 -> 独立 val A/B -> 接受新版本或保留 best
```

- 内环修复仍只接收与精确父代 code hash 和环境绑定的粗粒度真实失败。
- 外环默认不放入提交 agent 的每算子 20 分钟/20 万 token 搜索；若未来放入，其所有
  optimizer/reviewer token 与墙钟都必须进入同一硬预算。
- 外环 train 只读脱敏后的失败类别和已验证轨迹；val 与 train 严格不相交且不向改写器
  暴露内容；test 在版本选择结束后只使用一次。官方线上结果不作为可反复窥探的
  prompt 外环 train/val。
- 每次外环只改一个可归因假设，并保留 `system_prompt_version`、template version、
  完整 prompt hash、固定模型/种子/预算、样本划分哈希和 accept/reject 原因。
- 已完成的 system/provenance 基线只是外环的 round-0，不能因 focused contract test 通过就
  宣称 prompt 优化有性能收益。

### 运行时职责边界

不按“Evidence Store / Context Builder / Planner / Skill Scheduler / Renderer”五个新模块直接
落地。当前仓库已有 Candidate、EvaluationResult、cache、official history 和 repair guidance；
新增同类持久化会形成第二套事实源。收敛后的运行时关系是：

```text
                         BudgetController
                                |
Candidate / Evaluation / Cache / Official History
                                |
                           EvidenceView
                            /          \
                           v            v
                 PromptContextProjector  OperatorPolicy
                            \            /
                    ContractMatrix + MutationPlan
                                    |
                    MutationPromptSpec Registry
                                    |
                           Pure PromptRenderer
                                    |
                            DeepSeek LLM Gateway
                                    |
                         Static Gates / Executor
                                    |
                         append existing evidence
```

职责和所有权：

| 概念 | 决策 | 所有权 |
| --- | --- | --- |
| `BudgetController` | 必须独立；统一实际 API usage、保守预留、monotonic deadline 和 stop reason | 运行时基础设施 |
| `EvidenceView` | 只读聚合现有事实，不新增数据库，不摘要、不规划 | Candidate/cache/history 所有者 |
| `PromptContextProjector` | 确定性过滤、shape 分级、来源隔离、脱敏、裁剪并返回结构化上下文；不建服务 | 提示词工程 |
| `ContractMatrix` | 定义绝对冻结、计划可开放、默认可变和永久禁止表面；Prompt 与门禁共享语义 | 跨模块窄契约 |
| `OperatorPolicy` | 选择 mutation kind；可重放调度和 bandit 均属于 EA，不由 Renderer 实现 | EA 搜索策略 |
| `MutationPlan` | 把本次允许/冻结表面、父代、反馈范围和验证门显式化；不做搜索决策 | 提示词工程输入契约 |
| `MutationPromptSpec Registry` | 单一版本化来源；一次只激活一个 mutation 规范，可叠加有界 repair | 提示词工程 |
| `PromptRenderer` | 纯函数；不查库、不选策略、不调用模型、不更新权重 | 提示词工程 |
| `LLMGateway` | DeepSeek 调用、request fingerprint、usage、超时和错误；不得承担搜索决策 | LLM 运行时 |

Skill 权重和确定性轮换都属于 Adaptive Operator Selection，而不是 Prompt Engineering。UCT、逐步 LLM
Planner、插件系统和 multi-agent reviewer 在当前数据与预算地基未完成时均为 YAGNI；只有相同
总 token/墙钟 A/B 证明其优于“多生成并验证一个候选”后才能进入主路径。

## 6. 与现有代码的对接

### 已有基线单元

- `work/official_triton_agent/genetic_operators.py` 已有共享静态 system v2、唯一生产 mutation
  registry、纯 mutation/crossover Renderer、三种版本化规范、MutationPlan、Prompt v2 和有界
  repair overlay。
- `wlz_optimizer/stdlib_llm.py` 已按最终消息序列计算 `prompt_sha256`，并记录版本、请求指纹、
  usage、实际响应模型、墙钟、超时/重试状态和响应哈希；不记录完整 messages、endpoint、服务端
  错误正文、密钥或候选源码。
- `wlz_optimizer/budget.py` 已实现逐调用 `reserve/commit/release/uncertain` 账本并接入
  `StdlibOpenAIClient`；未知调用状态按 fail-closed 记账。
- `wlz_optimizer/prompt_context.py` 已将精确父代绑定、环境绑定、失败类别、聚合 rank/dtype、父代
  源码 AST 访存计数和官方性能摘要接入 Renderer；精确 shape、tensor/case 名和 raw log 不进入
  prompt，父代源码放在显式不可信数据分区。
- `wlz_optimizer/repair_guidance.py` 已实现精确证据、一次 lineage 尝试和剩余预算共同约束的
  `RepairDecision`；生成器传播 repair 次数，第二次 repair 在 LLM 调用前拒绝。
- `tests/test_official_prompt_contract.py` 已覆盖静态 system、Skill 渐进披露、repair overlay、
  消息角色边界和 brace escaping。
- `work/prompt_skill_lab.py` 与 `tests/test_prompt_skill_lab.py` 只用于隔离验证 doge-code 的发现、
  激活和纯 renderer 机制；未接入官方链，后续不能作为第二套生产 registry。

### 后续按证据再改

- Contract Matrix、MutationPlan 和不可信数据边界已经统一；后续 prompt 版本只能由跨家族等预算
  A/B 和真实 Ascend 结果驱动，不能因架构闭合直接宣称生成质量或性能提升。
- repair-vs-fresh 开发期 A/B 已拒绝当前 repair overlay；生产默认继续 fresh generation，只有新的
  等预算、跨轨迹证据推翻该结果后才重新启用 repair 生成路径。
- 生产 registry 已收敛；隔离 lab 不接入官方链，也不演化为通用 Agent/插件框架。
- 只有单调用 A/B 证明 reviewer 或多步流水线在同预算下提升有效候选率，才接入额外 LLM 调用。

### 明确不改

- 不改 `EAConfig`、`LLMInterface.generate()`、`Individual`、`GeneticOperators` 的对外签名。
- 不改 `optimizer_agent.py`、`executor.py`、`main.py`、Top-5 返回字段和输出文件命名。
- 不改现有静态门、Candidate schema、官方历史和 repair guidance 的所有权边界。
- 不先建 `wlz_optimizer/skill_executor.py`，不让比赛式 `work/official_triton_agent` 反向依赖一套新框架。
- 不把 `wlz_optimizer/genetic_operators.py` 的本地 mock 路径当成官方比赛生成路径。
- 不接入单个公开 case 的精确 ShapeObservation。一般源码/接口 shape contract 可在独立单元中
  接入，但必须先有分级投影、留出泛化门和不使用测试指纹的检查。

### 变更面归属与硬边界

借鉴 skill-opt 的“Skill 与 harness 分开归因”，但不开启 HarnessOpt。先判定失败的所有者，
再建立一个小验收单元：

| 失败面 | 可能现象 | 处理边界 |
| --- | --- | --- |
| prompt/Skill 内容 | 多个独立轨迹都忽略同一稳定契约 | 可做最小模板补丁和独立 A/B。 |
| 生成编排 | prompt 版本、消息角色、调用记录或预算器错误 | 在所有者模块单独修复，不向 prompt 堆叠规则。 |
| 静态门/executor | 候选被错分类、执行阶段丢失或环境未配置 | 独立代码单元；不由 prompt reviewer 代替真实门。 |
| 评测器/输入空间/官方接口 | 分数、case 或 schema 限制不利于当前假设 | 永久 denylist；不为提高通过率修改。 |

已完成的 system/registry/Renderer 是可测试基线，不是性能收益。Budget 和 PromptContext 目前是
未接线的隔离原型；OperatorPolicy 已存在但归 EA 所有。不得把这些项目合并报告为“P0 闭环完成”。

## 7. Token 预算估算

以下是基于本地源码字符长度的 token 规划估算，不是 tokenizer 或 API 实测，也不是
官方计费事实。正式数值必须读取 API `usage` 并按赛方统计口径校准。

本地只读校准：当前 21 个公开 baseline 源码的中位长度是 3,491 字符，最大 8,759 字符。以
`_count_expert_num_tokens` 为样本，现有 mutation 的 user + system 输入是 4,259 字符，crossover
是 6,792 字符；按 4 字符/token 粗估分别约 1,065 和 1,698 input token，均未包含模型输出。
代码的真实 tokenizer 比例可能不同，不能用该粗估执行硬停止。

| 调用/组件 | 规划 token 范围 | 说明 |
| --- | ---: | --- |
| 稳定 system message v2 | 150-300 / 每次调用 | 即使内容相同也不能假定在比赛统计中免费。 |
| mutation 指令与有界反馈 | 300-900 | 不含 kernel 源码。 |
| 每份父代 kernel 输入 | 500-3,000 | 按当前 corpus 保守规划，必须按 API usage 校准。 |
| 候选源码输出 | 1,000-3,500 | 当前配置单次 completion 上限是 4,096 token。 |
| **单父代生成合计** | **约 2,000-7,500** | 第一版 mutation 基线。 |
| **双父代 crossover 合计** | **约 3,000-10,500** | 两份源码都会进入输入。 |
| 分析 + 生成 + 审查 | **约 7,000-16,000** | 仅后续 A/B，三次调用都重复计入上下文。 |
| 再加一次修复 | **约 10,000-23,000** | 只对可信、可行动失败触发。 |

预算策略：

- 官方硬上限是每算子 200,000 token 和 20 分钟，任一先到即停。
- 每次 LLM 调用前先预留 `estimated_input + max_completion + safety_margin`，并检查预计调用时间与
  必要收尾时间；预留成功后才能发请求。
- 调用完成后用 API 实际 usage commit 并 release 未使用预留。usage 缺失使用显式保守上界；
  timeout、网络中断或未知调用状态不能自动按 0 token 结算，自动重试必须单独预留和记录。
- 先保留至少 20,000 token 和 2 分钟用于收尾、去重和形成合法 Top-5，搜索可用预算按
  180,000 token / 18 分钟规划；这是首版默认值，后续只能依据可追溯的 p90/p95 收尾观测调整。
- 按单父代生成约 5,000-7,500 token 粗算，理论上约 24-36 次；按三步约 12,000-16,000 token
  粗算，约 11-15 次。真实上限还会受 kernel 长度、crossover、修复、API latency 和 executor
  时间限制。
- 所有 Skill、multi-agent、review、repair token 合并到同一个计数器。缓存命中是否减少官方
  计费没有证据，预算器不得假定减少。
- SkillOpt-Lite 论文只说在受限 token 下选择高价值轨迹，没有可复用的 token/墙钟硬停止器
  或与基线等成本的证明；不得用该论文替代本项目预算实现。
- 当前参考 `LLMInterface` 的 `split()` 估算不准确且不强制停机；生产客户端现已接入调用级
  `reserve/commit/release/uncertain` 账本。后续多步调用必须复用该账本，不能建立旁路计数器。

## 8. Runtime 分阶段实施计划

每一项都是独立验收单元，完成并报告后停止，不跨项连做。排序的唯一目标是固定 20 万 token / 20
分钟内提高 best valid speedup，而不是追求模块齐全。

### P0：进入自适应搜索前必须完成

0. **Contract Matrix**（已完成）：system、Skill、结构化上下文和静态门已使用同一套冻结/开放
   边界；门不能验证的 wrapper/grid 仍冻结。
1. **Budget Ledger**（已完成）：`reserve/commit/release/uncertain` 与生产
   `StdlibOpenAIClient` 接线；未知调用状态 fail-closed；未修改 EA 流程。
2. **EvidenceView + PromptContext**（已完成）：精确父代绑定、环境绑定、失败类别投影、raw log
   不可见、父代源码不可信分隔区和 sanitization version 已接入 renderer。
3. **唯一 Registry/Renderer**（已完成）：生产路径只有 `MUTATION_SKILLS` 一份事实源；doge-code
   机制实验保留为通用隔离样品，不再保存第二套比赛 Skill 文案；每次只激活一个 Skill。
4. **MutationPlan**（已完成）：不可变计划已绑定父代 SHA-256、mutation kind、允许/冻结表面和
   计划版本；它不选择 mutation，不建立插件系统，也不承担 OperatorPolicy。
5. **请求可观测性**（已完成）：在 `prompt_sha256` 之外记录 request fingerprint、实际响应模型、
   usage、墙钟、retry/timeout、安全错误类别和响应哈希；A/B harness 另记录脱敏后的静态门漏斗，
   两条路径均不保存完整 prompt、源码、endpoint、服务端错误正文或秘密。
6. **等预算配对 A/B**（已完成开发期代理实验）：layout/packing、normalization、stateful/scan
   三个家族已在相同 token/墙钟上限下完成裸 DeepSeek A/B，并使用生产清理和静态门口径。
   该实验没有 Ascend correctness/performance，不能作为 prompt 性能接受证据。

确定性 round-robin、最低探索和 cost-aware bandit 均归 EA 搜索策略所有，不列为提示词工程 P0。
当前 20% 随机探索的可重放问题应交给 EA 单独验收，不能借本方案继续修改主线搜索。

### P1：可信反馈和 Prompt 成本优化（已完成开发期验收）

1. **确定性摘要**（已完成）：PromptContext v2 聚合失败类别、rank/dtype、父代源码 AST 访存操作
   和官方通过结果的 speedup/latency；不调用 LLM 总结，不注入精确 shape、tensor/case 名或 raw log。
2. **有界 Repair**（门控已完成，当前 overlay 已拒绝）：只有精确父代/观测绑定的可行动官方失败、
   lineage 尚未 repair 且预算充足时才允许一次尝试。真实失败轨迹的等预算开发期 A/B 显示 repair
   明显差于 fresh generation，因此生产默认不触发 overlay，保留门控和拒绝证据。
3. **MutationPromptSpec 压缩**（已完成一个假设）：Prompt v2 只删除 system、MutationPlan 和 Skill
   已覆盖的角色说明、任务复述与结束口号；接口规则、Skill 内容、非空变更和只输出源码协议不变。
   跨家族等上限 A/B 未降低静态有效/唯一候选数，并小幅降低实际 token，暂保留观察。

### P2：有时间再做的受控消融

1. 只在搜索停滞且剩余预算充足时试一次 LLM Planner；不得每个 child 都调用。
2. 只有官方接口真实暴露并能绑定环境时才注入 profiler/IR 摘要。
3. 门控 reviewer 或 multi-agent 仅在相同总预算 A/B 显著提高 Top-5 功能通过率时保留。

### P3：论文扩展，不进入当前主路径

- UCT/MCTS、学习型 OperatorPolicy、跨算子迁移、cost model、prompt 自进化和多 Agent critic。
- 插件系统只在至少两个独立实现和真实替换需求出现后再抽象。

### 开发期 prompt/Skill 版本实验协议

此协议从 P0-3 的 registry/renderer 收敛开始使用，不扩大 P0-1 预算单元：

1. **配对等预算**：同一 parent batch、算子/家族、模型配置和门禁下交替运行 A/B；每个变体使用
   相同总 token 与墙钟上限，而不是只保证调用次数相同。远程 seed 不保证严格可复现，必须重复
   多个独立 batch；一轮只改一个 prompt/Skill 假设。
2. **训练证据**：先按已验证失败类别聚类，至少两条独立记录支持才进入全局模板诊断；
   抽取通过记录反证，不处理单例、operator 名、精确 case shape/value 或未验证根因。
3. **数据隔离**：21 个算子按 reduction/normalization、scan/stateful、quantization、
   transpose/layout、scatter/gather、elementwise 等家族分层。train 用于诊断；密封 val 只返回
   聚合 accept/reject 指标；test 在版本选择结束后只跑一次。优先 family-stratified 或
   leave-one-operator/family-out，不做会混淆家族差异的简单随机切分。
4. **分层指标**：生成层记录 unique/non-noop、static/import/correctness admission；成本层记录
   token/unique valid、token/正确性准入、墙钟/准入候选、重复和失败调用成本；真实性能层只接受
   Ascend 远程或官方的 functional pass、相同预算 best valid speedup 和 Top-5 通过数。
5. **决策与停止**：主指标是相同 token/墙钟预算下的 best valid result；在没有真实 Ascend 时，
   只报告前两层代理指标。只有重复实验超过噪声且接口破坏、特例、重复、预算和 raw-log 泄漏
   护栏不退化才 accept；否则记录 flat/reject 并保留 best。

2026-07-23 的 round-0 裸 DeepSeek smoke 只覆盖 `_pack_seq_kernel`、每组 3 次静态检查：旧 prompt
与新 P0 prompt 均为 3/3 static pass，新 prompt 裸源码协议为 2/3，平均 token 高 11.8%。重复
candidate hash 表明样本不独立。该结果只能说明“未观察到新 prompt 优势”，不能证明任一版本的
Ascend correctness 或 speedup，也不满足本节的等预算跨家族接受协议。

同日 round-1 使用官方允许且 API 实际提供的 `deepseek-v4-pro`，仍只覆盖 `_pack_seq_kernel`、A/B
各 3 次：旧 prompt 为 3/3 裸源码、3/3 原始语法和接口门，共 5,241 token；当前 P0 prompt 为
2/3 裸源码、2/3 原始语法和接口门，共 6,413 token，token 高 22.4%。B 的失败样本含 Markdown
fence，生产 `_clean_code()` 会清理 fence，因此该原始失败不等于最终候选失败。两组各有 3 个唯一
hash，但样本仍太小且没有 Ascend 验证，只能继续判定“未观察到 P0 prompt 优势”。摘要位于被忽略的
`output/prompt-ab/p0-prompt-context-ab-20260723.json`，不保存 prompt、候选源码或凭据。

同日 round-2 使用 `deepseek-v4-pro` 完成跨家族、同上限实验：A/B 各调用 6 次，各自上限为
40,000 token / 300 秒。A 使用 19,822 token，5 个静态有效、3 个唯一有效；B 使用 22,498 token，
6 个静态有效、5 个唯一有效；两组裸源码协议均为 6/6。按实际 token 归一化后，B 的
valid/token 约高 6%，unique-valid/token 约高 46%，但 B 总 token 高约 13.5%。样本量仍小，且
静态有效不等于功能正确；因此结论仅是“B 在开发期唯一静态有效候选代理指标上值得保留观察”，
不是 Ascend prompt 接受结论。脱敏摘要位于被忽略的
`output/prompt-ab/p0-cross-family-equal-budget-20260723.json`，SHA-256 为
`28583c1e6bac89f418dca15f39ba0b6389e164a55ad3378d70fd58e77a387e43`。

同日 round-3 比较 P0 Prompt 快照 A 与 P1 压缩 Prompt B，仍使用三个家族、A/B 各 6 次及各自
40,000 token / 300 秒上限。A 使用 22,544 token，6 个静态有效、4 个唯一有效、裸源码协议 5/6；
B 使用 22,143 token，6 个静态有效、4 个唯一有效、裸源码协议 6/6。B 在代理有效性不退化时少用
约 1.8% token，但样本量仍小，只能保留该单一压缩假设继续观察。脱敏摘要位于
`output/prompt-ab/p1-cross-family-prompt-compression-20260723.json`，SHA-256 为
`f2ccecb0d8d88d6e13d8a3d20d13dc30a449ce693d5f6f32281b5768969c0c10`。

同日 round-4 使用 `_act_quant_kernel` 的真实官方 `runtime_error` 历史和精确父代哈希，比较当前
P1 fresh generation A 与 repair overlay B，各 3 次。A 使用 7,802 token，3 个静态有效、2 个
唯一有效、裸源码协议 3/3；B 使用 8,419 token，0 个静态有效、0 个唯一有效、裸源码协议 1/3。
因此当前 repair overlay 判定为 `reject`，不得因门控能力已经实现就自动进入生产默认路径。脱敏
摘要位于 `output/prompt-ab/p1-repair-vs-fresh-act-quant-20260723.json`，SHA-256 为
`2abbe8ebbd134807d464ecc8a328d7d63d0fc566249d71d3ce548573759717b0`。

P0、P1 至此闭合。P1 的“实现完成”只表示上下文、repair 门控、单假设压缩和开发期 A/B 已完成；
任何 prompt 版本的正式接受仍必须以真实 Ascend functional pass 和相同预算下的 best valid
speedup 为准。
