# Triton Optimization Agent 文档入口

状态：当前唯一文档入口
更新时间：2026-08-04

后续 AI 和维护者先读本页，再按任务进入专项文档。不要从文件名猜测哪个阶段报告仍然有效。

## 最新成果

**最新工程成果：**
[Chunk State 真实 launch 优化](11-最近成果-Chunk-State真实Launch优化.md)；
[Selective Scan 真实优化与选择纠偏](12-最近成果-Selective-Scan真实优化与选择纠偏.md)

结论摘要：两个原先仅有 neutral/equivalent 资格候选的算子，现已各增加一个真实
`num_warps` 单变量候选，并在本机 Ascend 910B4 的当前可见 shape 和一个隔离 shape 上通过
correctness 与 `B,C,C,B` paired benchmark。`_chunk_state_fwd_kernel` 两个 ratio 为
`0.93973/1.00268`；`_selective_scan_update_kernel` 为 `0.97473/0.97680`。纯 latency 排序仍会因
噪声把旧 comment-only selective candidate 列为 best，因此 selection lock 尚未完成，后续必须先
排除 AST 等价候选。

**当前全量资格基线：**
[本机 910B4 21 算子资格闭环](02-最近成果-本机910B4-21算子资格闭环.md)

结论摘要：当前 checkout 的 `21` 个算子均有当前可见 case 的 correctness 与 seeded ABBA paired
evidence，资格矩阵为 `21/21`。这只覆盖 `21 x 1` 可见 case 和本机 Ascend 910B4，不能替代官方
A2/A3、隐藏 case 或最终成绩。

**最新架构成果：**
[Triton Experience System 架构调研](00-最新成果-Triton-Experience-System架构调研.md)

结论摘要：

- 正式比赛 runtime、共享 schema 和离线开发工具合计已有 Candidate provenance、lineage、评测记录、
  失败历史、环境感知缓存、PromptContext 和预算控制，具备采集经验的原料；这些能力尚未全部接入
  同一条正式 runtime。
- 当前还没有跨 kernel 的 Experience Record、语义变更抽取、检索、Prompt 注入和使用效果归因。
- 当前 lineage / mutation history 只能作为经验的原始证据，不能直接当作经验库。
- 比赛主路径优先采用 `Single LLM + deterministic retrieval`；不先建立完整 Multi-Agent 链。
- 为先达到合格线，优先级仍是 correctness/evaluation、可信 benchmark、稳定 mutation 和等预算
  搜索基线。Experience 当前只建议先做 Phase 1 采集薄层。

该报告是架构分析，不表示 Experience System 已经实现。

## 事实权威顺序

发生冲突时按以下顺序判断：

1. 官方最新技术方案、当前任务正文、通知和该次平台原始结果。
2. 当前工作树源码、测试、manifest 和可复核运行输出。
3. 本页的当前状态和最新成果入口。
4. [研究与实施总计划](2026-Triton进化优化研究与实施总计划.md)中的研究方向、工程边界和路线。
5. 当前专项设计文档。
6. 带日期的历史报告和外部 `supply-doc` 归档。

“代码中存在字段”不代表官方平台认可该字段；“官方允许某能力”也不代表当前 Agent 已正确实现。

## 当前项目状态

截至 2026-08-03，本地标准库测试结果为：

```text
.venv/bin/python -m unittest discover -s tests -v
418 tests, OK, skipped=27
```

环境依赖跳过项不代表 CUDA、Triton、CANN 或 Ascend 路径通过。当前能力按接入层级分为：

- 正式比赛 runtime：LLM mutation/crossover、EA 搜索、由同一 `BudgetController` 约束的 LLM token
  与全流程墙钟、lineage 内失败反馈和 Top-5 输出；
- 共享 schema 与离线开发工具：更细的 Candidate/Evaluation 数据契约、静态/import/correctness 门、
  环境感知 cache/history、PromptContext、batch manifest 和 checkpoint；
- 本机 Ascend 910B4 + `msprof` 开发证据。

本机 qualification matrix 当前为 `21/21`，但证据范围仍是当前 checkout 的单个可见 case；逐算子
ratio、候选 hash、失败 probe 和 raw sidecar 以[最近成果报告](02-最近成果-本机910B4-21算子资格闭环.md)
为准。

后两层提供可复用证据和组件，但不能据此声称正式比赛 runtime 已经使用全部 schema、cache、history、
Prompt projector 或 checkpoint。

当前仍未完成：

- 不能把 profiler/process success 稳定拆成所有算子的独立 compile/correctness 事实；
- 没有在官方 A2/A3 上完成统一预算的全 case 功能和性能消融；
- 没有跨 kernel Experience Retrieval；
- 没有证明完整 Multi-Agent 比多生成并验证一个候选更划算。

本机 910B4 的 latency 或 ratio 只属于 `local_ascend_910b4` 开发证据，不是官方 speedup。

## 当前阅读路径

架构与实施：

- [研究与实施总计划](2026-Triton进化优化研究与实施总计划.md)
- [Experience System 最新架构调研](00-最新成果-Triton-Experience-System架构调研.md)
- [提示词工程与 Skill 系统设计](提示词工程与Skill系统设计方案.md)

官方约束与证据：

- [历史赛题与平台概览（2026-07-11 快照）](01-当前赛题与官方平台说明.md)：只用于理解选题和平台，
  其中“官方 Agent 源码尚未知”等状态已过期，不用于判断当前框架能力。
- [官方接口能力审计](2026-Triton官方接口能力审计.md)：当前框架事实入口；涉及平台身份和提交操作时
  还必须以 `AGENTS.md` 和 fresh authenticated page 为准。
- [官方评测四类问题纠错报告](2026-Triton官方评测四类问题纠错报告.md)
- [资料索引](source_index.md)

学习材料：

- [学习导读索引](learn/README.md)

官方 PDF、下载资料和网页快照保留在本机外部资料归档
`/workspace/user_data/supply-doc/sources/`。这些原始材料不因本次文档整理而改写、移动或删除。

## 文档维护规则

- 当前状态只在本页和总计划维护，不再新增“阶段交接”“下一步导向”式重复入口。
- 专项调研必须写清日期、证据范围、实现状态和被什么事实覆盖。
- 平台运行原文、artifact hash 和历史实验数字是证据快照，不做追溯性改写。
- 过期架构方案不保留在当前 `doc/`；需要追溯时查外部 `supply-doc` Git 历史。
- 新结论必须区分：已实现、已测试、真实设备开发证据、官方功能通过、官方性能结果。

## 本次整理边界

2026-08-03 本轮将当前文档集收敛为 11 个文件：删除 30 个 Rust-first、早期 mock、冲刺/阶段/交接稿、
失效学习导读和临时文件；保留赛题概览、官方接口/评测派生证据、Prompt 专项和当前计划。删除内容
没有从外部 `/workspace/user_data/supply-doc` 归档中抹除；官方 PDF、网页快照和运行原文也未改写。
