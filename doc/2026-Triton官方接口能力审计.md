# 2026 Triton 官方接口能力审计（G0）

状态：完成（官方框架 provenance 与静态代码接口审计）  
审计日期：2026-07-11  
适用赛题：基于进化算法的 Triton 自动优化系统  
上位计划：`doc/2026-Triton进化优化研究与实施总计划.md`

## 1. 结论

已登录的课程平台赛题页明确给出“初赛代码下载 / basic agent code framework”链接，目标为
GitCode 仓库 `lirui_ryan/Evolutionary-Algorithm-Based-TritonAscend-Optimization`。
因此该仓库可作为本赛题官方基础框架的一手来源，不再需要用公开参赛仓库推断接口。

官方仓库已按提交 `ef8c3bbc7bae6bdfa2af61722f9da14fd8ea5781` 原样保存在
`doc/sources/official_triton_agent/`，并生成固定版本快照
`doc/sources/official_triton_agent-ef8c3bbc.tar.gz`。本轮只做静态审计；由于当前没有已确认的
Ascend/CANN/msprof 环境和 API 凭据，没有运行官方优化流程，也没有产生真实性能结论。

当前阶段决策：

- G0 的官方规则、来源固定和静态代码接口审计完成。
- P0 official adapter 可进入静态适配工作，但本轮不实现。
- P1 可复用 `success/error` 做非常粗粒度失败记录；框架不提供 pass 级失败信息。
- P4 IR/profile 特征仍为 No-Go；框架没有 IR 输出，且未暴露所请求的 profiler 明细。
- 当前本地 mock 骨架没有应立即删除的源码。

## 2. 审计方法

本轮结论按以下证据链形成：

1. 核对官方技术方案和已登录课程平台返回的赛题正文。
2. 从赛题正文的“初赛代码下载”链接克隆 GitCode 仓库，并固定 commit、tree 和归档 hash。
3. 静态阅读入口、agent、executor、LLM 与进化循环源码，不导入或执行硬件相关代码。
4. 用标准库解析根目录 7 个 Python 文件的 AST，并统计 baseline 与随仓数据目录。

历史 worker 或公开参赛仓库的观察只作为检索线索；本节结论均由上述官方页面、固定源码或
本地可重复的静态命令直接复核。

## 3. 来源可信度

| 来源 | 身份判断 | 本轮用途 |
| --- | --- | --- |
| 2026 年 4 月赛题技术方案 | 官方一手材料 | 确认硬约束和评分边界 |
| 官方 GitLab `csc1/nscscc/compiler2026` | 官方资料仓库 | 验证公开目录是否包含 agent 源码 |
| 官方竞赛网站 `news.do` | 官方公开 API | 核对培训、技术支持与官方文档链接 |
| 已登录 `course.educg.net` 赛题页 | 官方课程/提交平台 | 确认本赛题正文和官方框架下载链接 |
| `lirui_ryan/Evolutionary-Algorithm-Based-TritonAscend-Optimization` | 赛题页指定的 GitCode 仓库 | 审计官方基础框架接口 |
| 本地固定版本源码与 tar.gz | 官方仓库的可复核快照 | 固定本轮审计对象 |
| `qinqinledao/Compiler2026-nwu` | 公开参赛仓库 | 二手观察可能的框架形态，不当作官方接口 |
| `T2026103582011617/ProteanTriton` | 公开参赛仓库 | 核对参赛 GitLab 项目是否带官方 seed |
| 本项目 `wlz_optimizer/` | 自研本地 mock | 判断已有契约和后续适配边界 |

来源固定结果：

- 课程平台任务：`contestID=1mTsU6jaSZ0`、`taskID=14955089`，动态 assignment
  标识为 `e-FLscI7uTE`。
- 赛题正文将 GitCode URL 明确标注为“初赛代码下载 / basic agent code framework”。
- commit：`ef8c3bbc7bae6bdfa2af61722f9da14fd8ea5781`，提交信息 `final version`，
  作者 `Yanqing22`，提交时间 `2026-06-12T16:23:16+08:00`。
- tree：`355a2bba369945164fc4d9d2db914c858d8a23f4`。
- 固定版本 tar.gz：74,052 bytes，SHA-256
  `c7c7d5cf2ebd2051bee4fec6f57250174869371cc97f82e45a77c6d3c9d72580`。
- `doc/sources/official_triton_agent/` 工作树在审计时保持 clean，文件未被本项目修改。

此前公开官方 GitLab 的目录未包含 agent 源码，公开课程说明也没有给出该下载地址；真正的
来源入口位于登录后的具体赛题正文。这解释了此前公开检索为什么没有取得框架。

公开参赛仓库仍不作为接口证据。当前官方身份来自赛题页对 GitCode 仓库的直接链接，而不是
源码中的“组委会提供”注释或文件名相似性。

## 4. 已确认的官方规则

官方文本位置：
`doc/sources/official_compiler2026_text/2026年全国大学生计算机系统能力大赛编译系统设计赛-编译系统挑战赛-基于进化算法的Triton自动优化系统-技术方案.txt`

| 能力/约束 | 已确认内容 | 原文行 |
| --- | --- | --- |
| 任务 | 在给定 agent 框架中用进化算法优化 Triton | 14-17 |
| 可改范围 | 可改进 agent 工具和 multi-agent 方法 | 18-24 |
| 运行预算 | 每算子 20 分钟或 20 万 token，先到即停 | 28-31 |
| 接口约束 | 必须使用提供环境和接口，不可改变调用方式 | 32-34 |
| 输入 | 大赛提供待优化 Triton 代码库 | 35-36 |
| 禁止项 | 不得手工修改 Triton 代码 | 37 |
| 多 agent | 所有 agent token 合计 | 38 |
| 输出数量 | 统一接口返回至多 5 个版本 | 39-40 |
| 功能门槛 | 至少一个非原样候选成功编译并通过全部测试 | 40-42 |
| 性能门槛 | 仅完全通过功能测试的代码参与 NPU 性能测试 | 44-53 |
| 评分 | Passing Rate 30%，性能 70% | 55-61 |
| 平台 | 鲲鹏 920、Ascend A2/A3、openEuler | 101-116 |
| 依赖 | 特殊情况下只能通过 pip 安装并说明 | 32-34 |
| 允许模型 | DeepSeek-V4/V3.2、Qwen3.5/3.6、Kimi-K2、GLM-4.6 | 156-157 |

技术方案没有给出 Python 类型、函数名、文件名、命令行参数或 JSON schema。

## 5. 官方接口能力矩阵

以下“官方代码状态”均针对固定提交 `ef8c3bbc` 的静态源码，不代表未运行环境中的行为已经
通过真机验证。

| 审计项 | 官方规则 | 官方代码状态 | 当前决策 |
| --- | --- | --- | --- |
| agent 初始化 | 必须走统一接口 | `setup(baseline_code, test_code, kernel_name="kernel", work_dir=None, test_case_id=1)` | 可写静态 adapter |
| 优化入口 | 每算子限时、统一调用 | `optimize(seed_codes, max_time=600) -> dict` | 保持签名，不假设限时生效 |
| 保存入口 | 提交至多 5 个版本 | `save_results(output_dir, kernel_name)` | 对齐现有文件命名 |
| executor 入口 | 编译、测试、性能评测 | `evaluate(code, timeout=1200) -> EvaluationResult` | 真机前只做类型适配 |
| `EvaluationResult` | 功能和性能需要 | `success/execution_time/speedup/fitness/error` | 无法拆分编译与正确性阶段 |
| optimize 返回 | 至多 5 个版本 | `best_code/best_fitness/speedup/generations/time_elapsed/llm_stats/top5_codes` | 可静态映射 schema |
| Top-5 元素 | 至多 5 个版本 | `code/fitness/generation/id` | adapter 保留 id 与代数 |
| 文件输出 | 提交 `output.zip` | `<kernel>_best.py`、`<kernel>_v1.py` 至 `_v5.py`、`<kernel>_stats.json` | 打包留给独立提交单元 |
| 20 分钟强制位置 | 先到上限即停 | `max_time` 只打印和计时，进化循环按 `max_generations` 执行 | 当前实现不满足自我强制 |
| 20 万 token 统计 | 先到上限即停 | 用 `prompt.split()` 与结果 `split()` 估算，未读取 API usage，未强制总上限 | 不作为合规预算器 |
| 编译结果字段 | 功能门槛需要 | 无独立 `compile_ok` 或编译错误类型 | pass-aware 签名 No-Go |
| 正确性结果字段 | 必须通过全部测试 | 测试脚本由 `msprof` 启动，无独立 `correctness_ok` 或误差详情 | pass-aware 签名 No-Go |
| 性能字段 | 真机 speedup 计分 | 只从 `OpBasicInfo.csv` 读取首个正的 `Task Duration(us)` | 没有 Ascend 时不填真实性能 |
| 失败输出 | 未规定 | 所有失败返回 `success=False`、固定 `error="Performance test failed"` | 只能粗粒度复用 |
| 原始日志 | 未规定 | 写入 `performance/<kernel>/get_prof.log`，不进入 `EvaluationResult` | adapter 不假设日志可用 |
| IR dump | 未规定 | 没有 IR dump 或 IR 特征 | P4 No-Go |
| profiler 指标 | 未规定 | 命令请求 MemoryDetail/Occupancy/PipeUtilization/Roofline，代码只解析时长 | 不声称明细可访问 |
| 并发/异步 | multi-agent token 合计 | 评测与进化循环均同步串行 | 不改变官方调用方式 |
| cache | 未规定 | 没有 cache 或环境 fingerprint | 自研 cache 必须保持环境感知边界 |
| 依赖声明 | 特殊依赖只能 pip 安装并说明 | 没有 `requirements.txt`、`pyproject.toml` 或环境文件 | 不自动安装依赖 |

## 6. 官方框架的静态审计发现

- `optimizer_agent.py:148-203` 接受 `max_time`，但只打印和记录 elapsed；实际
  `evolutionary_algorithm.py:264` 按固定代数运行，没有时间检查。
- `llm_interface.py:167-168` 用 `split()` 估算 token，没有读取 API usage，也没有 20 万上限。
- `optimizer_agent.py:183-202` 按 fitness 截取 Top-5，没有代码 hash 去重或非原样检查。
- `executor.py:44-51` 只有 `success/execution_time/speedup/fitness/error`，没有独立的编译、
  正确性和运行时状态。
- `executor.py:137-222` 将 msprof、测试进程、超时和其他异常压缩为相同失败结果；错误阶段
  不可区分。
- `executor.py:149-181` 请求 MemoryDetail、Occupancy、PipeUtilization、Roofline，但仅从
  `OpBasicInfo.csv` 解析 `Task Duration(us)`。
- 根目录 7 个 Python 文件均通过 `ast.parse`，这只证明语法可解析，不证明依赖或真机执行成功。
- `baseline/baseline.json` 有 50 个 kernel、每个 3 个 case，共 150 条 baseline 结果；
  `datasets/` 只有 21 个 kernel 目录，每个目录只有 1 个测试脚本。29 个 baseline kernel 没有
  随仓数据目录，不能静默补造缺失数据或假定评分集只有 21 个。

正确性测试可能通过被执行的测试脚本断言间接生效，但参赛 executor 没有提供独立
`correctness_ok` 或 mismatch 详情。静态 adapter 必须保留这种信息缺失，不能推导虚假的
compile/correctness 状态。

## 7. 当前本地骨架处置

### KEEP

- `schemas.py`：候选 provenance 和可空真实评测字段仍是 P0 的适配基础。
- `executors.py`：`Executor` protocol 与明确标注的本地静态检查可作为廉价预筛。
- `cache.py`：环境感知 JSONL cache 可用于 P0 replay 和 P1 失败复用。
- `genetic_operators.py`：当前注释变异只用于验证 provenance，不是研究实现，但测试仍依赖。
- `evolutionary_algorithm.py`：现有循环是 mock 契约验证点，后续逐阶段替换。
- `remote_ascend.py`：明确返回 `not_configured`，没有伪造真机能力。
- `io_utils.py`、`report.py`、dataset audit、scripts 和 tests：仍有本地验收用途。

### DELETE NOW

无。没有证据表明某个源码模块属于已放弃的 Rust-first、完整编译器或本地伪 NPU 方向。

### DEFER

- `executors.py:101-104` 的 `MockExecutor` 当前无引用且只修改 `kind`；等 P0 确定 executor
  结构后作为独立清理单元删除。
- `output/` 下的 mock/audit 结果是生成物，可在需要重新生成时清理，但不与 G0 文档修改混做。
- 注释变异与 `StubLlmClient` 在真实 candidate generator 接入后删除，而不是现在删除。

## 8. 官方框架固定状态与运行边界

| 项目 | 当前状态 |
| --- | --- |
| 官方来源 | 登录后的具体赛题正文直接链接 GitCode 仓库 |
| 本地源码 | `doc/sources/official_triton_agent/`，固定在 `ef8c3bbc` |
| 固定快照 | `doc/sources/official_triton_agent-ef8c3bbc.tar.gz`，SHA-256 已记录 |
| 静态可读性 | 根目录 Python 7/7 通过 AST 解析 |
| 依赖可复现性 | 未提供依赖声明，尚未确认 |
| 本机运行 | 未运行；不假定安装 Triton-Ascend、CANN、`msprof` 或模型 SDK |
| 真机评测 | 未配置 Ascend 环境，禁止生成或宣称真实 latency/speedup |

官方源码目录保持只读证据用途。后续适配代码应放在本项目自研模块中，不直接修改该目录；如需
验证不同官方版本，另行固定 commit 和 hash，不能覆盖本轮快照。

## 9. Go/No-Go 与解阻条件

| 项目 | 当前状态 | 解阻证据 |
| --- | --- | --- |
| H1 粗粒度失败复用 | Go，但信号很弱 | 可记录 `success=False` 和固定错误文本；不能区分失败阶段 |
| H1 pass-aware 失败签名 | No-Go | 需要官方返回编译阶段、测试阶段或 pass 信息 |
| P0 official adapter | Go（仅静态） | 签名和返回 schema 已固定；真机行为仍待验证 |
| P2 质量多样性 | 可做设计，不开始集成 | P0 有可重放评测记录 |
| P3 预算调度 | 可做设计，不开始集成 | 明确 token usage 与强制位置 |
| H2/P4 IR/profile 排序 | No-Go | 需要合法、稳定、低开销的 IR/profile 字段 |
| 真实性能结论 | No-Go | 需要 A2/A3 真机或官方评测结果 |

这里的 P0 “Go”只授权静态 schema/adapter 工作，不授权执行官方 NPU 流程、安装未声明依赖、
实现 SSH/HiDevLab 逻辑或把本地 proxy 标成 speedup。

## 10. 下一小步

定义一个静态 official-result adapter：只把固定提交中的 `EvaluationResult` 和
`optimize()` 返回字典映射到本地 schema，并用手工构造对象做单元测试；不导入官方硬件依赖，
不运行 NPU 代码。

## 11. 网络证据

- 大赛平台：<https://compiler.xtnl.org.cn/>
- 官方资料仓库：<https://gitlab.eduxiji.net/csc1/nscscc/compiler2026>
- 官方 repository-tree API：<https://gitlab.eduxiji.net/api/v4/projects/csc1%2Fnscscc%2Fcompiler2026/repository/tree?recursive=true&per_page=100>
- 课程平台竞赛页：<https://course.educg.net/course/6-366>
- 本赛题任务页：<https://course.educg.net/pages/contest/contest_submit.jsp?contestID=1mTsU6jaSZ0&taskID=14955089&my=false&contestCID=0>
- 官方基础框架（赛题页“初赛代码下载”）：<https://gitcode.com/lirui_ryan/Evolutionary-Algorithm-Based-TritonAscend-Optimization>
- 课程平台作品提交说明：<https://course.educg.net/sv2/indexexp/contest/contest.jsp?contestID=1mTsU6jaSZ0&tabDocID=8186665>
- 课程平台技术支持：<https://course.educg.net/sv2/indexexp/contest/contest.jsp?contestID=1mTsU6jaSZ0&tabDocID=10730949>
- 公开参赛项目历史核验：<https://gitlab.eduxiji.net/T2026103582011617/ProteanTriton/-/commits/main>
- 公开参赛仓库：<https://github.com/qinqinledao/Compiler2026-nwu>
