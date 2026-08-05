# 项目接手复盘与 Ascend Triton 适配说明

更新时间：2026-08-03

## 1. 当前状态

本阶段已经完成源码整理、真实 Ascend/NPU 冒烟和交付制品校验。提交给平台的是：

- `output/submission/official-agent-source-smoke-p2-g0-20260803.zip`
- SHA-256：`7b8a438d8dde8bddb5c21846eda2ec8aa4412e2d2090384149912d2f04f99a8a`
- 大小：`57,977 bytes`
- 入口：`Agent/main.py`
- 配置：population `2`、generation `0`、总 token budget `8192`
- 默认模型：`deepseek-v4-pro`

这个包的定位是 smoke，不是正式 scoring 包。它只验证平台能够加载 Agent、读取 baseline、调用 Ascend `msprof` 并保存结果；`generation=0` 意味着不会进行完整的进化搜索，`LLM call count=0` 也是预期行为。正式成绩仍以平台运行结果为准，不能由本地冒烟结果推断。

平台当前处于等待结果状态。本会话没有上传、重试或读取平台结果。

## 2. 接手时最容易踩坑的地方

### 2.1 “本机能跑”不等于 Ascend 能跑

仓库本地 `.venv` 是用于标准库测试的最小环境，不保证安装 Torch、Triton、`torch-npu`、CANN 或 `msprof`。Docker 也不能模拟真实 NPU 延迟。因此本地只能做语法、AST、接口、哈希、缓存和 mock 检查，不能伪造 `speedup` 或 `latency_ms`。

真实冒烟使用了独立的 Ascend Python 环境：Linux aarch64、Python 3.11.15、Torch 2.7.1、`torch-npu` 2.7.1.post4、Triton 3.5.0，并由 `msprof op` 产生实际 CSV。

### 2.2 旧文档与当前模型规则冲突

原始参赛说明仍出现 DeepSeek-V3 等旧模型示例，而当前运行规则要求 DeepSeek-V4 家族。运行时默认已收敛到 `deepseek-v4-pro`，并在模型列表中做白名单校验。后续改模型时必须同时确认：平台 allowlist、模型 ID、API endpoint 和 token 预算，不能只改 README。

### 2.3 API key 和 base URL 不能写进源码或制品

`llm_interface.py` 的约定是：优先读取配置注入值，其次读取 `API_KEY` 和 `API_URL` 环境变量；默认兼容地址是 `https://api.deepseek.com/v1`。API key 只应由运行环境注入，不能打印、提交、写入 ZIP、prompt、日志或子 agent 上下文。若平台提供不同的兼容 endpoint，应只通过 `API_URL` 覆盖，不要把 URL 和 key 混在代码里。

### 2.4 单测试用例会绕过多用例契约

早期路径在只有一个测试文件时直接使用单 case executor，导致多 case 证据结构无法统一，profiler observation 可能在 top-5 导出时被丢弃。现在 1 至 3 个连续 case 都走统一的 `MultiCaseContractExecutor`；case 编号缺失、重复或超过 3 都会 fail closed。

### 2.5 profiler 输出不能靠“找最新目录”猜

`msprof` 会生成 `run-* / OPPROF_* / OpBasicInfo.csv`。并发、残留目录或多个 `OPPROF_*` 都可能让“按 mtime 找最新”拿到错误结果。现在每次运行创建独立 `run-*` 目录，要求唯一 `OPPROF_*`，只读取一次有大小上限的 CSV 字节快照，并从快照同时计算耗时和 SHA-256，避免 TOCTOU。

### 2.6 多 case 的聚合值不能冒充单个官方测量

当前聚合定义是：总执行时间为已完成 case 时间之和，speedup 和 fitness 取成功 case 中最弱值，并明确写入 `official_aggregate=false`。这保证排序保守，但它不是平台最终评分公式的替代品。每个 case 的 baseline、执行时间、哈希和 profiler profile 都必须保留。

### 2.7 预算是 token 和墙钟的联合约束

只检查“调用次数”不够。现在 `BudgetController` 在 LLM 调用前预留 token/时间，成功后按真实 usage 提交，异常或 usage 不可信时按上界计费并标记 uncertain；进化循环和每个 executor 调用共享同一个 wall-clock deadline。预算耗尽后停止生成，不能通过重启优化器或重复 reservation 绕过上限。

### 2.8 最终归档与正式输出不是同一种东西

源码 Agent ZIP、生成的 `output.zip`、设计文档和视频材料属于不同交付物。当前归档只包含 `Agent/` 根目录和入口，不包含 datasets、运行输出、缓存、`.git` 或凭据。不要把 smoke 源码包直接当成最终全量材料包。

### 2.9 计划文档和平台浏览器上下文不完整

接手时 `AGENTS.md` 引用的总计划和学习文档并不在当前工作树，`doc/` 中只有凭据文件；因此架构判断必须以实际代码、测试和可复核产物为准，不能补写不存在的历史结论。当前环境也没有可用的 Chrome DevTools 登录快照，所以平台账号、任务身份和最终结果不能由本地推断。

## 3. 已完成的代码优化

### 3.1 Ascend executor 和 profiler 证据链

- 用 `shell=False` 参数列表调用 `msprof`，避免 shell 解析路径。
- 使用当前 `sys.executable` 启动测试，避免误用另一套 Python/Triton 环境。
- 每次评测建立隔离的 `performance/<kernel>/run-*` 目录。
- 只接受 `OpBasicInfo.csv` 中第一个名称完全匹配的 kernel 行。
- 对 CSV 设置 4 MiB 上限、UTF-8 解码和必需列检查；非法、空、非有限或非正耗时全部失败关闭。
- 保存相对 `executor_work_dir` 的 CSV 路径、run ID、解析规则、CSV SHA-256 和工具链指纹。

### 3.2 接口契约和多 case 评测

- 在昂贵的 `msprof` 前用 AST 比较函数签名、默认值、注解和 decorators，拦截接口漂移。
- 测试文件按 `test_<kernel>_<case_id>.py` 绑定，baseline 按 case 独立读取。
- 统一执行 1 至 3 个连续 case，共享单一 deadline；后续 case 的剩余时间会减少。
- case 证据升级为 schema v2，profile 只复制白名单字段，不接受任意 metadata。
- top-5 validator 校验 case 连续性、路径、SHA-256、toolchain fingerprint、时间与 speedup 公式的一致性，同时保留旧 schema v1 的读取兼容。

### 3.3 进化算法和候选 provenance

- 使用正权重的 rank selection，避免只有一个正 fitness 时概率向量退化。
- 使用去重后的 `(mu+lambda)` survivor selection，保留强父代并避免同一代码占据多个名额。
- crossover 或 mutation 异常时回退到较优父代，且保留 parent IDs、generation、lineage、mutation kind、model 和 prompt ID。
- 失败结果被压缩成有限类别，例如 syntax、import、timeout、runtime、correctness，不把原始日志直接塞入后续 prompt。
- top-5 只导出成功候选，排除 baseline 和重复代码；没有明确成功候选时主程序以失败退出。

### 3.4 LLM、预算和安全边界

- 默认模型固定在 `deepseek-v4-pro` 家族；模型切换只允许来自配置列表。
- 支持 DeepSeek OpenAI-compatible endpoint、代理和不同版本 `httpx` 的参数差异。
- 记录真实 usage metadata；缺失 usage 时按已预留上界结算，避免低估 token 消耗。
- API key 不进入 prompt、metadata、manifest 或归档；缺 key 时直接失败关闭。
- prompt context 只允许结构化统计和有限长度历史，避免把候选源码中的指令注入到 system prompt。

### 3.5 可审计源码打包

`scripts/build_official_agent_source_smoke.py` 现在使用固定 `Agent/` 布局、稳定 ZIP 时间戳、固定入口和 manifest。打包时会：

- 应用显式 smoke 覆盖值，不修改生产源码默认值。
- 编译所有 Python 文件，拒绝绝对路径、`..`、`.git`、`__pycache__`、`.pyc`、datasets、output 和潜在 key。
- 拒绝覆盖已有 ZIP/manifest，检查 ZIP 完整性和 20 MiB 上限。
- 写入每个源文件 SHA-256、归档 SHA-256、大小、配置和 `official_scoring_ready` 标志。

## 4. Triton 如何适配 Ascend 后端

这次适配没有把 CUDA 当作 Ascend，也没有硬编码 `soc_version`、CANN 路径、HiDevLab 主机或 SSH 凭据。核心思路是保留参赛框架的 Triton 算子接口，把后端差异集中在执行器和运行环境边界：

1. 由平台或真实 Ascend 环境提供 Triton-Ascend、`torch-npu`、CANN 和 `msprof`。
2. Agent 将候选 kernel 写入临时目录，重写测试中的 `from kernel import ...` / `import kernel`，保持测试输入和接口不变。
3. 执行器用当前 Ascend Python 解释器调用 `msprof op`，指定 kernel 名称和性能指标。
4. 只从 `OpBasicInfo.csv` 读取目标 kernel 的 `Task Duration(us)`，再与 organizer baseline JSON 中同一 case 的时间比较。
5. 采用框架约定的公式：`speedup = max(baseline_time / current_time - 1, 0)`，fitness 上限为 `2.0`。
6. 将每次真实观测及环境指纹写入 schema v2，供候选 provenance 和 top-5 输出复核。

因此，Triton 代码本身仍需遵守 Ascend Triton 后端支持的语法、装饰器、launch 参数和 dtype 约束；Agent 只负责在调用前做静态接口检查，在真实设备上做性能/正确性评测，不能在本地推测一个“Ascend speedup”。

## 5. 实际验证结果

### 自动化测试

```bash
.venv/bin/python -m unittest tests.test_official_source_packager -v
.venv/bin/python -m unittest discover -s tests -v
```

结果：打包器 2 项通过；全量 408 项通过、27 项因本机没有 CUDA/Torch/Triton 等条件跳过、0 失败。

### 真实 NPU smoke

算子：`_log_softmax_kernel`，baseline `751.320007 us`。

| case | profiler time | 结论 |
| --- | ---: | --- |
| seed 1 | `898.94 us` | 慢于 baseline，speedup `0` |
| seed 2 | `1052.301025 us` | 慢于 baseline，speedup `0` |

运行约 `182.56s`，LLM 调用 `0`。这证明了 Ascend 执行、`msprof` 采集、CSV 解析、schema v2 保存和 Agent 退出链路能够工作，不证明性能提升。

另有一个独立本地 case-1 候选 `857d2fef3baf`，仅把 `BLOCK_SIZE 1024` 改为 `2048`，配对 proxy ratio 为 `1.719870718x`。这是本地候选证据，不是官方 speedup，不能替代平台结果。

## 6. 后续接手者的检查顺序

1. 先核对平台账号、竞赛标题、contest/task/assignment/problem ID，再看运行结果；不要因为页面布局相似就提交到邻近任务。
2. 确认平台实际需要的是 smoke 源码包、正式 Agent 源码包还是另一个 `output.zip`，不要混用归档类型。
3. 在真实 Ascend 环境确认 `sys.executable`、Triton-Ascend、`torch-npu`、CANN 和 `msprof` 版本，并保存原始运行证据。
4. 正式 scoring 前把 population、generation、token 和 wall-clock 配置切回赛事要求，重新生成带新 SHA-256 的制品；不要沿用 `p2-g0` smoke 包。
5. 平台只报告上传接受、运行结束、功能通过和性能成绩中的哪一级，就只宣称哪一级；没有显式 latency/speedup 时保持 unknown。

## 7. 明确未完成或未宣称的事项

- 未在本会话代为上传、点击提交或重试平台任务。
- 没有 fresh authenticated page snapshot，因此平台任务身份仍需上传人复核。
- 没有把本地 proxy、Docker、CPU 或 mock 数值写成 Ascend 性能。
- 没有实现真实 SSH、rsync、HiDevLab、CANN 安装或远程清理逻辑。
- 没有声称 smoke 包已经是正式 scoring-ready 包，也没有声称真实 NPU 冒烟取得性能提升。

文件管理注意：仓库 `.gitignore` 默认忽略整个 `doc/`，本文件已写入工作区但不会自动进入普通提交；若后续需要纳入版本控制，只能显式添加本文件，不能放开整个目录或处理 `doc/testapi`。

