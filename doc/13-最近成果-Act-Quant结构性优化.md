# Act Quant 结构性 program-mapping 优化

状态：最新工程成果；本机 Ascend 910B4 开发证据，不是官方 A2/A3 成绩

日期：2026-08-04

## 1. 结论

`_act_quant_kernel` 原入选候选 `localv-d441fb23be18` 只显式增加
`num_warps=4`，当前可见 case 的 candidate/baseline ratio 为 `0.9992566758`。
这只能说明没有明显退化，不能称为可信优化。

本轮新增结构性候选 `localv-ce06de045cf9`：

- source SHA-256：
  `ce06de045cf999e1905d664dc4f644e58b7a06fc858fdd6f46e1fda210d71832`；
- baseline SHA-256：
  `40c69588fcf18137a6f4814b50aa46382bcbb1907d615d8d8be86b024b1e6c19`；
- mutation kind：`structural_group_per_program`；
- 当前 qualification matrix 已将其选为 `_act_quant_kernel.best_candidate_id`。

它把二维 grid 下“一个 program 同时处理最多 32 行、每行一个 128-column group”改为一维
grid 下“每个 `(row, quant_group)` 独立一个 program”。量化公式、scale 粒度、输出布局、公开
wrapper 签名和 JIT 参数签名保持不变。这是 program mapping 和工作集形状的改变，不是注释、
等价表达式或默认 launch 参数显式化。

## 2. 为什么做这个改动

baseline 的 `BLOCK_M=32` 会让每个 program 构造 `32x128` 工作集。当前可见输入
`x.shape=(2,4,256)` 展平后只有 `M=8`，因此 24/32 行被 mask；同时每个 program 内包含 32 个
彼此独立的 128 元素 reduction。

每个量化 group 的 scale 只依赖该行的 128 个连续元素，所以这些 group 可以独立映射到 program。
新 mapping 删除无效行、二维 row mask 和 program 内跨行工作集，但没有跨越 scale 的 128-column
语义边界。

该方向参考了 DeepSeek-V3 在一维连续 block 上“一量化组一个 program”的公开实现思路，随后按本题
现有 float16 输出、布尔 round-scale 语义、函数签名和 Ascend 本地合同独立改写，并以本机测试裁决，
没有把 CUDA 项目的性能结论直接移植到 Ascend：

- upstream revision：`9b4e9788e4a3a731f7567338ed15d3ec549ce03b`；
- reference：`inference/kernel.py` 的 `act_quant_kernel`；
- code license：MIT，`LICENSE-CODE`。

## 3. Correctness 证据

当前 checkout 的可见 case：

- 输入：`M=8, N=256, block_size=128`；
- 普通 scale 的 `y/scale` 与 PyTorch reference 比较；
- round-scale 路径执行并检查输出 shape；
- baseline 与 candidate 进程均返回 0；
- correctness artifact：
  `output/local-correctness/_act_quant_kernel/localv-ce06de045cf9.correctness.json`。

额外数值 probe：

- 覆盖 `M=1/8/16/64`；
- 覆盖 `N=128/256/512`；
- 普通 scale 与 round-scale 均对 `y` 和 `scale` 做 PyTorch reference 比较；
- baseline 与 candidate 均通过；
- probe：
  `output/overfit-probes/act-quant-flat/datasets/_act_quant_kernel/test__act_quant_kernel_2.py`；
- correctness artifact：
  `output/overfit-probes/act-quant-flat/local-correctness-ce06de045cf9.json`。

在首次性能实验后，候选曾被纠正一次：将数学等价但可能影响边界舍入的 `amax / 448` 恢复为
baseline 原表达式 `amax * (1 / 448)`，再从头重跑 correctness 和 performance。最终候选只保留
program mapping 假设。

## 4. Performance 证据

环境 fingerprint 为
`f821659614569a674c8403ecbee273c656270e0760dafcd3a646caae132cb662`，设备为
`Ascend910B4-1`，Python 为 `/usr/local/python3.11.15/bin/python`，两边 profiler 频率均为
1650 MHz。每个 shape 使用 `B,C,C,B`、每角色两次的 `msprof op` 配对序列。

| Shape / path | Baseline durations (us) | Candidate durations (us) | Candidate / baseline |
| --- | --- | --- | ---: |
| visible `M=8,N=256`, default scale | `120.902420, 120.962418` | `3.160063, 3.080062` | `0.0258000504` |
| probe `M=64,N=512`, round scale | `135.482712, 135.942719` | `21.300426, 21.240425` | `0.1567312644` |

raw paired artifacts：

- `output/local-paired/_act_quant_kernel/localv-ce06de045cf9.paired.json`；
- `output/overfit-probes/act-quant-flat/local-paired-ce06de045cf9.json`。

两个 ratio 都明显低于 `1.03` qualification 上限，并且不是原候选约 `1.0` 的噪声级差异。这里仍只
报告本机 candidate/baseline ratio，不称为官方 speedup。

## 5. 当前边界

- 本机证据只有当前公开脚本和一个自建 shape probe；官方 case 2/3、初赛其余 case 与隐藏 case
  未知。
- 本机是 Ascend 910B4，官方目标为 A2/A3；跨硬件排序尚未验证。
- `compile_status` 仍按当前证据模型保留为 `unknown`；进程成功和 profiler CSV 不能补造独立官方
  compile 事实。
- manifest 的 `import_evaluation` 仍为 `not_run`，该手工本地 qualification 候选不是
  scoring-ready submission artifact。
- 赛题要求由 Agent 自动优化；该候选目前是验证过的策略证据。正式提交路径必须让 optimizer 在
  预算内自动提出或确定性应用该策略，并重新经过正式 correctness admission。
- 大 `M`、更大 `N`、其他 `block_size` 和 A2/A3 上的 program 数量/调度代价仍需留出验证，不能把
  `M=8` 的比率外推到全部 shape。

## 6. 对“真实优化”的口径修正

当前 matrix 的 `change_class=substantive` 只表示候选不是 AST 等价或显式 neutral，并不表示收益强、
跨 shape 稳定或策略值得写入 Experience。后续 17 个已入选候选应继续按以下口径分层：

1. program mapping、数据复用、访存、reduction 或算法结构发生变化；
2. 有强数值 correctness，而不只是 shape/finite 检查；
3. 至少当前可见 shape 和一个隔离 shape 有配对性能证据；
4. ratio 接近 1.0 的 launch-only 候选只算 qualification fallback，不计为成功优化经验。

按这个口径，本轮 `_act_quant_kernel` 已从 questionable launch-only 候选推进为本机证据支持的
结构性优化；其余弱候选仍需逐个处理，不能由 `17/21 substantive_selected` 推断为 17 个真实优化。
