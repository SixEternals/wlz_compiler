# 04 优化升级 Plan（执行方版本）

更新时间：2026-08-04
用途：交付给执行方（Codex）按序实施。指挥方负责验收，执行方不做方案决策。
上位事实：`doc/2026编译系统设计赛大赛要求综合汇总.md` §5.1（官方规则）、
`doc/2026-Triton官方评测四类问题纠错报告.md`（两轮官方原始结果）、
`doc/211算子及总结.md`（当前 21 算子本机 matrix）。

## 0. 执行方必须先接受的五条事实

这五条决定了后面所有任务的优先级。不接受就会做错方向。

**事实 1：官方功能分是 0/100 二值，没有部分分。**
`综合汇总.md` §5.1：「至少一个版本成功编译并通过全部测试才有功能分 100；不能完全通过则功能分 0」。
平台 63 个任务 = 21 kernel × 3 task。一个 kernel 三个 task 全过才算功能通过。
当前官方状态：4 个算子 0/3、4 个算子部分失败 → **8 个算子的功能分是 0**，
不是「部分拿分」。

**事实 2：性能分只对功能完全通过的算子计算。**
所以 8 个功能失败的算子，性能分也是 0，无论本机 ratio 多好。
`_act_quant_kernel` 本机 ratio 0.0258（性能上限 200 分），但官方 2/3 → 实际 0 分。

**事实 3：官方分数公式与本机验收阈值不是同一个函数。**
`work/official_triton_agent/executor.py:340-345`：

```python
raw_speedup = self.baseline_time / current_time - 1
speedup = max(raw_speedup, 0.0)
fitness = min(speedup, 2.0)      # 200 分封顶
```

本机现行阈值是 ratio ≤ 1.03（不退化即通过）。换算成官方分数：

| 本机 ratio | 官方 speedup | 官方分数 |
| ---: | ---: | ---: |
| 0.0258 | 封顶 | 200 |
| 0.50 | 1.00 | 100 |
| 0.67 | 0.50 | 50 |
| 0.91 | 0.10 | 10 |
| 0.94 | 0.064 | 6 |
| 0.98 | 0.020 | 2 |
| 1.00 | 0 | 0 |

**ratio ≥ 0.94 的候选在官方记分上等于 0。** 当前 matrix 里 13 个 launch-only
候选加 4 个 neutral 候选全部落在这一档。继续在 `num_warps` / `num_stages`
上做单变量微调不产生分数。

**事实 4：官方允许每个 kernel 提交 5 个候选，当前只用了 1 个。**
`综合汇总.md` §5.1：「官方 `save_results()` 会为每个 kernel 写出 `<kernel>_best.py`、
`<kernel>_v1.py` 至 `<kernel>_v5.py`」，且「多个通过版本取最高分」。
5 个槽位是独立的风险分散机会：放 1 个激进候选 + 1 个保守候选 + baseline 近邻，
任何一个 3/3 通过就拿功能分，取最高性能分。
当前每算子只提交 1 个候选，等于自愿放弃 4 个免费重试。**这是当前最被浪费的规则。**

**事实 5：官方精度阈值是分段的，不是 `torch.allclose` 默认值。**
`综合汇总.md` §6.2：FP16 累积 <2048 用 `2^-8`、≥2048 用 `2^-7`；
BF16 是 `2^-7` / `2^-8`；FP32 分三段 `2^-11` / `2^-10` / `2^-9`。
判据是 golden 绝对值 ≥1 时看相对误差、<1 时看绝对误差，
`RE = (actual - golden) / (golden + 1e-7)`。
当前 `wlz_optimizer/correctness_oracle.py:416` 只有一个 `allclose` 策略
（`atol + rtol * |expected|`），与官方分段阈值不同构。
**这意味着本地 correctness 通过不等于官方精度通过，两个方向都可能出错。**

## 1. 任务总览与依赖

```
T1 官方精度门（对齐 §6.2）──┐
                             ├──> T4 五候选打包出货
T2 五候选槽位机制 ──────────┤
                             │
T3 防过拟合四门禁 ──────────┘
                             
T5 修 4 个 0/3 算子 ─────────> T4
T6 修 4 个部分失败算子 ──────> T4
T7 结构性优化（仅在 T4 交付后）
```

T1、T2、T3 可并行。T5、T6 依赖 T1（需要正确的精度判据才能验证修复）。
T4 是唯一的出货动作，依赖前面全部。T7 最后做。

## T1：实现官方分段精度门

**为什么排第一：** 没有正确的精度判据，T5/T6 的"修好了"是自我安慰。
官方 8 个算子失败中至少 3 个报 `accuracy check failed`，
而本地 oracle 用的容差与官方不同构。

**现状：** `wlz_optimizer/correctness_oracle.py:416` 只有一条：

```python
matches = policy.kind == "allclose" and absolute <= policy.atol + policy.rtol * abs(expected_float)
```

**要实现的判据**（`综合汇总.md` §6.2 原文）：

1. FP16/BF16 输入先转 FP32 计算，输出按 FP32 比较；FP32 直接 FP32 比较。
2. golden 出现 Inf/NaN → 该用例无效（不是失败，是 invalid，要分开记录）。
3. NPU 结果为 Inf/NaN：FP16 需先检查 FP32 golden 是否超出 FP16 表示范围，
   未超出则失败；FP32/BF16 在 golden 非 Inf/NaN 时出现 Inf/NaN 直接失败。
4. `|golden| >= 1` → 用相对误差 `RE = (actual - golden) / (golden + 1e-7)`；
   `|golden| < 1` → 用绝对误差 `AE = |actual - golden|`。
5. 阈值按 dtype 和累积计算次数分段：

| dtype | 累积次数 | 阈值 |
| --- | --- | --- |
| FP16 | < 2048 | `2^-8` |
| FP16 | >= 2048 | `2^-7` |
| BF16 | < 2048 | `2^-7` |
| BF16 | >= 2048 | `2^-8` |
| FP32 | < 2048 | `2^-11` |
| FP32 | 2048 ~ < 16384 | `2^-10` |
| FP32 | >= 16384 | `2^-9` |

「累积计算次数」按方案说明取一次累加参与的元素数量（reduce 维度长度或向量长度）。
每个算子的这个值要从 kernel 的 reduce 结构推导并显式记录，不要猜。
推导不出来时标 `accumulation_count=unknown` 并采用该 dtype 最严格的阈值。

**注意 BF16 那两行不是笔误。** 官方表格中 BF16 是 `<2048 → 2^-7`、`>=2048 → 2^-8`，
方向与 FP16/FP32 相反（阈值随累积次数变严）。照抄官方表，不要"修正"成单调的。
在代码注释里标一行说明这是官方表原值，避免后来者以为是 bug 改掉。

**验收条件：**
- 新增 `OraclePolicy` 类型支持分段阈值，保留现有 allclose 策略不删除（向后兼容测试）。
- 每个 dtype × 每个累积次数分段至少一个单元测试，含边界值（正好 2048、正好 16384）。
- 三个 Inf/NaN 分支各有测试；golden invalid 与 candidate 失败在返回值里可区分。
- 对 21 个算子逐个记录推导出的 dtype 和 accumulation_count，`unknown` 的单独列表。
- 现有 418 个测试仍全通过。

**不要做：** 不要删除或改写现有 allclose 路径；不要把官方阈值当成"参考值"再放宽；
不要因为某个算子用新门禁失败就调松阈值。

## T2：启用五候选槽位机制

**为什么重要：** 这是 ROI 最高的单项改动。官方规则给每个 kernel 5 个槽位、
取最高分、任一通过即得功能分；当前只用 1 个。等于每个算子白扔 4 次重试机会。

**现状：** `work/official_triton_agent/optimizer_agent.py:258` 的 `_get_top_k(5)`
已经能返回 5 个，但 `scripts/build_official_agent_batch_smoke.py`
按 operator 只选「第一个 `static_pass` + `passed=true` 的 manifest」，
最终每算子只打包 1 个候选。

**要实现的槽位策略**（每个 kernel 5 个槽位按风险梯度填充）：

| 槽位 | 内容 | 目的 |
| --- | --- | --- |
| v1 | 本机 ratio 最好的候选 | 争取高性能分 |
| v2 | 次优候选，且与 v1 的改动类型不同 | v1 若因某类问题失败，v2 不同源 |
| v3 | 最保守的非原样候选（launch-only 也可） | 功能分保底 |
| v4 | 与 v1 同策略但参数更保守的变体 | v1 是激进参数时的退一步版本 |
| v5 | 剩余最优，或留空 | 槽位不足时允许少于 5 个 |

**关键约束：**
- 五个候选必须**互不相同**且**都不等于 baseline**（官方：与原算子完全相同不得分）。
- 五个候选**必须都通过本地全部门禁**。不是"放几个赌运气的进去"——
  官方对失败候选没有惩罚，但提交未过门禁的候选会掩盖真实问题，
  且违反 `综合汇总.md` §2.4 的通用性要求。
- v1 和 v2 的改动类型必须不同（一个 launch-only 一个结构性，或两种不同结构性改动），
  这是分散风险的核心。两个都是 `num_warps` 变体等于只有一个候选。
- 槽位不足 5 个时输出实际数量，不要用 baseline 或重复候选凑满。

**验收条件：**
- 打包脚本对每个 kernel 输出 `<kernel>_v1.py` 到 `<kernel>_vN.py`（N ≤ 5）加 `<kernel>_best.py`。
- manifest 记录每个槽位的 candidate_id、sha256、改动类型标签、本机 ratio。
- 有测试验证：五候选互不相同、都不等于 baseline、v1/v2 改动类型不同。
- 有测试验证 N < 5 时不会用 baseline 或重复项填充。
- 目录名和文件名与官方 21 个槽位精确一致（`综合汇总.md` §5.1 提交契约）。

**不要做：** 不要为了填满槽位降低门禁标准；不要把 `_best.py` 当第 6 个候选；
不要改动 `_get_top_k` 的 fitness 排序逻辑（那是 T3 的事）。

## T3：四条防过拟合门禁

**为什么需要：** 决赛会追加 50 个不公开 case（`综合汇总.md` §5.1），
且官方明令禁止针对测试用例硬编码（§2.4）。但真实风险不在"用了公开 case"，
而在下面四个已经发生过的机制。每条都要写成确定性规则，不用 LLM 判断。

### 门禁 1 `no_semantic_change`

对候选和 parent 各做 AST 归一化：剥离注释与 docstring、规范化局部变量名、
规范化字符串字面量。归一化后 AST 相同则标记 `no_semantic_change`，
从 Top-5 候选池剔除，不参与 fitness 排序。

**动机（已发生）：** `_selective_scan_update_kernel` 的纯注释候选
`localv-4f49e502d013` ratio 0.9430，真实 launch 候选 `localv-5bbf8f9a80fa` ratio 0.9747。
纯注释改动因单 case 噪声排在真实改动前面。
另外 `_set_k_and_s_triton_kernel` 的候选 `fd113ce1` 只改了两个 f-string 报错文字
却被提交，官方 tc3 失败——这种候选本该在打包前被拒。

### 门禁 2 `timing_surface_moved`

比较候选与 parent 的 wrapper 部分。若候选新增 `torch.*` 张量算子调用、
`.cpu()`、`.item()`、`.contiguous()` 或额外 kernel launch，直接拒绝，不看 ratio。

**动机（已发生）：** `_pack_seq_kernel` 的 prefix-sum 候选 `ebfbd9806f27`
把工作挪进 wrapper 的 `torch.cumsum`，而 `msprof --kernel-name`
（`executor.py:243`）只计目标 kernel。目标 kernel 变快、端到端更慢。
本机因 ratio 1.0873 拒绝了它，但那是运气不是机制。

### 门禁 3 `shape_fingerprint`

拒绝对具体数值的相等比较分支和按观测 shape 建的查表。

实现要点：只拒绝 `ast.Eq` 比较到字面量整数、且该字面量等于某个可见 case 的
shape 值的情况。**允许** source-derived 计算（`triton.cdiv(N, BLOCK)`）
和 baseline 本来就有的 `<=` 分段（如 selective_scan 的 `if dstate <= 16`）。
避免误伤正常代码。

### 门禁 4 `holdout_required`

候选晋级 Top-5 前必须有至少一个 holdout case（搜索期未使用的 shape）
的 correctness 通过记录。缺失时标 `unqualified` 而非静默通过。
holdout case 的结果不得写入 mutation prompt 上下文。

**动机：** 当前 21×1 可见 case，搜索和验收是同一批，必然高估。
`doc/211算子及总结.md` 已写过这条五层验收，但只写在文档里没有执行。

**验收条件：**
- 每个门禁有单元测试，且用真实历史候选做反例：
  门禁 1 用 selective_scan 的 comment-only 候选和 `fd113ce1`；
  门禁 2 用 `ebfbd9806f27`；
  门禁 3 自造 `if N == 256` 样例；
  门禁 4 用只有单 case 证据的候选。
- 门禁触发时输出结构化原因字段，不是自由文本。
- 现有 418 个测试全通过。

**不要做：** 不要改 EA 搜索算法、population size 或 fitness 公式；
不要放宽现有 contract 检查；不要用 LLM 做这四个判断。

### T3 附注：允许的事

为避免执行方过度保守，以下**不算**过拟合：
用公开 case 的 shape 量级（非精确值）指导 tile 选择；
针对 Ascend 硬件特性的通用优化（对齐、访存合并、pipe 利用）；
从 baseline 源码结构推导的分段策略；在多个算子上都验证过的结构性模式。
判据只有一个：**这个改动在没见过的 shape 上还成立吗？**

## T5：修 4 个 0/3 算子

必读 `doc/2026-Triton官方评测四类问题纠错报告.md` §3.1–§3.4。
每个算子**只做单变量改动**，都是回退性质：

| 算子 | 候选 | 已定位的单一嫌疑点 | 做法 |
| --- | --- | --- | --- |
| `_copy_page_indices_kernel` | `f6321da2` | Python `range()` 的 stop 接了 Triton 运行时标量；`tl.arange(0, remainder)` 上界非 constexpr | 回退成 baseline 的 `tl.range` + mask 分块 |
| `_count_expert_num_tokens` | `76277d78` | 向量累加器改成标量累加器，循环内每次调 `tl.sum` | 回退成 `acc = tl.zeros((BLOCK_SIZE,))` + 末尾一次 `tl.sum` |
| `_state_passing_fwd_kernel` | `d3ab8399` | `BLOCK_SIZE` 默认值 16 → 128 | 只把 128 改回 16，保留提取 `mask_m` 的部分 |
| `_selective_scan_update_kernel` | `41f4c98a` | 所有 dstate 分段 `BLOCK_SIZE_M` 翻倍并新增 `num_stages`，官方报 accuracy failed | 回退成 baseline 的 launch profile |

**每个算子的流程：**
1. 先跑 baseline 本身，确认 baseline 在该 case 通过（建立 control）。
   这一步不能跳——`_set_k_and_s` 的教训是候选无语义变更却失败，
   说明不能假设 baseline 一定通过。
2. 再跑修正候选，用 T1 的官方分段精度门验证 correctness。
3. 然后跑 paired benchmark 拿 ratio。

**验收条件：**
- 4 个算子各有 baseline control 对照记录，不只有候选结果。
- 每个改动单变量、diff 行数小、能一句话说清。
- 明确报告每个算子修好后的 ratio。ratio ≥ 0.94 时标注
  "需后续结构性优化"但本轮不做。
- 修好的候选进入 T2 的 v3 保底槽位（它们是最保守的非原样候选）。

**不要做：** 不要一次改多个变量；不要顺手"改进"其他地方；
不要用 CUDA 通过代替 Ascend 通过；不要因为本地 correctness 过了就宣称官方会 3/3
（官方 tc2/tc3 参数未公开）。

## T6：修 4 个部分失败算子

| 算子 | 候选 | 官方状态 | 已知事实 |
| --- | --- | --- | --- |
| `_act_quant_kernel` | `35974f2c` | 2/3，tc3 runtime error | 候选把 `BLOCK_M` 32→64、固定 `num_stages=3/num_warps=4` |
| `_quantize_k_cache_fast_kernel` | `668d677d` | 1/3，tc2/tc3 accuracy failed | kernel 逻辑完全未改，只新增 `num_warps=4, num_stages=2` |
| `_set_k_and_s_triton_kernel` | `fd113ce1` | 2/3，tc3 accuracy failed | **kernel 和 launch 完全无语义变更**，只改了两个 f-string 报错文字 |
| `_per_group_transpose` | `bc3db2ee` | 3/3 → 2/3 跨轮退化 | 删除了 `tl.range` 分块循环，`BLOCK_SIZE_M` 16→32、`BLOCK_SIZE_K` 8→32 |

**处理要点：**

- `_set_k_and_s_triton_kernel` 是特殊情况：候选没有任何 kernel 语义变更却 tc3 失败。
  先确认 baseline 在同条件下是否也失败。若 baseline 也失败，说明是 baseline
  自身在 tc3 参数下的行为，需要真正的修复而非回退；若 baseline 通过，
  要核查提交 ZIP 内容与平台实际执行代码是否一致。这个算子的诊断结论
  比修复更重要，先给结论。
- `_per_group_transpose` 删掉 `tl.range` 后每个 program 只覆盖一个 m-tile，
  当某 expert 的 `num_tokens_of_expert` 大于 `BLOCK_SIZE_M × num_programs(1)`
  时存在漏算。恢复 `tl.range` 循环，tile 尺寸可保留或一并回退（记录哪个选择）。
  它同一 bytes 跨轮从 3/3 变 2/3（exit 139），说明存在条件敏感的越界，
  优先补 expert token 数超过单次 grid 覆盖范围的边界 case。
- `_act_quant_kernel` 官方用的是旧候选 `35974f2c`（2/3、0 分），
  但本机已有结构性候选 `localv-ce06de045cf9`（ratio 0.0258、性能上限 200）。
  本轮应验证新候选而不是修旧候选——但新候选也必须在扩展 case 上验证，
  不能因为本机 ratio 好就直接进 v1 槽位。

**验收条件：**
- 4 个算子各有 baseline control + 修正候选的 correctness 对照。
- 每个算子至少 4 个 shape：当前可见 shape、tile 整除边界、非整除边界、不同数量级。
- `_set_k_and_s` 给出明确诊断结论（baseline 是否自身失败）。
- `_per_group_transpose` 有 expert token 数超过单次 grid 覆盖的边界 case。
- 用 T1 的官方分段精度门，不用默认 allclose。

## T4：五候选打包出货

依赖 T1、T2、T3、T5、T6 全部完成。

**核心内容：** 把三个已有的高收益候选变现——它们从未进入任何官方 artifact：

| 算子 | 候选 | 本机 ratio | 官方分数上限 | 官方当前得分 |
| --- | --- | ---: | ---: | ---: |
| `_act_quant_kernel` | `localv-ce06de045cf9` | 0.0258 | 200 | 0.00 |
| `_fwd_kernel_ep_gather` | `localv-548c43fedfae` | 0.4806 | 108 | 0.62 |
| `_log_softmax_kernel` | `857d2fef3baf` | 0.5814 | 72 | 0.00 |

**每个候选晋级 v1 槽位前必须有的证据：**
- 至少 4 个 shape 的 correctness（用 T1 精度门）：当前可见 shape、
  tile 整除边界、非整除边界、不同数量级。
- act_quant 已有 `M=1/8/16/64 × N=128/256/512` probe，复用并补 paired benchmark。
- ep_gather 必须覆盖 token tail、top-k 变化、expert map 含负 expert id、
  hidden size 非整除。
- log_softmax 必须打开 `doc/211算子及总结.md` 提到的
  "当前入口注释掉的 small/3D/4D/dtype" 用例。
- 每个 shape 单独记录 correctness、paired ratio、raw samples。
- **报告里必须写出最差 shape 的 ratio，不是最好的。** 这是能否提交的依据。

**验收条件：**
- 21 个 kernel 各有 1–5 个候选槽位，全部通过 T1/T3 门禁。
- manifest 里每个 candidate_sha256 与 `output/` 下文件字节一致。
- 目录结构和文件命名与官方 21 槽位精确一致。
- 打包后**停下等审核，不提交官方评测**。

## T7：结构性优化（T4 交付后才做）

只有 T4 出货、拿到官方反馈后才动。方向参考已验证的三类结构性改动：
program mapping 重排（act_quant、ep_gather 的成功模式）、
block/index 表达式重写（chunk_cumsum，ratio 0.1569）、
索引位宽收窄（correct_attn，int64→int32）。

目标是把 13 个 launch-only 候选（ratio 0.94–1.03、官方 0–6 分）
换成结构性候选。**不要在 T4 之前做**：先把已有价值变现，再找新价值。

## 8. 期望收益与不确定性

若 T5、T6、T4 都成功：
- 功能通过从 13/21 提升到 21/21（8 个算子从 0 分进入性能池）；
- 三个高收益算子若在官方复现本机 ratio，性能 score 求和从 348 提升到约 727。

**必须说清的不确定性：**
- 本机 910B4 的 ratio 不等于官方 A2/A3 的 speedup。
- 官方 tc2/tc3 参数未公开，`baseline.json` 有 tc2/tc3 基线耗时但仓库无对应测试脚本，
  不能反推 shape/dtype，也不能自行补造。
- `_per_group_transpose` 已证明同一 bytes 跨轮结果会变。
- 上面的数字是"若本机 ratio 在官方复现"的上界，不是预测值。

**合格线的具体数值官方未公开**（`综合汇总.md` 与 `supply-doc` 中均无明确阈值），
因此本 plan 的目标是"功能分尽量全拿 + 已有大收益变现"，不对着假想分数线调参。

## 9. 需要指挥方决策的事（执行方不要自行决定）

1. 官方评测提交时机：T4 打包后必须停下等批准，逐 ZIP SHA 显式批准。
2. `综合汇总.md` §11 待确认事项中影响本 plan 的两条：
   模型清单是否含 `DeepSeek-R1`（当前默认 `deepseek-v4-pro`）、
   300 元 token 额度的计费口径。
3. 若 T5 修复后某算子 ratio 仍 ≥ 0.94，是否值得投入 T7 的结构性优化预算。

