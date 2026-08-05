# 2026 Triton 官方评测四类问题纠错报告

状态：待独立审核  
审计日期：2026-07-15  
上位计划：`doc/2026-Triton进化优化研究与实施总计划.md`  
相关入口：`doc/2026-Triton官方接口能力审计.md`

> **全文证据等级定义**（必须在前）：
> - **官方结果事实**：直接来自官方平台 `运行结果` 页面文字或 `raw-result` 截图/快照。不经推导，不进行补全。
> - **代码事实**：由已记录 revision/SHA 的静态源码、manifest、candidate 文件或上述官方文件的直接内容证实。
> - **高置信推断**：可从已确认的官方结果和代码事实中直接逻辑推得，且所有备选假说均被排除。
> - **研究假设**：合理解释但需更多对照实验才能确认；必须标注为假设。
> - **`unknown`**：平台未暴露、代码未提供或缺少直接证据；不得臆造编译日志、误差值、shape、dtype、Ascend IR、profiler 数据或官方内部聚合公式。

---

## 1. 结论摘要

两轮官方评测（2026-07-12 和 2026-07-15）的 `运行结果` 页面均显示 `17/21 kernels`。这一数字**不是**平台公开定义的指标——平台没有文档说明其精确语义。由失败任务清单反向推导可得出以下算术关系：

- **旧轮**：失败清单共 16 条 task，其中 4 个 0/3 kernel 贡献 12 条失败，3 个部分失败 kernel 贡献 4 条失败。其余 14 个 kernel 贡献 `14×3=42` 条成功 task，3 个部分失败 kernel 分别成功 1/2/2 条，共 5 条；总成功 task = `42+5=47`。`14+3=17` 与页面显示值匹配。
- **新轮**：失败清单共 17 条 task，其中 4 个 0/3 kernel 贡献 12 条失败，4 个部分失败 kernel 贡献 5 条失败。其余 13 个 kernel 贡献 `13×3=39` 条成功 task，4 个部分失败 kernel 分别成功 2/2/1/2 条，共 7 条；总成功 task = `39+7=46`。`13+4=17` 与页面显示值匹配。

因此页面 `17/21 kernels` 在数字上等于「失败清单中少于 3 条失败 task 的 kernel 数量」。**其精确官方语义（如「至少一个 tc 通过」「至少编译通过」或其他定义）仍为 `unknown`**，平台未公开此字段的定义。

新轮仅替换了 `_selective_scan_update_kernel` 一个候选（`eae3d41b` → `41f4c98a`），其余 20 个候选代码与旧轮完全相同。新轮 `_per_group_transpose`（代码未变）新增一个 `tc3 runtime failed (child exit status 139)`，使 derived 3/3 task-success 从 14 降至 13、partial task-success 从 3 升至 4。

新轮的 21 个候选均由 `scripts/generate_official_candidate.py` 经 `deepseek-v4-pro` 单轮变异生成并通过本地静态门禁。`output/official-runs/20260715-041611/submission.json` 将 `41f4c98a`（新 selective）与 `gpu_tests: 139/139 passed` 关联；对应 test 代码实际只为该候选定义一个固定 CUDA Triton smoke，覆盖 `batch=2,nheads=4,dim=64,dstate=16,ngroups=2,seed=0`、all-features/fresh-rerun 单一场景。因此 `139/139` 是当时完整 pytest 套件的总测试数，**不等于** selective 候选有 139 个独立测试 case。新轮其余 20 个候选没有同类本地 Triton-CUDA 或 Triton-Ascend 正确性记录。跨两轮共有 22 个不同候选（新轮仅替换 selective），旧 selective `eae3d41b` 也没有该 CUDA smoke 记录。

连续两次官方失败后，当前不应进行第三次盲提。

### 1.1 职责边界

本文定位为纠错与诊断报告，**不是修复实现计划，不得据此修改任何 kernel 代码**。以下分类用于澄清问题性质并给出优先级排序，不授权对任一候选或 baseline 做代码改动。

### 1.2 五步建议顺序（只作为问题清点优先级）

1. **固化真实 parent diff、baseline/control 与本地可复现证据**：先分清 dataset baseline、manifest parent 和 candidate，确认八个问题候选究竟改了什么，并记录现有门禁遗漏。
2. **先扩展本地 import/compile/correctness gate，覆盖这八个问题候选**：把已知可本地拦截的问题前移，不能继续依赖官方平台发现基础错误。
3. **针对四个 0/3 算子做最小单变量回退/修复**（第一类）：只生成可追溯、单变量、已通过本地 gate 的候选。
4. **处理四个部分失败；仅对目标算子自身 3/3 的候选做性能搜索**（第二、三类）：先扩大 case matrix，再进行参数调优。
5. **后期、显式预算批准下做官方验证；噪声实验是可选子项**（第四类）：每次官方提交仍需用户对具体 SHA 显式批准，不能把重复 3–5 次写成默认动作。

每一步的「输入、产物、完成证据、不做事项」详见 §8。

---

## 2. 两轮官方评测数据

### 2.1 证据来源

| 轮次 | 提交时间 | artifact SHA-256 | 原始结果证据 |
| --- | --- | --- | --- |
| 旧轮 | 2026-07-12 20:53 | `43b3d103f9caeef086e6ac685f44f84e7b5b67565b628167d985f00f5c697a18` | `output/official-runs/20260712-205309/raw-result.txt`、`submission/wlz_triton_real_agent_batch21_20260712.manifest.json` |
| 新轮 | 2026-07-15 04:20 | `2a66789946619e5d6fb1a8c27cd500fb855093f15e03501d9d48ecce8293748f` | `output/official-runs/20260715-041611/raw-result.json`、`output/official-runs/20260715-041611/final-page-snapshot.txt`、`submission/wlz_triton_selective_scan_gate_pass_20260715.manifest.json` |

两轮均使用统一入口 `scripts/build_official_agent_batch_smoke.py` 打包，输出格式为 `organizer-save-results-v1`。官方 agent 子仓库（`work/official_triton_agent`）revision 为 `ef8c3bb`，且 worktree 为 dirty 状态：4 个 tracked 文件被修改，另有 4 个 untracked 路径。主 workspace 不是真正 Git repo，其 revision 视为 `unknown`。

### 2.2 跨轮候选变更矩阵

新轮仅替换了一个候选：

| Operator | 旧轮 candidate_id | 新轮 candidate_id | SHA-256 变更 |
| --- | --- | --- | --- |
| `_selective_scan_update_kernel` | `eae3d41b` | `41f4c98a` | `0e3f97ba...` → `ada38a92...` |
| 其余 20 个 operator | 相同 | 相同 | 相同 |

（来源：对比两份 submission manifest `selections` 数组，所有字段逐项核对。）

### 2.3 平台结果差异

| 指标 | 旧轮 | 新轮 | Δ |
| --- | --- | --- | --- |
| 平台判定 | AC(47/63) | AC(46/63) | −1 task |
| 成功任务数 | 47 | 46 | −1 |
| 失败任务数 | 16 | 17 | +1 |
| kernel 计数（页面显示） | 17/21 | 17/21 | 0 |
| 总分 | 22.00 | 20.00 | −2.00 |
| avg_speedup 显示值 | 21.80 | 20.47 | −1.33 |

### 2.4 「17/21 kernels」的含义

**证据等级：代码事实（失败清单推导）+ unknown（官方精确定义）**

页面显示的 `17/21 kernels` 的精确官方语义**未公开**。平台没有文档定义其计数规则。由失败清单反推得出的算术关系见 §1 摘要：旧轮 `14 个 derived 3/3 + 3 个 partial = 17`，新轮 `13 个 derived 3/3 + 4 个 partial = 17`。此推导与 `47/63`（旧轮）和 `46/63`（新轮）自洽，但**不构成对官方字段语义的确认**——不得将推导公式写成官方定义。

按官方技术方案，功能项得分要求「至少一个非原样候选成功编译并通过全部测试」。失败清单可推导出零失败 task 的 kernel 数量（旧轮 14 个，新轮 13 个），但平台没有逐 kernel 明示完整 compile/correctness 阶段状态，因此本文把它们记为 `derived 3/3 task-success`，不把推导冒充官方独立确认。部分失败 kernel 严格来说不满足「全部测试通过」要求；但其 score 条目仍出现在平台 score 列表中——二者关系为 `unknown`，不能自行解释为「仅 tc1 参与性能计算」。

---

## 3. 第一类：四个全部失败（0/3 tc functional pass）

以下四个 kernel 在两轮中所有 case 均失败。旧轮 manifest 只能证明候选通过了**当时版本**的静态门禁：syntax/import/interface/launch 相关检查通过，而 `compile_ok`、`correctness_ok` 均为 `null`，且旧 manifest 没有 `triton_semantics_ok` 字段。新 selective `41f4c98a` 才额外记录后来加入的窄范围 Triton semantic check，并做过一个有限覆盖的 CUDA smoke。不能把当前 `LocalExecutor` 的能力追溯套用到旧候选。

表中的 `mutation_kind` 只是生成时抽到的 prompt/意图标签，不等于真实代码变化；根因分析以 manifest 的 `parent_path` 与 candidate 的逐行 diff 为准。另需区分：dataset 中无编号 `<op>.py` 是 baseline，manifest 可能选择 `<op>_1.py` 等 seed variant 作为真实 parent。下表优先列真实 parent，不能把所有 `_1.py` 都称为 baseline。

### 3.1 `_copy_page_indices_kernel`

| 项目 | 内容 |
| --- | --- |
| candidate_id | `f6321da2`（两轮相同） |
| 候选路径 | `output/real-agent-candidates/_copy_page_indices_kernel/f6321da2.py` |
| manifest parent（seed variant） | `work/official_triton_agent/datasets/_copy_page_indices_kernel/_copy_page_indices_kernel_1.py` |
| dataset baseline | `work/official_triton_agent/datasets/_copy_page_indices_kernel/_copy_page_indices_kernel.py` |
| test 路径 | `work/official_triton_agent/datasets/_copy_page_indices_kernel/test__copy_page_indices_kernel_1.py` |
| 旧轮结果 | tc1/tc2/tc3 全部 `runtime error (Traceback in log) (returncode=0)` |
| 新轮结果 | 同上，不变 |
| 已证实阶段 | 历史 manifest 状态为 `static_pass`；`compile_ok=null`、`correctness_ok=null`，且当时没有 `triton_semantics_ok` 字段 |
| mutation 类型 | `local_rewrite`，parent `seed-8b91bfd816ad`，generation 1 |

**候选 vs manifest parent 差异（代码事实）：**

Baseline 使用 `for i in tl.range(0, num_blocks, BLOCK_SIZE)` 配合 `mask=i + offset < num_blocks` 做分块循环。候选改为：

1. `num_blocks_aligned = num_blocks - (num_blocks % BLOCK_SIZE)` —— 用 Python `range(0, num_blocks_aligned, BLOCK_SIZE)`（非 `tl.range`）迭代对齐部分，其中 `num_blocks` 和 `num_blocks_aligned` 均为 Triton 标量（非 Python int）；
2. `remainder = num_blocks - num_blocks_aligned` 后使用 `tl.arange(0, remainder)` —— 其中 `remainder` 为运行时 Triton 标量。

Python `range()` 的 stop 参数使用 Triton 运行时标量，属于需真实编译确认的高风险 pattern；当前窄门禁仍不单独拒绝 runtime Python `range`。`tl.arange(0, remainder)` 的上界不是 constexpr，当前门禁会以 `dynamic_tl_arange` 拒绝，但该规则在旧候选生成时尚不存在。**官方 runtime 根因仍为 `unknown`**——Traceback 文本不可见，无法确认平台失败是否由其中哪一项触发。

**证据等级：** 官方事实（失败状态）+ 代码事实（候选 diff）+ unknown（根因）

### 3.2 `_count_expert_num_tokens`

| 项目 | 内容 |
| --- | --- |
| candidate_id | `76277d78`（两轮相同） |
| 候选路径 | `output/real-agent-candidates/_count_expert_num_tokens/76277d78.py` |
| manifest parent（也是 dataset baseline） | `work/official_triton_agent/datasets/_count_expert_num_tokens/_count_expert_num_tokens.py` |
| test 路径 | `work/official_triton_agent/datasets/_count_expert_num_tokens/test__count_expert_num_tokens_1.py` |
| 旧轮结果 | tc1/tc2/tc3 全部 `runtime error (Traceback in log) (returncode=0)` |
| 新轮结果 | 同上，不变 |
| 已证实阶段 | `local_static_pass` |
| mutation 类型 | `local_rewrite`，parent `seed-75b7860e1419`，generation 1 |

**候选 vs manifest parent 差异（代码事实）：**

Baseline 本身已有 Triton kernel（`_count_expert_num_tokens`）**和** wrapper（`count_expert_num_tokens`）两个函数。候选同样保留这两个函数。原报告中「baseline 仅有一个函数／候选新增 wrapper／可能错误入口」的说法**错误**。

实际 kernel 级差异：baseline 使用**向量累加器** `acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)` 配合 `acc = acc + has_curr_expert`，最后以一次 `tl.sum(acc)` 规约存储。候选改为**标量累加器** `acc = tl.zeros((1,), dtype=tl.int32)` 配合 `acc = acc + tl.sum(has_curr_expert)`，在循环内每次迭代执行 `tl.sum`。这是累加策略的变化（向量归约后累加 vs 每块局部归约后标量累加），**Ascend 上 Triton 对 `tl.sum` 在循环内的支持程度为 `unknown`**，官方 runtime 根因未确认。

**证据等级：** 官方事实 + 代码事实 + unknown（根因）

### 3.3 `_state_passing_fwd_kernel`

| 项目 | 内容 |
| --- | --- |
| candidate_id | `d3ab8399`（两轮相同） |
| 候选路径 | `output/real-agent-candidates/_state_passing_fwd_kernel/d3ab8399.py` |
| manifest parent（seed variant） | `work/official_triton_agent/datasets/_state_passing_fwd_kernel/_state_passing_fwd_kernel_1.py` |
| dataset baseline | `work/official_triton_agent/datasets/_state_passing_fwd_kernel/_state_passing_fwd_kernel.py` |
| test 路径 | `work/official_triton_agent/datasets/_state_passing_fwd_kernel/test__state_passing_fwd_kernel_1.py` |
| 旧轮结果 | tc1/tc2/tc3 全部 `runtime error (Traceback in log) (returncode=0)` |
| 新轮结果 | 同上，不变 |
| 已证实阶段 | `local_static_pass` |
| mutation 类型 | `strategy_change`，parent `seed-5f8bc4185a4d`，generation 1 |

**候选 vs manifest parent 差异（代码事实）：**

唯一影响 launch/tile 语义的差异是 kernel 签名中 `BLOCK_SIZE` 默认值从 `16` 改为 `128`：

```python
# baseline
BLOCK_SIZE: tl.constexpr = 16,
# candidate
BLOCK_SIZE: tl.constexpr = 128,
```

其余 diff 是把多处重复的 `offs_m < dim` 提取为 `mask_m`，未改变对应 mask 表达式。主要语义循环（`for c in range(nchunks)` 内的状态传递逻辑）未改。Kernel body 和 wrapper grid 均通过 `BLOCK_SIZE`/`META["BLOCK_SIZE"]` 引用该值，因此 16→128 会改变程序数、tile 宽度、内存访问和资源压力；是否因此触发官方 runtime error 仍为 `unknown`。

**诊断优先级：** 优先测试将 `BLOCK_SIZE=128` 恢复为 `16` 的对照候选，以隔离 tile size 是否为根因。

**证据等级：** 官方事实 + 代码事实 + unknown（根因）

### 3.4 `_selective_scan_update_kernel`

| 项目 | 旧轮 (eae3d41b) | 新轮 (41f4c98a) |
| --- | --- | --- |
| candidate_id | `eae3d41b` | `41f4c98a` |
| 候选路径 | `output/real-agent-candidates/_selective_scan_update_kernel/eae3d41b.py` | `output/real-agent-candidates-20260715-submit/_selective_scan_update_kernel/41f4c98a.py` |
| manifest parent | `work/official_triton_agent/datasets/_selective_scan_update_kernel/_selective_scan_update_kernel_1.py`（seed variant） | `work/official_triton_agent/datasets/_selective_scan_update_kernel/_selective_scan_update_kernel.py`（也是 dataset baseline） |
| test 路径 | `work/official_triton_agent/datasets/_selective_scan_update_kernel/test__selective_scan_update_kernel_1.py` | 同上 |
| mutation 类型 | `local_rewrite`，parent `seed-4f49e502d013`，generation 1 | `param_tuning`，parent `seed-fb9b4a704e2b`，generation 1 |
| 旧轮结果 | tc1/tc2/tc3 全部 `runtime error` | — |
| 新轮结果 | — | tc1/tc2/tc3 全部 `accuracy check failed` |
| 已证实阶段 | `local_static_pass` | manifest 为 `local_static_pass`；submission sidecar 记录 CUDA suite `139/139 passed`，其中本候选实际覆盖仅为一个固定 dstate=16 smoke |

**新候选 41f4c98a vs manifest parent `_selective_scan_update_kernel.py` 差异（代码事实）：**

Kernel 函数体**完全相同**（逐行比对确认）。唯一差异在 wrapper `selective_state_update` 中的 launch config 选择：

```python
# baseline
BLOCK_SIZE_M, num_warps = (
    (32, 4) if dstate <= 16
    else ((16, 4) if dstate <= 32
    else ((8, 4) if dstate <= 64
    else ((4, 4) if dstate <= 128
    else ((4, 8)))))
)

# candidate 41f4c98a
BLOCK_SIZE_M, num_warps, num_stages = (
    (64, 4, 4) if dstate <= 16
    else ((32, 4, 4) if dstate <= 32
    else ((16, 4, 3) if dstate <= 64
    else ((8, 4, 3) if dstate <= 128
    else ((4, 8, 2)))))
)
```

即：在所有 dstate 分段上将 `BLOCK_SIZE_M` 翻倍（dstate≤16: 32→64, dstate≤32: 16→32, dstate≤64: 8→16, dstate≤128: 4→8），并新增 `num_stages`。本地 CUDA smoke 仅覆盖 `dstate=16` 这一个分支。

旧候选 `eae3d41b` 的 heuristics decorator 含 `{{...}}` 双花括号。该文本能通过 `ast.parse`，但 Python 会把 `{{"key": value}}` 解释为“包含 dict 的 set”，在模块导入时求值 decorator 参数会触发 `TypeError: unhashable type: 'dict'`。这是可由源码独立复核的生成管线转义缺陷；它与平台粗粒度 `runtime error` 相容，但平台没有返回 Traceback，因此不能宣称已由官方日志确认就是这一根因。新候选已不含该缺陷。

**关键推断：**

- 新候选 `41f4c98a` 的官方结果为 `accuracy check failed (AssertionError)` → 平台已到达其 accuracy-check 结果路径。**不能据此补全平台内部 compile、launch 或比较流程的具体行为**——平台不暴露这些独立状态。
- 旧候选 `eae3d41b` 的官方结果为 `runtime error (Traceback in log)` → **不能断言「连运行阶段都未到达」**——平台不暴露阶段拆分，`runtime error` 仅说明平台在某个阶段（可能 compile、可能 launch、可能 kernel 执行）遇到了异常。
- `returncode=0` 只原样记录为平台失败条目中出现的字段；平台未定义其语义时，**不得声称它表示 wrapper 正常退出或内部 executor 状态**。
- CUDA smoke 通过（dstate=16 单一分支）≠ Ascend 全 case 通过。本地下载的公开测试脚本可见一个 smoke 配置，但平台 tc1/tc2/tc3 与该脚本或“初赛公开 50 case”的映射未公开。

**证据等级：** 官方事实（失败状态）+ 代码事实（候选 diff）+ unknown（accuracy mismatch 具体误差值、case 参数和根因）

---

## 4. 第二类：四个部分失败（1–2/3 tc functional pass）

### 4.1 `_act_quant_kernel`

| 项目 | 内容 |
| --- | --- |
| candidate_id | `35974f2c`（两轮相同） |
| 候选路径 | `output/real-agent-candidates/_act_quant_kernel/35974f2c.py` |
| manifest parent（seed variant） | `work/official_triton_agent/datasets/_act_quant_kernel/_act_quant_kernel_1.py` |
| dataset baseline | `work/official_triton_agent/datasets/_act_quant_kernel/_act_quant_kernel.py` |
| test 路径 | `work/official_triton_agent/datasets/_act_quant_kernel/test__act_quant_kernel_1.py` |
| 旧轮结果 | tc1 pass, tc2 pass, **tc3 `runtime error (Traceback in log) (returncode=0)`** |
| 新轮 result | 同上，不变 |
| score | 两轮均为 0.00 |
| mutation 类型 | `local_rewrite`，parent `seed-40c69588fcf1`，generation 1 |

**候选 vs manifest parent 差异（代码事实）：**

1. Wrapper launch config：`BLOCK_M` 从 `32` → `64`；固定使用 `num_stages=3, num_warps=4`，替代 baseline 按 `round_scale` 选择的 `num_stages=0 if round_scale else 2`（baseline 未显式设置 num_warps）。
2. 候选的 f-string assert 消息中出现双花括号 `{{block_size}}`（应打印实际值但实际打印字面量 `{block_size}`），属于生成管线 artifact，不影响运行路径（仅在 assert 失败时触发）。
3. Kernel 计算逻辑（quantize 数学、tl.load/tl.store pattern）未改。

本地下载的公开测试脚本可见一个输入配置，但它与平台 tc1/tc2/tc3 的映射未公开；tc3 runtime error 的根因仍为 `unknown`。Shape、dtype、launch config 或其他差异都只能作为待验证假设。

**证据等级：** 官方事实 + 代码事实 + unknown（tc3 差异与根因）

### 4.2 `_quantize_k_cache_fast_kernel`

| 项目 | 内容 |
| --- | --- |
| candidate_id | `668d677d`（两轮相同） |
| 候选路径 | `output/real-agent-candidates/_quantize_k_cache_fast_kernel/668d677d.py` |
| manifest parent（seed variant） | `work/official_triton_agent/datasets/_quantize_k_cache_fast_kernel/_quantize_k_cache_fast_kernel_1.py` |
| dataset baseline | `work/official_triton_agent/datasets/_quantize_k_cache_fast_kernel/_quantize_k_cache_fast_kernel.py` |
| test 路径 | `work/official_triton_agent/datasets/_quantize_k_cache_fast_kernel/test__quantize_k_cache_fast_kernel_1.py` |
| 旧轮结果 | tc1 pass, **tc2/tc3 `accuracy check failed (AssertionError) (returncode=0)`** |
| 新轮结果 | 同上，不变 |
| 旧轮 score | 0.72（avg_speedup=0.01） |
| 新轮 score | 6.10（avg_speedup=0.06；候选未变，属于跨轮显示值变化，原因 unknown） |
| mutation 类型 | `strategy_change`，parent `seed-c6e96b79773c`，generation 1 |

**候选 vs manifest parent 差异（代码事实）：**

Kernel 计算逻辑**完全未改**（逐行比对确认）。仅两处变更：

1. Launch 调用新增 `num_warps=4, num_stages=2`（baseline 未显式设置）；
2. 删除一行注释 `# assert num_blocks_per_token == 5  # Commented out for flexibility`。

tc2/tc3 accuracy failure 与 launch config 变化的因果关系**不能定论**——可能存在交互，也可能是 baseline 在对应测试条件下本身就有边界数值行为，需要对照实验确认。本地公开脚本与平台 tc1/tc2/tc3 的映射未公开。

**证据等级：** 官方事实 + 代码事实 + unknown（accuracy 细节与根因）

### 4.3 `_set_k_and_s_triton_kernel`

| 项目 | 内容 |
| --- | --- |
| candidate_id | `fd113ce1`（两轮相同） |
| 候选路径 | `output/real-agent-candidates/_set_k_and_s_triton_kernel/fd113ce1.py` |
| manifest parent（也是 dataset baseline） | `work/official_triton_agent/datasets/_set_k_and_s_triton_kernel/_set_k_and_s_triton_kernel.py` |
| test 路径 | `work/official_triton_agent/datasets/_set_k_and_s_triton_kernel/test__set_k_and_s_triton_kernel_1.py` |
| 旧轮结果 | tc1 pass, tc2 pass, **tc3 `accuracy check failed (AssertionError) (returncode=0)`** |
| 新轮结果 | 同上，不变 |
| 旧轮 score | 0.62（avg_speedup=0.01） |
| 新轮 score | 0.00（候选未变，属于跨轮显示值变化，原因 unknown） |
| mutation 类型 | `local_rewrite`，parent `seed-dbd704a0f9f0`，generation 1 |

**候选 vs manifest parent 差异（代码事实）：**

Kernel 函数体**完全相同**。Launch 调用**完全相同**（grid、参数、constexpr 值均未变）。唯一差异是两个 f-string 报错文字的格式变更：

```python
# baseline
f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}"
f"{loc.dtype=}"

# candidate
f"index_k_scale must be 1D or 2D, got shape {{index_k_scale.shape}}"
f"{{loc.dtype=}}"
```

双花括号在 f-string 中输出字面量花括号而非变量值（属于生成管线 artifact），但此代码仅在 `assert` 失败时执行，不影响正常路径。

**关键推断：** 候选对 kernel 和 launch 均无语义变更。tc3 accuracy fail **不能归因于候选的 kernel mutation**（不存在）。需要核查：

- 该 failure 是否为 baseline 本身在 tc3 官方参数下的固有行为；
- 提交的候选 ZIP 内容是否与平台实际执行的代码一致（即平台是否可能执行了 baseline 或其他版本）；
- 平台 tc3 对应的测试条件是什么，以及 baseline 在相同条件下是否也会 accuracy fail（本地公开脚本与平台 tc 的映射未公开）。

此外，此候选作为「优化候选」没有任何 kernel 语义变化——按合理流程，应在送测前被 no-op gate 识别并拒绝提交。

**证据等级：** 官方事实 + 代码事实 + unknown（tc3 failure 归属）

### 4.4 `_per_group_transpose`

| 项目 | 内容 |
| --- | --- |
| candidate_id | `bc3db2ee`（两轮相同） |
| 候选路径 | `output/real-agent-candidates/_per_group_transpose/bc3db2ee.py` |
| manifest parent（seed variant） | `work/official_triton_agent/datasets/_per_group_transpose/_per_group_transpose_1.py` |
| dataset baseline | `work/official_triton_agent/datasets/_per_group_transpose/_per_group_transpose.py` |
| test 路径 | `work/official_triton_agent/datasets/_per_group_transpose/test__per_group_transpose_1.py` |
| 旧轮结果 | tc1/tc2/tc3 全部通过，score=10.11 |
| 新轮结果 | tc1 pass, tc2 pass, **tc3 `runtime failed (child exit status 139) (returncode=0)`**，score=0.00 |
| mutation 类型 | `strategy_change`，parent `seed-16a03c94d7d0`，generation 1 |
| 旧轮状态 | derived 3/3 task-success |
| 新轮状态 | partial task-success（2/3）——新增失败 |

**候选 vs manifest parent 差异（代码事实）：**

1. Kernel 中**删除了 `tl.range` 分块循环**：
   ```python
   # baseline: 每个 program 通过 tl.range 循环覆盖多个 m-tile
   for start_m in tl.range(0, num_tokens_of_expert, BLOCK_SIZE_M * tl.num_programs(1)):
       m_coord = start_m + m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
       ...

   # candidate: 无循环，每个 program 仅覆盖一个 m-tile
   m_coord = m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
   ```

2. Tile 尺寸变更：`BLOCK_SIZE_M` 16→32，`BLOCK_SIZE_K` 8→32。

3. Grid 公式未变（`triton.cdiv((m + num_experts - 1) // num_experts, META["BLOCK_SIZE_M"])`）。

删除 `tl.range` 循环后，每个 program 只覆盖一个 m-tile。当某个 expert 的 `num_tokens_of_expert` 大于 `BLOCK_SIZE_M × num_programs(axis=1)`（即大于单次 grid 覆盖范围）时，存在漏算风险。但 tc3 exit 139 (SIGSEGV) 的精确根因 `unknown`——官方不暴露 Traceback 或内存访问细节。

**跨轮分析：**

该候选 bytes 在两轮之间完全未变（same `candidate_id`、same `candidate_sha256`）；两份完整 ZIP 并不相同，因为 selective entry 被替换。tc3 从「通过」变为「exit 139」。

这只能定性为**跨轮结果分歧 / 不稳定信号**，不能直接定性为「非确定性失败」。两次结果无法区分以下可能性：

- 官方评测环境变化（硬件状态、驱动版本、编译器缓存）；
- tc3 输入参数/调度差异（若不同轮次的 case 参数不完全相同）；
- 候选代码中存在对特定条件敏感的未定义行为（UB），旧轮恰好未触发；
- 其他 `unknown` 因素。

进一步分类至少需要先做相同输入、相同候选的本地重复运行和边界 case；只有在本地证据不足且官方实验另行获批时，才考虑重复同一官方 artifact。当前不排除任何一种可能性。

**证据等级：** 官方事实（exit 139 新增失败）+ 代码事实（候选未变 + diff）+ 研究假设（跨轮分歧原因）+ unknown（tc3 参数、SIGSEGV 触发位置）

---

## 5. 第三类：性能接近零（功能通过但分数极低）

### 5.1 定义与范围区分

以下讨论严格区分三类 kernel：

- **13 个 derived 3/3 task-success kernel**（新轮）：由失败清单反推，三个 tc 均未列为失败；这不是平台独立给出的逐 kernel compile/correctness 证明。
- **17 个 score-listed kernel**：平台 `Kernel得分详情` 列出 17 条 score 记录。其中包含 4 个部分失败 kernel（`_act_quant`、`_quantize_k_cache_fast`、`_set_k_and_s`、`_per_group_transpose`）。
- **4 个部分失败 kernel**：有 score 条目但有 tc 出现在失败清单中。

官方技术方案称「只有完全通过功能测试的代码才进入性能测试」。但当前平台仍列出部分失败 kernel 的 score。二者关系为 `unknown`——不能自行解释为「仅 tc1 参与性能计算」，也不能将 17 个 score-listed kernel 称为「17 个 fully correct kernel」。

### 5.2 新轮所有有 score 条目的 kernel（17 条）逐项对比

（来源：新轮 `raw-result.json` 与旧轮 `raw-result.txt` 的直接引用。avg_sp = avg_speedup。）

| # | Operator | 旧轮 score | 新轮 score | 旧轮 avg_sp | 新轮 avg_sp | candidate_id | 功能状态（新轮） |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `_chunk_cumsum_fwd_kernel` | 200.00 | 200.00 | 2.00 | 2.00 | `2f655153` | full pass |
| 2 | `_unpack_seq_triton_kernel` | 121.27 | 124.10 | 1.21 | 1.24 | `489d65d0` | full pass |
| 3 | `_pack_seq_kernel` | 14.15 | 6.99 | 0.14 | 0.07 | `5cdd4dbf` | full pass |
| 4 | `_quantize_k_cache_fast_kernel` | 0.72 | 6.10 | 0.01 | 0.06 | `668d677d` | partial (1/3) |
| 5 | `_correct_attn_cp_out_kernel` | 2.43 | 2.35 | 0.02 | 0.02 | `c504c3c2` | full pass |
| 6 | `_dequantize_k_cache_fast_kernel` | 2.78 | 2.03 | 0.03 | 0.02 | `5fa5f18c` | full pass |
| 7 | `_per_token_group_quant_8bit_colmajor` | 0.68 | 1.95 | 0.01 | 0.02 | `9ec9fc5d` | full pass |
| 8 | `_chunk_state_fwd_kernel` | 8.49 | 1.85 | 0.08 | 0.02 | `5ad9a16b` | full pass |
| 9 | `_trtllm_prefill_attn_kvfp8_dequant` | 6.51 | 1.19 | 0.07 | 0.01 | `26abf19d` | full pass |
| 10 | `_silu_mul_fp8_quant_deep_gemm` | 0.00 | 0.70 | 0.00 | 0.01 | `2fb7ad26` | full pass |
| 11 | `_fwd_kernel_ep_gather` | 2.65 | 0.62 | 0.03 | 0.01 | `4ac1ce77` | full pass |
| 12 | `_convert_req_index_to_global_index_kernel` | 0.13 | 0.18 | 0.00 | 0.00 | `9b1b6c01` | full pass |
| 13 | `_rms_norm_kernel` | 0.00 | 0.01 | 0.00 | 0.00 | `0363fd3c` | full pass |
| 14 | `_log_softmax_kernel` | 0.09 | 0.00 | 0.00 | 0.00 | `3f6a303e` | full pass |
| 15 | `_act_quant_kernel` | 0.00 | 0.00 | 0.00 | 0.00 | `35974f2c` | partial (2/3) |
| 16 | `_per_group_transpose` | 10.11 | 0.00 | 0.10 | 0.00 | `bc3db2ee` | partial (2/3) |
| 17 | `_set_k_and_s_triton_kernel` | 0.62 | 0.00 | 0.01 | 0.00 | `fd113ce1` | partial (2/3) |

### 5.3 avg_speedup 显示值的算术吻合

**新轮 17 条 score 直接求和（代码事实——由平台逐行结果直接求和）：**

```
200.00 + 124.10 + 6.99 + 6.10 + 2.35 + 2.03 + 1.95 + 1.85 + 1.19
+ 0.70 + 0.62 + 0.18 + 0.01 + 0.00 + 0.00 + 0.00 + 0.00
= 348.07
```

`348.07 ÷ 17 ≈ 20.4747`，平台显示 `avg_speedup=20.47`，**显示标签与算术吻合**。

旧轮同理：`370.63 ÷ 17 ≈ 21.8018`，显示 `21.80`，算术吻合。

**重要：** 在这两轮数据中，名为 `avg_speedup` 的显示值数值上等于 17 条 score 的未加权算术平均。这是可复核的观测关系；平台没有说明它是否是长期稳定的接口契约，也没有解释为何该字段使用 `avg_speedup` 命名。总分 `22.00`/`20.00` 的聚合公式仍为 `unknown`。

### 5.4 观测要点

- 未修改候选中存在多项上升（`_unpack_seq_triton_kernel` +2.3%，`_quantize_k_cache_fast_kernel` 的 score 显示值由 0.72→6.10、算术变化约 +747%，`_per_token_group_quant_8bit_colmajor` +187% 等），也有多项下降。`_quantize_k_cache_fast_kernel` 是 partial task-success，其 score 聚合含义仍为 `unknown`。因此**不能写「所有 kernel 分数系统性下移」**；只能说汇总显示值（avg_speedup 21.80→20.47，总分 22→20）下降，而个体升降并存。
- 除 `_chunk_cumsum_fwd_kernel`（200 分，满分）和 `_unpack_seq_triton_kernel`（~124 分）外，其余 derived 3/3 task-success kernel 的 score 分布在 0–7 之间，多数极低。

---

## 6. 第四类：波动（未修改候选的跨轮 score 变化与新增失败）

### 6.1 新增功能性失败

| Operator | 变化 | 描述 |
| --- | --- | --- |
| `_per_group_transpose` | 全通 → tc3 exit 139 | 见 §4.4 |

这是唯一的新增功能失败。其余 19 个未修改候选的功能 pass/fail 状态与旧轮完全一致。

### 6.2 代表性 score 变化（未修改、功能状态不变）

（只列变化超出 ±0.05 的 kernel。数据为平台逐行直接引用。）

| Operator | 旧轮 score | 新轮 score | Δ | 功能状态 |
| --- | ---: | ---: | ---: | --- |
| `_chunk_state_fwd_kernel` | 8.49 | 1.85 | −6.64 | full pass |
| `_quantize_k_cache_fast_kernel` | 0.72 | 6.10 | +5.38 | partial |
| `_trtllm_prefill_attn_kvfp8_dequant` | 6.51 | 1.19 | −5.32 | full pass |
| `_pack_seq_kernel` | 14.15 | 6.99 | −7.16 | full pass |
| `_dequantize_k_cache_fast_kernel` | 2.78 | 2.03 | −0.75 | full pass |
| `_fwd_kernel_ep_gather` | 2.65 | 0.62 | −2.03 | full pass |

（完整对比见 §5.2 表格。）

### 6.3 波动归因

**证据等级：研究假设**。当前只有一组跨轮对照，波动只能定性列举，不能量化置信区间或归因比例。进一步判断需要受控的本地重复观测；只有在预算与具体 SHA 另行获批时，才使用官方重复提交补充证据。可能因素包括：

- **测量噪声**：Ascend NPU latency 的自然波动（硬件温度/频率/负载等）——与小幅变化（±0.1~±2）一致，但不能解释 `_chunk_state_fwd_kernel`（−78%）或 `_quantize_k_cache_fast`（+747%）的幅度。
- **case-level 差异**：不同 tc 的 shape/dtype 不同，候选对某些参数组合敏感——`unknown`（无 tc 级 latency 分解）。
- **官方环境变化**：两轮之间评测服务器状态变化（负载、缓存、编译器状态等）。
- **`unknown`**：平台聚合计算的未公开因素。

在噪声基线建立之前，任何跨轮 score 变化不能直接解释为「优化改进」或「环境退化」。

---

## 7. 问题文件与符号索引

以下覆盖本报告中涉及的所有关键文件，均使用仓库相对路径。路径均已通过 `find`/`test` 验证存在。

### 7.1 失败/部分失败候选文件

| Operator | 候选路径 | candidate_id |
| --- | --- | --- |
| `_copy_page_indices_kernel` | `output/real-agent-candidates/_copy_page_indices_kernel/f6321da2.py` | `f6321da2` |
| `_count_expert_num_tokens` | `output/real-agent-candidates/_count_expert_num_tokens/76277d78.py` | `76277d78` |
| `_state_passing_fwd_kernel` | `output/real-agent-candidates/_state_passing_fwd_kernel/d3ab8399.py` | `d3ab8399` |
| `_selective_scan_update_kernel` (旧) | `output/real-agent-candidates/_selective_scan_update_kernel/eae3d41b.py` | `eae3d41b` |
| `_selective_scan_update_kernel` (新) | `output/real-agent-candidates-20260715-submit/_selective_scan_update_kernel/41f4c98a.py` | `41f4c98a` |
| `_act_quant_kernel` | `output/real-agent-candidates/_act_quant_kernel/35974f2c.py` | `35974f2c` |
| `_quantize_k_cache_fast_kernel` | `output/real-agent-candidates/_quantize_k_cache_fast_kernel/668d677d.py` | `668d677d` |
| `_set_k_and_s_triton_kernel` | `output/real-agent-candidates/_set_k_and_s_triton_kernel/fd113ce1.py` | `fd113ce1` |
| `_per_group_transpose` | `output/real-agent-candidates/_per_group_transpose/bc3db2ee.py` | `bc3db2ee` |

### 7.2 对应 manifest parent 与 test 文件

| Operator | manifest parent 路径 | test 路径 |
| --- | --- | --- |
| `_copy_page_indices_kernel` | `work/official_triton_agent/datasets/_copy_page_indices_kernel/_copy_page_indices_kernel_1.py` | `work/official_triton_agent/datasets/_copy_page_indices_kernel/test__copy_page_indices_kernel_1.py` |
| `_count_expert_num_tokens` | `work/official_triton_agent/datasets/_count_expert_num_tokens/_count_expert_num_tokens.py` | `work/official_triton_agent/datasets/_count_expert_num_tokens/test__count_expert_num_tokens_1.py` |
| `_state_passing_fwd_kernel` | `work/official_triton_agent/datasets/_state_passing_fwd_kernel/_state_passing_fwd_kernel_1.py` | `work/official_triton_agent/datasets/_state_passing_fwd_kernel/test__state_passing_fwd_kernel_1.py` |
| `_selective_scan_update_kernel` (旧 parent) | `work/official_triton_agent/datasets/_selective_scan_update_kernel/_selective_scan_update_kernel_1.py` | `work/official_triton_agent/datasets/_selective_scan_update_kernel/test__selective_scan_update_kernel_1.py` |
| `_selective_scan_update_kernel` (新 parent) | `work/official_triton_agent/datasets/_selective_scan_update_kernel/_selective_scan_update_kernel.py` | 同上 |
| `_act_quant_kernel` | `work/official_triton_agent/datasets/_act_quant_kernel/_act_quant_kernel_1.py` | `work/official_triton_agent/datasets/_act_quant_kernel/test__act_quant_kernel_1.py` |
| `_quantize_k_cache_fast_kernel` | `work/official_triton_agent/datasets/_quantize_k_cache_fast_kernel/_quantize_k_cache_fast_kernel_1.py` | `work/official_triton_agent/datasets/_quantize_k_cache_fast_kernel/test__quantize_k_cache_fast_kernel_1.py` |
| `_set_k_and_s_triton_kernel` | `work/official_triton_agent/datasets/_set_k_and_s_triton_kernel/_set_k_and_s_triton_kernel.py` | `work/official_triton_agent/datasets/_set_k_and_s_triton_kernel/test__set_k_and_s_triton_kernel_1.py` |
| `_per_group_transpose` | `work/official_triton_agent/datasets/_per_group_transpose/_per_group_transpose_1.py` | `work/official_triton_agent/datasets/_per_group_transpose/test__per_group_transpose_1.py` |

### 7.3 构建与评测脚本

| 文件 | 说明 |
| --- | --- |
| `scripts/generate_official_candidate.py` | 加载官方 genetic_operators 和 contract_executor，用 `deepseek-v4-pro` 执行单轮变异并做本地静态检查 |
| `scripts/build_official_agent_batch_smoke.py` | 按 operator name 从 candidate_root 中选取 `static_pass` + `passed=true` 的第一个 manifest，生成 `organizer-save-results-v1` ZIP |
| `wlz_optimizer/executors.py` | 当前版本 `LocalExecutor.evaluate()` 的静态门禁（syntax/import/signature/launch_contract/triton_semantics）；不执行真实编译或正确性测试。旧候选生成时的门禁能力以各自 manifest 为准 |
| `work/official_triton_agent/genetic_operators.py` | 下载的官方框架文件，当前工作树含本地 contract/prompt 修订；`mutate()` 根据 `mutation_type` 生成变异 prompt，不能把当前行号视为 pristine 官方版本 |
| `work/official_triton_agent/executor.py` | 比赛提供的 agent framework `TritonExecutor`：返回 schema 只有 `success/execution_time/speedup/fitness/error`。平台 UI 另有 runtime/accuracy/exit-139 粗分类；二者不能混成同一接口，也不能据此推断平台内部阶段 |
| `tests/test_torch_triton_local_smoke.py` | D2-local CUDA smoke：`_rms_norm_kernel` 和 `_selective_scan_update_kernel` 的正确性验证（batch=2,nheads=4,dim=64,dstate=16,ngroups=2,seed=0）；仅在 CUDA 环境可用 |

### 7.4 两轮官方结果

| 轮次 | 原始结果 | 提交清单 |
| --- | --- | --- |
| 旧轮 | `output/official-runs/20260712-205309/raw-result.txt` | `submission/wlz_triton_real_agent_batch21_20260712.manifest.json` |
| 新轮 | `output/official-runs/20260715-041611/raw-result.json` | `submission/wlz_triton_selective_scan_gate_pass_20260715.manifest.json` |

### 7.5 ZIP artifact

| 轮次 | ZIP 路径 | SHA-256 |
| --- | --- | --- |
| 旧轮 | `submission/wlz_triton_real_agent_batch21_20260712.zip` | `43b3d103f9caeef086e6ac685f44f84e7b5b67565b628167d985f00f5c697a18` |
| 新轮 | `submission/wlz_triton_selective_scan_gate_pass_20260715.zip` | `2a66789946619e5d6fb1a8c27cd500fb855093f15e03501d9d48ecce8293748f` |

---

## 8. 五步建议顺序详解

以下五步是**问题诊断优先级排序**，不是实施方案，不授权任何代码修改。每个官方提交仍需用户对具体 SHA 显式批准。

### 步骤 1：固化 parent diff、baseline/control 与现有证据

- **目的**：对八个问题候选分别确认 dataset baseline、manifest parent、candidate bytes 和真实 diff；同时建立 baseline/control 的本地可复现入口。
- **排序原因**：当前已经出现 `mutation_kind` 与真实 diff 不一致、seed variant 被误称 baseline、旧门禁能力被追溯套用等问题。若输入事实不稳定，后续 prompt、gate 和修复方向都会偏离。
- **输入**：八个 candidate manifest、其 `parent_path`、无编号 dataset baseline、公开 test 脚本、两轮原始结果。
- **产物**：八行证据矩阵，至少包含 parent SHA、candidate SHA、语义 diff、旧门禁字段、本地可复现状态、官方原始标签和 unknown 字段。
- **完成证据**：每个 SHA 可重算；每个路径存在；每条根因描述都能回指 diff 或明确标为假设/unknown。
- **不做事项**：不生成新候选，不提交官方评测，不把 baseline 手工改成参赛结果。

### 步骤 2：先扩展本地 import/compile/correctness gate

- **目的**：至少覆盖这八个问题候选，把可本地发现的错误拦在官方平台之前。
- **排序原因**：旧 `static_pass` 没有真实 compile/correctness 证明；动态 `tl.arange`、decorator 双花括号、no-op 语义候选等本可在本地拒绝。先修 gate，才能避免下一批候选重复同类错误。
- **输入**：步骤 1 的证据矩阵、现有 isolated worker/candidate runner/oracle、公开 tests。
- **产物**：可复现的 import、compile、runtime、correctness、timeout 分类；no-op/非 kernel 语义变化拒绝；逐 case 有界错误摘要。
- **完成证据**：旧 selective 在 import gate 被拒；copy 的动态 shape pattern在 semantic/compile gate 被拒；set 的仅报错字符串变化在 no-op gate 被拒；其余候选给出实际 pass/fail/unsupported，而不是伪造 Ascend 结论。
- **不做事项**：不声称 CUDA 等价于 Ascend，不产生本地 `speedup`，不先做 autotune。

### 步骤 3：为四个 0/3 算子生成并验证单变量候选

- **目的**：通过最小回退隔离四个 0/3 的高风险变更，并得到先通过本地 gate 的可追溯候选。
- **排序原因**：这些算子当前没有成功 task，是正确性风险最高的一组；但必须在步骤 2 后处理，不能再次让官方平台充当基础编译器报错器。
- **输入**：已通过本地 control 的 parent/baseline、步骤 2 gate、单一变更假设，例如恢复 `tl.range`、恢复 `BLOCK_SIZE=16`、修复 decorator、回退 selective launch profile。
- **产物**：每个算子至少一个单变量 candidate，记录 parent、diff、case signature 和 gate 结果。
- **完成证据**：候选在本地支持的全部公开/派生 case 上 fail-closed 通过；Ascend 专有缺口明确标为 unknown。后续官方验证仍需单独对具体 ZIP SHA 请求批准。
- **不做事项**：不一次改多个变量，不做多轮搜索，不以“至少一个官方 tc 通过”冒充最终 3/3 完成。

### 步骤 4：处理部分失败，并按算子进入性能搜索

- **目的**：先扩展 shape/dtype/layout/stateful case matrix 修复四个部分失败；对目标算子自身已有 derived 3/3 task-success 且通过本地 gate 的候选，才做参数调优或性能导向局部变异。
- **排序原因**：部分失败说明候选对边界条件敏感；直接 autotune 会把正确性缺陷与性能变量混在一起。另一方面，不必等待 21/21 全部修复后才优化已经独立过 gate 的算子。
- **输入**：逐算子 case catalog、correctness gate、官方 score 显示值、候选 launch/tile profile。
- **产物**：部分失败算子的边界复现与修复候选；已过 gate 算子的性能候选集和 provenance。
- **完成证据**：正确性候选先通过对应 gate；真实性能结论只来自获批后的官方/真实 Ascend结果，并与原始 artifact SHA 关联。
- **不做事项**：不把 score-listed 等同 fully correct，不对 partial candidate 做性能结论，不在满分算子上无目的消耗预算。

### 步骤 5：后期官方验证；噪声实验仅为可选子项

- **目的**：验证步骤 3/4 的明确假设。只有当候选稳定、官方配额明确且统计问题确实影响决策时，才设计相同 artifact 的噪声实验。
- **排序原因**：当前最缺的是可复现正确性证据，不是更多相同盲提。重复评测本身不会修复 0/3 或 partial failure。
- **输入**：已冻结 ZIP、完整本地验证记录、预先声明的官方验证问题；若做噪声实验，还需样本计划与预算上限。
- **产物**：按提交协议保存的官方 raw result；可选噪声实验保存每次样本、环境可见字段和样本量。
- **完成证据**：每次提交均有用户对具体 SHA 的显式批准；官方结果按 upload/completed/executor/functional/performance 分级记录。样本不足时不宣称显著或不显著。
- **不做事项**：不把“重复 3–5 次”写成默认要求，不盲重试，不用反复提交碰最高分。

---

## 9. 给独立审核者的问题清单

以下问题供独立 Claude 审核本报告时重点核查。每条的预期答案和证据位置已在相应章节标注。

1. **状态行声称「待独立审核」，全文是否仍有任何位置暗示状态为「完成」或「已验收」？**（检查 §1 之后所有断言）
2. **「17/21 kernels」的语义是否在任何地方被描述为已确认的官方定义，而非推导算术？**（检查 §1 摘要和 §2.4）
3. **新 selective candidate（41f4c98a）的 CUDA smoke 覆盖范围是否被准确描述（仅 batch=2,nheads=4,dim=64,dstate=16,ngroups=2,seed=0）？是否错误暗示 139 个 case 均为 selective 测试？**（检查 §3.4）
4. **`returncode=0` 是否被解释为「wrapper 正常退出」或「内部 executor 状态」？**（全文检索 `returncode`）
5. **`_per_group_transpose` 的跨轮 failure 是否被定性为「非确定性失败」而非「跨轮结果分歧 / 不稳定信号」？**（检查 §4.4 和 §6.1）
6. **是否出现「所有 kernel 分数系统性下移」或等价表述而未承认未修改候选中存在上升项？**（检查 §5.4 和 §6.3）
7. **低性能部分是否将 17 个 score-listed kernel 称为「17 个 fully correct kernel」或混淆 derived 3/3 task-success kernel（13）与 score-listed kernel（17）？**（检查 §5.1）
8. **`avg_speedup=20.47` 的公式是否被同时描述为「分母是否为 17 未知」又给出恰好匹配的算术？**（检查 §5.3）
9. **`_count_expert_num_tokens` 的 baseline 是否被错误描述为「仅有一个函数」？**（检查 §3.2）
10. **`_set_k_and_s_triton_kernel` 的 tc3 accuracy fail 是否被归因于 kernel mutation（而候选实际无 kernel 语义变更）？**（检查 §4.3）
11. **八个候选的实际 diff 是否被准确、完整地描述为代码事实？**（逐项核对 §3.1–§4.4 中各「候选 vs manifest parent 差异」段）
12. **所有 repo-relative 路径是否可通过 `find` 或 `test -f` 验证？**（随机抽查 §7 中 10 个路径）
13. **两个 ZIP SHA-256 是否与 submission manifest 一致？**
14. **五步顺序是否先固化 parent diff，再扩本地 gate，然后才生成 0/3 修复候选？步骤 5 是否把噪声实验标为可选且要求逐 SHA 显式批准？**（检查 §8）
15. **是否仍存在「阶段 1 9/10」等与本轮四类问题无关的过期进度断言？**（全文检索「阶段」）

---

## 10. 当前停止点

```
连续两次官方评测后状态：
  全部失败 kernel：4/21（19%）— 无任何 case 通过
  部分失败 kernel：4/21（19%）— 至少一个 case 失败
  derived 3/3 task-success：13/21（62%）— 三个 task 均未列入失败清单；多数 score 很低

下一步：先按五步顺序执行步骤 1（固化 parent diff、baseline/control 与现有证据），
不进行第三次官方提交。

官方 agent 子仓库 revision：ef8c3bb（worktree dirty：4 tracked modified + 4 untracked paths）
主 workspace revision：unknown（非真正 Git repo）
候选 SHA 与 ZIP SHA 见 §2.1 和 §7.5。
```

---

## 附录 A：全 21 kernel 两轮官方结果对照速查表

| # | Operator | 旧轮 score | 新轮 score | 旧轮 pass | 新轮 pass | candidate_id |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 1 | `_act_quant_kernel` | 0.00 | 0.00 | 2/3 | 2/3 | `35974f2c` |
| 2 | `_chunk_cumsum_fwd_kernel` | 200.00 | 200.00 | 3/3 | 3/3 | `2f655153` |
| 3 | `_chunk_state_fwd_kernel` | 8.49 | 1.85 | 3/3 | 3/3 | `5ad9a16b` |
| 4 | `_convert_req_index_to_global_index_kernel` | 0.13 | 0.18 | 3/3 | 3/3 | `9b1b6c01` |
| 5 | `_copy_page_indices_kernel` | 0.00 | 0.00 | **0/3** | **0/3** | `f6321da2` |
| 6 | `_correct_attn_cp_out_kernel` | 2.43 | 2.35 | 3/3 | 3/3 | `c504c3c2` |
| 7 | `_count_expert_num_tokens` | 0.00 | 0.00 | **0/3** | **0/3** | `76277d78` |
| 8 | `_dequantize_k_cache_fast_kernel` | 2.78 | 2.03 | 3/3 | 3/3 | `5fa5f18c` |
| 9 | `_fwd_kernel_ep_gather` | 2.65 | 0.62 | 3/3 | 3/3 | `4ac1ce77` |
| 10 | `_log_softmax_kernel` | 0.09 | 0.00 | 3/3 | 3/3 | `3f6a303e` |
| 11 | `_pack_seq_kernel` | 14.15 | 6.99 | 3/3 | 3/3 | `5cdd4dbf` |
| 12 | `_per_group_transpose` | 10.11 | 0.00 | 3/3 | **2/3** | `bc3db2ee` |
| 13 | `_per_token_group_quant_8bit_colmajor` | 0.68 | 1.95 | 3/3 | 3/3 | `9ec9fc5d` |
| 14 | `_quantize_k_cache_fast_kernel` | 0.72 | 6.10 | 1/3 | 1/3 | `668d677d` |
| 15 | `_rms_norm_kernel` | 0.00 | 0.01 | 3/3 | 3/3 | `0363fd3c` |
| 16 | `_selective_scan_update_kernel` | 0.00 | 0.00 | **0/3** | **0/3** | `eae3d41b` / `41f4c98a` |
| 17 | `_set_k_and_s_triton_kernel` | 0.62 | 0.00 | 2/3 | 2/3 | `fd113ce1` |
| 18 | `_silu_mul_fp8_quant_deep_gemm` | 0.00 | 0.70 | 3/3 | 3/3 | `2fb7ad26` |
| 19 | `_state_passing_fwd_kernel` | 0.00 | 0.00 | **0/3** | **0/3** | `d3ab8399` |
| 20 | `_trtllm_prefill_attn_kvfp8_dequant` | 6.51 | 1.19 | 3/3 | 3/3 | `26abf19d` |
| 21 | `_unpack_seq_triton_kernel` | 121.27 | 124.10 | 3/3 | 3/3 | `489d65d0` |

（pass 列由失败清单反推：`3/3`=三个 task 均未列为失败，`2/3`、`1/3`=部分 task 未列为失败，`**0/3**`=三个 task 全部列为失败；不是平台独立给出的逐 kernel compile/correctness 证明。score 为平台显示的原始 kernel 得分。）

## 附录 B：candidate manifest 完整字段说明

每个 `output/real-agent-candidates/<operator>/<id>.manifest.json` 包含：

- `candidate.id`、`candidate.code_hash`、`candidate.generation`、`candidate.mutation_kind`、`candidate.op_name`
- `candidate.parent_ids`（列表，可追溯生成时使用的 seed/parent）
- `candidate.metadata.official_operator_metadata`（mutation_type、operation、parent）
- `candidate.model_used`、`candidate.prompt_id`
- `parent_path`（生成时真实 parent 的绝对/相对路径，不保证它是无编号 dataset baseline）、`parent_sha256`
- `seed_sha256`（种子文件 SHA-256 列表，含双种子场景）
- `llm_stats`（调用次数、每次的 prompt_sha256、model、usage token 数）
- `static_evaluation`（本地静态检查完整结果，含 defined_functions、feature_counts、function_signatures、decorator_mismatches 等）

提交 manifest（如 `submission/wlz_triton_real_agent_batch21_20260712.manifest.json`）额外包含：
- `archive_entries`（ZIP 内文件列表）
- `selections`（每个 operator 的 candidate_id、candidate_sha256、candidate_manifest_path、operator）
- `artifact_kind`、`layout`、`scoring_intent`

两轮提交的 ZIP 和 manifest 均使用 `scripts/build_official_agent_batch_smoke.py` 生成，确保格式一致。
