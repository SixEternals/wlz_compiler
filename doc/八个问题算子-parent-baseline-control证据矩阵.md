# 八个问题算子 parent / baseline / control 证据矩阵

状态：步骤 1 证据已固化，未进入步骤 2  
固化日期：2026-07-19  
上位诊断：`doc/2026-Triton官方评测四类问题纠错报告.md`  
数据根目录：`work/official_triton_agent/datasets`

## 1. 范围与术语

本文只固化两轮官方评测中八个问题算子的现有证据，不运行新 gate，不生成候选，
不修改 Triton 代码，也不访问官方平台。

- **dataset baseline (B)**：`wlz_optimizer/io_utils.py:32` 定义的无编号 `<op>.py`。
- **manifest parent (P)**：candidate manifest 的 `parent_path`；可能是 baseline，也可能是
  `_1.py` seed variant。
- **candidate (C)**：实际打包候选。八个算子对应九个 candidate 版本，因为
  `_selective_scan_update_kernel` 在新轮替换过一次。
- **control**：可用于 baseline/candidate 对照的 test、EvaluationCase 或既有运行记录。
  `test 文件存在` 不等于 `control 已执行通过`。
- **官方标签**：原样来自两轮 `raw-result`；`returncode=0` 不作语义解释。

## 2. 文件身份矩阵

SHA 均为完整文件字节的 SHA-256。`P=B` 比较路径身份；“同字节”另行注明。

| 算子 / 版本 | B SHA-256 | P SHA-256 | P=B | C ID / C SHA-256 |
| --- | --- | --- | --- | --- |
| copy page indices | `3d8167f6a6e0f4fccc52bbccb174fc3b36cee0829f9b514161f4681997be9a1d` | `8b91bfd816add244b550f83d4a989d1bcbbc0b58af87eb8f6ae623f8326f5122` | 否 | `f6321da2` / `07b1fbf889b221940ed9c3243b143c9ca1e57825e44130ced6791b1b3cc55114` |
| count expert tokens | `75b7860e1419ec25757d85206957aaff8b604afddbd5d8a090c94a6a095bb306` | 同 B | 是 | `76277d78` / `840bbd72dd6fbabfd58e367612133dd2a03399aaac0ab612dbe7e94a8b23cb97` |
| state passing | `da9ff9c9487b2e829e020c3052709588de7614346cd27233c3c504492261097b` | `5f8bc4185a4d59b8dd260a142b060c11ac8dca7c7a01aa650e7bfe8c04485203` | 否 | `d3ab8399` / `5731ce120b7574762a81b465efec7dd76e47262f7aa9c9128ed2ee4aff013a03` |
| selective scan（旧） | `fb9b4a704e2b936e7e4145eb5a8b8a2bb6ca80d02b077d4093a10cd11b29589d` | `4f49e502d01364cf74c20a3b040ce0561eb018b8b200b4657aba9c56f8a4f4ff` | 否 | `eae3d41b` / `0e3f97ba6ba4a84a5e92883790c39a03d92c82c4d501a527e793241aaaf40d0c` |
| selective scan（新） | 同上 | 同 B | 是 | `41f4c98a` / `ada38a927318aee6f87c0d8d6862ff1d744748892ec60c4099ff7cfb109f71c5` |
| act quant | `40c69588fcf18137a6f4814b50aa46382bcbb1907d615d8d8be86b024b1e6c19` | 同 B（路径为 `_1.py`，字节相同） | 否 | `35974f2c` / `d22d0cece3706305f6d4f90bc55e6f414d6a5680a07a9f0985608cc29cfe92d7` |
| quantize K cache | `8d0787f7cc0f877b1681427fb8bcc6203ccce5da6cbb50fca2a161e7be64d526` | `c6e96b79773c1de22e7cbaf89483d3ab360dd659ab72b4af689c337215bd6d7b` | 否 | `668d677d` / `d24295e6f1aee4f9c5a52423561a55a2300df9dc78e657903cacda0b4a9d3af8` |
| set K and S | `dbd704a0f9f0d2c937e65f59323cfc5d8d279aed789517ba61aed0dc7aafa539` | 同 B | 是 | `fd113ce1` / `06d7886089f62066766fd2acf5b23eccd71ec366774944c73f66b1a6046b4cc1` |
| per-group transpose | `c39fb3c14f539f4bb6c23f2b70c829bdc2d033126445e0eb55eab1a99c900298` | `16a03c94d7d06678dd1240672cc1cf1d10867f4ef3cd08c0948e7ef7bb0cdfaa` | 否 | `bc3db2ee` / `90eb41a971ff676ed5d62452941b014404d0e568d85bf894c9a0536b6255baee` |

### 2.1 路径索引

表中 B/P/C 的具体路径如下；test 路径统一为同目录下
`test_<op>_1.py`。manifest 是 provenance 的权威入口。

| 算子 / 版本 | B 路径 | P 路径 | C 路径 | manifest 路径 |
| --- | --- | --- | --- | --- |
| copy | `work/official_triton_agent/datasets/_copy_page_indices_kernel/_copy_page_indices_kernel.py` | 同目录 `_copy_page_indices_kernel_1.py` | `output/real-agent-candidates/_copy_page_indices_kernel/f6321da2.py` | 同目录 `f6321da2.manifest.json` |
| count | `work/official_triton_agent/datasets/_count_expert_num_tokens/_count_expert_num_tokens.py` | 同 B | `output/real-agent-candidates/_count_expert_num_tokens/76277d78.py` | 同目录 `76277d78.manifest.json` |
| state | `work/official_triton_agent/datasets/_state_passing_fwd_kernel/_state_passing_fwd_kernel.py` | 同目录 `_state_passing_fwd_kernel_1.py` | `output/real-agent-candidates/_state_passing_fwd_kernel/d3ab8399.py` | 同目录 `d3ab8399.manifest.json` |
| selective（旧） | `work/official_triton_agent/datasets/_selective_scan_update_kernel/_selective_scan_update_kernel.py` | 同目录 `_selective_scan_update_kernel_1.py` | `output/real-agent-candidates/_selective_scan_update_kernel/eae3d41b.py` | 同目录 `eae3d41b.manifest.json` |
| selective（新） | 同上 B | 同 B | `output/real-agent-candidates-20260715-submit/_selective_scan_update_kernel/41f4c98a.py` | 同目录 `41f4c98a.manifest.json` |
| act | `work/official_triton_agent/datasets/_act_quant_kernel/_act_quant_kernel.py` | 同目录 `_act_quant_kernel_1.py` | `output/real-agent-candidates/_act_quant_kernel/35974f2c.py` | 同目录 `35974f2c.manifest.json` |
| quantize | `work/official_triton_agent/datasets/_quantize_k_cache_fast_kernel/_quantize_k_cache_fast_kernel.py` | 同目录 `_quantize_k_cache_fast_kernel_1.py` | `output/real-agent-candidates/_quantize_k_cache_fast_kernel/668d677d.py` | 同目录 `668d677d.manifest.json` |
| set | `work/official_triton_agent/datasets/_set_k_and_s_triton_kernel/_set_k_and_s_triton_kernel.py` | 同 B | `output/real-agent-candidates/_set_k_and_s_triton_kernel/fd113ce1.py` | 同目录 `fd113ce1.manifest.json` |
| transpose | `work/official_triton_agent/datasets/_per_group_transpose/_per_group_transpose.py` | 同目录 `_per_group_transpose_1.py` | `output/real-agent-candidates/_per_group_transpose/bc3db2ee.py` | 同目录 `bc3db2ee.manifest.json` |

新 selective 的 submit manifest 内部 `candidate_path` 仍指向
`output/real-agent-candidates-20260715-attempt2/.../41f4c98a.py`，而提交清单选择的是
`output/real-agent-candidates-20260715-submit/.../41f4c98a.manifest.json`。两处 `.py`
已经 `cmp` 验证字节相同，SHA 均为 `ada38a...f71c5`；矩阵使用 submit 路径表示实际
提交选择，保留该 provenance 路径差异而不静默改写 manifest。

## 3. 两段 diff 证据矩阵

`mutation_kind` 是生成意图标签；下表以实际 `B -> P -> C` diff 为准。

| 算子 | B -> P | P -> C（真实候选变化） | 可直接确认的风险 / unknown |
| --- | --- | --- | --- |
| copy page indices | 只删除许可证头和模块 docstring；计算体相同 | `tl.range + mask` 改成 runtime Python `range`；尾部使用动态 `tl.arange(0, remainder)` | 动态 `tl.arange` 是当前可静态识别风险；官方 runtime 根因仍 unknown |
| count expert tokens | P 就是 B | 向量 accumulator + 末尾 `tl.sum` 改为循环内 `tl.sum` + 标量 accumulator | Triton-Ascend 是否支持该循环内 reduction、是否导致官方错误均 unknown |
| state passing | **大幅实现差异**：状态/序列控制流重写；wrapper 的 `torch.npu.device` 改成 `torch.cuda.device` | `BLOCK_SIZE` 默认值 `16 -> 128`；重复 mask 提取为 `mask_m` | P 本身已有硬编码 CUDA 风险；不能只把 0/3 归因于最后一跳 tile 变化 |
| selective scan（旧） | 仅增加注释、docstring 和内联说明，未见计算表达式变化 | heuristics decorator 的 `{...}` 被污染为 `{{...}}`，另有局部源码重排 | 双花括号在 decorator 求值时预期触发 `TypeError`；平台未给 Traceback，官方根因不作最终确认 |
| selective scan（新） | P 就是 B | kernel body 相同；各 dstate 分支 `BLOCK_SIZE_M` 增大并新增 `num_stages` | 既有 CUDA smoke 仅覆盖 dstate=16；官方三 tc 参数和 accuracy mismatch 数值 unknown |
| act quant | P 与 B 字节相同 | wrapper `BLOCK_M 32 -> 64`；条件 stages 改为固定 `num_stages=3,num_warps=4`；f-string 双花括号 | kernel 数学未改；tc3 条件及 runtime 根因 unknown |
| quantize K cache | 只增加注释，计算体相同 | kernel 数学未改；launch 新增 `num_warps=4,num_stages=2`，另删注释 | launch 与 tc2/tc3 accuracy failure 的因果关系 unknown |
| set K and S | P 就是 B | 只把两个 f-string 插值改成字面量花括号；kernel/launch 不变 | 属于无 kernel 语义变化候选；tc3 accuracy failure 不能归因于计算改写 |
| per-group transpose | 只删除空行，计算体相同 | 删除覆盖多个 m-tile 的 `tl.range`；tile `16x8 -> 32x32` | 大 expert 存在尾部漏算风险；新轮 tc3 exit 139 的精确根因 unknown |

## 4. 历史门禁与 control 矩阵

八个旧候选 manifest 均为 `static_pass`，但只记录
`syntax/import/signature/launch_contract=true`；`compile_ok=null`、
`correctness_ok=null`，且没有 `triton_semantics_ok`。新 selective manifest 额外记录
`triton_semantics_ok=true`，但其 schema 中 `compile_ok/correctness_ok` 仍为 `null`。

| 算子 | 公开 test SHA-256 | case catalog | 既有本地运行证据 | 当前 control 状态 |
| --- | --- | --- | --- | --- |
| copy page indices | `13ecd37b2b57a116b4e3d5291a469e3ffe55ba7e66a96f0ec767e491e99040fc` | `unmaterialized` | 无 | `source_ready / not_run`；test 硬编码 NPU |
| count expert tokens | `a1617793fd39900448678ac1a8da7b93926f67f4ad83edfa5f2547ae3521e36d` | `unmaterialized` | 无 | `source_ready / not_run`；test 含 3 组输入并硬编码 NPU |
| state passing | `e2df3718ba2e8bbf0d1c763e79b4978dc770ab10f161f7bbacc473716aad1df0` | `unmaterialized` | 无 | `source_ready / not_run`；test 是 NPU main 脚本 |
| selective scan | `af063acf57be39ed697418cb8dac26c408a7915840893845c77c9472c62a3151` | `materialized_explicit_manifest`；signature `faf039e75001e59dc9e05428b037a605758f3befaf99f237b0da332db53ac36b` | submission sidecar 记录新 C 的 CUDA suite `139/139`；实际本算子只有 1 个固定 dstate=16 smoke | `materialized / narrow_cuda_record`; 不能外推官方 tc1–tc3 |
| act quant | `2ab41625c1f7b3d9e477569cfe536357f1af3808b00ec9ae9a3ef4c978e59339` | `unmaterialized` | 无 | `source_ready / not_run`；脚本可退到 CPU，但 Triton CPU 可执行性未证明 |
| quantize K cache | `c07e3c50d0099a4725d12699f45c56c6105afae7ca08a1ccc230d7c2b4fbbec2` | `unmaterialized` | 无 | `source_ready / not_run`；test 硬编码 NPU |
| set K and S | `5bebf3cc79f77add34f95e6065131a990fde9a389517372ddb6722b2341df99b` | `unmaterialized` | 无 | `source_ready / not_run`；test 硬编码 NPU |
| per-group transpose | `1a905ceb470c8584771ec6558222b5852f79d94103c17121b37864f7ee0b6ce1` | `unmaterialized` | 无 | `source_ready / not_run`；test 硬编码 NPU |

`unmaterialized` 的统一原因为 `missing_explicit_evaluation_contract`。当前没有本地
Triton-Ascend control，也没有证据表明平台 tc1/tc2/tc3 与这些单个公开脚本一一对应。

## 5. 官方原始标签矩阵

以下是 task 级反推，不把 `17/21 kernels` 当成官方字段定义。

| 算子 / candidate | 旧轮 | 新轮 | 稳定事实 |
| --- | --- | --- | --- |
| copy / `f6321da2` | tc1–3 runtime error | 同旧轮 | 0/3，两轮相同粗标签 |
| count / `76277d78` | tc1–3 runtime error | 同旧轮 | 0/3，两轮相同粗标签 |
| state / `d3ab8399` | tc1–3 runtime error | 同旧轮 | 0/3，两轮相同粗标签 |
| selective / `eae3d41b` -> `41f4c98a` | 旧 C：tc1–3 runtime error | 新 C：tc1–3 accuracy check failed | 候选已更换，只能确认标签路径变化，不能作同代码对照 |
| act / `35974f2c` | tc3 runtime error | 同旧轮 | derived 2/3，两轮相同粗标签 |
| quantize / `668d677d` | tc2–3 accuracy check failed | 同旧轮 | derived 1/3，两轮相同粗标签 |
| set / `fd113ce1` | tc3 accuracy check failed | 同旧轮 | derived 2/3；候选无计算语义变化 |
| transpose / `bc3db2ee` | 三个 task 未列失败 | tc3 child exit status 139 | 同 candidate bytes 出现跨轮结果分歧 |

证据文件：

- 旧轮：`output/official-runs/20260712-205309/raw-result.txt`
- 新轮：`output/official-runs/20260715-041611/raw-result.json`
- 候选选择：两份 `submission/*.manifest.json`

## 6. 步骤 1 完成边界

已完成：

- 八个算子的 B/P/C/test 路径均存在，九个 candidate 版本的 B/P/C SHA 已重算。
- B -> P 与 P -> C 分开记录，不再把 seed variant 当成 baseline。
- 历史静态门禁、control 物化、既有运行记录和官方标签分层记录。
- 每个根因描述都指向源码 diff，不能确认的官方内部信息保持 `unknown`。

尚未做（属于步骤 2 或以后）：

- 未物化其余七个公开 test 为 `EvaluationCase`。
- 未运行新的 import/compile/runtime/correctness control。
- 未验证 baseline、parent 或 candidate 在真实 Ascend 上的独立结果。
- 未新增 no-op、动态 shape、CUDA 引用或覆盖完整性的 gate。
- 未生成、打包或提交任何新候选。
