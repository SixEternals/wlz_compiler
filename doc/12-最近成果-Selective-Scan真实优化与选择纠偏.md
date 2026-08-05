# 最近成果 12：Selective Scan 真实优化与选择纠偏

更新时间：2026-08-04  
证据范围：`local_ascend_910b4_currently_visible_cases_not_official`

## 结论

候选 `localv-5bbf8f9a80fa` 只修改 `_selective_scan_update_kernel` 的一个 launch 决策：

```python
# dstate <= 16
(BLOCK_SIZE_M, num_warps) = (32, 2)  # baseline 为 (32, 4)
```

候选没有改变 kernel 数学表达式、wrapper 参数或其他 `dstate` 分支。它通过正式接口合同预检，
并在两个 `dstate<=16` shape 上通过数值 correctness 和串行 `B,C,C,B` paired `msprof`：

| case | shape | correctness | candidate / baseline median |
| --- | --- | --- | ---: |
| visible case 1 | `batch=2, heads=4, dim=64, dstate=16` | passed | `0.9747292067` |
| isolated case 2 | `batch=1, heads=2, dim=48, dstate=8` | passed | `0.9767981428` |

case 2 使用独立的向量化 PyTorch reference，并保存在 `output/overfit-probes/`；它不是官方
case 2。两组 profile 都来自 device 0、`1650 MHz`，环境 fingerprint 均为
`f821659614569a674c8403ecbee273c656270e0760dafcd3a646caae132cb662`。

## 可复核证据

- 候选源码：`output/real-agent-candidates/_selective_scan_update_kernel/localv-5bbf8f9a80fa.py`
- 候选 SHA-256：`5bbf8f9a80fadb68ff18de923340e2415f32b590d61821a3a234bec1a6f8ea88`
- manifest SHA-256：`60da7d2277e604be3079d43d7b57988085e510761488927f0be4479875d30ba8`
- baseline SHA-256：`fb9b4a704e2b936e7e4145eb5a8b8a2bb6ca80d02b077d4093a10cd11b29589d`
- visible test SHA-256：`af063acf57be39ed697418cb8dac26c408a7915840893845c77c9472c62a3151`
- shape-2 test SHA-256：`971b1addb6af8baa0869e2e1a9943953136171a1fdaaf153608b8b7ed1b1c94e`
- visible correctness：`output/local-correctness/_selective_scan_update_kernel/localv-5bbf8f9a80fa.correctness.json`
- visible paired：`output/local-paired/_selective_scan_update_kernel/localv-5bbf8f9a80fa.paired.json`
- shape-2 correctness：`output/overfit-probes/selective-case2/local-correctness.json`
- shape-2 paired：`output/overfit-probes/selective-case2/local-paired.json`

## Selection 纠偏

当前 qualification matrix 仍把旧 `localv-4f49e502d013` comment-only 候选列为该算子的
`best_candidate_id`，因为它在一次低样本测量中的 ratio 为 `0.9430254976`，低于真实候选的
`0.9747292067`。这不表示注释能加速 kernel，而是测量噪声和当前纯 latency 排序共同造成的
错误归因。

因此正式 selection 不能只取最小 ratio。后续最小规则应为：

1. 先要求 non-baseline、合同、全部 correctness 和环境绑定的 paired evidence；
2. 再排除 `comment_only`、`format_only`、`identifier_only` 和 AST 等价候选；
3. 最后在剩余真实 kernel/launch 候选中按同口径性能排序。

旧 sidecar 必须继续保留为噪声/等价对照，但不能进入 Experience 的成功策略或正式 selection。

## 边界

本结果只覆盖本机 Ascend 910B4 的两个 shape，不是官方 A2/A3 latency、官方 speedup 或全 case
functional pass。`num_warps=2` 对 `dstate>16` 的分支没有生效，也不能据此推广到其他分支。
