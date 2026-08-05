# 最近成果 11：`_chunk_state_fwd_kernel` 真实 Launch 优化

更新时间：2026-08-04  
证据范围：`local_ascend_910b4_currently_visible_cases_not_official`

## 结论

将 `_chunk_state_fwd_kernel` 的 wrapper launch 只增加一个显式参数：

```python
num_warps=2
```

其余 baseline 源码保持不变。候选 `localv-243d9727a611` 通过了正式接口合同预检、当前可见
case correctness，以及一个隔离的 shape probe。两个 shape 都完成同设备、同频率的串行
`B,C,C,B` `msprof` 配对，均在 `1.03` 容差内：

| case | shape 变化 | correctness | candidate / baseline median |
| --- | --- | --- | ---: |
| visible case 1 | `batch=2, seq=4096, heads=4, head_dim=16, dstate=8` | passed | `0.9397309946` |
| isolated case 2 | `batch=1, seq=1024, heads=2, head_dim=32, dstate=16` | passed | `1.0026768383` |

case 2 是留出 shape probe，不是官方 case 2；它被放在 `output/overfit-probes/`，没有改写
官方 dataset。当前 qualification matrix 仍为 `21/21`，并将该候选列为此算子的 best candidate。

## 可复核证据

- 候选源码：`output/real-agent-candidates/_chunk_state_fwd_kernel/localv-243d9727a611.py`
- 候选 SHA-256：`243d9727a611d9e642ab59bd4257fc4b81949d8324654253571b4bc7ce47a633`
- 候选 manifest SHA-256：`502d90f259bdd9a8c5782820286381cdd159b3cba7cb08886b8e0a31de9ca87e`
- baseline SHA-256：`a54615a220cfd258958e0cbb9d4fa5d4c0ee348551950bc7a35e1c1918823761`
- visible correctness：`output/local-correctness/_chunk_state_fwd_kernel/localv-243d9727a611.correctness.json`
- visible paired profile：`output/local-paired/_chunk_state_fwd_kernel/localv-243d9727a611.paired.json`
- shape-2 correctness：`output/overfit-probes/chunk-state-case2/local-correctness.json`
- shape-2 paired profile：`output/overfit-probes/chunk-state-case2/local-paired.json`
- environment fingerprint：`f821659614569a674c8403ecbee273c656270e0760dafcd3a646caae132cb662`

两组 profile 的设备均为 `0`，频率均为 `1650 MHz`，目标 profile 名称为
`_chunk_state_fwd_kernel_mix_aic`。sidecar 同时绑定 source/test/manifest hash，并保留 raw
CSV 与日志目录。

## 解释边界

这是一枚真实的 launch 参数变化，不再是注释、格式或等价表达式候选；但它仍只证明本机
910B4 上的两个可见/隔离 shape。它不证明官方 Ascend A2/A3 latency、初赛 50 case 或决赛
隐藏 case 的功能和性能，也不应直接写成官方 speedup。

## 后续处理

保留原 `localv-0ff6b5fc0e82` comment-only candidate 作为可回退证据，不删除失败/旧结果。下一枚
实验应继续采用单变量 launch 或 tile 参数，并在至少一个不同 shape 上重放；neutral 候选不计入
Experience 的“成功策略”统计。
