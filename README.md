# PPT视频网盘地址1
我用夸克网盘给你分享了「比赛项目PPT视频讲解」，点击链接或复制整段内容，打开「夸克APP」即可获取。
/~3ae73a0jNk~:/
链接：https://pan.quark.cn/s/68a74881d396
链接有问题可以联系fengfeng_feng20@163.com

# Triton Optimization Agent

面向 2026 华为毕昇杯“基于进化算法的 Triton 自动优化系统”赛题的研究与工程项目。

项目的核心思路是：使用大模型提出 Triton kernel 的 mutation/crossover 候选，由进化算法管理候选搜索，再通过静态检查、正确性测试和真实硬件性能测量决定候选去留。

> 大模型提供搜索空间，进化算法组织搜索过程，真实评测提供可信反馈，失败历史帮助系统避免重复试错。

## 项目能力

- 以 Python 为主的比赛 runtime 和离线开发工具；
- LLM mutation/crossover、候选 lineage、去重、排序和 Top-5 输出；
- 统一管理 LLM Token、全流程墙钟和评测预算；
- 语法、接口、导入、编译、正确性和性能的分层验证；
- 保存成功与失败候选的 provenance、环境指纹和评测证据；
- 本机 Ascend 910B4 的 correctness 与 paired benchmark 开发闭环。

## 当前证据边界

当前本机 qualification matrix 为 `21/21`，覆盖当前 checkout 的 21 个公开算子和当前可见 case。该结果属于 `local_ascend_910b4` 开发证据，不是官方 Ascend A2/A3 成绩，也不代表隐藏 case 通过或最终比赛排名。

Experience Retrieval 目前是架构规划方向，尚未接入完整正式 runtime；完整 Multi-Agent 也尚未被证明在同等预算下优于多生成候选。

## 目录入口

- `doc/README.md`：当前文档入口和项目状态；
- `doc/2026-Triton进化优化研究与实施总计划.md`：研究假设、架构边界和实施顺序；
- `doc/00-最新成果-Triton-Experience-System架构调研.md`：Experience System 架构分析；
- `doc/02-最近成果-本机910B4-21算子资格闭环.md`：本机 Ascend 开发证据；
- `doc/项目介绍PPT文案.md`：比赛介绍 PPT 文案底稿；
- `scripts/build_project_intro_ppt.py`：PPTX 生成脚本；
- `doc/项目介绍PPT.pptx`：生成的比赛介绍PPT。

## 本地验证

使用仓库统一 Python 环境：

```bash
WLZ_PYTHON="$PWD/.venv/bin/python"
"$WLZ_PYTHON" -m unittest discover -s tests -v
```

本机 Ascend 运行需要明确配置的 CANN、设备和测试环境，不应把普通本地环境或 CUDA proxy 结果当作官方评测结果。

## 合规与提交边界

项目不得针对特定测试用例硬编码、探测评测环境或绕过评测。第三方代码、开源组件和 AI 辅助生成内容应按比赛要求记录来源、许可证、使用范围和人工审查情况。
