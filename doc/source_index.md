# Triton 赛题资料索引

状态：当前资料入口；不复制或改写官方原始材料
更新时间：2026-08-03

## 1. 先读什么

1. `doc/README.md`：当前唯一文档入口和事实权威顺序。
2. `doc/02-最近成果-本机910B4-21算子资格闭环.md`：当前 21 算子的本机资格与证据边界。
3. `doc/00-最新成果-Triton-Experience-System架构调研.md`：最新架构成果。
4. `doc/2026-Triton进化优化研究与实施总计划.md`：当前研究方向、边界和实施顺序。
5. 本索引：定位官方原始资料和固定证据。

发生冲突时，官方最新材料和该次平台原始结果高于内部 Markdown；当前源码、测试和可复核输出高于
历史计划。文件名含“官方”的内部报告仍是项目审计/解读，不是赛方原文。

## 2. 原始资料边界

官方 PDF、下载 archive、网页快照和培训材料保留在仓库外的本机资料目录：

```text
/workspace/user_data/supply-doc/sources/
```

该目录的原始材料不因项目文档升级而改写、移动或删除。仓库 `.gitignore` 继续忽略：

```text
doc/sources/
doc/repos/
```

因此当前 Git 仓库中的 `doc/` 只保存项目分析、计划、学习导读和索引。需要复核官方原文时，从外部
资料目录读取；不要为了让 Markdown 链接方便而复制大型 archive 进仓库。

## 3. 当前主赛题核心原文

核心中文技术方案：

```text
/workspace/user_data/supply-doc/sources/official_compiler2026_archive/compiler2026-main/
2026年全国大学生计算机系统能力大赛编译系统设计赛-编译系统挑战赛-
基于进化算法的Triton自动优化系统-技术方案.pdf
```

已核验 SHA-256：

```text
fbd4ec3ecff3b6a7202cb2b692be3aed4ee12658915b375ad757ebdb482a31bd
```

同目录包含章程、英文版和其他赛题技术方案。其他赛题材料只能用于区分边界，不能据此宣称本项目
同时参加 Triton-Ascend 或 AscendNPU IR 赛题。

核心原文支持的约束摘要：

- 给定 Agent 框架内实现进化算法；
- 每算子 20 分钟或 20 万 token；
- Multi-Agent token 合计；
- 最多返回 5 个版本；
- 至少一个非原样候选成功编译并通过全部测试；
- 只有功能通过候选进入性能测试；
- 初赛公开 50 case，决赛追加 50 个隐藏 case；
- 官方目标为鲲鹏 920、Ascend A2/A3、openEuler。

“官方只给 LLM API，运行时不允许依赖 Claude Code/Codex CLI”是当前实现边界，不应伪装成上述 PDF
中的逐字条款，除非后续任务正文或官方通知提供明确原文。

## 4. 当前在线任务身份

任何上传、提交、取消或 IDE 操作前，都必须按 `AGENTS.md` 从 fresh authenticated page 复核：

```text
Contest ID:    1mTsU6jaSZ0
Task ID:       14955089
Assignment ID: 47585
Problem ID:    3153461
Task title:    2026年编译挑战赛-基于进化算法的Triton自动优化系统
```

课程平台任务 URL：

<https://course.educg.net/pages/contest/contest_submit.jsp?contestID=1mTsU6jaSZ0&taskID=14955089&my=false&contestCID=0>

这些 ID 用于防止误操作附近赛题；它们不是对当前登录账号或页面状态的替代。登录失效、标题/ID 不
一致或出现 CAPTCHA 时必须停止。

## 5. 外部资料目录结构

以下路径均相对 `/workspace/user_data/supply-doc/sources/`：

| 路径 | 内容 | 当前用途 |
| --- | --- | --- |
| `official_compiler2026_archive/` | 2026 官方 archive 和 PDF | 赛题规则最高等级本地证据 |
| `official_compiler2026_text/` | PDF 文本提取结果 | 搜索辅助；歧义时回到 PDF |
| `compiler.educg.net/` | 官网 API、网页和静态资源快照 | 固定日期的网站证据，不代表当前页面 |
| `official_compiler2025_archive/` | 2025 官方历史资料 | 背景参考，不覆盖 2026 规则 |
| `pan.educg.net/` | 培训课件、视频和 manifest | 学习资料，不是全部内容都与本赛题相关 |
| `logs/` | 下载和网络诊断日志 | 只用于复核采集过程 |

外部 `supply-doc` 还保存内部历史 Markdown 和自己的 Git 历史。被当前 `doc/` 删除的 Rust-first、
mock、冲刺、交接和阶段导向稿需要追溯时从那里查看，不重新放回当前阅读路径。

## 6. 当前内部证据文档

以下是内部分析，不是官方原文：

- `doc/02-最近成果-本机910B4-21算子资格闭环.md`：当前可见 case 的 21 算子本机 qualification 矩阵和原始 sidecar 入口。
- `doc/01-当前赛题与官方平台说明.md`：赛题和平台概览，在线流程仍需 fresh page 复核。
- `doc/2026-Triton官方接口能力审计.md`：绑定固定官方框架版本的接口审计。
- `doc/2026-Triton官方评测四类问题纠错报告.md`：历史官方运行结果的派生分析。

历史数字、测试数、硬件状态和“下一步”只在其日期/commit 范围内成立。当前实现以 `doc/README.md`、
总计划和当前代码为准。

## 7. 公开技术入口

- 大赛/课程平台：<https://course.educg.net/course/6-366>
- 2026 官方资料 GitLab：<https://gitlab.eduxiji.net/csc1/nscscc/compiler2026>
- Triton-Ascend：<https://github.com/triton-lang/triton-ascend>
- Triton 文档：<https://triton-lang.org/main/index.html>

研究参考必须固定论文版本、仓库 commit、许可证和访问日期。公开项目能证明某机制存在，不能证明
它适合 Ascend、适合本赛题预算或优于当前基线。

## 8. 安全和维护

- 不读取、打印、提交或发送 API key、cookie、session token、浏览器 profile 或
  `work/official_triton_agent/set_env/set_api.sh`。
- 下载脚本和网页快照不自动执行；先检查来源、路径和副作用。
- 官方原始材料只追加新版本，不追溯性修改旧版本。
- 新官方通知到达时，先保存原文和 hash，再更新本索引与总计划中的派生结论。
- 平台上传或提交必须遵守 `AGENTS.md` 的 artifact gate 和特定 hash 明确批准要求。
