# AGENTS.md

This file defines how AI coding agents must work in this repository.

The main goal is not maximum autonomy. The main goal is small, reviewable
changes that a human can actually validate.

## Project Context

This repository is for the 2026 Huawei BiSheng Cup compiler challenge research
and tooling, with the current practical direction focused on Triton automatic
optimization and low-cost local development before real Ascend evaluation.

Important local documents:

- `doc/README.md`
- `doc/00-最新成果-Triton-Experience-System架构调研.md`
- `doc/2026-Triton进化优化研究与实施总计划.md`
- `doc/source_index.md`
- `doc/提示词工程与Skill系统设计方案.md`
- `doc/learn/README.md`
- `doc/learn/导读写作规范.md`

Read `doc/README.md` first. The Experience System report is the latest
architecture result, but it is analysis only and does not mean Experience
retrieval has been implemented. Old stage reports and superseded plans are kept
outside the current documentation set in the local `supply-doc` Git history;
they are not authoritative for current architecture, hardware assumptions, or
implementation order.

Before changing architecture, executor behavior, scoring, cache format,
Experience behavior, or remote Ascend workflow, read `doc/README.md`, the master
plan, and the relevant document first.

Before adding or modifying learning guides under `doc/learn/`, read
`doc/learn/README.md` and `doc/learn/导读写作规范.md`.

## Unified Local Python Environment

Use the repository-local `.venv` as the single default Python development and
test environment:

```bash
python3 -m venv .venv  # only when .venv does not exist
export WLZ_PYTHON="$PWD/.venv/bin/python"
```

- Prefer `"$WLZ_PYTHON"` or `.venv/bin/python` in agent commands; shell
  activation state is not reliable across non-interactive tool calls.
- `.venv/` is generated and ignored. Do not commit it or create a separate
  environment per agent.
- The environment is intentionally minimal and does not inherit system or Conda
  site-packages. Do not install packages ad hoc. Handle any dependency change as
  its own acceptance unit under the dependency rules below.
- The default local suite uses the standard-library `unittest` runner. `pytest`,
  Torch, Triton, CANN, and `torch-npu` are not assumed to be installed.
- Do not point `WLZ_TRITON_PYTHON` at an unrelated shared environment by default.
  CUDA, Triton, or Ascend checks require an explicitly scoped environment task
  and must report the exact interpreter and hardware boundary.

Canonical verification command:

```bash
"$WLZ_PYTHON" -m unittest discover -s tests -v
```

## External Model Workers

When the user requests multi-agent work, or a bounded research/review subtask
can be delegated independently, Codex remains the primary agent and may use the
`agent-link` MCP server to call the configured `claude` worker.

- The `claude` worker is pinned in `~/.agent-link/config.json` to the Claude CLI
  `opus` alias. In this environment that alias maps to `deepseek-v4-pro[1m]`.
- Do not pass a model override in an `agent-link` tool call. The profile disables
  model overrides so the worker cannot silently fall back to Sonnet or Haiku.
- Delegate one concrete, bounded task and require an inspectable result.
- The primary Codex agent owns scope decisions, integration, and verification.
- Parallel workers are appropriate for read-only research or review. Do not let
  multiple write-capable workers edit the same files concurrently.
- Never send credentials, private tokens, or unrelated workspace content to a
  worker prompt.

## Version Control Workflow

This is a colocated Git and Jujutsu repository. Both tools use the same Git
objects and are compatible when their responsibilities remain explicit.

- Use `jj` for daily changes, descriptions, rebases, and bookmark movement.
- Use `git` for GitHub compatibility and read-only inspection such as
  `git status`, `git diff`, and `git log`.
- Do not run history-changing Git and JJ commands concurrently. After adopting
  JJ, avoid routine `git commit`, `git reset`, and `git rebase`.
- Keep accepted history on the tracked `main` bookmark. Verify the acceptance
  unit before moving `main` or pushing it.
- Push accepted work with `jj git push --bookmark main`. Generated output,
  submission archives, downloaded sources, and credentials remain untracked.
- Never commit or print `work/official_triton_agent/set_env/set_api.sh`.

## Core Working Principle

Every task must be split into a small acceptance unit.

A small acceptance unit is a change that:

- has one clear goal,
- can be reviewed by a human in about 5-15 minutes,
- has a concrete verification command or concrete artifact,
- does not silently decide unrelated architecture,
- stops after completion instead of continuing into the next feature.

Do not implement "the whole skeleton", "the full optimizer", "the complete
remote runner", or similarly broad tasks in one pass.

## Code Size Discipline

Code bloat is prohibited. Implement each acceptance unit with the smallest
change that satisfies its concrete behavior and verification.

- Reuse existing protocols, schemas, workers, and helpers before adding a new
  module or parallel abstraction.
- Do not duplicate logic to create a feature-specific framework when an
  existing component can be extended without blurring its responsibility.
- Do not add speculative extension points, compatibility layers, wrappers, or
  configuration fields without a current caller and test.
- Prefer one explicit branch over a new class hierarchy for one narrow mode.
- If the minimal implementation would still exceed the normal scope limits,
  stop and report the expansion instead of hiding it in generated boilerplate.

## Default Scope Limits

Unless the user explicitly asks for a larger change, a single work unit should
stay within these limits:

- At most 3 source files changed.
- At most 1 test file changed.
- At most 1 documentation file changed.
- About 200 net new lines or fewer.
- One concept per turn.

If the work needs to exceed these limits, stop and provide a scope-expansion
report. Do not keep coding.

Examples of one concept:

- Define `Candidate` and `EvaluationResult`.
- Add JSONL cache read/write.
- Add syntax-only local executor checks.
- Add top-5 ranking from existing result objects.
- Add one mock CLI that already has supporting modules.

Examples that are too large:

- Build the whole optimizer skeleton.
- Add local, simulator, remote, cache, ranking, and CLI in one turn.
- Refactor project layout while also changing behavior.
- Add real LLM calls while also changing evolutionary search.

## Architecture Boundaries

The first implementation path is Python-first. Rust can be introduced later as
an optional orchestrator or analysis tool, but the official competition-style
entry path must remain easy to run in Python unless the user explicitly changes
that direction.

Use these conceptual boundaries:

- Candidate/schema code owns stable data contracts.
- Local executor owns cheap local validation only.
- Cache owns result reuse and environment-aware cache keys.
- Ranking owns ordering and top-5 selection.
- Genetic operators own candidate generation.
- Evolutionary algorithm owns search flow.
- Remote Ascend executor owns real remote evaluation, but only after remote
  environment details exist.

Do not blur these boundaries without explaining why.

## Runtime Model Policy

- Competition runtime models must come from the official Appendix 2 allowlist.
- Project default model family is DeepSeek-V4 (current ID: `deepseek-v4-pro`).
- The `ENGINE` fallback default in `work/official_triton_agent/config.py` must
  stay within the DeepSeek-V4 family.
- Never set models outside the allowlist (e.g. DeepSeek-V3) as defaults.

## Ascend And Performance Rules

The local machine must not be treated as a real Ascend evaluator.

Local code may:

- parse Python,
- inspect AST,
- normalize code,
- hash candidates,
- classify obvious errors,
- run mock/proxy checks,
- optionally use Triton interpreter if explicitly added later.

Local code must not:

- invent real Ascend latency,
- label proxy scores as real speedup,
- assume `torch-npu`, CANN, or Triton-Ascend is installed,
- hard-code an Ascend `soc_version`,
- hard-code HiDevLab paths or credentials,
- treat Docker as a virtual NPU.

Real `speedup`, `latency_ms`, and final performance ranking must come from a
real remote Ascend executor or official evaluation output.

If a value is local-only, call it `proxy_score`, `local_score`, or similar. Do
not call it `speedup`.

## Remote Ascend Rules

Do not implement real SSH, rsync, HiDevLab, CANN, or `msprof` logic until the
user provides or confirms a real remote environment.

Before remote implementation, it is acceptable to define:

- interface types,
- configuration schemas,
- `not_configured` behavior,
- mock remote evaluation for local flow testing.

Remote code must not contain:

- real credentials,
- private hostnames unless supplied by the user for that specific task,
- hard-coded usernames,
- hard-coded absolute remote work directories,
- destructive cleanup commands.

## Official Platform Submission Protocol

The CourseGrading online task is a real external evaluation surface. Treat any
upload, submit, cancel, IDE action, or retry as an external side effect. Browser
inspection is read-only by default.

### Fixed Task Identity

For the current Triton evolutionary-optimization task, verify all of these
before doing anything stateful:

- Contest ID: `1mTsU6jaSZ0`.
- Task ID: `14955089`.
- Assignment ID shown by the loaded task: `47585`.
- Problem ID used by the result frame: `3153461`.
- Task title: `2026年编译挑战赛-基于进化算法的Triton自动优化系统`.
- Task URL:
  `https://course.educg.net/pages/contest/contest_submit.jsp?contestID=1mTsU6jaSZ0&taskID=14955089&my=false&contestCID=0`.

Stop if the account, title, IDs, or URL do not match. Never submit to a nearby
contest tab merely because its page layout looks the same.

### Browser And Session Boundary

- Use the configured `chrome-devtools` MCP with the dedicated Chrome profile.
- The Chrome DevTools endpoint must remain bound to `127.0.0.1`; never expose it
  on `0.0.0.0` or a public/LAN interface.
- Never export, print, copy, or send browser cookies, session tokens, passwords,
  or profile files to a terminal command, log, prompt, or sub-agent.
- First call `list_pages`, select the exact task page, and take a fresh snapshot.
  Accessibility UIDs are ephemeral; never reuse a UID from an older snapshot.
- Read-only page inspection may be delegated to a bounded sub-agent. Uploading,
  submitting, cancelling, starting an evaluation, or consuming official quota
  remains owned by the primary agent.
- Do not inspect unrelated logged-in tabs. If login has expired, stop and ask
  the user to log in manually.
- If the site presents a CAPTCHA, anti-bot challenge, unexpected consent page,
  or destructive confirmation, stop. Do not bypass it.

### Known Online Form

The authenticated task page currently exposes:

- UI label `提交源文件` and a required file input named `FILE1`.
- Accepted extensions stated by the page: `rar` and `zip`.
- Maximum archive size enforced by the page: `20,971,520` bytes (20 MiB).
- The visible multipart form resolves to
  `/assignment/programOJPList.jsp?proNum=1&assignID=47585`, but the JavaScript
  submit path sends a `FormData` field named `file` to
  `/assignment/showOJPProcessMsg.jsp?problemID=3153461&assignID=47585&doSubmit=true&wtime=<seconds>`.
- JavaScript-controlled buttons `提 交` and `取消`.
- A result area labelled `运行结果`, backed by
  `/assignment/showOJPProcessMsg.jsp?problemID=3153461&assignID=47585`.
- An `进入在线IDE` action, which must not be opened unless the current task
  explicitly requires it.

These facts describe the online evaluation form, not the complete final-round
delivery contract. Keep these artifact classes distinct:

- an agent source archive used by the online task;
- generated optimized operators packaged as `output.zip` when required;
- the final agent source, design PDF, and preliminary-round MP4 materials.

Do not assume one archive satisfies all three purposes. If the expected archive
layout is not evidenced by the current task instructions or organizer docs,
stop and ask for clarification instead of trial-submitting guessed layouts.
The task text explicitly says to package the generated `output/` directory as
`output.zip`; it does not currently specify the outer layout that combines the
agent source, PDF, and MP4. Do not invent that outer layout. Avoid Chinese
archive filenames because the page renames them before upload.

### Pre-Submission Gate

Before requesting permission to submit, the primary agent must report:

- exact local artifact path, byte size, and SHA-256;
- whether this is a smoke artifact or a scoring/final artifact;
- archive integrity result (`unzip -t` or the RAR equivalent);
- archive root listing and expected entry point;
- exact focused and full test commands already run, including failures;
- confirmation that the archive contains no `.secrets`, API keys, cookies,
  `.git`, `__pycache__`, `.pyc`, local output, or unintended absolute/`..` paths;
- target task title and IDs from a fresh authenticated page snapshot;
- expected model/token/time budget and whether the submission can consume paid
  or organizer-provided quota.

Smoke configuration must be labelled as smoke configuration. It must not be
presented as the final competition configuration.

### Explicit Approval Requirement

Uploading a file and clicking `提 交` require explicit user approval for the
specific artifact hash and target task. A previous generic `continue`, approval
to inspect the page, or approval for an older artifact is not sufficient.

Before the click, state in one short message:

```text
准备提交：<artifact path>
SHA-256：<hash>
目标：<task title / taskID / assignID>
影响：将上传文件并触发官方运行，可能消耗评测和模型额度。
```

Wait for confirmation. Do not combine this approval request with unrelated
questions.

### Submission Execution

After explicit approval:

1. Take a fresh snapshot and re-check account, task title, IDs, and current
   result state.
2. Use the file input from that snapshot to select the approved artifact.
3. Take another snapshot and verify the displayed filename before submitting.
4. Click `提 交` exactly once.
5. Handle only an expected confirmation dialog. Stop on unexpected wording.
6. Do not click `取消`, open the IDE, navigate away, refresh during upload, or
   submit a second time while a run is queued or active.

The page enforces at least 10 seconds between submissions. That is only a UI
rate limit, not permission to retry after 10 seconds. A retry still requires a
diagnosis and fresh explicit approval.

### Result Collection And Meaning

After submission, read the `运行结果` area without modifying the page. Poll at
a conservative interval, normally 10-30 seconds, and respect the official
20-minute-per-operator limit. Keep the user informed during a long run.

The result frame internally polls
`/assignment/showOJPProcessJSON.jsp?assignID=47585&problemID=3153461&userID=<current-user>`.
For the first JSON entry, `ret == "0"` means the platform is still polling;
`ret != "0"` means the platform process reached a terminal state. The website
polls every 2 seconds, but agents should not add another aggressive polling
loop. A terminal `ret` is a completion signal only, not a pass signal.

Record an official-run artifact under `output/official-runs/<timestamp>/` with:

- submitted filename, size, SHA-256, and source revision when available;
- task URL, IDs, account display name, submission time, and completion time;
- exact raw result text and status transitions;
- a screenshot or accessibility snapshot of the final result;
- any visible compile, correctness, timeout, runtime, token, latency, score, or
  speedup fields, preserving missing fields as unknown.

Report result levels separately:

1. `upload accepted`: transport/form submission succeeded.
2. `platform run completed`: the platform stopped queuing/running.
3. `official executor success`: only if the platform explicitly says so.
4. `functional pass`: only if compilation and all required correctness tests
   are explicitly confirmed.
5. `performance result`: only from an explicit official latency/speedup/score.

Never infer a higher level from a lower one. In particular, HTTP success,
`运行结束`, a numeric `1`, or an aggregate `success` flag does not by itself
prove compilation, full correctness, or performance success.

On timeout, login loss, ambiguous output, or failure, preserve the raw result
and stop. Diagnose before requesting approval for another submission; never
blindly retry an identical artifact.

## Cache Rules

Cache is a first-class feature because NPU time is expensive.

Cache keys should be environment-aware. At minimum, distinguish:

- operator name,
- input shape/signature when available,
- normalized code hash,
- executor kind,
- environment fingerprint.

Cache successful results, compile failures, correctness failures, timeouts, and
runtime failures. Repeated failures are also expensive.

Cache files should be human-readable unless there is a clear reason otherwise.
Prefer JSONL for append-only result logs.

## Candidate Provenance Rules

Every candidate must preserve provenance.

Track at least:

- candidate id,
- operator name,
- code hash,
- parent ids,
- generation,
- mutation/crossover kind,
- model used if any,
- prompt id if any,
- evaluation status,
- metadata.

Do not generate anonymous candidate files that cannot be traced back to their
origin.

## Reporting Protocol

Agents must report work in a way that supports human acceptance. Do not provide
vague summaries.

### Before Editing

Before making file edits, provide a short boundary report:

```md
本轮验收单元：<one clear small goal>

计划改动：
- <file>: <what will change>
- <file>: <what will change>

不做事项：
- 不做 <explicitly excluded work>
- 不改 <explicitly excluded area>

验收方式：
- `<command>`
- 人工重点看 <file/behavior>

如果发现需要扩大范围，我会先停下来说明。
```

Keep this short. The goal is to expose scope, not to write an essay.

### When Scope Expands

If the current task requires broader changes than expected, stop and report:

```md
范围超出，暂停。

原因：
- <why the original scope is insufficient>

需要新增的改动：
- <new file/module/dependency/behavior>

建议拆分：
1. <small follow-up task>
2. <small follow-up task>

等待确认后再继续。
```

Do not continue coding after this report unless the user confirms.

### Final Acceptance Report

At the end of each completed work unit, report in this format:

```md
状态：完成 / 部分完成 / 未完成

本轮目标：
- <original small goal>

实际改动：
- <file>: <specific behavior changed>
- <file>: <specific behavior changed>

验收结果：
- `<command>`: 通过 / 失败 / 未运行
- `<command>`: 通过 / 失败 / 未运行

人工验收入口：
- <first file or artifact to inspect>
- <specific behavior or boundary to review>

未做事项：
- <intentionally not done>
- <future work not done>

风险和注意：
- <known risk, or "暂无已知风险">

下一小步建议：
- <one next small step only>
```

If tests were not run, say so and explain why. Do not write "tests pass" without
the exact command.

### Forbidden Reporting Patterns

Do not report only:

- "已优化"
- "已完善"
- "已增强"
- "测试通过"
- "完成了骨架"

Always name concrete files, concrete behavior, and concrete verification.

Do not hide failed commands. Failed verification is useful information.

Do not list many future ideas in the final report. Suggest only one next small
step.

## Testing Rules

Prefer targeted tests for the current acceptance unit.

When changing behavior:

- add or update a focused test,
- run the focused test,
- run the broader test suite if it is cheap and available.

For this repository, common commands may include:

```bash
WLZ_PYTHON="$PWD/.venv/bin/python"
"$WLZ_PYTHON" -m unittest tests.test_local_mock -v
"$WLZ_PYTHON" -m unittest discover -s tests -v
"$WLZ_PYTHON" scripts/run_local_mock.py --help
```

Do not introduce network-dependent tests unless the user explicitly asks.

Do not make tests depend on Ascend hardware, `torch-npu`, CANN, or remote
credentials unless the task is explicitly about real remote evaluation.

## Dependency Rules

Do not add new runtime dependencies without a clear reason.

If a new dependency is needed:

- stop before adding it,
- explain why the standard library or existing dependencies are insufficient,
- state where it will be declared,
- wait for confirmation unless the user explicitly requested that dependency.

Prefer standard-library implementations for early skeleton work.

## File And Artifact Rules

Do not commit generated cache, output, or `__pycache__` artifacts.

Do not edit `doc/` unless the task is documentation work or the user explicitly
asks for doc updates.

Do not rewrite downloaded official materials under `doc/sources/`.

When adding `doc/learn/` guides, use the `NN-主题导读.md` naming convention,
avoid duplicate numbering, and include a concrete human acceptance entry.

Do not perform broad formatting-only rewrites.

Do not delete user-created files or generated experiment output unless the user
explicitly asks.

## Review Expectations

Human review should be able to answer these questions quickly:

- What was this unit supposed to do?
- Which files changed?
- What command verifies it?
- What should I inspect manually?
- What was intentionally not done?
- What is the next smallest useful step?

If the answer is not obvious from the final report and diff, the work unit was
too large or reported poorly.

## Suggested Work Sequence

For the current optimizer skeleton, prefer this order:

1. Minimal importable package and test setup.
2. Core schemas: `Candidate`, `EvaluationResult`, `EvalContext`.
3. Code normalization and stable code hash.
4. Local static-check executor.
5. JSONL cache.
6. Mock candidate generator.
7. Ranking and top-5 selection.
8. Local mock CLI and output artifacts.
9. Remote Ascend config/interface stub.
10. Documentation for the local mock loop.

Do not skip directly to real LLM, real remote Ascend, or simulator integration
before the local mock loop is reviewable.
