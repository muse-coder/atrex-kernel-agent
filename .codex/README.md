# Codex Agent for IterKernel

`.codex/` 是 IterKernel 的 Codex 适配层。它不替换 `.claude/`，而是把同一套
RLCR / artifact-first kernel 优化流程映射到 Codex 的运行方式。

## 文件

- `prompts/optimize-kernel.md`：Codex master 的主 playbook。
- `agents/master.md`：战略层职责。
- `agents/analysis.md`：诊断角色契约。
- `agents/code-impl.md`：从头实现角色契约。
- `agents/code-iter.md`：渐进修改角色契约。
- `../AGENTS.md`：Codex 在仓库根默认读取的项目说明。
- `../scripts/codex_round_guard.py`：Codex 手动门槛检查，替代 Claude hooks 的关键约束。

## 推荐启动

```bash
scripts/launch_codex_task.sh campaigns/operators/example_task
```

常用环境变量：

```bash
CODEX_BIN=codex                 # Codex 可执行文件
CODEX_MODEL=gpt-5               # 传给 codex -m
CODEX_SANDBOX=danger-full-access # 默认允许 /tmp campaign repo 和 GPU 工具
CODEX_APPROVAL=never            # 默认不打断自动流程
IK_NO_CODEX=1                   # 只创建 worktree，不启动 Codex
```

启动器会从 `IK_BASE_BRANCH`（默认当前分支 HEAD）创建新 worktree，因此该 ref 必须已经包含
`AGENTS.md`、`.codex/` 和 `scripts/codex_round_guard.py`。如果这些文件还只是当前工作区的
未提交改动，请先提交，或直接在当前工作区手动启动 Codex。

## 手动运行

在仓库根启动 Codex 后，对它说：

```text
读取 AGENTS.md 和 .codex/prompts/optimize-kernel.md，然后用 Codex master 流程优化：
<kernel 描述>
```

进入 code-iter 轮时，编辑 `solution/` 前必须显式执行：

```bash
python scripts/codex_round_guard.py pre-edit /tmp/<slug> <N>
```

读完本轮 `direction.md` 后执行：

```bash
python scripts/codex_round_guard.py mark-direction-read /tmp/<slug> <N>
```
