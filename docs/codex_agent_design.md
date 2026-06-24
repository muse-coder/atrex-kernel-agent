# Codex Agent Design

本文记录 IterKernel 的 Codex 适配层。Claude Code 版本仍在 `.claude/` 中；Codex 版本
在不改变核心 RLCR 设计的前提下，换掉运行入口和 hook 机制。

## 映射关系

| Claude 版 | Codex 版 | 说明 |
|---|---|---|
| `CLAUDE.md` | `AGENTS.md` | 仓库级 agent 说明 |
| `.claude/commands/optimize-kernel.md` | `.codex/prompts/optimize-kernel.md` | 主 playbook |
| `.claude/agents/*.md` | `.codex/agents/*.md` | 角色契约 |
| `.claude/settings.json` hooks | `scripts/codex_round_guard.py` | Codex 显式门槛检查 |
| `scripts/launch_task.sh` | `scripts/launch_codex_task.sh` | worktree + agent 启动器 |

## 保留不变的契约

- 每个 campaign 使用 `/tmp/<slug>/` 独立 git repo。
- baseline/candidate 对称 ABI、固定 workload、destination-passing、CUDA-event A/B 交错。
- performance verdict 只认 NCU `gpu__time_duration`。
- 每轮必须有 correctness、NCU、5 类静态产物和 `analysis.md`。
- 版本文件模型：`solution/<family>_v<N>.<ext>`，vN 对应 `rounds/r<N>/`。
- code-iter 轮只允许 `cp v<N-1> -> v<N>` 后 Edit 一个 lever。
- analysis 只 recommend 枯竭，master 才 decide。

## Codex 差异

Claude 版依赖 SessionStart / PreToolUse hooks 自动注入规则并阻止跳过门槛。Codex 版不假设
这些 hook 存在，因此把关键门槛变成显式命令：

```bash
python scripts/codex_round_guard.py mark-direction-read /tmp/<slug> <N>
python scripts/codex_round_guard.py pre-edit /tmp/<slug> <N>
```

这不是完整安全沙箱；它是 Codex agent 必须执行的过程检查。真正的防重写仍依赖版本文件模型、
one-lever diff 和 analysis 的事后核查。

## 执行模型

Codex 默认单会话顺序执行四个角色。这样避免依赖某个 Codex surface 是否提供 subagent。
如果未来运行环境提供稳定的 subagent 工具，可以按同一契约扩展，但必须保持：

- master 是唯一 spawner。
- subagent 不互相通信。
- prompt/return 只给路径和轮号，实质信息只走磁盘 artifact。
- pivot analysis 使用 fresh 实例。

## 启动器

`scripts/launch_codex_task.sh` 复用 Claude 版 worktree 思路，但启动 Codex：

- 默认 worktree 根：`../IterKernel-codex-worktrees`
- 默认 branch 前缀：`ik-codex/`
- 默认 review base 前缀：`ik-codex-base/`
- 默认 sandbox：`danger-full-access`，因为 campaign repo 在 `/tmp/<slug>/`，并且需要 GPU 工具。
