# IterKernel Codex Agent

本文件是 Codex 在本仓库中的默认入口说明。Claude Code 版本仍保留在
`CLAUDE.md` 和 `.claude/` 下；Codex 版本的入口在 `.codex/` 下。

## 启动方式

- 日常启动：`scripts/launch_codex_task.sh <kernel-task-dir>`
- 手动启动：在仓库根目录启动 Codex 后，先读 `.codex/prompts/optimize-kernel.md`
  和本文件，再按用户给出的 kernel 描述执行。

## Codex 适配原则

Codex 版保留 IterKernel 的核心设计：master 定战略，analysis 诊断，code-impl 从头实现，
code-iter 渐进修改，所有实质信息通过磁盘 artifact 交接。

Codex 与 Claude Code 的差异：

- Codex 不依赖 Claude slash command。主 playbook 是
  `.codex/prompts/optimize-kernel.md`。
- Codex 不默认拥有 `.claude/settings.json` 中的 SessionStart / PreToolUse hooks。
  进入迭代阶段后，编辑 `solution/` 前必须显式运行
  `python scripts/codex_round_guard.py pre-edit <campaign-dir> <round>`。
- Codex 不依赖 Claude `Agent` / `SendMessage` 语义。默认由一个 Codex 会话按角色契约
  顺序执行；如果当前 Codex 运行环境提供 subagent 工具，也必须保持 master-only spawn、
  扁平树和磁盘 artifact 契约。

## 硬性要求

1. candidate kernel 必须从头设计并实现。任何旧 campaign、库 kernel、抄来的 kernel
   都只能作为参考，不能作为代码起点。
2. baseline 必须是现成库中实测最快者。PyTorch/cuBLAS 和 FlashInfer AOT 都要测，取快者。
3. 性能结论只认 NCU kernel duration，也就是 `gpu__time_duration`。benchmark wall-clock
   只作 sanity。
4. 完成判据是达到该 shape roofline 上限的 90% 以上，不是“打平 baseline”。
5. 每轮只改一个 lever。code-iter 轮必须 `cp v<N-1> -> v<N>`，再只 Edit `v<N>`。
6. 性能退化不回退。退化是数据，继续分析并前进；只有编译失败或正确性失败可按错误恢复流程
   回到上次正确版本。
7. 每轮必须生成 correctness 标记、NCU 报告、5 类静态产物和 `analysis.md`，否则不能进入
   下一轮代码修改。

## 仓库边界

本 agent 仓库只保存规则、模板、launcher 和 campaign 占位目录。实际 kernel 代码、
benchmark 结果和 `.rlcr/` 过程状态保存在 `/tmp/<slug>/` 的独立 git repo 中，不提交到
本仓库。

## Codex 角色契约

每次切换角色前先读对应文件：

- master：`.codex/agents/master.md`
- analysis：`.codex/agents/analysis.md`
- code-impl：`.codex/agents/code-impl.md`
- code-iter：`.codex/agents/code-iter.md`

Claude 版 `.claude/commands/optimize-kernel.md` 仍可作为完整历史 playbook 参考；Codex
执行时以 `.codex/` 文件为入口，以 `docs/` 下的 benchmark/correctness/optimization
规则为共享约束。
