# IterKernel

**渐进式优化 Kernel Agent — Iterative GPU Kernel Optimization**

IterKernel 提供一套可复现、证据优先的 GPU kernel 优化框架:固定 workload
shape、对称 baseline/candidate ABI、CUDA-event A/B 交错计时、正确性门禁、
Nsight Compute 证据、迭代式 agent 优化（RLCR）。

## 仓库结构

```text
AGENTS.md                      Codex 版仓库级 agent 入口
.codex/                        Codex 版 prompt 与角色契约
  prompts/optimize-kernel.md     Codex master playbook
  agents/                        master / analysis / code-impl / code-iter 契约

docs/                          规则与模板
  kernel_optimization_rules.md   优化护栏（正确性优先、对称 ABI、证据支撑）
  benchmark_contract.md          benchmark 契约（A/B 交错、CUDA-event、来源追溯）
  correctness_contract.md        正确性要求（poison、oracle、回归网格）
  benchmark_template.py          标准 benchmark harness（复制到 bench/ 使用）

templates/                     任务模板
  example_task/                  复制这个目录来创建新的 kernel 优化任务
    prompt.md                    任务卡模板
    config.toml                  build/benchmark 默认值模板
    baseline/.gitkeep
    solution/.gitkeep
    bench/.gitkeep
    docs/.gitkeep

campaigns/operators/           campaign 目录占位（只保留 .gitkeep，
                               实际 kernel 代码在 /tmp/<slug>/ 独立 repo）

scripts/                       可选 worktree 启动器
  launch_task.sh                 为 agent 仓库建隔离 worktree 并起 Claude
  launch_codex_task.sh           为 agent 仓库建隔离 worktree 并起 Codex
  codex_round_guard.py           Codex 版手动过程门槛检查
  launch_tasks/                  每个任务一个启动脚本

external/                      知识子模块（可选）
  KernelWiki/                    Blackwell/Hopper kernel 优化知识库
  ncu-report-skill/              Nsight Compute profiling 方法论
```

## 环境准备

```bash
# 克隆仓库
git clone <repo-url> && cd IterKernel
git submodule update --init --recursive

# 安装 FlashInfer AOT（默认 baseline 来源，必须用最新源码的 AOT 编译版）
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
FLASHINFER_ENABLE_AOT=1 pip install -e . -v
cd ..
```

**必须从最新 main 源码编译 AOT 版本**（`FLASHINFER_ENABLE_AOT=1`），不要用
pip 预编译 wheel（版本滞后）,也不要用 JIT 模式（运行时编译有额外开销,
不代表部署态性能）。AOT 版预编译所有 cubin,是 FlashInfer 实际部署时的路径,
性能最强。编译要求见
[FlashInfer 安装文档](https://docs.flashinfer.ai/installation.html)。

## 快速开始

在 IterKernel 仓库根目录打开 Claude Code，直接运行 slash 命令：

```text
/optimize-kernel <kernel 描述>
```

例如：

```text
/optimize-kernel 在 RTX PRO 5000 上优化 M=1024 的 FP8 GEMM，baseline 用 FlashInfer AOT
```

命令会由当前会话里的 **master agent** 编排完成整个优化流程：master 负责战略判断，
并用 Agent 工具 spawn `analysis` / `code-impl` / `code-iter` 三类 subagent；不调
Workflow，完整编排见 `.claude/commands/optimize-kernel.md`：

1. 理解需求、检测目标 GPU（`nvidia-smi`）、推断 workload shapes
2. 为该 campaign 创建**独立 git 仓库** `/tmp/<slug>/`；在 agent 仓库
   `campaigns/operators/<slug>/` 只保留目录结构（`.gitkeep`）
3. Profile baseline（NCU 实测 + PTX/SASS 静态分析）
4. 分析瓶颈、设计 kernel 架构、模块分解（`// MODULE: <id>` 标记）
5. 按复杂度选择实现原语（纯 CUDA+PTX / CUTLASS / CuTe DSL），从零实现 candidate
6. 进入 RLCR 模块循环，逐模块增量迭代直到达到停止条件

所有 kernel 代码的修改和 commit 都发生在 `/tmp/<slug>/` 独立 repo 中，
不提交到 agent 仓库（见 CLAUDE.md「Kernel 代码仓库隔离」）。

### Codex 版本

Codex 适配层保留同一套 RLCR / artifact-first 设计，但入口换成 `AGENTS.md` 与
`.codex/`：

```bash
scripts/launch_codex_task.sh campaigns/operators/example_task
```

这会创建任务专属 worktree，并启动 Codex 读取 `AGENTS.md` 与
`.codex/prompts/optimize-kernel.md`。Codex 版不依赖 Claude hooks；进入 code-iter 轮时，
读完 `direction.md` 后、编辑 `solution/` 前，必须通过
`scripts/codex_round_guard.py` 显式检查过程门槛。设计说明见
[`docs/codex_agent_design.md`](docs/codex_agent_design.md)。

## 任务生命周期

每个 campaign 在 `/tmp/<slug>/` 独立 git 仓库中进行（kernel 代码不提交到
agent 仓库）。独立 repo 的目录结构：

```text
prompt.md       任务卡：要优化什么 kernel、约束、第一里程碑
config.toml     build/benchmark 默认值
baseline/       参考实现（对照组）
solution/       你的优化版本
bench/          独立 benchmark + 正确性 harness
docs/           运行日志、profile 笔记、结果、决策记录
.rlcr/          RLCR 循环状态与每轮记录
```

核心原则是**对称**：baseline 和 candidate 通过相同的本地接口、固定 workload、
预分配 output、CUDA-event 计时、A/B 交错采样、严格正确性检查来比较。

## RLCR 迭代循环

`/optimize-kernel` 在 Step 7 进入逐模块的 RLCR 迭代，由 master 编排 `code-iter`
和 `analysis` 交替完成。轮次用**全局递增编号 N**，**每轮一个目录** `.rlcr/current/rounds/r<N>/`
存放该轮全部产物（**本地，不 commit**）。一轮三段（对应命令文档 Step 7a/7b/7c）：

1. **实现（7a）** — 读 `rounds/r<N>/direction.md`，对目标 MODULE 做**增量 Edit**
   （严禁 Write 覆盖 `solution/` 文件），`benchmark.py --correctness-only` + benchmark 粗筛，commit 代码
2. **Profile（7b）** — NCU 实测 + PTX/SASS 静态分析，全部存入 `rounds/r<N>/`
3. **分析（7c）** — 综合 NCU/SASS 证据写 `rounds/r<N>/analysis.md`，给出 verdict、更新
   `state.md`、写下一轮 `rounds/r<N+1>/direction.md`

每轮只 commit `solution/` 代码；`.rlcr/`（含每轮目录）整个 gitignore，过程
记录与 profile 数据只留本地。**每轮硬约束的完整清单以 `.claude/commands/optimize-kernel.md`
Step 7 为权威**（CLAUDE.md「RLCR 每轮硬约束」是其检查清单镜像）。

停止条件（**无轮次上限**，只要有进展/有方向就继续）：
- **roofline efficiency ≥ 90%** → 优化成功，全部结束
- 某模块持续无进展 → 转去优化下一个模块（不是停止，之后可回来再试）
- 所有模块暂无新方向 → 拓宽搜索空间找新方向，不轻易判定"到顶"
- 仅在 roofline 达标或用户明确叫停时停止

每轮迭代开始前必须刷新上下文：任务卡、当前 benchmark 证据、KernelWiki。
RLCR 状态保存在独立 repo 的 `.rlcr/current/` 目录下。

## 可选：worktree 启动器

日常使用直接在仓库根运行 `/optimize-kernel` 即可。`scripts/launch_task.sh`
是一个可选便捷启动器：为 agent 仓库创建隔离 git worktree 并在其中起 Claude
Code，你再在里面运行 `/optimize-kernel`。环境变量：

```bash
IK_BASE_BRANCH=<ref>          # worktree 的基准分支（默认当前分支）
IK_NO_CLAUDE=1                # 只建 worktree 不起 Claude
IK_BASH_BIN=/path/to/bash     # 指定 bash 4+（macOS 3.2 不支持）
CLAUDE_MODEL=opus             # Claude 模型（默认 opus）
CLAUDE_EFFORT=max             # Claude effort（默认 max）
```

## 外部知识（可选）

```bash
git submodule update --init --recursive
```

- **KernelWiki** — Blackwell/Hopper kernel 优化知识库
  （tcgen05、NVFP4、FA4、DeepGEMM、CUTLASS/FlashInfer PR 引用）
- **ncu-report-skill** — Nsight Compute profiling 方法论与报告分析

## 核心文档

- [`docs/kernel_optimization_rules.md`](docs/kernel_optimization_rules.md) — 优化护栏
- [`docs/benchmark_contract.md`](docs/benchmark_contract.md) — benchmark 规则
- [`docs/correctness_contract.md`](docs/correctness_contract.md) — 正确性要求
