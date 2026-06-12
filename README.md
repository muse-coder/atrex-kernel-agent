# IterKernel

**渐进式优化 Kernel Agent — Iterative GPU Kernel Optimization**

IterKernel 提供一套可复现、证据优先的 GPU kernel 优化框架:固定 workload
shape、对称 baseline/candidate ABI、CUDA-event A/B 交错计时、正确性门禁、
Nsight Compute 证据、迭代式 agent 优化（RLCR）。

## 仓库结构

```text
docs/                          规则与模板
  kernel_optimization_rules.md   优化护栏（正确性优先、对称 ABI、evidence-backed）
  benchmark_contract.md          benchmark 契约（A/B 交错、CUDA-event、provenance）
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

campaigns/operators/           你的 kernel 优化任务放这里

scripts/                       启动脚本
  launch_task.sh                 通用任务启动器（建 worktree + 起 Claude + RLCR）
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

### 1. 创建一个 kernel 优化任务

```bash
cp -r templates/example_task campaigns/operators/b200_my_kernel__multi_shape

# 编辑任务卡和配置
vim campaigns/operators/b200_my_kernel__multi_shape/prompt.md
vim campaigns/operators/b200_my_kernel__multi_shape/config.toml
```

### 2. 启动 agent

```bash
scripts/launch_task.sh campaigns/operators/b200_my_kernel__multi_shape
```

设置 `IK_NO_CLAUDE=1` 只准备 worktree 不起 Claude。

### 3. 在 Claude Code 内启动 RLCR 循环

启动器会生成 `.rlcr/draft.md`（包含任务卡和约束）。先检查并完善它，保存为
`.rlcr/plan.md`，然后启动循环：

```text
/project:rlcr .rlcr/plan.md --base-branch <printed-ik-base-branch>
```

这会启动一个 Ultracode Workflow，自动迭代直到达到停止条件：
- **Roofline efficiency ≥ 90%** → 优化成功，停止
- **连续 50 轮无进展** → 卡住，停止

无轮次上限，只要有进展就继续优化。全部在 Claude Code 内部闭环，无外部依赖。

## 任务生命周期

每个 kernel 优化任务的目录结构：

```text
prompt.md       任务卡：要优化什么 kernel、约束、第一里程碑
config.toml     build/benchmark 默认值
baseline/       参考实现（对照组）
solution/       你的优化版本
bench/          独立 benchmark + 正确性 harness
docs/           run log、profile 笔记、结果、决策记录
```

核心原则是**对称**：baseline 和 candidate 通过相同的本地接口、固定 workload、
预分配 output、CUDA-event 计时、A/B 交错采样、严格正确性检查来比较。

## RLCR 迭代循环

使用 Claude Code 内置 subagent 驱动（`/project:rlcr`），不依赖外部工具：

1. **Coder subagent** — 写/改 `solution/kernel.cu`，跑 correctness + benchmark
2. **Analyst subagent** — 审查 diff + 跑 NCU profiling + roofline 分析 → 找瓶颈 →
   给出 P0-P3 问题列表 + 下一步优化方向 + roofline efficiency + verdict
3. **Workflow 循环** — 程序化控制：efficiency ≥ 90% 则成功停止，连续 50 轮无进展
   则停止，无轮次上限

每轮迭代开始前必须刷新上下文：任务卡、当前 benchmark 证据、KernelWiki。
状态和每轮记录保存在 `.rlcr/<timestamp>/` 目录下。

## 环境变量

```bash
IK_BASE_BRANCH=<ref>          # worktree 的基准分支（默认当前分支）
IK_NO_CLAUDE=1                # 只建 worktree 不起 Claude
IK_BASH_BIN=/path/to/bash     # 指定 bash 4+（macOS 3.2 不支持）
CLAUDE_MODEL=opus              # Claude 模型（默认 opus）
CLAUDE_EFFORT=max              # Claude effort（默认 max）
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
