# IterKernel

**渐进式优化 Kernel Agent — Iterative GPU Kernel Optimization Framework**

IterKernel provides a reproducible, evidence-first framework for autonomous GPU
kernel optimization. It turns real serving-framework kernels into structured
optimization tasks: frozen production shapes, copied upstream baselines,
symmetric benchmarks, correctness gates, Nsight Compute evidence, and
iterative agent refinement (RLCR) in one place.

Derived from [KDA-Pilot](https://github.com/BBuf/KDA-Pilot), generalized to be
framework-agnostic — not tied to any specific upstream (SGLang, vLLM, PyTorch,
etc.).

## What's Inside

```text
docs/                          Framework rules and templates
  kernel_optimization_rules.md   optimization guardrails (correctness-first,
                                 symmetric ABI, evidence-backed no-go, ...)
  benchmark_contract.md          standalone benchmark contract (A/B interleaved,
                                 CUDA-event timing, provenance, ...)
  correctness_contract.md        correctness requirements (poison, oracle, grid)
  benchmark_template.py          standard benchmark harness (copy to bench/)
  llm_kernel_workflow_rules.md   LLM serving kernel discovery workflow

templates/                     Task directory template
  example_task/                  copy this to start a new kernel task
    prompt.md                    task card template
    config.toml                  build/benchmark defaults template
    baseline/.gitkeep
    solution/.gitkeep
    bench/.gitkeep
    docs/.gitkeep

campaigns/                     Your kernel optimization campaigns
  operators/                     standalone operator tasks (norm, rope, ...)
  llm/                           LLM kernel-workflow campaigns

scripts/                       Launcher scripts
  launch_task.sh                 generic task launcher (worktree + Claude)
  launch_tasks/                  per-task launcher wrappers

external/                      Knowledge submodules (optional)
  KernelWiki/                    Blackwell/Hopper kernel optimization wiki
  ncu-report-skill/              Nsight Compute profiling skill
```

## Quick Start

### 1. Create a new kernel task

```bash
# Copy the template
cp -r templates/example_task campaigns/operators/b200_my_kernel__multi_shape

# Edit prompt.md and config.toml for your kernel
vim campaigns/operators/b200_my_kernel__multi_shape/prompt.md
vim campaigns/operators/b200_my_kernel__multi_shape/config.toml
```

### 2. Launch with the agent

```bash
# Create a per-task launcher (optional but recommended)
cat > scripts/launch_tasks/k01_b200_my_kernel.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IK_LAUNCHER_NAME="${IK_LAUNCHER_NAME:-$(basename "$0")}"
export IK_LAUNCHER_NAME
exec "$SCRIPT_DIR/../launch_task.sh" "campaigns/operators/b200_my_kernel__multi_shape" "$@"
EOF
chmod +x scripts/launch_tasks/k01_b200_my_kernel.sh

# Run it
scripts/launch_tasks/k01_b200_my_kernel.sh
```

Or launch directly:

```bash
scripts/launch_task.sh campaigns/operators/b200_my_kernel__multi_shape
```

Set `IK_NO_CLAUDE=1` to prepare the worktree without launching Claude.

### 3. Inside Claude Code, start the RLCR loop

```text
/humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct
/humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --base-branch <printed-ik-base-branch>
```

## Task Lifecycle

Every kernel task follows the same shape:

```text
prompt.md       task card for the agent
config.toml     benchmark/build defaults
baseline/       copied upstream baseline source
solution/       optimized candidate source
bench/          standalone benchmark and correctness harness
docs/           run logs, profile notes, results, decision ledger
```

The key rule is **symmetry**: baseline and candidate must be compared through
matching local interfaces, fixed workload rows, preallocated outputs,
CUDA-event timing, interleaved A/B sampling, strict correctness checks, and
full provenance.

## RLCR Loop

The optimization loop uses the [Humanize](https://github.com/PolyArch/humanize)
plugin for Claude Code:

1. **Claude implements** — writes/modifies `solution/kernel.cu`, runs bench
2. **Codex reviews** — independently reviews the diff against the base branch
3. **Claude corrects** — addresses review findings, re-benchmarks
4. **Repeat** — bounded by `--max N` rounds

Each iteration must refresh context from the task prompt, rules, current
evidence, and knowledge skills (KernelWiki, ncu-report-skill) before choosing
the next edit.

## Environment Overrides

```bash
IK_BASE_BRANCH=<ref>          # base branch for worktree (default: current)
IK_NO_CLAUDE=1                # prepare worktree without launching Claude
IK_BASH_BIN=/path/to/bash     # force modern bash (macOS 3.2 rejected)
CLAUDE_MODEL=opus              # Claude model (default: opus)
CLAUDE_EFFORT=max              # Claude effort (default: max)
```

## External Knowledge (Optional)

```bash
git submodule update --init --recursive
```

- **KernelWiki** — Blackwell/Hopper kernel optimization knowledge base
  (tcgen05, NVFP4, FA4, DeepGEMM, upstream PR references)
- **ncu-report-skill** — Nsight Compute profiling methodology and report
  analysis

## Key Documents

- [`docs/kernel_optimization_rules.md`](docs/kernel_optimization_rules.md) —
  optimization guardrails
- [`docs/benchmark_contract.md`](docs/benchmark_contract.md) — benchmark rules
- [`docs/correctness_contract.md`](docs/correctness_contract.md) — correctness
  requirements
- [`docs/llm_kernel_workflow_rules.md`](docs/llm_kernel_workflow_rules.md) —
  LLM kernel discovery workflow
