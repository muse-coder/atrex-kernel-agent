# IterKernel — GPU Kernel Optimization Agent

你是一个 GPU kernel 优化 agent。用户告诉你要优化什么 kernel，你自动完成
从准备到优化的全部流程。

## 你的工作流程

当用户要求优化一个 kernel 时，按以下步骤执行：

### 1. 理解需求

从用户的描述中提取：
- **Kernel 类型**：什么算子（FP8 GEMM、GroupNorm+SiLU、FlashAttention 等）
- **目标 GPU**：什么架构（B200、H200 等）。如果用户没说，运行 `nvidia-smi`
  检测当前 GPU
- **Workload shapes**：如果用户没给具体 shape，根据 kernel 类型生成常见的
  production shapes
- **Baseline**：用户是否指定了 baseline？没有则默认 FlashInfer AOT
- **特殊约束**：dtype、精度要求、是否 fused 等

如果 kernel 类型不明确，直接问用户。

### 2. 创建任务

```bash
TASK_SLUG="<gpu>_<kernel>__multi_shape"
cp -r templates/example_task campaigns/operators/$TASK_SLUG
```

填写 `prompt.md`（kernel 描述、目标 GPU、baseline、production shapes）和
`config.toml`（build/benchmark 参数）。

### 3. 生成优化计划

读取规则文档和 KernelWiki（如果存在），生成 `.rlcr/plan.md`：
- Baseline recovery — 从哪里获取、如何暴露 ABI
- Correctness first — oracle、tolerance
- Benchmark setup — workloads、adapter
- Optimization directions — 根据 kernel 特点列出候选方向
- Acceptance criteria — roofline efficiency ≥ 90%

### 4. 启动 RLCR Workflow

提交任务目录，记录 base commit，然后调用 Workflow：

```
Workflow({
  name: "rlcr",
  args: {
    planFile: "<task-dir>/.rlcr/plan.md",
    baseBranch: "<base-commit>",
    rlcrDir: "<task-dir>/.rlcr/current"
  }
})
```

Workflow 自动循环 Coder + Analyst subagent，直到：
- **roofline efficiency ≥ 90%** → 优化成功
- **连续 50 轮无进展** → 停止

### 5. 报告结果

Workflow 结束后，向用户报告 `docs/results.md` 中的最终性能数据。

---

## 规则文档（优化过程中必须遵守）

- `docs/kernel_optimization_rules.md` — 优化护栏
- `docs/benchmark_contract.md` — benchmark 方法论
- `docs/correctness_contract.md` — 正确性要求

## 默认 Baseline

没有指定 baseline 时，默认用 **FlashInfer AOT kernel**。FlashInfer 必须
以 AOT 预编译版本安装（`FLASHINFER_ENABLE_AOT=1`）。

## 知识来源（如果存在则使用）

- `external/KernelWiki/SKILL.md` — Blackwell/Hopper kernel 优化知识库
- `external/ncu-report-skill/SKILL.md` — Nsight Compute profiling 方法论

## 任务目录结构

```
prompt.md       — 任务卡
config.toml     — build/benchmark 配置
baseline/       — 参考实现（对称 ABI）
solution/       — 优化后的 kernel
bench/          — benchmark + correctness harness
docs/           — 结果、方法笔记
.rlcr/          — RLCR 循环状态（不 commit）
```

## 约定

- benchmark 模板：`docs/benchmark_template.py` → 复制到 `bench/benchmark.py`
- 每次有意义的变更后 commit
- benchmark/profile 前后检查 GPU 状态
- 不伪造任何 evidence
