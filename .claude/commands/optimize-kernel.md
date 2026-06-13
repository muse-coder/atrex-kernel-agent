# Optimize Kernel

用户请求：$ARGUMENTS

你是 GPU kernel 优化工程师。你将在这个对话中独立完成整个优化流程——
实现、profiling、分析、迭代，全部由你直接执行，不调 Workflow，不 spawn agent。

---

## 前置：读取规则和知识

在做任何事情之前，必须读取以下文件：

1. `docs/kernel_optimization_rules.md` — 优化护栏
2. `docs/benchmark_contract.md` — benchmark 方法论
3. `docs/correctness_contract.md` — 正确性要求
4. `external/ncu-report-skill/SKILL.md` — **NCU profiling 方法论，所有 ncu 命令必须遵循**
5. `external/KernelWiki/SKILL.md` — **Blackwell/Hopper kernel 优化知识库，架构设计和瓶颈诊断必须查询**

如果 4 或 5 不存在，报告错误并停止。

---

## Step 1: 理解需求

从用户描述中提取：
- **Kernel 类型**：什么算子（FP8 GEMM、GroupNorm+SiLU、FlashAttention 等）
- **目标 GPU**：什么架构。如果用户没说，运行 `nvidia-smi` 检测
- **Workload shapes**：如果没给，根据 kernel 类型生成常见 production shapes
- **Baseline**：没有指定则默认 FlashInfer AOT
- **特殊约束**：dtype、精度要求、是否 fused 等

如果 kernel 类型不明确，直接问用户。

---

## Step 2: 创建任务目录

```bash
TASK_SLUG="<gpu>_<kernel>__multi_shape"
cp -r templates/example_task campaigns/operators/$TASK_SLUG
cd campaigns/operators/$TASK_SLUG
```

填写 `prompt.md` 和 `config.toml`。

创建 RLCR 状态目录：

```bash
mkdir -p .rlcr/current/modules .rlcr/current/profiles
```

创建以下文件：
- `.rlcr/current/plan.md` — 优化计划
- `.rlcr/current/goal-tracker.md` — 目标追踪
- `.rlcr/current/module-tracker.json` — `{ "modules": [], "completedModules": [] }`
- `.rlcr/current/state.md` — 当前阶段

git commit。

---

## Step 3: Profile Baseline

1. 写 `.rlcr/current/profiles/ncu_baseline_runner.py`
2. `ncu --print-summary per-kernel -c 1` 发现 kernel 名
3. `ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters -k "regex:<NAME>" -c 1 -o .rlcr/current/profiles/baseline`
4. `ncu --import .rlcr/current/profiles/baseline.ncu-rep --page details > .rlcr/current/profiles/baseline-details.txt`
5. `python bench/benchmark.py --device cuda:0`

---

## Step 4: 分析 Baseline + 设计 Kernel 架构

**查询 KernelWiki**：用 `python3 external/KernelWiki/scripts/query.py` 搜索相关
kernel 案例和优化技术。

读取 NCU 数据，分析：
- Primary bound（compute/memory/latency/barrier）— 引用具体 NCU metrics
- Baseline 做得好的地方（学习）
- Baseline 做得差的地方（机会）
- 资源利用：registers、smem、occupancy、TC utilization

写 `.rlcr/current/baseline-analysis.md`。

设计新 kernel 架构：
- Tile sizes、CTA shape、warp layout
- Pipeline structure、async loading
- Shared memory layout
- Key PTX 指令
- Module decomposition（`// MODULE: <id>` 标记）
- Performance ceiling 推导

写 `.rlcr/current/kernel-architecture.md` 和 `.rlcr/current/direction.md`。

git commit。

---

## Step 5: 实现完整 Kernel

### 代码约束

- CUDA C++ only — no Triton, no CuTe DSL
- Raw PTX inline assembly（TMA、WGMMA/UMMA、mbarrier、fence）
- 薄封装（一个 inline function = 一条 PTX 指令，DeepGEMM 风格）
- **FORBIDDEN**:
  - `#include "cutlass/*.h"` 或 `#include "cute/*.hpp"`（`cutlass/numeric_types.h` 除外）
  - `cutlass::gemm::collective::CollectiveBuilder`
  - `cutlass::gemm::kernel::GemmUniversal`
  - `cutlass::gemm::device::GemmUniversalAdapter`
  - `cutlass::epilogue::collective::CollectiveBuilder`
  - `using namespace cute`
  - Any CuTe layout algebra

### 实现

1. 按 `direction.md` 从头实现完整 CUDA kernel，写在 `solution/`
2. 插入 `// MODULE: <id> BEGIN/END` 标记
3. 写 benchmark adapter
4. `python bench/correctness.py` — 全部通过
5. `python bench/benchmark.py` — 记录结果
6. git commit: "initial kernel implementation"
7. 写 `.rlcr/current/initial-implementation-summary.md`

---

## Step 6: Profile 新 Kernel + 模块分解

1. 写 `.rlcr/current/profiles/ncu_candidate_runner.py`
2. NCU profile → `.rlcr/current/profiles/initial`
3. 导出 details、cubin、SASS、PTX、res-usage
4. 对比 initial vs baseline NCU 数据
5. 验证 MODULE 标记，用 NCU source-level 数据算每模块 runtime fraction
6. 写 `.rlcr/current/decomposition.md`
7. Gap analysis → 每模块瓶颈定位
8. 全局优化策略 → 写 `.rlcr/current/global-strategy.md`
9. 写第一个模块的 `.rlcr/current/modules/<id>/round-0-direction.md`
10. 更新 `module-tracker.json`，git commit

---

## Step 7: Module Loop — RLCR 迭代

按 suggestedOrder 对每个模块循环，每模块最多 15 轮。
每轮你直接依次完成三件事：

### 7a. 实现优化

1. 读 `modules/<id>/round-N-direction.md`
2. 如有上轮 P0/P1 issues 先修复
3. 聚焦 `// MODULE: <id>` 范围，允许必要的外部最小改动
4. `python bench/correctness.py` + `python bench/benchmark.py`
5. git commit: "<id> round N: <描述>"
6. 写 `modules/<id>/round-N-summary.md`

### 7b. Profile

1. `mkdir -p .rlcr/current/profiles/<id>-rN`
2. NCU `--set full` → `profiles/<id>-rN/candidate`
3. 导出 details
4. SASS dump: cubin、res-usage、SASS

### 7c. 分析 + 下一轮方向

1. **Scope check** — 是否有超范围改动
2. **Theory vs Actual** — 对比 direction.md 预测值 vs 实际
   - |gap| < 20% → aligned
   - 否则归因：implementation gap 还是 theory error
3. **PTX/SASS deep analysis** — register delta、spill、instruction count
4. **NCU metrics** — throughput、stalls、TC utilization（vs 上轮 + vs baseline）
5. **Strategy trajectory** — 是否偏离 roadmap（>10% → 修正策略）
6. 写 `modules/<id>/round-N-analysis.md`
7. 按 verdict 决定下一步：

| Verdict | 动作 |
|---|---|
| **CONTINUE** | 写 `round-(N+1)-direction.md`，继续下一轮 |
| **MODULE_COMPLETE** | 结束此模块，进入 Integration |
| **MODULE_STALLED** × 5 | 跳到下一模块 |
| **STRATEGY_REVISION_NEEDED** | 重新分析瓶颈，更新 `global-strategy.md` 和模块顺序 |

### 停止条件

- **roofline efficiency ≥ 90%** → 全部结束
- 单模块连续 5 轮无进展 → 跳到下一模块
- 所有模块完成 → 进入 Finalize

---

## Step 8: Integration（每个模块完成后）

1. NCU full kernel profile
2. 对比 baseline 整体性能
3. 如有 regression → NCU + SASS 诊断，写 `regression-analysis.md`
4. 更新 `module-tracker.json`、`goal-tracker.md`
5. 写下一个模块的 `round-0-direction.md`
6. git commit

---

## Step 9: Finalize

1. 写 `docs/results.md`：
   - Per-module contribution breakdown
   - Theory accuracy summary
   - Final per-shape performance, geomean speedup
   - GPU info, roofline summary
2. 写 `.rlcr/current/complete-summary.md`
3. 更新 `.rlcr/current/state.md`
4. git commit

---

## Step 10: 报告结果

向用户报告 `docs/results.md` 中的最终性能数据。
