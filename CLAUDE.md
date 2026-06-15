# IterKernel — GPU Kernel Optimization Agent

GPU kernel 优化项目。使用 `/optimize-kernel <kernel描述>` 启动优化流程。

## 实现约束

优化后的 kernel 必须是 **CUDA C++**，底层使用 PTX inline assembly 直接操控
硬件（TMA、WGMMA/UMMA、mbarrier、fence 等）。优先使用 DeepGEMM 风格的薄封装
（一个 inline 函数对应一条 PTX 指令），尽量不用 CUTLASS/CuTe 等多层模板抽象。
但如果从零实现某些功能确实过于复杂（如复杂的 epilogue fusion、多阶段 pipeline
编排等），允许选择性使用 CUTLASS/CuTe 的部分模板来简化实现，但需在
`docs/draft.md` 中记录使用理由。

## 硬性要求（最高优先级，不可违反）

1. **候选 kernel 必须从头设计并实现（design & implement from scratch）。**
   - 任务的核心交付物是"我自己从零写的 kernel"。**严禁**把任何已有实现
     （上一个 campaign 的 kernel、库 kernel、网上抄来的 kernel）当作起点去
     "继续迭代 / 修补"。即使存在一个一模一样 shape 的旧 campaign,也**不得**
     在它的 kernel 上接着改——必须新开一个空文件,从 PTX 薄封装、warp 角色
     划分、主循环、epilogue 全部自己重新设计实现。
   - 旧 campaign / 库实现只能作为**学习与对比参考**（看它的 NCU、看它的 SASS、
     借鉴架构思路），不能作为代码起点。
   - **不要把精力花在 benchmark harness 的开销/公平性修补上当作"优化"**——
     优化指的是 kernel 本身的架构与指令级工作。harness 问题只在影响正确对比
     时顺手修正,不是任务目标。

2. **Baseline 必须对标"当前最强的现成实现"。**
   - baseline = 该算子在本 GPU 上**最快的现成库实现**,通常在 PyTorch
     （cuBLAS / `torch._scaled_mm` 等）与 **FlashInfer 库** 之间取**实测更快**
     的那个。两个都要测,选快的当 baseline,并在 `docs/baseline_source.md`
     记录实测对比与选择理由。
   - 绝不拿一个弱 baseline 来"虚假取胜"。打赢的是当前 SOTA 现成实现。

3. **性能必须用 NCU 实测作准（authoritative）。**
   - baseline vs candidate 的快慢、每轮的进退、最终选最优版,**一律以 NCU 实测
     的 kernel duration（`gpu__time_duration` / kernel time）为准**。
   - `bench/benchmark.py` 的 wall-clock 仅作辅助参考,不能作为性能结论的唯一
     依据（含 Python dispatch / 包装层开销,会掩盖 kernel 真实表现）。
   - baseline 与 candidate 用**同一套 ncu 命令、同一块空闲 GPU** profile,引用
     具体 metric 数值得出结论。ncu 命令遵循 `external/ncu-report-skill/SKILL.md`。

## 规则文档（优化过程中必须遵守）

- `docs/kernel_optimization_rules.md` — 优化护栏
- `docs/benchmark_contract.md` — benchmark 方法论
- `docs/correctness_contract.md` — 正确性要求
- `docs/kernel_optimization_lessons.md` — 历史经验教训（fragment layout、swizzle trade-off、调试策略等）

## Baseline 选择（对标当前最强现成实现）

Baseline 必须是该算子在目标 GPU 上**最快的现成库实现**。流程：
1. 测 PyTorch 路径（cuBLAS，如 `torch._scaled_mm` / `torch.mm`）。
2. 测 **FlashInfer 库**路径（AOT 预编译,`FLASHINFER_ENABLE_AOT=1`）。
3. 取两者中**实测更快**的作为 baseline；在 `docs/baseline_source.md` 记录
   两者的实测延迟、版本/commit、入口函数、选择理由。

baseline 与 candidate 必须用对称的 ABI 与计时方式（均 destination-passing,
无单边多余开销）。不得用弱 baseline 取巧。

## 知识来源（如果存在则使用）

- `external/KernelWiki/SKILL.md` — Blackwell/Hopper kernel 优化知识库
- `external/ncu-report-skill/SKILL.md` — Nsight Compute profiling 方法论
- `external/CudaSkill/cuda_skill/references/ptx-isa.md` — PTX ISA 文档搜索入口
  （完整文档在 `external/CudaSkill/cuda_skill/references/ptx-docs/`）。**分析
  PTX/SASS 或修改 PTX 指令时,必须查阅:指令语义、操作数约束、fragment
  layout、Target ISA Notes(SM 版本支持)**

## Kernel 代码仓库隔离

**每个 campaign 使用独立的 git 仓库管理 kernel 代码，不在本 agent 仓库中提交。**

- 独立 repo 位置：`/tmp/<campaign_slug>/`（如 `/tmp/rtxpro5000_fp8_gemm__m1024/`）
- 本仓库 `campaigns/operators/<slug>/` 只保留目录结构（`.gitkeep`），不含实际 kernel 代码
- 所有 kernel 代码修改、优化迭代的 commit 在独立 repo 中进行
- benchmark 实验结果（`results.jsonl`）不提交到本仓库

## 任务目录结构

独立 repo 内的目录结构：

```
prompt.md       — 任务卡
config.toml     — build/benchmark 配置
baseline/       — 参考实现（对称 ABI）
solution/       — 优化后的 kernel
bench/          — benchmark + correctness harness
docs/           — 结果、方法笔记
.rlcr/          — RLCR 循环状态
```

## RLCR 每轮硬约束（不可跳过）

每一轮优化必须按顺序完成以下步骤，**缺任何一步不得进入下一轮**：

轮次用**全局递增编号 N**，每轮全部产物放进一个目录 `.rlcr/current/rounds/r<N>/`
（本地，不 commit）：

1. **实现前**：读 `rounds/r<N>/direction.md`（首轮需先写）
2. **实现后**：`git diff` 检查 MODULE 边界 + 防重写机械检查
3. **验证**：correctness 通过 + benchmark 记录
4. **Profile**：NCU 实测 + PTX/SASS 静态分析（5 类静态产物每轮必生成），保存到 `rounds/r<N>/`
5. **文档**：写 `rounds/r<N>/summary.md`（diff 统计 + 因果关系）
6. **分析**：写 `rounds/r<N>/analysis.md`（NCU 数值 + SASS 证据 + verdict）
7. **方向**：写 `rounds/r<N+1>/direction.md`（或标记模块完成）
8. **提交**：git commit（**只提交 `solution/` 代码**；`.rlcr/` 含每轮目录不进 git）

**即使 context 被压缩、即使性能已达标、即使"看起来不需要"，都不能省略。
这是硬约束，不是建议。违反时立即停下补齐再继续。**

## 约定

- benchmark 模板：`docs/benchmark_template.py` → 复制到 `bench/benchmark.py`
- 每次有意义的变更后在独立 repo 中 commit
- benchmark/profile 前后检查 GPU 状态
- 不伪造任何 evidence
