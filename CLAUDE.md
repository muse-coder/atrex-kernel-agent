# IterKernel — GPU Kernel Optimization Agent

GPU kernel 优化项目。使用 `/optimize-kernel <kernel描述>` 启动优化流程。

## 实现约束

优化后的 kernel 必须是 **CUDA C++**，底层使用 PTX inline assembly 直接操控
硬件（TMA、WGMMA/UMMA、mbarrier、fence 等）。优先使用 DeepGEMM 风格的薄封装
（一个 inline 函数对应一条 PTX 指令），尽量不用 CUTLASS/CuTe 等多层模板抽象。
但如果从零实现某些功能确实过于复杂（如复杂的 epilogue fusion、多阶段 pipeline
编排等），允许选择性使用 CUTLASS/CuTe 的部分模板来简化实现，但需在
`docs/draft.md` 中记录使用理由。Baseline 可以是任何实现（FlashInfer、
CUTLASS 等）。

## 规则文档（优化过程中必须遵守）

- `docs/kernel_optimization_rules.md` — 优化护栏
- `docs/benchmark_contract.md` — benchmark 方法论
- `docs/correctness_contract.md` — 正确性要求
- `docs/kernel_optimization_lessons.md` — 历史经验教训（fragment layout、swizzle trade-off、调试策略等）

## 默认 Baseline

没有指定 baseline 时，默认用 **FlashInfer AOT kernel**。FlashInfer 必须
以 AOT 预编译版本安装（`FLASHINFER_ENABLE_AOT=1`）。

## 知识来源（如果存在则使用）

- `external/KernelWiki/SKILL.md` — Blackwell/Hopper kernel 优化知识库
- `external/ncu-report-skill/SKILL.md` — Nsight Compute profiling 方法论

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

1. **实现前**：读 `modules/<id>/round-N-direction.md`（首轮需先写）
2. **实现后**：`git diff` 检查 MODULE 边界 + 防重写机械检查
3. **验证**：correctness 通过 + benchmark 记录
4. **Profile**：NCU 实测 + PTX/SASS 静态分析，保存到 `profiles/<id>-rN/`
5. **文档**：写 `modules/<id>/round-N-summary.md`（diff 统计 + 因果关系）
6. **分析**：写 `modules/<id>/round-N-analysis.md`（NCU 数值 + SASS 证据 + verdict）
7. **方向**：写 `modules/<id>/round-(N+1)-direction.md`（或标记模块完成）
8. **提交**：git commit

**即使 context 被压缩、即使性能已达标、即使"看起来不需要"，都不能省略。
这是硬约束，不是建议。违反时立即停下补齐再继续。**

## 约定

- benchmark 模板：`docs/benchmark_template.py` → 复制到 `bench/benchmark.py`
- 每次有意义的变更后在独立 repo 中 commit
- benchmark/profile 前后检查 GPU 状态
- 不伪造任何 evidence
