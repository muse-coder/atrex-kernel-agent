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
