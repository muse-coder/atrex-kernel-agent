# IterKernel — GPU Kernel Optimization Agent

GPU kernel 优化项目。使用 `/optimize-kernel <kernel描述>` 启动优化流程。

## 多 agent 架构（master + analysis + code1 + code2）

> 摘要。**权威单一来源是 `.claude/commands/optimize-kernel.md`「## 多 agent 编排」节**；
> 各角色契约在 `.claude/agents/{analysis,code-impl,code-iter}.md`。

`/optimize-kernel` 跑的会话 agent 是 **master（战略层 orchestrator）**：它**不亲手写
kernel、不读原始 SASS dump**，而是用 **Agent 工具** spawn 三类 subagent，靠**磁盘
artifact 交接（不在 prompt 里塞尝试历史）**。**不用 Workflow**——这是串行+自适应判断+
交互式 master 的循环，是 Agent 工具的正命。

- **master**：定/重定优化总纲、扣「丢弃当前总纲」扳机（按 ceiling 判据 + pathology
  checklist，双向）、守 90% roofline 目标、持 `architecture-ledger.md`、commit/finalize。
  **唯一 spawner**（扁平树，不嵌套 spawn）。
- **analysis**（`subagent_type: analysis`）：clean-context 诊断，读 NCU/SASS/PTX/源码，
  按**绝对 roofline** 判 verdict，写 `analysis.md`+`summary.md`+下轮 `direction.md`，
  **recommend**（不 decide）枯竭。只读代码。
- **code1 = code-impl**：从头实现（Write 新文件）；**code2 = code-iter**：渐进优化
  （**无 Write 工具**，靠 `cp` 上一版 + `Edit` 一处，从工具层无法凭空重写）。两者各自跑
  correctness + 5 类 SASS 产物 + NCU。
- **源文件版本命名**：每个产代码的轮 N 都写成独立版本文件 `solution/<family>_v<N>.<ext>`
  （`v` 数字 = 全局轮号，round 1 = 初始 v1；code2 轮 N = `cp v<N-1>` + 一 lever）。全部
  v1..vn 留存，便于跨轮 diff 与 finalize 按 NCU 选最优 vN。权威规则见 optimize-kernel.md
  「源文件版本命名」。

**双循环**：内循环 `code2↔analysis`（master 不插手，反应式）；analysis 报枯竭才上浮到
外循环由 master 判丢弃/re-arch。现有所有硬纪律（FROM SCRATCH / 防重写 / NCU 权威 /
完整产物门槛 / 90% roofline）不变，只是分摊到各角色；hook 按 cwd+file_path 校验，子 agent
照常受约束。

## 实现约束（原语三选一，按算子复杂度评估后定）

candidate **不固定 CUDA**。master 在 Step 4d-0 评估算子复杂度 + 达 ≥90% roofline 所需
抽象层级后，从三者选一（Triton 不在候选集）：

1. **纯 CUDA + PTX 薄封装**（DeepGEMM 风格，一函数=一条 PTX）—— 结构简单~中等、或瓶颈
   在需要手工调度的指令级控制、或不贴标准模板时。此路径内**禁止**混入 CUTLASS
   Collective/Builder/GemmUniversal*/CuTe layout 代数（`cutlass/numeric_types.h` 仅作
   dtype 定义例外）。
2. **CUTLASS**（C++ 模板，Collective/Builder/epilogue fusion/CuTe layout 代数均**允许**）
   —— 标准/近标准 GEMM/conv/attention，手写多阶段 pipeline 成本过高时。
3. **CuTe DSL**（Python）—— 要 CUTLASS 级抽象但需更灵活 fusion/更快迭代、裸 PTX 不现实时。

无论选哪个：**FROM SCRATCH**（自己用所选原语搭，不复制任何已有实现——CUTLASS 路径=
自己用其构件组装而非抄现成 kernel）、每轮 5 类 SASS 产物+NCU、渐进 Edit/一轮一 lever、
退化不回退、≥90% roofline。所选原语 + build/profile 命令写进 `config.toml`。**权威清单
以 `.claude/commands/optimize-kernel.md` Step 4d-0 + Step 5「代码约束」为准**（冲突时以该
命令文档为准）。此约束只针对从零实现的 candidate；baseline 可用任意现成库实现。

## 硬性要求（最高优先级，不可违反）

> 以下为**摘要**。完整正文与边界以 `.claude/commands/optimize-kernel.md`
> 「全局铁律」为**权威单一来源**；运行时由
> `.claude/hooks/inject_hard_requirements.py` 在每次 SessionStart/压缩后重新
> 注入（抗 context 压缩）。**改这些规则时三处需同步：本节 + 全局铁律 + inject hook。**

1. **从头实现**：candidate 必须从零设计实现，**严禁**以任何已有实现（旧 campaign /
   库 / 抄来的 kernel）为代码起点；旧实现只作学习/对比参考（读其 NCU/SASS）。不要
   把 benchmark harness 的开销/公平性修补当成"优化"——优化 = kernel 架构与指令级工作。
2. **最强 baseline**：baseline = 现成库中**实测最快**者（PyTorch cuBLAS 与 FlashInfer
   AOT 都测、取快者），记录于 `docs/baseline_source.md`；绝不打弱 baseline。
3. **性能以 NCU 为准**：快慢 / 进退 / 选最优一律看 NCU kernel duration
   (`gpu__time_duration`)，baseline 与 candidate 用同一套 ncu 命令、同一块空闲 GPU；
   `bench/benchmark.py` wall-clock 仅作辅助。
4. **完成判据 = 达到该 shape 的 ≥90% roofline 上限**（compute-bound 即 90% spec
   峰值；memory-bound 取 90% memory roofline，可能低于 spec 峰值）；baseline 只是
   必须超过的下限参照，**"打平 baseline" ≠ 完成**。

## 规则文档（优化过程中必须遵守）

- `docs/kernel_optimization_rules.md` — 优化护栏
- `docs/benchmark_contract.md` — benchmark 方法论
- `docs/correctness_contract.md` — 正确性要求
- `docs/kernel_optimization_lessons.md` — 历史经验教训（fragment layout、swizzle trade-off、调试策略等）

## Baseline 选择（对标当前最强现成实现）

摘要：测 PyTorch（cuBLAS，如 `torch._scaled_mm`）与 FlashInfer（AOT 预编译，
`FLASHINFER_ENABLE_AOT=1`）两条路径，取**实测更快**者为 baseline，在
`docs/baseline_source.md` 记录延迟 / 版本 / 入口 / 选择理由；baseline 与 candidate
对称 ABI 与计时（destination-passing，无单边开销）。完整流程见
`.claude/commands/optimize-kernel.md` Step 1。

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

> 这是 `.claude/commands/optimize-kernel.md` **Step 7（7a/7b/7c）的检查清单镜像**，
> 完整定义以 Step 7 为权威；改流程时两处同步。

每一轮优化必须按顺序完成以下步骤，**缺任何一步不得进入下一轮**。轮次用**全局递增
编号 N**，每轮全部产物放进 `.rlcr/current/rounds/r<N>/`（本地，不 commit）：

1. **实现前（7a）**：读 `rounds/r<N>/direction.md`（首轮需先写）；并把
   `.rlcr/current/state.md` 设为 `当前轮: r<N>`（完整产物门槛 hook 据此判定上一轮）
2. **实现后（7a）**：`git diff` 检查 MODULE 边界 + 防重写机械检查
3. **验证（7a）**：`benchmark.py --correctness-only` 通过 + benchmark 记录（粗筛）
4. **Profile（7b）**：NCU 实测 + PTX/SASS 静态分析（5 类静态产物每轮必生成），存 `rounds/r<N>/`
5. **文档（7c）**：analysis 从版本间 diff / git diff 重建并写 `rounds/r<N>/summary.md`
   （diff 统计 + 因果关系）
6. **分析（7c）**：写 `rounds/r<N>/analysis.md`（NCU 数值 + SASS 证据 + verdict；进退一律以 NCU 判）
7. **状态（7c）**：更新 `state.md`（当前轮号 / verdict / 最新 NCU duration / 下一步 direction）——抗压缩恢复依赖它
8. **方向（7c）**：写 `rounds/r<N+1>/direction.md`（或标记模块完成）
9. **提交**：git commit（**只提交 `solution/` 代码**；`.rlcr/` 含每轮目录不进 git）

**即使 context 被压缩、即使性能已达标、即使"看起来不需要"，都不能省略。
这是硬约束，不是建议。违反时立即停下补齐再继续。**

## 约定

- benchmark 模板：`docs/benchmark_template.py` → 复制到 `bench/benchmark.py`
- 每次有意义的变更后在独立 repo 中 commit
- benchmark/profile 前后检查 GPU 状态
- 不伪造任何 evidence
