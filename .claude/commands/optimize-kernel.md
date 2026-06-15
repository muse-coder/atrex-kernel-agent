# Optimize Kernel

用户请求：$ARGUMENTS

你是 GPU kernel 优化工程师。你将在这个对话中独立完成整个优化流程——
实现、profiling、分析、迭代，全部由你直接执行，不调 Workflow，不 spawn agent。

### 全局铁律

0. **设计即上限**：第一步（Step 4d）的架构设计就要做到优化上限——直接采用打赢
   目标 baseline 所必需的全部核心技术，并通过 4d-ceiling 的结构上限门槛
   （结构上限 ≥ 目标效率）才能进入实现。不允许"先简单版再渐进爬"。
1. **渐进式修改**：Step 5 首次实现后，**在同一架构内**的所有后续修改（Step 7+）
   必须是增量 Edit，**严禁用 Write 覆盖 solution/ 下的任何文件**，每轮只改一个
   优化点。**边界**：此约束管的是"既定架构内的迭代纪律"，**不**约束"换架构"
   ——当分析表明架构本身赢不了时，必须 STRATEGY_REVISION→重新设计→从头实现新
   架构（合法，见 Step 4d 澄清与 Step 5）。
2. **每次改动后 `git diff`**：确认改动范围与目标一致，非目标区域未被修改。
3. **Regression guard**：每轮 benchmark 后对比上轮整体性能，退化 >5% 则停下分析，不继续下一轮。
4. **不丢失历史**：每轮 commit 保留完整 git 历史，禁止 amend/rebase/force-push。

### 错误恢复流程（编译失败 / 精度错误时）

当增量 Edit 导致编译失败或精度不通过时，**严格按以下顺序处理**：

1. **读编译错误 / 精度 diff**，定位具体出错的行
2. **用 Edit 做针对性修复**（只改报错相关的行），不要扩大修改范围
3. 重新编译 / 跑精度测试
4. 如果连续 3 次 Edit 修复仍然失败：
   - `git diff HEAD -- solution/` 检查累积改动量
   - 如果累积改动已经偏离太远，**`git checkout HEAD -- solution/` 回退到上次 commit 的状态**
   - 重新读 direction.md，缩小优化目标，用更小的改动重试
5. **绝对禁止的行为**：
   - ❌ 编译不过 → Write 重写整个文件
   - ❌ 精度不对 → 把整个 kernel 函数重写
   - ❌ 连续报错 → 放弃 MODULE 边界，大范围改动
   - ✅ 正确做法永远是：小 Edit → 编译 → 验证 → 再小 Edit

### 防重写机械检查

每次对 `solution/` 做完修改后，**必须执行**：

```bash
# 1. 查看改动是否在目标 MODULE 边界内
#    提取目标模块的行范围，检查 diff 的行号是否落在范围内
git diff HEAD -- solution/ | grep "^@@" 
#    对比 MODULE: <id> BEGIN/END 的行号

# 2. 检查非目标区域是否被修改
#    对 solution/ 下每个文件，检查 MODULE 标记外的代码是否有 diff
git diff HEAD -- solution/ | head -100
```

检查标准（看改动的因果关系，不是看行数）：
- **目标 MODULE 内的改动** → 正常，无论多少行
- **MODULE 外有改动且能说明因果链**（本模块改了 X → 导致 Y 接口/布局/参数必须跟着变）→ 正常，在 summary 中记录因果关系
- **MODULE 外有改动但说不清因果关系** → **立即停下**，`git checkout HEAD -- solution/` 回退，重新规划
- **整个函数或文件被删除重写**（diff 显示大段连续 `-` 后跟大段连续 `+`） → **立即停下**，这是重写不是渐进修改。回退到上次 commit

---

## 前置：读取规则和知识

在做任何事情之前，必须读取以下文件：

1. `docs/kernel_optimization_rules.md` — 优化护栏
2. `docs/benchmark_contract.md` — benchmark 方法论
3. `docs/correctness_contract.md` — 正确性要求
4. `docs/kernel_optimization_lessons.md` — **历史经验教训，包含 fragment layout、swizzle trade-off、SM 架构能力、调试策略等，必须在实现前阅读以避免重复踩坑**
5. `external/ncu-report-skill/SKILL.md` — **NCU profiling 方法论，所有 ncu 命令必须遵循**
6. `external/KernelWiki/SKILL.md` — **Blackwell/Hopper kernel 优化知识库，架构设计和瓶颈诊断必须查询**
7. `external/CudaSkill/cuda_skill/references/ptx-isa.md` — **PTX ISA 文档搜索入口**
   （完整文档 `external/CudaSkill/cuda_skill/references/ptx-docs/`）。无需通读，
   但**分析 PTX/SASS 和设计/修改 PTX 指令时必须按需查阅**（见 Step 4b/4c、7c）

如果 5 或 6 不存在，报告错误并停止。若 7 不存在（子模块未初始化），
回退到在线 PTX ISA 文档查询。

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

## Step 2: 创建任务目录（独立 git 仓库）

每个 campaign 使用独立 git 仓库，不在 agent 仓库中提交 kernel 代码。

```bash
TASK_SLUG="<gpu>_<kernel>__<shape_desc>"

# 1. 在 /tmp/ 下创建独立 repo
mkdir -p /tmp/$TASK_SLUG
cp -r templates/example_task/. /tmp/$TASK_SLUG/   # 末尾的 . 连 .gitignore 一起拷
cd /tmp/$TASK_SLUG
git init   # .gitignore 已就位：.rlcr/ 等过程产物本地保留、不提交

# 2. 在 agent 仓库保留空目录结构（仅 .gitkeep）
mkdir -p $AGENT_REPO/campaigns/operators/$TASK_SLUG/{baseline,bench,solution,docs}
touch $AGENT_REPO/campaigns/operators/$TASK_SLUG/{baseline,bench,solution,docs}/.gitkeep
```

在独立 repo 中填写 `prompt.md` 和 `config.toml`。

创建 RLCR 状态目录：

```bash
mkdir -p .rlcr/current/rounds .rlcr/current/profiles
```

创建以下文件：
- `.rlcr/current/plan.md` — 优化计划
- `.rlcr/current/goal-tracker.md` — 目标追踪
- `.rlcr/current/module-tracker.json` — `{ "modules": [], "completedModules": [] }`
- `.rlcr/current/state.md` — 当前阶段

在独立 repo 中 git commit。后续所有 kernel 代码修改、benchmark 结果都在此 repo 中提交。

---

## Step 3: Profile Baseline

### 3a. NCU 实测数据

1. 写 `.rlcr/current/profiles/ncu_baseline_runner.py`
2. `ncu --print-summary per-kernel -c 1` 发现 kernel 名
3. `ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters -k "regex:<NAME>" -c 1 -o .rlcr/current/profiles/baseline`
4. `ncu --import .rlcr/current/profiles/baseline.ncu-rep --page details > .rlcr/current/profiles/baseline-details.txt`
5. `ncu --import .rlcr/current/profiles/baseline.ncu-rep --csv > .rlcr/current/profiles/baseline-metrics.csv`
6. `python bench/benchmark.py --device cuda:0`

### 3b. 静态代码分析（PTX / SASS / 汇编）

读 `config.toml` 获取 arch（如 sm_120）。找到 baseline kernel 源文件。

```bash
# PTX 中间表示
nvcc -ptx -lineinfo -arch=sm_<ARCH> <source.cu> -o .rlcr/current/profiles/baseline.ptx

# Cubin 二进制
nvcc -cubin -lineinfo -arch=sm_<ARCH> <source.cu> -o .rlcr/current/profiles/baseline.cubin

# SASS 反汇编（GPU 原生指令）
cuobjdump -sass .rlcr/current/profiles/baseline.cubin > .rlcr/current/profiles/baseline-sass.txt

# 寄存器 / shared memory 资源使用
cuobjdump -res-usage .rlcr/current/profiles/baseline.cubin > .rlcr/current/profiles/baseline-res-usage.txt

# 详细反汇编（含控制流、predicate、barrier 信息）
nvdisasm -gi -sf .rlcr/current/profiles/baseline.cubin > .rlcr/current/profiles/baseline-nvdisasm.txt
```

如果 baseline 是 library kernel（FlashInfer/CUTLASS 预编译），无法从源码编译时，
用 `cuobjdump -sass` 直接从 .so 中提取 SASS：
```bash
cuobjdump -sass -fun <kernel_name> <library.so> > .rlcr/current/profiles/baseline-sass.txt
```

---

## Step 4: 分析 Baseline + 设计 Kernel 架构

**查询 KernelWiki**：用 `python3 external/KernelWiki/scripts/query.py` 搜索相关
kernel 案例和优化技术。

### 4a. NCU 实测分析

读取 NCU 数据（baseline-details.txt、baseline-metrics.csv），分析：
- Primary bound（compute/memory/latency/barrier）— 引用具体 NCU metric 值
- SM throughput、DRAM bandwidth、L2/L1 hit rates
- Tensor Core utilization（`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`）
- Warp stall 分布（`smsp__pcsamp_warps_issue_stalled_*`）
- Achieved occupancy vs theoretical occupancy
- Pipeline overlap 质量（compute/memory overlap ratio）

### 4b. PTX/SASS 静态分析

读取 baseline-ptx、baseline-sass.txt、baseline-res-usage.txt、baseline-nvdisasm.txt，分析：
- **寄存器压力**：每线程寄存器数、是否有 spill（SASS 中的 `STL`/`LDL` 指令 = local memory spill）
- **指令统计**：总 SASS 指令数、指令类型分布（compute/memory/control/sync）
- **Tensor Core 指令**：HMMA/UMMA/WGMMA 指令出现频率和位置
- **内存访问模式**：LDG/STG 指令的 cache 修饰符（`.L1`/`.L2`/`.CONSTANT`）、向量化宽度（`.64`/`.128`）
- **Bank conflict 风险**：LDS/STS 指令的 stride 模式
- **同步指令**：`BAR.SYNC`、`MEMBAR`、`mbarrier` 相关指令的位置和频率
- **Dual-issue 机会**：连续独立指令的 scheduling 质量
- **循环结构**：PTX 中的分支和循环展开程度
- **Predicated execution**：条件执行指令的比例

**查 PTX ISA 文档**：分析中遇到任何语义/约束不确定的指令（MMA fragment
layout、cp.async/TMA、ldmatrix、mbarrier、cache 修饰符、指令的 SM 版本支持等），
必须到 `external/CudaSkill/cuda_skill/references/ptx-docs/`（入口
`ptx-isa.md`）查清楚，不靠猜测或 CuTe/CUTLASS 抽象推断（见
`docs/kernel_optimization_lessons.md` §1、§2）。把查到的依据写进分析文档。

### 4c. 综合诊断

综合 NCU 实测 + PTX/SASS 静态证据，定位：
- Baseline 做得好的地方（学习）
- Baseline 做得差的地方（改进机会），每条必须有 NCU metric 或 SASS 指令证据

写 `.rlcr/current/baseline-analysis.md`（必须包含具体数值，不允许模糊描述）。

### 4d. 设计 Kernel 架构

> **核心原则：第一步就把设计做到上限（design to the ceiling）。**
> 初始架构必须直接奔着性能上限去——**从第一版就采用打赢目标 baseline 所必需
> 的全部核心技术**（如 warp specialization、ldmatrix、TMA、最优 tile/swizzle、
> 异步流水线等）。**严禁**先设计一个"correctness-first 的简单版"再指望靠 RLCR
> 渐进爬上去：核心架构技术（warp 角色划分、ldmatrix vs 手写 LDS、同步机制）是
> **架构骨架,不是后期能 bolt-on 的 tweak**——简单架构的效率天花板是焊死的,
> 后续每轮只会在那个被焊死的上限里找局部最优,永远赢不了。RLCR 迭代是在一个
> **已经逼近上限**的架构上做精调,不是从 0.45x 往上挪。

基于分析结果设计新 kernel 架构：
- Tile sizes、CTA shape、warp layout（含 warp specialization：producer/consumer 角色划分）
- Pipeline structure、async loading（cp.async / TMA / mbarrier 编排）
- Fragment 加载方式（ldmatrix 优先；手写 LDS 仅在 ldmatrix 不适用且已验证无 bank conflict 时）
- Shared memory layout（含 swizzle/padding 策略避免 bank conflict）
- Key PTX 指令选择（引用 SASS 分析中发现的瓶颈；先做 ldmatrix/mbarrier/TMA 的编译可行性验证）
- Module decomposition（`// MODULE: <id>` 标记）
- 寄存器预算（参考 baseline res-usage 设定目标）

#### 4d-ceiling. 结构上限分析（强制门槛，不可跳过）

在 roofline（硬件算力/带宽下限）之外，**必须额外推导"所选候选架构本身的效率
上限"**，并与 baseline 的实测效率对比：

1. **硬件 roofline**：compute floor / memory floor（标注 compute/memory bound）。
2. **结构上限（structural ceiling）**：**这个具体架构**最多能到峰值的百分之几？
   逐项问：
   - 加载与计算是否被 per-step 全块 barrier 串行化？（→ 上限被 barrier 压低）
   - fragment 取数有无 bank conflict / 是否用了 ldmatrix？
   - occupancy 被什么 cap（寄存器/smem）？能否藏住延迟？
   - 是否复刻了 baseline 达到其效率所用的关键技术？缺哪条、各扣多少效率？
3. **决策门槛**：若 `结构上限 < baseline 实测效率`（或 < roofline 90% 目标），
   则**当前设计注定打不赢——禁止进入 Step 5**。必须回到本步重新设计,补齐
   baseline 的使能技术,直到结构上限 ≥ 目标,再实现。
4. 若判断"打赢 baseline 必须做重写级工作"（如必须 warp-specialized + ldmatrix
   从头实现），**在此处就明确写出来并告知用户**工作量与取舍,而不是先写一版
   注定输的简单实现。

把硬件 roofline + 结构上限 + 决策结论写进 `.rlcr/current/kernel-architecture.md`。

> **关于"渐进式硬约束"的边界（重要澄清）**：全局铁律的"严禁重写"管的是
> **在一个既定架构内迭代时**的纪律（别用 Write 覆盖、别一报错就整文件重写）。
> 它**绝不意味着**"架构选错了也只能将就"。当结构上限分析（4d-ceiling）或 Step 7
> 的迭代证据表明**整体思路/架构本身赢不了**时，正确动作是
> **STRATEGY_REVISION → 重新设计架构 → 从头实现一版新 candidate**（这是合法且
> 必要的，不算违反渐进式约束；见 Step 5 关于重新实现的说明）。

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
7. **创建渐进式修改锁**：`touch .rlcr/current/.initial-impl-done`
   - 此 marker 一旦存在，项目 hook（`.claude/hooks/block_solution_rewrite.py`）
     会强制两条规则（基于磁盘状态，**context 压缩也冲不掉**）：
     - **防重写**：拦截对该 campaign `solution/` 的 Write 覆盖，以及 shell
       重写（`>`/`>>`/`tee`/`sed -i`/`truncate`/`dd` 重定向到 solution/）。
     - **先读方向再改**：迭代期对 locked `solution/` 的 Edit，必须先 Read 当前轮
       的 `rounds/r<N>/direction.md`（最新的那个）才放行；没读会被拦。
     - marker 不存在时两条都不生效（首次实现照常用 Write）。
   - 后续所有 solution/ 改动必须用 Edit；动手前先读本轮 direction（hook 会强制）
8. 写 `.rlcr/current/initial-implementation-summary.md`

### 重新实现（re-architecture，当 4d-ceiling 或 Step 7 判定需换架构时）

这是合法且必要的，不是"被禁止的重写"。re-architecture = 一次**新的 Step 5 首次
实现**，所以它合法地重置迭代锁（防重写 marker 管的是"迭代期"，不是"换架构"）。
流程：
1. 先更新 `.rlcr/current/kernel-architecture.md`：写清"为何旧架构上限不够"
   （引用结构上限分析 + 实测证据）与新架构如何达到目标上限。
2. **显式重置迭代锁**：`rm .rlcr/current/.initial-impl-done`（hook 会拦截对
   locked solution/ 的**任何** Write，包括新文件——所以必须先解锁，这是 marker
   的正常生命周期，不是绕过）。在 commit message / 文档里写明这是 authorized
   re-architecture。
3. 新架构**写成 solution/ 下的新源文件**（如 `kernel_v2.cu`），保留旧实现供
   对比，git 历史完整；不去 Write 覆盖旧文件本身。
4. 把 candidate ABI / adapter 切到新文件；旧文件在新版验证更快后用 `git rm` 删除。
5. 新文件插 MODULE 标记，跑 correctness + benchmark，commit
   "re-architecture: <新架构> (initial)"，**重新 `touch .initial-impl-done`**
   重新上锁，然后对**新文件**进入 Step 6/7 的渐进迭代。

---

## Step 6: Profile 新 Kernel + 模块分解

### 6a. NCU 实测

1. 写 `.rlcr/current/profiles/ncu_candidate_runner.py`
2. NCU profile：
   ```bash
   ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters \
     -k "regex:<NAME>" -c 1 -o .rlcr/current/profiles/initial \
     python .rlcr/current/profiles/ncu_candidate_runner.py
   ncu --import .rlcr/current/profiles/initial.ncu-rep --page details > .rlcr/current/profiles/initial-details.txt
   ncu --import .rlcr/current/profiles/initial.ncu-rep --csv > .rlcr/current/profiles/initial-metrics.csv
   ```

### 6b. 静态代码分析

```bash
nvcc -ptx -lineinfo -arch=sm_<ARCH> <source.cu> -o .rlcr/current/profiles/initial.ptx
nvcc -cubin -lineinfo -arch=sm_<ARCH> <source.cu> -o .rlcr/current/profiles/initial.cubin
cuobjdump -sass .rlcr/current/profiles/initial.cubin > .rlcr/current/profiles/initial-sass.txt
cuobjdump -res-usage .rlcr/current/profiles/initial.cubin > .rlcr/current/profiles/initial-res-usage.txt
nvdisasm -gi -sf .rlcr/current/profiles/initial.cubin > .rlcr/current/profiles/initial-nvdisasm.txt
```

### 6c. 对比分析 + 模块分解

分解前先读 `docs/module_decomposition_guide.md`（分解原则、典型 GEMM/Attention/
Reduction 分解、共享资源识别、优化顺序）。

1. NCU metrics 对比：initial vs baseline（throughput、bandwidth、stalls、TC util）
2. SASS 对比：指令数、寄存器数、spill 数、循环结构差异
3. 验证 MODULE 标记，用 NCU source-level 数据算每模块 runtime fraction
4. 写 `.rlcr/current/decomposition.md`
5. Gap analysis → 每模块瓶颈定位（NCU 证据 + SASS 证据）
6. 全局优化策略 → 写 `.rlcr/current/global-strategy.md`
7. 写第一轮方向 `.rlcr/current/rounds/r1/direction.md`（在文档内标明本轮目标模块）
8. 更新 `module-tracker.json`，git commit（只提交 `solution/` 代码与 `docs/`）

---

## Step 7: Module Loop — RLCR 迭代

按 suggestedOrder 对每个模块循环。**无轮次上限**——只要 roofline 未达 90% 且
仍有可尝试的方向，就继续优化；某模块卡住就转下一个模块或拓宽搜索，仅在目标
达成或用户明确叫停时停止。轮次用**全局递增编号 N**（跨模块连续，不按模块分
目录；目标模块写进文档内容里）。

**每轮一个目录（本地，不 commit）**：本轮全部产物放进 `.rlcr/current/rounds/r<N>/`：
- 文档：`direction.md`、`summary.md`、`analysis.md`
- profile / 静态分析：`candidate.ptx`、`candidate-sass.txt`、`candidate.cubin`、
  `candidate-res-usage.txt`、`candidate-nvdisasm.txt`、`candidate.ncu-rep`、
  `candidate-details.txt`、`candidate-metrics.csv`

`.rlcr/` 整个被 `.gitignore`，所以这些分析记录与数据**只留本地、不进 git**；
**每轮只 commit `solution/` 代码**（交付物 `docs/results.md` 在 Finalize 时提交）。

每轮你直接依次完成三件事：

### 7a. 实现优化（渐进式修改，严禁重写）

**核心约束：每轮只做增量修改，不重写文件**

1. 读 `rounds/r<N>/direction.md`（**必读**：hook 会拦截"未读方向就
   Edit solution/"，没读这一步后面改不动）
2. 如有上轮 P0/P1 issues 先修复
3. **修改前**：
   - `git stash` 或确认工作区干净
   - 定位 `// MODULE: <id> BEGIN` 到 `// MODULE: <id> END` 的行范围
4. **修改时**：
   - **必须使用 Edit 工具**做针对性修改，**禁止用 Write 覆盖整个文件**
   - 主改动在目标 MODULE 内；MODULE 外的改动必须是被主改动**因果驱动**的联动：
     - ✅ 共享 helper 函数签名/实现变更（被本模块调用）
     - ✅ shared memory 总量、launch config（smem_size、grid/block）
     - ✅ 数据流接口适配（上下游模块的读写格式跟着变）
     - ✅ pipeline 编排联动（prologue 改了 stage 数，mainloop/epilogue 的 barrier 跟着调）
     - ✅ 寄存器策略全局调整（occupancy 变化导致）
     - ❌ 与本轮优化目标无因果关系的代码改动
   - 每次 Edit 只改一个逻辑点（一条优化策略），不要一次改多个不相关的地方
5. **修改后验证**：
   - `git diff -- solution/` — 检查所有改动
   - MODULE 内的改动：正常
   - MODULE 外的改动：**每一处都必须在 rounds/r<N>/summary.md 中说明因果关系**（"改了 X 是因为模块内改了 Y，导致 Z 接口不兼容"）。无法说明因果关系的外部改动 → 回退
   - `python bench/correctness.py` — 正确性必须通过
   - `python bench/benchmark.py` — 记录性能
6. **Regression check**：对比本轮 vs 上轮的整体 kernel 性能
   - 如果整体性能下降 > 5% 且不在预期内（direction.md 未预测到），立即 `git diff` 分析原因
   - 如果是其他模块被意外影响，`git checkout -- <affected files>` 回退非目标改动
7. git commit: "r<N> (<id>): <描述>" — **只提交 `solution/` 代码**（`.rlcr/` 不进 git）
8. 写 `rounds/r<N>/summary.md`，其中包含本轮 diff 统计（改了哪些文件、多少行）

### 7b. Profile（NCU 实测 + 静态分析）

本轮所有 profile/静态分析产物都写进本轮目录 `.rlcr/current/rounds/r<N>/`
（本地，不 commit）。**5 类静态产物每轮都必须生成**，不可省略。

```bash
mkdir -p .rlcr/current/rounds/r<N>
RD=.rlcr/current/rounds/r<N>

# NCU 实测
ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters \
  -k "regex:<NAME>" -c 1 -o $RD/candidate \
  python .rlcr/current/profiles/ncu_candidate_runner.py
ncu --import $RD/candidate.ncu-rep --page details > $RD/candidate-details.txt
ncu --import $RD/candidate.ncu-rep --csv > $RD/candidate-metrics.csv

# 静态代码分析（PTX / SASS / 资源占用 / 反汇编 —— 每轮必做）
nvcc -ptx   -lineinfo -arch=sm_<ARCH> <source.cu> -o $RD/candidate.ptx
nvcc -cubin -lineinfo -arch=sm_<ARCH> <source.cu> -o $RD/candidate.cubin
cuobjdump -sass       $RD/candidate.cubin > $RD/candidate-sass.txt
cuobjdump -res-usage  $RD/candidate.cubin > $RD/candidate-res-usage.txt
nvdisasm  -gi -sf     $RD/candidate.cubin > $RD/candidate-nvdisasm.txt
```

### 7c. 分析 + 下一轮方向

1. **Scope check** — 是否有超范围改动

2. **Theory vs Actual** — 对比 direction.md 预测值 vs 实际
   - |gap| < 20% → aligned
   - 否则归因：implementation gap 还是 theory error

3. **NCU 实测对比**（当前 vs 上轮 vs baseline，引用具体数值）：
   - SM throughput（`sm__throughput.avg.pct_of_peak_sustained_elapsed`）
   - DRAM bandwidth（`dram__bytes.sum.per_second`）
   - L2/L1 hit rates
   - Warp stall 分布变化（哪类 stall 增加/减少）
   - Tensor Core utilization
   - Achieved occupancy
   - NCU rules engine（`--page details` 中的 Est. Speedup 建议）

4. **PTX/SASS 静态对比**（当前 vs 上轮）：
   - 寄存器数量 delta（res-usage）
   - Spill 检测：`STL`/`LDL` 指令数量变化
   - 总 SASS 指令数变化
   - 关键指令变化：HMMA/UMMA/WGMMA 数量、LDG/STG 向量化宽度、LDS/STS 访问模式
   - 循环展开程度变化（通过 branch 指令密度判断）
   - Dual-issue 质量：连续独立指令比例
   - 新增/消失的 `BAR.SYNC`、`MEMBAR` 等同步指令
   - nvdisasm 控制流图变化（分支、predicated execution）
   - **查 PTX ISA 文档**：解读指令变化、或本轮 direction 要换用某条 PTX 指令时，
     先到 `external/CudaSkill/.../ptx-docs/` 查其语义、操作数约束、fragment
     layout、Target ISA Notes（SM 支持），再下结论。`rounds/r<N>/analysis.md` 中
     涉及 PTX 指令选择的结论必须引用 ISA 文档章节号作为依据。

5. **Strategy trajectory** — 是否偏离 roadmap（>10% → 修正策略）

6. 写 `rounds/r<N>/analysis.md`（每条结论必须附 NCU metric 值或 SASS 指令证据）
7. 按 verdict 决定下一步：

| Verdict | 动作 |
|---|---|
| **CONTINUE** | 写 `rounds/r<N+1>/direction.md`，继续下一轮 |
| **MODULE_COMPLETE** | 结束此模块，进入 Integration |
| **MODULE_STALLED** | 本模块持续无进展 → 转下一模块（不是停止；之后可回来再试） |
| **STRATEGY_REVISION_NEEDED** | 重新分析瓶颈，更新 `global-strategy.md` 和模块顺序 |

### 停止条件

**无轮次上限。** 只在以下情况结束/转向：

- **roofline efficiency ≥ 90%** → 全部结束
- 某模块持续无进展 → 转去优化下一个模块（不是停止；之后可回来再试）
- 所有模块都暂时无新方向 → **不要轻易判定"到顶"**：拓宽搜索空间（查
  KernelWiki、PTX ISA 文档、公开资料/论文/开源 kernel）找新方向再试
- 所有模块完成且 roofline 达标 → 进入 Finalize
- 否则只要还有可尝试的方向就继续；**仅在目标达成或用户明确叫停时停止**

---

## Step 8: Integration（每个模块完成后）

1. `git diff <module-start-commit>..HEAD -- solution/` — 汇总本模块所有改动范围
2. NCU full kernel profile + SASS 静态分析
3. 对比 baseline 整体性能：
   - 整体 speedup vs baseline
   - **Per-module regression check**：检查每个已完成模块的 source-level NCU metrics，确认之前优化的模块没有退化
4. 如有 regression：
   - NCU + SASS 诊断根因
   - 如果是模块间干扰（如 shared memory 布局冲突、寄存器压力传导），写 `regression-analysis.md` 并修复
   - 修复后重新 benchmark 确认
5. 更新 `module-tracker.json`、`goal-tracker.md`
6. 写下一轮方向 `rounds/r<N+1>/direction.md`（标明下一个目标模块）
7. git commit

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
