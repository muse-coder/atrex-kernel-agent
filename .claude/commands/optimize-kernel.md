# 优化 Kernel

用户请求：$ARGUMENTS

你是 GPU kernel 优化工程师。你将在这个对话中独立完成整个优化流程——
实现、profiling、分析、迭代，全部由你直接执行，不调 Workflow，不 spawn agent。

### 全局铁律

> 本节是硬性要求的**权威单一来源**。CLAUDE.md「硬性要求」是其摘要、
> `.claude/hooks/inject_hard_requirements.py` 是其运行时镜像——**改本节时同步那两处**。

-2. **候选 kernel 必须从头设计并实现（FROM SCRATCH）**：核心交付物是"你自己从零
   写的 kernel"。**严禁**以任何已有实现（上一个 campaign 的 kernel、库 kernel、
   抄来的 kernel）作为代码起点去"继续迭代/修补"。即使发现一个**同 shape、同 GPU
   的旧 campaign**,也绝不在它的 .cu 上接着改——必须新开空文件,把 PTX 薄封装、
   warp 角色划分、主循环、epilogue 全部自己重新设计写出来。旧 campaign / 库实现
   只能作为**学习与对比参考**（读它的 NCU/SASS、借鉴架构思路）,不能当起点。
   一打开任务就开始设计并写新 kernel,不要先去捡现成 kernel 的便宜。
-1. **不要把 harness/开销修补当作"优化"**：优化 = kernel 本身的架构与指令级工作
   （tile/warp-spec/pipeline/swizzle/PTX 指令选择等）。benchmark 包装层的开销、
   公平性问题只在**影响正确对比**时顺手修正,绝不作为优化目标或拿来充当"成果"。
   若发现自己在改 `.py` 包装、`.item()`、`copy_`、计时代码而不是 `.cu` kernel,
   说明跑偏了,立即回到 kernel。
-0.5. **性能必须用 NCU 实测（authoritative）**：判断 baseline 与 candidate 的
   性能、判断每轮快了还是慢了、最终选最优版,**都以 NCU 实测的 kernel 时间
   （`gpu__time_duration` / kernel duration）为准**。`bench/benchmark.py` 的
   wall-clock 只作辅助参考,不得作为性能结论的唯一依据（它含 Python dispatch、
   包装层开销,会掩盖 kernel 真实表现）。每轮 baseline 与 candidate 都要用
   **同一套 NCU 命令**在**同一块空闲 GPU**上 profile,引用具体 metric 数值得出
   "快/慢/持平"。NCU 命令遵循 `external/ncu-report-skill/SKILL.md`。
0. **设计即上限**：第一步（Step 4d）的架构设计就要做到优化上限——直接采用打赢
   目标 baseline 所必需的全部核心技术，并通过 4d-ceiling 的结构上限门槛
   （结构上限 ≥ 目标效率）才能进入实现。不允许"先简单版再渐进爬"。
1. **渐进式修改**：Step 5 首次实现后，**在同一架构内**的所有后续修改（Step 7+）
   必须是增量 Edit，**严禁用 Write 覆盖 solution/ 下的任何文件**，每轮只改一个
   优化点。**边界**：此约束管的是"既定架构内的迭代纪律"，**不**约束"换架构"
   ——当分析表明架构本身赢不了时，必须 STRATEGY_REVISION→重新设计→从头实现新
   架构（合法，见 Step 4d 澄清与 Step 5）。
2. **每次改动后 `git diff`**：确认改动范围与目标一致，非目标区域未被修改。
3. **退化不回退（no performance revert）**：性能退化**不触发回退**。退化只是
   数据——分析原因、记录，然后**继续前进**。允许直接在退化版本上叠加下一步：
   增量优化必然遇到"某一步下降"，但**在那个下降的版本上再改一步往往就对了**
   （局部低谷≠死路；如 ldmatrix-A 单独退化、与 warp-spec 组合就转正）。
   **不做** `git checkout HEAD -- solution/` 式的性能回退。
4. **每轮都 commit，git 历史即安全网**：每一轮（无论快慢）都 commit 保留完整
   历史，禁止 amend/rebase/force-push。因为每轮都在历史里，**任何一轮都能取回**，
   所以根本不需要主动回退。最优交付物在 **Finalize 时**从所有已提交轮次里按
   benchmark 选出（`git checkout <最优 commit> -- solution/`，见 Step 9）。
5. **唯一的例外是正确性/编译失败**：错的代码不能 benchmark，必须修到能跑对
   （见"错误恢复流程"）。这是"修到正确"，不是"性能回退"。
6. **SASS 分析硬门槛（gating，不可跳过）**：每一轮**必须**先完成 5 类静态产物
   （`candidate.ptx/.cubin/candidate-sass.txt/candidate-res-usage.txt/
   candidate-nvdisasm.txt`）+ NCU 实测，并写进 `rounds/r<N>/analysis.md`，
   **才能开始下一轮的 solution/ 代码修改**。这由 hook
   (`block_solution_rewrite.py` 的 SASS GATE) 机械强制：下一轮 Edit solution/ 时，
   若上一轮 `rounds/r<N-1>/candidate-sass.txt` 不存在则被拦截。**所有轮次（含
   re-architecture 里程碑）都必须用 `.rlcr/current/rounds/r<N>/` 目录结构**，
   否则 hook 的 direction/SASS 门槛失效（这正是 v2/v3 里程碑当初绕过 SASS 的原因）。

### 错误恢复流程（编译失败 / 精度错误时）

当增量 Edit 导致编译失败或精度不通过时，**严格按以下顺序处理**：

1. **读编译错误 / 精度 diff**，定位具体出错的行
2. **用 Edit 做针对性修复**（只改报错相关的行），不要扩大修改范围
3. 重新编译 / 跑精度测试
4. 如果连续 3 次 Edit 修复仍然失败（**仅正确性/编译失败**，铁律 #5 的例外）：
   - `git diff HEAD -- solution/` 检查累积改动量
   - 如果累积改动已经偏离太远，`git checkout HEAD -- solution/` 回到上次能跑对的
     commit（这是"修到正确"，因为错的代码无法 benchmark；不是性能回退）
   - 重新读 direction.md，缩小目标，用更小的改动重试
   - 注意：这只适用于编译/精度失败。**性能退化绝不回退**（铁律 #3）。
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
- **目标 GPU 与 arch（必须实测，禁止硬编码/猜测）**：**一律先运行 `nvidia-smi`
  查清楚型号与 compute capability**，即使用户给了型号也要核对：
  ```bash
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
  ```
  把 compute_cap（形如 `12.0`）转成**完整 nvcc arch 串**：去掉点 → `sm_120`，
  再补架构专用后缀 `a` → **`sm_120a`**（TMA / ldmatrix / fp8 mma 等指令需要
  `sm_XXXa` 变体）。把检测到的 `target_gpu` 与 `arch`（如 `sm_120a`）写进
  `config.toml`，**后续所有 `nvcc -arch=<ARCH>` 都直接用这个完整串**（不要再拼
  `sm_` 前缀，避免 `sm_sm_120a`）。绝不沿用模板里的占位/示例值。
- **Workload shapes**：如果没给，根据 kernel 类型生成常见 production shapes
- **Baseline**：对标**当前最强的现成库实现**。实测对比 PyTorch（cuBLAS，如
  `torch._scaled_mm`/`torch.mm`）与 **FlashInfer 库**（AOT 预编译）两条路径,
  取**更快**的那个当 baseline,在 `docs/baseline_source.md` 记录两者实测延迟、
  版本、入口与选择理由。不得用弱 baseline 取巧。baseline 与 candidate 必须对称
  ABI/计时（均 destination-passing,无单边多余开销）。
- **特殊约束**：dtype、精度要求、是否 fused 等
- **完成目标（PRIMARY，必须显式设定并写进 goal-tracker）**：**达到该 shape 的
  roofline 上限的 ≥90%**（roofline efficiency ≥ 90%）。roofline 上限 =
  `min(compute 峰值, 带宽×算术强度)`，**先判这个 shape 是 compute-bound 还是
  memory-bound**（算 AI 与脊点 `峰值/带宽` 比大小）：
  - **compute-bound**（AI > 脊点）→ 上限 = spec 峰值，目标 = 90% spec 峰值。
    例：sm_120 FP8 峰值 516 TF、M=1024/N=10240/K=4096 的 AI≈1283 ≫ 脊点≈384 →
    compute-bound → 目标 ≥464 TF / ≤t_floor/0.9。
  - **memory-bound**（AI < 脊点）→ 上限 = 带宽×AI，**低于 spec 峰值**；目标 =
    90% 的 memory roofline（µs 下限 = 搬运字节数 / 带宽 / 0.9）。**此时绝不能拿
    "90% spec 峰值"当目标——它物理不可达。**
  - 即：「90% spec 峰值」只是 compute-bound 时的特例，通用判据始终是「90% roofline」。
  - **这是完成判据,不是 baseline。** baseline(最强现成库)只是**对比参照**,
    用来判断"赢没赢现成实现",**不是完成线**。
  - **关键:90% roofline 上限常常 > baseline 实测效率**(库实现往往只到上限的
    85-90%)。所以「打平 baseline」**通常不等于**「达到 90% roofline 上限」——
    后者更高,可能要求**超过 baseline**。两个数都要在 goal-tracker 里写清楚
    (目标=90% roofline 上限;参照=baseline),**不要把"打平 baseline"误当成完成**。
  - 若实测发现连 SOTA 库都远低于 90%(如 88%),说明 90% 对该 shape 可能触及
    物理上限:此时如实告知用户"90% 可能不可达、当前 SOTA=X%",由用户决定是否
    放宽目标——但在得到用户确认前,仍以 90% roofline 上限为目标继续推进(wave-quant/
    stream-K、调度等所有杠杆都要试)。

如果 kernel 类型不明确，直接问用户。

---

## Step 2: 创建任务目录（独立 git 仓库）

每个 campaign 使用独立 git 仓库，不在 agent 仓库中提交 kernel 代码。

```bash
TASK_SLUG="<gpu>_<kernel>__<shape_desc>"

# 1. 在 /tmp/ 下创建独立 repo
mkdir -p /tmp/$TASK_SLUG
cd /tmp/$TASK_SLUG
git init   # 先 init

# 2. 只拷"目录骨架 + .gitignore"，**不要**拷 prompt.md / config.toml
#    （这两个文件你马上要从零写成真正内容；如果先 cp 占位符过来，它们就成了
#     "已存在文件"，之后用 Write 覆盖会被 "File has not been read yet" 拦住，
#     白白多一次 Read 仪式。所以这里直接不拷它们，用 Write 新建即可。）
cp $AGENT_REPO/templates/example_task/.gitignore /tmp/$TASK_SLUG/
mkdir -p /tmp/$TASK_SLUG/{baseline,bench,solution,docs}
#    .gitignore 已就位：.rlcr/ 等过程产物本地保留、不提交

# 3. 在 agent 仓库保留空目录结构（仅 .gitkeep）
mkdir -p $AGENT_REPO/campaigns/operators/$TASK_SLUG/{baseline,bench,solution,docs}
touch $AGENT_REPO/campaigns/operators/$TASK_SLUG/{baseline,bench,solution,docs}/.gitkeep
```

用 **Write 新建** `prompt.md` 和 `config.toml`（参考 `templates/example_task/`
里的同名文件作为格式样板，但不要 cp 过来再覆盖）。

> **关于 "File has not been read yet" 报错**：Write 工具对**已存在**的文件会要求
> "先用 Read 工具读过才能覆盖"（`cat` 不算，它只认 Read 工具的调用记录）。所以
> 规则是：**新建文件 → 直接 Write；要改已存在的文件 → 先 Read 再 Write，或干脆用
> Edit 做增量修改。** 本步把 prompt.md/config.toml 留给 Write 新建，正是为了绕开
> 这个无谓的摩擦。

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

读 `config.toml` 获取 arch（完整 nvcc 形式，如 `sm_120a`，由 Step 1 的
`nvidia-smi` 检测写入）。找到 baseline kernel 源文件。下面命令里的 `<ARCH>`
就是这个完整串，直接 `-arch=<ARCH>`（即 `-arch=sm_120a`）。

```bash
# PTX 中间表示
nvcc -ptx -lineinfo -arch=<ARCH> <source.cu> -o .rlcr/current/profiles/baseline.ptx

# Cubin 二进制
nvcc -cubin -lineinfo -arch=<ARCH> <source.cu> -o .rlcr/current/profiles/baseline.cubin

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
> 目标是 **≥90% roofline 上限**（compute-bound 时即 90% spec 峰值；memory-bound
> 时为 90% 的 memory roofline，见 Step 1）（不是"追平 baseline"——目标本就高于
> baseline）。初始架构必须直接奔着这个上限去——**从第一版就采用达到 ≥90% 上限
> 所必需的全部核心技术**（如 warp specialization、ldmatrix、TMA、最优 tile/swizzle、
> 异步流水线、stream-K 消除 wave quantization 等）。**严禁**先设计一个"correctness-first 的简单版"再指望靠 RLCR
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
上限"**，并与 **≥90% roofline 上限这个目标**对比（baseline 只是必须超过的下限
参照,不是目标线）：

1. **硬件 roofline**：先判 compute-bound 还是 memory-bound（算 AI vs 脊点,见
   Step 1），取 `min(compute 峰值, 带宽×AI)` 作为 roofline 上限。算出 **90%
   roofline 上限对应的目标 TF / µs**。
2. **结构上限（structural ceiling）**：**这个具体架构**最多能到 roofline 上限的
   百分之几？逐项问（每条扣多少效率,凑出结构上限百分比）：
   - 加载与计算是否被 per-step 全块 barrier 串行化？（→ 上限被 barrier 压低）
   - fragment 取数有无 bank conflict / 是否用了 ldmatrix？
   - occupancy / 寄存器墙能否藏住 MMA 延迟（运行时 wait）？
   - **wave quantization**：#CTA 能否整除 #SM？凑不齐就要 stream-K/persistent,否则
     尾波损失（如本例 110=2·5·11,而 2^a·5^b 的 tile 永远凑不成 110 倍数 → 必失 ~3%）。
   - 还缺哪些把利用率推到 90% roofline 上限的技术？
3. **决策门槛**：若 `结构上限 < 90% roofline 上限`，则**当前设计达不到目标——禁止
   进入 Step 5**。必须回到本步重新设计,补齐使能技术,直到结构上限 ≥ 90% roofline
   上限,再实现。（同时结构上限必须 > baseline,否则连现成实现都赢不了。）
4. 若判断"达到 90% roofline 上限必须做重写级工作"（如 warp-specialized + ldmatrix
   + stream-K 从头实现），**在此处就明确写出来并告知用户**工作量与取舍。若连 SOTA
   库都远低于 90%(实测得知),说明 90% 可能触及该 shape 物理上限——如实告知用户当前
   SOTA 百分比,由用户决定是否放宽,在确认前仍以 90% roofline 上限为目标。

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

- 仅限 CUDA C++ —— 不用 Triton，不用 CuTe DSL
- 裸 PTX inline assembly（TMA、WGMMA/UMMA、mbarrier、fence）
- 薄封装（一个 inline function = 一条 PTX 指令，DeepGEMM 风格）
- **禁止**：
  - `#include "cutlass/*.h"` 或 `#include "cute/*.hpp"`（`cutlass/numeric_types.h` 除外）
  - `cutlass::gemm::collective::CollectiveBuilder`
  - `cutlass::gemm::kernel::GemmUniversal`
  - `cutlass::gemm::device::GemmUniversalAdapter`
  - `cutlass::epilogue::collective::CollectiveBuilder`
  - `using namespace cute`
  - 任何 CuTe layout algebra

### 实现

1. 按 `direction.md` **从头**实现完整 CUDA kernel，写在 `solution/`（新开空文件,
   不复制/不继承任何已有 kernel——见全局铁律 -2 FROM SCRATCH）
2. 插入 `// MODULE: <id> BEGIN/END` 标记
3. 写 benchmark adapter
4. `python bench/benchmark.py --correctness-only` — 正确性全部通过（正确性 gate 已
   内置在 benchmark.py：poison + oracle compare；`--correctness-only` 只跑这步、跳过计时）
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

这是合法且必要的，不是"被禁止的重写"。re-architecture = 写一个**全新的源文件**。
**锁全程保持上锁,不要 `rm`** —— hook 已改为「只拦覆盖已存在文件、放行新文件」,
所以写新架构文件本来就放行,无需也**不要**摘锁(摘了忘补回来 = 纪律静默失效,
这正是历史踩过的坑)。流程：
1. 先更新 `.rlcr/current/kernel-architecture.md`：写清"为何旧架构上限不够"
   （引用结构上限分析 + 实测证据）与新架构如何达到 ≥90% roofline 上限目标。
2. 新架构**直接 Write 一个新源文件**（锁开着也放行,因为是新文件不是覆盖），
   文件名按**全局递增轮次编号**命名（如 re-arch 在第 8 轮就叫 `<family>_r8.cu`，
   保持"文件名↔轮次"单调对应）。**不要**用 `v2`/`v3` 这类与轮次脱钩的名字，
   **也不要** `rm`/`touch` 那个锁。保留旧实现供对比；绝不 Write 覆盖旧文件本身
   （覆盖已存在文件仍被 hook 拦）。
3. 把 candidate ABI / adapter 切到新文件；旧文件在新版验证更快后用 `git rm` 删除。
4. 新文件插 MODULE 标记，跑 correctness + benchmark，写 `rounds/r<N>/` 的
   direction/summary/analysis + 5 类 SASS 产物，commit "re-architecture: <新架构>
   (initial)"，然后对**新文件**进入 Step 6/7 的渐进迭代（锁一直在,hook 全程强制）。

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
nvcc -ptx -lineinfo -arch=<ARCH> <source.cu> -o .rlcr/current/profiles/initial.ptx
nvcc -cubin -lineinfo -arch=<ARCH> <source.cu> -o .rlcr/current/profiles/initial.cubin
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

## Step 7: 模块循环 — RLCR 迭代

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
1.5. **本轮一开始就把 `.rlcr/current/state.md` 的当前轮号设为 N**——写一行
   **精确格式 `当前轮: r<N>`**（如 `当前轮: r8`）。SASS 门槛 hook 读这一行作为
   **权威轮号**，据此检查"上一轮 r\<N-1\> 的 `candidate-sass.txt` 是否就绪"；不再
   靠目录排序猜。**轮号必须在改 solution/ 之前就更新好**，否则 hook 会按上一轮的
   号判定、检查错对象（这正是历史上 SASS 被绕过的根因）。
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
   - `python bench/benchmark.py --correctness-only` — 正确性必须通过（**这一步是
     gate：错的代码不能进入 profile/benchmark**）
   - `python bench/benchmark.py` — 仅作**粗筛 sanity**（量级是否合理、有没有跑飞），
     **不在此处下"快了/慢了"的结论**。wall-clock 含 dispatch/包装层开销，不作性能
     判据（铁律 -0.5）。
6. **Regression check（分析，不回退）—— 判据只用 NCU**：本轮 vs 上轮的"快了/
   慢了/持平"**一律以 7b 的 NCU kernel duration（`gpu__time_duration`）为准**，
   在 7c 完成 NCU 实测后才下结论，**不得用 `bench/benchmark.py` 的 wall-clock 判
   进退**。
   - 如果 NCU duration 较上轮上升 > 5% 且不在预期内（direction.md 未预测到），
     立即 `git diff` 分析原因并写进 analysis.md。**但不回退**（铁律 #3）——commit
     本轮，继续前进；下一轮可在此基础上叠加（局部下降常被后续修改转正）。最优版
     在 Finalize 按 NCU 选出。
   - 仅当某改动**破坏正确性**时才 `git checkout`（错误恢复流程，铁律 #5 例外）。
   - 注意时序：本步只是"分析判进退"的占位说明，真正的数值对比发生在 7b（跑 NCU）
     之后的 7c；commit（第 7 步）可以先做（git 历史即安全网），进退结论落在 7c。
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
nvcc -ptx   -lineinfo -arch=<ARCH> <source.cu> -o $RD/candidate.ptx
nvcc -cubin -lineinfo -arch=<ARCH> <source.cu> -o $RD/candidate.cubin
cuobjdump -sass       $RD/candidate.cubin > $RD/candidate-sass.txt
cuobjdump -res-usage  $RD/candidate.cubin > $RD/candidate-res-usage.txt
nvdisasm  -gi -sf     $RD/candidate.cubin > $RD/candidate-nvdisasm.txt
```

### 7c. 分析 + 下一轮方向

1. **范围检查** — 是否有超范围改动

2. **理论 vs 实际** — 对比 direction.md 预测值 vs 实际
   - |gap| < 20% → 一致
   - 否则归因：实现差距（implementation gap）还是理论错误（theory error）

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
   - **调度层分析（强制，不能只看计数）**：每轮都必须从「统计层」（指令计数 /
     寄存器 / spill / bank-conflict）深入到「**调度 / 因果层**」——读热循环 SASS 的
     指令**发射顺序**：
     - **ptxas 有没有按你的意图排？** 还是把你的改动重排/抵消了？（手改和 inline
       PTX 都只是**提示**，ptxas 会二次调度——唯一确认方式是读 emit 出来的 SASS。
       这正是「能不能搏得动」的实测：`cuobjdump -sass` 看 ptxas 保留了多少 + NCU
       看目标 stall 有没有降。**绝不能只凭推断**。）
     - stall 到底卡在哪两条指令之间？`NOP` 填充 / scoreboard 等待 / `*DEPBAR`？
       math-pipe 指令（HMMA/QMMA…）是背靠背还是被 NOP 隔开？
     - **给每个 gap 归类**：**依赖气泡**（RAW 链——可用重排 / 多累加器 / 软件流水
       打破）vs **流水线吞吐 stall**（math pipe 本身吃不下——独立指令之间也夹 NOP；
       重排无用，只能加并发流或减总指令）。这个分类直接决定某优化「是否可能有效」。
     - **把每个 NCU stall 数对应到具体 SASS 模式**：别只写「wait=5.0」，要写
       「wait=5.0 ← Lxxxx 处 QMMA 突发之间夹 NOP 但累加器互相独立 ⇒ 吞吐 bound、
       非可重排依赖」。判「持平/退化」的轮次**必须**给出 SASS 证据说明**为什么**
       （ptxas 重排掉了？spill？RAW 链？），不能只凭 wall-clock 推断。

5. **策略轨迹** — 是否偏离 roadmap（>10% → 修正策略）

6. 写 `rounds/r<N>/analysis.md`（每条结论必须附 NCU metric 值或 SASS 指令证据）
7. **更新 `.rlcr/current/state.md`（每轮必做，抗压缩恢复依赖它）**：写明
   当前轮号 N、当前目标 module、本轮 verdict、最新 NCU duration、下一步 direction
   指向哪一轮。SessionStart hook 会把这份 state.md 作为「进度恢复卡」重注入
   context，所以它必须反映最新进度，否则压缩后会按过时状态续做。
8. 按 verdict 决定下一步：

| 判定（Verdict） | 动作 |
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

> **一个方向做到极致仍未达标 → 换思路，别在同一方向上继续微调（强制）**：
> 当某个优化**方向/架构**已被推到极致（增量调参只剩持平或退化，且 SASS 调度层
> 分析表明剩余瓶颈在该方向内**不可约**——如「NOP 是吞吐 stall 非依赖气泡」、
> 「再加 warp 必然 throttle 爆」、「更多累加器必然 spill」），**不要再在这个方向
> 上磨**。必须**主动换一条根本不同的思路**——通常是**更激进的并发结构 / 整体
> re-architecture**（例：从「per-step 全块 barrier」换到 warp specialization；从
> cp.async 换 TMA；从单累加器换 cooperative ping-pong 把更多 math 流喂进流水线
> 又不过载；persistent / cluster / split-K 等）。判断「方向已到极致」必须有**调度层
> SASS 证据 + 至少一次反向尝试的实测**（如 r14 加 warp、r15 加累加器都退化）撑腰，
> 不能只凭直觉。换思路属于已授权的自主决策（见下），不要因为「是个大改」就停下来
> 问或就此收尾——**目标没达成且还有根本不同的思路没试，就不算到顶**。

> **自主决策（不要为已授权的决定征求许可）**：本流程内的决定——继续下一轮、
> 转模块、拓宽搜索、**乃至 STRATEGY_REVISION→re-architecture**——都已被本 skill
> 授权，必须**自主执行，不要用 AskUserQuestion 去问用户"要不要做"**。即使某步
> token/工作量很大（如换架构重写），也不是征求许可的理由——授权已经给了。只有
> 遇到**真正越出任务范围的岔路**，或用户主动要求介入时，才停下来问。

---

## Step 8: 集成（Integration，每个模块完成后）

1. `git diff <module-start-commit>..HEAD -- solution/` — 汇总本模块所有改动范围
2. NCU full kernel profile + SASS 静态分析
3. 对比 baseline 整体性能（**以 NCU duration 为准**）：
   - 整体 speedup vs baseline = `baseline NCU duration / candidate NCU duration`
     （`gpu__time_duration`），**不用 wall-clock 算 speedup**（铁律 -0.5）
   - **逐模块退化检查（per-module regression check）**：检查每个已完成模块的 source-level NCU metrics，确认之前优化的模块没有退化
4. 如有 regression（以 NCU 判定）：
   - NCU + SASS 诊断根因
   - 如果是模块间干扰（如 shared memory 布局冲突、寄存器压力传导），写 `regression-analysis.md` 并修复
   - 修复后重新跑 NCU + correctness 确认
5. 更新 `module-tracker.json`、`goal-tracker.md`
6. 写下一轮方向 `rounds/r<N+1>/direction.md`（标明下一个目标模块）
7. git commit

---

## Step 9: Finalize

0. **选出最优版本（因为不回退，最优不一定是 HEAD）—— 排名只用 NCU**：扫所有已
   提交轮次，挑出**正确且 NCU kernel duration（`gpu__time_duration`）最短**的那一轮
   commit。**"最快"一律以 NCU 实测排序，不用 `bench/benchmark.py` 的 wall-clock**
   （铁律 -0.5）。若各轮的 NCU 记录已随 `.rlcr/` 清理而不全，则对候选的几轮 commit
   逐一 `git checkout` 后用**同一套 NCU 命令、同一块空闲 GPU** 重测 duration 再排名，
   不要凭 wall-clock 推断。选定后若它不是 HEAD，`git checkout <最优 commit> --
   solution/`（或在其上 cherry-pick 后续仍有效的修改）定为交付物，并重跑一次 NCU +
   correctness 确认。记录"哪一轮胜出 + NCU duration + 为什么"。
   （配合铁律 #3 退化不回退：过程允许走低谷，最优在此按 NCU 一次性选出。）
1. 写 `docs/results.md`：
   - 最优版本是哪一轮、对比各轮 **NCU duration**（含走过的低谷，体现 no-revert 探索）；
     wall-clock 如要列只作辅助参考列，不作排名/结论依据
   - 逐模块贡献拆解
   - 理论准确度总结
   - 最终每个 shape 的性能与 geomean speedup —— **均按 NCU duration 计算**
     （`baseline / candidate` 的 `gpu__time_duration`）
   - GPU 信息、roofline 总结（compute/memory bound 判定 + 达到 roofline 上限的百分比）
2. 写 `.rlcr/current/complete-summary.md`
3. 更新 `.rlcr/current/state.md`
4. git commit

---

## Step 10: 报告结果

向用户报告 `docs/results.md` 中的最终性能数据。
