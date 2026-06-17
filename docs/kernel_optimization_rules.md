# Kernel 优化规则

这些规则适用于每个 kernel 优化任务。它们建立起让优化循环保持诚实且可复现的
护栏。

## 实现语言与抽象层级

优化后的 kernel（solution/）必须用 **CUDA C++** 编写。candidate 实现不得使用
Triton 或任何其他高层 kernel DSL。

优先使用 CUDA 中最原始的控制构造：

- **PTX inline assembly**（`asm volatile`）用于硬件特定操作：
  `cp.async.bulk`（TMA）、`wgmma.mma_async` / `tcgen05.mma`、`mbarrier`、
  `fence.proxy.async`、`setmaxnreg`、named barriers（`bar.sync`）等。
- **薄封装（thin wrappers）** over PTX 是可以接受的——一个 inline 函数对应一条
  PTX 指令，无状态、无抽象。DeepGEMM 风格，而非 CUTLASS 风格。
- **禁止使用** CUTLASS 的 Collective/Builder/Pipeline 抽象、`GemmUniversal*`
  以及 CuTe layout 代数。**唯一允许**的 CUTLASS 头文件是 `cutlass/numeric_types.h`
  （仅用于 dtype 定义）。**权威禁止清单以 `.claude/commands/optimize-kernel.md`
  Step 5「代码约束」为准**——本文件与其冲突时以该命令文档为准。注意这与 baseline
  无关：baseline 可以是任意现成库实现（见下），约束只针对你从零实现的 candidate。

为什么有此偏好：高度抽象的框架会让人更难推理硬件实际执行了什么。当每条 PTX
指令都在源码中可见时，分析者可以把 NCU metric 对应到具体代码，编码者也可以做
有针对性的修改。candidate 的核心交付价值正是这种「指令级可见、可归因」的实现，
所以即使某些功能（复杂 epilogue fusion、多阶段 pipeline 编排、高级 layout 变换）
从零写更费劲，也必须自己用 PTX 薄封装实现，而不是退回到 CUTLASS/CuTe 模板。

推荐（薄封装）：
```cuda
__device__ void tma_load(void* smem, uint64_t* mbar, ...) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier..."
                 : : "r"(...), "l"(...) : "memory");
}
```

baseline 可以使用任意实现（FlashInfer、CUTLASS、Triton）用于对比。

## Baseline 与 Candidate 配对

每个任务都必须以两个本地实现收尾：

- `baseline/`：参考 kernel 实现，通过任务 benchmark ABI 暴露。
- `solution/`：优化后的实现，通过完全相同的任务 benchmark ABI 暴露。

当任务 prompt 指定了某个特定 baseline 时，使用该实现。当没有给出特定 baseline
时，默认以对应的 FlashInfer kernel 作为参考实现。FlashInfer 必须从最新源码安装
为 AOT 预编译（cubin）版本（`FLASHINFER_ENABLE_AOT=1 pip install -e .`），使
baseline 反映其部署配置下可获得的最强实现。不要把 JIT/Python 模式当作
baseline——JIT 有运行时编译开销，不能代表生产性能。构建说明见 README。

在 `docs/baseline_source.md` 中记录 baseline 的来源：
- source：FlashInfer（默认）或其他指定实现
- version / commit
- 所使用的确切函数或 kernel 入口点
- 为适配 ABI 所做的任何本地修改

如果 baseline kernel 是 CUDA/C++ CUDA，则 baseline 与优化后的 candidate 必须使
用相同的本地注册/导出/构建方式。不要把 baseline 通过一个 wrapper 暴露，而把
candidate 通过一条更轻的直连路径暴露。

如果 baseline 实现是 Triton、CuTe DSL 或 Python，把它保持在 `baseline/` 内，并
构建一个本地 adapter，使其与 candidate adapter 具有相同的调用签名、参数顺序、
stream 行为和输出分配策略。

每次 CUDA launch 都必须使用 PyTorch 的当前 stream，例如
`at::cuda::getCurrentCUDAStream()`。

## 编译 Flag

只要编译 flag 可能改变数值或代码生成，它们在 baseline 与 candidate 之间就必须
对称。

不要传 `--use_fast_math`，除非 baseline 已经在用它，并且 candidate 使用完全相
同的 flag。默认是不开 fast math。

不要只给一侧添加额外的 `nvcc` flag、架构特定开关或数学模式 flag。在
`docs/benchmark_method.md` 中记录所有编译 flag。

## 远程 GPU 规则

针对某个特定 GPU 架构的任务，必须在该架构上验证并 benchmark。

在 GPU 工作之前，检查 `nvidia-smi`，选择一块没有活跃 compute 进程且没有明显内
存占用的 GPU。在当前运行中对 baseline、candidate、正确性、benchmark、profiling
和 NCU 命令一致地使用所选的那块 GPU。

在任务的 `docs/run_log.md` 或 `docs/results.md` 中记录主机、GPU id、GPU 型号，
以及运行前/后的 GPU 状态。

为构建、benchmark 日志、profiler trace 和 NCU 报告使用任务专属的工作区。不要把
产物写进另一个任务的工作区。

## 正确性优先于性能

优化之前，确认：

- baseline 源文件；
- 可调用的参数与标量参数；
- 生产 workload 行；
- 标准回归网格（如果在任务 prompt 或 `docs/correctness_contract.md` 中有
  定义）。

最终 candidate 必须通过生产 workload 正确性检查和标准回归网格，任何 benchmark
结果才算数。

保留显式的 NaN/Inf 检查。使用任务正确性契约中的 tolerance，除非任务在
`docs/benchmark_method.md` 中记录了更严格的任务局部 tolerance。

## 面向天花板设计（架构选择）

初始架构必须**面向性能天花板**设计，而不是设计成一个刻意简单的「正确性优先
v1」再逐步往上爬。第一版设计必须从一开始就采用达到目标效率所必需的每一项核心
技术（例如 warp specialization、ldmatrix、TMA、最优 tile/swizzle、异步多阶段
pipeline）。这些是架构骨架，而非事后加上的小调整——一个简单的结构有着任何增量
RLCR 调优都无法突破的硬效率天花板。

> **完成判据（PRIMARY）= 达到该 shape 的 roofline 上限的 ≥90%**，其中
> roofline 上限 = `min(compute 峰值, 带宽×算术强度)`：compute-bound 时即
> 90% spec 峰值；memory-bound 时为 90% 的 memory roofline（**低于 spec 峰值，
> 拿 spec 峰值当目标会物理不可达**）。先算 AI 与脊点判定属于哪类。
> **baseline（最强现成库实现）不是完成线，只是必须超过的下限参照。** 关键：
> 90% roofline 上限常常 **> baseline 实测效率**（库实现往往只到上限 85-90%），
> 所以「打平 baseline」**通常不等于**「完成」。完整定义与 `nvidia-smi`/roofline
> 推导以 `.claude/commands/optimize-kernel.md` Step 1 + Step 4d-ceiling 为准
> （本文件与其冲突时以该命令文档为准）。

在实现之前，除了硬件 roofline 之外，还要做一次**结构天花板分析**：

- 硬件 roofline = compute/memory 下限（GPU + 问题的固有属性）。先判 compute-/
  memory-bound，取 `min(compute 峰值, 带宽×AI)` 作为 roofline 上限，算出 **90%
  roofline 上限对应的目标 TF / µs**。
- 结构天花板 = **所选 candidate 架构**所能达到的最大效率。明确地自问：load/
  compute 是否被每步的 block barrier 串行化了？fragment load 是否无 bank
  conflict（ldmatrix）？是什么限制了 occupancy？wave quantization 是否凑不齐
  #SM？还缺哪些把利用率推到 90% roofline 上限的使能技术，每个缺失的技术各损失
  多少效率？

**门槛（Gate）：** 如果 `结构天花板 < 90% roofline 上限`，该设计达不到目标——
禁止进入实现，必须在写代码前重新设计、补齐使能技术，直到结构天花板 ≥ 90%
roofline 上限。（同时结构天花板必须 **> baseline 实测效率**，否则连现成实现都
赢不了。）如果达到 90% roofline 上限确实需要重写级别的工作量，那就提前说明，而
不是交付一个注定达不到目标的简单版本。若实测发现连 SOTA 库都远低于 90%，说明 90%
可能触及该 shape 的物理上限：如实告知用户当前 SOTA 百分比，由用户决定是否放宽，
在确认前仍以 90% roofline 上限为目标。

增量编辑护栏（迭代期间不覆盖文件）约束的是**在某一选定架构内部**的纪律。它绝不
会强迫接受一个注定要输的架构：当分析表明架构本身无法获胜时，从头重新设计并重新
实现（一个新的 candidate 源文件）——那是正确的行动，而非违规。

### 不因性能回退；每轮都 commit；定稿时选最优

性能回退**不**触发回滚。增量优化不可避免地会撞上某一步使性能下降，但在那一步
*之上*再做一次改动，常常能把它变成净赢（局部下探不是死胡同；例如某个优化单独看
是回退，但与后续改动组合起来却是净赢）。所以：

- 性能回退时**不要** `git checkout` 回滚。分析并记录它为何回退，commit 该轮，
  然后继续前进（你可以直接在回退版本之上构建）。
- **每轮都 commit**（无论变快还是变慢）。git 历史就是安全网——任何一轮都可恢
  复，因此不需要主动回滚。
- **在定稿时选择交付物**：扫描所有已 commit 各轮的 **NCU kernel duration
  （`gpu__time_duration`）**，`git checkout` 出那个既正确又最快的版本作为最终
  solution（"最快"以 NCU 为准，不用 wall-clock；见 optimize-kernel Step 9.0）。
- **唯一**的例外是正确性/编译失败：错误代码无法被 benchmark，所以恢复到上一个
  可用状态（那是「修到正确」，不是性能回滚）。

## Benchmark 与证据

以 `docs/benchmark_template.py` 作为计时 harness 的起点。调优开始后不要变更
workload、tolerance、分数聚合或计时规则，除非 baseline 与 candidate 都重新测
量。

每一次迭代在选择下一次编辑、benchmark 运行、profiling 运行或 no-go 结论之前，都
必须刷新其 kernel 优化上下文。该刷新包括任务 prompt、当前 benchmark 证据，以及
可用的知识 skill（例如 `external/KernelWiki/SKILL.md`、
`external/ncu-report-skill/SKILL.md`）。

当需要 NCU profiling 时，如果有 `external/ncu-report-skill/SKILL.md` 则遵循它。
把 profile harness、报告、分析和摘要保存在任务专属目录中，并用得到的证据来选择
下一次编辑，而不是猜测。

最终的性能结论必须报告：

- 每个 workload 的 median、mean、std、min、p10 和 p90 延迟；
- 生产 workload 上的等权几何平均加速比；
- 确切的命令行；
- baseline 与 candidate 的源码 hash；
- GPU 主机/id/型号以及空闲状态证据。

当一个正确的 candidate 尚未明显达成目标，或者 profiler 证据会改变下一次编辑时，
使用 Nsight Compute。最终的改进或 no-go 必须包含一个 roofline 式的解释：估算搬
运的字节数、有用的标量或向量运算、相关时的实测带宽和/或 FLOP/s，以及当前的瓶颈
（active bound）或阻塞点（blocker）。

不要因为第一个 candidate 输了就定稿成 no-go。一个 no-go 需要 baseline 数字、至
少一次有论证的 candidate 尝试、正确性状态、benchmark 证据，以及一个明确指出的瓶
颈或阻塞点。

## 底层汇编分析

每一轮优化都必须包含 PTX/SASS 静态分析。这不是可选步骤——
每次修改 kernel 代码后，都必须重新生成 cubin 并检查寄存器用量、指令模式和
编译器行为的变化。静态分析和 NCU 运行时 profiling 互补：NCU 告诉你"慢在
哪"，汇编分析告诉你"编译器做了什么导致慢"。

### 工具链

| 层级 | 命令 | 输出 | 关注点 |
|-------|---------|--------|------------------|
| PTX | `nvcc -ptx -arch=sm_100a file.cu` | `.ptx` 虚拟汇编 | 算法逻辑、指令选择、循环结构 |
| Cubin | `nvcc -cubin -arch=sm_100a file.cu` | `.cubin` 二进制 | 供下面工具使用的中间产物 |
| SASS | `cuobjdump -sass file.cubin` | 真实机器码 | 指令调度、dual-issue 配对、寄存器分配 |
| Resources | `cuobjdump -res-usage file.cubin` | 寄存器/共享内存用量 | occupancy 瓶颈诊断 |
| Source map | `nvdisasm -gi file.cubin` | SASS + 源码行号 | 定位哪行源码生成了哪些指令 |
| JIT .so | `cuobjdump -sass xxx.so` | 来自共享库的 SASS | FlashInfer JIT / AOT 产物分析 |

把 `sm_100a` 替换为实际目标架构（例如 Hopper 用 `sm_90a`）。`-arch` flag 必须
与被 benchmark 的 GPU 匹配。

### 每轮必须检查

- **寄存器压力**：`cuobjdump -res-usage` 显示每个 kernel 的寄存器数。如果
  occupancy 受寄存器限制，检查 PTX/SASS 找出不必要的活跃变量或溢出到 local
  memory（SASS 中的 `STL`/`LDL`）。
- **指令组成（instruction mix）**：`cuobjdump -sass` 揭示实际发射的指令。检查
  不必要的类型转换、冗余的 MOV、错失的常量折叠，或次优的数学序列（例如用了完整
  的 `DFMA` 而非 `FFMA`）。
- **双发射（dual-issue）分析**：在 SASS 输出中，寻找本可以一起发射却没有的指令
  对——通常由寄存器依赖链引起。
- **共享内存 bank conflict**：`nvdisasm -gi` 把 SASS load/store 映射回源码行。
  与 NCU 的 bank-conflict metric 交叉对照，定位出问题的访问模式。
- **Baseline 对比**：为 baseline 和 candidate 都生成 SASS，并排对比指令数、循环
  展开（loop unrolling）和内存访问模式。
- **FlashInfer 分析**：对 FlashInfer 的 `.so` 使用 `cuobjdump -sass`，在源码不
  可得或 JIT 编译选择不明确时，检查 baseline 实际生成的代码。
- **调度层分析（强制，不只是计数）**：每一轮都必须越过*统计*层（计数 / 寄存器
  / 溢出 / bank-conflict）抵达*调度 / 因果*层。阅读热点循环 SASS 指令的**排
  序**，而不只是总数：
  - `ptxas` 是保留了你想要的调度，还是重排/撤销了你的改动？（手工编辑和 inline
    PTX 只是*提示*——`ptxas` 会重新调度。唯一确认方式是阅读发射出的 SASS。）
  - 停顿究竟坐落在哪里——`NOP` 填充 / scoreboard 等待 / `*DEPBAR`——在哪两条指
    令之间？数学管线指令（HMMA/QMMA/等）是背靠背发射的，还是被 NOP 隔开了？
  - 给每个间隙分类：**依赖气泡（dependency bubble）**（一条 RAW 链，你可以通过
    重排 / 更多累加器 / 软件 pipelining 打破它）vs **管线吞吐停顿（pipe-
    throughput stall）**（数学管线就是无法接受另一条 op——NOP 出现在*相互独立*
    的 op 之间；重排不会有帮助，只有更多并发流或更少总 op 才行）。这个分类决定
    了某个修复是否根本可行。
  - **把每个 NCU 停顿数字都绑定到一个具体的 SASS 模式上。** 例如不要写
    "wait = 5.0"；要写 "wait = 5.0 ← QMMA 突发在 SASS Lxxxx 处被 NOP 隔开，
    尽管累加器相互独立 ⇒ tensor 管线吞吐受限，而非可重排的依赖。" 一轮被判为
    "持平"或"回退"的，必须给出*为什么*的 SASS 证据（ptxas 把它重排掉了？溢出
    了？RAW 链？），而不是单凭 wall-clock 推断。

### 记录

执行汇编分析时，把发现记录在该轮的分析文件
（`.rlcr/current/rounds/r<N>/analysis.md`）中。包含：kernel 名称、寄存器数、共享
内存用量、观察到的关键 SASS 模式，以及任何指导了优化方向的可行动洞见。

## PR 范围

kernel 优化完成后，最终 commit 只能包含：

- baseline、优化后的 solution、本地 ABI、benchmark adapter 以及正确性/benchmark
  harness 的 kernel 源码；
- 逐 shape 的 baseline-vs-candidate 性能对比和最终结论，通常放在
  `docs/results.md`；
- 复现该结果所需的小型方法/溯源说明。

不要 commit 中间优化产物，例如原始 NCU 报告、Nsight trace、profiler 运行目录、
临时 harness 二进制、构建输出、草稿日志、失败实验转储或大型 benchmark JSONL
文件，除非有明确要求。

## Shape 特化

当 benchmark 或 profiler 证据表明不同的 workload 分桶（bucket）需要不同的 block
size、向量宽度、内存 layout 或寄存器压力权衡时，允许使用 shape 特化 kernel、
模板变体、autotune 表和 dispatcher。

使用特化时，写一份 `docs/dispatch.md`，包含：

- 分桶条件；
- 所选的 baseline 与 candidate 入口点；
- 每桶的延迟和加速比；
- 该桶使用此实现的原因。

当证据表明多个 shape 分桶需要不同实现时，不要强行使用一个通用 kernel。

## 既有工作与探索

在任何一次迭代中确定实现策略之前，当可用的知识 skill 可能改变设计时，阅读或查询
它们：CUTLASS/CuTe、CUDA samples、PyTorch、KernelWiki，以及任务局部的 NCU 证据。

在 `docs/draft.md`、`docs/results.md` 或 `docs/research.md` 中记录采纳/否决的想
法。让优化尝试保持有界且有证据支撑。

## 完成标准

一个任务只有在满足以下条件时才算完成：

- `baseline/`、`solution/`、`bench/` 和 `docs/` 包含所需的本地产物；
- 生产 workload 正确性通过；
- 标准回归正确性通过（如适用）；
- benchmark 结果使用标准独立计时规则；
- NCU 或一份清晰的 roofline 式分析解释了最终结果或阻塞点；
- `docs/results.md` 总结了最终命令、逐 shape 性能对比、结果和结论；
- 已暂存（staged）的 diff 排除了原始 profiling、NCU、临时构建和草稿产物。
