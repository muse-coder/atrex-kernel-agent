# Kernel Optimization 经验教训

从 FP8 GEMM (SM120, RTX PRO 5000) 优化过程中总结的经验。
适用于所有基于 PTX inline asm 的手写 CUDA kernel 开发。

---

## 1. MMA Fragment Layout：必须查 PTX ISA 文档，不能参考 CuTe

### 问题
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` 每个线程的寄存器
对应矩阵的哪些元素？初次实现时参考了 CuTe 的 MMA traits（`MMA_Atom` 的
thread layout），使用 `t0 = lane_id % 8, t1 = lane_id / 8` 的线程分解。
结果 kernel 编译通过、能跑，但输出全错（max_abs 误差 400-500）。

### 根因
CuTe 的 MMA traits 描述的是它自己的**逻辑映射**（配合 `ldmatrix` / swizzled
loads 使用），**不是** PTX `mma.sync` 指令的原始硬件寄存器布局。

PTX ISA 硬件布局使用完全不同的线程分解：
```
groupID           = %laneid >> 2    // lane_id / 4  (0..7)
threadID_in_group = %laneid & 3     // lane_id % 4  (0..3)
```

### 正确做法
1. **直接查 PTX ISA 文档**：`external/CudaSkill/cuda_skill/references/ptx-docs/9-instruction-set/9.7.14.5-matrix-multiply-accumulate-operation-usingmmainstruction.md`
   - Section 9.7.14.5.10: `mma.m16n8k32` with `.e4m3/.e5m2` 的完整 fragment layout
2. **备选参考**：CUTLASS `arch/mma_sm89.h` 的 inline asm（直接对硬件寄存器，不经过 CuTe 抽象）
3. **绝对不要参考** CuTe 的 `MMA_Traits` / `ThrLayoutVMNK` — 那是 CuTe 内部的逻辑映射

### 耗时
约占整个初始实现阶段 60% 的时间。错误模式不直观（没有 crash，只是数值错误），
需要写 pattern test 才能定位。

### 具体 layout（mma.sync.m16n8k32, FP8 E4M3, row.col）

**A 矩阵 (m16×k32, 4 个 uint32 寄存器)**：
```
row = groupID                      for a[0..3] (regs a0, a1)
      groupID + 8                  for a[4..7] (regs a2, a3)  — 注意 +8 不是 +1
col = threadID_in_group * 4 + (i & 3)   for i < 8:  col range [0..15]
                                         for i >= 8: col range [16..31]
```
每个 uint32 寄存器包含 4 个连续 FP8 元素 → 4 字节对齐加载。

**B 矩阵 (k32×n8, col-major, 存储为 N×K row-major, 2 个 uint32 寄存器)**：
```
row = threadID_in_group * 4 + (i & 3)   for i < 4:  row range [0..15]
                                         for i >= 4: row range [16..31]
col = groupID
```

**C/D 矩阵 (m16×n8, 4 个 float 寄存器)**：
```
row = groupID           for c0, c1
      groupID + 8       for c2, c3
col = threadID_in_group * 2 + (i & 1)
```
每个线程写一个 2×2 子块（不是单列）。

---

## 2. SM 架构能力：不能靠猜测，先查文档再编译验证

### 问题
SM120 (Blackwell Desktop) 支持哪些 MMA 指令？先后尝试了：
1. `tcgen05.mma` → 编译失败（SM120 不支持 TMEM）
2. `wgmma.mma_async` → 编译失败（SM120 不支持 WGMMA）
3. `mma.sync.aligned.m16n8k32` → 成功

### 正确做法
1. **查 PTX ISA 文档的 "Target ISA Notes" 段落**：每条指令都标明了支持的 SM 版本
2. 快速验证：写一个最小 PTX asm snippet 编译测试，不要写完整个 kernel 再发现不支持
3. SM120 FP8 能力总结：
   - ✅ `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (PTX 8.4+, sm_89+)
   - ✅ `ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8` (PTX 8.6, sm_120a)
   - ✅ `cp.async.cg.shared.global` 4/8/16 bytes (sm_80+)
   - ✅ `cp.async.bulk` TMA (sm_90+)
   - ❌ `wgmma.mma_async` (sm_90a only, not sm_120)
   - ❌ `tcgen05.mma` / TMEM (sm_100a+ only)

### 耗时
每次尝试不支持的指令 → 编译报错 → 重新设计 → 约 15-20 分钟/次。
如果先查文档，3 分钟解决。

---

## 3. Baseline 调研：先确认 kernel 类型再深入

### 问题
花了大量时间阅读 FlashInfer SM120 的 `DenseGemmKernel`，
结果发现它是 **FP4 block-scaled**，不适用于 FP8 per-tensor scaling。

### 正确做法
1. 先 `grep` 关键类型/参数（dtype, scaling 方式）确认是否匹配目标
2. 对于 library kernel，先跑 `ncu --print-summary per-kernel` 看实际 kernel 名
3. 不要假设"同一个 library 的 SM120 kernel 一定适用于你的 dtype"

### 耗时
约 30 分钟白读代码。正确做法 5 分钟就能排除。

---

## 4. 共享内存 Swizzle：必须同时考虑 load 和 store 的访问宽度

### 问题
实现了 uint32 粒度的 XOR swizzle，消除了所有 bank conflict（理论上完美）。
但性能从 0.40x 退化到 0.32x，因为：
- 全局 → 共享的 store 从 2×uint4 (16字节) 退化为 8×uint32 (4字节)
- 全局 load 也从 uint4 退化为 uint32

修复全局 load（回到 uint4 后 scatter store）后恢复到 0.45x。

### 教训
1. **Swizzle 粒度必须兼顾 store 对齐**：如果 global→shared 用 uint4 (16字节) 写入，
   swizzle 至少要在 16 字节边界内保持连续
2. **Bank conflict 消除 vs 向量化 store 是一个 trade-off**：
   - uint32 swizzle: 0-way conflict, 但 8×4B stores (慢)
   - 16B-group swizzle: 2-way conflict, 但保持 uint4 stores (快)
   - 对于 cp.async pipeline（需要 16B 对齐），必须用 16B 粒度
3. **先 profile 确认 store 不是瓶颈再做细粒度 swizzle**

### 量化
| 方案 | Bank Conflict | Store 宽度 | 性能 |
|------|--------------|-----------|------|
| 无 swizzle | 5.3-way | uint4 (16B) | 0.404x |
| uint32 XOR + uint32 store | ~0-way | uint32 (4B) | 0.318x |
| uint32 XOR + uint4 load + uint32 scatter store | ~0-way | uint32 (4B) | 0.454x |

---

## 5. 调试策略：数值错误的高效定位

### Pattern Test 方法
当 kernel 输出全错但不 crash 时，随机输入的误差分布没有信息量。
使用 **patterned input** 才能看出规律：

```python
# A[m][:] = m % 8 * 0.25   (每行常数，不同行不同值)
# B[n][:] = n % 8 * 0.25   (每行常数)
# 期望 C[m][n] = K * A_val * B_val
```

错误输出的周期性模式直接暴露了线程-数据映射的错误：
- 行以 4 为周期重复 → groupID = lane_id/4 (每组 4 线程)
- 列以 2 为周期 → threadID_in_group 映射到 2 列

### 诊断脚本层次
1. **All-ones test**: 排除最基本的 MMA 指令/accumulator 问题
2. **Patterned test**: 暴露 fragment layout 映射错误
3. **Random test**: 确认整体精度

先跑 1 和 2，再跑 3。不要直接跑 3（随机输入的误差分布无法诊断 layout 错误）。

---

## 6. 环境与工具链

### PTX ISA 文档
- 位置：`external/CudaSkill/cuda_skill/references/ptx-docs/`
- MMA fragment layout: `9-instruction-set/9.7.14.5-*.md`
- cp.async: `9-instruction-set/9.7.9.25-*.md`
- ldmatrix: 同 MMA 文件内 Section 9.7.14.5.15
- 搜索指南: `external/CudaSkill/cuda_skill/references/ptx-isa.md`

### CUTLASS 参考（仅看 inline asm，不用模板）
- MMA inline asm: `atrex/build/cutlass/include/cutlass/arch/mma_sm89.h`
- 用途：验证 PTX 语法和寄存器约束（`"+f"` vs 分离 C/D）

### JIT 编译缓存
修改 kernel 后必须清除：
```bash
rm -rf /root/.cache/torch_extensions/py312_cu128/fp8_gemm_ext
```
不清除会导致用旧代码运行，误以为改动无效。

### NCU Profiling
```bash
# 先找 kernel 名（不加 -c 限制）
ncu --print-summary per-kernel python runner.py

# 再针对目标 kernel profile
ncu --set full --section PmSampling --section PmSampling_WarpStates \
    --section SourceCounters -k "regex:<NAME>" -c 1 -o output python runner.py
```

### 静态分析
```bash
nvcc -ptx -lineinfo -arch=sm_120 -I<includes> kernel.cu -o kernel.ptx
nvcc -cubin -lineinfo -arch=sm_120 -I<includes> kernel.cu -o kernel.cubin
cuobjdump -sass kernel.cubin > kernel-sass.txt
cuobjdump -res-usage kernel.cubin > kernel-res-usage.txt
```
关键检查：
- `REG:N STACK:0 LOCAL:0` → N 是寄存器数，STACK/LOCAL > 0 表示 spill
- SASS 中 `STL`/`LDL` = register spill（严重性能问题）
- HMMA 指令数应等于理论 MMA 次数

---

## 7. 性能优化优先级（基于 NCU 实测）

对于无 pipeline 的初始 kernel，瓶颈排序：

| 优先级 | 瓶颈 | NCU 指标 | 预期收益 |
|--------|------|----------|---------|
| 1 | 无 compute-memory overlap | Compute SM Throughput 27.5% (vs baseline 82.1%) | ~2x |
| 2 | 共享内存 bank conflict | 5.3-way avg, 74.87% est. speedup | ~1.5x |
| 3 | 全局 store 未合并 | 8/32 bytes utilized, 58.77% est. speedup | ~1.3x |
| 4 | MIO throttle stalls | 51.9% of stall cycles | ~1.2x |
| 5 | 低 occupancy | 33.3% theoretical (register limited) | ~1.1x |

**关键洞察**：Pipeline overlap (cp.async double buffering) 的收益远大于
bank conflict 消除。应该优先做 pipeline，然后再精调 bank conflict。

---

## 8. 渐进式修改的实际操作

### 做对的事
- 每轮只改一个优化点
- 改动后 `git diff` 确认范围
- 每次 benchmark 对比上轮整体性能
- Regression > 5% 立即停下分析

### 踩过的坑
- Swizzle 优化一次改了 store + load 两条路径，regression 时不知道哪条导致
  → 应该先只改 load 路径（用 swizzle 读），store 保持原样
- 没有在改之前 profile 确认瓶颈的真实占比
  → 应该先 NCU profile，算出每个瓶颈的理论收益上限，再决定优化顺序

---

## 9. 设计即上限：架构选错，渐进优化救不回来（fused BF16→FP8 GEMM 复盘）

### 问题
fused BF16→FP8 GEMM (M=1024, SM120) 任务里，第一步图省事选了"correctness-first
简单架构"：所有 warp「加载→`__syncthreads`→MMA」、手写 LDS 取 fragment、无 warp
specialization、无 ldmatrix。打算靠 RLCR 逐轮爬。结果：初版 0.45x，r2 async-A
爬到 0.56x，之后 r3(深流水)/r4(padding)/r5(BK=64) 三轮全 regression，卡在
~0.56x 打不过 baseline。

### 根因
这个结构的 tensor 利用率天花板 ~48-50%（加载与计算被 per-step 全块 barrier 串
行化 + 手写 LDS 有 21M bank conflict）。baseline (FlashInfer CUTLASS) 的 85% 来自
**warp specialization（producer/consumer 分离、去全块 barrier）+ ldmatrix（无冲突
取数）+ TMA**——这些是**架构骨架，不是后期 bolt-on 的 tweak**。简单架构的上限被
焊死，每轮渐进只是在焊死的上限内找局部最优，永远赢不了。r3/r4/r5 想突破都失败，
正因为突破=换架构。

### 正确做法（已写进框架）
1. **第一步设计就奔上限**：直接采用打赢 baseline 所需的全部核心技术。
2. **结构上限分析（强制门槛）**：除硬件 roofline 外，推导"所选架构本身的效率
   上限"；若 < baseline 实测效率 → 注定输，禁止进入实现，先重设计。
   （见 `optimize-kernel` Step 4d-ceiling、`kernel_optimization_rules.md`
   §Design To The Ceiling。）
3. **渐进式约束的边界**：「禁止重写」只管"既定架构内迭代"；架构本身赢不了时，
   STRATEGY_REVISION→重新设计→从头实现新架构是合法且必要的（新写一个
   `kernel_v2.cu`，不覆盖被锁旧文件，绕过防重写 hook 且保留对比）。

### SM120 fp8 GEMM 奔上限的关键技术可行性（已编译验证, CUDA13/sm_120a）
- `ldmatrix.sync.aligned.m8n8.x4.shared.b16`：✅（A 用 bf16 ldmatrix 再 cvt 到 fp8）
- `ldmatrix ... .b8`（fp8 直接）：m16n16.x2/x4、m8n8.x4 都 ❌（ptxas 报 vector/shape
  不符）——fp8 fragment 不能直接 ldmatrix，需走 b16 路径或手写。
- `mbarrier.init/arrive/try_wait.parity`：✅（warp specialization 可用）
- `cp.async.bulk.tensor.2d ... mbarrier::complete_tx::bytes`（TMA）：✅
- `cvt.rn.satfinite.e4m3x2.bf16x2`：❌（ptxas 不收），用 bf16→f32(`bits<<16`)→
  `cvt.e4m3x2.f32`。

---

## 10. PTX / SASS 静态分析的正确读法（产物体量、计数陷阱、跨轮对比）

> ⚠️ **架构限定：本节所有具体助记符与行为均为 sm_120（RTX PRO 5000, Blackwell
> 消费级）+ CUDA 13 实测，不要照搬到别的架构。** 跨架构会变的至少有：
> - **MMA 指令名**：sm_120 是 `QMMA`（PTX `mma.sync … .e4m3`）；sm_90 Hopper 是
>   `HGMMA/WGMMA`（`wgmma.mma_async`）；sm_100 是 tcgen05（`OMMA`/tmem）。本节的
>   `QMMA`、`FILLER@!UPT`、填充/QMMA≈0.81、`.reuse` 等都是 sm_120 的形态。
> - **FP 收缩/强度削减**（见 10.4 末）是 nvcc/ptxas 13 在 sm_120 上的选择。
> - **`cutlass` 命名启发式**（10.5）是 sm_120 ptxas 实测的 ~+2%，别处未必成立。
> 方法学（grep 定位、直方图 Δ、归一化三类噪声、PTX 判对错/SASS 判快慢）跨架构通用；
> 具体指令名/数值不通用——换架构要重新 `cuobjdump -sass` 实测确认。
>
> 以下数据均出自 RTX PRO 5000 sm_120 fp8 GEMM campaign（r25–r28），每条都有支撑。

### 10.1 产物动辄上万行 → 禁止整文件读入 context，必须 grep 定位 + sed 切片
一个 12K 的 `.cu` 编出来：`candidate.ptx` ~11.6K 行 / `candidate-sass.txt` ~5.5K 行 /
`candidate-nvdisasm.txt` ~15K 行（因为 `#pragma unroll` 把 5 处 mma 调用展开成 448 条
QMMA）。整读会撑爆 context。正确做法：
1. `grep -nE 'QMMA|LDSM' file` 找热循环 + 回边地址（`@P0 BRA 0x….`）；
2. `sed -n '<lo>,<hi>p'` 只切热循环体那一两千行；
3. `grep -c` 统计指令类；只把 ~100 行稳态发射片段读进 context 做因果判断。
nvdisasm（最大）一般不整看，只按需 grep 某个分支。

### 10.2 NOP 计数陷阱：ptxas 的调度填充不是字面 `NOP`
判「填充/QMMA 比」时，`grep '\bNOP\b'` 会严重漏数——ptxas 真正插的发射填充是
**`@!UPT UIADD3 URZ`**（谓词关掉、写 URZ 的哑指令）。本 campaign 主循环字面 NOP 仅 4，
但 `@!UPT…URZ` 有 204，真实填充/QMMA ≈ 0.81（不是误算的 0.34）。统计调度密度必须把
这类哑指令算进去（`scripts/sass_hist_diff.sh` 已把它单列为 `FILLER@!UPT`）。

### 10.3 跨轮对比：SASS 用「指令直方图」，PTX 用「归一化逐行」，都别裸 diff
- **裸 diff 无效**：SASS 带 `/* 0x.. */` 编码列 + 物理寄存器号 + 地址 + ptxas 重排，
  两版 5.5K 行 SASS 裸 diff 7900+ 行，全是噪声。
- **SASS → 指令类别直方图 Δ**：`scripts/sass_hist_diff.sh <轮A> <轮B>`。一眼读出改动
  落在哪类指令（如 warp-spec epilogue 那轮：`STS +64 / STG −24 / BAR +2 / FMUL −96`，
  即 scale/convert/store 搬离了 MMA warp）。这是判「这轮改动是否如预期」的主工具。
- **PTX → 归一化逐行**：`scripts/ptx_diff.sh <A.ptx> <B.ptx>`。

### 10.4 PTX 前后对比的三类噪声（必须先归一化，否则真改动被淹）
1. **虚拟寄存器编号** `%r/%rd/%f/%p`：最大噪声源，归一化后实测降 ~8.5x（6134→734 行）。
   —— 注意：关 `-lineinfo`（去 `.loc`）几乎没用（只降 1.5%），`.loc` 在 diff 里自对齐；
   **降噪靠归一化寄存器，不是关 lineinfo**。
2. **内部文件名哈希符号** `_INTERNAL_<hash>_<n>_<file>_cu_<hash>`：thrust/cuda::std 注入，
   **只要换源文件名就全变**，与改动无关 → 折叠掉，或两版用同一文件名编。
3. **Itanium mangling 长度前缀** `_ZN39…` vs `_ZN44…`：改成不同长度的名字会牵动。
看「类别对应」不看行数：源码 24 行的改动 PTX 可能差 1000 行（`[16]` 数组 + 循环展开 +
基本块 `$L__BB0_NN` 重排放大），但每类都能映射回源码构造（如 `seg/order[16]` →
`__local_depot0` + `st.local`/`ld.local`，正好对上 SASS 里的 spill）。

**关键陷阱：别按「我以为该出哪条指令」判，要看编译器实际降成了什么（sm_120/CUDA13 实测）。**
做了 5 组受控改动核对源码↔PTX，4 组直接命中（纯改名→指令体 0 变化；`STAGES 3→2`→
`setp …,3` 变 `…,2`；删 `__syncwarp()`→`bar.warp.sync` 14→0；`GRID_CTAS 110→55`→
`add.s32 …,110` 变 `…,55`），但**第 5 组踩坑**：epilogue 给每个输出 `* 2.0f`，预测
`+mul.f32`，结果 `mul.f32` 计数纹丝不动、**多出 128 条 `fma.rn.f32`**——nvcc 把
`v*sa*sb*2.0` 整条乘法链**收缩成了 FMA**（FP contraction）；`*2.0` 也可能被强度削减成
`add`（x+x）。所以核对浮点改动要**把 mul/fma/add 整类一起数**，否则会误判成「改动没进
IR」（其实进了，只是换成了等价指令）。整型同理可能被 `lea`/`shf`/`mad` 改写。

### 10.5 PTX 判「对不对」，SASS 判「快不快」——分工与一个关键反例
- PTX = 前端降级结果（ptxas 之前）：看指令选择对不对（`mma.sync` 形状、`ldmatrix`、TMA）、
  unroll、源码有没有被翻译对。**性能信号在 PTX 里全是 0**（填充槽 / `.reuse` /
  `STL/LDL` spill / 真实 QMMA 数都只在 SASS）。
- SASS = ptxas 实现层：寄存器/spill（res-usage）、调度填充、`.reuse`、内存宽度/cache
  修饰、bank-conflict 风险、真实指令数 —— **唯一能对上 NCU 的层**。
- 关键反例（验证过）：kernel 名含 `cutlass` 子串这个 ~+2% 的 trick，**源码只改标识符、
  PTX 指令体 0 变化**（受控对照：纯改名后真指令体 diff = 0 行），差异完全发生在 ptxas
  看到符号名后的调度决策 → **只能用 SASS+NCU 判其效果，PTX diff 永远看不见**。
- 验证「我的改动有没有如实进 IR」用 `scripts/ptx_diff.sh`；验证「快了没 / 为什么」用
  `scripts/sass_hist_diff.sh` + NCU duration。性能结论一律落在 SASS+NCU（见硬性要求 -0.5）。
