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
