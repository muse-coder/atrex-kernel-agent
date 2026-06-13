# Module Decomposition Guide

Decomposer agent 参考文档：如何将 GPU kernel 拆分为可独立优化的模块。

## 分解原则

1. **同步点是天然边界** — barrier、fence、`__syncthreads()` 前后是不同模块
2. **Warp-role 分支是并行边界** — producer/consumer warp 各自一个模块
3. **模块粒度** — 最小 ~10-20 SASS 指令，最大 8-10 个模块/kernel
4. **标记格式** — `// MODULE: <id> BEGIN` / `// MODULE: <id> END`，id 用 kebab-case

## 典型分解

### GEMM (pipeline software-pipelining style)

| Module ID | 描述 | 典型运行时间占比 |
|---|---|---|
| `prologue` | TMA pipeline fill — 前 N 个 stage 的异步加载 | 5-10% |
| `mainloop-load` | K-dimension 循环中的 TMA cp.async 发射 | 15-25% |
| `mainloop-mma` | Tensor Core MMA (wgmma/tcgen05.mma) 指令 | 40-60% |
| `mainloop-barrier` | mbarrier arrive/wait, fence, pipeline 控制 | 5-10% |
| `epilogue-compute` | 累加器类型转换、bias add、activation | 5-15% |
| `epilogue-store` | TMA/全局内存写回 | 5-10% |

### Attention (FlashAttention style)

| Module ID | 描述 |
|---|---|
| `qk-load` | Q、K tile 加载到 smem/register |
| `score-compute` | QK^T matmul |
| `softmax` | Online softmax: row-max, exp, row-sum |
| `pv-compute` | score @ V matmul |
| `output-accumulate` | 跨 K-tile 累加器 rescale + 累加 |
| `output-store` | 最终输出写回全局内存 |

### Reduction (multi-level)

| Module ID | 描述 |
|---|---|
| `thread-local` | 每个线程的局部归约 |
| `warp-reduce` | warp 内 shuffle reduce |
| `block-reduce` | smem + `__syncthreads()` 跨 warp 归约 |
| `grid-reduce` | atomicAdd 或多阶段 global 归约 |

### Fused Ops

每个被 fuse 的算子一个模块，加上 `data-load` 和 `data-store` 各一个。

## 共享资源识别

必须记录跨模块共享的资源：

| 类型 | 示例 | 优化影响 |
|---|---|---|
| smem buffer | `smem_A`, `smem_B` | 修改 layout/padding 影响所有使用者 |
| barrier | `mbarrier[N]` | 修改 arrive/wait 时序影响 pipeline |
| pipeline state | stage 计数、phase bit | 修改 stage 数影响 prologue 和 mainloop |
| register accumulator | 跨循环迭代的累加器 | 修改精度/布局影响 epilogue |

## 优化顺序

1. **按运行时间占比降序** — 优化最热的模块收益最大
2. **尊重依赖** — 如果 B 依赖 A 的输出格式，先稳定 A
3. **先 mainloop 后 prologue/epilogue** — mainloop 通常占 60%+ 运行时间
4. **barrier/sync 模块最后** — 它们的优化通常是调整时序，需要其他模块先稳定
