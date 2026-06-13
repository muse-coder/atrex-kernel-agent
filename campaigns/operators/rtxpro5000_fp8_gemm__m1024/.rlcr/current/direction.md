# Initial Kernel Implementation

## Kernel Type and Semantics

**FP8 GEMM with per-tensor scaling**: C[M,N] = (scale_a * scale_b) * A[M,K] @ B[N,K]^T

- A: (M, K) FP8 E4M3, row-major, contiguous
- B: (N, K) FP8 E4M3, row-major (stored as N x K; transposed during MMA)
- C: (M, N) BF16, row-major, contiguous
- scale_a, scale_b: FP32 per-tensor scalars
- Primary shape: M=1024, N=10240, K=4096
- Regression shapes: M=3584/8192/16384, N=10240, K=4096
- GPU: NVIDIA RTX PRO 5000 72GB Blackwell (110 SMs, SM 12.0, `sm_120`)
- Compile with `-arch=sm_120`

## Architecture Overview

The kernel targets 128x256x64 tile shape (vs baseline's 64x128x64) to reduce
wave quantization from 5.82 waves to 1.45 waves on 110 SMs. See
`kernel-architecture.md` for full derivation.

| Parameter | Value |
|-----------|-------|
| CTA tile | 128 x 256 x 64 |
| Block size | 256 threads (8 warps, organized as 8x1x1 or 4x2x1) |
| Warp tile | 64 x 64 |
| K-tile | 64 (FP8 = 64 bytes per row) |
| Pipeline stages | 4 (quad-buffered A + B in smem) |
| Shared memory | ~96 KB (4 stages x 24 KB) |
| MMA instruction | QMMA.SF.16832.F32.E4M3.E4M3.E8 |
| Registers/thread | Target < 255 (aim for ~208) |

Grid: `dim3(ceil(N/256) * ceil(M/128), 1, 1)` = `40 * 8 = 320` blocks for
primary shape.

## Module Structure

The kernel is organized into 7 modules with `// MODULE: <id> BEGIN/END` markers.
Implement all modules in a single kernel function.

### Module List

```
// MODULE: prologue BEGIN
// ... barrier init, pipeline fill
// MODULE: prologue END

// MODULE: mainloop-tma BEGIN
// ... TMA load dispatch per K-iteration
// MODULE: mainloop-tma END

// MODULE: mainloop-smem-load BEGIN
// ... LDSM from shared memory to registers
// MODULE: mainloop-smem-load END

// MODULE: mainloop-mma BEGIN
// ... QMMA tensor core computation
// MODULE: mainloop-mma END

// MODULE: mainloop-barrier BEGIN
// ... mbarrier arrive/wait/fence
// MODULE: mainloop-barrier END

// MODULE: epilogue-convert BEGIN
// ... FP32 to BF16 conversion with scale
// MODULE: epilogue-convert END

// MODULE: epilogue-store BEGIN
// ... Output store to global memory
// MODULE: epilogue-store END
```

## Shared Memory Layout

Total shared memory: 4 stages x (A_tile + B_tile) per stage.

```
A_tile per stage: 128 rows x 64 cols x 1 byte (FP8) = 8192 bytes
B_tile per stage: 256 rows x 64 cols x 1 byte (FP8) = 16384 bytes
Per stage: 24576 bytes = 24 KB
Total: 4 x 24 KB = 96 KB (fits in 102.4 KB smem config)

Layout in shared memory:
  [stage 0] A: offset 0,        size 8192   B: offset 8192,    size 16384
  [stage 1] A: offset 24576,    size 8192   B: offset 32768,   size 16384
  [stage 2] A: offset 49152,    size 8192   B: offset 57344,   size 16384
  [stage 3] A: offset 73728,    size 8192   B: offset 81920,   size 16384
```

Use 128-byte XOR swizzle for both A and B tiles to eliminate bank conflicts
during LDSM loads. The swizzle function:
```
swizzled_offset = offset ^ ((offset >> 7) & 0x7) << 4;
```
This swizzles the 128-byte-aligned groups to avoid N-way bank conflicts.

Set shared memory via `cudaFuncSetAttribute`:
```cpp
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
```

## Implementation Details Per Module

### Module: prologue

**Purpose**: Initialize mbarriers for 4 pipeline stages and pre-fill the first
4 stages with TMA loads.

**Key operations**:
1. Compute CTA tile position: `tile_m = blockIdx / N_tiles`, `tile_n = blockIdx % N_tiles`
2. Compute per-thread identifiers: warp_id, lane_id, warp_m, warp_n
3. Thread 0 (or warp 0, lane 0 via uniform predicate):
   - Initialize 4 mbarriers with expected transaction bytes
   - Issue 4 pairs of TMA loads (A tile + B tile for each stage)
4. All threads: zero accumulator registers

**PTX for mbarrier init** (thin wrapper):
```cuda
__device__ void mbarrier_init(uint64_t* mbar, uint32_t expected_count) {
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_shared_to_generic(mbar)), "r"(expected_count)
    );
}
```

**PTX for TMA load** (A tile, 2D):
```cuda
__device__ void tma_load_2d(void* smem_ptr, uint64_t* mbar, uint64_t tma_desc,
                            int32_t coord_x, int32_t coord_y) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"((uint32_t)__cvta_shared_to_generic(smem_ptr)),
           "l"(tma_desc), "r"(coord_x), "r"(coord_y),
           "r"((uint32_t)__cvta_shared_to_generic(mbar))
        : "memory"
    );
}
```

**Accumulator zeroing**: Use `CS2R R_i, SRZ` pattern (clear-and-set-register to
zero). In CUDA C++, just zero-initialize the accumulator array.

**Synchronization**: `BAR.SYNC 0` after prologue to ensure all threads are ready.

### Module: mainloop-tma

**Purpose**: Issue TMA loads for the next pipeline stage.

**Key operations** (per K-iteration):
1. Compute TMA descriptor coordinates for A and B
2. Thread 0 issues TMA load for A tile at current K offset
3. Thread 0 issues TMA load for B tile at current K offset
4. Arrive on mbarrier with expected bytes

**Data flow**:
- Reads: K iteration counter, TMA descriptors (from constant memory)
- Writes: Shared memory buffers (asynchronously), mbarrier state

**PTX for arrive with transaction bytes**:
```cuda
__device__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_shared_to_generic(mbar)), "r"(tx_bytes)
    );
}
```

### Module: mainloop-smem-load

**Purpose**: Load A and B tile fragments from shared memory to registers using
LDSM instructions.

**Key operations**:
- Wait on mbarrier for current stage (data ready)
- Use `LDSM.16.M88.4` (ldmatrix.sync.aligned.m8n8.x4) to load fragments
- Each warp loads its portion of A (2 LDSM per M-step = 4 total) and B
  (8 LDSM for 8 N-steps)

**PTX for ldmatrix**:
```cuda
__device__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                            const void* smem_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"((uint32_t)__cvta_shared_to_generic(smem_ptr))
    );
}
```

**Register budget**: ~48 registers for A/B fragments (12 LDSM x 4 regs = 48)

### Module: mainloop-mma

**Purpose**: Execute QMMA tensor core operations for the current K-tile.

**Key operations**:
- 32 QMMA instructions per warp (4 M-steps x 8 N-steps) for K=32
- Repeat for K=64 (2 K-steps within the K-tile) = 64 QMMA total
- Actually the baseline does K=64 in one QMMA pass with K=32 granularity,
  issuing 32 QMMAs total. Follow the same pattern.

**PTX for QMMA** (Blackwell SM120 FP8 MMA):
```cuda
// This maps to QMMA.SF.16832.F32.E4M3.E4M3.E8
// The exact PTX syntax for SM120 QMMA needs verification.
// Alternative: use the mma.sync PTX if QMMA is cuBLAS-internal only.
// For SM120, the standard PTX MMA for FP8:
__device__ void mma_fp8_16832(float* d, uint32_t* a, uint32_t* b, float* c,
                               float scale_a, float scale_b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}
```

Note: The baseline uses `QMMA.SF` which is a cuBLAS-internal instruction that
may differ from the standard PTX `mma.sync` path. If QMMA is not accessible
via standard PTX, use `mma.sync.aligned.m16n8k32` instead. The SM120 PTX ISA
should support FP8 MMA via the standard `mma.sync` PTX instruction.

**Register budget**: 128 registers for accumulators (32 groups x 4 regs)

### Module: mainloop-barrier

**Purpose**: Manage pipeline barrier synchronization between TMA loads and MMA
consumption.

**Key operations**:
1. `mbarrier.try_wait.parity` to wait for current stage data
2. `fence.proxy.async` after wait to ensure data visibility
3. `mbarrier.arrive` to signal that current stage buffer is consumed
4. Advance pipeline stage counter

**PTX for try_wait**:
```cuda
__device__ bool mbarrier_try_wait_parity(uint64_t* mbar, uint32_t phase) {
    uint32_t result;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}"
        : "=r"(result)
        : "r"((uint32_t)__cvta_shared_to_generic(mbar)), "r"(phase)
    );
    return result != 0;
}
```

**Synchronization pattern**:
```
for k = 0 to K/64 - 1:
    wait(mbar[k % NUM_STAGES], phase)
    fence.proxy.async
    // MODULE: mainloop-smem-load (LDSM)
    // MODULE: mainloop-mma (QMMA)
    // MODULE: mainloop-tma (issue next TMA if k + NUM_STAGES < K/64)
    arrive(mbar[k % NUM_STAGES])
    phase ^= 1  (when wrapping around stages)
```

### Module: epilogue-convert

**Purpose**: Apply per-tensor scale and convert FP32 accumulators to BF16.

**Key operations**:
1. Load scale = scale_a * scale_b into a register (precomputed in prologue)
2. For each accumulator group: `acc[i] = acc[i] * scale`
3. Convert pairs of FP32 values to packed BF16: `F2FP.BF16.F32.PACK_AB`

**PTX for F32-to-BF16 conversion**:
```cuda
__device__ uint32_t f32x2_to_bf16x2(float a, float b) {
    uint32_t result;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(b), "f"(a));
    return result;
}
```

**Register flow**: Reads accumulator registers (R8-R71), writes BF16 packed
values into temporary registers for store.

### Module: epilogue-store

**Purpose**: Write BF16 output to global memory.

**Key operations**:
1. Compute output address: `C_ptr + (tile_m * 128 + warp_m * 64 + local_row) * N + tile_n * 256 + warp_n * 64 + local_col`
2. Store packed BF16 values using vectorized stores

**Preferred: 128-bit vectorized stores**:
```cuda
// Store 8 BF16 values (16 bytes) per instruction
__device__ void stg_128bit(void* ptr, uint4 data) {
    asm volatile(
        "st.global.v4.b32 [%0], {%1, %2, %3, %4};"
        :: "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w)
        : "memory"
    );
}
```

Using 128-bit stores (STG.128) instead of 16-bit stores (STG.U16) reduces
store transactions by 8x, directly addressing the baseline's 50% excessive
store sector issue.

**Store pattern**: Each thread stores its portion of the 64x64 warp tile.
With 32 threads per warp and 64x64 = 4096 BF16 elements:
- Each thread stores 128 BF16 elements = 256 bytes = 16 stores of 128 bits

## Baseline Weaknesses to Exploit

1. **Wave quantization (5.82 waves -> 1.45 waves)**: The 128x256 tile reduces
   total tiles from 1280 to 320, dramatically reducing tail waste. This is the
   single biggest improvement opportunity.

2. **Uncoalesced epilogue stores (50% excessive sectors)**: The baseline uses
   individual 16-bit STG.E.U16 stores. Our kernel uses 128-bit vectorized stores,
   reducing store transaction count by 8x.

3. **Excessive epilogue length (~400 SASS instructions)**: The baseline's
   epilogue is disproportionately long because it interleaves FFMA scale, F2FP
   convert, and STG store for each element individually. Our kernel batches
   the operations: all scales first, all converts, then all stores.

4. **Pipeline depth mismatch**: The baseline uses 8 pipeline stages with only
   50 KB shared memory. Our 4-stage pipeline with 96 KB shared memory is better
   matched to the tile size and K-dimension.

## Performance Target

- **Minimum**: >= 90% roofline efficiency (>= 376 TFLOPS, <= 228 us)
- **Stretch goal**: >= 94% roofline efficiency (>= 393 TFLOPS, <= 218 us)

### Derivation

Peak FP8 TFLOPS on RTX PRO 5000: ~418 TFLOPS
FLOPS for M=1024, N=10240, K=4096: 2 * 1024 * 10240 * 4096 = 85.9 GFLOPS

At 90% efficiency: 85.9e9 / (418e12 * 0.90) = 228 us
At 94%: 85.9e9 / (418e12 * 0.94) = 219 us

The baseline achieves 223 us (92.2% measured by NCU). Our target is achievable
primarily through better tile shape selection (1.45 vs 5.82 waves), with
epilogue improvements providing additional margin.

For regression shapes (M=3584/8192/16384), the larger tile should perform at
least as well since those shapes have better wave coverage in both configurations.
