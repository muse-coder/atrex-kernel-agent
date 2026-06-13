# Kernel Architecture: FP8 GEMM for RTX PRO 5000 (SM 12.0)

## 1. Problem Specification

Compute C = scale * (A @ B^T) where:
- A: (M, K) in FP8 E4M3, row-major
- B: (N, K) in FP8 E4M3, row-major (transposed for GEMM: B^T is K x N)
- C: (M, N) in BF16, row-major
- scale = scale_a * scale_b (per-tensor FP32 scalars)
- Primary shape: M=1024, N=10240, K=4096

## 2. Tile Configuration

### 2.1 CTA Tile: 128 x 256 x 64

Rationale:
- M_tiles = 1024/128 = 8
- N_tiles = 10240/256 = 40
- Total tiles = 320
- Waves on 110 SMs (2 blocks/SM max) = 320 / 220 = **1.45 waves**
- Last wave: 100 blocks on 110 SMs = 90.9% utilization
- vs baseline: 1280 tiles / 220 = 5.82 waves (much worse tail)

The 128x256 tile dramatically improves wave efficiency from 5.82 waves to 1.45
waves, reducing tail waste from ~18% to ~9%.

### 2.2 K-Tile: 64

- K_iterations = 4096 / 64 = 64 iterations per mainloop
- Each K-tile: A fragment is 128x64 (8192 bytes FP8), B fragment is 256x64 (16384 bytes FP8)
- Total per iteration: 24576 bytes = 24 KB

### 2.3 Alternative: 64x256x64

If register pressure prevents 128x256:
- M_tiles = 1024/64 = 16, N_tiles = 10240/256 = 40
- Total = 640 tiles, waves = 640/220 = 2.91 waves
- Still better than baseline's 5.82 waves

## 3. CTA Shape and Warp Layout

### Block Size: 256 threads (8 warps)

Warp layout for 128x256 tile:
- 2 warps along M (each covers 64 rows)
- 4 warps along N (each covers 64 columns)
- Total: 8 warps per CTA

Each warp computes a 64x64 output tile using QMMA.16832 instructions:
- 64/16 = 4 M-steps, 64/8 = 8 N-steps per warp
- 4 * 8 = 32 QMMA instructions per warp per K-tile iteration

### Warp Specialization

Two roles:
- **Producer warp** (warp 0): Issues TMA loads, manages barriers
- **Consumer warps** (warps 1-7): Execute QMMA tensor core operations

Consumer warp 0 also participates in MMA when not issuing TMA loads, using
predicated TMA issue. The actual MMA work distribution is:
- Warps 0-1: M-group 0 (rows 0-63)
- Warps 2-3: M-group 0 second half or M-group 1
- Warps 4-7: N-dimension coverage

Actually, for simplicity and to match the proven baseline pattern, use
**cooperative** warp execution (all warps execute MMA, one warp also handles TMA):
- All 8 warps execute QMMA
- Thread 0 of warp 0 additionally issues TMA loads (via uniform predicate UP0)
- This matches the baseline's pattern which works well

## 4. Pipeline Structure

### 4.1 Software Pipeline: 4 stages

Rationale for reducing from baseline's 8 stages to 4:
- Fewer stages = fewer shared memory buffers = more smem for larger tiles
- 4 stages still provides good latency hiding for TMA loads
- Shared memory budget: 4 * (128*64 + 256*64) * 1 byte = 4 * 24576 = 98304 bytes = 96 KB
- This fits within the 102.4 KB shared memory configuration

If 96 KB is too tight with padding/swizzle, reduce to 3 stages:
- 3 * 24576 = 73728 bytes = 72 KB (comfortable)

### 4.2 Pipeline Phases

```
Stage 0: [TMA_A, TMA_B] -> smem_buf[0]
Stage 1: [TMA_A, TMA_B] -> smem_buf[1]
Stage 2: [TMA_A, TMA_B] -> smem_buf[2]
Stage 3: [TMA_A, TMA_B] -> smem_buf[3]
```

Prologue fills stages 0-3, then mainloop begins consuming stage 0 while
loading into the next available stage.

## 5. Shared Memory Layout

### 5.1 Double/Quad Buffered A and B

```
smem layout (4 stages, total ~96 KB):

  Stage 0:  A[128x64] = 8192 bytes  |  B[256x64] = 16384 bytes  = 24576 bytes
  Stage 1:  A[128x64] = 8192 bytes  |  B[256x64] = 16384 bytes  = 24576 bytes
  Stage 2:  A[128x64] = 8192 bytes  |  B[256x64] = 16384 bytes  = 24576 bytes
  Stage 3:  A[128x64] = 8192 bytes  |  B[256x64] = 16384 bytes  = 24576 bytes
                                                            Total = 98304 bytes
```

### 5.2 Swizzle Pattern

Use 128-byte swizzle for FP8 data to avoid bank conflicts during LDSM loads:
- For A tile (128x64, FP8): swizzle every 128 bytes along the K dimension
- For B tile (256x64, FP8): swizzle every 128 bytes along the K dimension
- The swizzle pattern XORs bits of the row index into the column address

The baseline uses LDSM.16.M88.4 which loads 128 bytes per warp (4 x 32 bytes).
Our kernel should match this access pattern.

## 6. MMA Instruction Selection

### QMMA.SF.16832.F32.E4M3.E4M3.E8

This is the Blackwell SM120 native FP8 tensor core instruction:
- Shape: M=16, N=8, K=32
- Inputs: FP8 E4M3 (A and B)
- Accumulator: FP32
- Scale factor: per-tensor (SF mode)
- E8 suffix: 8-element output grouping

Per warp, per K-tile iteration (K=64, so 2 K-steps of 32):
- 4 M-steps * 8 N-steps * 2 K-steps = 64 QMMA instructions per warp
- But the baseline uses 32 QMMAs per iteration with K=64, meaning it processes
  K=32 per QMMA and two A-tiles feed into one set of B-tiles

### Register Budget per Warp

Accumulator registers for 64x64 output tile:
- (64/16) * (64/8) = 4 * 8 = 32 accumulator groups
- Each group: 4 FP32 registers = 128 bytes
- Total accumulators: 32 * 4 = 128 registers

Input registers:
- A fragments: loaded via LDSM, ~16-32 registers
- B fragments: loaded via LDSM, ~32-64 registers
- Address/index registers: ~20 registers
- TMA descriptor registers: ~10 uniform registers

Total estimate: 128 (accum) + 50 (A/B fragments) + 30 (misc) = ~208 registers
This is under the 255 register limit, leaving headroom.

## 7. Epilogue Design

### 7.1 TMA Store Epilogue (Key Improvement)

Instead of the baseline's element-by-element STG.E.EF.U16 stores, use
**TMA store** (cp.async.bulk) for the epilogue:

1. Convert accumulators from FP32 to BF16 in registers (F2FP.BF16.F32.PACK)
2. Store BF16 values to shared memory
3. Issue TMA store from shared memory to global memory

This eliminates the uncoalesced store pattern (50% excessive sectors in baseline).

However, TMA stores on SM120 may have constraints. If TMA store is not available
or practical for the output layout, fall back to:

### 7.2 Vectorized STG Epilogue (Fallback)

- Pack 4 BF16 values into 64-bit register pairs
- Use STG.E.64 (64-bit stores) instead of STG.E.U16 (16-bit stores)
- This gives 4x fewer store transactions

## 8. Address Computation and Tiling

### CTA-to-Tile Mapping

For M=1024, N=10240 with 128x256 tiles:
- M_tiles = 8, N_tiles = 40
- Use 2D grid: gridDim.x = N_tiles = 40, gridDim.y = M_tiles = 8
- Or 1D grid with linearized tile index and row-major tile mapping

### Swizzled Tile Order

Use a swizzled tile mapping to improve L2 locality:
```
tile_idx = blockIdx.x
tile_m = (tile_idx / N_tiles) // could swap to tile_n first
tile_n = (tile_idx % N_tiles)
// Apply swizzle: partition N_tiles into groups of num_sm_tiles
```

## 9. Module Decomposition

| Module ID | Description | Estimated Runtime % |
|-----------|-------------|---------------------|
| `prologue` | Initialize barriers, fill pipeline stages 0-3 with TMA loads | 5% |
| `mainloop-tma` | Per-iteration TMA load dispatch (A and B tiles) | 10% |
| `mainloop-smem-load` | LDSM from shared memory to registers | 10% |
| `mainloop-mma` | QMMA tensor core computation | 55% |
| `mainloop-barrier` | mbarrier arrive/wait/fence synchronization | 5% |
| `epilogue-convert` | FP32 accumulator to BF16 conversion + scale | 5% |
| `epilogue-store` | Output store (TMA store or vectorized STG) | 10% |

## 10. Key PTX Instructions

| Operation | PTX/SASS Instruction | Purpose |
|-----------|---------------------|---------|
| TMA Load | `cp.async.bulk.tensor.2d` / `UTMALDG.3D` | Global->Shared async load |
| MMA | `mma.sync.aligned.m16n8k32.f32.e4m3.e4m3` / `QMMA.SF.16832` | FP8 tensor core |
| Barrier Init | `mbarrier.init` / `SYNCS.EXCH.64` | Initialize mbarrier |
| Barrier Arrive | `mbarrier.arrive.expect_tx` / `SYNCS.ARRIVE.TRANS64.RED` | Signal TMA completion |
| Barrier Wait | `mbarrier.try_wait.parity` / `SYNCS.PHASECHK.TRANS64.TRYWAIT` | Wait for data |
| Smem Load | `ldmatrix.sync.aligned.m8n8.x4` / `LDSM.16.M88.4` | Shared->Register for MMA |
| Convert | `cvt.rn.bf16x2.f32` / `F2FP.BF16.F32.PACK_AB` | FP32->BF16 conversion |
| Fence | `fence.proxy.async` | Ensure async operations visible |
| Block Sync | `bar.sync` / `BAR.SYNC` | CTA-wide synchronization |

## 11. Performance Ceiling

### Theoretical Minimum Latency

FLOPS = 2 * 1024 * 10240 * 4096 = 85,899,345,920 FLOPS
At 418 TFLOPS peak: 85.9e9 / 418e12 = 205.5 us

### Memory Transfer

Data read: A = 1024*4096*1 = 4 MB, B = 10240*4096*1 = 40 MB, total = 44 MB
Data write: C = 1024*10240*2 = 20 MB
Total: 64 MB

At L2 bandwidth (~4.51 TB/s from NCU): 64 MB / 4.51 TB/s = 14.2 us
At DRAM bandwidth (~207 GB/s): 64 MB / 207 GB/s = 309 us
L2 hit rate is 95.14%, so effective bandwidth is very high for this shape.

### Expected Achievement

With 128x256 tiles (1.45 waves vs 5.82 waves):
- Wave efficiency improvement: ~8% reduction in tail waste
- Expected latency: ~213-218 us
- Expected TFLOPS: ~394-403 TFLOPS
- Expected roofline efficiency: ~94-96%

Target: **>= 90% roofline efficiency** (>= 376 TFLOPS, <= 228 us)

The main improvement vector is the tile shape change reducing wave quantization.
Secondary improvements from TMA stores and tighter epilogue should provide
additional 2-3% gains.
