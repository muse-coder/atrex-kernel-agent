# Baseline Analysis: cuBLAS FP8 GEMM on RTX PRO 5000

## Kernel Identified

- **Name**: `nvjet_sm120_qqtst_mma_64x128x64_8_32x64x64_tmaAB_bz_TNNN`
- **Source**: cuBLAS JIT (NVRTC) via `torch._scaled_mm`
- **Shape**: M=1024, N=10240, K=4096 (FP8 E4M3 x FP8 E4M3 -> BF16)
- **GPU**: NVIDIA RTX PRO 5000 72GB Blackwell (SM 12.0, 110 SMs)
- **Duration**: 223.17 us
- **Achieved TFLOPS**: 2*1024*10240*4096 / 223.17e-6 / 1e12 = 385.3 TFLOPS
- **Roofline Efficiency**: ~92.2% (vs ~418 TFLOPS peak FP8)

## 1. Primary Bound: Compute (Tensor Pipe)

NCU identifies this workload as **compute-bound** with the tensor pipe as the
dominant bottleneck:

| Metric | Value | Source |
|--------|-------|--------|
| Compute (SM) Throughput | 81.73% | Speed Of Light |
| Memory Throughput | 80.33% | Speed Of Light |
| Tensor (FP) Pipeline | 81.73% | Compute Workload Analysis |
| SM Busy | 81.73% | Compute Workload Analysis |
| Pipe Tensor Cycles Active | 81.73% | Throughput Breakdown |

The tensor pipe is the highest-utilized pipeline. NCU Rule
"HighPipeUtilization" confirms: "Tensor is the highest-utilized pipeline (81.7%)
based on elapsed cycles... It's dominated by its Tensor (FP) sub-pipeline. The
pipeline is over-utilized and likely a performance bottleneck."

The stall analysis confirms this: 47.8% of stall cycles are "stall_math" (waiting
for execution pipe to be available), which is characteristic of a compute-bound
kernel where warps spend most time waiting for tensor core operations to complete.

## 2. What the Baseline Does Well

### 2.1 Tile and MMA Selection
The kernel uses tile size **64x128xK** with QMMA.SF.16832 (16x8x32 shaped MMA
with FP8 E4M3 inputs, FP32 accumulation, and scale factor). The kernel name
encodes: `64x128x64` CTA tile, `8` pipeline stages, `32x64x64` per-warp tile.

- **QMMA.SF.16832.F32.E4M3.E4M3.E8**: Blackwell's native FP8 tensor core
  instruction with per-tensor scale factor (SF) support. The `.E8` suffix
  indicates 8-element accumulation mode.
- 32 QMMA instructions per mainloop iteration (16 for first A-tile x 8 B-tiles,
  16 for second A-tile x 8 B-tiles), providing excellent MMA density.

### 2.2 TMA-Based Loading
The kernel uses `UTMALDG.3D` (Uniform TMA Load Global, 3D descriptor) for
asynchronous global-to-shared memory transfers. This is the correct SM120
approach for bulk data movement.

### 2.3 Barrier Synchronization
Uses `SYNCS.EXCH.64` for mbarrier initialization, `SYNCS.ARRIVE.TRANS64.RED` for
TMA arrival signaling, and `SYNCS.PHASECHK.TRANS64.TRYWAIT` for phase-based
waiting. This is the Blackwell mbarrier synchronization pattern with transaction
counting.

### 2.4 No Spills
- Local Memory Spilling: 0 bytes
- Shared Memory Spilling: 0 bytes
- Stack Size: 1024 bytes (minimal)

### 2.5 Warp Specialization
The kernel uses warp specialization with UP0 predicate guarding TMA loads and
mbarrier operations. Thread 0 within warp 0 (lane 0, warp 0) acts as the TMA
producer. The `@UP0` predicates on UTMALDG and SYNCS instructions confirm this
pattern — only one warp handles data loading while others compute.

### 2.6 Pipeline Depth
The 8-stage pipeline (`8` in the kernel name) with `SYNCS.EXCH.64` initialization
for 8 barriers (offsets 0x0 through 0x38) provides deep prefetching to hide
global memory latency.

## 3. What the Baseline Does Poorly

### 3.1 Wave Quantization / Tail Effect
- Grid: 1280 blocks (1D), 110 SMs
- Waves per SM: 5.82 = 1280 / 110 / 2 (2 blocks/SM)
- With 2 blocks per SM, effective waves = 1280 / (110*2) = 5.82
- Last wave utilization: 0.82 * 110 * 2 = ~180 blocks active out of 220 slots
- **Tile decomposition**: M_tiles = 1024/64 = 16, N_tiles = 10240/128 = 80,
  total = 1280 tiles. This creates 5.82 waves with significant tail waste.

### 3.2 Very Low Occupancy
- Theoretical Occupancy: **16.67%** (2 warps per scheduler, 8 warps per SM)
- Achieved Occupancy: **16.30%**
- Block limit: **2 blocks/SM** (limited by both registers=255 AND shared memory=50.18KB)
- 128 threads/block = 4 warps/block, so 8 warps/SM total
- SM 12.0 has 48 warps max per SM, so only 8/48 = 16.67% utilized

This is inherent to large-tile GEMM kernels that maximize register file usage
for accumulator storage. 255 registers per thread is the hardware maximum.

### 3.3 Low Scheduler Utilization
- Eligible Warps Per Scheduler: 0.18 (out of 2.0 active)
- Issue Slots Busy: 15.53%
- No Eligible: 83.74% of cycles
- Warp Cycles Per Issued Instruction: 12.02 cycles

The scheduler is starved for eligible warps because:
1. Only 2 active warps per scheduler (low occupancy)
2. High-latency tensor core operations dominate
3. 47.8% of stall time is "stall_math" (waiting for tensor pipe)

### 3.4 Uncoalesced Global Memory Access
NCU reports: "50% excessive sectors" for global access patterns.
- Global loads: only 4/32 bytes utilized per sector (12.5% efficiency)
- Global stores: only 16/32 bytes utilized per sector (50% efficiency)
- L2 Sector Promotion Misses: 44.56%

The store pattern uses `STG.E.EF.U16` (16-bit BF16 stores) with complex address
calculations. The per-element F2FP conversion + store pattern generates many
small stores rather than bulk TMA stores.

### 3.5 Epilogue Inefficiency
The epilogue section is extremely long — approximately 400+ SASS instructions
devoted to:
1. FFMA for scale multiplication (R_i * scale)
2. F2FP.BF16.F32.PACK_AB for FP32->BF16 conversion
3. Individual STG.E.EF.U16 stores (16-bit per element)

The epilogue stores output in a highly serial, element-by-element fashion. Each
output element requires: FFMA (scale) + F2FP (convert) + STG (store), repeated
for all 16 accumulator groups x 4 elements each = 64 F2FP + 64 STG + 64 FFMA.

### 3.6 Barrier Stalls in Mainloop
The `BAR.SYNC.DEFER_BLOCKING 0x0` instruction at address 0x7f3e0976e3c0 shows
2356 stall samples — the highest-stalled instruction. This is the barrier
synchronization between TMA load completion and MMA consumption. The TMA producer
warp must complete loading before consumer warps can begin computing.

### 3.7 NANOSLEEP Spin-Wait
The kernel uses `NANOSLEEP.RAND.WARP.SYNCS` (532+619 stall samples in the wait
loop) for mbarrier waiting. This burns cycles in a spin-wait loop with random
backoff, suggesting the TMA loads are not always ready when the MMA pipeline
needs data.

## 4. Resource Utilization

| Resource | Value | Limit | Notes |
|----------|-------|-------|-------|
| Registers/Thread | 255 | 255 | Maximum — occupancy limiter |
| Shared Memory/Block | 50.18 KB (dynamic) | 102.4 KB config | ~49% of config |
| Block Size | 128 threads (64x2) | - | 4 warps |
| Blocks/SM | 2 | 24 max | Limited by regs + smem |
| Occupancy | 16.67% theoretical | 100% | 8/48 warps |
| Achieved Occupancy | 16.30% | 16.67% | 97.8% of theoretical |

## 5. Pipeline Structure

- **8 pipeline stages** (from kernel name suffix `_8_`)
- Prologue: 8x SYNCS.EXCH.64 to initialize barriers, then fill pipeline with
  first batch of TMA loads
- Mainloop: Each iteration loads A-tile and B-tile via UTMALDG, waits on
  barrier, loads from shared memory via LDSM, computes 32 QMMA instructions,
  signals next-stage barrier
- Epilogue: Scale accumulator, convert FP32->BF16, store via STG

### Mainloop Detail (from SASS)
Per iteration:
1. UIMAD/UIADD3 to compute TMA descriptor offsets
2. UTMALDG.3D for A-tile load, UTMALDG.3D for B-tile load
3. BAR.SYNC.DEFER_BLOCKING to wait for TMA completion
4. 12x LDSM.16.M88.4 to load A/B tiles from smem to registers
5. BAR.SYNC again
6. 32x QMMA.SF.16832.F32.E4M3.E4M3.E8 for the MMA computation
7. Barrier arrive/wait for next stage
8. Update K-loop counter, branch back

## 6. SASS Analysis Summary

### Instruction Mix (Mainloop)
- **QMMA**: 32 per iteration, dominant compute instruction (~80% of active cycles)
- **LDSM.16.M88.4**: 12 per iteration (shared memory to register loads)
- **UTMALDG.3D**: 2 per iteration (TMA global to shared)
- **BAR.SYNC**: 2 per iteration (barrier synchronization)
- **SYNCS.ARRIVE/PHASECHK**: mbarrier signaling
- **UIMAD/UIADD3**: Uniform register address arithmetic

### Key Observations
1. The QMMA instructions show consistent stall_math stalls (~300 samples each),
   confirming the tensor pipe is saturated
2. The first QMMA after barrier sync shows very high stall (2186 samples) — this
   is the pipeline startup latency after a barrier
3. LDSM instructions show stall_mio stalls, indicating shared memory access
   latency between TMA completion and register load
4. The epilogue is disproportionately long for a compute-bound kernel

### Accumulator Register Layout
The kernel uses R8-R71 (64 registers) for FP32 accumulators, organized as:
- R8-R11, R12-R15, ..., R68-R71 = 16 groups of 4 registers
- Each group holds one 16x8 accumulator tile
- Total: 16 tiles * 4 regs * 32 bits = 2048 bits of accumulator per thread

## 7. Opportunities Summary

| Opportunity | Potential Impact | Confidence |
|-------------|-----------------|------------|
| Better tile shape (128x256 or 64x256) | 5-10% | High — fewer waves, better utilization |
| Swap A/B for M << N | 3-5% | Medium — may improve tile coverage |
| TMA-based epilogue store | 3-7% | High — eliminate uncoalesced stores |
| Reduce register pressure to <255 | 2-4% | Medium — may enable higher occupancy |
| Persistent kernel / Stream-K | 5-10% | Medium — eliminates tail effect entirely |
| Tighter prologue/epilogue | 2-3% | Medium — reduce non-MMA overhead |
