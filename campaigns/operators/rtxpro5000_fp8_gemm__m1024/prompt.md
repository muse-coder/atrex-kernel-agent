# rtxpro5000_fp8_gemm__m1024

Target GPU: NVIDIA RTX PRO 5000 72GB Blackwell (110 SMs, SM 12.0).

Baseline kernel entry point(s):

- `torch._scaled_mm` (PyTorch cuBLAS FP8 GEMM)

Goal: optimize FP8 GEMM (E4M3 × E4M3 → BF16) for the production shape
M=1024, N=10240, K=4096 on RTX PRO 5000 Blackwell. This shape showed the
worst roofline efficiency (88.1%, 368.6 TFLOPS) among 4 candidate shapes
benchmarked at both BF16 and FP8 precision.

## Context

These shapes come from a transformer model's linear layers:
- input: (M, K) BF16 → cast to FP8 E4M3
- weight: (N, K) FP8 E4M3
- output: (M, N) BF16
- Per-tensor scaling (scale_a=1.0, scale_b=1.0)

## Benchmark Shapes

Primary (optimization target):
- M=1024, K=4096, N=10240 — 88.1% efficiency, 0.233ms

Regression (must not degrade):
- M=3584, K=4096, N=10240 — 99.8% efficiency, 0.720ms
- M=8192, K=4096, N=10240 — 105.4% efficiency, 1.558ms
- M=16384, K=4096, N=10240 — 92.6% efficiency, 3.547ms

## GPU Properties

- NVIDIA RTX PRO 5000 72GB Blackwell
- 110 SMs, compute capability 12.0
- FP8 peak: ~418 TFLOPS (estimated)
- BF16 peak: ~209 TFLOPS (estimated)

## Analysis of M=1024 Bottleneck

With M=1024, K=4096, N=10240:
- 2*M*N*K = 85.9 GFLOPS
- Theoretical min at 418 TFLOPS: 0.205ms
- Current: 0.233ms → 88.1% efficiency
- Tile 128×128: M_tiles=8, N_tiles=80, total=640 tiles, 5.8 waves on 110 SMs
  → last wave: 90/110 = 81.8% utilization
- Tile 128×256: M_tiles=8, N_tiles=40, total=320 tiles, 2.9 waves
  → last wave: 100/110 = 90.9% utilization

Likely bottlenecks:
1. Wave quantization / tail effect at small M
2. Suboptimal tile shape for this aspect ratio (M << N)
3. Possible swap-AB benefit (treating as N×K @ K×M)

Before writing an optimized kernel, read and follow:

- `../../docs/benchmark_contract.md`
- `../../docs/kernel_optimization_rules.md`
- `../../docs/correctness_contract.md`

Required first milestone:

1. Place the reference kernel implementation into `baseline/`.
2. Record the baseline's origin in `docs/baseline_source.md`.
3. Expose the baseline through local low-overhead ABI entry points.
4. Expose the candidate through the exact same ABI in `solution/`.
5. Create `bench/workloads.json`, copy the standard template to
   `bench/benchmark.py`, implement `bench/adapter.py`, and create
   `bench/correctness.py`.

All benchmark code must call only files in this task directory.
