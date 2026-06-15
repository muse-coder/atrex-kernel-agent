"""Benchmark BF16 GEMM vs FP8 GEMM across shapes to find worst performer."""
import torch
import torch.utils.benchmark as benchmark
import json

SHAPES = [
    (16384, 4096, 10240),
    (8192, 4096, 10240),
    (3584, 4096, 10240),
    (1024, 4096, 10240),
]

WARMUP = 10
NUM_RUNS = 100

def bench_bf16_gemm(M, K, N, device="cuda"):
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(N, K, dtype=torch.bfloat16, device=device)  # (N, K) so we do A @ B.T

    # Warmup
    for _ in range(WARMUP):
        torch.mm(A, B.t())
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    times = []
    for _ in range(NUM_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(A, B.t())
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    times.sort()
    median_ms = times[len(times) // 2]
    flops = 2.0 * M * N * K
    tflops = flops / (median_ms * 1e-3) / 1e12
    return median_ms, tflops, times


def bench_fp8_gemm(M, K, N, device="cuda"):
    # FP8: input BF16 -> cast to FP8 E4M3, weight FP8 E4M3
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    A_fp8 = A_bf16.to(torch.float8_e4m3fn)
    B_fp8 = B_bf16.to(torch.float8_e4m3fn)

    scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
    scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(WARMUP):
        torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median_ms = times[len(times) // 2]
    flops = 2.0 * M * N * K
    tflops = flops / (median_ms * 1e-3) / 1e12
    return median_ms, tflops, times


def get_peak_tflops(gpu_name):
    """Approximate peak TFLOPS for known GPUs."""
    if "PRO 5000" in gpu_name or "5000" in gpu_name:
        # RTX PRO 5000 Blackwell - approximate peaks
        return {
            "bf16": 209.2,  # BF16 Tensor Core
            "fp8": 418.4,   # FP8 Tensor Core (2x BF16)
        }
    return {"bf16": 200.0, "fp8": 400.0}


if __name__ == "__main__":
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    peaks = get_peak_tflops(gpu_name)

    results = []

    print("=" * 100)
    print(f"{'Shape (M,K,N)':<25} {'Type':<8} {'Median(ms)':<12} {'TFLOPS':<10} {'Peak%':<10} {'Efficiency'}")
    print("=" * 100)

    for M, K, N in SHAPES:
        # BF16
        bf16_ms, bf16_tflops, bf16_times = bench_bf16_gemm(M, K, N)
        bf16_eff = bf16_tflops / peaks["bf16"] * 100

        # FP8
        fp8_ms, fp8_tflops, fp8_times = bench_fp8_gemm(M, K, N)
        fp8_eff = fp8_tflops / peaks["fp8"] * 100

        print(f"({M:>5},{K:>4},{N:>5})      BF16    {bf16_ms:>9.3f}ms  {bf16_tflops:>7.1f}    {bf16_eff:>6.1f}%")
        print(f"{'':25} FP8     {fp8_ms:>9.3f}ms  {fp8_tflops:>7.1f}    {fp8_eff:>6.1f}%")
        print(f"{'':25} FP8/BF16 speedup: {bf16_ms/fp8_ms:.2f}x")
        print("-" * 100)

        results.append({
            "M": M, "K": K, "N": N,
            "bf16_median_ms": bf16_ms,
            "bf16_tflops": bf16_tflops,
            "bf16_efficiency": bf16_eff,
            "fp8_median_ms": fp8_ms,
            "fp8_tflops": fp8_tflops,
            "fp8_efficiency": fp8_eff,
            "fp8_speedup_over_bf16": bf16_ms / fp8_ms,
        })

    # Find worst efficiency
    print("\n" + "=" * 100)
    print("EFFICIENCY RANKING (lower = worse = more optimization opportunity)")
    print("=" * 100)

    all_entries = []
    for r in results:
        all_entries.append(("BF16", r["M"], r["K"], r["N"], r["bf16_efficiency"], r["bf16_tflops"], r["bf16_median_ms"]))
        all_entries.append(("FP8", r["M"], r["K"], r["N"], r["fp8_efficiency"], r["fp8_tflops"], r["fp8_median_ms"]))

    all_entries.sort(key=lambda x: x[4])

    for i, (dtype, M, K, N, eff, tflops, ms) in enumerate(all_entries):
        marker = " <-- WORST" if i == 0 else ""
        print(f"  {dtype:<5} ({M:>5},{K:>4},{N:>5})  {eff:>6.1f}%  {tflops:>7.1f} TFLOPS  {ms:>9.3f}ms{marker}")

    # Save results
    with open("/home/moudi.mou/agent/IterKernel/bench_shapes_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to bench_shapes_results.json")
