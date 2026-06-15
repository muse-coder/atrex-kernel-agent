"""Benchmark FlashInfer BF16 and FP8 GEMM on GB200.

Shapes: (M, N, K) with N=10240, K=4096, M in {16384, 8192, 3584, 1024}.
- BF16 GEMM:  input(M,K) bf16 × weight(K,N) bf16 → output(M,N) bf16
- FP8 GEMM:   input(M,K) bf16 → internally cast to fp8_e4m3 × weight(N,K) fp8_e4m3 → output(M,N) bf16
"""

import torch
import time
import json
import sys

SHAPES = [
    (16384, 10240, 4096),
    (8192,  10240, 4096),
    (3584,  10240, 4096),
    (1024,  10240, 4096),
]

WARMUP = 50
ITERS = 200

def bench_bf16_torch(M, N, K, device="cuda"):
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    for _ in range(WARMUP):
        torch.mm(A, B)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / ITERS
    flops = 2.0 * M * N * K
    tflops = flops / elapsed / 1e12
    return elapsed * 1e6, tflops  # us, TFLOPS

def bench_fp8_flashinfer(M, N, K, device="cuda"):
    try:
        from flashinfer.gemm import bmm_fp8
    except ImportError:
        pass

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    scale_a = torch.ones(1, dtype=torch.float32, device=device)
    scale_b = torch.ones(1, dtype=torch.float32, device=device)

    # FlashInfer's fp8 GEMM: try segmented or cutlass-based
    # First try the cutlass fp8 gemm
    try:
        from flashinfer.gemm import cutlass_segment_gemm
        # segment_gemm expects batched layout; use single segment for plain GEMM
        seg_indptr = torch.tensor([0, M], dtype=torch.int64, device=device)
        weight_indices = torch.tensor([0], dtype=torch.int64, device=device)

        # For FP8 segment gemm: input is fp8, weight is fp8
        A_fp8 = A_bf16.to(torch.float8_e4m3fn)

        for _ in range(WARMUP):
            cutlass_segment_gemm(A_fp8, W_fp8.unsqueeze(0), seg_indptr, weight_indices, scale_a, scale_b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(ITERS):
            cutlass_segment_gemm(A_fp8, W_fp8.unsqueeze(0), seg_indptr, weight_indices, scale_a, scale_b)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / ITERS
        flops = 2.0 * M * N * K
        tflops = flops / elapsed / 1e12
        return elapsed * 1e6, tflops, "cutlass_segment_gemm"
    except Exception as e:
        print(f"  cutlass_segment_gemm failed: {e}", file=sys.stderr)

    # Fallback: try torch._scaled_mm (uses cuBLAS FP8)
    try:
        A_fp8 = A_bf16.to(torch.float8_e4m3fn)
        # _scaled_mm: (M,K) fp8 × (K,N) fp8 with per-tensor scales
        W_t = W_fp8.t().contiguous()  # (K, N)
        for _ in range(WARMUP):
            torch._scaled_mm(A_fp8, W_t, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(ITERS):
            torch._scaled_mm(A_fp8, W_t, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / ITERS
        flops = 2.0 * M * N * K
        tflops = flops / elapsed / 1e12
        return elapsed * 1e6, tflops, "torch._scaled_mm"
    except Exception as e:
        print(f"  torch._scaled_mm failed: {e}", file=sys.stderr)
        return None, None, "FAILED"

def bench_fp8_torch_scaled_mm(M, N, K, device="cuda"):
    """Direct torch._scaled_mm benchmark as reference."""
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    A_fp8 = A_bf16.to(torch.float8_e4m3fn)
    W_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    W_t = W_fp8.t().contiguous()  # (K, N)
    scale_a = torch.ones(1, dtype=torch.float32, device=device)
    scale_b = torch.ones(1, dtype=torch.float32, device=device)

    for _ in range(WARMUP):
        torch._scaled_mm(A_fp8, W_t, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        torch._scaled_mm(A_fp8, W_t, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / ITERS
    flops = 2.0 * M * N * K
    tflops = flops / elapsed / 1e12
    return elapsed * 1e6, tflops

def main():
    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FlashInfer version: ", end="")
    try:
        import flashinfer
        print(flashinfer.__version__)
    except:
        print("N/A")
    print()

    # Check available FlashInfer GEMM APIs
    print("=== Available FlashInfer GEMM APIs ===")
    try:
        import flashinfer.gemm as fg
        apis = [x for x in dir(fg) if not x.startswith('_')]
        print(f"  flashinfer.gemm: {apis}")
    except Exception as e:
        print(f"  flashinfer.gemm import error: {e}")
    print()

    results = []

    print("=" * 90)
    print(f"{'Shape (M,N,K)':<22} {'BF16 (us)':>10} {'BF16 TFLOPS':>12} {'FP8 (us)':>10} {'FP8 TFLOPS':>12} {'FP8 Method':>22}")
    print("-" * 90)

    for M, N, K in SHAPES:
        bf16_us, bf16_tflops = bench_bf16_torch(M, N, K)
        fp8_us, fp8_tflops, fp8_method = bench_fp8_flashinfer(M, N, K)
        scaled_mm_us, scaled_mm_tflops = bench_fp8_torch_scaled_mm(M, N, K)

        fp8_str_us = f"{fp8_us:.1f}" if fp8_us else "FAIL"
        fp8_str_tf = f"{fp8_tflops:.1f}" if fp8_tflops else "FAIL"

        print(f"({M:>5},{N:>5},{K:>4})  {bf16_us:>10.1f} {bf16_tflops:>12.1f} {fp8_str_us:>10} {fp8_str_tf:>12} {fp8_method:>22}")
        print(f"{'':>22} {'scaled_mm:':>10} {scaled_mm_us:>10.1f}us {scaled_mm_tflops:>10.1f} TFLOPS")

        results.append({
            "M": M, "N": N, "K": K,
            "bf16_us": round(bf16_us, 1),
            "bf16_tflops": round(bf16_tflops, 1),
            "fp8_us": round(fp8_us, 1) if fp8_us else None,
            "fp8_tflops": round(fp8_tflops, 1) if fp8_tflops else None,
            "fp8_method": fp8_method,
            "scaled_mm_us": round(scaled_mm_us, 1),
            "scaled_mm_tflops": round(scaled_mm_tflops, 1),
        })

    print("=" * 90)

    # GB200 theoretical peaks (per GPU)
    # BF16 Tensor Core: ~2250 TFLOPS (with sparsity), ~1125 TFLOPS (dense)
    # FP8 Tensor Core: ~4500 TFLOPS (with sparsity), ~2250 TFLOPS (dense)
    print("\n=== Roofline Reference (GB200 per-GPU, dense) ===")
    print("  BF16 Tensor Core peak: ~1125 TFLOPS")
    print("  FP8  Tensor Core peak: ~2250 TFLOPS")
    print("  HBM bandwidth: ~8 TB/s")
    print()

    for r in results:
        M, N, K = r["M"], r["N"], r["K"]
        bf16_eff = r["bf16_tflops"] / 1125.0 * 100
        fp8_eff = (r["fp8_tflops"] / 2250.0 * 100) if r["fp8_tflops"] else 0
        smm_eff = r["scaled_mm_tflops"] / 2250.0 * 100
        print(f"  ({M:>5},{N:>5},{K:>4})  BF16 eff: {bf16_eff:>5.1f}%  FP8 eff: {fp8_eff:>5.1f}%  scaled_mm eff: {smm_eff:>5.1f}%")

    # Save JSON
    with open("/home/admin/gemm_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /home/admin/gemm_baseline_results.json")

if __name__ == "__main__":
    main()
