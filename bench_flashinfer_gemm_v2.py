"""FlashInfer FP8/BF16 GEMM kernels for ncu profiling on GB200.

Usage:
  # Warmup run (no profiling)
  python bench_flashinfer_gemm_v2.py --warmup

  # Profile a specific kernel+shape with ncu:
  ncu --set full --kernel-id ::regex:<kernel_filter>: \
      python bench_flashinfer_gemm_v2.py --kernel <name> --M <M>

  # Or profile all in one shot (long):
  ncu --set full -o gemm_profile python bench_flashinfer_gemm_v2.py
"""

import torch
import math
import argparse
import sys

SHAPES = [
    (16384, 10240, 4096),
    (8192,  10240, 4096),
    (3584,  10240, 4096),
    (1024,  10240, 4096),
]


def run_torch_mm_bf16(M, N, K):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    out = torch.mm(A, B)
    torch.cuda.synchronize()
    return out


def run_fi_mm_bf16_cudnn(M, N, K):
    from flashinfer.gemm import mm_bf16
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    out = mm_bf16(A, B, backend="cudnn")
    torch.cuda.synchronize()
    return out


def run_torch_scaled_mm_fp8(M, N, K):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn).t().contiguous()
    sa = torch.ones(1, dtype=torch.float32, device="cuda")
    sb = torch.ones(1, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    out = torch._scaled_mm(A, W, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    return out


def run_fi_fp8_blockscaled(M, N, K):
    from flashinfer.gemm import gemm_fp8_nt_blockscaled
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    block = 128
    sa = torch.ones(math.ceil(M / block), math.ceil(K / block), dtype=torch.float32, device="cuda")
    sb = torch.ones(math.ceil(N / block), math.ceil(K / block), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    out = gemm_fp8_nt_blockscaled(A, W, sa, sb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    return out


KERNELS = {
    "torch_mm_bf16":      run_torch_mm_bf16,
    "fi_cudnn_bf16":      run_fi_mm_bf16_cudnn,
    "torch_scaled_mm":    run_torch_scaled_mm_fp8,
    "fi_fp8_blockscaled": run_fi_fp8_blockscaled,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, default=None,
                        help=f"Which kernel to run. Choices: {list(KERNELS.keys())}. Default: all")
    parser.add_argument("--M", type=int, default=None,
                        help="Specific M to test. Default: all shapes")
    parser.add_argument("--warmup", action="store_true",
                        help="Run warmup only (no profiling target)")
    parser.add_argument("--iters", type=int, default=1,
                        help="Number of iterations per kernel (default 1 for ncu)")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)

    shapes = SHAPES
    if args.M is not None:
        shapes = [(args.M, 10240, 4096)]

    kernels = KERNELS
    if args.kernel is not None:
        if args.kernel not in KERNELS:
            print(f"Unknown kernel: {args.kernel}. Choose from {list(KERNELS.keys())}", file=sys.stderr)
            sys.exit(1)
        kernels = {args.kernel: KERNELS[args.kernel]}

    if args.warmup:
        print("Warmup run...", file=sys.stderr)
        for name, fn in kernels.items():
            for M, N, K in shapes:
                try:
                    fn(M, N, K)
                    print(f"  {name} M={M}: OK", file=sys.stderr)
                except Exception as e:
                    print(f"  {name} M={M}: FAILED - {e}", file=sys.stderr)
        print("Warmup done.", file=sys.stderr)
        return

    # Warmup phase (not captured by ncu if using --kernel-id to skip)
    for name, fn in kernels.items():
        for M, N, K in shapes:
            try:
                fn(M, N, K)
            except Exception:
                pass
    torch.cuda.synchronize()

    # Profiling phase - use cudaProfilerApi to bracket
    torch.cuda.cudart().cudaProfilerStart()

    for name, fn in kernels.items():
        for M, N, K in shapes:
            print(f">>> {name} M={M} N={N} K={K}", file=sys.stderr)
            for _ in range(args.iters):
                try:
                    fn(M, N, K)
                except Exception as e:
                    print(f"  FAILED: {e}", file=sys.stderr)
            torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
