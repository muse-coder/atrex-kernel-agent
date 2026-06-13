"""FP8 GEMM candidate kernel using CUTLASS 3.x for SM120 (Blackwell).

Compiles the CUDA kernel via torch.utils.cpp_extension.load() on first import.
Subsequent imports use the cached compiled module.
"""
from __future__ import annotations

import os
import torch
from pathlib import Path

_MODULE = None
_SOLUTION_DIR = Path(__file__).resolve().parent
_CUTLASS_ROOT = Path(os.environ.get(
    "CUTLASS_ROOT",
    "/home/moudi.mou/opensource/cutlass"
))


def _get_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    from torch.utils.cpp_extension import load

    include_dirs = [
        str(_CUTLASS_ROOT / "include"),
        str(_CUTLASS_ROOT / "tools" / "util" / "include"),
    ]

    _MODULE = load(
        name="fp8_gemm_sm120",
        sources=[str(_SOLUTION_DIR / "fp8_gemm_kernel.cu")],
        extra_include_paths=include_dirs,
        extra_cuda_cflags=[
            "-std=c++17",
            "-arch=sm_120a",
            "-O3",
            "--expt-relaxed-constexpr",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )
    return _MODULE


def fp8_gemm_candidate(
    A: torch.Tensor,       # (M, K) float8_e4m3fn
    B: torch.Tensor,       # (N, K) float8_e4m3fn
    scale_a: torch.Tensor, # scalar float32
    scale_b: torch.Tensor, # scalar float32
    C: torch.Tensor,       # (M, N) bfloat16, output
) -> None:
    """FP8 GEMM: C = (scale_a * scale_b) * A @ B^T"""
    mod = _get_module()
    mod.fp8_gemm(A, B, scale_a, scale_b, C)
