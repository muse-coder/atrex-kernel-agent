"""Python wrapper for FP8 GEMM CUDA kernel. JIT compiles on first import."""
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_dir = Path(__file__).resolve().parent
_ext = None


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext
    _ext = load(
        name="fp8_gemm_ext",
        sources=[str(_dir / "fp8_gemm_kernel.cu")],
        extra_cuda_cflags=[
            "-arch=sm_120",
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        verbose=False,
    )
    return _ext


def fp8_gemm_candidate(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    C: torch.Tensor,
) -> None:
    ext = _get_ext()
    ext.fp8_gemm_launch(A, B, scale_a.item(), scale_b.item(), C)
