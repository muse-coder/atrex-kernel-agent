"""Baseline FP8 GEMM using torch._scaled_mm (cuBLAS)."""
import torch


def fp8_gemm_baseline(
    A: torch.Tensor,       # (M, K) float8_e4m3fn
    B: torch.Tensor,       # (N, K) float8_e4m3fn
    scale_a: torch.Tensor, # scalar float32
    scale_b: torch.Tensor, # scalar float32
    C: torch.Tensor,       # (M, N) bfloat16, output
) -> None:
    result = torch._scaled_mm(
        A, B.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=C.dtype,
    )
    C.copy_(result)
