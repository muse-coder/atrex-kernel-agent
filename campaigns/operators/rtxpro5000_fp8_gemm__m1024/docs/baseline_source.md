# Baseline Source

- **source**: PyTorch `torch._scaled_mm` (cuBLAS FP8 GEMM backend)
- **version**: PyTorch 2.10.0+cu130, CUDA 13.0
- **entry point**: `torch._scaled_mm(A, B.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=bfloat16)`
- **local modifications**: none, direct API call wrapped in destination-passing adapter
- **dtype**: FP8 E4M3 × FP8 E4M3 → BF16 with per-tensor FP32 scales
