# NVFP4 and Block-Scaled Narrow Precision


**Last updated**: 2026-06-30

## Overview

NVFP4 is NVIDIA's 4-bit floating-point format (E2M1) with block scaling, native to Blackwell tensor cores.

## Format Details

```
E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
Representable values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

Block scaling: every 16 FP4 elements share one FP8 E4M3 scale factor
Two-level: per-block E4M3 scale × per-tensor FP32 global scale

Quantization:   q_i = cast_FP4(x_i / (s_global * s_block))
Dequantization: x_hat_i = s_global * s_block * deq_FP4(q_i)
```

## tcgen05 Variants for FP4

| Variant | Description | Throughput vs Hopper |
|---|---|---|
| `tcgen05.mma.mxf4.block_scale` | MX FP4 with block scaling | **4×** |
| `tcgen05.mma.mxf4nvf4.block_scale` | NVFP4 + MX FP4 flexible scaling | **4×** |

## PTX for FP4 Conversion

```ptx
// Convert two FP4 values to two FP16 values
cvt.rn.f16x2.e2m1x2 result, packed_fp4;

// Byte unpacking (faster than bitwise extraction)
mov.b32 {tmp0, tmp1, tmp2, tmp3}, packed_data;
```

## NVFP4 vs MXFP4

| Aspect | NVFP4 | MXFP4 |
|---|---|---|
| Scale format | E4M3 (fractional) | UE8M0 (power-of-2 only) |
| Block size | 16 elements | 32 elements |
| Scale precision | Non-power-of-2 | Power-of-2 only |
| Quantization error | Lower | Higher |

## Block-Scaled MMA Implementation

Block-scaled tcgen05 paths load data and scale factors separately. MXF8/MXFP8
uses an E8M0 scale per 32 elements; NVFP4 uses an FP8 scale per 16 elements.
The hardware applies those scales during MMA, but the kernel must still stage
the scale tensors with layouts compatible with the selected MMA atom.

```python
tiled_mma = cute.make_tiled_mma(
    cute.SM100_MMA_F32MXF8MXF8F32_SS_TN,
)

# Data and scale tiles have independent TMA loads.
cute.copy(tma_data_a, data_a_gmem, data_a_smem)
cute.copy(tma_scale_a, scale_a_gmem, scale_a_smem)
acc = cute.gemm(tiled_mma, data_a_smem, scale_a_smem, acc)
```

Choose the format from the numerical budget and operator support, then measure
the full path: scale loads and quantization/dequantization can dominate a small
GEMM even when narrow-precision MMA throughput is high.

## Related
- [Fine-Grained FP8/FP4 Quantization](fine-grained-quantization.md) -- Scaling strategies
- [nvfp4-gemm](../kernels/nvfp4-gemm.md) -- NVFP4 GEMM kernel
- [nvfp4-gemv](../kernels/nvfp4-gemv.md) -- NVFP4 GEMV kernel
