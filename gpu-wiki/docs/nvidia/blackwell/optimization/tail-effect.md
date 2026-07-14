# Tail Effect -- Last Wave Underutilization


**Last updated**: 2026-07-14

## Symptom

Performance drops for problem sizes where total_tiles % num_SMs != 0. The last wave of tiles runs with many SMs idle.

## Likely Causes

1. **Wave quantization**: Grid of N tiles on M SMs takes ceil(N/M) waves; last wave may use only N%M SMs
2. **Static assignment**: stride-by-gridDim leaves remainder tiles on few SMs
3. **Non-persistent launch**: each kernel launch has fixed grid, no dynamic rebalancing

## Candidate Techniques

| Technique | Effect |
|---|---|
| [CLC](../features/clc.md) | Running clusters take over coordinates of not-yet-launched clusters |
| [CLC](../features/clc.md) | Dynamic persistent scheduling; compare with static persistent and Stream-K choices |
| [Tile scheduling](tile-scheduling.md) | Raster order, swizzle patterns for better distribution |

## Example

```
// Hypothetical resident capacity: 142 one-CTA clusters
// Grid: 150 one-CTA clusters
// Without dynamic redistribution, the final 8 clusters form a sparse wave.
// With CLC, earlier finishers can cancel pending launches and process their IDs.
```

## Caveats
- CLC helps only while the grid still contains not-yet-launched clusters
- For very large problems, tail effect is amortized across many waves
- CLC requires `sm_100` or higher; it cannot create parallelism when the entire grid is smaller than machine capacity


## Related

- [Not Reaching Peak FLOPS](compute-bound.md)
- [Low SM Utilization](low-sm-utilization.md)
- [Memory Bandwidth Bound](memory-bound.md)
- [MoE Expert Load Imbalance](moe-load-imbalance.md)
- [Pipeline Stalls](pipeline-stalls.md)
- [CK Tile Quantized GEMM and MX Format](../../../amd/common/ck-quantization-mx.md)
