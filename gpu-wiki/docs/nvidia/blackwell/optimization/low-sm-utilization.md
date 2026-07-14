# Low SM Utilization


**Last updated**: 2026-07-14

## Symptom

SM utilization below 60% despite sufficient occupancy. Nsight Compute shows idle SMs during portions of kernel execution.

## Likely Causes

1. **Tail effect**: Last wave of tiles leaves most SMs idle (see [tail-effect](tail-effect.md))
2. **Load imbalance**: Some tiles take longer than others (variable computation per tile)
3. **Static scheduling**: Fixed tile-to-SM assignment doesn't adapt to runtime conditions
4. **Grid too small**: Fewer threadblocks than SMs

## Candidate Techniques

| Technique | Applicability | Effect |
|---|---|---|
| [CLC](../features/clc.md) | `sm_100` or higher | Faster clusters can take over pending cluster coordinates |
| [CLC](../features/clc.md) | `sm_100` or higher | Dynamic persistent scheduling when pending clusters remain |
| [Tile scheduling](tile-scheduling.md) | SM90+ | Better L2 locality, reduce load variance |

## Examples

```
// Local Gluon sample, GB200, 8192^3 GEMM:
// Static: 1040.13 TFLOPS
// CLC:    1080.74 TFLOPS (+3.9% in that run)
```

## Caveats
- CLC cannot help if there is no pending cluster launch to cancel
- Persistent kernels complicate debugging and profiling
- For non-persistent kernels, ensure grid size >> SM count


## Related

- [Not Reaching Peak FLOPS](compute-bound.md)
- [Memory Bandwidth Bound](memory-bound.md)
- [MoE Expert Load Imbalance](moe-load-imbalance.md)
- [Pipeline Stalls](pipeline-stalls.md)
- [Register Pressure -- Low Occupancy](register-pressure.md)
- [Occupancy Optimization](../../../amd/common/occupancy-optimization.md)
- [Occupancy Tuning Differences Across Architectures](../../common/occupancy-tuning-by-arch.md)
- [Composable Kernel (CK) Architecture Overview](../../../amd/common/ck-architecture-overview.md)
