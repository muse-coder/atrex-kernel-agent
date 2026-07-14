# nvidia/blackwell/optimization

Cross-feature optimization methods and symptom-driven diagnosis. Feature-specific
strategy belongs in [`../features/`](../features/).

## Files

| File | Description |
|------|------|
| [cache-policy.md](cache-policy.md) | Cache Policy Differentiation |
| [chunk-parallelism.md](chunk-parallelism.md) | Chunk-Based Parallelism |
| [compute-bound.md](compute-bound.md) | Diagnosis: not reaching peak FLOPS |
| [cuda-gemm-optimization-ladder.md](cuda-gemm-optimization-ladder.md) | CUDA GEMM Optimization Ladder |
| [double-buffering.md](double-buffering.md) | Double/Multi-Buffering Patterns |
| [epilogue-fusion.md](epilogue-fusion.md) | Epilogue Fusion |
| [kernel-fusion.md](kernel-fusion.md) | Kernel Fusion |
| [low-sm-utilization.md](low-sm-utilization.md) | Diagnosis: low SM utilization |
| [memory-bound.md](memory-bound.md) | Diagnosis: memory bandwidth bound |
| [moe-load-imbalance.md](moe-load-imbalance.md) | Diagnosis: MoE expert load imbalance |
| [ping-pong-scheduling.md](ping-pong-scheduling.md) | Ping-Pong Scheduling |
| [pipeline-stages.md](pipeline-stages.md) | Software Pipelining and Multi-Stage Buffering |
| [pipeline-stalls.md](pipeline-stalls.md) | Diagnosis: pipeline stalls |
| [register-budgeting.md](register-budgeting.md) | Register Budgeting |
| [register-pressure.md](register-pressure.md) | Diagnosis: register pressure and low occupancy |
| [software-exp.md](software-exp.md) | Software-Emulated Exponential |
| [swizzling.md](swizzling.md) | Nsight Compute command to check shared memory bank conflicts |
| [tail-effect.md](tail-effect.md) | Diagnosis: last-wave underutilization |
| [tile-scheduling.md](tile-scheduling.md) | Work allocation versus software tile mapping on Blackwell |
| [vectorized-loads.md](vectorized-loads.md) | nvcc compilation with register budgeting |
| [warp-specialization.md](warp-specialization.md) | Warp Specialization on Blackwell |
