# amd/common/profiling

AMD profiling and performance-modeling entry point. Start with the general
rocprofv3 workflow, use Roofline to classify the bound, then escalate to the
architecture-specific ATT guide only when instruction-level evidence is needed.

## Workflow

1. [rocprofv3.md](rocprofv3.md) — trace dispatches, collect counters, and filter the target kernel.
2. [roofline.md](roofline.md) — determine compute- versus memory-bound behavior and a realistic ceiling.
3. Use the matching ATT guide for source/ISA hotspots: [CDNA3 / gfx942](gfx942-gluon-att.md) or [CDNA4 / gfx950](gfx950-gluon-att.md).
4. [rocprof-trace-decoder.md](rocprof-trace-decoder.md) — install or use the ATT decoder library when it is not available from ROCm.

## Tools

| Tool | Role |
|------|------|
| `rocprofv3` | Kernel traces, hardware counters, and Advanced Thread Trace collection |
| ROCm Compute Profiler | Counter analysis and Roofline visualization |
| ROCm Systems Profiler | End-to-end application tracing |
| PyTorch Profiler | CPU/GPU timeline and Perfetto visualization |
| NPKit | Fine-grained RCCL kernel tracing |

## Files

| File | Description |
|------|------|
| [rocprofv3.md](rocprofv3.md) | General rocprofv3 tracing and counter collection |
| [roofline.md](roofline.md) | Tile-level Roofline and utilization methodology |
| [gfx942-gluon-att.md](gfx942-gluon-att.md) | CDNA3/Gluon instruction-level ATT workflow |
| [gfx950-gluon-att.md](gfx950-gluon-att.md) | CDNA4/Gluon ATT and counter workflow |
| [rocprof-trace-decoder.md](rocprof-trace-decoder.md) | Thread-trace decoder library usage |

## Related

- [NVIDIA profiling](../../../nvidia/common/profiling/README.md)
- [AMD common optimization](../README.md)
