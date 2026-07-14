# Programmatic Dependent Launch / Grid Dependency Control


**Last updated**: 2026-06-30

## Overview

PDL/GDC allows overlapping execution of dependent kernel launches. The primary kernel signals it is finishing; the secondary kernel begins before the primary fully completes.

## How It Works

```cuda
// Primary kernel signals near completion
cudaGridDependencySynchronize();  // or PTX equivalent

// Secondary kernel can start overlapping with primary's tail
// Enabled by default on SM100 (opt-in on SM90)
```

## Blackwell Default Behavior

On SM100, PDL is **enabled by default** -- no opt-in needed. This means:
- Back-to-back kernel launches naturally overlap
- Memory fences ensure correctness for dependent data
- Reduces kernel launch gaps in compute-heavy pipelines

## When It Matters
- Chains of small kernels (e.g., MoE dispatch → compute → combine)
- Pipeline-parallel training with many sequential kernel launches
- Reduces overall wall-clock time without code changes on Blackwell

## Practical Launch Pattern

For a dependent pipeline such as GEMM → Softmax → GEMM, declare the dependency
at launch rather than synchronizing the host between kernels. The exact launch
API is framework-dependent; the following CuTeDSL-style pseudocode shows the
intent:

```python
cute.cluster_launch(
    softmax_kernel,
    grid,
    block,
    args=(qk_result, ...),
    dependency=current_kernel,
)
```

Use PDL to overlap the predecessor's tail with the successor's prologue. It is
separate from CLC: PDL coordinates dependent kernel launches, while CLC
redistributes work inside one grid.

## Related
- [Cluster Launch Control](clc.md) -- Dynamic persistent scheduling for work redistribution
