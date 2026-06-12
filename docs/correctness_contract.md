# Correctness Contract

This document defines the correctness requirements that every kernel
optimization task must satisfy before benchmark results are considered valid.

## General Principles

1. **Correctness before performance.** No benchmark result counts until the
   candidate passes both the production workload correctness checks and the
   canonical regression grid.

2. **Poison before check.** Output buffers must be filled with NaN (floating
   point) or a sentinel value (integer) before every correctness call so that
   stale-output and skipped-kernel bugs are visible.

3. **NaN/Inf preservation.** Preserve explicit NaN/Inf checks in the
   correctness harness. If the baseline kernel never produces NaN/Inf on valid
   input, the candidate must not produce them either.

4. **Oracle comparison.** Compare candidate output against an independent
   PyTorch/math oracle when practical. If a full oracle is expensive, at
   minimum compare against the baseline plus targeted oracle rows.

## Defining a Regression Grid

Each task should define a regression grid in its `prompt.md` or a separate
`docs/correctness_contract.md` file. The grid specifies:

- **Shapes**: representative tensor dimensions covering typical and edge cases.
- **Dtypes**: all dtypes the kernel is expected to support (e.g. fp16, bf16,
  fp32).
- **Scalar parameters**: any configurable scalars (eps, num_groups, etc.).
- **Layout variants**: contiguous, channels-last, strided, etc.
- **Tolerances**: per-dtype absolute and relative tolerance.

Example:

```yaml
regression_grid:
  shapes:
    - [1, 64, 32, 32]
    - [4, 256, 16, 16]
    - [1, 128, 20, 256, 256]
  dtypes: [float16, bfloat16, float32]
  num_groups: [32]
  eps: [1e-5, 1e-6]
  tolerances:
    float16:  {atol: 3e-3, rtol: 3e-3}
    bfloat16: {atol: 7e-2, rtol: 2e-2}
    float32:  {atol: 1e-5, rtol: 1e-5}
```

## When to Update Tolerances

- Do not relax tolerances to make a failing candidate pass.
- If the task records a stricter task-local tolerance in
  `docs/benchmark_method.md`, use the stricter value.
- If evidence shows the baseline itself exceeds a grid tolerance on specific
  shapes, document it and adjust only that cell.
