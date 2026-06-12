# Standalone Benchmark Contract

This repository benchmarks kernel optimizations using local, self-contained
harnesses. Both baseline and candidate run from local source inside the task
directory.

## Hard Rules

- Baseline and candidate must expose matching local entry points. Any wrapper
  overhead included for one side must be included for the other side.
- Every task must contain two local implementations: reference implementation in
  `baseline/` and the optimized implementation in `solution/`.
- Prefer local direct CUDA ABI for both sides: output tensors passed last,
  `destination_passing_style = true`.
- Every CUDA launch must use PyTorch's current stream:
  `at::cuda::getCurrentCUDAStream()`.
- If the baseline is CUDA/C++ CUDA, expose baseline and candidate through the
  same local registration/export/build style.
- Do not pass `--use_fast_math` unless the baseline already uses it and the
  candidate uses the exact same flag.
- If the baseline is Triton, CuTe DSL, or Python, keep it local and build a
  local baseline adapter with the same benchmark ABI used by the candidate.
- Workloads are frozen before tuning. Changing workloads, tolerances, scoring,
  or benchmark timing rules requires deleting old results and remeasuring both
  baseline and candidate.

## Required Directory Contents After First Milestone

```text
baseline/
  reference kernel source files
  kernel.cu or binding.py exposing the baseline ABI
solution/
  kernel.cu or binding.py exposing the candidate ABI
bench/
  workloads.json
  benchmark.py
  adapter.py
  correctness.py
  results.jsonl
docs/
  baseline_source.md
  benchmark_method.md
  run_log.md
config.toml
```

`bench/benchmark.py` must start from `docs/benchmark_template.py`. Do not
invent a different timing harness unless the template has a documented bug and
both baseline and candidate are remeasured after the fix.

## ABI Pattern

For pure CUDA, use a local direct-symbol CUDA pattern:

```cuda
#include <ATen/cuda/CUDAContext.h>

void my_kernel(TensorView input, TensorView output) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // launch <<<grid, block, shmem, stream>>>
}
```

The task may use one exported function per baseline entry point, or a single
exported function with an explicit selector argument. Baseline and candidate
must use the same choice.

## Workload Rules

- `bench/workloads.json` is the source of truth.
- Include every production shape the task is expected to optimize and a small
  regression grid for edge layouts/dtypes.
- Each workload records the function/selector, tensor shapes, dtypes, strides,
  scalar parameters, tolerance, random seed, and whether it is included in the
  headline score.
- Do not silently skip a production workload. Any missing baseline, missing
  candidate, compile failure, runtime failure, or correctness failure makes the
  benchmark invalid.

## Timing Rules

- Run each workload in an isolated subprocess when possible.
- Generate fresh random inputs for each trial; inputs may be stable inside a
  trial but must change between trials.
- Preallocate output tensors before timing. Timed regions must not include input
  generation, Python setup, JIT build, imports, allocation, or data restoration.
- Warm both baseline and candidate before measurement.
- Use CUDA events for GPU time. Use wrapper-inclusive wall-clock only as a
  secondary diagnostic.
- Use inner-loop amplification: record N back-to-back invocations inside one
  event pair and divide by N. Increase N until the sample is at least about
  1000 us or N reaches the configured cap.
- Use interleaved A/B sampling per trial to cancel clock and thermal drift:
  baseline, candidate, baseline, candidate, or the reverse order selected by a
  deterministic seed.
- Report median, mean, std, min, p10, p90 for both sides on every workload.
- Primary speedup per workload is `baseline_median_us / candidate_median_us`.
- Primary headline is equal-weight geometric mean over all production workloads.
  Also report arithmetic mean as a secondary tracking metric.

Recommended defaults:

```toml
[benchmark]
warmup_runs = 10
iterations = 200
num_trials = 7
inner_iterations_min = 1
inner_iterations_max = 4096
target_sample_us = 1000
timeout_seconds = 600
use_isolated_runner = true
```

## Correctness Rules

- Compare candidate and baseline against an independent PyTorch/math oracle when
  practical. If a full oracle is expensive, at minimum compare candidate against
  the baseline plus targeted oracle rows.
- Check shapes, dtypes, NaN/Inf, and tolerance per output.
- Poison output buffers before each correctness run so stale-output and skipped
  kernel bugs are visible.

## Provenance

Every benchmark result must record:

- task slug and target GPU
- baseline and candidate source hash
- exact command
- CUDA, PyTorch, compiler versions
- GPU model, GPU id, and idle state before/after
- workload count and trial/iteration/inner-loop settings
- correctness summary

Do not keep benchmark numbers without this provenance.
