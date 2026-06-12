# Kernel Optimization Rules

These rules apply to every kernel optimization task. They establish the
guardrails that keep the optimization loop honest and reproducible.

## Baseline And Candidate Pairing

Each task must end with two local implementations:

- `baseline/`: the reference kernel implementation, exposed through the task
  benchmark ABI.
- `solution/`: the optimized implementation exposed through the exact same
  task benchmark ABI.

When the task prompt specifies a particular baseline, use that implementation.
When no specific baseline is given, default to the corresponding FlashInfer
kernel as the reference implementation. FlashInfer must be installed as the
AOT pre-compiled (cubin) version from the latest source
(`FLASHINFER_ENABLE_AOT=1 pip install -e .`) so the baseline reflects the
strongest available implementation in its deployment configuration. Do not
use the JIT/Python mode as the baseline — JIT has runtime compilation
overhead that does not represent production performance. See README for
build instructions.

Record the baseline's origin in `docs/baseline_source.md`:
- source: FlashInfer (default) or other specified implementation
- version / commit
- the exact function or kernel entry point used
- any local modifications made to adapt the ABI

If the baseline kernel is CUDA/C++ CUDA, the baseline and the optimized
candidate must use the same local registration/export/build style. Do not
expose the baseline through one wrapper and the candidate through a lighter
direct path.

If the baseline implementation is Triton, CuTe DSL, or Python, keep it inside
`baseline/` and build a local adapter that has the same call signature,
argument ordering, stream behavior, and output allocation policy as the
candidate adapter.

Every CUDA launch must use PyTorch's current stream, for example
`at::cuda::getCurrentCUDAStream()`.

## Compile Flags

Compile flags must be symmetric between baseline and candidate whenever they can
change numerics or code generation.

Do not pass `--use_fast_math` unless the baseline already uses it and the
candidate uses the exact same flag. The default is no fast math.

Do not add extra `nvcc` flags, architecture-specific toggles, or math-mode flags
to only one side. Record all compile flags in `docs/benchmark_method.md`.

## Remote GPU Rule

Tasks targeting a specific GPU architecture must validate and benchmark on that
architecture.

Before GPU work, inspect `nvidia-smi` and choose a GPU with no active compute
processes and no meaningful memory occupancy. Use the selected GPU consistently
for baseline, candidate, correctness, benchmark, profiling, and NCU commands in
the current run.

Record the host, GPU id, GPU model, and before/after GPU state in the task's
`docs/run_log.md` or `docs/results.md`.

Use a task-owned workspace for builds, benchmark logs, profiler traces, and NCU
reports. Do not write artifacts into another task's workspace.

## Correctness Before Performance

Before optimization, identify:

- the baseline source file(s);
- the callable arguments and scalar parameters;
- the production workload rows;
- the canonical regression grid (if defined in the task prompt or
  `docs/correctness_contract.md`).

The final candidate must pass the production workload correctness checks and the
canonical regression grid before any benchmark result counts.

Preserve explicit NaN/Inf checks. Use tolerances from the task's correctness
contract unless the task records a stricter task-local tolerance in
`docs/benchmark_method.md`.

## Benchmark And Evidence

Use `docs/benchmark_template.py` as the timing harness starting point. Do not
change workloads, tolerances, score aggregation, or timing rules after tuning
starts unless both baseline and candidate are remeasured.

Every iteration must refresh its kernel-optimization context before choosing the
next edit, benchmark run, profiling run, or no-go conclusion. That refresh
includes the task prompt, current benchmark evidence, and available knowledge
skills (e.g. `external/KernelWiki/SKILL.md`, `external/ncu-report-skill/SKILL.md`).

When NCU profiling is needed, follow `external/ncu-report-skill/SKILL.md` if
available. Keep the profile harness, reports, analysis, and summary in a
task-owned directory, and use the resulting evidence to choose the next edit
instead of guessing.

A final performance claim must report:

- median, mean, std, min, p10, and p90 latency per workload;
- equal-weight geometric mean speedup over production workloads;
- exact command lines;
- baseline and candidate source hash;
- GPU host/id/model and idle-state evidence.

Use Nsight Compute when a correct candidate is not clearly target-complete or
when profiler evidence would change the next edit. A final improvement or no-go
must include a roofline-style explanation: estimated bytes moved, useful scalar
or vector operations, achieved bandwidth and/or FLOP/s when relevant, and the
active bound or blocker.

Do not finalize a no-go because the first candidate loses. A no-go needs
baseline numbers, at least one reasoned candidate attempt, correctness status,
benchmark evidence, and a named active bound or blocker.

## PR Scope

After a kernel is optimized, the final commit must include only:

- the kernel source for baseline, optimized solution, local ABI, benchmark
  adapter, and correctness/benchmark harness;
- the per-shape baseline-vs-candidate performance comparison and final
  conclusion, normally in `docs/results.md`;
- small method/provenance notes needed to reproduce the result.

Do not commit intermediate optimization artifacts such as raw NCU reports,
Nsight traces, profiler run directories, temporary harness binaries, build
outputs, scratch logs, failed experiment dumps, or large benchmark JSONL files
unless explicitly requested.

## Shape Specialization

Shape-specialized kernels, template variants, autotune tables, and dispatchers
are allowed when benchmark or profiler evidence shows that different workload
buckets need different block sizes, vector widths, memory layouts, or register
pressure tradeoffs.

When specialization is used, write `docs/dispatch.md` with:

- the bucket condition;
- the selected baseline and candidate entry point;
- per-bucket latency and speedup;
- the reason that bucket uses this implementation.

Do not force one universal kernel when evidence shows multiple shape buckets
need different implementations.

## Prior Art And Exploration

Before settling on an implementation strategy in any iteration, read or query
available knowledge skills when they could change the design: CUTLASS/CuTe,
CUDA samples, PyTorch, KernelWiki, and task-local NCU evidence.

Record kept/rejected ideas in `docs/draft.md`, `docs/results.md`, or
`docs/research.md`. Keep optimization attempts bounded and evidence-backed.

## Completion Bar

A task is complete only when:

- `baseline/`, `solution/`, `bench/`, and `docs/` contain the required local
  artifacts;
- production workload correctness passes;
- canonical regression correctness passes (if applicable);
- the benchmark result uses the standard standalone timing rules;
- NCU or a clear roofline-style analysis explains the final result or blocker;
- `docs/results.md` summarizes the final command, per-shape performance
  comparison, result, and conclusion;
- the staged diff excludes raw profiling, NCU, temporary build, and scratch
  artifacts.
