# Kernel Optimization Rules

These rules apply to every kernel optimization task. They establish the
guardrails that keep the optimization loop honest and reproducible.

## Implementation Language And Abstraction Level

The optimized kernel (solution/) must be written in **CUDA C++**. Do not use
Triton or any other high-level kernel DSL for the candidate implementation.

Prefer the most primitive control constructs available to CUDA:

- **PTX inline assembly** (`asm volatile`) for hardware-specific operations:
  `cp.async.bulk` (TMA), `wgmma.mma_async` / `tcgen05.mma`, `mbarrier`,
  `fence.proxy.async`, `setmaxnreg`, named barriers (`bar.sync`), etc.
- **Thin wrappers** over PTX are acceptable — one inline function per PTX
  instruction, no state, no abstraction. DeepGEMM-style, not CUTLASS-style.
- **Prefer not to use** CUTLASS Collective/Builder/Pipeline abstractions or
  CuTe layout algebra. However, if implementing certain functionality from
  scratch is genuinely too complex (e.g. complex epilogue fusion, multi-stage
  pipeline orchestration, advanced layout transformations), selectively using
  CUTLASS/CuTe templates is allowed. When doing so, document the rationale
  in `docs/draft.md` — explain what was too complex to rewrite and why the
  CUTLASS/CuTe component was chosen.

Why this preference: heavily abstracted frameworks make it harder to reason
about what the hardware actually executes. When each PTX instruction is
visible in the source, the analyst can match NCU metrics to specific code,
and the coder can make targeted changes. But pragmatism matters — if a raw
implementation would take disproportionate effort for marginal visibility
gain, use the library and move on.

Preferred (thin wrappers):
```cuda
__device__ void tma_load(void* smem, uint64_t* mbar, ...) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier..."
                 : : "r"(...), "l"(...) : "memory");
}
```

Use with caution (CUTLASS-style — only when raw implementation is impractical):
```cpp
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<...>;
CollectiveMainloop collective;
collective(accum, tCrA, tCrB, ...);  // PTX visibility is reduced
```

The baseline may use any implementation (FlashInfer, CUTLASS, Triton) for
comparison.

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

## Design To The Ceiling (Architecture Selection)

The initial architecture must be designed **to the performance ceiling**, not as
a deliberately-simple "correctness-first v1" to be climbed incrementally. The
first design must adopt, from the start, every core technique the strongest
baseline uses to reach its measured efficiency (e.g. warp specialization,
ldmatrix, TMA, optimal tile/swizzle, async multi-stage pipeline). These are
architectural skeleton, not bolt-on tweaks — a simple structure has a hard
efficiency ceiling that no amount of incremental RLCR tuning can break.

Before implementing, perform a **structural-ceiling analysis** in addition to the
hardware roofline:

- Hardware roofline = compute/memory floor (a property of the GPU + problem).
- Structural ceiling = the max efficiency the **chosen candidate architecture**
  can reach. Ask explicitly: is load/compute serialized by a per-step block
  barrier? Are fragment loads bank-conflict-free (ldmatrix)? What caps occupancy?
  Does it replicate the baseline's enabling techniques, and if not, how much
  efficiency does each missing technique cost?

**Gate:** if `structural ceiling < baseline measured efficiency` (or < the
roofline-efficiency target), the design will lose — redesign before writing code.
If beating the baseline genuinely requires a rewrite-level effort, say so up front
rather than shipping a doomed simple version.

The incremental-edit guardrail (no file overwrites during iteration) governs
discipline **within a chosen architecture**. It never forces accepting a losing
architecture: when analysis shows the architecture itself cannot win, redesign
and re-implement from scratch (a new candidate source file) — that is the correct
action, not a violation.

### No performance revert; commit every round; select best at finalize

A performance regression does **not** trigger a rollback. Incremental
optimization inevitably hits a step that lowers performance, but a further change
*on top of* that step often turns it into a win (a local dip is not a dead end;
e.g. an optimization that regresses standalone but composes into a win with a
subsequent change). So:

- **Do not** `git checkout`-revert on a performance regression. Analyze and
  record why it regressed, commit the round, and keep moving forward (you may
  build directly on the regressed version).
- **Commit every round** (fast or slow). The git history is the safety net —
  any round is recoverable, so no active revert is needed.
- **Select the deliverable at finalization**: scan all committed rounds' bench
  numbers and `git checkout` the correct-and-fastest one as the final solution.
- The **only** exception is a correctness/compile failure: incorrect code cannot
  be benchmarked, so restore the last working state (that is "fix to correct,"
  not a performance revert).

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

## Low-Level Assembly Analysis

Every optimization round must include PTX/SASS static analysis.这不是可选步骤——
每次修改 kernel 代码后，都必须重新生成 cubin 并检查寄存器用量、指令模式和
编译器行为的变化。静态分析和 NCU 运行时 profiling 互补：NCU 告诉你"慢在
哪"，汇编分析告诉你"编译器做了什么导致慢"。

### Toolchain

| Level | Command | Output | What to look for |
|-------|---------|--------|------------------|
| PTX | `nvcc -ptx -arch=sm_100a file.cu` | `.ptx` virtual assembly | Algorithm logic, instruction selection, loop structure |
| Cubin | `nvcc -cubin -arch=sm_100a file.cu` | `.cubin` binary | Intermediate artifact for the tools below |
| SASS | `cuobjdump -sass file.cubin` | Real machine code | Instruction scheduling, dual-issue pairing, register allocation |
| Resources | `cuobjdump -res-usage file.cubin` | Register/shared mem usage | Occupancy bottleneck diagnosis |
| Source map | `nvdisasm -gi file.cubin` | SASS + source line numbers | Locate which source line generates which instructions |
| JIT .so | `cuobjdump -sass xxx.so` | SASS from shared library | FlashInfer JIT / AOT product analysis |

Replace `sm_100a` with the actual target architecture (e.g. `sm_90a` for
Hopper). The `-arch` flag must match the GPU being benchmarked.

### Every Round Must Check

- **Register pressure**: `cuobjdump -res-usage` shows register count per
  kernel. If occupancy is limited by registers, inspect PTX/SASS to find
  unnecessary live variables or spills to local memory (`STL`/`LDL` in SASS).
- **Instruction mix**: `cuobjdump -sass` reveals actual issued instructions.
  Check for unnecessary type conversions, redundant MOVs, missed constant
  folding, or sub-optimal math sequences (e.g. full `DFMA` instead of `FFMA`).
- **Dual-issue analysis**: In SASS output, look for instruction pairs that
  could issue together but don't — often caused by register dependency chains.
- **Shared memory bank conflicts**: `nvdisasm -gi` maps SASS loads/stores back
  to source lines. Cross-reference with NCU's bank-conflict metrics to identify
  the offending access pattern.
- **Baseline comparison**: Generate SASS for both baseline and candidate to
  compare instruction counts, loop unrolling, and memory access patterns
  side-by-side.
- **FlashInfer analysis**: Use `cuobjdump -sass` on the FlashInfer `.so` to
  inspect the baseline's actual generated code when the source is not available
  or when JIT compilation choices are unclear.

### Recording

When assembly analysis is performed, record findings in the round analysis
file (`.rlcr/current/rounds/r<N>/analysis.md`). Include: kernel name, register count, shared
memory usage, key SASS patterns observed, and any actionable insight that
informed the optimization direction.

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
