# <task_slug>

Target GPU: <target_gpu>.

Target upstream entry points to copy as local baseline:

- `<framework.module:entry_function>`

Goal: optimize <kernel_description> for the production shape set on <arch>.

Before writing an optimized kernel, read and follow:

- `../../docs/benchmark_contract.md`
- `../../docs/kernel_optimization_rules.md`
- `../../docs/correctness_contract.md`

Required first milestone:

1. Copy the relevant upstream source files for these entry points into
   `baseline/`.
2. Record upstream URL, commit, and copied files in `docs/baseline_source.md`.
3. Expose the copied baseline through local low-overhead ABI entry points.
4. Expose the candidate through the exact same ABI in `solution/`.
5. Create `bench/workloads.json`, copy the standard template to
   `bench/benchmark.py`, implement `bench/adapter.py`, and create
   `bench/correctness.py`.

Do not import, patch, or monkey-patch the upstream framework during correctness
or benchmark runs. All benchmark code must call only files in this task
directory.
