# <task_slug>

Target GPU: <target_gpu>.

Baseline kernel entry point(s):

- `<module_or_file:entry_function>`

Goal: optimize <kernel_description> for the production shape set on <arch>.

Before writing an optimized kernel, read and follow:

- `../../docs/benchmark_contract.md`
- `../../docs/kernel_optimization_rules.md`
- `../../docs/correctness_contract.md`

Required first milestone:

1. Place the reference kernel implementation into `baseline/`.
2. Record the baseline's origin in `docs/baseline_source.md`.
3. Expose the baseline through local low-overhead ABI entry points.
4. Expose the candidate through the exact same ABI in `solution/`.
5. Create `bench/workloads.json`, copy the standard template to
   `bench/benchmark.py`, implement `bench/adapter.py`, and create
   `bench/correctness.py`.

All benchmark code must call only files in this task directory.
