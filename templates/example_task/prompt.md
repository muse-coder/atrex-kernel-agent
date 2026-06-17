# <task_slug>

目标 GPU：<target_gpu>。

Baseline kernel 入口点：

- `<module_or_file:entry_function>`

目标：在 <arch> 上针对生产 shape 集优化 <kernel_description>。

在编写优化 kernel 之前，请阅读并遵循：

- `../../docs/benchmark_contract.md`
- `../../docs/kernel_optimization_rules.md`
- `../../docs/correctness_contract.md`

必须完成的第一个里程碑：

1. 把参考 kernel 实现放入 `baseline/`。
2. 在 `docs/baseline_source.md` 中记录 baseline 的来源。
3. 通过本地低开销 ABI 入口点暴露 baseline。
4. 在 `solution/` 中通过完全相同的 ABI 暴露 candidate。
5. 创建 `bench/workloads.json`，把标准模板复制到
   `bench/benchmark.py`，实现 `bench/adapter.py`。正确性已内置在 benchmark.py
   （poison + oracle compare），用 `python bench/benchmark.py --correctness-only`
   单独跑正确性 gate，无需另写 `correctness.py`。

所有 benchmark 代码只能调用本任务目录内的文件。
