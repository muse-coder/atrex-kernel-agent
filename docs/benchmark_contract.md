# 独立 Benchmark 契约

本仓库使用本地的、自包含的 harness 来 benchmark kernel 优化。baseline 与
candidate 都从任务目录内的本地源码运行。

## 硬性规则

- baseline 与 candidate 必须暴露相匹配的本地入口点。为其中一方计入的任何
  wrapper 开销，也必须为另一方计入。
- 每个任务都必须包含两个本地实现：参考实现放在 `baseline/`，优化后的实现放
  在 `solution/`。
- 两侧都优先使用本地直连 CUDA ABI：输出 tensor 作为最后一个参数传入，
  `destination_passing_style = true`。
- 每次 CUDA launch 都必须使用 PyTorch 的当前 stream：
  `at::cuda::getCurrentCUDAStream()`。
- 如果 baseline 是 CUDA/C++ CUDA，则 baseline 与 candidate 必须通过相同的本地
  注册/导出/构建方式暴露。
- 不要传 `--use_fast_math`，除非 baseline 已经在用它，并且 candidate 使用完全
  相同的 flag。
- 如果 baseline 是 CuTe DSL 或 Python，把它保持在本地，并构建一个本地
  baseline adapter，使用与 candidate 相同的 benchmark ABI。
- workload 在调优开始前就冻结。变更 workload、tolerance、评分或 benchmark 计时
  规则，都需要删除旧结果并重新测量 baseline 与 candidate 两侧。

## 首个里程碑后必需的目录内容

```text
baseline/
  reference kernel source files
  kernel.cu or binding.py exposing the baseline ABI
solution/
  kernel.cu or binding.py exposing the candidate ABI
bench/
  workloads.json
  benchmark.py        # 含内置正确性 gate：python benchmark.py --correctness-only
  adapter.py
  results.jsonl
docs/
  baseline_source.md
  benchmark_method.md
  run_log.md
config.toml
```

`bench/benchmark.py` 必须从 `docs/benchmark_template.py` 开始。不要自行发明另一
套计时 harness，除非模板存在已记录在案的 bug，并且修复后 baseline 与 candidate
都重新测量。

## ABI 模式

对于纯 CUDA，使用本地直连符号的 CUDA 模式：

```cuda
#include <ATen/cuda/CUDAContext.h>

void my_kernel(TensorView input, TensorView output) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // launch <<<grid, block, shmem, stream>>>
}
```

任务可以为每个 baseline 入口点导出一个函数，或者用一个带显式 selector 参数的
单一导出函数。baseline 与 candidate 必须采用相同的选择。

## Workload 规则

- `bench/workloads.json` 是唯一可信来源（source of truth）。
- 包含任务预期要优化的每一个生产 shape，外加一个用于边缘 layout/dtype 的小型
  回归网格。
- 每个 workload 记录函数/selector、tensor shape、dtype、stride、标量参数、
  tolerance、随机种子，以及它是否计入头条分数。
- 不要悄悄跳过某个生产 workload。任何缺失的 baseline、缺失的 candidate、编译
  失败、运行时失败或正确性失败，都会使该 benchmark 无效。

## 计时规则

- 尽可能在隔离的子进程中运行每个 workload。
- 为每次 trial 生成全新的随机输入；输入在一次 trial 内部可以保持稳定，但在不同
  trial 之间必须变化。
- 在计时之前预分配输出 tensor。计时区间不得包含输入生成、Python 设置、JIT
  构建、import、分配或数据恢复。
- 在测量之前对 baseline 和 candidate 都做预热（warm）。
- 用 CUDA events 计 GPU 时间。包含 wrapper 的 wall-clock 仅作为次要诊断。
- 使用内层循环放大：在一对 event 之间记录 N 次背靠背调用，再除以 N。增大 N
  直到样本至少约 1000 us 或 N 达到配置上限。
- 每次 trial 使用交错 A/B 采样以抵消时钟与热漂移：baseline、candidate、
  baseline、candidate，或由确定性种子选定的相反顺序。
- 在每个 workload 上为两侧都报告 median、mean、std、min、p10、p90。
- 每个 workload 的主加速比为 `baseline_median_us / candidate_median_us`。
- 主头条指标为所有生产 workload 上的等权几何平均（geometric mean）。同时报告
  算术平均作为次要跟踪指标。

推荐默认值：

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

## 正确性规则

- 在可行时，把 candidate 与 baseline 都与一个独立的 PyTorch/数学 oracle 对比。
  如果完整 oracle 代价过高，至少把 candidate 与 baseline 加上有针对性的 oracle
  行做对比。
- 对每个输出检查 shape、dtype、NaN/Inf 和 tolerance。
- 在每次正确性运行前对输出 buffer 投毒（poison），使陈旧输出和被跳过的 kernel
  bug 可见。

## 溯源（Provenance）

每个 benchmark 结果都必须记录：

- 任务 slug 与目标 GPU
- baseline 与 candidate 的源码 hash
- 确切命令
- CUDA、PyTorch、编译器版本
- GPU 型号、GPU id，以及运行前/后的空闲状态
- workload 数量与 trial/iteration/inner-loop 设置
- 正确性摘要

没有这些溯源信息，不要保留 benchmark 数字。
