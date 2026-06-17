# 正确性契约

本文档定义每个 kernel 优化任务在 benchmark 结果被视为有效之前，必须满足的正确
性要求。

## 总体原则

1. **正确性优先于性能。** 在 candidate 同时通过生产 workload 正确性检查和标准
   回归网格之前，任何 benchmark 结果都不算数。

2. **检查前先投毒。** 在每次正确性调用之前，输出 buffer 必须填入 NaN（浮点）
   或哨兵值（整数），以便陈旧输出和被跳过的 kernel bug 可见。

3. **保留 NaN/Inf。** 在正确性 harness 中保留显式的 NaN/Inf 检查。如果 baseline
   kernel 在合法输入上从不产生 NaN/Inf，那么 candidate 也不得产生。

4. **Oracle 对比。** 在可行时，把 candidate 输出与一个独立的 PyTorch/数学
   oracle 对比。如果完整 oracle 代价过高，至少与 baseline 加上有针对性的
   oracle 行做对比。

## 定义回归网格

每个任务都应在其 `prompt.md` 或一个单独的 `docs/correctness_contract.md` 文件
中定义一个回归网格。该网格指定：

- **Shapes**：覆盖典型情形和边缘情形的代表性 tensor 维度。
- **Dtypes**：kernel 预期支持的所有 dtype（例如 fp16、bf16、fp32）。
- **标量参数**：任何可配置的标量（eps、num_groups 等）。
- **Layout 变体**：contiguous、channels-last、strided 等。
- **Tolerances**：按 dtype 划分的绝对与相对 tolerance。

示例：

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

## 何时更新 Tolerance

- 不要为了让一个失败的 candidate 通过而放宽 tolerance。
- 如果任务在 `docs/benchmark_method.md` 中记录了更严格的任务局部 tolerance，
  使用更严格的那个值。
- 如果有证据表明 baseline 自身在特定 shape 上超出了某个网格 tolerance，记录下
  来，并只调整那一个单元格。
