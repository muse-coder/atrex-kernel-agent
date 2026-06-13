# Optimize Kernel

用户请求：$ARGUMENTS

## 工作流程

按以下步骤执行 kernel 优化：

### 1. 理解需求

从用户的描述中提取：
- **Kernel 类型**：什么算子（FP8 GEMM、GroupNorm+SiLU、FlashAttention 等）
- **目标 GPU**：什么架构（B200、H200 等）。如果用户没说，运行 `nvidia-smi`
  检测当前 GPU
- **Workload shapes**：如果用户没给具体 shape，根据 kernel 类型生成常见的
  production shapes
- **Baseline**：用户是否指定了 baseline？没有则默认 FlashInfer AOT
- **特殊约束**：dtype、精度要求、是否 fused 等

如果 kernel 类型不明确，直接问用户。

### 2. 创建任务

```bash
TASK_SLUG="<gpu>_<kernel>__multi_shape"
cp -r templates/example_task campaigns/operators/$TASK_SLUG
```

填写 `prompt.md`（kernel 描述、目标 GPU、baseline、production shapes）和
`config.toml`（build/benchmark 参数）。

### 3. 生成优化计划

读取规则文档和 KernelWiki（如果存在），生成 `.rlcr/plan.md`：
- Baseline recovery — 从哪里获取、如何暴露 ABI
- Correctness first — oracle、tolerance
- Benchmark setup — workloads、adapter
- Optimization directions — 根据 kernel 特点列出候选方向
- Acceptance criteria — roofline efficiency ≥ 90%

### 4. 启动 RLCR Workflow

提交任务目录，记录 base commit，然后调用 Workflow：

```
Workflow({
  name: "rlcr",
  args: {
    planFile: "<task-dir>/.rlcr/plan.md",
    baseBranch: "<base-commit>",
    rlcrDir: "<task-dir>/.rlcr/current"
  }
})
```

Workflow 用**三个 agent 分工协作**执行渐进式模块化优化：

**Coder agent** — 读取 Analyst 的方向文档 → 修改 CUDA 代码（聚焦当前目标模块，
但如果优化需要联动修改其他位置以通过编译/正确性测试，允许最小范围的外部改动）
→ 跑 correctness + benchmark。不跑 NCU，不做性能分析。

**Profiler agent** — 跑 NCU（`--set full`、`--set source`）、导出 cubin、
`cuobjdump -res-usage`、`cuobjdump -sass`、`nvcc -ptx`。纯数据采集，不分析
数据，不改代码。

**Analyst agent** — 读取 Profiler 导出的 NCU 指标 + PTX/SASS + CUDA 源码 →
性能分析（瓶颈定位、theory vs actual 归因、PTX/SASS 深度分析）→ 列举问题 →
写出按轮次保存的优化方向文档 `round-N-direction.md`，供 Coder 下一轮读取。

信息流：`Coder → Profiler → Analyst → round-N-direction.md → Coder`

流程：
1. **Round 0 — 初始实现**
   - **Profiler** 跑 baseline（FlashInfer/CUTLASS 等）的 NCU + PTX/SASS
   - **Analyst** 深度分析 baseline 性能 → 设计新 kernel 架构（tile size、
     pipeline 结构、smem layout、PTX 指令选择）→ 写完整的架构设计文档
   - **Coder** 按架构设计**从头实现完整 CUDA kernel**（不是在 baseline 上改）
   - **Profiler** 对新 kernel 跑 NCU + PTX/SASS
   - **Analyst** 分析新 kernel → 分解模块（插 MODULE 标记）→ 对比 baseline →
     设计全局优化策略 + 写首份 round-0-direction.md
2. **Module Loop** — 对每个模块循环（最多 15 轮/模块），每轮三步：
   - **Coder**: 读 round-N-direction.md → 实现优化 → correctness + benchmark
   - **Profiler**: 对新版本跑 NCU + 导出 cubin/PTX/SASS
   - **Analyst**: 读 Profiler 数据 + Coder 改动 + kernel 源码 → theory vs actual
     归因 → 写新 round-(N+1)-direction.md（含问题清单 + 下一步方向）
3. **Integration** — 每个模块完成后 Profiler 跑全 kernel profile，Analyst 做
   集成分析，回退时做 NCU + PTX/SASS 诊断
4. **Finalize** — per-module contribution breakdown + 理论准确度总结

停止条件：
- **roofline efficiency ≥ 90%** → 优化成功
- **单模块连续 5 轮无进展** → 跳到下一模块
- **所有模块完成** → 输出最终报告

### 5. 报告结果

Workflow 结束后，向用户报告 `docs/results.md` 中的最终性能数据。
