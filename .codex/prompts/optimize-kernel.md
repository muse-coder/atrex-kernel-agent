# Optimize Kernel - Codex Master Playbook

用户请求由启动器或当前对话提供。你是 Codex master agent，负责完整编排 IterKernel 的
GPU kernel 优化流程。

## 入口

启动后先建立这些变量：

```bash
AGENT_REPO="$(git rev-parse --show-toplevel)"
```

如果用户从 `scripts/launch_codex_task.sh` 进入，启动 prompt 会给出：

- `TASK_DIR`：agent repo 内的任务占位目录。
- `SOURCE_DRAFT`：任务草稿，通常是 `<TASK_DIR>/.rlcr/draft.md`。

如果用户手动给 kernel 描述，则直接用该描述创建 campaign。

## Codex 执行模型

Codex 版不依赖 Claude slash command 和 hooks。默认由当前 Codex 会话顺序扮演四个角色：

1. master：战略和状态机。
2. analysis：只读证据，写分析和下一轮 direction。
3. code-impl：从头实现新架构。
4. code-iter：复制上一版，单 lever 增量修改。

每次进入角色前先读对应 `.codex/agents/*.md`。如果运行环境有 subagent 工具，可以使用，
但必须保持 master-only spawn、扁平树、磁盘 artifact 交接。

## 全局铁律

- candidate kernel 必须从头实现；旧实现只能参考，不能作为代码起点。
- baseline 必须取 PyTorch/cuBLAS 与 FlashInfer AOT 中 NCU 实测最快者。
- 性能结论只认 NCU `gpu__time_duration`。
- 完成目标是该 shape roofline 上限的 90% 以上。
- 每轮一个 lever，退化不回退。
- correctness 未过不能 profile；correctness evidence、NCU 和静态产物不能伪造。
- 每轮都要 commit；`.rlcr/` 不进 git。

## Step 1 - 需求和 GPU

1. 读用户请求、`TASK_DIR/prompt.md` 或 `SOURCE_DRAFT`。
2. 运行 `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`，记录 target GPU
   和 compute capability。
3. 推断 workload shapes、kernel 语义、正确性 oracle 和 baseline 候选。

## Step 2 - 创建 campaign repo

1. 生成 slug，例如 `rtxpro5000_fp8gemm__m1024n10240k4096`。
2. 创建 `/tmp/<slug>/` 独立 git repo。
3. 从任务模板复制 `prompt.md`、`config.toml`、`baseline/`、`solution/`、`bench/`、`docs/`。
4. 在 agent repo 的 `campaigns/operators/<slug>/` 只保留 `.gitkeep`。
5. 初始化 `.rlcr/current/rounds/`、`state.md`、`summary.md`、`architecture-ledger.md`、
   `goal-tracker.md`、`module-tracker.json`。

## Step 3 - Baseline

1. 实现对称 ABI 的 baseline adapter。
2. 测 PyTorch/cuBLAS 和 FlashInfer AOT，取 NCU `gpu__time_duration` 更快者。
3. 写 `docs/baseline_source.md` 和 `.rlcr/current/goal-tracker.md`。
4. baseline 与 candidate 必须 destination-passing、固定 workload、预分配 output、
   CUDA-event A/B 交错计时。

## Step 4 - 架构设计

1. 用 analysis 角色读 baseline NCU/SASS/PTX，写 `baseline-analysis.md`。
2. master 评估算子复杂度，选择 primitive：
   - `cuda_ptx`
   - `cutlass`
   - `cutedsl`
3. 写 `kernel-architecture.md`：tile、CTA/warp roles、pipeline、smem layout、MMA 指令、
   epilogue、MODULE 分解、寄存器预算、预期 NCU 指标。
4. 做 4d-ceiling 检查：结构上限必须能达到 90% roofline，否则先重设架构。

## Step 5 - 从头实现

进入 `code-impl` 角色，按 `.codex/agents/code-impl.md` 执行 round 1 / v1：

- 写 `solution/<family>_v1.<ext>`。
- correctness 全过。
- 生成 NCU 和 5 类静态产物。
- `touch .rlcr/current/.initial-impl-done`。
- git commit。

## Step 6 - 分解和首轮 direction

进入 `analysis` 角色：

- 读 v1 evidence。
- 写 `decomposition.md` 或更新 `module-tracker.json`。
- 写 `rounds/r2/direction.md`。
- 更新 `summary.md` 和 `state.md`。

## Step 7 - RLCR 内循环

每轮 N 从 2 开始：

1. master 更新 `state.md`：`当前轮: r<N>`。
2. 进入 `code-iter`：
   - 读 `rounds/r<N>/direction.md`。
   - 运行 `codex_round_guard.py mark-direction-read`。
   - 运行 `codex_round_guard.py pre-edit`。
   - `cp v<N-1> -> v<N>`，Edit 一个 lever。
   - correctness、benchmark sanity、NCU、静态产物。
   - git commit。
3. 进入 `analysis`：
   - 读本轮 evidence 和 `diff v<N-1> v<N>`。
   - 写 `rounds/r<N>/analysis.md`、`rounds/r<N>/summary.md`、更新 `summary.md`、`state.md`。
   - CONTINUE 时写 `rounds/r<N+1>/direction.md`。

上一轮产物不齐时不得进入下一轮。Codex 版用下面命令显式检查：

```bash
python "$AGENT_REPO/scripts/codex_round_guard.py" pre-edit "$CAMPAIGN_DIR" <N>
```

## Step 8 - 枯竭和 re-architecture

analysis 只能 recommend 枯竭。master 做决定前必须：

1. 用 fresh analysis pass 重新读磁盘 evidence。
2. 按 ceiling 而不是当前数字判断。
3. 逐项检查 pathology：错瓶颈、缺基本功、结构焊死、重复循环、过度工程、正确性墙。
4. 若证据不足，强制继续一轮或换模块。
5. 若证据充分，更新 `architecture-ledger.md` 和 `kernel-architecture.md`，进入新的
   `code-impl` re-arch 轮，写新的 `v<N>`。

## Step 9 - Finalize

达到 90% roofline 后：

1. 扫描全部 `solution/*_v<N>.*` 和对应 NCU evidence。
2. 以 NCU duration 选最优版本。
3. 写 `docs/results.md`，包括 baseline、roofline、最优版本、关键 SASS/NCU 证据、
   正确性证据和可复现命令。
4. 必要时把高价值 lesson 回灌到 agent repo 文档；不要提交 campaign `.rlcr/`。

## Step 10 - 中断恢复

任何恢复或 context 压缩后：

1. 读 `AGENTS.md`。
2. 读本文件。
3. 读 `CAMPAIGN_DIR/.rlcr/current/state.md`、`summary.md`、最新 `rounds/r<N>/`。
4. 从未完成的最小步骤继续，不重开已经存在的轮次。
