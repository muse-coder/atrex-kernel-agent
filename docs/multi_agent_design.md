# IterKernel 多 agent 设计

> **本文是设计总览（架构 + 逻辑图）。所有硬规则的权威单一来源是
> `.claude/commands/optimize-kernel.md`「## 多 agent 编排」节**；本文与其冲突时以该命令
> 文档为准。各角色契约见 `.claude/agents/{analysis,code-impl,code-iter}.md`。

## 1. 一句话总览

`/optimize-kernel` 的会话 agent 是 **master（战略层 orchestrator）**，它**不亲手写
kernel、不读原始 SASS dump**，而是用 **Agent 工具** spawn 三类 subagent
（`analysis` / `code-impl=code1` / `code-iter=code2`）。**信息只通过磁盘文档交接**
（spawn prompt / return 只是指针），**不用 Workflow**（这是串行 + 自适应判断 + 交互式
master 的循环，是 Agent 工具的正命，不是 Workflow 的并行批处理）。

## 2. 四个角色

| 角色 | 是谁 | 职责 | 工具 |
|---|---|---|---|
| **master** | 跑命令的会话 agent | 定/重定总纲；扣「丢弃总纲」扳机（ceiling + pathology，双向）；守 ≥90% roofline；持 `architecture-ledger`；commit/finalize；**唯一 spawner** | 写战略文档 + spawn |
| **analysis** | `subagent_type: analysis` | 读 NCU+SASS+PTX+源码/diff，按**绝对 roofline** 判 verdict，写 `analysis.md`+`summary.md`+下轮 `direction.md`；**recommend**（不 decide）枯竭 | 只读代码；写 `.rlcr/` 文档 |
| **code1** | `subagent_type: code-impl` | **从头实现**：按总纲 Write 新版本文件 `_v<N>` + correctness + 5 类产物。首次实现 + 每次 re-arch | Write/Edit |
| **code2** | `subagent_type: code-iter` | **渐进优化**：`cp v<N-1>→v<N>` + 对 v<N> 改一个 lever + correctness + 5 类产物 | **无 Write**；cp(Bash)+Edit |

## 3. 图 A — 角色拓扑 + 磁盘总线

```
                    ┌──────────────────────────────────────────────────┐
                    │  master  = 跑 /optimize-kernel 的会话 agent          │
                    │  战略层 · 唯一 spawner · 不写kernel/不读原始SASS dump │
                    │  定总纲 │ 丢弃判定 │ 守90%roofline │ commit │ finalize│
                    └───┬───────────────┬───────────────┬──────────────┘
        spawn(prompt=路径+轮号+目标,无历史)│               │
         ┌──────────────┘               │               └──────────────┐
         ▼                              ▼                              ▼
   ┌────────────┐               ┌────────────┐                 ┌────────────┐
   │   code1    │               │   code2    │                 │  analysis  │
   │ code-impl  │               │ code-iter  │                 │            │
   │ 从头实现    │               │ 渐进 cp+Edit│                 │ 诊断+判进退 │
   │ Write _v<N>│               │ 仅 cp+Edit  │                 │ 只读代码    │
   │ 一架构一次  │               │ 续用·一架构  │                 │routine续用 │
   └─────┬──────┘               └─────┬──────┘                 │/pivot fresh│
         │                            │                        └─────┬──────┘
   读总纲│写代码+产物         读direction│cp+Edit+产物        读产物+diff│写诊断文档
         │                            │                              │
         ▼                            ▼                              ▼
   ╔══════════════════════════════════════════════════════════════════════╗
   ║        磁盘 artifact 总线   /tmp/<slug>/.rlcr/current/   (唯一真相源)     ║
   ║                                                                        ║
   ║  战略: kernel-architecture.md(总纲) · architecture-ledger.md · goal-    ║
   ║        tracker.md · module-tracker.json · state.md · summary.md(索引)   ║
   ║  每轮: rounds/rN/{direction.md, candidate.*(5类SASS+ncu), analysis.md,  ║
   ║        summary.md} · solution/<family>_v<N>.<ext>                        ║
   ║  机械: .initial-impl-done · .direction-read-marker                      ║
   ╚══════════════════════════════════════════════════════════════════════╝
         ▲
         │  hook 读磁盘状态机械强制门槛(与 agent 身份无关,子agent照拦):
         │   · 防重写: 覆盖已存在 solution/ → deny ; 新文件(cp/Write) → 放行
         │   · 先读方向: 编辑前没读 rN/direction.md → deny
         │   · SASS门槛: r(N-1)/candidate-sass.txt 缺 → deny rN 编辑
         └── subagent 之间从不直接通信;一切配对 = "读对方写的文档"
```

## 4. 图 B — 端到端生命周期（master 主状态机）

```
 [Step1-2] master: 读需求 + nvidia-smi 定 arch + 建 /tmp/slug 仓库 + .rlcr 骨架
     │
     ▼
 [Step3+4abc] master ──spawn──▶ analysis(baseline模式) → 写 baseline-analysis.md
     ▼
 [Step4d-0] master: 评估算子复杂度 → 选原语 {纯CUDA+PTX | CUTLASS | CuTe DSL}
 [Step4d + 4d-ceiling] master: 设计总纲, 过"结构上限 ≥ 90%roofline"门槛
     │   写 kernel-architecture.md + architecture-ledger(首条) + config.toml(原语/命令)
     ▼
 [Step5] master ──spawn──▶ code1(从头实现, round1=v1, FROM SCRATCH)
     │   correctness gate + 5类产物 + ncu + touch .initial-impl-done
     ▼
 [Step6] master ──spawn──▶ analysis: 模块分解 + 写首个迭代轮 rounds/r2/direction.md
     ▼
 ┌───────────────────────── 内循环 (Step7, 见图C; 从 round2 起) ───────────────┐
 │   每轮: master设轮号 → code2 cp+改一lever → analysis判进退                    │
 │   verdict=CONTINUE ─────────────────────────────▶ 继续下一轮               │
 │   verdict=MODULE_COMPLETE ──────────────────────▶ [Step8 集成] → 下一模块  │
 │   analysis 报「接近枯竭」 ───────────────────────▶ 上浮到 ↓                 │
 └───────────────────────────────────────────────────────┬────────────────┘
                                                          ▼
                                    [丢弃门槛, 见图D]  master 判定
                              强制再磨一轮 │            │ 判丢弃(证据足)
                              (证据不足/低谷)│            ▼
                                  回内循环   改总纲+追加ledger → spawn 新 code1
                                            (re-arch, 新文件 _v<N>, 保持锁不rm) → 回内循环
     ┌─────────────────────────────────────────────────────────────────┐
     │  退出条件: roofline efficiency ≥ 90%  (唯一"完成";无轮次上限)          │
     └─────────────────────────────────────────────────────────────────┘
     ▼
 [Step9] master finalize: 扫所有 v*, 按 NCU duration 选最优 vN; 回灌 living lessons
     ▼
 [Step10] master 报告 docs/results.md
```

## 5. 图 C — 内循环一轮（时序 + 版本演化 + hook 门槛）

```
 master                         磁盘 solution/                    code2
   │ 设 state.md「当前轮:rN」                                        │
   │ spawn code2(上一版 v<N-1> 路径, rN)                            │
   ├──────────────────────────────────────────────────────────────▶│
   │                    cp <family>_v<N-1> → <family>_v<N>          │ (Bash,新文件,hook放行)
   │                    Read rN/direction.md  ◀─────────────────────┤ (hook:先读方向✓)
   │                    Edit v<N> 改一个 lever ◀────────────────────┤ (hook:防重写✓ SASS门槛查r(N-1)✓)
   │                    adapter import → v<N>                       │
   │                    diff v<N-1> v<N> 自查(只一 lever)            │
   │                    correctness gate + 生成 rN/candidate.*       │
   │                    git commit "rN: ..."                        │
   │◀── return:改了哪个lever/correctness/产物齐 ────────────────────┤
   │ spawn analysis(rN)                                             │
   │   analysis 读 v<N> + diff v<N-1>↔v<N> + 产物 + summary.md轨迹    │
   │   调度层SASS分析 + 锚绝对roofline + 测量诚信查                    │
   │   写 rN/analysis.md + rN/summary.md + summary索引 + r(N+1)/direction + state.md
   │◀── return:verdict|当前%roofline+gap|下轮方向 或「枯竭建议+最强证据」
   │ 读 return: CONTINUE→回顶(rN+1) ; 枯竭建议→图D
```

```
 版本演化(solution/ 累积, 全留存):
   v1(初始,code1) ─cp+1lever→ v2 ─→ v3 ─ … ─→ v7
                                                  │ master 判丢弃 → re-arch
                                          v8(code1 从零新架构) ─cp+1lever→ v9 ─→ …
   finalize: 扫所有 v*, 按 NCU duration 选最优那个 vN 直接定为交付(无需 git 考古)
```

> **关键不变式**：先落盘(写 analysis.md/state.md/summary.md)再 return → 即使 spawn 中途
> 压缩、return 丢了，master 重读磁盘即可恢复 verdict。return 只是指针。

## 6. 图 D — master「丢弃当前总纲」决策树（外循环铰链）

```
         analysis 报「接近枯竭」(带证据: SASS不可约 + 反向实验退化 + %roofline + 轨迹)
                                  │
                    ┌─ 关1: judge by ceiling, not current ─┐
                    │  当前方向结构上限 vs 备选上限?          │
                    │  (绝不因"现在更慢"拒换;绝不因"这轮退"丢)│
                    └───────────────┬─────────────────────┘
                    ┌─ 关2: pathology checklist 逐条核 ─────────────────┐
                    │  错瓶颈? 缺基本功(warp-spec/TMA/ldmatrix/streamK)? │
                    │  结构焊死(吞吐stall/throttle/spill)? 重复循环(查ledger)?│
                    │  过度工程? 正确性墙?                                │
                    └───────────────┬──────────────────────────────────┘
                       硬证据齐全(缺一不可)?
                    ┌───── 否 ─────┐         ┌───── 是 ─────┐
                    ▼              │         │              ▼
            强制再磨一轮            │         │      判丢弃:
       master 让 analysis 写新     │         │   1.更新 kernel-architecture.md
       direction → code2 继续      │         │   2.追加 architecture-ledger 一条
            (回内循环)             │         │   3.spawn 新 code1 (FROM SCRATCH,
                                              │     新版本文件 _v<N>,保持锁不rm)
                                              │   4.回内循环 (注:换原语也在此, 4d-0 重选)
```

## 7. 图 E — 上下文 / 压缩生命周期

```
 角色        实例寿命              跨轮上下文            压缩时
 ─────────  ──────────────────  ──────────────────  ────────────────────────────
 master     整个 campaign        会累积→会被压缩       SessionStart hook 重注入(铁律+
                                                     进度卡) + 重读 state/summary/
                                                     ledger/当前轮 → 接着 spawn 缺的角色
 code2      一个架构             SendMessage 续用      续用涨大→压缩→重读活跃版本文件恢复
                                (保留"手感")
 analysis   routine: 一个架构    routine 续用         pivot 用fresh本就无历史;续用压缩
            pivot:   一次性      pivot fresh(故意忘)   →靠 summary.md趋势+analysis.md重建
 code1      一次性               无                   一次性,几乎不压缩

 不变式: summary.md(趋势) + 每轮 analysis.md(附NCU/SASS证据) 必须自足到能重建轨迹。
        "fresh ≠ 看不见历史" —— 靠读磁盘看连续效果(summary趋势 + sass_hist_diff),不靠记忆。
 reservation: rounds/rN/ 有 direction.md 无 analysis.md ⇒ 该轮已开未完,续做不重开。
```

## 8. 源文件版本命名（v<N>，code1/code2 都遵守）

- 每个产代码的轮 N 都写成独立版本文件 `solution/<family>_v<N>.<ext>`（C++ `.cu`，
  CuTe DSL `.py`），**`v` 的数字 = 全局轮号 N**，与 `rounds/r<N>/` 一一对应。
- **round 1 = 初始从头实现（code1）= v1。**
  - **code2 渐进轮 N**：`cp v<N-1> v<N>`（Bash 建新文件，hook 放行）→ 对 v<N> 做**一个
    lever 的 Edit**。"渐进" = `diff v<N-1> v<N>` 只含一个 lever。
  - **code1 re-arch 轮 N**：直接 Write 全新 `_v<N>`（FROM SCRATCH）。
- code2 **无 Write 工具**：`cp`+`Edit` 而非 Write——因为 Write 新文件会**绕过** hook 的
  "先读方向"+SASS 门槛（只在 Edit 触发），cp 出 v<N> 再 Edit 则门槛照常生效。
- 全部 v1..vn 留在 `solution/`：跨轮直接 diff；finalize 按 NCU 选最优 vN。

## 9. 实现原语三选一（Step 4d-0，按算子复杂度评估）

candidate **不固定 CUDA**。master 先评估算子复杂度 + 达 ≥90% roofline 所需抽象层级，三选一
（Triton 不在候选集），选择写进 `kernel-architecture.md`/`architecture-ledger`/`config.toml`：

| 原语 | 何时选 |
|---|---|
| **纯 CUDA + PTX 薄封装** | 结构简单~中等；瓶颈在需手工调度的指令级控制；不贴标准模板 |
| **CUTLASS**（C++ 模板，Collective/Builder/CuTe layout 均允许） | 标准/近标准 GEMM/conv/attn，手写多阶段 pipeline 成本过高 |
| **CuTe DSL**（Python） | 要 CUTLASS 级抽象但需更灵活 fusion/更快迭代，裸 PTX 不现实 |

无论选哪个：FROM SCRATCH、每轮 5 类 SASS 产物 + NCU、渐进 cp+Edit/一 lever、退化不回退、
≥90% roofline。SASS 静态分析对三者都成立（CuTe DSL 从 JIT cubin 提取）；build/profile 命令
随原语，写进 `config.toml`（C++ 用 nvcc；CuTe DSL JIT 后 dump cubin）。

## 10. 与 auto-gpu-kernel 的关系（借鉴 / 分叉）

设计哲学源自 `auto-gpu-kernel`（Agent 工具 + artifact 即契约 + clean context + 反应式
specialist），但按 IterKernel 的约束调整：

**借鉴**：artifact 即契约、clean context 抗偏见、反应式不固定 cadence、judge by ceiling、
pathology checklist、`summary.md` 滚动索引、living lessons（prune）、folder reservation、
亚噪声用 paired 对比。

**分叉（IterKernel 特有）**：
- **测量**：auto-gpu-kernel 用 **CUPTI**（远程 Triton on Modal，NCU 不便、Triton 编译器
  不透明）；IterKernel 用 **NCU**（自控 GPU + CUDA/PTX，NCU 的 stall/roofline/source-
  counter 可行动）。NCU 为权威（铁律 -0.5），wall-clock 仅辅助。
- **静态分析**：auto-gpu-kernel 不做 SASS（且禁止静态预测寄存器压力）；IterKernel **每轮
  5 类 SASS 产物 + 调度层因果分析**是一等纪律（读的是已编译 SASS，不是"预测"）。
- **战略层**：auto-gpu-kernel 把 re-arch 写成硬规则（卡住就 Gluon 重写）；IterKernel 有
  独立 **master 战略层**，按 ceiling + pathology + 硬证据判丢弃，持 architecture-ledger
  防绕圈。
- **对抗自欺**：IterKernel 自欺史更深（ptxas symbol-name 效应、FP 塌缩、wall-clock 误判），
  故 analysis 契约内置测量诚信清单 + pivot 用 fresh 实例做无偏裁判。
