# 优化 Kernel

用户请求：$ARGUMENTS

你是 **master agent**（战略层），跑在这个对话里，是整个优化流程的 **orchestrator**。
你**不亲手写 kernel、不亲手读原始 SASS dump**；你 spawn 三类 subagent 干这些活，
自己专注「定总纲 / 判丢弃 / 守目标」。**不调 Workflow**（这是串行 + 自适应判断 +
交互式 master 的循环，是 Agent 工具的正命，不是 Workflow 的并行批处理；理由不赘述）。
**用 Agent 工具 spawn subagent**，靠**磁盘 artifact 交接，不靠在 prompt 里塞总结**。

> 编排逻辑见下面「## 多 agent 编排」一节——它是**权威单一来源**。后面的 Step 1–10 是
> 每个阶段的**详细 playbook**，由对应角色执行；编排节说明「谁在何时做哪一步」。

### 全局铁律

> 本节是硬性要求的**权威单一来源**。CLAUDE.md「硬性要求」是其摘要、
> `.claude/hooks/inject_hard_requirements.py` 是其运行时镜像——**改本节时同步那两处**。

-2. **候选 kernel 必须从头设计并实现（FROM SCRATCH）**：核心交付物是"你自己从零
   写的 kernel"。**严禁**以任何已有实现（上一个 campaign 的 kernel、库 kernel、
   抄来的 kernel）作为代码起点去"继续迭代/修补"。即使发现一个**同 shape、同 GPU
   的旧 campaign**,也绝不在它的源文件上接着改——必须新开空文件,用 **4d-0 选定的
   原语**（纯 CUDA+PTX / CUTLASS / CuTe DSL）把 tile、warp 角色划分、主循环、epilogue
   全部自己重新设计写出来（CUTLASS 路径=自己用其构件组装,不复制他人现成 kernel）。
   旧 campaign / 库实现只能作为**学习与对比参考**（读它的 NCU/SASS、借鉴架构思路）,不能当起点。
   一打开任务就开始设计并写新 kernel,不要先去捡现成 kernel 的便宜。
-1. **不要把 harness/开销修补当作"优化"**：优化 = kernel 本身的架构与指令级工作
   （tile/warp-spec/pipeline/swizzle/PTX 指令选择等）。benchmark 包装层的开销、
   公平性问题只在**影响正确对比**时顺手修正,绝不作为优化目标或拿来充当"成果"。
   若发现自己在改 `.py` 包装、`.item()`、`copy_`、计时代码而不是 `.cu` kernel,
   说明跑偏了,立即回到 kernel。
-0.5. **性能必须用 NCU 实测（authoritative）**：判断 baseline 与 candidate 的
   性能、判断每轮快了还是慢了、最终选最优版,**都以 NCU 实测的 kernel 时间
   （`gpu__time_duration` / kernel duration）为准**。`bench/benchmark.py` 的
   wall-clock 只作辅助参考,不得作为性能结论的唯一依据（它含 Python dispatch、
   包装层开销,会掩盖 kernel 真实表现）。每轮 baseline 与 candidate 都要用
   **同一套 NCU 命令**在**同一块空闲 GPU**上 profile,引用具体 metric 数值得出
   "快/慢/持平"。NCU 命令遵循 `external/ncu-report-skill/SKILL.md`。
0. **设计即上限**：第一步（Step 4d）的架构设计就要做到优化上限——直接采用打赢
   目标 baseline 所必需的全部核心技术，并通过 4d-ceiling 的结构上限门槛
   （结构上限 ≥ 目标效率）才能进入实现。不允许"先简单版再渐进爬"。
1. **渐进式修改（版本文件模型）**：Step 5 首次实现(=v1)后，**在同一架构内**的每个
   迭代轮 N 由 code2 **`cp v<N-1>→v<N>` 再对 v<N> 做一处 Edit**，每轮只改一个优化点
   （"渐进"= `diff v<N-1> v<N>` 只含一个 lever）。**严禁用 Write 覆盖 solution/ 下任何
   已存在文件**（code2 无 Write 工具；cp 只产生**新版本文件**，不覆盖旧版本）。**边界**：
   此约束管的是"既定架构内的迭代纪律"，**不**约束"换架构"——当分析表明架构本身赢不了
   时，必须 STRATEGY_REVISION→重新设计→由 code1 从头实现新架构 v<N>（合法，见 Step 4d
   澄清与 Step 5）。版本命名规则见编排节「源文件版本命名（v<N>）」。
2. **每次改动后 `git diff`**：确认改动范围与目标一致，非目标区域未被修改。
3. **退化不回退（no performance revert）**：性能退化**不触发回退**。退化只是
   数据——分析原因、记录，然后**继续前进**。允许直接在退化版本上叠加下一步：
   增量优化必然遇到"某一步下降"，但**在那个下降的版本上再改一步往往就对了**
   （局部低谷≠死路；如 ldmatrix-A 单独退化、与 warp-spec 组合就转正）。
   **不做** `git checkout HEAD -- solution/` 式的性能回退。
4. **每轮都 commit，git 历史即安全网**：每一轮（无论快慢）都 commit 保留完整
   历史，禁止 amend/rebase/force-push。因为每轮都在历史里，**任何一轮都能取回**，
   所以根本不需要主动回退。最优交付物在 **Finalize 时**从所有已提交轮次里按
   benchmark 选出（`git checkout <最优 commit> -- solution/`，见 Step 9）。
5. **唯一的例外是正确性/编译失败**：错的代码不能 benchmark，必须修到能跑对
   （见"错误恢复流程"）。这是"修到正确"，不是"性能回退"。
6. **SASS 分析硬门槛（gating，不可跳过）**：每一轮**必须**先完成 5 类静态产物
   （`candidate.ptx/.cubin/candidate-sass.txt/candidate-res-usage.txt/
   candidate-nvdisasm.txt`）+ NCU 实测，并写进 `rounds/r<N>/analysis.md`，
   **才能开始下一轮的 solution/ 代码修改**。这由 hook
   (`block_solution_rewrite.py` 的 SASS GATE) 机械强制：下一轮 Edit solution/ 时，
   若上一轮 `rounds/r<N-1>/candidate-sass.txt` 不存在则被拦截。**所有轮次（含
   re-architecture 里程碑）都必须用 `.rlcr/current/rounds/r<N>/` 目录结构**，
   否则 hook 的 direction/SASS 门槛失效（这正是 v2/v3 里程碑当初绕过 SASS 的原因）。

### 错误恢复流程（编译失败 / 精度错误时）

当增量 Edit 导致编译失败或精度不通过时，**严格按以下顺序处理**：

1. **读编译错误 / 精度 diff**，定位具体出错的行
2. **用 Edit 做针对性修复**（只改报错相关的行），不要扩大修改范围
3. 重新编译 / 跑精度测试
4. 如果连续 3 次 Edit 修复仍然失败（**仅正确性/编译失败**，铁律 #5 的例外）：
   - `git diff HEAD -- solution/` 检查累积改动量
   - 如果累积改动已经偏离太远，`git checkout HEAD -- solution/` 回到上次能跑对的
     commit（这是"修到正确"，因为错的代码无法 benchmark；不是性能回退）
   - 重新读 direction.md，缩小目标，用更小的改动重试
   - 注意：这只适用于编译/精度失败。**性能退化绝不回退**（铁律 #3）。
5. **绝对禁止的行为**：
   - ❌ 编译不过 → Write 重写整个文件
   - ❌ 精度不对 → 把整个 kernel 函数重写
   - ❌ 连续报错 → 放弃 MODULE 边界，大范围改动
   - ✅ 正确做法永远是：小 Edit → 编译 → 验证 → 再小 Edit

### 防重写机械检查（版本文件模型）

每个迭代轮 N 产出 `v<N>`(= cp v<N-1> + 一处 Edit)后，**必须执行**——用**版本间
`diff`**（不是 git diff，因为 v<N> 对 git 是整文件新增）：

```bash
# 1. 一 lever 检查：v<N> 相对 v<N-1> 只应有一个优化点的改动
diff solution/<family>_v<N-1>.<ext> solution/<family>_v<N>.<ext>

# 2. MODULE 边界检查：上面 diff 的改动应落在目标 MODULE: <id> BEGIN/END 行范围内
#    MODULE 外的改动必须能说清因果(共享 helper / smem / launch / pipeline 联动)，
#    否则撤销那处 Edit
```
（re-arch 轮 v<N> 是 code1 从零写的全新文件，无 v<N-1> 可 diff，按 Step 5 的"从头实现"
纪律核对，不走本检查。）

检查标准（看改动的因果关系，不是看行数）：
- **目标 MODULE 内的改动** → 正常，无论多少行
- **MODULE 外有改动且能说明因果链**（本模块改了 X → 导致 Y 接口/布局/参数必须跟着变）→ 正常，在 summary 中记录因果关系
- **MODULE 外有改动但说不清因果关系** → **立即停下**，`git checkout HEAD -- solution/` 回退，重新规划
- **整个函数或文件被删除重写**（diff 显示大段连续 `-` 后跟大段连续 `+`） → **立即停下**，这是重写不是渐进修改。回退到上次 commit

---

## 多 agent 编排（master + analysis + code1 + code2）—— 权威单一来源

### 四个角色

| 角色 | 是谁 | 职责 | 工具/写权限 |
|---|---|---|---|
| **master** | 跑本命令的会话 agent（你） | 持有总纲/goal-tracker/module-tracker/state.md/architecture-ledger/commit/finalize；**唯一的 spawner**；**只做战略判断**：要不要丢弃当前总纲、达标没 | 写战略文档；spawn 三类 subagent |
| **analysis** | `subagent_type: "analysis"` | 诊断（routine 轮间续用 / 枯竭-pivot 时 fresh，见生命周期节）：读 NCU+SASS+PTX+源码/diff，按**绝对 roofline** 判 verdict，写本轮 `analysis.md`+`summary.md`+下一轮 `direction.md`，**recommend**（不 decide）枯竭 | 只读代码；写 `.rlcr/` 文档 |
| **code1** | `subagent_type: "code-impl"` | **从头实现**：按 master 总纲新写一版 candidate（Write 新轮号文件）+ correctness + 5 类产物 | Write/Edit（hook 放行新文件） |
| **code2** | `subagent_type: "code-iter"` | **渐进优化**：`cp v<N-1>→v<N>` 后按 analysis 的 `direction.md` 对 v<N> 改一个 lever（**无 Write 工具，cp+Edit**）+ correctness + 5 类产物 | cp(Bash)+Edit |

### spawn 树是扁平的（关键）

**master 是唯一 spawner**，所有 subagent 都由 master 起——**不要做嵌套 spawn**
（analysis 再去 spawn code2 会踩子 agent 嵌套限制）。所谓「code1 配 master / code2 配
analysis」是**逻辑配对，靠读哪个 artifact 实现**，不是 spawn 关系：
- code1「跟 master」= code1 读 master 写的总纲 `kernel-architecture.md`
- code2「跟 analysis」= code2 读 analysis 写的 `rounds/r<N>/direction.md`

**spawn 每个 subagent 时，prompt 里只给最小上下文**（不塞尝试历史）：
`AGENT_REPO=<本 agent 仓库绝对路径>`、`CAMPAIGN_DIR=<当前 /tmp/slug>`、**本轮轮号 N**、
目标 module、（code2 还要）**上一版文件 `v<N-1>` 路径**（它 cp 成 `v<N>` 再改）。其余一律让 subagent 从磁盘自己读。

### Agent 间信息交互：全靠文档（artifact 即契约）

**铁则：agent 之间的实质信息只通过磁盘文档（`.rlcr/current/` 下的 artifact）流动。**
spawn 的 `prompt` 与 subagent 的 return 只是**指针**（"叫谁去读哪份文档"），不承载实质
内容。**subagent 之间从不直接通信**——所谓"配对"都是"读对方写的文档"。这样才同时拿到
clean context、抗压缩、单一真相、互不直连。下表是**完整的读写契约**（谁写→谁读）：

| 文档（相对 `.rlcr/current/`） | 写者 | 读者 | 作用 |
|---|---|---|---|
| `kernel-architecture.md`（总纲） | master | code1 | 架构蓝图 + 4d-0 选定的原语 |
| `architecture-ledger.md` | master | master / analysis | 战略层反绕圈台账（试过的架构/ceiling/弃因） |
| `goal-tracker.md` | master | analysis | 绝对 roofline 目标 + baseline 参照 |
| `module-tracker.json` | master | master / analysis | 模块清单与完成状态 |
| `baseline-analysis.md` + `profiles/baseline.*` | analysis(baseline 模式) | master | 设计总纲的依据 |
| `decomposition.md` / `global-strategy.md` | analysis(Step 6) | master | 模块分解 + 全局策略 |
| `rounds/r<N>/direction.md` | analysis | code2 | **本轮精细方向**（核心配对） |
| `solution/<family>_v<N>.<ext>` + `rounds/r<N>/candidate.*`（ptx/cubin/sass/res-usage/nvdisasm/ncu-rep/metrics） | code1 / code2 | analysis | 本轮版本源码 + 5 类 SASS 产物 + NCU |
| `rounds/r<N>/analysis.md` | analysis | master / 下一轮 analysis | 完整证据 + verdict |
| `rounds/r<N>/summary.md` | analysis（从 git diff 重建） | master | 本轮 diff 统计 |
| `summary.md`（滚动索引，一行一轮） | analysis | master / analysis | 轨迹（平时扫它，不重读原始 dump） |
| living lessons（campaign 副本） | analysis（写/prune） | 全体 | 教训沉淀；master finalize 时回灌全局 |
| `state.md`（`当前轮: r<N>` 行 / verdict / 最新 NCU duration） | master（轮号行）+ analysis（verdict/duration） | 全体 + **hook** + SessionStart 进度卡 | 进度 / 抗压缩恢复 |
| `.initial-impl-done` / `.direction-read-marker`（机械 marker） | code1（建锁）/ hook（读 direction 时刷新） | **hook** | 防重写 / 先读方向 / SASS 门槛的判定依据 |

> 非文档的两个指针通道（不可消除的最小量）：**spawn prompt**（master→subagent，给路径+
> 轮号+目标）、**return text**（subagent→master，给 verdict/指向哪份文档，限字数）。
> 二者都**不复述历史**——要历史就读 `summary.md` / `analysis.md`。

### subagent 生命周期与上下文策略（保留 vs 故意丢弃）

「优化器不该每轮失忆，诊断器故意失忆」——两个角色的上下文策略**相反**，按需选定：

| 角色 | 跨轮上下文 | 寿命 | 为什么 |
|---|---|---|---|
| **code2** | **SendMessage 续用同一实例**（保留对 kernel 的「手感」：MODULE 边界、试过什么、寄存器预算、layout 怪癖） | **一个架构**（master re-arch 才丢弃、起新 code2） | 连续性有价值，每轮丢掉重读既浪费又丢细微判断 |
| **analysis（routine 轮间分析）** | **SendMessage 续用同一实例** | 一个架构 | 连续看「这轮 lever 有没有用 + 连续几轮趋势」更顺、更自然；带着对轨迹的理解判进退 |
| **analysis（枯竭/pivot 诊断）** | **起一个 fresh 实例** | 一次性 | 这是最怕「舍不得自己主意」的决策点；新鲜眼睛抗确认偏差/抗重复循环（IterKernel 自欺史需要），可兼作对抗式复核 |
| **code1** | 每次新实例 | 一个架构（一次性） | 每代架构本就是从零实现，无续用语义 |

> **对齐 auto-gpu-kernel**：它的主循环是**连续**做 routine 测/判的，只有 `research`
> agent（诊断 plateau / 是否 pivot）才 fresh。这里 routine analysis 续用 = 对应它的主
> 循环；pivot analysis fresh = 对应它的 research。不要把 freshness 套到每一轮。

- **续用不与 hook 冲突**：续用的 code2/analysis 每轮仍须 Read 本轮 `direction.md`/产物
  （hook 强制），marker 按文件 mtime 判，照常放行。
- **不变式（无论续用还是 fresh 都要守）**：`summary.md`（趋势）+ 每轮 `analysis.md`
  （附 NCU/SASS 证据）必须**自足到能重建轨迹**——因为 pivot 用 fresh 实例、且续用实例
  攒久了照样会被压缩、得从磁盘恢复。「fresh ≠ 看不见历史」：fresh analysis 靠**读**
  summary.md 趋势 + 下钻相关轮 analysis.md + `sass_hist_diff.sh` 看效果，而非靠记忆。

### 源文件版本命名（v<N>，按全局轮号）—— code1 / code2 都遵守

**每个产代码的轮 N 都把源码写成一个独立版本文件** `solution/<family>_v<N>.<ext>`
（C++ 路径 `.cu`，CuTe DSL 路径 `.py`）。**`v` 的数字 = 全局轮号 N**，与 `rounds/r<N>/`
一一对应（vN ↔ rN）。

- **round 1 = 初始从头实现（code1）= v1。** 之后：
  - **code2 渐进轮 N**：`cp solution/<family>_v<N-1>.<ext> solution/<family>_v<N>.<ext>`
    （Bash 建**新文件**，防重写 hook 放行新文件）→ 对 **v<N>** 做**一个 lever 的 Edit**
    （Edit 触发"先读方向"+SASS 门槛，纪律/门槛不丢）。"渐进"= `diff v<N-1> v<N>` 只含
    一个 lever（analysis 核对）。
  - **code1 re-arch 轮 N**：直接 Write 全新 `<family>_v<N>.<ext>`（FROM SCRATCH）。
- code2 **仍无 Write 工具**：靠 `cp`(Bash)+`Edit` 实现"复制上一版 + 改一处"，从工具层
  杜绝凭空整文件重写；no-rewrite 由 cp-自上版 + analysis 的"一 lever diff"核对共同保证。
- 每轮把 benchmark adapter / ncu runner 的 import 指向本轮 **v<N>**。
- 全部 v1..vn 留在 `solution/`（每轮源码都是一等文件）：跨轮直接 diff；finalize 按 NCU
  选最优 vN 即 `git checkout`/直接用该文件，无需 git 考古。

### 两个循环

```
外循环（战略，master 主导）
  analysis(读 baseline) → master 定总纲(Step 4d + 4d-ceiling) + 记 architecture-ledger
     → code1 从头实现(Step 5)
  ……（内循环若干轮）……
  analysis 报「接近枯竭」 → master 丢弃门槛判定 → 若丢弃：改总纲 + ledger → code1 重写新文件

内循环（战术，analysis↔code2，master 不插手）
  master 设 state.md「当前轮: r<N>」 → spawn code2(cp v<N-1>→v<N>, 读 r<N>/direction.md,
     对 v<N> 改一个 lever, 产物) → spawn analysis(读产物+diff v<N-1>↔v<N>, 写
     analysis.md/summary/r<N+1>direction, verdict)
     → master 读 analysis 的精简 return：CONTINUE 就继续内循环；「枯竭建议」才上浮到外循环
```

**反应式，不固定 cadence**：master 只在 analysis 报上「接近枯竭 / 触发 pathology」时
介入战略判断，**不每轮重度干预**内循环；不达标也不主动喊停（无轮次上限）。

### master 的「丢弃当前总纲」门槛（外循环的铰链）

analysis 只给**带证据的枯竭建议**；**扣丢弃扳机的是 master**（提方向的人不判自己方向
死活，避免确认偏差）。master 收到枯竭建议后，按下面两关判定，**双向**（可强制丢弃，
也可强制再磨）：

1. **judge by ceiling, not current**：比的是**架构的结构上限**（4d-ceiling），不是当前
   NCU 数字。新架构 r1 比成熟旧架构慢是正常的——只要它结构上限更高就值得换。**绝不**
   因「当前更慢」拒绝换架构，也**绝不**因「这轮退了一点」就丢弃（铁律#3：局部低谷≠死路，
   读全轨迹 `summary.md` 确认不是低谷）。
2. **pathology checklist**（逐条核，命中才有理由丢弃 / 才知道往哪重定）：
   - **错瓶颈**：在 memory-bound kernel 上磨 compute（或反之）——重判 bound 再定向。
   - **缺基本功**：warp-spec / TMA / ldmatrix / stream-K 等该上的核心技术没上 → 这不是
     「枯竭」，是总纲一开始没到 ceiling，应 re-arch 补齐。
   - **结构焊死**：瓶颈是吞吐 stall（非依赖气泡）/ 再加并发必 throttle / 更多累加器必
     spill——本方向结构不可约 → 换**根本不同的并发结构**。
   - **重复循环**：换名字绕回试过的架构（查 `architecture-ledger.md`）→ 禁止重试，另寻。
   - **过度工程**：复杂度本身堵死了进一步优化 → 简化或换路。
   - **正确性墙**：连续不同思路都精度/编译失败 → 多半是数值/算法,先解 correctness。

   **丢弃的硬证据**（缺一不准丢，对应 analysis 的枯竭门槛）：SASS 调度层证据(结构不可约)
   + 至少一次反向实验实测退化 + 当前 %roofline + 全轨迹（非低谷）。证据不足 → **强制
   再磨一轮**（让 analysis 写新 direction、code2 继续）。

   判丢弃 → 更新 `kernel-architecture.md`（写清旧架构上限为何不够 + 新架构如何达 ≥90%
   roofline）+ 追加 `architecture-ledger.md` 一条（旧架构、实测 ceiling、弃因、证据）
   → spawn **新 code1** 从头实现（新轮号文件，FROM SCRATCH，**保持锁、不 rm**）→ 回内循环。

### 抗压缩 / resume 约定（folder reservation）

某轮目录 `rounds/r<N>/` 里 `direction.md` 存在但 `analysis.md` 不存在 ⇒ **该轮已开未完**，
按 Step 7 的 a/b/c 续做未完成的产物，**不要重开新轮、不要跳过未生成的产物**。master
每轮把 `state.md` 的「当前轮: r<N>」与 verdict/最新 NCU duration 维护好（SASS 门槛 hook
与 SessionStart 进度卡都依赖它）。

### 新增 artifact（都在 `.rlcr/current/`，不 commit）

- `architecture-ledger.md`（master 持有）—— 战略层反绕圈记忆：试过哪些**架构/并发结构**、
  各自实测到的 ceiling、为何弃（附证据）。每次 re-arch 前必查、之后必追加。
- `summary.md`（analysis 维护）—— **一行一轮**的滚动索引：
  `r<N> | 改了什么 | NCU <µs> | <X>% roofline | <verdict>`。analysis/master 平时**扫它
  重建轨迹**，只在需要时下钻读某轮原始 dump（解决长 campaign 的 context 缩放）。
- per-campaign **living lessons**（analysis 维护）—— 发现某条 lesson 是死胡同/错误就
  **prune 掉**，沉淀有效的；master 在 finalize 时把可复用教训回灌 `AGENT_REPO/docs/
  kernel_optimization_lessons.md`。

### 角色 ↔ Step 映射

- **Step 1–2**（理解需求 / 建仓）：master。
- **Step 3 + 4a/4b/4c**（profile+分析 baseline）：master spawn **analysis**（baseline 模式）。
- **Step 4d + 4d-ceiling**（定总纲 + 结构上限门槛）：master，写 `kernel-architecture.md`
  + `architecture-ledger.md` 首条。
- **Step 5**（首次从头实现 = **round 1 = v1**）：master spawn **code1**（写 `<family>_v1`）。
- **Step 6**（profile v1 + 模块分解 + 写**首个迭代轮** `rounds/r2/direction.md`）：master
  spawn **analysis**（v1 的分析落 `rounds/r1/`；r1 是初始实现轮）。
- **Step 7 内循环**（从 round 2 起）：每轮 master 设 state 轮号 → spawn **code2**
  （cp v<N-1>→v<N> + 7a 改一 lever + 7b 产物）→ spawn **analysis**（7b 解读 + 7c
  分析/verdict/下轮 direction）。verdict=CONTINUE 继续；枯竭建议上浮到 master 丢弃门槛。
- **Step 8**（集成）/ **Step 9**（finalize，按 NCU 选最优）/ **Step 10**（报告）：master
  （集成的逐模块退化检查可 spawn analysis 复核）。

> 下面 Step 1–10 的所有硬纪律（FROM SCRATCH、防重写、NCU 权威、SASS 门槛、4d-ceiling、
> 90% roofline、退化不回退）**全部不变**，只是分摊到上述角色执行。各角色契约见
> `.claude/agents/{analysis,code-impl,code-iter}.md`。

---

## 前置：读取规则和知识

在做任何事情之前，必须读取以下文件（路径相对 `AGENT_REPO`）。**分工**：master 必读
1–4（护栏/契约/教训，战略判断要用）；NCU/PTX/KernelWiki 的深读由 **analysis/code**
subagent 在各自契约里按需做（master spawn 时把 `AGENT_REPO` 路径传给它们即可，不必自己
通读 5–7）：

1. `docs/kernel_optimization_rules.md` — 优化护栏
2. `docs/benchmark_contract.md` — benchmark 方法论
3. `docs/correctness_contract.md` — 正确性要求
4. `docs/kernel_optimization_lessons.md` — **历史经验教训，包含 fragment layout、swizzle trade-off、SM 架构能力、调试策略等，必须在实现前阅读以避免重复踩坑**
5. `external/ncu-report-skill/SKILL.md` — **NCU profiling 方法论，所有 ncu 命令必须遵循**
6. `external/KernelWiki/SKILL.md` — **Blackwell/Hopper kernel 优化知识库，架构设计和瓶颈诊断必须查询**
7. `external/CudaSkill/cuda_skill/references/ptx-isa.md` — **PTX ISA 文档搜索入口**
   （完整文档 `external/CudaSkill/cuda_skill/references/ptx-docs/`）。无需通读，
   但**分析 PTX/SASS 和设计/修改 PTX 指令时必须按需查阅**（见 Step 4b/4c、7c）

如果 5 或 6 不存在，报告错误并停止。若 7 不存在（子模块未初始化），
回退到在线 PTX ISA 文档查询。

---

## Step 1: 理解需求

从用户描述中提取：
- **Kernel 类型**：什么算子（FP8 GEMM、GroupNorm+SiLU、FlashAttention 等）
- **目标 GPU 与 arch（必须实测，禁止硬编码/猜测）**：**一律先运行 `nvidia-smi`
  查清楚型号与 compute capability**，即使用户给了型号也要核对：
  ```bash
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
  ```
  把 compute_cap（形如 `12.0`）转成**完整 nvcc arch 串**：去掉点 → `sm_120`，
  再补架构专用后缀 `a` → **`sm_120a`**（TMA / ldmatrix / fp8 mma 等指令需要
  `sm_XXXa` 变体）。把检测到的 `target_gpu` 与 `arch`（如 `sm_120a`）写进
  `config.toml`，**后续所有 `nvcc -arch=<ARCH>` 都直接用这个完整串**（不要再拼
  `sm_` 前缀，避免 `sm_sm_120a`）。绝不沿用模板里的占位/示例值。
- **Workload shapes**：如果没给，根据 kernel 类型生成常见 production shapes
- **Baseline**：对标**当前最强的现成库实现**。实测对比 PyTorch（cuBLAS，如
  `torch._scaled_mm`/`torch.mm`）与 **FlashInfer 库**（AOT 预编译）两条路径,
  取**更快**的那个当 baseline,在 `docs/baseline_source.md` 记录两者实测延迟、
  版本、入口与选择理由。不得用弱 baseline 取巧。baseline 与 candidate 必须对称
  ABI/计时（均 destination-passing,无单边多余开销）。
- **特殊约束**：dtype、精度要求、是否 fused 等
- **完成目标（PRIMARY，必须显式设定并写进 goal-tracker）**：**达到该 shape 的
  roofline 上限的 ≥90%**（roofline efficiency ≥ 90%）。roofline 上限 =
  `min(compute 峰值, 带宽×算术强度)`，**先判这个 shape 是 compute-bound 还是
  memory-bound**（算 AI 与脊点 `峰值/带宽` 比大小）：
  - **compute-bound**（AI > 脊点）→ 上限 = spec 峰值，目标 = 90% spec 峰值。
    例：sm_120 FP8 峰值 516 TF、M=1024/N=10240/K=4096 的 AI≈1283 ≫ 脊点≈384 →
    compute-bound → 目标 ≥464 TF / ≤t_floor/0.9。
  - **memory-bound**（AI < 脊点）→ 上限 = 带宽×AI，**低于 spec 峰值**；目标 =
    90% 的 memory roofline（µs 下限 = 搬运字节数 / 带宽 / 0.9）。**此时绝不能拿
    "90% spec 峰值"当目标——它物理不可达。**
  - 即：「90% spec 峰值」只是 compute-bound 时的特例，通用判据始终是「90% roofline」。
  - **这是完成判据,不是 baseline。** baseline(最强现成库)只是**对比参照**,
    用来判断"赢没赢现成实现",**不是完成线**。
  - **关键:90% roofline 上限常常 > baseline 实测效率**(库实现往往只到上限的
    85-90%)。所以「打平 baseline」**通常不等于**「达到 90% roofline 上限」——
    后者更高,可能要求**超过 baseline**。两个数都要在 goal-tracker 里写清楚
    (目标=90% roofline 上限;参照=baseline),**不要把"打平 baseline"误当成完成**。
  - 若实测发现连 SOTA 库都远低于 90%(如 88%),说明 90% 对该 shape 可能触及
    物理上限:此时如实告知用户"90% 可能不可达、当前 SOTA=X%",由用户决定是否
    放宽目标——但在得到用户确认前,仍以 90% roofline 上限为目标继续推进(wave-quant/
    stream-K、调度等所有杠杆都要试)。

如果 kernel 类型不明确，直接问用户。

---

## Step 2: 创建任务目录（独立 git 仓库）

每个 campaign 使用独立 git 仓库，不在 agent 仓库中提交 kernel 代码。

```bash
TASK_SLUG="<gpu>_<kernel>__<shape_desc>"

# 1. 在 /tmp/ 下创建独立 repo
mkdir -p /tmp/$TASK_SLUG
cd /tmp/$TASK_SLUG
git init   # 先 init

# 2. 只拷"目录骨架 + .gitignore"，**不要**拷 prompt.md / config.toml
#    （这两个文件你马上要从零写成真正内容；如果先 cp 占位符过来，它们就成了
#     "已存在文件"，之后用 Write 覆盖会被 "File has not been read yet" 拦住，
#     白白多一次 Read 仪式。所以这里直接不拷它们，用 Write 新建即可。）
cp $AGENT_REPO/templates/example_task/.gitignore /tmp/$TASK_SLUG/
mkdir -p /tmp/$TASK_SLUG/{baseline,bench,solution,docs}
#    .gitignore 已就位：.rlcr/ 等过程产物本地保留、不提交

# 3. 在 agent 仓库保留空目录结构（仅 .gitkeep）
mkdir -p $AGENT_REPO/campaigns/operators/$TASK_SLUG/{baseline,bench,solution,docs}
touch $AGENT_REPO/campaigns/operators/$TASK_SLUG/{baseline,bench,solution,docs}/.gitkeep
```

用 **Write 新建** `prompt.md` 和 `config.toml`（参考 `templates/example_task/`
里的同名文件作为格式样板，但不要 cp 过来再覆盖）。

> **关于 "File has not been read yet" 报错**：Write 工具对**已存在**的文件会要求
> "先用 Read 工具读过才能覆盖"（`cat` 不算，它只认 Read 工具的调用记录）。所以
> 规则是：**新建文件 → 直接 Write；要改已存在的文件 → 先 Read 再 Write，或干脆用
> Edit 做增量修改。** 本步把 prompt.md/config.toml 留给 Write 新建，正是为了绕开
> 这个无谓的摩擦。

创建 RLCR 状态目录：

```bash
mkdir -p .rlcr/current/rounds .rlcr/current/profiles
```

创建以下文件：
- `.rlcr/current/plan.md` — 优化计划
- `.rlcr/current/goal-tracker.md` — 目标追踪（目标=90% roofline 上限；参照=baseline）
- `.rlcr/current/module-tracker.json` — `{ "modules": [], "completedModules": [] }`
- `.rlcr/current/state.md` — 当前阶段（含「当前轮: r<N>」权威轮号行，由 master 每轮维护）
- `.rlcr/current/architecture-ledger.md` — **（master 持有）战略层反绕圈台账**：试过哪些
  架构/并发结构、各自实测 ceiling、弃因+证据。初始为空，每次 re-arch 前查、后追加。
- `.rlcr/current/summary.md` — **（analysis 维护）一行一轮的滚动索引**，初始只有表头
  `r<N> | 改了什么 | NCU(µs) | %roofline | verdict`。

在独立 repo 中 git commit。后续所有 kernel 代码修改、benchmark 结果都在此 repo 中提交。

---

## Step 3: Profile Baseline

### 3a. NCU 实测数据

1. 写 `.rlcr/current/profiles/ncu_baseline_runner.py`
2. `ncu --print-summary per-kernel -c 1` 发现 kernel 名
3. `ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters -k "regex:<NAME>" -c 1 -o .rlcr/current/profiles/baseline`
4. `ncu --import .rlcr/current/profiles/baseline.ncu-rep --page details > .rlcr/current/profiles/baseline-details.txt`
5. `ncu --import .rlcr/current/profiles/baseline.ncu-rep --csv > .rlcr/current/profiles/baseline-metrics.csv`
6. `python bench/benchmark.py --device cuda:0`

### 3b. 静态代码分析（PTX / SASS / 汇编）

读 `config.toml` 获取 arch（完整 nvcc 形式，如 `sm_120a`，由 Step 1 的
`nvidia-smi` 检测写入）。找到 baseline kernel 源文件。下面命令里的 `<ARCH>`
就是这个完整串，直接 `-arch=<ARCH>`（即 `-arch=sm_120a`）。

```bash
# PTX 中间表示
nvcc -ptx -lineinfo -arch=<ARCH> <source.cu> -o .rlcr/current/profiles/baseline.ptx

# Cubin 二进制
nvcc -cubin -lineinfo -arch=<ARCH> <source.cu> -o .rlcr/current/profiles/baseline.cubin

# SASS 反汇编（GPU 原生指令）
cuobjdump -sass .rlcr/current/profiles/baseline.cubin > .rlcr/current/profiles/baseline-sass.txt

# 寄存器 / shared memory 资源使用
cuobjdump -res-usage .rlcr/current/profiles/baseline.cubin > .rlcr/current/profiles/baseline-res-usage.txt

# 详细反汇编（含控制流、predicate、barrier 信息）
nvdisasm -gi -sf .rlcr/current/profiles/baseline.cubin > .rlcr/current/profiles/baseline-nvdisasm.txt
```

如果 baseline 是 library kernel（FlashInfer/CUTLASS 预编译），无法从源码编译时，
用 `cuobjdump -sass` 直接从 .so 中提取 SASS：
```bash
cuobjdump -sass -fun <kernel_name> <library.so> > .rlcr/current/profiles/baseline-sass.txt
```

---

## Step 4: 分析 Baseline + 设计 Kernel 架构

**查询 KernelWiki**：用 `python3 external/KernelWiki/scripts/query.py` 搜索相关
kernel 案例和优化技术。

### 4a. NCU 实测分析

读取 NCU 数据（baseline-details.txt、baseline-metrics.csv），分析：
- Primary bound（compute/memory/latency/barrier）— 引用具体 NCU metric 值
- SM throughput、DRAM bandwidth、L2/L1 hit rates
- Tensor Core utilization（`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`）
- Warp stall 分布（`smsp__pcsamp_warps_issue_stalled_*`）
- Achieved occupancy vs theoretical occupancy
- Pipeline overlap 质量（compute/memory overlap ratio）

### 4b. PTX/SASS 静态分析

读取 baseline-ptx、baseline-sass.txt、baseline-res-usage.txt、baseline-nvdisasm.txt，分析：
- **寄存器压力**：每线程寄存器数、是否有 spill（SASS 中的 `STL`/`LDL` 指令 = local memory spill）
- **指令统计**：总 SASS 指令数、指令类型分布（compute/memory/control/sync）
- **Tensor Core 指令**：HMMA/UMMA/WGMMA 指令出现频率和位置
- **内存访问模式**：LDG/STG 指令的 cache 修饰符（`.L1`/`.L2`/`.CONSTANT`）、向量化宽度（`.64`/`.128`）
- **Bank conflict 风险**：LDS/STS 指令的 stride 模式
- **同步指令**：`BAR.SYNC`、`MEMBAR`、`mbarrier` 相关指令的位置和频率
- **Dual-issue 机会**：连续独立指令的 scheduling 质量
- **循环结构**：PTX 中的分支和循环展开程度
- **Predicated execution**：条件执行指令的比例

**查 PTX ISA 文档**：分析中遇到任何语义/约束不确定的指令（MMA fragment
layout、cp.async/TMA、ldmatrix、mbarrier、cache 修饰符、指令的 SM 版本支持等），
必须到 `external/CudaSkill/cuda_skill/references/ptx-docs/`（入口
`ptx-isa.md`）查清楚，不靠猜测或 CuTe/CUTLASS 抽象推断（见
`docs/kernel_optimization_lessons.md` §1、§2）。把查到的依据写进分析文档。

### 4c. 综合诊断

综合 NCU 实测 + PTX/SASS 静态证据，定位：
- Baseline 做得好的地方（学习）
- Baseline 做得差的地方（改进机会），每条必须有 NCU metric 或 SASS 指令证据

写 `.rlcr/current/baseline-analysis.md`（必须包含具体数值，不允许模糊描述）。

### 4d-0. 实现原语选择（先评估算子复杂度，再选语言/原语）

**candidate 不强制 CUDA**。在设计架构前，master 先评估**算子复杂度 + 达到 ≥90%
roofline 所需的抽象层级**，在下面三者中选一（选择理由写进 `kernel-architecture.md` +
`architecture-ledger.md`，并把所选原语与对应 **build/profile 命令**写进 `config.toml`）：

| 原语 | 何时选 | 控制力 / 工作量 |
|---|---|---|
| **纯 CUDA + PTX 薄封装** | 算子结构简单~中等；或瓶颈在某条指令的手工调度（需要库藏不住的指令级控制）；或算子不贴合标准 GEMM/conv 模板 | 控制力最强 / 工作量最大（项目原生强项） |
| **CUTLASS**（C++ 模板：Collective/Builder/GemmUniversal/epilogue fusion/CuTe layout 代数均**允许**） | 算子是标准/近标准 GEMM/conv/attention，CUTLASS 已有逼近 SOTA 的构件，手写多阶段 pipeline/warp-spec/TMA 成本过高 | 控制力中 / 工作量低、起点高 |
| **CuTe DSL**（Python） | 要 CUTLASS 级抽象（layout / copy & MMA atom / pipeline），但需要比 C++ 模板更灵活的自定义 fusion 或更快迭代，且裸 PTX 不现实 | 控制力中高 / 迭代最快 |

> Triton 不在候选集。选定后**这一代架构内统一用该原语**，不要半路混搭；**re-arch
> 可以换原语**（在 ledger 写明换因）。
>
> **无论选哪个，下列纪律全部不变**：FROM SCRATCH（不接任何已有实现为代码起点——
> 「从零」= 用所选原语**自己搭**，CUTLASS 路径=自己用 CUTLASS 构件组装而非复制别人的
> CUTLASS kernel）、每轮 5 类 SASS 产物 + NCU 实测、渐进 Edit / MODULE 边界 / 一轮一
> lever、退化不回退、≥90% roofline 完成判据。
>
> **方法学随原语自适应**：SASS 静态分析（`cuobjdump`/`nvdisasm`）对三者都成立（CuTe
> DSL 从 JIT 产出的 cubin 提取）；但 build/profile 具体命令按 `config.toml`——C++ 用
> `nvcc`，CuTe DSL 用其 JIT 后再 dump cubin。**详细原语约束见 Step 5「代码约束」。**

### 4d. 设计 Kernel 架构

> **核心原则：第一步就把设计做到上限（design to the ceiling）。**
> 目标是 **≥90% roofline 上限**（compute-bound 时即 90% spec 峰值；memory-bound
> 时为 90% 的 memory roofline，见 Step 1）（不是"追平 baseline"——目标本就高于
> baseline）。初始架构必须直接奔着这个上限去——**从第一版就采用达到 ≥90% 上限
> 所必需的全部核心技术**（如 warp specialization、ldmatrix、TMA、最优 tile/swizzle、
> 异步流水线、stream-K 消除 wave quantization 等）。**严禁**先设计一个"correctness-first 的简单版"再指望靠 RLCR
> 渐进爬上去：核心架构技术（warp 角色划分、ldmatrix vs 手写 LDS、同步机制）是
> **架构骨架,不是后期能 bolt-on 的 tweak**——简单架构的效率天花板是焊死的,
> 后续每轮只会在那个被焊死的上限里找局部最优,永远赢不了。RLCR 迭代是在一个
> **已经逼近上限**的架构上做精调,不是从 0.45x 往上挪。

基于分析结果设计新 kernel 架构：
- Tile sizes、CTA shape、warp layout（含 warp specialization：producer/consumer 角色划分）
- Pipeline structure、async loading（cp.async / TMA / mbarrier 编排）
- Fragment 加载方式（ldmatrix 优先；手写 LDS 仅在 ldmatrix 不适用且已验证无 bank conflict 时）
- Shared memory layout（含 swizzle/padding 策略避免 bank conflict）
- Key PTX 指令选择（引用 SASS 分析中发现的瓶颈；先做 ldmatrix/mbarrier/TMA 的编译可行性验证）
- Module decomposition（`// MODULE: <id>` 标记）
- 寄存器预算（参考 baseline res-usage 设定目标）

#### 4d-ceiling. 结构上限分析（强制门槛，不可跳过）

在 roofline（硬件算力/带宽下限）之外，**必须额外推导"所选候选架构本身的效率
上限"**，并与 **≥90% roofline 上限这个目标**对比（baseline 只是必须超过的下限
参照,不是目标线）：

1. **硬件 roofline**：先判 compute-bound 还是 memory-bound（算 AI vs 脊点,见
   Step 1），取 `min(compute 峰值, 带宽×AI)` 作为 roofline 上限。算出 **90%
   roofline 上限对应的目标 TF / µs**。
2. **结构上限（structural ceiling）**：**这个具体架构**最多能到 roofline 上限的
   百分之几？逐项问（每条扣多少效率,凑出结构上限百分比）：
   - 加载与计算是否被 per-step 全块 barrier 串行化？（→ 上限被 barrier 压低）
   - fragment 取数有无 bank conflict / 是否用了 ldmatrix？
   - occupancy / 寄存器墙能否藏住 MMA 延迟（运行时 wait）？
   - **wave quantization**：#CTA 能否整除 #SM？凑不齐就要 stream-K/persistent,否则
     尾波损失（如本例 110=2·5·11,而 2^a·5^b 的 tile 永远凑不成 110 倍数 → 必失 ~3%）。
   - 还缺哪些把利用率推到 90% roofline 上限的技术？
3. **决策门槛**：若 `结构上限 < 90% roofline 上限`，则**当前设计达不到目标——禁止
   进入 Step 5**。必须回到本步重新设计,补齐使能技术,直到结构上限 ≥ 90% roofline
   上限,再实现。（同时结构上限必须 > baseline,否则连现成实现都赢不了。）
4. 若判断"达到 90% roofline 上限必须做重写级工作"（如 warp-specialized + ldmatrix
   + stream-K 从头实现），**在此处就明确写出来并告知用户**工作量与取舍。若连 SOTA
   库都远低于 90%(实测得知),说明 90% 可能触及该 shape 物理上限——如实告知用户当前
   SOTA 百分比,由用户决定是否放宽,在确认前仍以 90% roofline 上限为目标。

把硬件 roofline + 结构上限 + 决策结论写进 `.rlcr/current/kernel-architecture.md`。

> **关于"渐进式硬约束"的边界（重要澄清）**：全局铁律的"严禁重写"管的是
> **在一个既定架构内迭代时**的纪律（别用 Write 覆盖、别一报错就整文件重写）。
> 它**绝不意味着**"架构选错了也只能将就"。当结构上限分析（4d-ceiling）或 Step 7
> 的迭代证据表明**整体思路/架构本身赢不了**时，正确动作是
> **STRATEGY_REVISION → 重新设计架构 → 从头实现一版新 candidate**（这是合法且
> 必要的，不算违反渐进式约束；见 Step 5 关于重新实现的说明）。

写 `.rlcr/current/kernel-architecture.md` 和 `.rlcr/current/direction.md`。

git commit。

---

## Step 5: 实现完整 Kernel

> **执行者：master spawn `code-impl`（code1）。** 本步全部纪律即 code1 契约。master 只
> 负责把总纲 `kernel-architecture.md` 备好、传 `AGENT_REPO`/`CAMPAIGN_DIR`/轮号，然后读
> code1 的 return（新文件路径 / correctness / 初始 NCU / 总纲是否全落地）。

### 代码约束（按 4d-0 选定的原语，三选一；本节是权威清单）

实现语言/原语**不固定 CUDA**，按 Step 4d-0 评估算子复杂度后选定的那一个执行。
Triton 不在候选集。无论哪条路径都必须 FROM SCRATCH（不复制/不继承任何已有实现）。

**A. 纯 CUDA + PTX 薄封装**（默认强项路径）
- CUDA C++ + 裸 PTX inline assembly（TMA、WGMMA/UMMA、mbarrier、fence）
- DeepGEMM 风格薄封装（一个 inline function = 一条 PTX 指令）
- 此路径**禁止**退回高层模板：`cutlass::*` 的 Collective/Builder/GemmUniversal*/
  epilogue CollectiveBuilder、`using namespace cute`、任何 CuTe layout 代数、
  `#include "cute/*.hpp"`（`cutlass/numeric_types.h` 仅作 dtype 定义可用）。
  —— 既然选了"纯手写"，就不要半路混进 CUTLASS/CuTe 抽象。

**B. CUTLASS**（C++ 模板）
- **允许**使用 CUTLASS 的 Collective/Builder、GemmUniversal*、epilogue fusion、
  CuTe layout 代数等全部构件——这正是选它的目的：用现成高性能构件快速逼近 ceiling。
- 仍须 FROM SCRATCH：自己用 CUTLASS 构件**组装**本算子的 kernel，**不复制**他人/旧
  campaign 现成的 CUTLASS kernel 文件再改。
- 调参（tile/stage/cluster/epilogue 等模板参数）按渐进纪律：一轮一 lever、Edit、
  MODULE 边界。

**C. CuTe DSL**（Python）
- 用 CuTe DSL 的 layout / copy & MMA atom / pipeline 等抽象自定义实现。
- FROM SCRATCH 同上：自己写，不接已有 .py 实现。
- 源文件是 `.py`；build/profile（JIT 后 dump cubin、`cuobjdump`/`nvdisasm` 取 SASS）
  命令以 `config.toml` 声明为准。

> 三条路径共有：每轮 5 类 SASS 产物 + NCU（SASS 对 JIT 产物同样提取）、渐进 Edit /
> 一轮一 lever、退化不回退、≥90% roofline。**`config.toml` 必须记录所选原语与对应
> build/profile 命令**，下文及 code agent 里写 `nvcc …` 处，按 config.toml 实际命令执行。

### 实现

1. 按 `direction.md` 用 **4d-0 选定的原语从头**实现完整 kernel，写在 `solution/`
   （新开空文件,不复制/不继承任何已有 kernel——见全局铁律 -2 FROM SCRATCH）
2. 插入 `// MODULE: <id> BEGIN/END` 标记
3. 写 benchmark adapter
4. `python bench/benchmark.py --correctness-only` — 正确性全部通过（正确性 gate 已
   内置在 benchmark.py：poison + oracle compare；`--correctness-only` 只跑这步、跳过计时）
5. `python bench/benchmark.py` — 记录结果
6. git commit: "initial kernel implementation"
7. **创建渐进式修改锁**：`touch .rlcr/current/.initial-impl-done`
   - 此 marker 一旦存在，项目 hook（`.claude/hooks/block_solution_rewrite.py`）
     会强制两条规则（基于磁盘状态，**context 压缩也冲不掉**）：
     - **防重写**：拦截对该 campaign `solution/` 的 Write 覆盖，以及 shell
       重写（`>`/`>>`/`tee`/`sed -i`/`truncate`/`dd` 重定向到 solution/）。
     - **先读方向再改**：迭代期对 locked `solution/` 的 Edit，必须先 Read 当前轮
       的 `rounds/r<N>/direction.md`（最新的那个）才放行；没读会被拦。
     - marker 不存在时两条都不生效（首次实现照常用 Write）。
   - 后续所有 solution/ 改动必须用 Edit；动手前先读本轮 direction（hook 会强制）
8. 写 `.rlcr/current/initial-implementation-summary.md`

### 重新实现（re-architecture，当 4d-ceiling 或 Step 7 判定需换架构时）

这是合法且必要的，不是"被禁止的重写"。re-architecture = 写一个**全新的源文件**。
**锁全程保持上锁,不要 `rm`** —— hook 已改为「只拦覆盖已存在文件、放行新文件」,
所以写新架构文件本来就放行,无需也**不要**摘锁(摘了忘补回来 = 纪律静默失效,
这正是历史踩过的坑)。流程：
1. 先更新 `.rlcr/current/kernel-architecture.md`：写清"为何旧架构上限不够"
   （引用结构上限分析 + 实测证据）与新架构如何达到 ≥90% roofline 上限目标。
2. 新架构**直接 Write 一个新源文件**（锁开着也放行,因为是新文件不是覆盖），
   文件名按**版本=全局轮号**命名 `<family>_v<N>.<ext>`（如 re-arch 在第 8 轮就叫
   `<family>_v8.cu`，保持"文件名↔轮次"单调对应；见编排节「源文件版本命名」）。
   `v` 的数字必须 = 轮号 N，**不要**用与轮次脱钩的随意编号，**也不要** `rm`/`touch`
   那个锁。保留旧版本文件供对比；绝不 Write 覆盖旧文件本身（覆盖已存在文件仍被 hook 拦）。
3. 把 candidate ABI / adapter 切到新文件；旧文件在新版验证更快后用 `git rm` 删除。
4. 新文件插 MODULE 标记，跑 correctness + benchmark，写 `rounds/r<N>/` 的
   direction/summary/analysis + 5 类 SASS 产物，commit "re-architecture: <新架构>
   (initial)"，然后对**新文件**进入 Step 6/7 的渐进迭代（锁一直在,hook 全程强制）。

---

## Step 6: Profile 新 Kernel + 模块分解

> **执行者：master spawn `analysis`。** 初版 = round 1 = v1，其分析落 `rounds/r1/`。
> analysis 读 v1 产物做对比+模块分解，写 `decomposition.md`/`global-strategy.md` 与
> **首个迭代轮** `rounds/r2/direction.md`（round 2 起进内循环）；master 据其 return 更新
> `module-tracker.json` 后进入 Step 7 内循环。

### 6a. NCU 实测

1. 写 `.rlcr/current/profiles/ncu_candidate_runner.py`
2. NCU profile：
   ```bash
   ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters \
     -k "regex:<NAME>" -c 1 -o .rlcr/current/profiles/initial \
     python .rlcr/current/profiles/ncu_candidate_runner.py
   ncu --import .rlcr/current/profiles/initial.ncu-rep --page details > .rlcr/current/profiles/initial-details.txt
   ncu --import .rlcr/current/profiles/initial.ncu-rep --csv > .rlcr/current/profiles/initial-metrics.csv
   ```

### 6b. 静态代码分析

```bash
nvcc -ptx -lineinfo -arch=<ARCH> <source.cu> -o .rlcr/current/profiles/initial.ptx
nvcc -cubin -lineinfo -arch=<ARCH> <source.cu> -o .rlcr/current/profiles/initial.cubin
cuobjdump -sass .rlcr/current/profiles/initial.cubin > .rlcr/current/profiles/initial-sass.txt
cuobjdump -res-usage .rlcr/current/profiles/initial.cubin > .rlcr/current/profiles/initial-res-usage.txt
nvdisasm -gi -sf .rlcr/current/profiles/initial.cubin > .rlcr/current/profiles/initial-nvdisasm.txt
```

### 6c. 对比分析 + 模块分解

分解前先读 `docs/module_decomposition_guide.md`（分解原则、典型 GEMM/Attention/
Reduction 分解、共享资源识别、优化顺序）。

1. NCU metrics 对比：initial vs baseline（throughput、bandwidth、stalls、TC util）
2. SASS 对比：指令数、寄存器数、spill 数、循环结构差异
3. 验证 MODULE 标记，用 NCU source-level 数据算每模块 runtime fraction
4. 写 `.rlcr/current/decomposition.md`
5. Gap analysis → 每模块瓶颈定位（NCU 证据 + SASS 证据）
6. 全局优化策略 → 写 `.rlcr/current/global-strategy.md`
7. 写**首个迭代轮**方向 `.rlcr/current/rounds/r2/direction.md`（round 1=初始 v1；首个
   迭代轮 = round 2 → code2 产出 v2。在文档内标明本轮目标模块）
8. 更新 `module-tracker.json`，git commit（只提交 `solution/` 代码与 `docs/`）

---

## Step 7: 模块循环 — RLCR 迭代

> **执行者：内循环由 master 编排 `code-iter`（code2）↔ `analysis`。** 每轮 master：
> ① 设 `state.md`「当前轮: r<N>」②spawn code2（7a 改一个 lever + 7b 产物）③spawn
> analysis（7b 解读 + 7c 写 analysis/summary/下轮 direction + verdict）④读 analysis 的
> 精简 return。verdict=CONTINUE → 下一轮；analysis 报「接近枯竭」→ 上浮到「多 agent
> 编排」节的 **master 丢弃门槛**（ceiling 判据 + pathology checklist）。下面 7a/7b/7c 的
> 纪律即 code2/analysis 各自契约的镜像。

按 suggestedOrder 对每个模块循环。**无轮次上限**——只要 roofline 未达 90% 且
仍有可尝试的方向，就继续优化；某模块卡住就转下一个模块或拓宽搜索，仅在目标
达成或用户明确叫停时停止。轮次用**全局递增编号 N**（跨模块连续，不按模块分
目录；目标模块写进文档内容里）。

**每轮一个目录（本地，不 commit）**：本轮全部产物放进 `.rlcr/current/rounds/r<N>/`：
- 文档：`direction.md`、`summary.md`、`analysis.md`
- profile / 静态分析：`candidate.ptx`、`candidate-sass.txt`、`candidate.cubin`、
  `candidate-res-usage.txt`、`candidate-nvdisasm.txt`、`candidate.ncu-rep`、
  `candidate-details.txt`、`candidate-metrics.csv`

`.rlcr/` 整个被 `.gitignore`，所以这些分析记录与数据**只留本地、不进 git**；
**每轮只 commit `solution/` 代码**（交付物 `docs/results.md` 在 Finalize 时提交）。

每轮你直接依次完成三件事：

### 7a. 实现优化（渐进式修改，严禁重写）

**核心约束：每轮只做增量修改，不重写文件**

1. 读 `rounds/r<N>/direction.md`（**必读**：hook 会拦截"未读方向就
   Edit solution/"，没读这一步后面改不动）
1.5. **本轮一开始就把 `.rlcr/current/state.md` 的当前轮号设为 N**——写一行
   **精确格式 `当前轮: r<N>`**（如 `当前轮: r8`）。SASS 门槛 hook 读这一行作为
   **权威轮号**，据此检查"上一轮 r\<N-1\> 的 `candidate-sass.txt` 是否就绪"；不再
   靠目录排序猜。**轮号必须在改 solution/ 之前就更新好**，否则 hook 会按上一轮的
   号判定、检查错对象（这正是历史上 SASS 被绕过的根因）。
2. 如有上轮 P0/P1 issues 先修复
3. **修改前**：
   - `cp solution/<family>_v<N-1>.<ext> solution/<family>_v<N>.<ext>`（建本轮版本文件），
     Read v<N>，定位 `// MODULE: <id> BEGIN` 到 `// MODULE: <id> END` 的行范围
4. **修改时**：
   - **只对 v<N> 用 Edit 工具**做针对性修改（code2 无 Write 工具；**不从零重写整文件**）
   - 主改动在目标 MODULE 内；MODULE 外的改动必须是被主改动**因果驱动**的联动：
     - ✅ 共享 helper 函数签名/实现变更（被本模块调用）
     - ✅ shared memory 总量、launch config（smem_size、grid/block）
     - ✅ 数据流接口适配（上下游模块的读写格式跟着变）
     - ✅ pipeline 编排联动（prologue 改了 stage 数，mainloop/epilogue 的 barrier 跟着调）
     - ✅ 寄存器策略全局调整（occupancy 变化导致）
     - ❌ 与本轮优化目标无因果关系的代码改动
   - 每次 Edit 只改一个逻辑点（一条优化策略），不要一次改多个不相关的地方
5. **修改后验证**：
   - **`diff solution/<family>_v<N-1>.<ext> solution/<family>_v<N>.<ext>`** — 检查本轮所有
     改动（v<N> 对 git 是整文件新增，故用版本间 diff，不用 `git diff`）
   - MODULE 内的改动：正常
   - MODULE 外的改动：**每一处都必须在 rounds/r<N>/summary.md 中说明因果关系**（"改了 X 是因为模块内改了 Y，导致 Z 接口不兼容"）。无法说明因果关系的外部改动 → 撤销那处 Edit
   - `python bench/benchmark.py --correctness-only` — 正确性必须通过（**这一步是
     gate：错的代码不能进入 profile/benchmark**）
   - `python bench/benchmark.py` — 仅作**粗筛 sanity**（量级是否合理、有没有跑飞），
     **不在此处下"快了/慢了"的结论**。wall-clock 含 dispatch/包装层开销，不作性能
     判据（铁律 -0.5）。
6. **Regression check（分析，不回退）—— 判据只用 NCU**：本轮 vs 上轮的"快了/
   慢了/持平"**一律以 7b 的 NCU kernel duration（`gpu__time_duration`）为准**，
   在 7c 完成 NCU 实测后才下结论，**不得用 `bench/benchmark.py` 的 wall-clock 判
   进退**。
   - 如果 NCU duration 较上轮上升 > 5% 且不在预期内（direction.md 未预测到），
     立即 `git diff` 分析原因并写进 analysis.md。**但不回退**（铁律 #3）——commit
     本轮，继续前进；下一轮可在此基础上叠加（局部下降常被后续修改转正）。最优版
     在 Finalize 按 NCU 选出。
   - 仅当某改动**破坏正确性**时才 `git checkout`（错误恢复流程，铁律 #5 例外）。
   - 注意时序：本步只是"分析判进退"的占位说明，真正的数值对比发生在 7b（跑 NCU）
     之后的 7c；commit（第 7 步）可以先做（git 历史即安全网），进退结论落在 7c。
7. git commit: "r<N> (<id>): <描述>" — **只提交 `solution/` 代码**（`.rlcr/` 不进 git）
8. 写 `rounds/r<N>/summary.md`，其中包含本轮 diff 统计（改了哪些文件、多少行）

### 7b. Profile（NCU 实测 + 静态分析）

本轮所有 profile/静态分析产物都写进本轮目录 `.rlcr/current/rounds/r<N>/`
（本地，不 commit）。**5 类静态产物每轮都必须生成**，不可省略。

```bash
mkdir -p .rlcr/current/rounds/r<N>
RD=.rlcr/current/rounds/r<N>

# NCU 实测
ncu --set full --section PmSampling --section PmSampling_WarpStates --section SourceCounters \
  -k "regex:<NAME>" -c 1 -o $RD/candidate \
  python .rlcr/current/profiles/ncu_candidate_runner.py
ncu --import $RD/candidate.ncu-rep --page details > $RD/candidate-details.txt
ncu --import $RD/candidate.ncu-rep --csv > $RD/candidate-metrics.csv

# 静态代码分析（PTX / SASS / 资源占用 / 反汇编 —— 每轮必做）
nvcc -ptx   -lineinfo -arch=<ARCH> <source.cu> -o $RD/candidate.ptx
nvcc -cubin -lineinfo -arch=<ARCH> <source.cu> -o $RD/candidate.cubin
cuobjdump -sass       $RD/candidate.cubin > $RD/candidate-sass.txt
cuobjdump -res-usage  $RD/candidate.cubin > $RD/candidate-res-usage.txt
nvdisasm  -gi -sf     $RD/candidate.cubin > $RD/candidate-nvdisasm.txt
```

### 7c. 分析 + 下一轮方向

1. **范围检查** — 是否有超范围改动

2. **理论 vs 实际** — 对比 direction.md 预测值 vs 实际
   - |gap| < 20% → 一致
   - 否则归因：实现差距（implementation gap）还是理论错误（theory error）

3. **NCU 实测对比**（当前 vs 上轮 vs baseline，引用具体数值）：
   - SM throughput（`sm__throughput.avg.pct_of_peak_sustained_elapsed`）
   - DRAM bandwidth（`dram__bytes.sum.per_second`）
   - L2/L1 hit rates
   - Warp stall 分布变化（哪类 stall 增加/减少）
   - Tensor Core utilization
   - Achieved occupancy
   - NCU rules engine（`--page details` 中的 Est. Speedup 建议）

4. **PTX/SASS 静态对比**（当前 vs 上轮）：
   - 寄存器数量 delta（res-usage）
   - Spill 检测：`STL`/`LDL` 指令数量变化
   - 总 SASS 指令数变化
   - 关键指令变化：HMMA/UMMA/WGMMA 数量、LDG/STG 向量化宽度、LDS/STS 访问模式
   - 循环展开程度变化（通过 branch 指令密度判断）
   - Dual-issue 质量：连续独立指令比例
   - 新增/消失的 `BAR.SYNC`、`MEMBAR` 等同步指令
   - nvdisasm 控制流图变化（分支、predicated execution）
   - **查 PTX ISA 文档**：解读指令变化、或本轮 direction 要换用某条 PTX 指令时，
     先到 `external/CudaSkill/.../ptx-docs/` 查其语义、操作数约束、fragment
     layout、Target ISA Notes（SM 支持），再下结论。`rounds/r<N>/analysis.md` 中
     涉及 PTX 指令选择的结论必须引用 ISA 文档章节号作为依据。
   - **调度层分析（强制，不能只看计数）**：每轮都必须从「统计层」（指令计数 /
     寄存器 / spill / bank-conflict）深入到「**调度 / 因果层**」——读热循环 SASS 的
     指令**发射顺序**：
     - **ptxas 有没有按你的意图排？** 还是把你的改动重排/抵消了？（手改和 inline
       PTX 都只是**提示**，ptxas 会二次调度——唯一确认方式是读 emit 出来的 SASS。
       这正是「能不能搏得动」的实测：`cuobjdump -sass` 看 ptxas 保留了多少 + NCU
       看目标 stall 有没有降。**绝不能只凭推断**。）
     - stall 到底卡在哪两条指令之间？`NOP` 填充 / scoreboard 等待 / `*DEPBAR`？
       math-pipe 指令（HMMA/QMMA…）是背靠背还是被 NOP 隔开？
     - **给每个 gap 归类**：**依赖气泡**（RAW 链——可用重排 / 多累加器 / 软件流水
       打破）vs **流水线吞吐 stall**（math pipe 本身吃不下——独立指令之间也夹 NOP；
       重排无用，只能加并发流或减总指令）。这个分类直接决定某优化「是否可能有效」。
     - **把每个 NCU stall 数对应到具体 SASS 模式**：别只写「wait=5.0」，要写
       「wait=5.0 ← Lxxxx 处 QMMA 突发之间夹 NOP 但累加器互相独立 ⇒ 吞吐 bound、
       非可重排依赖」。判「持平/退化」的轮次**必须**给出 SASS 证据说明**为什么**
       （ptxas 重排掉了？spill？RAW 链？），不能只凭 wall-clock 推断。

5. **策略轨迹** — 是否偏离 roadmap（>10% → 修正策略）

6. 写 `rounds/r<N>/analysis.md`（每条结论必须附 NCU metric 值或 SASS 指令证据）
7. **更新 `.rlcr/current/state.md`（每轮必做，抗压缩恢复依赖它）**：写明
   当前轮号 N、当前目标 module、本轮 verdict、最新 NCU duration、下一步 direction
   指向哪一轮。SessionStart hook 会把这份 state.md 作为「进度恢复卡」重注入
   context，所以它必须反映最新进度，否则压缩后会按过时状态续做。
8. 按 verdict 决定下一步：

| 判定（Verdict） | 动作 |
|---|---|
| **CONTINUE** | 写 `rounds/r<N+1>/direction.md`，继续下一轮 |
| **MODULE_COMPLETE** | 结束此模块，进入 Integration |
| **MODULE_STALLED** | 本模块持续无进展 → 转下一模块（不是停止；之后可回来再试） |
| **STRATEGY_REVISION_NEEDED** | 重新分析瓶颈，更新 `global-strategy.md` 和模块顺序 |

### 停止条件

**无轮次上限。** 只在以下情况结束/转向：

- **roofline efficiency ≥ 90%** → 全部结束
- 某模块持续无进展 → 转去优化下一个模块（不是停止；之后可回来再试）
- 所有模块都暂时无新方向 → **不要轻易判定"到顶"**：拓宽搜索空间（查
  KernelWiki、PTX ISA 文档、公开资料/论文/开源 kernel）找新方向再试
- 所有模块完成且 roofline 达标 → 进入 Finalize
- 否则只要还有可尝试的方向就继续；**仅在目标达成或用户明确叫停时停止**

> **一个方向做到极致仍未达标 → 换思路，别在同一方向上继续微调（强制）**：
> 当某个优化**方向/架构**已被推到极致（增量调参只剩持平或退化，且 SASS 调度层
> 分析表明剩余瓶颈在该方向内**不可约**——如「NOP 是吞吐 stall 非依赖气泡」、
> 「再加 warp 必然 throttle 爆」、「更多累加器必然 spill」），**不要再在这个方向
> 上磨**。必须**主动换一条根本不同的思路**——通常是**更激进的并发结构 / 整体
> re-architecture**（例：从「per-step 全块 barrier」换到 warp specialization；从
> cp.async 换 TMA；从单累加器换 cooperative ping-pong 把更多 math 流喂进流水线
> 又不过载；persistent / cluster / split-K 等）。判断「方向已到极致」必须有**调度层
> SASS 证据 + 至少一次反向尝试的实测**（如 r14 加 warp、r15 加累加器都退化）撑腰，
> 不能只凭直觉。换思路属于已授权的自主决策（见下），不要因为「是个大改」就停下来
> 问或就此收尾——**目标没达成且还有根本不同的思路没试，就不算到顶**。

> **自主决策（不要为已授权的决定征求许可）**：本流程内的决定——继续下一轮、
> 转模块、拓宽搜索、**乃至 STRATEGY_REVISION→re-architecture**——都已被本 skill
> 授权，必须**自主执行，不要用 AskUserQuestion 去问用户"要不要做"**。即使某步
> token/工作量很大（如换架构重写），也不是征求许可的理由——授权已经给了。只有
> 遇到**真正越出任务范围的岔路**，或用户主动要求介入时，才停下来问。

---

## Step 8: 集成（Integration，每个模块完成后）

1. `git diff <module-start-commit>..HEAD -- solution/` — 汇总本模块所有改动范围
2. NCU full kernel profile + SASS 静态分析
3. 对比 baseline 整体性能（**以 NCU duration 为准**）：
   - 整体 speedup vs baseline = `baseline NCU duration / candidate NCU duration`
     （`gpu__time_duration`），**不用 wall-clock 算 speedup**（铁律 -0.5）
   - **逐模块退化检查（per-module regression check）**：检查每个已完成模块的 source-level NCU metrics，确认之前优化的模块没有退化
4. 如有 regression（以 NCU 判定）：
   - NCU + SASS 诊断根因
   - 如果是模块间干扰（如 shared memory 布局冲突、寄存器压力传导），写 `regression-analysis.md` 并修复
   - 修复后重新跑 NCU + correctness 确认
5. 更新 `module-tracker.json`、`goal-tracker.md`
6. 写下一轮方向 `rounds/r<N+1>/direction.md`（标明下一个目标模块）
7. git commit

---

## Step 9: Finalize

0. **选出最优版本（因为不回退，最优不一定是 HEAD）—— 排名只用 NCU**：扫所有已
   提交轮次，挑出**正确且 NCU kernel duration（`gpu__time_duration`）最短**的那一轮
   commit。**"最快"一律以 NCU 实测排序，不用 `bench/benchmark.py` 的 wall-clock**
   （铁律 -0.5）。若各轮的 NCU 记录已随 `.rlcr/` 清理而不全，则对候选的几轮 commit
   逐一 `git checkout` 后用**同一套 NCU 命令、同一块空闲 GPU** 重测 duration 再排名，
   不要凭 wall-clock 推断。选定后若它不是 HEAD，`git checkout <最优 commit> --
   solution/`（或在其上 cherry-pick 后续仍有效的修改）定为交付物，并重跑一次 NCU +
   correctness 确认。记录"哪一轮胜出 + NCU duration + 为什么"。
   （配合铁律 #3 退化不回退：过程允许走低谷，最优在此按 NCU 一次性选出。）
1. 写 `docs/results.md`：
   - 最优版本是哪一轮、对比各轮 **NCU duration**（含走过的低谷，体现 no-revert 探索）；
     wall-clock 如要列只作辅助参考列，不作排名/结论依据
   - 逐模块贡献拆解
   - 理论准确度总结
   - 最终每个 shape 的性能与 geomean speedup —— **均按 NCU duration 计算**
     （`baseline / candidate` 的 `gpu__time_duration`）
   - GPU 信息、roofline 总结（compute/memory bound 判定 + 达到 roofline 上限的百分比）
2. 写 `.rlcr/current/complete-summary.md`
3. 更新 `.rlcr/current/state.md`
4. git commit

---

## Step 10: 报告结果

向用户报告 `docs/results.md` 中的最终性能数据。
