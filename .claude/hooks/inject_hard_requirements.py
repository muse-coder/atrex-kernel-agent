#!/usr/bin/env python3
"""SessionStart hook：把 IterKernel 硬性要求 + 当前进度重新注入到 context。

两类内容即使在对话被压缩后也必须继续生效，所以都走 SessionStart 的 stdout
（SessionStart 会在 启动 / 恢复 / 清空 / **压缩** 时触发，stdout 会被加入到全新
的 context 窗口——这就是「抗 context 压缩」的机制）：

1. 硬性要求（REQUIREMENTS）：静态规则，恒定不变。
2. 进度恢复卡（build_progress_block）：动态状态，从磁盘上活跃 campaign 的
   ``.rlcr/current/`` 读出——当前阶段、目标、最新一轮做到哪了。压缩往往发生在
   某一轮中途，只重注入规则不够：agent 还需要知道「我现在第几轮、在优化哪个
   module、上一轮结论、下一步 direction、完整产物门槛过没过」，否则容易丢线头、
   重复上一轮或跳过没做完的 profile。

输出走 stdout；对于 SessionStart hook，Claude Code 会把 stdout 加入 context。
任何错误都 fail-open（只打印 REQUIREMENTS，绝不让 hook 打断 session）。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

# ⚠️ 本文本是 `.claude/commands/optimize-kernel.md`「全局铁律」+ CLAUDE.md
#    「硬性要求」的**运行时镜像**（权威源是 optimize-kernel.md）。因为它要在运行时
#    注入 context、无法"指向另一个文件让模型去读"，所以必须保留这份副本。
#    改铁律时三处同步：optimize-kernel.md 全局铁律 / CLAUDE.md 硬性要求 / 这里。
REQUIREMENTS = """\
[IterKernel 硬性要求 —— 始终生效，每次 context 压缩后都会重新注入]

1. 从头实现：candidate kernel 必须从头设计并实现。严禁把任何已有实现
   （上一个 campaign 的 kernel、库 kernel、抄来的代码）当作代码起点去
   「继续/修补」。即使存在一个 shape+GPU 完全相同的旧 campaign，也只能作为
   参考（看它的 NCU/SASS、借鉴思路）——新开空文件，用 Step 4d-0 选定的原语
   （纯 CUDA+PTX / CUTLASS / CuTe DSL，按算子复杂度评估而定，不固定 CUDA）自己
   从零写出 kernel（CUTLASS 路径=自己用其构件组装，不复制现成 kernel）。

2. 最强 baseline：baseline = 现成库中最快的实现。
   PyTorch（cuBLAS / torch._scaled_mm）和 FlashInfer 两者都要测，取更快的，
   并在 docs/baseline_source.md 记录对比。绝不打赢一个弱 baseline。

3. 性能以 NCU 为准：baseline 与 candidate 的快慢、每一轮的进展、以及最终
   最优版本，一律以 NCU 的 kernel duration（gpu__time_duration）判定。
   bench wall-clock 仅作辅助。

4. 完成判据 = 达到该 shape 的 roofline 上限的 ≥90%（compute-bound 取 compute
   roofline、memory-bound 取 memory roofline 的 90%；后者可能低于 spec 峰值，
   拿 spec 峰值当目标会物理不可达），不是"打平 baseline"。
   baseline 只是必须超过的下限参照；90% roofline 上限常常 > baseline 实测效率，
   所以「打平 baseline」≠ 完成。目标与参照都要写进 goal-tracker。

另外：不要把 benchmark harness / 开销修补（.py 包装层、.item()、
copy_、计时代码）误当成「优化」。优化 = kernel 本身的架构与指令级工作。
如果你在改这些而不是 .cu，就跑偏了——回到 kernel 上。

完整文本见：CLAUDE.md「硬性要求」+ .claude/commands/optimize-kernel.md 全局铁律 -2/-1/-0.5。"""


# --- 进度恢复卡 -----------------------------------------------------------------

# 每轮必生成的产物（用于体检最新一轮做到哪了；顺序即展示顺序）。
ROUND_ARTIFACTS = [
    "direction.md", "summary.md", "analysis.md",
    "candidate.ptx", "candidate.cubin", "candidate-sass.txt",
    "candidate-res-usage.txt", "candidate-nvdisasm.txt",
    "candidate.ncu-rep", "candidate-details.txt", "candidate-metrics.csv",
    "correctness-pass.txt",
]
GATE_ARTIFACTS = [
    "analysis.md", "correctness-pass.txt",
    "candidate.ptx", "candidate.cubin", "candidate-sass.txt",
    "candidate-res-usage.txt", "candidate-nvdisasm.txt",
    "candidate.ncu-rep", "candidate-details.txt", "candidate-metrics.csv",
]
# 单个文件最多注入多少字符 / 行，避免把 context 撑爆。
MAX_CHARS = 2500
MAX_LINES = 45
# 扫描活跃 campaign 的根目录（B/C 未改前，campaign 都在 /tmp/<slug>/）。
SCAN_GLOBS = ["/tmp/*"]


def _find_campaign_roots(cwd: str | None) -> list[Path]:
    """返回所有含 .rlcr/current/ 的目录。优先 cwd 向上找，再扫 SCAN_GLOBS。"""
    roots: list[Path] = []
    # 1) cwd 向上逐级找（agent 真在 campaign 里跑时命中）。
    if cwd:
        p = Path(cwd).expanduser()
        for cand in [p, *p.parents]:
            if (cand / ".rlcr" / "current").is_dir():
                roots.append(cand)
                break
    # 2) 扫已知根目录。
    for pattern in SCAN_GLOBS:
        base = Path(pattern).parent
        glob = Path(pattern).name
        try:
            for d in base.glob(glob):
                if (d / ".rlcr" / "current").is_dir() and d not in roots:
                    roots.append(d)
        except OSError:
            pass
    return roots


def _active_campaign(cwd: str | None) -> Path | None:
    """活跃 campaign = .rlcr/current 最近被改过的那个（最可能是当前在做的）。"""
    roots = _find_campaign_roots(cwd)
    if not roots:
        return None

    def mtime(root: Path) -> float:
        try:
            return (root / ".rlcr" / "current").stat().st_mtime
        except OSError:
            return 0.0

    return max(roots, key=mtime)


def _excerpt(path: Path) -> str | None:
    """读文件前若干行，截断到上限。读不到返回 None。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    clipped = lines[:MAX_LINES]
    out = "\n".join(clipped)
    if len(out) > MAX_CHARS:
        out = out[:MAX_CHARS] + " …(截断)"
    if len(lines) > MAX_LINES:
        out += f"\n…(还有 {len(lines) - MAX_LINES} 行，见原文件)"
    return out.strip() or None


def _latest_round(cur: Path) -> tuple[int, Path] | None:
    rounds = cur / "rounds"
    best: tuple[int, Path] | None = None
    try:
        for d in rounds.iterdir():
            name = d.name
            if d.is_dir() and name.startswith("r") and name[1:].isdigit():
                n = int(name[1:])
                if best is None or n > best[0]:
                    best = (n, d)
    except OSError:
        return None
    return best


def build_progress_block(cwd: str | None) -> str | None:
    root = _active_campaign(cwd)
    if root is None:
        return None
    cur = root / ".rlcr" / "current"
    try:
        ts = datetime.fromtimestamp(cur.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        ts = "?"

    parts: list[str] = [
        "[IterKernel 进度恢复 —— 压缩后请先核对「做到哪了」再继续]",
        f"活跃 campaign: {root}   (.rlcr/current 最后修改: {ts})",
        "（若这不是你正在做的 campaign，以 cwd 下的为准；本卡是磁盘快照，best-effort。）",
    ]

    for label, rel in (("state.md", "state.md"),
                       ("goal-tracker.md", "goal-tracker.md")):
        ex = _excerpt(cur / rel)
        if ex:
            parts.append(f"\n--- {label} ---\n{ex}")

    lr = _latest_round(cur)
    if lr is not None:
        n, rdir = lr
        present = []
        for art in ROUND_ARTIFACTS:
            mark = "✓" if (rdir / art).exists() else "✗"
            present.append(f"{art}{mark}")
        missing_gate = [name for name in GATE_ARTIFACTS if not (rdir / name).exists()]
        parts.append(
            f"\n--- 最新轮 r{n}（{rdir.relative_to(root)}）---\n"
            + "  ".join(present)
            + "\n本轮完整产物门槛："
            + ("已满足" if not missing_gate else "未满足——下一轮改 solution/ 会被 hook 拦；缺失: " + ", ".join(missing_gate))
        )
        dir_ex = _excerpt(rdir / "direction.md")
        if dir_ex:
            parts.append(f"\n--- r{n}/direction.md ---\n{dir_ex}")

    parts.append(
        "\n恢复后应做：核对当前轮号/目标 module/上一轮 verdict，"
        "按 optimize-kernel.md Step 7 的 a/b/c 顺序续做未完成的步骤；"
        "不要重头开新轮，也不要跳过未生成的产物。"
    )
    return "\n".join(parts)


def main() -> int:
    # 读取 stdin 上的 hook JSON payload（SessionStart 会带 cwd 等字段）。
    cwd = None
    try:
        raw = sys.stdin.read()
        if raw.strip():
            payload = json.loads(raw)
            cwd = payload.get("cwd")
    except Exception:
        cwd = None

    # 1) 静态规则（恒定）。
    print(REQUIREMENTS)

    # 2) 动态进度卡（best-effort，fail-open）。
    try:
        block = build_progress_block(cwd)
    except Exception:
        block = None
    if block:
        print("\n" + block)

    return 0


if __name__ == "__main__":
    sys.exit(main())
