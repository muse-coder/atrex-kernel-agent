#!/usr/bin/env python3
"""IterKernel solution/ 守卫（PreToolUse + PostToolUse hook）。

两类抗压缩保障，以磁盘上的状态为依据，因此能在任何 context 摘要后存活
（模型的记忆无关紧要——它们在每次工具调用时作为独立进程运行）：

注：**防重写不再由本 hook 机械强制**（原先拦 ``Write`` 覆盖 + shell 重写的那一段
已移除）。改为靠 code2 无 ``Write`` 工具 + ``cp v<N-1>→v<N>`` 版本文件模型 +
analysis 的「一 lever diff」人工核对来自律。本 hook 只保留下面两道依赖
``.initial-impl-done`` 锁的 process 门槛。

1. 方向已读门槛（process 层）。在迭代阶段（``.initial-impl-done`` 标记存在时），
   对已锁定的 ``solution/`` 做 ``Edit`` 会被拒绝，除非*当前*轮的方向文件
   （``.rlcr/current/rounds/r<N>/direction.md`` 最新者，否则
   ``.rlcr/current/direction.md``）自其上次写入后已被读过。读它（PostToolUse）
   会刷新 ``.rlcr/current/.direction-read-marker``。

   ⚠️ 已知局限（#6）：``.direction-read-marker`` 是**全局单文件**，PostToolUse 拿不到
   「是哪个 subagent 读的」，所以**任何** agent（master/analysis）Read 当前 direction
   都会刷新它，从而让 code2「漏读也能 Edit」。hook 层无法归因 reader 身份；因此靠**契约
   不变式**补强：**活跃轮的 direction.md 只由 code2 读，master/analysis 只写不回读**
   （见 optimize-kernel.md Step 7c、analysis.md、code-iter.md）。code2 仍被要求每轮亲自读。

2. 完整产物门槛（process 层）。在第 N 轮（N>=2）对已锁定的 ``solution/`` 做
   ``Edit`` 会被拒绝，直到上一轮生成了完整 NCU、5 类静态产物和 ``analysis.md``。
   强制执行「每轮都要做 profile/静态分析/诊断，否则不进入下一轮代码修改」。
   第 1 轮永不设门槛。

调用方式：
  pre  → handle_pre_tool_use   (matcher: Write|Edit|MultiEdit)
  post → handle_post_tool_use  (matcher: Read)

决策协议：deny == 在 stdout 打印 PreToolUse ``permissionDecision: deny`` JSON，
exit 0。Allow == exit 0，无输出。任何错误都 fail-open，使守卫永远不会
打断一次 session。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MARKER_REL = (".rlcr", "current", ".initial-impl-done")
DIRECTION_READ_MARKER_REL = (".rlcr", "current", ".direction-read-marker")

DIRECTION_REASON = (
    "📖 本轮优化方向尚未阅读。编辑 solution/ 前必须先 Read 当前轮的方向文件："
    "{path}。读完它再做增量 Edit（即使 context 被压缩也不能跳过这一步）。"
)

# 上一轮必须完整落盘的 profile / 静态分析 / 诊断产物。缺任一项都不能进入下一轮
# solution/ 编辑；这把文档里的“5 类静态产物 + NCU + analysis.md”变成机械门槛。
REQUIRED_PREV_ROUND_ARTIFACTS = [
    "candidate.ptx",
    "candidate.cubin",
    "candidate-sass.txt",
    "candidate-res-usage.txt",
    "candidate-nvdisasm.txt",
    "candidate.ncu-rep",
    "candidate-details.txt",
    "candidate-metrics.csv",
    "correctness-pass.txt",  # benchmark.py 仅在全部 workload 正确时落；缺=上一轮 kernel 未证明正确
    "analysis.md",
]
ROUND_ARTIFACTS_REASON = (
    "🔬 上一轮（{path}）的 profile/静态分析/诊断产物尚未完整生成，缺失：{missing}。"
    "硬门槛：每一轮都必须先完成 NCU（candidate.ncu-rep/details/metrics）、5 类静态产物"
    "（candidate.ptx/.cubin/candidate-sass.txt/candidate-res-usage.txt/"
    "candidate-nvdisasm.txt）并写进 analysis.md，才能开始下一轮的 solution/ 代码修改。"
    "请先补齐上一轮产物（即使 context 被压缩也不能跳过）。"
)


def emit_deny(reason: str) -> int:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }, ensure_ascii=False))
    return 0


def emit_warn(msg: str) -> None:
    """非阻塞：把一条警告注入到模型 context（放行该工具）。"""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": msg,
        }
    }, ensure_ascii=False))


UNLOCKED_WARN = (
    "⚠️ 渐进式修改锁缺失：{root}/.rlcr/current/.initial-impl-done 不存在,但该 "
    "campaign 已进入迭代(存在 rounds/r<N>/)。说明锁在某次 re-arch 摘掉后没补回来,"
    "先读 direction / 完整产物硬门槛**当前全部失效**(防重写本就靠 code2 无 Write + "
    "cp 模型自律,不依赖锁)。请立刻 "
    "`touch {root}/.rlcr/current/.initial-impl-done` 恢复纪律,再继续改 solution/。"
)


def resolve_path(raw: str, cwd: str | None) -> Path:
    p = Path(raw.strip().strip('"').strip("'")).expanduser()
    if not p.is_absolute() and cwd:
        p = Path(cwd).expanduser() / p
    return p


def locked_campaign_root(raw: str, cwd: str | None) -> Path | None:
    """仅当 ``raw`` 位于一个*已锁定*的 solution/ 目录下时，返回该 campaign 根目录。

    已锁定 == 路径中有一个恰好为 ``solution`` 的段，且
    ``<parent>/.rlcr/current/.initial-impl-done`` 存在。精确段匹配可避免
    像 ``profiles/solution-sass.txt`` 这样的误判。
    """
    if not raw or "solution" not in raw:
        return None
    parts = resolve_path(raw, cwd).parts
    for i, seg in enumerate(parts):
        if seg == "solution" and i > 0:
            root = Path(*parts[:i])
            if root.joinpath(*MARKER_REL).exists():
                return root
    return None


def campaign_root_for_solution(raw: str, cwd: str | None) -> Path | None:
    """对任何位于 ``solution`` 段下的路径返回其 campaign 根目录，
    无论锁状态如何（用于未锁定状态的警告）。"""
    if not raw or "solution" not in raw:
        return None
    parts = resolve_path(raw, cwd).parts
    for i, seg in enumerate(parts):
        if seg == "solution" and i > 0:
            return Path(*parts[:i])
    return None


def unlocked_but_iterating(root: Path) -> bool:
    """当锁标记缺失、但 campaign 已越过初始实现阶段（存在 >=1 个 rounds/r<N>/
    目录）时返回 True。这就是「锁被摘掉后从未恢复」的状态 -> 强制约束被悄悄
    关闭。在初始实现之前的阶段（还没有 rounds 目录）不会触发，以避免误判。"""
    if root.joinpath(*MARKER_REL).exists():
        return False
    rounds = root / ".rlcr" / "current" / "rounds"
    try:
        return any(re.fullmatch(r"r\d+", d.name) and d.is_dir() for d in rounds.iterdir())
    except OSError:
        return False


def current_direction_file(root: Path) -> Path | None:
    """最新一轮的方向文件。

    新布局：.rlcr/current/rounds/r<N>/direction.md  （每轮一个目录）
    旧布局：.rlcr/current/modules/<id>/round-*-direction.md
    兜底：  .rlcr/current/direction.md  （初始）
    """
    cur = root / ".rlcr" / "current"
    candidates: list[Path] = []
    for base, pattern in ((cur / "rounds", "r*/direction.md"),
                          (cur / "modules", "*/round-*-direction.md")):
        try:
            candidates += list(base.glob(pattern))
        except OSError:
            pass
    if candidates:
        try:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except OSError:
            return candidates[0]
    fallback = cur / "direction.md"
    return fallback if fallback.exists() else None


# --- 方向已读门槛 --------------------------------------------------------------
def direction_unread(root: Path) -> Path | None:
    """如果当前方向文件存在但尚未被读过，则返回它。"""
    cur = current_direction_file(root)
    if cur is None:
        return None  # 没有可读的 -> 永不死锁
    read_marker = root.joinpath(*DIRECTION_READ_MARKER_REL)
    try:
        if not read_marker.exists() or cur.stat().st_mtime > read_marker.stat().st_mtime:
            return cur
    except OSError:
        return cur
    return None


# state.md 里声明「当前轮」的行，作为轮号的**权威来源**（由 Step 7a 每轮设定）。
# 形如：`当前轮: r3` / `current_round = 3` / `当前轮号：r3`（大小写、r 前缀、
# 中英冒号、=/: 都容忍）。
_ROUND_DECL_RE = re.compile(
    r"(?:current[_ ]round|当前轮号?)\s*[:：=]\s*r?(\d+)", re.IGNORECASE
)


def declared_current_round(cur: Path) -> int | None:
    """从 .rlcr/current/state.md 读出 agent 声明的当前轮号 N（权威来源）。

    取第一处匹配（state.md 顶部那条声明），读不到/没声明返回 None。"""
    try:
        text = (cur / "state.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _ROUND_DECL_RE.search(text)
    return int(m.group(1)) if m else None


def prev_round_artifacts_missing(root: Path) -> tuple[Path, list[str]] | None:
    """上一轮产物门槛：在当前轮编辑 solution/ 会被阻止，直到**上一轮**生成完整
    NCU、静态分析和 analysis.md。第 1 轮（没有上一轮）永不设门槛。

    「上一轮是谁」的判定（修掉旧的 `dirs[-2]` 错位 bug）：
    旧实现把「目录号第二大的」当上一轮，依赖 agent 建 round 目录的时机——簿记
    一错位就检查错轮次（甚至放过该拦的 SASS 跳过）。现在改为：
      1) 权威路径：读 state.md 声明的当前轮号 N（Step 7a 每轮设定），上一轮 =
         **磁盘上存在的、轮号 < N 的最大轮**。不受目录创建早晚影响。
      2) 兜底：state.md 没有可解析的轮号时，回退到旧的「第二大目录」启发式
         （保证不比以前差，且永不死锁）。"""
    cur = root / ".rlcr" / "current"
    rounds = cur / "rounds"
    dirs: dict[int, Path] = {}
    try:
        for d in rounds.iterdir():
            m = re.fullmatch(r"r(\d+)", d.name)
            if m and d.is_dir():
                dirs[int(m.group(1))] = d
    except OSError:
        return None
    if not dirs:
        return None  # 还没有任何轮 -> 无可设门槛的对象

    n = declared_current_round(cur)
    if n is not None:
        prev_nums = [k for k in dirs if k < n]
        if not prev_nums:
            return None  # 第 1 轮 / 没有更早的轮 -> 不设门槛
        prev_dir = dirs[max(prev_nums)]
    else:
        # 兜底：旧启发式（第二大目录）。
        nums = sorted(dirs)
        if len(nums) < 2:
            return None
        prev_dir = dirs[nums[-2]]

    missing = [
        name for name in REQUIRED_PREV_ROUND_ARTIFACTS
        if not (prev_dir / name).exists()
    ]
    return (prev_dir, missing) if missing else None


def handle_pre_tool_use(payload: dict) -> int:
    tool = (payload.get("tool_name") or "").lower()
    tool_input = payload.get("tool_input") or {}
    cwd = payload.get("cwd")

    if tool == "write":
        # 防重写已不再由本 hook 强制：Write 一律放行（靠 code2 无 Write 工具 +
        # cp v<N-1>→v<N> 版本文件模型 + analysis 的一-lever diff 人工核对自律）。
        # 仅在锁缺失但已进入迭代时给一条非阻塞提醒——先读方向 / 完整产物两道门槛
        # 依赖该锁。
        fp = tool_input.get("file_path", "")
        wroot = campaign_root_for_solution(fp, cwd)
        if wroot is not None and unlocked_but_iterating(wroot):
            emit_warn(UNLOCKED_WARN.format(root=wroot))
        return 0

    if "edit" in tool:  # Edit / MultiEdit
        fp = tool_input.get("file_path", "")
        root = locked_campaign_root(fp, cwd)
        if root is None:
            wroot = campaign_root_for_solution(fp, cwd)
            if wroot is not None and unlocked_but_iterating(wroot):
                emit_warn(UNLOCKED_WARN.format(root=wroot))
            return 0
        cur = direction_unread(root)
        if cur:
            return emit_deny(DIRECTION_REASON.format(path=cur))
        missing = prev_round_artifacts_missing(root)
        if missing:
            prev_dir, missing_names = missing
            return emit_deny(
                ROUND_ARTIFACTS_REASON.format(
                    path=prev_dir,
                    missing=", ".join(missing_names),
                )
            )
        return 0

    return 0


def handle_post_tool_use(payload: dict) -> int:
    """当当前方向文件被读取时，刷新方向已读标记。"""
    tool = (payload.get("tool_name") or "").lower()
    if "read" not in tool:
        return 0
    tool_input = payload.get("tool_input") or {}
    raw = tool_input.get("file_path", "")
    if not raw:
        return 0
    path = resolve_path(raw, payload.get("cwd"))
    name = path.name
    if not (name.endswith("-direction.md") or name == "direction.md"):
        return 0
    parts = path.parts
    if ".rlcr" not in parts:
        return 0
    root = Path(*parts[: parts.index(".rlcr")])
    cur = current_direction_file(root)
    try:
        if cur is not None and path.resolve() == cur.resolve():
            root.joinpath(*DIRECTION_READ_MARKER_REL).touch()
    except OSError:
        pass
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre"
    raw = sys.stdin.read().strip()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return 0  # fail-open（出错时放行）
    if mode == "post":
        return handle_post_tool_use(payload)
    return handle_pre_tool_use(payload)


if __name__ == "__main__":
    raise SystemExit(main())
