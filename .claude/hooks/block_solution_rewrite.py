#!/usr/bin/env python3
"""IterKernel solution/ guards (PreToolUse + PostToolUse hook).

Two compaction-proof guarantees, keyed off on-disk state so they survive any
context summarization (the model's memory is irrelevant — these run as a
separate process on every tool call):

1. ANTI-REWRITE (means layer).  Once a campaign's
   ``<root>/.rlcr/current/.initial-impl-done`` marker exists, ``solution/``
   files may only change via incremental ``Edit``. Wholesale ``Write`` and
   shell overwrites (``>`` ``>>`` ``tee`` ``sed -i`` ``truncate`` ``dd`` into
   solution/) are denied.

2. DIRECTION-READ GATE (process layer).  In the iteration phase (marker
   present), an ``Edit`` to a locked ``solution/`` is denied unless the *current*
   round direction file (newest ``round-*-direction.md`` under
   ``.rlcr/current/modules/``, else ``.rlcr/current/direction.md``) has been
   read since it was last written. Reading it (PostToolUse) refreshes
   ``.rlcr/current/.direction-read-marker``.

3. SASS GATE (process layer).  An ``Edit`` to a locked ``solution/`` in round N
   (N>=2) is denied until the PREVIOUS round produced its SASS/static analysis
   (``.rlcr/current/rounds/r<N-1>/candidate-sass.txt`` on disk). Enforces "每轮
   都要做 SASS, 否则不进入下一轮代码修改". Round 1 is never gated.

Invocation:
  pre  → handle_pre_tool_use   (matcher: Write|Edit|MultiEdit|Bash)
  post → handle_post_tool_use  (matcher: Read)

Decision protocol: deny == print PreToolUse ``permissionDecision: deny`` JSON
on stdout, exit 0. Allow == exit 0, no output. Fail-open on any error so the
guard never breaks a session.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MARKER_REL = (".rlcr", "current", ".initial-impl-done")
DIRECTION_READ_MARKER_REL = (".rlcr", "current", ".direction-read-marker")

REWRITE_REASON = (
    "🔒 渐进式修改锁已激活（{root}/.rlcr/current/.initial-impl-done 存在）。"
    "禁止用 Write 覆盖**已存在**的 solution/ 文件（既定架构内只能用 Edit 增量修改）。"
    "若要回退请用 `git checkout HEAD -- solution/`，再用小步 Edit 重试。"
    "（注：re-architecture 写**新文件**是放行的——直接 Write 新的轮次编号源文件即可，"
    "无需也不要 `rm` 这个锁。）"
)

DIRECTION_REASON = (
    "📖 本轮优化方向尚未阅读。编辑 solution/ 前必须先 Read 当前轮的方向文件："
    "{path}。读完它再做增量 Edit（即使 context 被压缩也不能跳过这一步）。"
)

# Representative of the 5 mandatory per-round static products. Its presence in a
# round dir means that round's SASS/static analysis was generated.
SASS_ARTIFACT = "candidate-sass.txt"
SASS_REASON = (
    "🔬 上一轮（{path}）的 SASS/静态分析尚未生成。硬门槛：每一轮都必须先完成 5 类"
    "静态产物（candidate.ptx/.cubin/candidate-sass.txt/candidate-res-usage.txt/"
    "candidate-nvdisasm.txt）并写进 analysis.md，才能开始下一轮的 solution/ 代码修改。"
    "请先为上一轮生成静态分析（即使 context 被压缩也不能跳过）。"
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
    """Non-blocking: inject a warning into the model context (allows the tool)."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": msg,
        }
    }, ensure_ascii=False))


UNLOCKED_WARN = (
    "⚠️ 渐进式修改锁缺失：{root}/.rlcr/current/.initial-impl-done 不存在,但该 "
    "campaign 已进入迭代(存在 rounds/r<N>/)。说明锁在某次 re-arch 摘掉后没补回来,"
    "防重写 / 先读 direction / SASS 硬门槛**当前全部失效**。请立刻 "
    "`touch {root}/.rlcr/current/.initial-impl-done` 恢复纪律,再继续改 solution/。"
)


def resolve_path(raw: str, cwd: str | None) -> Path:
    p = Path(raw.strip().strip('"').strip("'")).expanduser()
    if not p.is_absolute() and cwd:
        p = Path(cwd).expanduser() / p
    return p


def locked_campaign_root(raw: str, cwd: str | None) -> Path | None:
    """Return the campaign root iff ``raw`` is under a *locked* solution/ dir.

    Locked == path has a segment exactly ``solution`` and
    ``<parent>/.rlcr/current/.initial-impl-done`` exists. Exact-segment match
    avoids false positives like ``profiles/solution-sass.txt``.
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
    """Return the campaign root for any path under a ``solution`` segment,
    regardless of lock state (used for the unlocked-state warning)."""
    if not raw or "solution" not in raw:
        return None
    parts = resolve_path(raw, cwd).parts
    for i, seg in enumerate(parts):
        if seg == "solution" and i > 0:
            return Path(*parts[:i])
    return None


def unlocked_but_iterating(root: Path) -> bool:
    """True if the lock marker is MISSING yet the campaign is already past the
    initial impl (>=1 rounds/r<N>/ dir exists). That is the 'lock got dropped and
    never restored' state -> enforcement is silently off. Not triggered during the
    pre-initial-impl phase (no rounds dirs yet) to avoid false positives."""
    if root.joinpath(*MARKER_REL).exists():
        return False
    rounds = root / ".rlcr" / "current" / "rounds"
    try:
        return any(re.fullmatch(r"r\d+", d.name) and d.is_dir() for d in rounds.iterdir())
    except OSError:
        return False


def current_direction_file(root: Path) -> Path | None:
    """Newest round direction file.

    New layout: .rlcr/current/rounds/r<N>/direction.md  (one dir per round)
    Legacy:     .rlcr/current/modules/<id>/round-*-direction.md
    Fallback:   .rlcr/current/direction.md  (initial)
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


# --- shell overwrite detection -------------------------------------------------
_REDIRECT_RE = re.compile(r"""\d*>>?\s*("[^"]*"|'[^']*'|[^\s;|&)]+)""")
_SED_INPLACE_RE = re.compile(r"\bsed\b[^|;&]*-i")
_TEE_RE = re.compile(r"\btee\b")
_TRUNC_RE = re.compile(r"\b(?:truncate|dd)\b")


def _tokens(command: str) -> list[str]:
    return re.findall(r"""(?:"[^"]*"|'[^']*'|[^\s;|&]+)""", command)


def bash_overwrite_root(command: str, cwd: str | None) -> Path | None:
    if not command or "solution" not in command:
        return None
    candidates: list[str] = [m.group(1) for m in _REDIRECT_RE.finditer(command)]
    if _SED_INPLACE_RE.search(command) or _TEE_RE.search(command) or _TRUNC_RE.search(command):
        candidates.extend(_tokens(command))
    for cand in candidates:
        root = locked_campaign_root(cand, cwd)
        if root is not None:
            return root
    return None


# --- direction-read gate -------------------------------------------------------
def direction_unread(root: Path) -> Path | None:
    """Return the current direction file if it exists but has not been read."""
    cur = current_direction_file(root)
    if cur is None:
        return None  # nothing to read -> never deadlock
    read_marker = root.joinpath(*DIRECTION_READ_MARKER_REL)
    try:
        if not read_marker.exists() or cur.stat().st_mtime > read_marker.stat().st_mtime:
            return cur
    except OSError:
        return cur
    return None


def prev_round_sass_missing(root: Path) -> Path | None:
    """SASS GATE: editing solution/ in the current round is blocked until the
    PREVIOUS round produced its SASS/static analysis. Enforces "每轮都要做 SASS,
    否则不进入下一轮代码修改". Keyed on rounds/r<N>/candidate-sass.txt on disk
    (compaction-proof). Round 1 (no previous) is never gated."""
    rounds = root / ".rlcr" / "current" / "rounds"
    dirs: list[tuple[int, Path]] = []
    try:
        for d in rounds.iterdir():
            m = re.fullmatch(r"r(\d+)", d.name)
            if m and d.is_dir():
                dirs.append((int(m.group(1)), d))
    except OSError:
        return None
    if len(dirs) < 2:
        return None  # round 1 or none -> nothing to gate on
    dirs.sort()
    prev_dir = dirs[-2][1]  # round before the newest
    artifact = prev_dir / SASS_ARTIFACT
    return None if artifact.exists() else artifact


def handle_pre_tool_use(payload: dict) -> int:
    tool = (payload.get("tool_name") or "").lower()
    tool_input = payload.get("tool_input") or {}
    cwd = payload.get("cwd")

    if tool == "write":
        fp = tool_input.get("file_path", "")
        root = locked_campaign_root(fp, cwd)
        if root is not None:
            # Allow creating a NEW file (re-architecture writes a new round-numbered
            # source) — there is nothing to "rewrite". Only block OVERWRITING an
            # existing solution file (the lazy full-file-rewrite the lock guards
            # against). This removes the need to ever `rm` the lock for a re-arch.
            if resolve_path(fp, cwd).exists():
                return emit_deny(REWRITE_REASON.format(root=root))
            return 0
        # unlocked-state safety net
        wroot = campaign_root_for_solution(fp, cwd)
        if wroot is not None and unlocked_but_iterating(wroot):
            emit_warn(UNLOCKED_WARN.format(root=wroot))
        return 0

    if "bash" in tool:
        root = bash_overwrite_root(tool_input.get("command", ""), cwd)
        return emit_deny(REWRITE_REASON.format(root=root)) if root else 0

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
        sass = prev_round_sass_missing(root)
        if sass:
            return emit_deny(SASS_REASON.format(path=sass.parent))
        return 0

    return 0


def handle_post_tool_use(payload: dict) -> int:
    """Refresh the direction-read marker when the CURRENT direction is read."""
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
        return 0  # fail-open
    if mode == "post":
        return handle_post_tool_use(payload)
    return handle_pre_tool_use(payload)


if __name__ == "__main__":
    raise SystemExit(main())
