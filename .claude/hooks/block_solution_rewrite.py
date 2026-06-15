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
    "solution/ 首次实现后只能用 Edit 增量修改，禁止 Write 覆盖或 shell 重定向重写。"
    "若要回退请用 `git checkout HEAD -- solution/`，再用小步 Edit 重试。"
)

DIRECTION_REASON = (
    "📖 本轮优化方向尚未阅读。编辑 solution/ 前必须先 Read 当前轮的方向文件："
    "{path}。读完它再做增量 Edit（即使 context 被压缩也不能跳过这一步）。"
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


def current_direction_file(root: Path) -> Path | None:
    """Newest round-*-direction.md under modules/, else initial direction.md."""
    try:
        candidates = list((root / ".rlcr" / "current" / "modules").glob("*/round-*-direction.md"))
    except OSError:
        candidates = []
    if candidates:
        try:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except OSError:
            return candidates[0]
    fallback = root / ".rlcr" / "current" / "direction.md"
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


def handle_pre_tool_use(payload: dict) -> int:
    tool = (payload.get("tool_name") or "").lower()
    tool_input = payload.get("tool_input") or {}
    cwd = payload.get("cwd")

    if tool == "write":
        root = locked_campaign_root(tool_input.get("file_path", ""), cwd)
        return emit_deny(REWRITE_REASON.format(root=root)) if root else 0

    if "bash" in tool:
        root = bash_overwrite_root(tool_input.get("command", ""), cwd)
        return emit_deny(REWRITE_REASON.format(root=root)) if root else 0

    if "edit" in tool:  # Edit / MultiEdit
        root = locked_campaign_root(tool_input.get("file_path", ""), cwd)
        if root is None:
            return 0
        cur = direction_unread(root)
        return emit_deny(DIRECTION_REASON.format(path=cur)) if cur else 0

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
