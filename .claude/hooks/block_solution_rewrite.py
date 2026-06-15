#!/usr/bin/env python3
"""IterKernel anti-rewrite guard (PreToolUse hook).

Enforces the optimize-kernel.md rule: once a campaign's first kernel
implementation is committed, ``solution/`` files may only change via incremental
``Edit``. Wholesale ``Write`` rewrites (or shell overwrites) are blocked.

Trigger key: the per-campaign marker ``<root>/.rlcr/current/.initial-impl-done``
(created in optimize-kernel.md Step 5.7). The guard does nothing until that
marker exists, so it never interferes with the first implementation or with any
repo that is not an IterKernel campaign.

Matcher (project .claude/settings.json): ``Write|Bash``. ``Edit``/``MultiEdit``
are the sanctioned incremental tools and are intentionally never matched.

Decision protocol: print a PreToolUse ``permissionDecision: deny`` JSON on
stdout and exit 0 (per Claude Code hooks docs). Allow == exit 0, no output.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MARKER_REL = (".rlcr", "current", ".initial-impl-done")

DENY_REASON = (
    "🔒 渐进式修改锁已激活（{root}/.rlcr/current/.initial-impl-done 存在）。"
    "solution/ 首次实现后只能用 Edit 增量修改，禁止 Write 覆盖或 shell 重定向重写。"
    "若要回退请用 `git checkout HEAD -- solution/`，再用小步 Edit 重试。"
)


def deny(root: Path) -> int:
    reason = DENY_REASON.format(root=root)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }, ensure_ascii=False))
    return 0


def locked_campaign_root(raw: str, cwd: str | None) -> Path | None:
    """Return the campaign root if ``raw`` points under a *locked* solution/ dir.

    A path is locked when it contains a path segment exactly equal to
    ``solution`` and ``<parent-of-solution>/.rlcr/current/.initial-impl-done``
    exists. Requiring an exact segment (not a substring) avoids false positives
    like ``profiles/solution-sass.txt``.
    """
    if not raw:
        return None
    raw = raw.strip().strip('"').strip("'")
    if "solution" not in raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute() and cwd:
        path = Path(cwd).expanduser() / path
    parts = path.parts
    for i, seg in enumerate(parts):
        if seg == "solution" and i > 0:
            root = Path(*parts[:i])
            if root.joinpath(*MARKER_REL).exists():
                return root
    return None


def check_write(tool_input: dict, cwd: str | None) -> Path | None:
    return locked_campaign_root(tool_input.get("file_path", ""), cwd)


# Shell tokens that indicate the command writes/overwrites a file in place.
_REDIRECT_RE = re.compile(r"""\d*>>?\s*("[^"]*"|'[^']*'|[^\s;|&)]+)""")
_SED_INPLACE_RE = re.compile(r"\bsed\b[^|;&]*-i")
_TEE_RE = re.compile(r"\btee\b")
_TRUNC_RE = re.compile(r"\b(?:truncate|dd)\b")


def _tokens(command: str) -> list[str]:
    return re.findall(r"""(?:"[^"]*"|'[^']*'|[^\s;|&]+)""", command)


def check_bash(tool_input: dict, cwd: str | None) -> Path | None:
    """Block shell commands that overwrite a locked solution/ file.

    Conservative: only redirection targets, ``tee`` args, ``sed -i`` file args,
    and ``truncate``/``dd`` targets are treated as write targets. Read-only uses
    of solution/ (``nvcc solution/k.cu -o x``, ``git checkout -- solution/``,
    ``cp solution/k.cu backup``) are NOT flagged because solution/ is not a
    write target there.
    """
    command = tool_input.get("command", "")
    if not command or "solution" not in command:
        return None

    candidates: list[str] = [m.group(1) for m in _REDIRECT_RE.finditer(command)]

    # In-place/streaming writers: their file arguments are the write target.
    if _SED_INPLACE_RE.search(command) or _TEE_RE.search(command) or _TRUNC_RE.search(command):
        candidates.extend(_tokens(command))

    for cand in candidates:
        root = locked_campaign_root(cand, cwd)
        if root is not None:
            return root
    return None


def main() -> int:
    raw = sys.stdin.read().strip()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return 0  # fail-open on malformed input; never break the session

    tool_name = (payload.get("tool_name") or "").lower()
    tool_input = payload.get("tool_input") or {}
    cwd = payload.get("cwd")

    if tool_name == "write":
        root = check_write(tool_input, cwd)
    elif tool_name == "bash":
        root = check_bash(tool_input, cwd)
    else:
        root = None

    if root is not None:
        return deny(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
