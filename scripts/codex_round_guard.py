#!/usr/bin/env python3
"""Manual process guard for the Codex IterKernel agent.

Claude Code runs IterKernel checks through hooks. Codex does not rely on those
hooks, so the Codex role contracts call this script explicitly:

  python scripts/codex_round_guard.py mark-direction-read /tmp/<slug> 3
  python scripts/codex_round_guard.py pre-edit /tmp/<slug> 3
  python scripts/codex_round_guard.py status /tmp/<slug>

The guard intentionally enforces only process gates that can be checked from
disk. It does not replace the one-lever diff review done by analysis.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED_PREV_ROUND_ARTIFACTS = [
    "candidate.ptx",
    "candidate.cubin",
    "candidate-sass.txt",
    "candidate-res-usage.txt",
    "candidate-nvdisasm.txt",
    "candidate.ncu-rep",
    "candidate-details.txt",
    "candidate-metrics.csv",
    "correctness-pass.txt",
    "analysis.md",
]

ROUND_DECL_RE = re.compile(
    r"(?:current[_ ]round|当前轮号?)\s*[:：=]\s*r?(\d+)", re.IGNORECASE
)


def die(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 2


def campaign_root(raw: str) -> Path:
    root = Path(raw).expanduser().resolve()
    return root


def current_dir(root: Path) -> Path:
    return root / ".rlcr" / "current"


def rounds_dir(root: Path) -> Path:
    return current_dir(root) / "rounds"


def initial_impl_done(root: Path) -> bool:
    return (current_dir(root) / ".initial-impl-done").exists()


def direction_marker(root: Path) -> Path:
    return current_dir(root) / ".direction-read-marker"


def direction_file(root: Path, round_num: int | None) -> Path | None:
    cur = current_dir(root)
    if round_num is not None:
        path = cur / "rounds" / f"r{round_num}" / "direction.md"
        return path if path.exists() else None

    candidates: list[Path] = []
    try:
        candidates.extend((cur / "rounds").glob("r*/direction.md"))
    except OSError:
        pass
    try:
        candidates.extend((cur / "modules").glob("*/round-*-direction.md"))
    except OSError:
        pass
    fallback = cur / "direction.md"
    if fallback.exists():
        candidates.append(fallback)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_declared_round(root: Path) -> int | None:
    try:
        text = (current_dir(root) / "state.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = ROUND_DECL_RE.search(text)
    return int(match.group(1)) if match else None


def round_numbers(root: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    try:
        entries = list(rounds_dir(root).iterdir())
    except OSError:
        return found
    for entry in entries:
        match = re.fullmatch(r"r(\d+)", entry.name)
        if match and entry.is_dir():
            found[int(match.group(1))] = entry
    return found


def previous_round_dir(root: Path, round_num: int | None) -> Path | None:
    nums = round_numbers(root)
    if not nums:
        return None
    n = round_num if round_num is not None else read_declared_round(root)
    if n is not None:
        prev = [num for num in nums if num < n]
        return nums[max(prev)] if prev else None
    ordered = sorted(nums)
    if len(ordered) < 2:
        return None
    return nums[ordered[-2]]


def missing_previous_artifacts(root: Path, round_num: int | None) -> tuple[Path, list[str]] | None:
    prev = previous_round_dir(root, round_num)
    if prev is None:
        return None
    missing = [name for name in REQUIRED_PREV_ROUND_ARTIFACTS if not (prev / name).exists()]
    return (prev, missing) if missing else None


def direction_unread(root: Path, round_num: int | None) -> Path | None:
    path = direction_file(root, round_num)
    if path is None:
        return None
    marker = direction_marker(root)
    try:
        if not marker.exists() or path.stat().st_mtime > marker.stat().st_mtime:
            return path
    except OSError:
        return path
    return None


def command_mark_direction_read(root: Path, round_num: int | None) -> int:
    if not current_dir(root).is_dir():
        return die(f"not a campaign with .rlcr/current: {root}")
    path = direction_file(root, round_num)
    if path is None:
        return die(f"direction.md not found for round {round_num or '<latest>'}: {root}")
    marker = direction_marker(root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    print(f"OK: marked direction as read: {path}")
    return 0


def command_pre_edit(root: Path, round_num: int | None) -> int:
    if not current_dir(root).is_dir():
        return die(f"not a campaign with .rlcr/current: {root}")
    if not initial_impl_done(root):
        print("OK: initial implementation lock is not active yet; pre-edit gate skipped.")
        return 0

    if round_num is not None and direction_file(root, round_num) is None:
        return die(f"direction.md not found for round {round_num}: {root}")

    unread = direction_unread(root, round_num)
    if unread is not None:
        return die(
            "current direction has not been marked as read. Read it, then run "
            f"`codex_round_guard.py mark-direction-read {root} {round_num or ''}`. "
            f"direction={unread}"
        )

    missing = missing_previous_artifacts(root, round_num)
    if missing is not None:
        prev, names = missing
        return die(
            f"previous round artifacts are incomplete: {prev}; missing: {', '.join(names)}"
        )

    print("OK: Codex pre-edit gates passed.")
    return 0


def command_status(root: Path) -> int:
    if not current_dir(root).is_dir():
        return die(f"not a campaign with .rlcr/current: {root}")
    declared = read_declared_round(root)
    nums = sorted(round_numbers(root))
    latest = nums[-1] if nums else None
    direction = direction_file(root, declared or latest)
    unread = direction_unread(root, declared or latest)
    missing = missing_previous_artifacts(root, declared or latest)

    print(f"campaign: {root}")
    print(f"initial_impl_done: {initial_impl_done(root)}")
    print(f"declared_round: {declared}")
    print(f"latest_round: {latest}")
    print(f"direction: {direction}")
    print(f"direction_read: {unread is None}")
    if missing is None:
        print("previous_round_artifacts: complete or not applicable")
    else:
        prev, names = missing
        print(f"previous_round_artifacts: incomplete ({prev})")
        print("missing: " + ", ".join(names))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["mark-direction-read", "pre-edit", "status"],
        help="guard operation to run",
    )
    parser.add_argument("campaign_dir", help="path to /tmp/<slug> campaign repo")
    parser.add_argument("round", nargs="?", type=int, help="current round number N")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = campaign_root(args.campaign_dir)
    if args.command == "mark-direction-read":
        return command_mark_direction_read(root, args.round)
    if args.command == "pre-edit":
        return command_pre_edit(root, args.round)
    if args.command == "status":
        return command_status(root)
    return die(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
