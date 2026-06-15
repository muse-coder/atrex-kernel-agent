#!/usr/bin/env python3
"""SessionStart hook: re-inject IterKernel HARD REQUIREMENTS into context.

These three rules must stay in effect even after the conversation is compacted.
SessionStart fires on startup / resume / clear / **compact**, so this script's
stdout is re-added to the fresh context window right after a compaction —
making the rules context-compression-proof (the same guarantee mechanism as
block_solution_rewrite.py, which is disk/process based rather than context based).

Output goes to stdout; for SessionStart hooks Claude Code adds stdout to context.
"""
import json
import sys

REQUIREMENTS = """\
[IterKernel HARD REQUIREMENTS — always in effect, re-injected after every context compaction]

1. FROM SCRATCH: the candidate kernel MUST be designed and implemented from
   scratch. NEVER take an existing implementation (a prior campaign's kernel, a
   library kernel, copied code) as a code starting point to "continue/patch".
   Even an identical shape+GPU prior campaign is reference-only (read its
   NCU/SASS, borrow ideas) — open a NEW empty .cu and write the PTX wrappers,
   warp roles, mainloop and epilogue yourself.

2. STRONGEST BASELINE: baseline = the fastest available library implementation.
   Measure BOTH PyTorch (cuBLAS / torch._scaled_mm) and FlashInfer, pick the
   faster, and record the comparison in docs/baseline_source.md. Never beat a
   weak baseline.

3. NCU IS AUTHORITATIVE FOR PERFORMANCE: judge baseline-vs-candidate, each
   round's progress, and the final best version by NCU kernel duration
   (gpu__time_duration). bench wall-clock is secondary only.

Also: do NOT mistake benchmark-harness / overhead fixes (.py wrapper, .item(),
copy_, timing code) for "optimization". Optimization = the kernel's architecture
and instruction-level work. If you're editing those instead of the .cu, you've
gone off-track — return to the kernel.

Full text: CLAUDE.md "硬性要求" + .claude/commands/optimize-kernel.md 全局铁律 -2/-1/-0.5."""


def main() -> int:
    # Read (and ignore) the hook JSON payload on stdin if present.
    try:
        sys.stdin.read()
    except Exception:
        pass
    # For SessionStart, stdout is injected into the model context.
    print(REQUIREMENTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
