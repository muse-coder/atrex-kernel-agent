#!/usr/bin/env python3
"""Persist and validate post-change NVIDIA SASS/PTX evidence for one iteration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import extract_nvidia_asm as asm


def extract_ptx(cubin: Path) -> str | None:
    if not shutil.which("cuobjdump"):
        return None
    result = subprocess.run(["cuobjdump", "--dump-ptx", str(cubin)],
                            capture_output=True, text=True, timeout=60)
    return result.stdout if result.returncode == 0 else None


def evaluate_sass(sass: str, resource: str | None, arch: str,
                  expected_checks: list[str]) -> dict:
    expected = asm.analyze_expected_instructions(sass, arch)
    spills = asm.analyze_spills(sass, resource)
    widths = asm.analyze_load_width(sass)
    checks = {
        "tensor-core": expected["has_tensor_core"],
        "async-copy": expected["has_async_copy"],
        "no-spills": not spills["has_spills"],
        "vectorized-global-load": widths["global_load"]["128"] > 0,
    }
    failed = [name for name in expected_checks if not checks[name]]
    return {
        "expected": expected_checks,
        "checks": checks,
        "passed": not failed,
        "failed_expectations": failed,
        "spills": spills,
        "expected_instructions": expected,
        "load_width": widths,
        "scalar_fallback": asm.analyze_scalar_fallback(sass),
        "instruction_mix": asm.analyze_instruction_mix(sass),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NVIDIA ISA evidence after a candidate kernel change.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ncu-rep", type=Path, help="Candidate NCU report; preferred for JIT kernels")
    source.add_argument("--cubin", type=Path, help="Candidate cubin or .so")
    parser.add_argument("--arch", required=True, choices=("sm90", "sm100"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expect", action="append", default=[],
                        choices=("tensor-core", "async-copy", "no-spills", "vectorized-global-load"))
    parser.add_argument("--dump-ptx", action="store_true",
                        help="Dump PTX when --cubin is provided; unavailable PTX is recorded, not invented.")
    args = parser.parse_args()

    if args.ncu_rep:
        sass, resource = asm.extract_sass_from_ncu_rep(str(args.ncu_rep))
        source_path = str(args.ncu_rep)
    else:
        sass = asm.extract_sass_from_cubin(str(args.cubin))
        resource = asm.extract_ptxas_stats(str(args.cubin))
        source_path = str(args.cubin)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sass_path = args.output_dir / "candidate.sass"
    sass_path.write_text(sass, encoding="utf-8")
    evaluation = evaluate_sass(sass, resource, args.arch, args.expect)
    ptx_path = None
    if args.dump_ptx and args.cubin:
        ptx = extract_ptx(args.cubin)
        if ptx is not None:
            ptx_path = args.output_dir / "candidate.ptx"
            ptx_path.write_text(ptx, encoding="utf-8")

    result = {
        "platform": "nvidia",
        "arch": args.arch,
        "source": source_path,
        "sass": sass_path.name,
        "ptx": ptx_path.name if ptx_path else None,
        **evaluation,
    }
    report = args.output_dir / "isa_validation.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"ISA validation: {'PASS' if result['passed'] else 'FAIL'} ({report})")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
