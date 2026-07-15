#!/usr/bin/env python3
"""Classify rocprofv3 ATT/PMC/ASM artifacts into the shared profile contract.

The classifier is deliberately conservative: it emits a symptom only when an
artifact contains a direct counter or instruction-level signal. Missing data is
reported as ``insufficient-evidence`` rather than guessed from source code.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from profile_summary import build_summary, write_summary


def numeric(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def csv_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return ""


def att_signal(path: Path) -> tuple[bool, bool, bool]:
    """Return (memory stall, pipeline stall, register spill) from an ATT CSV."""
    memory = pipeline = spill = False
    try:
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            for row in csv.DictReader(handle):
                instruction = row.get("Instruction", "").lower()
                stall = numeric(row.get("Stall", "")) or 0.0
                if "scratch" in instruction:
                    spill = True
                if stall > 0 and any(token in instruction for token in
                                     ("buffer_load", "flat_load", "global_load", "ds_read", "ds_write")):
                    memory = True
                if stall > 0 and any(token in instruction for token in
                                     ("s_waitcnt", "s_barrier", "s_wait", "s_sendmsg")):
                    pipeline = True
    except (OSError, csv.Error):
        pass
    return memory, pipeline, spill


def pmc_nonzero(path: Path, counter: str) -> bool:
    """Return true only when a named PMC has an observed non-zero value."""
    try:
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    if key and counter in key.lower() and (numeric(value or "") or 0.0) > 0:
                        return True
    except (OSError, csv.Error):
        pass
    return False


def classify(profile_dir: Path) -> dict:
    evidence: list[str] = []
    localize: list[str] = []
    findings: list[dict] = []
    symptoms: set[str] = set()

    for stats in sorted((profile_dir / "att").rglob("stats_*.csv")):
        rel = stats.relative_to(profile_dir).as_posix()
        evidence.append(rel)
        memory, pipeline, spill = att_signal(stats)
        if memory:
            symptoms.add("memory-bound")
            localize.append(rel)
            findings.append({"pattern": "att-memory-stall", "confidence": "medium",
                             "evidence": f"memory instruction with Stall > 0 in {rel}"})
        if pipeline:
            symptoms.add("pipeline-stalls")
            localize.append(rel)
            findings.append({"pattern": "att-wait-stall", "confidence": "medium",
                             "evidence": f"wait/barrier instruction with Stall > 0 in {rel}"})
        if spill:
            symptoms.add("register-pressure")
            localize.append(rel)
            findings.append({"pattern": "att-scratch", "confidence": "high",
                             "evidence": f"scratch instruction in {rel}"})

    for pmc in sorted((profile_dir / "pmc").rglob("*.csv")):
        rel = pmc.relative_to(profile_dir).as_posix()
        evidence.append(rel)
        if pmc_nonzero(pmc, "spi_ra_vgpr_sgpr_full_csn"):
            symptoms.add("register-pressure")
            localize.append(rel)
            findings.append({"pattern": "pmc-vgpr-pressure", "confidence": "medium",
                             "evidence": f"VGPR pressure counter collected in {rel}"})
        if pmc_nonzero(pmc, "sq_lds_bank_conflict") or pmc_nonzero(pmc, "tcp_tcc_miss"):
            symptoms.add("memory-bound")
            localize.append(rel)
            findings.append({"pattern": "pmc-memory", "confidence": "low",
                             "evidence": f"memory-pressure counter collected in {rel}"})

    for asm in sorted((profile_dir / "asm").rglob("*")):
        if not asm.is_file() or asm.suffix not in {".s", ".amdgcn"}:
            continue
        rel = asm.relative_to(profile_dir).as_posix()
        evidence.append(rel)
        text = csv_text(asm)
        if "scratch_load" in text or "scratch_store" in text:
            symptoms.add("register-pressure")
            localize.append(rel)
            findings.append({"pattern": "asm-scratch", "confidence": "high",
                             "evidence": f"scratch load/store in {rel}"})

    status = "complete" if symptoms else "insufficient-evidence"
    if not evidence:
        reason = "No ATT stats, PMC CSV, or AMDGCN assembly artifacts were produced."
    elif not symptoms:
        reason = "Artifacts were collected, but none contained a direct supported bottleneck signal."
    else:
        reason = None
    return build_summary(platform="amd", classification_status=status, evidence=evidence,
                         symptoms=sorted(symptoms), localize=localize,
                         findings=findings, reason=reason)


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify rocprofv3 artifacts into summary.json.")
    parser.add_argument("--profile-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = classify(args.profile_dir)
    write_summary(args.profile_dir, summary, "ROCm Profile Summary")
    print(f"Summary saved to: {args.profile_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
