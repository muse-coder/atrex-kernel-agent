#!/usr/bin/env python3
"""Shared machine-readable contract for NVIDIA and AMD profile diagnosis."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CLASSIFICATION_STATUSES = {"complete", "insufficient-evidence", "blocked", "skipped"}


def build_summary(*, platform: str, classification_status: str,
                  evidence: list[str], symptoms: list[str] | None = None,
                  localize: list[str] | None = None,
                  findings: list[dict[str, Any]] | None = None,
                  reason: str | None = None) -> dict[str, Any]:
    if classification_status not in CLASSIFICATION_STATUSES:
        raise ValueError(f"unknown classification status: {classification_status}")
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform,
        "collector_status": "complete",
        "classification_status": classification_status,
        "evidence": sorted(set(evidence)),
        "symptoms": sorted(set(symptoms or [])),
        "localize": sorted(set(localize or [])),
        "findings": findings or [],
        "reason": reason,
        "suggested_queries": [
            {"section": "optimization", "symptom": symptom}
            for symptom in sorted(set(symptoms or []))
        ],
    }


def render_text(summary: dict[str, Any], heading: str) -> str:
    lines = [f"===== {heading} =====", f"Platform: {summary['platform']}",
             f"Classification Status: {summary['classification_status']}"]
    if summary["reason"]:
        lines.append(f"Reason: {summary['reason']}")
    if summary["evidence"]:
        lines.extend(["", "Evidence:"] + [f"  - {path}" for path in summary["evidence"]])
    lines.extend(["", "===== Diagnosis ====="])
    if summary["symptoms"]:
        lines.append("SYMPTOMS: " + ", ".join(summary["symptoms"]))
    else:
        lines.append("SYMPTOMS: none")
    if summary["localize"]:
        lines.append("LOCALIZE: " + ", ".join(summary["localize"]))
    if summary["suggested_queries"]:
        lines.extend(["", "===== Suggested Queries ====="])
        lines.extend(f"gpu-wiki: --section {item['section']} --symptom {item['symptom']}"
                     for item in summary["suggested_queries"])
    return "\n".join(lines) + "\n"


def write_summary(output_dir: Path, summary: dict[str, Any], heading: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.txt").write_text(render_text(summary, heading), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a blocked/skipped profile summary contract.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=("nvidia", "amd"))
    parser.add_argument("--status", required=True, choices=sorted(CLASSIFICATION_STATUSES))
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence", action="append", default=[])
    args = parser.parse_args()
    summary = build_summary(platform=args.platform, classification_status=args.status,
                            evidence=args.evidence, reason=args.reason)
    write_summary(args.output_dir, summary, "Profile Summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
