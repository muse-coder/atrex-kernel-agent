#!/usr/bin/env python3
"""Correctness check for FP8 GEMM candidate vs baseline (torch._scaled_mm).

Verifies:
1. Output buffer poisoning (NaN fill before each call)
2. No NaN/Inf in outputs
3. Numerical closeness within FP8 tolerances
4. All workload shapes pass
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "solution"))

from fp8_gemm_baseline import fp8_gemm_baseline
from fp8_gemm_candidate import fp8_gemm_candidate


def check_workload(wl: dict, device: torch.device) -> dict:
    M = wl["shapes"]["M"]
    N = wl["shapes"]["N"]
    K = wl["shapes"]["K"]
    atol = float(wl.get("atol", 0.125))
    rtol = float(wl.get("rtol", 0.05))
    wl_id = wl["id"]

    torch.manual_seed(42)
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    A = A_bf16.to(torch.float8_e4m3fn)
    B = B_bf16.to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
    scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Baseline
    C_base = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    C_base.fill_(float("nan"))  # poison
    fp8_gemm_baseline(A, B, scale_a, scale_b, C_base)
    torch.cuda.synchronize()

    if torch.isnan(C_base).any() or torch.isinf(C_base).any():
        return {"id": wl_id, "status": "FAIL",
                "message": "baseline output has NaN/Inf"}

    # Candidate
    C_cand = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    C_cand.fill_(float("nan"))  # poison
    fp8_gemm_candidate(A, B, scale_a, scale_b, C_cand)
    torch.cuda.synchronize()

    if torch.isnan(C_cand).any():
        return {"id": wl_id, "status": "FAIL",
                "message": "candidate output has NaN"}
    if torch.isinf(C_cand).any():
        return {"id": wl_id, "status": "FAIL",
                "message": "candidate output has Inf"}

    # Compare
    diff = (C_base.float() - C_cand.float()).abs()
    max_abs = diff.max().item()
    denom = C_base.float().abs().clamp_min(1e-12)
    max_rel = (diff / denom).max().item()

    ok = torch.all(diff <= (atol + rtol * C_base.float().abs()))
    if not ok:
        # Find first failing element for diagnostics
        fail_mask = diff > (atol + rtol * C_base.float().abs())
        fail_idx = fail_mask.nonzero()[0]
        r, c = fail_idx[0].item(), fail_idx[1].item()
        return {
            "id": wl_id, "status": "FAIL",
            "message": (
                f"exceeds tolerance atol={atol} rtol={rtol} "
                f"max_abs={max_abs:.6f} max_rel={max_rel:.6f} "
                f"first_fail=({r},{c}) base={C_base[r,c].item():.6f} "
                f"cand={C_cand[r,c].item():.6f}"
            ),
        }

    return {
        "id": wl_id, "status": "PASS",
        "max_abs": max_abs, "max_rel": max_rel,
        "message": f"OK max_abs={max_abs:.6f} max_rel={max_rel:.6f}",
    }


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    workloads = json.loads((BENCH_DIR / "workloads.json").read_text())
    print(f"Running correctness checks for {len(workloads)} workloads...")
    print()

    all_pass = True
    for wl in workloads:
        result = check_workload(wl, device)
        status = result["status"]
        if status != "PASS":
            all_pass = False
        print(f"  [{status}] {result['id']}: {result['message']}")

    print()
    if all_pass:
        print("ALL WORKLOADS PASSED")
        return 0
    else:
        print("SOME WORKLOADS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
