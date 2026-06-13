"""Adapter for FP8 GEMM benchmark: baseline (torch._scaled_mm) vs candidate."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "solution"))


def _get_dtype(name: str):
    return getattr(torch, name)


def make_case(workload: dict, *, device: torch.device, seed: int) -> dict:
    shapes = workload["shapes"]
    M, N, K = shapes["M"], shapes["N"], shapes["K"]
    dtype_a = _get_dtype(workload.get("dtype_a", "float8_e4m3fn"))
    dtype_b = _get_dtype(workload.get("dtype_b", "float8_e4m3fn"))
    dtype_out = _get_dtype(workload.get("dtype_out", "bfloat16"))

    torch.manual_seed(seed)
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    A = A_bf16.to(dtype_a)
    B = B_bf16.to(dtype_b)

    scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
    scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)

    inputs = {
        "A": A,
        "B": B,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "dtype_out": dtype_out,
    }

    baseline_outputs = {
        "C": torch.empty(M, N, dtype=dtype_out, device=device),
    }
    candidate_outputs = {
        "C": torch.empty(M, N, dtype=dtype_out, device=device),
    }

    tolerance = {
        "atol": float(workload.get("atol", 0.125)),
        "rtol": float(workload.get("rtol", 0.05)),
    }

    return {
        "inputs": inputs,
        "baseline_outputs": baseline_outputs,
        "candidate_outputs": candidate_outputs,
        "tolerance": tolerance,
    }


def call_baseline(workload: dict, inputs: dict, outputs: dict) -> None:
    A = inputs["A"]
    B = inputs["B"]
    scale_a = inputs["scale_a"]
    scale_b = inputs["scale_b"]
    dtype_out = inputs["dtype_out"]

    result = torch._scaled_mm(
        A, B.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=dtype_out,
    )
    outputs["C"].copy_(result)


def call_candidate(workload: dict, inputs: dict, outputs: dict) -> None:
    try:
        from fp8_gemm_candidate import fp8_gemm_candidate
        fp8_gemm_candidate(
            inputs["A"], inputs["B"], inputs["scale_a"], inputs["scale_b"],
            outputs["C"]
        )
    except ImportError:
        A = inputs["A"]
        B = inputs["B"]
        scale_a = inputs["scale_a"]
        scale_b = inputs["scale_b"]
        dtype_out = inputs["dtype_out"]
        result = torch._scaled_mm(
            A, B.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=dtype_out,
        )
        outputs["C"].copy_(result)
