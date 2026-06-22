#!/usr/bin/env python3
"""
B300 (Blackwell Ultra, per-GPU) roofline / theoretical-ceiling calculator.

Given an operator (GEMM or attention), a shape, and a dtype, compute:
  - total FLOPs and HBM bytes moved (ideal lower bound: inputs read once, output written once)
  - arithmetic intensity AI = FLOPs / bytes
  - ridge point = peak_compute / peak_bandwidth
  - bound type (compute vs memory) by comparing AI vs ridge
  - roofline ceiling (TFLOPS) and the time floor t_floor (us)
  - the 90% completion target (us and TFLOPS), matching IterKernel hard rule #4

Optionally, pass --time-us <measured NCU duration> to also print achieved
TFLOPS / bandwidth and %-of-roofline utilization.

============================ DATA DISCIPLINE ============================
This tool refuses to invent peaks. It only auto-uses a compute peak when a
named source states it for that (dtype, sparsity), tagged by provenance, and
it will NOT manufacture a completion line for anything ambiguous.

  Auto-used B300 per-GPU peaks (each provenance tagged in the output):
    FP4 sparse = 18.0 PFLOPS  -- public DGX B300 guide (144 PF FP4 inference / 8;
                                 NVIDIA does not split dense vs sparse)
    FP4 dense  = 13.5 PFLOPS  -- LOCAL user-provided datasheet in
                                 docs/b300_hardware_specs.md (node 108 PF*/8);
                                 NOT in the public DGX B300 guide. The NVIDIA
                                 Blackwell Ultra blog cites 15 PF dense NVFP4 for
                                 some SKUs -> pass --peak-tflops 15000 to use it.

  REQUIRE --peak-tflops (no trustworthy built-in target -> hard error):
    FP8  -- the only public figure is 72 PF FP8 (node) = 9 PF/GPU with
            dense/sparse UNSTATED; using it as a dense target could be 2x off.
            The error prints 9000 so you can opt in deliberately.
    bf16 / fp16 / tf32 / fp32 -- not in any B300 source.

  HBM bandwidth: the B300 datasheet does NOT list it. Default 8 TB/s is the
  B200 HBM3e *reference* and only sets the ridge / memory ceiling as an
  estimate. Override with --bandwidth-tb-s once you have a measured/official
  B300 number (nvidia-smi or a memcpy bench). Non-datasheet values warn.
========================================================================

Caveats on the work models:
  - Attention bytes is the FlashAttention IDEAL LOWER BOUND (Q,K,V read once,
    O written once; the S×S score matrix never touches HBM). A real kernel may
    re-read K/V tiles, so measured bytes >= this. The memory ceiling is optimistic.
  - --causal multiplies score/PV FLOPs by ~0.5; an approximation (diagonal tiles
    are partially computed, so the true factor is slightly above 0.5).
  - FP4 inputs carry block-scale metadata (NVFP4: one shared FP8 scale per 16
    values). This tool adds ceil(elems/block)*scale_bytes PER input tensor.
    Tune with --fp4-scale-block / --fp4-scale-bytes, or disable with
    --fp4-scale-block 0. FP4 output is not supported (its scale is not modeled).

Examples:
  scripts/roofline_b300.py gemm --shape 4096,4096,4096 --dtype fp4 --sparse
  scripts/roofline_b300.py gemm --shape 1024,10240,4096 --dtype fp8 --peak-tflops 9000
  scripts/roofline_b300.py gemm --shape 1024,10240,4096 --dtype fp8 --peak-tflops 9000 --time-us 200
  scripts/roofline_b300.py attn --shape 1,32,4096,128 --dtype bf16 --peak-tflops 2250
  scripts/roofline_b300.py attn --shape 1,32,8192,128 --dtype bf16 --causal \
        --attn-eff 0.75 --peak-tflops 2250
"""

import argparse
import math
import sys

# --------------------------------------------------------------------------
# B300 per-GPU compute peaks (TFLOPS). Only (dtype, sparsity) pairs a named
# source states. value, provenance.
# --------------------------------------------------------------------------
KNOWN_PEAKS_TFLOPS = {
    "fp4": {
        "sparse": (18000.0, "DGX B300 guide (144PF FP4 inference/8; dense/sparse unsplit)"),
        "dense":  (13500.0, "LOCAL user datasheet docs/b300_hardware_specs.md (node 108PF*/8; not in public guide)"),
    },
}

# dtypes whose only public figure is an ambiguous headline -> refuse to
# auto-use; the error cites the number so the user can opt in via --peak-tflops.
HEADLINE_ONLY_TFLOPS = {
    "fp8": (9000.0, "72PF FP8 node/8; dense/sparse UNSTATED"),
}

# HBM bandwidth: B300 datasheet missing it. B200 HBM3e = 8 TB/s reference only.
BANDWIDTH_TB_S = (8.0, "reference(B200 HBM3e) -- B300 datasheet missing")

# Per-GPU memory: 288 GB per the public NVIDIA DGX B300 user guide (8x288=2.3TB).
# Note: docs/b300_hardware_specs.md (a user-provided datasheet) says 2.1TB/8
# = 262.5 GB. Informational only (not used in roofline). Conflict unresolved.
MEMORY_GB = 288.0

DTYPE_BYTES = {
    "fp4": 0.5, "fp8": 1.0, "bf16": 2.0, "fp16": 2.0, "tf32": 4.0, "fp32": 4.0,
}
# Output / accumulation precision implied by the input dtype.
DEFAULT_OUT_DTYPE = {
    "fp4": "bf16", "fp8": "bf16", "bf16": "bf16",
    "fp16": "fp16", "tf32": "fp32", "fp32": "fp32",
}

WARN = "\033[33m"
RST = "\033[0m"
_warned = set()


def _color(s, c):
    return f"{c}{s}{RST}" if sys.stdout.isatty() else s


def warn_once(msg):
    if msg not in _warned:
        _warned.add(msg)
        print(_color(f"  ⚠ {msg}", WARN), file=sys.stderr)


# --------------------------------------------------------------------------
# Spec resolution -- never invents a value
# --------------------------------------------------------------------------
def resolve_peak_tflops(dtype, sparse, override):
    if override is not None:
        if override <= 0:
            sys.exit("error: --peak-tflops must be > 0")
        return override, "override(--peak-tflops)"

    table = KNOWN_PEAKS_TFLOPS.get(dtype)
    if table:
        if sparse:
            if "sparse" in table:
                return table["sparse"]
            sys.exit(f"error: no built-in sparse peak for '{dtype}'. Pass --peak-tflops.")
        if "dense" in table:
            val, prov = table["dense"]
            if dtype == "fp4":
                warn_once("fp4 dense=13.5 PF/GPU is a LOCAL datasheet value (not the "
                          "public DGX B300 guide); NVIDIA Blackwell Ultra blog cites "
                          "15 PF dense NVFP4 for some SKUs -- pass --peak-tflops 15000 to use it.")
            return val, prov
        sys.exit(f"error: no built-in dense peak for '{dtype}'. Pass --peak-tflops.")

    if dtype in HEADLINE_ONLY_TFLOPS:
        v, prov = HEADLINE_ONLY_TFLOPS[dtype]
        sys.exit(
            f"error: '{dtype}' has no trustworthy dense/sparse B300 peak -- the only "
            f"public figure is an ambiguous headline {v:.0f} TFLOPS/GPU [{prov}]. "
            f"Using it as a dense target could be ~2x off, so pass it deliberately: "
            f"--peak-tflops {v:.0f}"
        )

    sys.exit(f"error: no built-in B300 peak for dtype '{dtype}'. Pass --peak-tflops <TFLOPS> "
             f"(e.g. a B200-class bf16 dense ~2250).")


def resolve_bandwidth(override):
    if override is not None:
        if override <= 0:
            sys.exit("error: --bandwidth-tb-s must be > 0")
        return override, "override(--bandwidth-tb-s)"
    return BANDWIDTH_TB_S


def elem_bytes(dtype):
    if dtype not in DTYPE_BYTES:
        sys.exit(f"error: unknown dtype '{dtype}'. Known: {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[dtype]


# --------------------------------------------------------------------------
# FP4 block-scale metadata: one shared scale per `block` values, per tensor.
# ceil() per tensor so small shapes still cost >= 1 scale.
# --------------------------------------------------------------------------
def fp4_scale_bytes(elem_counts, block, scale_bytes):
    if not block:  # 0 -> disabled
        return 0.0, 0
    n_scales = sum(math.ceil(c / block) for c in elem_counts)
    return n_scales * scale_bytes, n_scales


# --------------------------------------------------------------------------
# Operator FLOPs / bytes models
# --------------------------------------------------------------------------
def gemm_work(M, N, K, dtype, out_dtype, fp4_block, fp4_scale_b):
    flops = 2.0 * M * N * K
    b_in = elem_bytes(dtype)
    b_out = elem_bytes(out_dtype)
    bytes_ = b_in * (M * K + K * N) + b_out * (M * N)  # read A,B + write C
    scale_note = ""
    if dtype == "fp4":
        sb, n = fp4_scale_bytes([M * K, K * N], fp4_block, fp4_scale_b)  # A, B
        bytes_ += sb
        scale_note = (f"\n  + fp4 block-scale: ceil(M*K/{fp4_block})+ceil(K*N/{fp4_block}) "
                      f"= {n} scales x {fp4_scale_b}B = {sb/1e6:.3f} MB"
                      if fp4_block else "\n  + fp4 block-scale: disabled")
    detail = (f"FLOPs = 2*M*N*K = 2*{M}*{N}*{K}\n"
              f"  bytes = {b_in}*(M*K + K*N) + {b_out}*(M*N)  [read A,B once + write C]"
              f"{scale_note}")
    return flops, bytes_, detail


def attn_work(B, H, S, d, dtype, causal, kv_len, fp4_block, fp4_scale_b):
    Lq = S
    Lk = kv_len if kv_len else S
    flops = 4.0 * B * H * Lq * Lk * d
    causal_note = ""
    if causal:
        flops *= 0.5
        causal_note = " * ~0.5 (causal, approx)"
    b_in = elem_bytes(dtype)
    b_out = elem_bytes(DEFAULT_OUT_DTYPE[dtype])  # O written at accumulation precision
    qkv_elems = B * H * Lq * d + 2 * (B * H * Lk * d)  # Q + K + V
    o_elems = B * H * Lq * d                            # O
    bytes_ = b_in * qkv_elems + b_out * o_elems
    scale_note = ""
    if dtype == "fp4":
        sb, n = fp4_scale_bytes([B * H * Lq * d, B * H * Lk * d, B * H * Lk * d],
                                fp4_block, fp4_scale_b)  # Q, K, V only (O is b_out)
        bytes_ += sb
        scale_note = (f"\n  + fp4 block-scale (Q,K,V): {n} scales x {fp4_scale_b}B = {sb/1e6:.3f} MB"
                      if fp4_block else "\n  + fp4 block-scale: disabled")
    detail = (f"FLOPs = 4*B*H*Lq*Lk*d = 4*{B}*{H}*{Lq}*{Lk}*{d}{causal_note}  [QK^T + P@V]\n"
              f"  bytes = {b_in}*B*H*(Lq+2*Lk)*d [Q,K,V read] + {b_out}*B*H*Lq*d [O write]"
              f"{scale_note}\n"
              f"  NOTE: IDEAL LOWER BOUND (no S×S to HBM); a real kernel may re-read K/V.")
    return flops, bytes_, detail


# --------------------------------------------------------------------------
# Formatting / report
# --------------------------------------------------------------------------
def fmt_flops(f):
    for unit, div in (("PFLOP", 1e15), ("TFLOP", 1e12), ("GFLOP", 1e9), ("MFLOP", 1e6)):
        if f >= div:
            return f"{f/div:.3f} {unit}"
    return f"{f:.0f} FLOP"


def fmt_bytes(b):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b/div:.3f} {unit}"
    return f"{b:.0f} B"


def report(op_name, flops, bytes_, detail, peak_tflops, peak_prov,
           bw_tb_s, bw_prov, dtype, target_frac, attn_eff, time_us):
    ai = flops / bytes_
    ridge = peak_tflops / bw_tb_s
    compute_bound = ai >= ridge

    peak_flops = peak_tflops * 1e12
    peak_bw = bw_tb_s * 1e12

    if compute_bound:
        ceil_flops = peak_flops
        t_floor_s = flops / peak_flops
        bound = "COMPUTE-bound"
    else:
        ceil_flops = peak_bw * ai
        t_floor_s = bytes_ / peak_bw
        bound = "MEMORY-bound"

    eff = attn_eff if attn_eff is not None else 1.0
    eff_ceil_flops = ceil_flops * eff
    eff_t_floor_s = t_floor_s / eff
    target_t_s = eff_t_floor_s / target_frac
    target_flops = eff_ceil_flops * target_frac

    # Flag a completion bar built on a non-datasheet / override peak.
    headline_est = ("override" in peak_prov) or ("datasheet" not in peak_prov and "guide" not in peak_prov)
    bar_tag = "  (NOTE: target rests on a user-supplied/estimate peak)" if "override" in peak_prov else ""

    print(f"\n=== {op_name} | dtype={dtype} | B300 (per-GPU) ===")
    print(detail)
    print(f"\n  total FLOPs        : {fmt_flops(flops)}")
    print(f"  HBM bytes (ideal)  : {fmt_bytes(bytes_)}")
    print(f"  arithmetic intensity AI = {ai:.2f} FLOP/byte")

    print(f"\n  peak compute       : {peak_tflops:,.0f} TFLOPS   [{peak_prov}]")
    print(f"  HBM bandwidth      : {bw_tb_s:.3f} TB/s       [{bw_prov}]")
    print(f"  ridge point        : {ridge:.1f} FLOP/byte")
    if "datasheet" not in peak_prov and "guide" not in peak_prov and "override" not in peak_prov:
        warn_once(f"peak compute is {peak_prov} -- verify or pass --peak-tflops")
    if "datasheet" not in bw_prov and "override" not in bw_prov:
        warn_once(f"HBM bandwidth is {bw_prov} -- ridge & memory ceiling are estimates; pass --bandwidth-tb-s")

    print(f"\n  >> {bound}  (AI {'≥' if compute_bound else '<'} ridge)")
    if eff != 1.0:
        print(f"  >> attention structural efficiency applied: ×{eff:.2f}")
    print(f"\n  roofline ceiling   : {eff_ceil_flops/1e12:,.1f} TFLOPS")
    print(f"  time floor t_floor : {eff_t_floor_s*1e6:.2f} us   (fastest physically possible)")
    print(f"  {int(target_frac*100)}% target        : ≤ {target_t_s*1e6:.2f} us   "
          f"(≥ {target_flops/1e12:,.1f} TFLOPS)   <-- completion bar{bar_tag}")

    if time_us is not None:
        if time_us <= 0:
            sys.exit("error: --time-us must be > 0")
        t_s = time_us * 1e-6
        ach_flops = flops / t_s
        ach_bw = bytes_ / t_s
        print(f"\n  --- measured (time = {time_us:.2f} us) ---")
        print(f"  achieved compute   : {ach_flops/1e12:,.1f} TFLOPS  "
              f"({100*ach_flops/eff_ceil_flops:.1f}% of roofline ceiling)")
        print(f"  achieved bandwidth : {ach_bw/1e12:.3f} TB/s    "
              f"({100*ach_bw/peak_bw:.1f}% of peak HBM)")
        verdict = "DONE (≥ target)" if t_s <= target_t_s else "below target"
        print(f"  vs {int(target_frac*100)}% target     : {verdict}")
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _pos_ints(s, n, names):
    try:
        vals = [int(x) for x in s.split(",")]
    except ValueError:
        sys.exit(f"error: --shape must be {names} (integers)")
    if len(vals) != n or any(v <= 0 for v in vals):
        sys.exit(f"error: --shape must be {n} positive integers: {names}")
    return vals


def _frac(x, name):
    if not (0.0 < x <= 1.0):
        sys.exit(f"error: {name} must be in (0, 1]")
    return x


def main():
    p = argparse.ArgumentParser(
        description="B300 per-GPU roofline / theoretical-ceiling calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="op", required=True)

    g = sub.add_parser("gemm", help="C[M,N] = A[M,K] @ B[K,N]")
    g.add_argument("--shape", required=True, help="M,N,K")
    g.add_argument("--out-dtype", help="output dtype (default: bf16 for fp4/fp8, else same)")

    a = sub.add_parser("attn", help="FlashAttention-style attention")
    a.add_argument("--shape", required=True, help="B,H,S,d (batch, heads, seqlen, head_dim)")
    a.add_argument("--causal", action="store_true", help="causal mask (~half the FLOPs, approx)")
    a.add_argument("--kv-len", type=int, default=0, help="KV length if != S (cross/decode)")
    a.add_argument("--attn-eff", type=float, default=None,
                   help="attention structural efficiency in (0,1], e.g. 0.75 (softmax overhead)")

    for s in (g, a):
        s.add_argument("--dtype", required=True, choices=sorted(DTYPE_BYTES))
        s.add_argument("--sparse", action="store_true", help="use 2:4 sparse peak (default dense)")
        s.add_argument("--peak-tflops", type=float, help="override peak compute (TFLOPS)")
        s.add_argument("--bandwidth-tb-s", type=float, help="override HBM bandwidth (TB/s)")
        s.add_argument("--target-frac", type=float, default=0.90,
                       help="completion fraction of roofline in (0,1] (default 0.90)")
        s.add_argument("--time-us", type=float, help="measured kernel duration (NCU) to score utilization")
        s.add_argument("--fp4-scale-block", type=int, default=16,
                       help="fp4 block-scale group size (NVFP4=16); 0 disables")
        s.add_argument("--fp4-scale-bytes", type=float, default=1.0,
                       help="bytes per fp4 block scale (E4M3=1)")

    args = p.parse_args()

    target_frac = _frac(args.target_frac, "--target-frac")
    if args.fp4_scale_block < 0:
        sys.exit("error: --fp4-scale-block must be 0 (disabled) or > 0")
    if args.fp4_scale_bytes < 0:
        sys.exit("error: --fp4-scale-bytes must be >= 0")

    peak, peak_prov = resolve_peak_tflops(args.dtype, args.sparse, args.peak_tflops)
    bw, bw_prov = resolve_bandwidth(args.bandwidth_tb_s)

    if args.op == "gemm":
        M, N, K = _pos_ints(args.shape, 3, "M,N,K")
        out_dtype = args.out_dtype or DEFAULT_OUT_DTYPE[args.dtype]
        if out_dtype not in DTYPE_BYTES:
            sys.exit(f"error: unknown --out-dtype '{out_dtype}'")
        if out_dtype == "fp4":
            sys.exit("error: --out-dtype fp4 not supported (output block-scale not modeled); "
                     "use bf16/fp16/fp32")
        flops, bytes_, detail = gemm_work(M, N, K, args.dtype, out_dtype,
                                          args.fp4_scale_block, args.fp4_scale_bytes)
        name = f"GEMM M={M} N={N} K={K}"
        attn_eff = None
    else:
        B, H, S, d = _pos_ints(args.shape, 4, "B,H,S,d")
        if args.kv_len < 0:
            sys.exit("error: --kv-len must be >= 0")
        attn_eff = _frac(args.attn_eff, "--attn-eff") if args.attn_eff is not None else None
        flops, bytes_, detail = attn_work(B, H, S, d, args.dtype, args.causal, args.kv_len,
                                          args.fp4_scale_block, args.fp4_scale_bytes)
        name = f"ATTENTION B={B} H={H} S={S} d={d}" + (" causal" if args.causal else " full")

    report(name, flops, bytes_, detail, peak, peak_prov, bw, bw_prov,
           args.dtype, target_frac, attn_eff, args.time_us)


if __name__ == "__main__":
    main()
