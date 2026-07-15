# Iteration V<N> Plan

## Input Evidence
<profile evidence from Stage 1, compared with the previous iteration report when available>

## Search Log

Each row records one search action. Rules:
- **Layer**: L1 = gpu-wiki (docs/ + 3rdparty/ + reference-kernels/), L2 = reference-projects/, L3 = public net
- **New?**: Yes if this (Source, Query, Finding) triple has NOT appeared in any prior v*_plan.md Search Log; No otherwise
- The table MUST contain at least one row with New? = Yes (novelty constraint)
- Optimization action MUST be derived from at least one New? = Yes entry

| Source | Layer | Query | Finding | New? | Actionability |
|--------|-------|-------|---------|------|---------------|
| | L1 | | | | |
| | L2 | | | | |
| | L3 | | | | |

## This Iteration's Optimization Action
<choose exactly one optimization category>

## Performance Expectation and ISA Escalation (NVIDIA only)

- State the expected measurable effect from the Roofline model and this action
  (for example, increased tensor-core utilization, reduced spill risk, or a
  bounded latency/throughput improvement).
- **Do not** run SASS/PTX inspection merely because a plan exists.
- Escalate after the post-change performance test only when observed performance
  is materially below the modeled/plan expectation, regresses without an
  explained trade-off, or fails to make the expected resource/utilization move.
- On escalation, inspect the candidate `.ncu-rep` first; use a cubin/PTX only
  when the discrepancy may be caused by compiler lowering. Candidate checks may
  include `tensor-core`, `async-copy`, `no-spills`, and
  `vectorized-global-load`.

## Expected Impact
<how the action improves the current bottleneck and ISA targets>

## Risks and Rollback
<why it might fail and how to roll back>
