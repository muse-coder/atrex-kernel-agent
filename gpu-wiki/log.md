# GPU Wiki Log

Append-only, chronological record of what changed in this wiki and when. Newest
entries go at the top. Each entry starts with a fixed, grep-able prefix so the
history can be sliced with plain tools:

```text
## [YYYY-MM-DD] <op> | <subject>
```

`<op>` is one of `ingest`, `query`, `lint`, `normalize`. List the docs that were
touched and any `RELATIONS.md` updates in the body. See `CLAUDE.md` for the
operations that append here.

```text
# last 5 entries
grep "^## \[" log.md | head -5
# everything ingested
grep "^## \[" log.md | grep " ingest | "
```

---

## [2026-07-14] lint | validate Blackwell feature-centric reorganization

Regenerated the flat index; self-containment, index, and structure gates pass.
Ran `test_query.py`, `test_check_structure.py`, and
`test_check_self_contained.py` (72 tests total); all pass. `git diff --check`
also passes.

## [2026-07-14] normalize | consolidate Blackwell wiki around features

Reorganized `docs/nvidia/blackwell/` from eleven topic/form directories into
five top-level entry points: `features/`, `optimization/`, `programming/`,
`kernels/`, and `articles/`. Moved architecture-porting pages beside their
TCGEN05/TMEM features, folded CLC hands-on and persistent-scheduling guidance
into `features/clc.md`, moved general hands-on material into the owning feature,
kernel, or optimization area, and placed symptom pages under
`optimization/patterns/`. Rewrote relative links, regenerated the index, and
updated routing/catalog pages and `RELATIONS.md` references.

## [2026-07-14] lint | validate Blackwell CLC refactor

Rebuilt and checked the flat index, passed self-containment and structure gates,
ran 72 script unit tests, and scanned for the removed fake APIs and obsolete
SM90/SM100-only architecture claims. No residual matches or structural findings.

## [2026-07-14] ingest | refactor Blackwell CLC knowledge

Rebuilt `docs/nvidia/blackwell/hardware/clc.md` from the PTX ISA 8.6 contract
and local Gluon/CuTeDSL implementations, then aligned the hands-on, persistent
kernel, tile scheduling, PTX, architecture comparison, pattern, kernel, and
source-specific article pages. Removed invented queue/acquire/`clusterctl` APIs,
separated CLC work allocation from software rasterization, documented failure
and multicast rules, and corrected CLC availability to `sm_100` or higher,
including SM120. Updated the three local catalogs and `docs/RELATIONS.md`.

## [2026-07-14] lint | verify CLC architecture-boundary corrections

Ran the self-containment, generated-index, structure, and whitespace checks after
the CLC corrections; all checks pass with no structure findings.

## [2026-07-14] ingest | correct CLC architecture boundary and terminology

Corrected the CuTeDSL architecture reference from SM90+ to Blackwell SM100+,
expanded CLC consistently as Cluster Launch Control in the QuACK GEMM article,
and made the Blackwell hands-on page explicit that Hopper has Thread Block
Clusters but not CLC. Added the Hopper-vs-Blackwell distinction to
`docs/RELATIONS.md` and regenerated `docs/index.md`.

## [2026-07-01] merge | adopt PR #2 vendor-first reorg + rebuild query

Merged PR #2 (vendor-first + NVIDIA architecture-first restructure; removed
redundant non-kernel-optimization docs). Rebuilt the governance layer for the new
layout: `query.py` now matches by path segment so `--arch blackwell` excludes
`blackwell-geforce` (sm120) and Hopper; `build_index` groups by vendor/architecture;
regenerated `docs/index.md`; updated the `CLAUDE.md` taxonomy, README entry points,
and the `gpu-kernel-research` agent L1 scope. All checkers and unit tests pass.

## [2026-06-30] ingest | architecture-scoped query tool

Added `scripts/query.py` (+ `scripts/test_query.py`): keyword search over `docs/`
filtered by `--arch` / `--vendor` / `--dsl`, derived from the path taxonomy, so a
Blackwell query never returns Hopper or CDNA pages (architecture-neutral and
general pages are always included). Wired it into the `gpu-kernel-research` agent's
L1 retrieval step and the `CLAUDE.md` Query operation, added a README routing row,
and added the unit test to the CI gate.

## [2026-06-30] normalize | page-schema sweep + strict structural gate

Normalized page conventions across `docs/`: renamed every `## Related Docs` /
`## Related Documents` heading to `## Related`, fixed `**Last Updated**` →
`**Last updated**` casing, and added a missing `# H1` title (plus a one-line
summary where the body was table/code-first) to 11 pages. Made `build_index` and
`check_structure` fence-aware so `#` lines inside code blocks are no longer
mistaken for titles, regenerated `docs/index.md`, added
`scripts/test_check_structure.py`, and flipped the CI gate to
`check_structure.py --strict` (title + summary now enforced; missing-`## Related`,
orphans, and RELATIONS staleness remain advisory).

## [2026-06-30] lint | bootstrap structural baseline

Ran `scripts/check-self-contained.py` (green) and `scripts/check_structure.py`.
Baseline structural findings: `missing-related` on most legacy pages,
`missing-summary` on a small set, one `missing-h1`
(`docs/ref-docs/nvidia/common/nvidia-ptx-sync-and-async.md`), zero orphans. These
are tracked as non-blocking warnings until the Phase 2 normalization sweep.

## [2026-06-30] ingest | governance layer (index.md, log.md, schema)

Added the wiki-governance layer: generated `docs/index.md` (flat catalog),
created this `log.md`, rewrote `CLAUDE.md` into the operations schema, and pointed
`AGENTS.md`/`README.md` at them. No content pages moved; directory taxonomy and
relative-link convention unchanged.
## [2026-07-14] normalize | further simplify Blackwell navigation

Flattened the thin `optimization/patterns/` branch into `optimization/`, moved
feature-adjacent quantization guidance into `features/`, and collapsed Gluon and
Triton programming notes and pitfalls into the programming root. Kept the
larger CuTeDSL collection nested, left `articles/` unchanged, repaired all
relative links, and regenerated the flat index.

## [2026-07-14] lint | validate simplified Blackwell layout

Passed self-containment, generated-index, structure, and whitespace checks, plus
the 72 query, structure, and self-containment script tests.
## [2026-07-14] ingest | structured Blackwell optimization retrieval

Extended `scripts/query.py` with repeatable `--section`, `--exclude-section`,
`--kernel-type`, and profiler-aligned `--symptom` filters. Kernel types use only
stable title/path vocabulary, while symptoms select the diagnosis cards named by
`tools/classify_ncu.py`. Updated the research-agent L1 procedure and wiki routing
examples so Blackwell GEMM retrieval separates operator cases, profile diagnosis,
and hardware mechanisms.

## [2026-07-14] lint | validate structured wiki retrieval

Passed 21 query-tool unit tests, index synchronization, self-containment,
structure, and whitespace checks. A Blackwell GEMM implementation search now
returns 16 role- and operator-filtered pages; a `pipeline-stalls` diagnosis query
returns its single optimization card.
## [2026-07-14] ingest | register operator-level wiki retrieval

Added `query.py --operator` with canonical names and aliases for GEMM/GEMV,
MoE, FlashAttention, GDN, MLA, NSA, Mamba, normalization, Softmax, convolution,
AllReduce, and Paged Attention. Operator matching uses title/path markers only.
Added a Blackwell-kernel inventory test that requires every current page in
`docs/nvidia/blackwell/kernels/` to have a registered operator route; a missing
mapping now fails CI instead of silently becoming unsearchable.

## [2026-07-14] lint | validate operator coverage and scoped retrieval

Passed the 22 query tests, generated-index, self-containment, structure, and
whitespace checks. Verified GDN and FlashAttention each resolve to one Blackwell
kernel page, while Paged Attention correctly reports no Blackwell operator page.
