#!/usr/bin/env python3

"""Architecture-scoped keyword search over the gpu-wiki docs.

The wiki is organized vendor-first (`docs/{generic,nvidia,amd}/`) with NVIDIA
split by architecture (`hopper`/`blackwell`/`blackwell-geforce`) and AMD by
`gfx942`/`gfx950`. That taxonomy lives in each file's path, so a scoped search is
a path filter plus keyword ranking. The point is isolation: a ``--arch blackwell``
query returns Blackwell (and architecture-neutral) pages and never leaks Hopper,
CDNA, or Blackwell-GeForce (sm120) results.

Examples:
    python3 gpu-wiki/scripts/query.py "bank conflict" --arch blackwell
    python3 gpu-wiki/scripts/query.py "gemm" --arch b200 --vendor nvidia --section kernels --kernel-type gemm
    python3 gpu-wiki/scripts/query.py --arch b200 --operator gdn --section kernels
    python3 gpu-wiki/scripts/query.py --arch b200 --section optimization --symptom pipeline-stalls
    python3 gpu-wiki/scripts/query.py "flash attention" --arch cdna3 --dsl flydsl
    python3 gpu-wiki/scripts/query.py --list-arch

Matching is by path *segment* (a directory name) or a filename substring, so
`blackwell` matches `nvidia/blackwell/...` but not `nvidia/blackwell-geforce/...`.
A page is kept for a requested filter value when it carries that value's token OR
carries no token from that dimension at all (neutral / cross-arch pages such as
`nvidia/common/...` or `RELATIONS.md`). It is dropped only when it belongs to a
*different* value of the same dimension. `generic` pages are vendor-neutral and
survive any `--vendor` filter.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import build_index  # same directory; reuse title/summary extraction + fence masking


# canonical value -> tokens identifying it as a path segment or filename substring
ARCH_ALIASES = {
    "hopper": {"hopper", "sm90", "h100", "h20", "h200"},
    "blackwell": {"blackwell", "sm100", "sm103", "b200", "b300"},
    "blackwell-geforce": {"blackwell-geforce", "sm120"},
    "cdna3": {"cdna3", "gfx942", "mi300x", "mi308x"},
    "cdna4": {"cdna4", "gfx950", "mi355x"},
    "rdna4": {"rdna4", "gfx1250"},
    "ampere": {"ampere", "sm80", "a100"},
}
# `generic` is intentionally omitted: generic pages are vendor-neutral and should
# survive any --vendor filter rather than being treated as a competing vendor.
VENDOR_ALIASES = {
    "nvidia": {"nvidia"},
    "amd": {"amd"},
}
DSL_ALIASES = {
    "cutedsl": {"cutedsl"},
    "flydsl": {"flydsl"},
    "gluon": {"gluon"},
    "triton": {"triton"},
    "cuda": {"cuda"},
}

# `--symptom` consumes the controlled vocabulary emitted by
# `tools/classify_ncu.py` in a profiler `summary.txt`.  A symptom is a diagnosis
# card, not a fuzzy keyword: it therefore selects the card by its stable file
# name.  The card itself owns the candidate techniques and links to open next.
SYMPTOMS = {
    "compute-bound",
    "low-sm-utilization",
    "memory-bound",
    "pipeline-stalls",
    "register-pressure",
    "tail-effect",
    "moe-load-imbalance",
}

# Kernel type deliberately uses only stable path/title vocabulary, never body
# text.  This keeps an attention page that merely mentions GEMM out of a GEMM
# implementation search.  Profiling diagnosis is intentionally a separate
# `--symptom` query because strategy cards usually do not name one operator.
KERNEL_TYPE_ALIASES = {
    "gemm": {"gemm", "matmul", "matrix multiplication"},
    "gemv": {"gemv", "matrix vector"},
    "attention": {"attention", "flashmla", "mla"},
    "moe": {"moe", "mixture of experts", "expert"},
    "norm": {"norm", "rmsnorm", "layernorm"},
    "reduction": {"reduction", "softmax"},
}
KERNEL_TYPE_INPUT_ALIASES = {
    "matmul": "gemm",
    "matrix-multiply": "gemm",
    "matrix-multiplication": "gemm",
    "mixture-of-experts": "moe",
    "layer-norm": "norm",
    "rms-norm": "norm",
}


@dataclass(frozen=True)
class Operator:
    """A stable operator identifier, user spellings, and title/path markers."""

    aliases: frozenset[str]
    markers: frozenset[str]


# `--operator` is more specific than `--kernel-type`.  Match only title/path
# vocabulary, so an acronym such as GDN maps to its canonical page without
# letting unrelated body-text mentions become false positives.
OPERATORS = {
    "gemm": Operator(frozenset({"gemm", "matmul", "matrix-multiplication"}),
                     frozenset({"gemm", "matmul", "matrix multiplication"})),
    "gemv": Operator(frozenset({"gemv", "matrix-vector"}), frozenset({"gemv", "matrix vector"})),
    "grouped-gemm": Operator(frozenset({"grouped-gemm", "grouped-matmul"}),
                             frozenset({"grouped gemm"})),
    "gated-dual-gemm": Operator(frozenset({"gated-dual-gemm", "gate-up-gemm"}),
                                 frozenset({"gated dual gemm"})),
    "moe": Operator(frozenset({"moe", "mixture-of-experts", "fused-moe"}),
                    frozenset({"moe", "mixture of experts", "expert"})),
    "flash-attention": Operator(frozenset({"flash-attention", "flashattention", "fa", "fa4"}),
                                frozenset({"flash attention", "flashattention"})),
    "paged-attention": Operator(frozenset({"paged-attention", "paged-attn"}),
                                frozenset({"paged attention", "paged-attention"})),
    "mla": Operator(frozenset({"mla", "flashmla", "multi-head-latent-attention"}),
                    frozenset({"mla", "flashmla", "multi-head latent attention", "multi latent attention"})),
    "sparse-mla": Operator(frozenset({"sparse-mla", "sparse-multi-latent-attention"}),
                           frozenset({"sparse mla", "sparse multi"})),
    "gdn": Operator(frozenset({"gdn", "gated-delta-net", "gateddeltanet"}),
                    frozenset({"gated delta net", "gateddeltanet"})),
    "nsa": Operator(frozenset({"nsa", "native-sparse-attention"}),
                    frozenset({"native sparse attention"})),
    "mamba": Operator(frozenset({"mamba", "mamba2", "ssm"}),
                       frozenset({"mamba", "state space model", "ssm"})),
    "norm": Operator(frozenset({"norm", "rmsnorm", "layernorm", "rms-norm", "layer-norm"}),
                     frozenset({"rmsnorm", "rms norm", "layernorm", "layer norm"})),
    "softmax": Operator(frozenset({"softmax", "softmax-reduce"}), frozenset({"softmax"})),
    "conv": Operator(frozenset({"conv", "convolution"}), frozenset({"conv", "convolution"})),
    "allreduce": Operator(frozenset({"allreduce", "all-reduce"}), frozenset({"allreduce", "all reduce"})),
}

# non-canonical spellings a user might pass on the command line
ARCH_INPUT_ALIASES = {
    "sm90": "hopper",
    "h100": "hopper",
    "h20": "hopper",
    "sm100": "blackwell",
    "sm103": "blackwell",
    "b200": "blackwell",
    "b300": "blackwell",
    "geforce": "blackwell-geforce",
    "sm120": "blackwell-geforce",
    "gfx942": "cdna3",
    "mi300x": "cdna3",
    "mi308x": "cdna3",
    "gfx950": "cdna4",
    "mi355x": "cdna4",
    "gfx1250": "rdna4",
}

TITLE_WEIGHT = 3
SUMMARY_WEIGHT = 2
BODY_WEIGHT = 1


@dataclass
class Page:
    rel_path: str  # relative to docs/
    title: str
    summary: str
    segments: tuple[str, ...]  # path components (dirs + filename), lowercased
    filename: str  # lowercased filename
    keyword_blob: str  # path words for keyword scoring
    body: str


def path_segments(rel_path: str) -> tuple[str, ...]:
    return tuple(part for part in rel_path.lower().split("/") if part)


def keyword_blob(rel_path: str) -> str:
    text = rel_path.lower()
    for ch in "/-_.":
        text = text.replace(ch, " ")
    return text


def dimension_values(page: "Page", aliases: dict[str, set[str]]) -> set[str]:
    """Which canonical values of a dimension does this page belong to?"""
    found = set()
    for value, value_tokens in aliases.items():
        if any(token in page.segments or token in page.filename for token in value_tokens):
            found.add(value)
    return found


def matches_dimension(page: "Page", aliases: dict[str, set[str]], requested: set[str]) -> bool:
    if not requested:
        return True
    present = dimension_values(page, aliases)
    if not present:
        return True  # neutral page (no token in this dimension) — never excluded
    return bool(present & requested)


def normalize_path_selector(value: str) -> tuple[str, ...]:
    """Normalize a --section selector into docs-relative path components."""
    parts = tuple(part.lower() for part in value.strip("/ ").split("/") if part)
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(value)
    return parts


def has_path_selector(page: "Page", selector: tuple[str, ...]) -> bool:
    """Match a selector as one contiguous sequence of path components."""
    width = len(selector)
    return any(page.segments[index:index + width] == selector
               for index in range(len(page.segments) - width + 1))


def matches_sections(page: "Page", include: tuple[tuple[str, ...], ...],
                     exclude: tuple[tuple[str, ...], ...]) -> bool:
    """Include any requested knowledge role/path and exclude explicit paths."""
    if exclude and any(has_path_selector(page, selector) for selector in exclude):
        return False
    return not include or any(has_path_selector(page, selector) for selector in include)


def page_symptoms(page: "Page") -> set[str]:
    """Return profiler symptoms represented by a stable diagnosis-card filename."""
    stem = page.filename.removesuffix(".md")
    return {stem} if stem in SYMPTOMS else set()


def resolve_kernel_type(value: str) -> Optional[str]:
    value = value.lower().replace("_", "-")
    if value in KERNEL_TYPE_ALIASES:
        return value
    return KERNEL_TYPE_INPUT_ALIASES.get(value)


def page_kernel_types(page: "Page") -> set[str]:
    """Classify kernel types from title/path only, avoiding body-text false positives."""
    stable_text = f"{page.title.lower()} {page.keyword_blob}"
    return {
        kernel_type
        for kernel_type, terms in KERNEL_TYPE_ALIASES.items()
        if any(term in stable_text for term in terms)
    }


def resolve_operator(value: str) -> Optional[str]:
    value = value.lower().replace("_", "-")
    for operator, definition in OPERATORS.items():
        if value == operator or value in definition.aliases:
            return operator
    return None


def page_operators(page: "Page") -> set[str]:
    """Return specific operators present in a page's stable title/path text."""
    stable_text = f"{page.title.lower()} {page.keyword_blob}"
    return {
        operator
        for operator, definition in OPERATORS.items()
        if any(marker in stable_text for marker in definition.markers)
    }


def load_pages(docs_dir: Path) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(docs_dir.rglob("*.md")):
        if path.name in {"README.md", "index.md"}:
            continue
        rel = path.relative_to(docs_dir).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        masked = build_index.mask_fences(text.splitlines())
        title, title_index = build_index.extract_title(masked, path.stem)
        summary = build_index.extract_summary(masked, title_index)
        pages.append(
            Page(
                rel_path=rel,
                title=title,
                summary=summary,
                segments=path_segments(rel),
                filename=path.name.lower(),
                keyword_blob=keyword_blob(rel),
                body=text.lower(),
            )
        )
    return pages


def resolve_arch(value: str) -> Optional[str]:
    value = value.lower()
    if value in ARCH_ALIASES:
        return value
    return ARCH_INPUT_ALIASES.get(value)


def score_page(page: Page, terms: list[str], match_any: bool) -> int:
    title_l = page.title.lower()
    summary_l = page.summary.lower()
    matched_terms = 0
    score = 0
    for term in terms:
        if term in title_l:
            score += TITLE_WEIGHT
            matched_terms += 1
        elif term in summary_l or term in page.keyword_blob:
            score += SUMMARY_WEIGHT
            matched_terms += 1
        elif term in page.body:
            score += BODY_WEIGHT
            matched_terms += 1
    if matched_terms == 0:
        return 0
    if not match_any and matched_terms < len(terms):
        return 0  # AND semantics: every term must appear somewhere
    return score


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Architecture-scoped search over gpu-wiki docs.")
    parser.add_argument("query", nargs="*", help="Keywords to search for.")
    parser.add_argument("--root", default="gpu-wiki", help="Path to the gpu-wiki root.")
    parser.add_argument("--arch", action="append", default=[], help="Restrict to an architecture (repeatable).")
    parser.add_argument("--vendor", action="append", default=[], help="Restrict to nvidia / amd (repeatable).")
    parser.add_argument("--dsl", action="append", default=[], help="Restrict to cutedsl / flydsl / gluon / triton / cuda.")
    parser.add_argument("--section", action="append", default=[],
                        help="Keep docs whose path contains this role/path (repeatable; e.g. kernels, optimization).")
    parser.add_argument("--exclude-section", action="append", default=[],
                        help="Drop docs whose path contains this role/path (repeatable; e.g. articles).")
    parser.add_argument("--kernel-type", action="append", default=[],
                        help="Keep pages with a stable kernel type in title/path (e.g. gemm, attention, moe).")
    parser.add_argument("--operator", action="append", default=[],
                        help="Keep pages for a specific operator (e.g. gdn, flash-attention, paged-attention).")
    parser.add_argument("--symptom", action="append", default=[],
                        help="Select diagnosis cards from profiler SYMPTOMS (repeatable).")
    parser.add_argument("--any", dest="match_any", action="store_true", help="Match any keyword (default: all).")
    parser.add_argument("--limit", type=int, default=20, help="Maximum results to print.")
    parser.add_argument("--list-arch", action="store_true", help="List known architecture values and exit.")
    parser.add_argument("--list-operators", action="store_true", help="List known operator identifiers and exit.")
    args = parser.parse_args(argv)

    if args.list_arch:
        for value, tokens in ARCH_ALIASES.items():
            print(f"{value}: {', '.join(sorted(tokens))}")
        return 0
    if args.list_operators:
        for operator, definition in sorted(OPERATORS.items()):
            aliases = ", ".join(sorted(alias for alias in definition.aliases if alias != operator))
            print(f"{operator}: {aliases}" if aliases else operator)
        return 0

    docs_dir = Path(args.root) / "docs"
    if not docs_dir.is_dir():
        print(f"ERROR docs-not-found {docs_dir}:1 path does not exist", file=sys.stderr)
        return 1

    requested_arch: set[str] = set()
    for value in args.arch:
        resolved = resolve_arch(value)
        if resolved is None:
            print(f"ERROR unknown-arch {value}: try --list-arch", file=sys.stderr)
            return 1
        requested_arch.add(resolved)
    requested_vendor = {v.lower() for v in args.vendor}
    requested_dsl = {v.lower() for v in args.dsl}
    try:
        requested_sections = tuple(normalize_path_selector(value) for value in args.section)
        excluded_sections = tuple(normalize_path_selector(value) for value in args.exclude_section)
    except ValueError as exc:
        print(f"ERROR invalid-section {exc.args[0]}: use a relative docs path or path segment", file=sys.stderr)
        return 1
    requested_symptoms = {value.lower().replace("_", "-") for value in args.symptom}
    unknown_symptoms = requested_symptoms - SYMPTOMS
    if unknown_symptoms:
        print("ERROR unknown-symptom " + ", ".join(sorted(unknown_symptoms)) +
              ": valid values are " + ", ".join(sorted(SYMPTOMS)), file=sys.stderr)
        return 1
    requested_kernel_types: set[str] = set()
    for value in args.kernel_type:
        resolved = resolve_kernel_type(value)
        if resolved is None:
            print("ERROR unknown-kernel-type " + value + ": valid values are " +
                  ", ".join(sorted(KERNEL_TYPE_ALIASES)), file=sys.stderr)
            return 1
        requested_kernel_types.add(resolved)
    requested_operators: set[str] = set()
    for value in args.operator:
        resolved = resolve_operator(value)
        if resolved is None:
            print("ERROR unknown-operator " + value + ": try --list-operators", file=sys.stderr)
            return 1
        requested_operators.add(resolved)

    pages = load_pages(docs_dir)
    scoped = [
        page
        for page in pages
        if matches_dimension(page, ARCH_ALIASES, requested_arch)
        and matches_dimension(page, VENDOR_ALIASES, requested_vendor)
        and matches_dimension(page, DSL_ALIASES, requested_dsl)
        and matches_sections(page, requested_sections, excluded_sections)
        and (not requested_symptoms or bool(page_symptoms(page) & requested_symptoms))
        and (not requested_kernel_types or bool(page_kernel_types(page) & requested_kernel_types))
        and (not requested_operators or bool(page_operators(page) & requested_operators))
    ]

    filters = []
    if requested_arch:
        filters.append(f"arch={','.join(sorted(requested_arch))}")
    if requested_vendor:
        filters.append(f"vendor={','.join(sorted(requested_vendor))}")
    if requested_dsl:
        filters.append(f"dsl={','.join(sorted(requested_dsl))}")
    if requested_sections:
        filters.append("section=" + ",".join("/".join(value) for value in requested_sections))
    if excluded_sections:
        filters.append("exclude-section=" + ",".join("/".join(value) for value in excluded_sections))
    if requested_symptoms:
        filters.append(f"symptom={','.join(sorted(requested_symptoms))}")
    if requested_kernel_types:
        filters.append(f"kernel-type={','.join(sorted(requested_kernel_types))}")
    if requested_operators:
        filters.append(f"operator={','.join(sorted(requested_operators))}")
    scope_desc = "; ".join(filters) if filters else "no filter"
    print(f"scope: {scope_desc} — {len(scoped)}/{len(pages)} pages in scope")

    # Split every positional arg on whitespace so a quoted multi-word query
    # ("moe gemm") behaves the same as separate tokens (moe gemm) — each word is
    # an independent keyword (AND by default, OR with --any), not a literal phrase.
    terms = [word.lower() for arg in args.query for word in arg.split()]
    if not terms:
        print("(no keywords given; listing in-scope pages)")
        for page in scoped[: args.limit]:
            print(f"  docs/{page.rel_path} — {page.summary}")
        return 0

    ranked = sorted(
        ((score_page(page, terms, args.match_any), page) for page in scoped),
        key=lambda item: (-item[0], item[1].rel_path),
    )
    hits = [(score, page) for score, page in ranked if score > 0]
    print(f'{len(hits)} match(es) for "{" ".join(terms)}"')
    for score, page in hits[: args.limit]:
        print(f"  [{score}] docs/{page.rel_path} — {page.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
