import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import query  # noqa: E402


# Every Blackwell operator page in kernels/ must have at least one stable
# --operator route.  Adding or removing a page requires updating this inventory,
# making coverage an explicit reviewable contract rather than best effort.
BLACKWELL_KERNEL_OPERATORS = {
    "deepgemm.md": {"gemm"},
    "distributed-gemm-allreduce.md": {"gemm", "allreduce"},
    "flash-attention-4.md": {"flash-attention"},
    "flashmla.md": {"mla"},
    "fp8-block-scale-gemm.md": {"gemm"},
    "fused-moe.md": {"moe"},
    "gated-delta-net.md": {"gdn"},
    "gated-dual-gemm.md": {"gated-dual-gemm", "gemm"},
    "grouped-gemm.md": {"grouped-gemm", "gemm"},
    "mla-decode.md": {"mla"},
    "nsa.md": {"nsa"},
    "nvfp4-gemm.md": {"gemm"},
    "nvfp4-gemv.md": {"gemv"},
    "sparse-mla.md": {"sparse-mla", "mla"},
}


def make_page(rel, title="", summary="", body=""):
    return query.Page(
        rel_path=rel,
        title=title,
        summary=summary,
        segments=query.path_segments(rel),
        filename=rel.rsplit("/", 1)[-1].lower(),
        keyword_blob=query.keyword_blob(rel),
        body=body.lower(),
    )


class DimensionTests(unittest.TestCase):
    def test_arch_from_path_segment(self):
        page = make_page("nvidia/hopper/cutedsl/flash_fwd.md")
        self.assertEqual({"hopper"}, query.dimension_values(page, query.ARCH_ALIASES))

    def test_arch_from_filename(self):
        page = make_page("amd/hardware-specs/hardware_specs_mi300x.md")
        self.assertEqual({"cdna3"}, query.dimension_values(page, query.ARCH_ALIASES))

    def test_blackwell_and_geforce_are_distinct(self):
        # the whole point: blackwell must NOT match blackwell-geforce (sm120)
        geforce = make_page("nvidia/blackwell-geforce/cutedsl/sm120-gemm.md")
        self.assertEqual({"blackwell-geforce"}, query.dimension_values(geforce, query.ARCH_ALIASES))
        self.assertFalse(query.matches_dimension(geforce, query.ARCH_ALIASES, {"blackwell"}))
        self.assertTrue(query.matches_dimension(geforce, query.ARCH_ALIASES, {"blackwell-geforce"}))

        datacenter = make_page("nvidia/blackwell/techniques/swizzling.md")
        self.assertEqual({"blackwell"}, query.dimension_values(datacenter, query.ARCH_ALIASES))
        self.assertFalse(query.matches_dimension(datacenter, query.ARCH_ALIASES, {"blackwell-geforce"}))

    def test_neutral_page_has_no_arch(self):
        page = make_page("nvidia/common/ptx/ptx-instruction-set.md")
        self.assertEqual(set(), query.dimension_values(page, query.ARCH_ALIASES))

    def test_blackwell_filter_excludes_hopper(self):
        hopper = make_page("nvidia/hopper/hands-on/warp-specialization.md")
        self.assertFalse(query.matches_dimension(hopper, query.ARCH_ALIASES, {"blackwell"}))

    def test_neutral_survives_any_arch_filter(self):
        neutral = make_page("generic/gpu-execution-model.md")
        self.assertTrue(query.matches_dimension(neutral, query.ARCH_ALIASES, {"blackwell"}))
        self.assertTrue(query.matches_dimension(neutral, query.ARCH_ALIASES, {"hopper"}))

    def test_generic_is_vendor_neutral(self):
        generic = make_page("generic/gpu-execution-model.md")
        self.assertTrue(query.matches_dimension(generic, query.VENDOR_ALIASES, {"nvidia"}))
        amd = make_page("amd/common/amd-mfma-matrix-cores.md")
        self.assertFalse(query.matches_dimension(amd, query.VENDOR_ALIASES, {"nvidia"}))

    def test_resolve_arch_aliases(self):
        self.assertEqual("hopper", query.resolve_arch("sm90"))
        self.assertEqual("cdna3", query.resolve_arch("gfx942"))
        self.assertEqual("blackwell-geforce", query.resolve_arch("sm120"))
        self.assertEqual("blackwell", query.resolve_arch("blackwell"))
        self.assertIsNone(query.resolve_arch("nonsense"))


class KnowledgeRoleTests(unittest.TestCase):
    def test_section_selector_matches_a_path_component(self):
        kernel = make_page("nvidia/blackwell/kernels/nvfp4-gemm.md")
        self.assertTrue(query.matches_sections(kernel, (("kernels",),), ()))
        self.assertFalse(query.matches_sections(kernel, (("optimization",),), ()))

    def test_section_selector_accepts_a_full_path_fragment(self):
        optimization = make_page("nvidia/blackwell/optimization/compute-bound.md")
        selector = query.normalize_path_selector("nvidia/blackwell/optimization")
        self.assertTrue(query.matches_sections(optimization, (selector,), ()))

    def test_exclude_section_wins_over_include(self):
        article = make_page("nvidia/blackwell/articles/gemm-analysis.md")
        self.assertFalse(query.matches_sections(article, (), (("articles",),)))

    def test_symptom_is_identified_by_diagnosis_card_filename(self):
        symptom = make_page("nvidia/blackwell/optimization/pipeline-stalls.md")
        ordinary = make_page("nvidia/blackwell/optimization/pipeline-stages.md")
        self.assertEqual({"pipeline-stalls"}, query.page_symptoms(symptom))
        self.assertEqual(set(), query.page_symptoms(ordinary))

    def test_kernel_type_uses_title_and_path_not_body(self):
        gemm = make_page("nvidia/blackwell/kernels/deepgemm.md", title="DeepGEMM")
        attention = make_page("nvidia/blackwell/kernels/flashmla.md", title="FlashMLA",
                              body="This attention kernel contains GEMM operations.")
        self.assertIn("gemm", query.page_kernel_types(gemm))
        self.assertNotIn("gemm", query.page_kernel_types(attention))
        self.assertIn("attention", query.page_kernel_types(attention))

    def test_kernel_type_aliases(self):
        self.assertEqual("gemm", query.resolve_kernel_type("matmul"))
        self.assertEqual("moe", query.resolve_kernel_type("mixture-of-experts"))
        self.assertIsNone(query.resolve_kernel_type("nonsense"))


class ScoringTests(unittest.TestCase):
    def test_title_match_scores_higher_than_body(self):
        page = make_page("a.md", title="Bank conflict swizzle", summary="s", body="body text")
        self.assertEqual(query.TITLE_WEIGHT, query.score_page(page, ["bank"], match_any=False))

    def test_and_semantics_requires_all_terms(self):
        page = make_page("a.md", title="Bank conflict", summary="s", body="nothing else")
        self.assertEqual(0, query.score_page(page, ["bank", "missingterm"], match_any=False))

    def test_any_semantics_allows_partial(self):
        page = make_page("a.md", title="Bank conflict", summary="s", body="nothing else")
        self.assertGreater(query.score_page(page, ["bank", "missingterm"], match_any=True), 0)


class EndToEndTests(unittest.TestCase):
    def make_wiki(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name) / "gpu-wiki"
        (root / "docs").mkdir(parents=True)
        self.addCleanup(temp.cleanup)
        return root

    def write(self, root, rel, text):
        path = root / "docs" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_scoped_search_excludes_other_arch(self):
        root = self.make_wiki()
        self.write(root, "nvidia/blackwell/techniques/x.md", "# X\n\nUses tcgen05 pipeline.\n")
        self.write(root, "nvidia/hopper/hands-on/y.md", "# Y\n\nUses wgmma pipeline.\n")
        self.write(root, "nvidia/blackwell-geforce/cutedsl/z.md", "# Z\n\nsm120 nvfp4 gemm.\n")
        pages = query.load_pages(root / "docs")
        scoped = {p.rel_path for p in pages if query.matches_dimension(p, query.ARCH_ALIASES, {"blackwell"})}
        self.assertIn("nvidia/blackwell/techniques/x.md", scoped)
        self.assertNotIn("nvidia/hopper/hands-on/y.md", scoped)
        self.assertNotIn("nvidia/blackwell-geforce/cutedsl/z.md", scoped)

    def test_quoted_multiword_query_splits_into_terms(self):
        # a single quoted arg "moe gemm" must behave as two AND-ed keywords, not a
        # literal-phrase substring search
        root = self.make_wiki()
        self.write(root, "nvidia/blackwell/kernels/fused-moe.md", "# Fused MoE Dual GEMM\n\nRouting plus dual GEMM.\n")
        self.assertEqual(0, query.main(["moe gemm", "--root", str(root), "--vendor", "nvidia"]))
        page = query.load_pages(root / "docs")[0]
        self.assertGreater(query.score_page(page, ["moe", "gemm"], match_any=False), 0)

    def test_section_and_symptom_filters_select_one_diagnosis_card(self):
        root = self.make_wiki()
        self.write(root, "nvidia/blackwell/optimization/pipeline-stalls.md",
                   "# Pipeline stalls\n\nTMA pipeline diagnosis.\n")
        self.write(root, "nvidia/blackwell/optimization/pipeline-stages.md",
                   "# Pipeline stages\n\nPipeline tuning.\n")
        self.assertEqual(0, query.main([
            "--root", str(root), "--arch", "b200", "--section", "optimization",
            "--symptom", "pipeline_stalls",
        ]))

    def test_kernel_type_filter_excludes_body_only_mentions(self):
        root = self.make_wiki()
        self.write(root, "nvidia/blackwell/kernels/deepgemm.md", "# DeepGEMM\n\nGEMM kernel.\n")
        self.write(root, "nvidia/blackwell/kernels/flashmla.md", "# FlashMLA\n\nAttention uses GEMM.\n")
        self.assertEqual(0, query.main([
            "gemm", "--root", str(root), "--arch", "b200", "--section", "kernels",
            "--kernel-type", "gemm",
        ]))

    def test_every_blackwell_kernel_page_has_a_registered_operator(self):
        docs = SCRIPTS_DIR.parent / "docs"
        pages = query.load_pages(docs)
        blackwell_kernels = {
            Path(page.rel_path).name: page
            for page in pages
            if Path(page.rel_path).parent.as_posix() == "nvidia/blackwell/kernels"
        }
        self.assertEqual(set(BLACKWELL_KERNEL_OPERATORS), set(blackwell_kernels))
        for filename, expected_operators in BLACKWELL_KERNEL_OPERATORS.items():
            with self.subTest(filename=filename):
                self.assertTrue(expected_operators & query.page_operators(blackwell_kernels[filename]))


if __name__ == "__main__":
    unittest.main()
