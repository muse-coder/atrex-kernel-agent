import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import classify_rocprof  # noqa: E402
import profile_summary  # noqa: E402


class ProfileSummaryTests(unittest.TestCase):
    def test_write_summary_creates_machine_and_human_contracts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = profile_summary.build_summary(
                platform="nvidia", classification_status="complete",
                evidence=["analysis/metrics.json"], symptoms=["pipeline-stalls"],
            )
            profile_summary.write_summary(root, summary, "NCU Profile Summary")
            payload = json.loads((root / "summary.json").read_text())
            self.assertEqual("complete", payload["classification_status"])
            self.assertIn("SYMPTOMS: pipeline-stalls", (root / "summary.txt").read_text())

    def test_amd_att_scratch_and_wait_are_classified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stats = root / "att" / "stats_run.csv"
            stats.parent.mkdir(parents=True)
            stats.write_text(
                "Instruction,Stall,Hitcount\n"
                "s_waitcnt lgkmcnt(0),12,4\n"
                "scratch_store_dword,3,2\n",
                encoding="utf-8",
            )
            summary = classify_rocprof.classify(root)
            self.assertEqual("complete", summary["classification_status"])
            self.assertIn("pipeline-stalls", summary["symptoms"])
            self.assertIn("register-pressure", summary["symptoms"])

    def test_amd_missing_artifacts_is_not_classified(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = classify_rocprof.classify(Path(temp))
            self.assertEqual("insufficient-evidence", summary["classification_status"])
            self.assertEqual([], summary["symptoms"])


if __name__ == "__main__":
    unittest.main()
