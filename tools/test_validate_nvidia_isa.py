import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import validate_nvidia_isa  # noqa: E402


class NvidiaIsaValidationTests(unittest.TestCase):
    def test_expected_sm100_lowering_passes(self):
        sass = """
/*0000*/ TCGEN05.MMA;
/*0010*/ UTMALDG.128;
/*0020*/ LDG.E.128;
"""
        result = validate_nvidia_isa.evaluate_sass(
            sass, None, "sm100", ["tensor-core", "async-copy", "no-spills", "vectorized-global-load"])
        self.assertTrue(result["passed"])

    def test_spill_fails_no_spills_expectation(self):
        result = validate_nvidia_isa.evaluate_sass("/*0000*/ STL.64;", None, "sm100", ["no-spills"])
        self.assertFalse(result["passed"])
        self.assertEqual(["no-spills"], result["failed_expectations"])


if __name__ == "__main__":
    unittest.main()
