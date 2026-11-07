"""Source-fit diagnostic checks only: no torch import, training, or source assets."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/source_fit52.py"
SPEC = importlib.util.spec_from_file_location("source_fit_diagnostics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourceFitDiagnosticTests(unittest.TestCase):
    def zero_row_check(self):
        check = {"mask_equal": True, "max_latent_difference": 0.0,
                 "max_standardized_property_difference": 0.0}
        check.update(MODULE.compare_modeled_property(np.full(3, np.nan),
                                                     np.full(3, np.nan),
                                                     np.zeros(3, dtype=bool)))
        return check

    def test_zero_rows_are_not_reported_as_zero_property_error(self):
        check = self.zero_row_check()
        self.assertEqual(check["modeled_property_comparison_rows"], 0)
        self.assertEqual(check["modeled_property_comparison"], "not_applicable_no_compatible_rows")
        self.assertIsNone(check["max_modeled_property_difference"])
        self.assertTrue(MODULE.feature_parity_passed(check))

    def test_property_difference_uses_only_compatible_rows(self):
        check = MODULE.compare_modeled_property([2.0, np.nan, 5.0],
                                                 [1.5, np.nan, 5.0],
                                                 [True, False, True])
        self.assertEqual(check["modeled_property_comparison_rows"], 2)
        self.assertEqual(check["modeled_property_comparison"], "compared")
        self.assertEqual(check["max_modeled_property_difference"], 0.5)

    def test_exact_compatible_rows_pass(self):
        check = {"mask_equal": True, "max_latent_difference": 0.0,
                 "max_standardized_property_difference": 0.0}
        check.update(MODULE.compare_modeled_property([2.0, np.nan, 5.0],
                                                     [2.0, np.nan, 5.0],
                                                     [True, False, True]))
        self.assertTrue(MODULE.feature_parity_passed(check))

    def test_mask_mismatch_fails_without_compatible_rows(self):
        check = self.zero_row_check()
        check["mask_equal"] = False
        self.assertFalse(MODULE.feature_parity_passed(check))

    def test_latent_mismatch_fails_without_compatible_rows(self):
        check = self.zero_row_check()
        check["max_latent_difference"] = 1e-3
        self.assertFalse(MODULE.feature_parity_passed(check))

    def test_missing_latent_difference_is_rejected(self):
        check = self.zero_row_check()
        check["max_latent_difference"] = None
        self.assertFalse(MODULE.feature_parity_passed(check))


if __name__ == "__main__":
    unittest.main()
