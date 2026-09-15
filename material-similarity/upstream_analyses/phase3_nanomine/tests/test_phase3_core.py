from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase3_core import (  # noqa: E402
    exact_set_distance,
    map_interval_distance,
    sequence_edit_distance,
    soft_set_distance,
    weighted_composite,
)


class DistanceTests(unittest.TestCase):
    def test_exact_and_soft_identity(self):
        self.assertEqual(exact_set_distance(["Silica"], ["Silica"]), 0.0)
        self.assertEqual(exact_set_distance(["Silica"], ["Alumina"]), 1.0)
        self.assertLess(soft_set_distance(["poly(lactic acid)"], ["polylactic acid"]), 0.5)

    def test_sequence_order_and_repetition(self):
        base = ["mixing", "heating", "cooling"]
        self.assertEqual(sequence_edit_distance(base, base), 0.0)
        self.assertGreater(sequence_edit_distance(base, ["mixing", "cooling", "heating"]), 0.0)
        self.assertGreater(sequence_edit_distance(base, ["mixing", "heating", "cooling", "heating"]), 0.0)

    def test_missing_is_interval_not_imputation(self):
        value = {"kind": "numeric", "value": 0.2, "scale_key": "fraction"}
        other = {"kind": "numeric", "value": 0.4, "scale_key": "fraction"}
        lower, reported, upper, coverage = map_interval_distance(
            {"mass": [value], "volume": [value]},
            {"mass": [other]},
            {"fraction": 1.0},
        )
        self.assertAlmostEqual(reported, 0.2)
        self.assertAlmostEqual(lower, 0.1)
        self.assertAlmostEqual(upper, 0.6)
        self.assertAlmostEqual(coverage, 0.5)

    def test_no_common_variable_is_undefined_reported(self):
        value = {"kind": "numeric", "value": 0.2, "scale_key": "fraction"}
        lower, reported, upper, coverage = map_interval_distance(
            {"mass": [value]}, {"volume": [value]}, {"fraction": 1.0}
        )
        self.assertEqual(lower, 0.0)
        self.assertTrue(math.isnan(reported))
        self.assertEqual(upper, 1.0)
        self.assertEqual(coverage, 0.0)

    def test_reported_composite_renormalizes_available_facets(self):
        facets = {
            "known": {"reported": np.array([0.2], dtype=np.float32)},
            "missing": {"reported": np.array([np.nan], dtype=np.float32)},
        }
        result = weighted_composite(facets, {"known": 0.2, "missing": 0.8}, "reported")
        self.assertAlmostEqual(float(result[0]), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()

