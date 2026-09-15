from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase1_core import (  # noqa: E402
    composition_distance_matrix,
    composition_matrices,
    mix_distances,
    parse_formula,
)
from run_phase1 import select_pilot  # noqa: E402


class FormulaParserTest(unittest.TestCase):
    def test_numeric_and_parenthetical_formula(self):
        parsed = parse_formula("Ca3(PO4)2")
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(sum(parsed.values()), 1.0)
        self.assertAlmostEqual(parsed["Ca"], 3 / 13)
        self.assertAlmostEqual(parsed["P"], 2 / 13)
        self.assertAlmostEqual(parsed["O"], 8 / 13)

    def test_hydrate(self):
        parsed = parse_formula("CuSO4·5H2O")
        self.assertIsNotNone(parsed)
        self.assertEqual(set(parsed), {"Cu", "S", "O", "H"})

    def test_variable_formula_rejected(self):
        self.assertIsNone(parse_formula("La1-xSrxMnO3"))


class DistanceTest(unittest.TestCase):
    def test_composition_distance_is_symmetric(self):
        formulas = [parse_formula(x) for x in ("NaCl", "KCl", "SiO2")]
        exact, groups, periods, binary = composition_matrices(formulas)
        distance = composition_distance_matrix(exact, groups, periods, binary)
        np.testing.assert_allclose(distance, distance.T)
        np.testing.assert_allclose(np.diag(distance), 0)
        self.assertLess(distance[0, 1], distance[0, 2])

    def test_missing_block_is_not_a_similarity_feature(self):
        composition = np.asarray([[0.0, 0.4], [0.4, 0.0]], dtype=np.float32)
        block = np.ones((2, 2), dtype=np.float32)
        available = np.zeros((2, 2), dtype=bool)
        mixed = mix_distances(composition, [(block, available)])
        np.testing.assert_allclose(mixed, composition)


class PilotOrderTest(unittest.TestCase):
    def test_selection_and_row_order_are_source_order_invariant(self):
        records = []
        for sid in ("paper-a", "paper-b", "paper-c"):
            for sample_id in ("1", "2", "3"):
                records.append({"SID": sid, "sample_id": sample_id})
        config = {
            "seed": 17,
            "cohort": {
                "minimum_samples_per_sid_before_sampling": 2,
                "maximum_samples_per_sid": 3,
                "pilot_target_samples": 9,
            },
        }
        forward, _ = select_pilot(records, config)
        reverse, _ = select_pilot(list(reversed(records)), config)
        forward_ids = [(row["SID"], row["sample_id"]) for row in forward]
        reverse_ids = [(row["SID"], row["sample_id"]) for row in reverse]
        self.assertEqual(forward_ids, reverse_ids)


if __name__ == "__main__":
    unittest.main()
