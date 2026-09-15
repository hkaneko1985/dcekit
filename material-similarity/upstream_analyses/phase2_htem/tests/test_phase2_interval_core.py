import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase2_interval_core import (  # noqa: E402
    FACETS,
    FIELD_SPECS,
    field_pair_matrices,
    material_distance,
    process_interval_distances,
    raw_slots,
    evaluate_labels,
)


class Phase2IntervalCoreTests(unittest.TestCase):
    def test_null_slots_are_retained_as_unreported(self):
        self.assertEqual(raw_slots('[1,null,0]'), [1, None, 0])

    def test_partial_numeric_vector_coverage(self):
        distance, coverage, _ = field_pair_matrices(
            ['[10,null,30]', '[20,40,null]'], "numeric_array", False
        )
        self.assertAlmostEqual(float(coverage[0, 1]), 2 / 3)
        self.assertGreater(float(distance[0, 1]), 0)

    def test_joint_missing_is_not_similarity_evidence(self):
        _, coverage, _ = field_pair_matrices([None, None], "numeric", False)
        self.assertEqual(float(coverage[0, 1]), 0.0)

    def test_interval_without_imputation(self):
        shape = (2, 2)
        distances = {field: np.zeros(shape, dtype=np.float32) for field in FIELD_SPECS}
        coverage = {field: np.zeros(shape, dtype=np.float32) for field in FIELD_SPECS}
        first = next(iter(FIELD_SPECS))
        distances[first][0, 1] = distances[first][1, 0] = 0.5
        coverage[first][0, 1] = coverage[first][1, 0] = 1.0
        process, q, _ = process_interval_distances(distances, coverage, [0.25] * len(FACETS))
        self.assertLessEqual(float(process["optimistic"][0, 1]), float(process["reported"][0, 1]))
        self.assertLessEqual(float(process["reported"][0, 1]), float(process["pessimistic"][0, 1]))
        self.assertGreater(float(process["pessimistic"][0, 1] - process["optimistic"][0, 1]), 0)
        self.assertGreater(float(q[0, 1]), 0)

    def test_material_distance_keeps_composition_floor(self):
        composition = np.asarray([[0, 0.3], [0.3, 0]], dtype=np.float32)
        process = np.asarray([[0, 0.8], [0.8, 0]], dtype=np.float32)
        result = material_distance(composition, process, 0.75)
        self.assertTrue(np.all(result >= composition - 1e-7))
        self.assertAlmostEqual(float(result[0, 1]), 0.3 + 0.7 * 0.75 * 0.8, places=6)

    def test_source_matching_removes_source_only_signal(self):
        labels = np.asarray([0, 0, 1, 1])
        positives = {
            "pair_i": np.asarray([0, 2]),
            "pair_j": np.asarray([1, 3]),
            "pair_study": np.asarray(["a", "b"], dtype=object),
            "pair_component": np.asarray(["a", "b"], dtype=object),
        }
        # Repeated unlabeled entries supply at least 50 source-stratum pairs.
        ui = np.asarray(([0, 2] * 60), dtype=np.int32)
        uj = np.asarray(([1, 3] * 60), dtype=np.int32)
        composition = np.zeros((4, 4), dtype=np.float32)
        result = evaluate_labels(
            labels, positives, ui, uj, composition, 1, {},
            np.asarray(["same:x", "same:y"], dtype=object),
            np.asarray((["same:x", "same:y"] * 60), dtype=object),
        )
        self.assertAlmostEqual(result["all_source_matched"]["excess"], 0.0)


if __name__ == "__main__":
    unittest.main()
