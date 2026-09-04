#!/usr/bin/env python3
"""Regression tests for reviewer-driven methodological corrections."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from physics_simulators import dataset_specs
from reanalyze_reference_results import add_interval_scores
from run_bayesian_optimization_validation import (
    exact_wilcoxon_signed_rank,
    pairwise_summary,
)
from run_physics_validation import _candidate_bounds, select_candidate_set


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "physics_validation_results"


class RevisionInvariantTests(unittest.TestCase):
    def test_bo_inference_uses_six_independent_primary_datasets(self) -> None:
        runs = pd.read_csv(RESULTS / "bo_run_summary.csv")
        clustered, descriptive = pairwise_summary(runs[runs["dataset"] != "D5"])
        self.assertTrue((clustered["dataset_count"] == 6).all())
        self.assertTrue((clustered["independent_unit"] == "dataset").all())
        self.assertTrue((descriptive["pair_count"] == 30).all())
        sequential = clustered[
            (clustered["regime"] == "sequential")
            & (clustered["baseline"] == "RBF-GPR-UCB")
        ].iloc[0]
        self.assertAlmostEqual(float(sequential["wilcoxon_p"]), 0.375)

    def test_exact_signed_rank_discards_zero_differences(self) -> None:
        statistic, p_value = exact_wilcoxon_signed_rank(
            np.array([1.0, 2.0, 3.0, 0.0])
        )
        self.assertEqual(statistic, 0.0)
        self.assertEqual(p_value, 0.25)

    def test_d7_bounds_do_not_depend_on_test_inputs(self) -> None:
        spec = next(item for item in dataset_specs() if item.key == "D7")
        train = np.array([[0.0, 1.0], [2.0, 5.0]])
        test_a = np.array([[3.0, 6.0]])
        test_b = np.array([[3000.0, 6000.0]])
        np.testing.assert_allclose(
            _candidate_bounds(spec, train, test_a),
            _candidate_bounds(spec, train, test_b),
        )

    def test_out_of_bounds_baseline_candidates_are_rejected_not_clipped(self) -> None:
        frame = pd.DataFrame(
            {"x1": [10.0, 11.0], "x2": [10.0, 11.0], "log_score": [0.0, -1.0]}
        )
        candidates, projected, _ = select_candidate_set(
            frame, np.array([[0.0, 1.0], [0.0, 1.0]]), top_k=2
        )
        self.assertEqual(candidates.shape, (0, 2))
        self.assertFalse(projected)

    def test_interval_score_penalizes_missed_intervals(self) -> None:
        raw = pd.DataFrame(
            {
                "predicted_mean": [0.0, 0.0],
                "actual": [0.0, 10.0],
                "interval_width_90": [2.0, 2.0],
            }
        )
        scored = add_interval_scores(raw)
        self.assertLess(scored.loc[0, "interval_score_90"], scored.loc[1, "interval_score_90"])

    def test_archived_manifest_is_machine_readable_and_flags_d5(self) -> None:
        manifest = pd.read_csv(RESULTS / "dataset_manifest.csv")
        d5 = manifest[manifest["dataset"] == "D5"].iloc[0]
        self.assertFalse(bool(d5["primary_analysis"]))
        self.assertAlmostEqual(float(d5["train_success_rate"]), 104 / 125)
        self.assertAlmostEqual(float(d5["test_success_rate"]), 42 / 65)


if __name__ == "__main__":
    unittest.main()
