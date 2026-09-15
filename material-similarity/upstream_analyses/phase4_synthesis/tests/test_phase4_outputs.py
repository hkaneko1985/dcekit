import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class Phase4OutputTests(unittest.TestCase):
    def test_expected_datasets_and_counts(self):
        frame = pd.read_csv(RESULTS / "cross_dataset_diagnostics.csv").set_index("dataset")
        self.assertEqual(
            set(frame.index),
            {"Starrydata", "HTEM", "NanoMine", "PNCExtract"},
        )
        self.assertEqual(int(frame.loc["Starrydata", "samples"]), 5001)
        self.assertEqual(int(frame.loc["HTEM", "samples"]), 1891)
        self.assertEqual(int(frame.loc["NanoMine", "samples"]), 832)
        self.assertEqual(int(frame.loc["PNCExtract", "samples"]), 1103)

    def test_core_cross_domain_directions(self):
        frame = pd.read_csv(RESULTS / "cross_dataset_diagnostics.csv").set_index("dataset")
        for dataset in ["Starrydata", "HTEM", "NanoMine"]:
            self.assertGreater(frame.loc[dataset, "context_gain_median"], 0)
        self.assertLess(frame.loc["PNCExtract", "context_gain_median"], 0)

    def test_views_are_valid_and_nonidentical(self):
        frame = pd.read_csv(RESULTS / "view_disagreement_pairs.csv")
        self.assertTrue(frame["ari"].between(-1, 1).all())
        summary = pd.read_csv(RESULTS / "view_disagreement_summary.csv").set_index("dataset")
        self.assertTrue((summary["median"] < 0.9).all())

    def test_global_local_direction_reversal(self):
        frame = pd.read_csv(RESULTS / "objective_disagreement.csv")
        pnc = frame.loc[
            (frame["dataset"] == "PNCExtract")
            & (frame["view"] == "strict_loading")
        ].iloc[0]
        self.assertLess(pnc["global_gain"], 0)
        self.assertGreater(pnc["local_gain"], 0)
        htem = frame.loc[frame["dataset"] == "HTEM"]
        joint = (htem["improves_global"] & htem["improves_local"]).mean()
        self.assertAlmostEqual(joint, 0.046153846153846156)

    def test_decision_and_manifest(self):
        decision = json.loads(
            (RESULTS / "phase4_decision.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            decision["decision"],
            "GO_FOR_METHODS_PAPER_HOLD_FOR_PERFORMANCE_TRANSFER_CLAIMS",
        )
        self.assertTrue(all(decision["criteria"].values()))
        manifest = json.loads(
            (ROOT / "data" / "upstream_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["files"]), 17)
        for item in manifest["files"].values():
            self.assertEqual(len(item["sha256"]), 64)

    def test_report_has_no_control_characters(self):
        text = (ROOT / "report" / "PHASE4_REPORT_JA.md").read_text(encoding="utf-8")
        bad = [
            char
            for char in text
            if ord(char) < 32 and char not in {"\n", "\r", "\t"}
        ]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
