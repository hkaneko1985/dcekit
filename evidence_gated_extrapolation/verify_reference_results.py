#!/usr/bin/env python3
"""Fast integrity checks for the revised archived paper results.

This script reads archived outputs; it does not refit models.  Use the run_*
scripts for computational reproduction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "physics_validation_results"


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(f"Expected {expected}, obtained {actual}")


def main() -> None:
    required = [
        "forward_summary.csv",
        "strong_baseline_summary.csv",
        "forward_dataset_level_comparisons.csv",
        "forward_variation_summary.csv",
        "inverse_metrics_raw.csv",
        "inverse_metrics_fair_raw.csv",
        "inverse_decision_utility_overall.csv",
        "inverse_variation_summary.csv",
        "bo_pairwise_summary.csv",
        "bo_pairwise_repetition_descriptive.csv",
        "bo_dataset_summary.csv",
        "bo_uncertainty_summary.csv",
        "dataset_generation_audit.csv",
        "run_manifest.json",
    ]
    missing = [name for name in required if not (RESULTS / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing reference outputs: {missing}")

    forward = pd.read_csv(RESULTS / "forward_summary.csv")
    if set(forward["dataset"]) != {f"D{i}" for i in range(1, 8)}:
        raise AssertionError("Forward summary does not contain D1-D7")
    forward_comparison = pd.read_csv(
        RESULTS / "forward_dataset_level_comparisons.csv"
    )
    fixed_gpr = forward_comparison[
        (forward_comparison["proposed"] == "AD-ADE-GPR")
        & (forward_comparison["baseline"] == "GPR")
    ].iloc[0]
    fixed_svr = forward_comparison[
        (forward_comparison["proposed"] == "AD-ADE-SVR")
        & (forward_comparison["baseline"] == "SVR")
    ].iloc[0]
    if tuple(fixed_gpr[["win_count", "loss_count"]]) != (5, 1):
        raise AssertionError("Unexpected primary GPR-family result")
    if tuple(fixed_svr[["win_count", "loss_count"]]) != (6, 0):
        raise AssertionError("Unexpected primary SVR-family result")
    close(float(fixed_gpr["wilcoxon_p"]), 0.0625)
    close(float(fixed_svr["wilcoxon_p"]), 0.03125)

    inverse = pd.read_csv(RESULTS / "inverse_metrics_raw.csv")
    egc = inverse[inverse["method"].eq("EGC-GMR")]
    counts = egc["rejection_reason"].value_counts().to_dict()
    if len(egc) != 168 or counts != {
        "supported": 153,
        "no_feasible_candidate": 10,
        "unsupported_evidence": 5,
    }:
        raise AssertionError(f"Unexpected EGC-GMR decisions: {counts}")

    inverse_utility = pd.read_csv(
        RESULTS / "inverse_decision_utility_overall.csv"
    ).set_index("method")
    close(float(inverse_utility.loc["EGC-GMR", "verified_answer_rate"]), 0.875)
    close(
        float(inverse_utility.loc["EGC-GMR", "success_rate_nrmse_le_1_0"]),
        0.816667,
    )

    forward_shell = pd.read_csv(RESULTS / "forward_variation_summary.csv")
    deep = forward_shell[forward_shell["rho"].eq(32.0)]
    gpr = deep[deep["method"].eq("RBF-GPR")]
    ade_gpr = deep[deep["method"].eq("AD-ADE-GPR")]
    if len(gpr) != 7 or not np.allclose(gpr["directional_sd_median"], 0.0):
        raise AssertionError("RBF-GPR deep-shell collapse check failed")
    if len(ade_gpr) != 7 or not np.all(ade_gpr["directional_sd_median"] > 0):
        raise AssertionError("AD-ADE-GPR direction-dependence check failed")

    inverse_shell = pd.read_csv(RESULTS / "inverse_variation_summary.csv")
    egc_deep = inverse_shell[
        inverse_shell["method"].eq("EGC-GMR")
        & inverse_shell["rho"].isin([16.0, 32.0])
    ]
    if len(egc_deep) != 14 or not np.allclose(egc_deep["supported_rate"], 0.0):
        raise AssertionError("EGC-GMR deep-shell rejection check failed")

    paired = pd.read_csv(RESULTS / "bo_pairwise_summary.csv")
    sequential = paired[
        paired["regime"].eq("sequential")
        & paired["baseline"].eq("RBF-GPR-UCB")
    ].iloc[0]
    batch = paired[
        paired["regime"].eq("batch")
        & paired["baseline"].eq("RBF-GPR top-3")
    ].iloc[0]
    if tuple(
        sequential[["dataset_win_count", "dataset_tie_count", "dataset_loss_count"]]
    ) != (3, 2, 1):
        raise AssertionError("Unexpected sequential Bayesian-optimization counts")
    if tuple(
        batch[["dataset_win_count", "dataset_tie_count", "dataset_loss_count"]]
    ) != (3, 1, 2):
        raise AssertionError("Unexpected batch Bayesian-optimization counts")
    close(float(sequential["wilcoxon_p"]), 0.375)
    close(float(batch["wilcoxon_p"]), 0.4375)

    uncertainty = pd.read_csv(RESULTS / "bo_uncertainty_summary.csv")
    primary_uncertainty = uncertainty[uncertainty["dataset"] != "D5"]
    proposed_uncertainty = primary_uncertainty[
        primary_uncertainty["method"] == "EG-AD-ADE uncertainty"
    ]
    close(
        float(proposed_uncertainty["coverage_90_supported_median"].median()),
        0.953488,
    )
    close(
        float(proposed_uncertainty["interval_width_90_supported_median"].median()),
        5.140879,
    )

    audit = pd.read_csv(RESULTS / "dataset_generation_audit.csv").set_index("dataset")
    if int(audit.loc["D5", "training_failure_count"]) != 21:
        raise AssertionError("Unexpected D5 training failure count")
    if int(audit.loc["D5", "external_failure_count"]) != 23:
        raise AssertionError("Unexpected D5 external failure count")

    print("REVISED_REFERENCE_RESULTS_VERIFIED")
    print("Forward primary: AD-ADE vs fixed RBF = GPR 5/6; SVR 6/6")
    print("Inverse primary: EGC verified-answer rate 0.875; success@NRMSE<=1 = 0.817")
    print("BO dataset-level: sequential 3/2/1 (p=0.375); batch 3/1/2 (p=0.438)")
    print("D5 is exploratory: 21/125 training and 23/65 external simulations failed")


if __name__ == "__main__":
    main()
