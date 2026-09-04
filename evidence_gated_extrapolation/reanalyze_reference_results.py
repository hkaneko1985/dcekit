#!/usr/bin/env python3
"""Reviewer-driven reanalysis of the archived reference results.

This script does not regenerate physical simulations.  It (i) changes the
independent unit for BO inference from repetition to dataset, (ii) adds proper
interval-score and sharpness summaries, (iii) applies the same no-clipping
feasibility rule to all inverse methods, (iv) treats D5 as exploratory, and
(v) combines the original methods with training-only-tuned strong baselines.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physics_simulators import dataset_specs
from run_bayesian_optimization_validation import (
    exact_wilcoxon_signed_rank,
    make_uncertainty_batch_figure,
    pairwise_summary,
    summarize_uncertainty,
)
from run_physics_validation import (
    dataset_manifest,
    summarize_inverse,
    summarize_inverse_decision_utility,
    summarize_inverse_matched,
)


RESULTS = Path(__file__).resolve().parent / "physics_validation_results"
PRIMARY_DATASETS = ("D1", "D2", "D3", "D4", "D6", "D7")
PRIMARY_INVERSE_DATASETS = ("D1", "D2", "D3", "D4", "D6")


def add_interval_scores(raw: pd.DataFrame) -> pd.DataFrame:
    revised = raw.copy()
    half_width = revised["interval_width_90"] / 2.0
    lower = revised["predicted_mean"] - half_width
    upper = revised["predicted_mean"] + half_width
    actual = revised["actual"]
    alpha = 0.10
    revised["interval_lower_90"] = lower
    revised["interval_upper_90"] = upper
    revised["interval_score_90"] = (
        upper
        - lower
        + (2.0 / alpha) * (lower - actual) * (actual < lower)
        + (2.0 / alpha) * (actual - upper) * (actual > upper)
    )
    return revised


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def forward_reanalysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    original = pd.read_csv(RESULTS / "forward_summary.csv")
    strong = pd.read_csv(RESULTS / "strong_baseline_summary.csv")
    original_combined = original[
        [
            "dataset",
            "dataset_label",
            "model",
            "test_sd_nrmse_median",
            "test_range_nrmse_median",
            "nrmse_median",
        ]
    ].rename(columns={"nrmse_median": "train_sd_nrmse_median"})
    original_combined["primary_analysis"] = original_combined["dataset"] != "D5"
    strong_combined = strong[
        [
            "dataset",
            "dataset_label",
            "model",
            "test_sd_nrmse_median",
            "test_range_nrmse_median",
            "train_sd_nrmse_median",
            "primary_analysis",
        ]
    ]
    combined = pd.concat([original_combined, strong_combined], ignore_index=True)
    combined.to_csv(RESULTS / "forward_all_model_summary.csv", index=False)

    lookup = combined.set_index(["dataset", "model"])["test_sd_nrmse_median"]
    comparisons = [
        ("AD-ADE-GPR", "GPR", "fixed RBF control"),
        ("AD-ADE-SVR", "SVR", "fixed RBF control"),
        ("AD-ADE-GPR", "Tuned-RBF-GPR", "training-shell-tuned RBF"),
        ("AD-ADE-SVR", "Tuned-RBF-SVR", "training-shell-tuned RBF"),
        ("AD-ADE-GPR", "Tuned-Ridge", "global linear extrapolation"),
        ("AD-ADE-SVR", "Tuned-Ridge", "global linear extrapolation"),
    ]
    rows: list[dict[str, object]] = []
    for proposed, baseline, baseline_role in comparisons:
        difference = pd.Series(
            {
                dataset: float(
                    lookup.loc[(dataset, baseline)] - lookup.loc[(dataset, proposed)]
                )
                for dataset in PRIMARY_DATASETS
            }
        )
        statistic, p_value = exact_wilcoxon_signed_rank(
            difference.to_numpy(dtype=float)
        )
        rows.append(
            {
                "proposed": proposed,
                "baseline": baseline,
                "baseline_role": baseline_role,
                "independent_unit": "dataset",
                "dataset_count": len(difference),
                "win_count": int(np.sum(difference > 1e-12)),
                "tie_count": int(np.sum(np.abs(difference) <= 1e-12)),
                "loss_count": int(np.sum(difference < -1e-12)),
                "median_test_sd_nrmse_reduction": float(np.median(difference)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["holm_p"] = holm_adjust(comparison["wilcoxon_p"].tolist())
    comparison.to_csv(RESULTS / "forward_dataset_level_comparisons.csv", index=False)
    make_forward_figure(combined, RESULTS / "fig_forward_strong_baselines.png")
    return combined, comparison


def make_forward_figure(summary: pd.DataFrame, output: Path) -> None:
    datasets = list(PRIMARY_DATASETS)
    positions = np.arange(len(datasets), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.6), sharex=True, constrained_layout=True)
    panels = [
        (
            axes[0],
            ["GPR", "Tuned-RBF-GPR", "AD-ADE-GPR", "Tuned-Ridge"],
            ["Fixed RBF-GPR", "Tuned RBF-GPR", "AD-ADE-GPR", "Tuned Ridge"],
            ["#98A2B3", "#0E7C86", "#0B5FA5", "#D97706"],
            "(a) GPR-family comparison",
        ),
        (
            axes[1],
            ["SVR", "Tuned-RBF-SVR", "AD-ADE-SVR", "Tuned-Ridge"],
            ["Fixed RBF-SVR", "Tuned RBF-SVR", "AD-ADE-SVR", "Tuned Ridge"],
            ["#98A2B3", "#0E7C86", "#0B5FA5", "#D97706"],
            "(b) SVR-family comparison",
        ),
    ]
    width = 0.19
    for axis, methods, labels, colors, title in panels:
        for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
            values = (
                summary[summary["model"] == method]
                .set_index("dataset")
                .reindex(datasets)["test_sd_nrmse_median"]
                .to_numpy(dtype=float)
            )
            axis.bar(
                positions + (index - 1.5) * width,
                values,
                width,
                label=label,
                color=color,
            )
        axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_ylabel("External-test-SD NRMSE")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#D0D5DD", alpha=0.55, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=4, fontsize=7, loc="upper center")
    axes[1].set_xticks(positions, datasets)
    axes[1].set_xlabel("D5 is excluded from the primary comparison because of non-uniform simulator failures")
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def inverse_reanalysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RESULTS / "inverse_metrics_raw.csv")
    fair = raw.copy()
    unfair_projection = fair["method"].isin(["Absolute-GMR", "Transition-GMR"]) & (
        fair["projected_to_bounds"] > 0.5
    )
    metric_columns = [
        "response_nrmse",
        "topk_response_nrmse",
        "top5_response_nrmse",
        "input_standardized_error",
        "topk_input_standardized_error",
        "top5_input_standardized_error",
    ]
    candidate_columns = [column for column in fair.columns if column.startswith("candidate_")]
    fair.loc[unfair_projection, metric_columns + candidate_columns] = np.nan
    fair.loc[unfair_projection, "candidate_count"] = 0
    fair.loc[unfair_projection, "supported_target"] = 0.0
    fair.loc[unfair_projection, "simulator_verified"] = 0.0
    fair.loc[unfair_projection, "rejection_reason"] = "no_feasible_candidate"
    fair.loc[unfair_projection, "no_feasible_candidate"] = 1.0
    fair["posthoc_common_no_clipping_rule"] = True
    fair.to_csv(RESULTS / "inverse_metrics_fair_raw.csv", index=False)

    summary = summarize_inverse(fair)
    matched = summarize_inverse_matched(fair)
    by_dataset, overall = summarize_inverse_decision_utility(fair)
    summary.to_csv(RESULTS / "inverse_summary_fair.csv", index=False)
    matched.to_csv(RESULTS / "inverse_matched_summary_fair.csv", index=False)
    by_dataset.to_csv(RESULTS / "inverse_decision_utility_by_dataset.csv", index=False)
    overall.to_csv(RESULTS / "inverse_decision_utility_overall.csv", index=False)
    return by_dataset, overall


def bo_reanalysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_summary = pd.read_csv(RESULTS / "bo_run_summary.csv")
    clustered, repetition = pairwise_summary(
        run_summary[run_summary["dataset"] != "D5"]
    )
    clustered_all, _ = pairwise_summary(run_summary)
    clustered.to_csv(RESULTS / "bo_pairwise_summary.csv", index=False)
    clustered_all.to_csv(
        RESULTS / "bo_pairwise_all_datasets_exploratory.csv", index=False
    )
    repetition.to_csv(
        RESULTS / "bo_pairwise_repetition_descriptive.csv", index=False
    )

    uncertainty_raw = add_interval_scores(
        pd.read_csv(RESULTS / "bo_uncertainty_raw.csv")
    )
    uncertainty_raw.to_csv(
        RESULTS / "bo_uncertainty_scored_raw.csv", index=False
    )
    uncertainty_summary = summarize_uncertainty(uncertainty_raw)
    uncertainty_summary.to_csv(RESULTS / "bo_uncertainty_summary.csv", index=False)
    dataset_summary = pd.read_csv(RESULTS / "bo_dataset_summary.csv")
    make_uncertainty_batch_figure(
        uncertainty_summary,
        dataset_summary,
        RESULTS / "fig_bo_uncertainty_batch.png",
    )

    statuses = pd.read_csv(RESULTS / "bo_run_status.csv")
    primary_runs = run_summary[run_summary["dataset"] != "D5"].copy()
    primary_statuses = statuses[statuses["dataset"] != "D5"].copy()
    tradeoff = primary_runs.groupby(["regime", "method"], as_index=False).agg(
        final_quality_median=("final_quality", "median"),
        final_quality_q25=("final_quality", lambda s: s.quantile(0.25)),
        final_quality_q75=("final_quality", lambda s: s.quantile(0.75)),
    )
    stop = primary_statuses.groupby(["regime", "method"], as_index=False).agg(
        unsupported_stop_rate=(
            "stop_reason",
            lambda s: float(np.mean(s != "budget_reached")),
        )
    )
    tradeoff = tradeoff.merge(stop, on=["regime", "method"], how="left")
    tradeoff["analysis_set"] = "primary six datasets; D5 excluded"
    tradeoff.to_csv(RESULTS / "bo_risk_performance_tradeoff.csv", index=False)

    exploratory_tradeoff = run_summary.groupby(
        ["regime", "method"], as_index=False
    ).agg(
        final_quality_median=("final_quality", "median"),
        final_quality_q25=("final_quality", lambda s: s.quantile(0.25)),
        final_quality_q75=("final_quality", lambda s: s.quantile(0.75)),
    )
    exploratory_stop = statuses.groupby(["regime", "method"], as_index=False).agg(
        unsupported_stop_rate=(
            "stop_reason",
            lambda s: float(np.mean(s != "budget_reached")),
        )
    )
    exploratory_tradeoff = exploratory_tradeoff.merge(
        exploratory_stop, on=["regime", "method"], how="left"
    )
    exploratory_tradeoff["analysis_set"] = "all seven datasets; exploratory"
    exploratory_tradeoff.to_csv(
        RESULTS / "bo_risk_performance_tradeoff_all_datasets_exploratory.csv",
        index=False,
    )
    return clustered, uncertainty_summary


def repair_manifest_and_audit() -> pd.DataFrame:
    frames = {
        spec.key: pd.read_csv(RESULTS / "datasets" / f"{spec.key}_data.csv")
        for spec in dataset_specs()
    }
    manifest = dataset_manifest(dataset_specs(), frames)
    manifest.to_csv(RESULTS / "dataset_manifest.csv", index=False)
    audit = manifest[
        [
            "dataset",
            "n_train_requested",
            "n_train",
            "train_success_rate",
            "n_test_requested",
            "n_test",
            "test_success_rate",
            "primary_analysis",
        ]
    ].copy()
    audit["training_failure_count"] = audit["n_train_requested"] - audit["n_train"]
    audit["external_failure_count"] = audit["n_test_requested"] - audit["n_test"]
    audit.to_csv(RESULTS / "dataset_generation_audit.csv", index=False)
    return audit


def main() -> None:
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    RESULTS = args.results_dir.resolve()
    audit = repair_manifest_and_audit()
    forward, forward_comparison = forward_reanalysis()
    inverse_by_dataset, inverse_overall = inverse_reanalysis()
    bo_clustered, uncertainty = bo_reanalysis()

    d4 = pd.read_csv(RESULTS / "datasets" / "D4_data.csv")
    purity = "desorption_co2_purity"
    train = d4[d4["split"] == "train"][purity]
    test = d4[d4["split"] == "test"][purity]
    key_results = {
        "primary_forward_datasets": list(PRIMARY_DATASETS),
        "exploratory_forward_dataset": "D5",
        "primary_inverse_datasets": list(PRIMARY_INVERSE_DATASETS),
        "D5_training_success_rate": float(
            audit.loc[audit["dataset"] == "D5", "train_success_rate"].iloc[0]
        ),
        "D5_external_success_rate": float(
            audit.loc[audit["dataset"] == "D5", "test_success_rate"].iloc[0]
        ),
        "D4_purity_train_sd": float(train.std(ddof=1)),
        "D4_purity_test_sd": float(test.std(ddof=1)),
        "forward_dataset_level_comparisons": forward_comparison.to_dict("records"),
        "inverse_all_target_overall": inverse_overall.to_dict("records"),
        "bo_dataset_clustered_comparisons": bo_clustered.to_dict("records"),
        "bo_uncertainty_cross_dataset_medians": uncertainty[
            uncertainty["dataset"] != "D5"
        ].groupby("method")[
            [
                "coverage_90_supported_median",
                "interval_width_90_supported_median",
                "interval_score_90_supported_median",
                "rmse_supported_median",
                "support_rate",
            ]
        ]
        .median()
        .reset_index()
        .to_dict("records"),
    }
    (RESULTS / "revision_key_results.json").write_text(
        json.dumps(key_results, indent=2), encoding="utf-8"
    )
    manifest_path = RESULTS / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest["reviewer_driven_reanalysis"] = {
        "date": "2026-09-02",
        "primary_forward_and_bo_datasets": list(PRIMARY_DATASETS),
        "primary_inverse_datasets": list(PRIMARY_INVERSE_DATASETS),
        "D5_status": "exploratory because simulator-success filtering was non-uniform",
        "forward_primary_metric": "mean response-wise external-test-SD NRMSE",
        "strong_baselines": [
            "training-shell-tuned RBF-GPR",
            "training-shell-tuned RBF-SVR",
            "training-shell-tuned ridge",
        ],
        "bo_inference": "Wilcoxon test on one median paired difference per dataset; repetitions are descriptive",
        "inverse_feasibility": "common discard-without-clipping rule",
        "inverse_all_target_penalty": 2.0,
        "uncertainty": "coverage, interval width, RMSE, and proper interval score",
        "outputs": [
            "forward_all_model_summary.csv",
            "forward_dataset_level_comparisons.csv",
            "inverse_metrics_fair_raw.csv",
            "inverse_decision_utility_by_dataset.csv",
            "inverse_decision_utility_overall.csv",
            "bo_pairwise_summary.csv",
            "bo_pairwise_all_datasets_exploratory.csv",
            "bo_pairwise_repetition_descriptive.csv",
            "bo_uncertainty_scored_raw.csv",
            "dataset_generation_audit.csv",
            "revision_key_results.json",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("REANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
