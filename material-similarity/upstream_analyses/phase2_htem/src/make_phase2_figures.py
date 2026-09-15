#!/usr/bin/env python3
"""Create compact scientific summary figures for the HTEM Phase 2 run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"


def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_grid() -> list[dict[str, str]]:
    with (RESULTS / "phase2_grid.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    result = load_json("phase2_results.json")
    posthoc = load_json("phase2_posthoc_sensitivity.json")
    grid = load_grid()

    selected = result["descriptively_ranked_configurations"]
    diagnostics = {row["descriptive_rank"]: row for row in result["diagnostics"]}
    sensitivity = {row["descriptive_rank"]: row for row in posthoc["selected_configuration_sensitivity"]}

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 140,
        "savefig.dpi": 220,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)

    # A. Selected configurations: source-adjusted improvement and robustness.
    ax = axes[0, 0]
    ranks = np.arange(1, len(selected) + 1)
    observed = np.asarray([row["source_matched_delta_vs_composition_same_k"] for row in selected])
    ci = np.asarray([
        sensitivity[index]["all_libraries_source_matched_component_bootstrap"]["ci95"]
        for index in ranks
    ])
    shuffle95 = np.asarray([
        diagnostics[index]["process_shuffle_within_element_system_and_instrument"]
        ["source_matched_delta"]["percentile_95"]
        for index in ranks
    ])
    yerr = np.vstack([observed - ci[:, 0], ci[:, 1] - observed])
    colors = ["#1167b1" if index in result["context_supported_diagnostic_ranks"] else "#999999" for index in ranks]
    ax.bar(ranks, observed, color=colors, alpha=0.86, label="Observed delta")
    ax.errorbar(ranks, observed, yerr=yerr, fmt="none", ecolor="#222222", capsize=3, lw=1, label="Component bootstrap 95% CI")
    ax.scatter(ranks, shuffle95, marker="^", s=42, color="#d95f02", zorder=3, label="Source-preserving shuffle 95th pct.")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(ranks, [f"R{index}" for index in ranks])
    ax.set_ylabel("Source-adjusted delta vs composition")
    ax.set_title("A. Descriptively selected configurations")
    ax.legend(frameon=False, loc="upper right")

    # B. Global-cluster support versus local-retrieval support.
    ax = axes[0, 1]
    palette = {"reported": "#1b9e77", "optimistic": "#7570b3", "pessimistic": "#d95f02"}
    for variant in ["reported", "optimistic", "pessimistic"]:
        rows = [row for row in grid if row["profile"] != "COMPOSITION_ONLY" and row["variant"] == variant]
        x = np.asarray([number(row, "source_matched_delta_vs_composition_same_k") for row in rows])
        y = np.asarray([number(row, "mrr_delta_vs_composition") for row in rows])
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=10, alpha=0.34, color=palette[variant], edgecolors="none", label=variant)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xlabel("Source-adjusted cluster delta")
    ax.set_ylabel("MRR delta vs composition")
    ax.set_title("B. Global clustering and local retrieval disagree")
    ax.legend(frameon=False)

    # C. Prespecified lambda/resolution grid for the strongest descriptive profile.
    ax = axes[1, 0]
    rows = [row for row in grid if row["profile"] == "SOURCE_ONLY" and row["variant"] == "reported"]
    lambdas = sorted({number(row, "process_lambda") for row in rows})
    clusters = sorted({int(float(row["clusters"])) for row in rows})
    matrix = np.full((len(clusters), len(lambdas)), np.nan)
    for row in rows:
        i = clusters.index(int(float(row["clusters"])))
        j = lambdas.index(number(row, "process_lambda"))
        matrix[i, j] = number(row, "source_matched_delta_vs_composition_same_k")
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.20, vmax=0.20, origin="lower")
    ax.set_xticks(range(len(lambdas)), [f"{value:g}" for value in lambdas])
    ax.set_yticks(range(len(clusters)), [str(value) for value in clusters])
    ax.set_xlabel("Process weight lambda")
    ax.set_ylabel("Number of clusters")
    ax.set_title("C. SOURCE_ONLY / reported: full resolution slice")
    fig.colorbar(image, ax=ax, shrink=0.82, label="Source-adjusted delta")

    # D. Missing-information bound dependence.
    ax = axes[1, 1]
    width = 0.34
    optimistic = np.asarray([diagnostics[index]["ari_against_uncertainty_variants"]["optimistic"] for index in ranks])
    pessimistic = np.asarray([diagnostics[index]["ari_against_uncertainty_variants"]["pessimistic"] for index in ranks])
    ax.bar(ranks - width / 2, optimistic, width, color="#7570b3", label="reported vs optimistic")
    ax.bar(ranks + width / 2, pessimistic, width, color="#e6ab02", label="reported vs pessimistic")
    ax.set_xticks(ranks, [f"R{index}" for index in ranks])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Adjusted Rand index")
    ax.set_title("D. Dependence on missing-setting bounds")
    ax.legend(frameon=False)

    fig.suptitle("HTEM Phase 2 — composition/process interval similarity", fontsize=13, fontweight="bold")
    for suffix in ["png", "svg"]:
        fig.savefig(FIGURES / f"phase2_summary.{suffix}", bbox_inches="tight")
    plt.close(fig)

    # Compact hypothesis figure for the strongest cross-study/cross-instrument pair.
    pairs = load_json("interesting_pairs.json")["cross_study_cross_instrument_hypotheses"]
    pair = pairs[0]
    fields = pair["shared_process_fields"]
    labels = [row["field"].replace("deposition_", "").replace("_", " ") for row in fields]
    similarities = [row["conditional_similarity"] for row in fields]
    coverage = [row["slot_coverage"] for row in fields]
    order = np.arange(len(fields))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.3), constrained_layout=True)
    ax.barh(order, similarities, color="#2b8cbe", alpha=0.86, label="Conditional similarity")
    ax.scatter(coverage, order, color="#e34a33", marker="D", s=32, label="Pairwise reporting coverage")
    ax.set_yticks(order, labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Similarity / coverage")
    ax.set_title(
        f"Cross-study, cross-instrument hypothesis: library {pair['sample_library_id_1']} vs {pair['sample_library_id_2']}\n"
        f"{pair['element_system_1']}; consensus={pair['coassignment_fraction_across_grid_at_k150']:.1%}, "
        f"composition distance={pair['composition_distance']:.3f}"
    )
    ax.legend(frameon=False, loc="lower right")
    for suffix in ["png", "svg"]:
        fig.savefig(FIGURES / f"phase2_cross_instrument_pair.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
