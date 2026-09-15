#!/usr/bin/env python3
"""Create static scientific figures for the Phase 3 report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

VIEW_MAP = {
    "material_only": "Material only",
    "balanced": "Balanced",
    "sequence_focus": "Sequence focus",
    "settings_focus": "Settings focus",
    "numeric_settings_focus": "Numeric settings",
    "process_only": "Process only",
    "missing_mask_only": "Missing mask",
}


def save(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sentinel_grid() -> pd.DataFrame:
    grid = pd.read_csv(RESULTS / "phase3_grid.csv")
    specs = [
        ("material_only", "identity_strict", "equal", 0.0),
        ("balanced", "full_strict", "equal", 0.5),
        ("sequence_focus", "identity_soft", "sequence", 0.5),
        ("settings_focus", "identity_soft", "settings_all", 0.5),
        ("numeric_settings_focus", "identity_soft", "settings_numeric", 0.5),
        ("process_only", "identity_soft", "equal", 1.0),
    ]
    rows = []
    for name, material, process, weight in specs:
        subset = grid[
            (grid.material_profile == material)
            & (grid.process_profile == process)
            & np.isclose(grid.process_lambda, weight)
        ].copy()
        subset["view"] = name
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def make_summary() -> None:
    sent = sentinel_grid()
    retrieval = pd.read_csv(RESULTS / "sentinel_retrieval.csv")
    facet = pd.read_csv(RESULTS / "facet_audit.csv")
    missing = pd.read_csv(RESULTS / "missingness_control.csv")
    k50 = sent[(sent.bound == "reported") & (sent.requested_clusters == 50)].copy()
    missing50 = missing[missing.k == 50].copy()
    display = pd.concat([
        k50[["view", "same_paper_recall_lift", "paper_ari"]],
        missing50.rename(columns={"view": "view"})[["view", "same_paper_recall_lift", "paper_ari"]],
    ])
    order = list(VIEW_MAP)
    display["order"] = display.view.map({name: i for i, name in enumerate(order)})
    display = display.sort_values("order")
    labels = [VIEW_MAP[x] for x in display.view]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2", "#9C9C9C"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.barh(labels[::-1], display.same_paper_recall_lift.to_numpy()[::-1], color=colors[::-1])
    ax.set_xlabel("Same-paper co-cluster lift")
    ax.set_title("A. Weak-label enrichment at k=50")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[0, 1]
    ax.barh(labels[::-1], display.paper_ari.to_numpy()[::-1], color=colors[::-1])
    ax.set_xlabel("Adjusted Rand index vs paper ID")
    ax.set_title("B. Paper-label agreement at k=50")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 0]
    rep = retrieval[retrieval.bound == "reported"].copy()
    rep["order"] = rep.view.map({name: i for i, name in enumerate(order)})
    rep = rep.sort_values("order")
    ax.barh([VIEW_MAP[x] for x in rep.view][::-1], rep.mrr_midrank.to_numpy()[::-1], color=colors[:6][::-1])
    ax.axvline(float(missing50.mrr_midrank.iloc[0]), color="#666666", ls="--", lw=1.5, label="Missing-mask MRR")
    ax.set_xlabel("Tie-aware MRR for same-paper neighbor")
    ax.set_title("C. Local retrieval (reported distance)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 1]
    chosen = facet[facet.facet.isin([
        "matrix_strict", "filler_strict", "surface_strict", "loading", "descriptor",
        "process_family", "step_type", "sequence", "settings_all", "settings_numeric",
    ])].copy()
    label_map = {
        "matrix_strict": "Matrix", "filler_strict": "Filler", "surface_strict": "Surface",
        "loading": "Loading", "descriptor": "Descriptor", "process_family": "Method",
        "step_type": "Step type", "sequence": "Sequence", "settings_all": "All settings",
        "settings_numeric": "Numeric settings",
    }
    ax.barh([label_map[x] for x in chosen.facet][::-1], chosen.mean_interval_width.to_numpy()[::-1], color="#EDC948")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean optimistic–pessimistic width")
    ax.set_title("D. Uncertainty caused by unreported fields")
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("NanoMine Phase 3: multi-view similarity gives purpose-dependent structure", fontsize=15)
    save(fig, "phase3_summary")


def make_sensitivity() -> None:
    inventory = pd.read_csv(ROOT / "data" / "process_family_inventory.csv")
    stability = pd.read_csv(RESULTS / "bound_stability.csv")
    control = pd.read_csv(RESULTS / "process_shuffle_control_summary.csv")
    breadth = pd.read_csv(RESULTS / "pncextract_breadth_grid.csv")
    operations = inventory[inventory.record_type == "operation"].pivot_table(
        index="name", columns="process_family", values="prevalence", aggfunc="max", fill_value=0
    )
    operations = operations.loc[operations.max(axis=1).sort_values(ascending=False).index[:10]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(operations.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_yticks(range(len(operations)), [x.title() for x in operations.index])
    ax.set_xticks(range(len(operations.columns)), [x.replace(" Processing", "") for x in operations.columns], rotation=25, ha="right")
    ax.set_title("A. Reported operation prevalence by process family")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="Sample prevalence")

    ax = axes[0, 1]
    for k, color in ((20, "#4C78A8"), (50, "#E15759")):
        subset = stability[stability.requested_clusters == k].groupby("process_lambda")["ari_optimistic_pessimistic"].median()
        ax.plot(subset.index, subset.values, marker="o", label=f"k={k}", color=color)
    ax.set_xlabel("Process weight λ")
    ax.set_ylabel("Median ARI: optimistic vs pessimistic")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("B. Missingness-bound stability")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    subset = control[(control.k == 50) & (control.metric == "paper_ari")].copy()
    x = np.arange(len(subset))
    width = 0.34
    ax.bar(x - width / 2, subset.original, width, label="Original", color="#59A14F")
    ax.bar(x + width / 2, subset.shuffled_mean, width, yerr=subset.shuffled_sd, label="Shuffled", color="#BAB0AC")
    ax.set_xticks(x, ["Order", "Setting values"])
    ax.set_ylabel("Paper ARI at k=50")
    ax.set_title("C. Process controls preserving masks")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    b50 = breadth[(breadth.bound == "reported") & (breadth.requested_clusters == 50)].sort_values("same_paper_recall_lift")
    ax.barh(b50.profile.str.replace("_", " ").str.title(), b50.same_paper_recall_lift, color="#B07AA1")
    ax.set_xlabel("Same-paper co-cluster lift")
    ax.set_title("D. PNCExtract composition-only breadth check")
    ax.grid(axis="x", alpha=0.25)
    fig.suptitle("NanoMine Phase 3: process inventory and robustness checks", fontsize=15)
    save(fig, "phase3_sensitivity")


if __name__ == "__main__":
    make_summary()
    make_sensitivity()

