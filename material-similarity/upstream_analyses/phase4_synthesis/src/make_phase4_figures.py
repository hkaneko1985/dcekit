#!/usr/bin/env python3
"""Create publication-style figures for the Phase 4 synthesis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def overview() -> None:
    diag = pd.read_csv(RESULTS / "cross_dataset_diagnostics.csv")
    aris = pd.read_csv(RESULTS / "view_disagreement_pairs.csv")
    objectives = pd.read_csv(RESULTS / "objective_disagreement.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # A: direction consistency across the richer-metadata grids.
    context = diag.loc[
        diag["dataset"].isin(["Starrydata", "HTEM", "NanoMine"]),
        ["dataset", "context_gain_positive_fraction"],
    ].copy()
    context["percent"] = 100 * context["context_gain_positive_fraction"]
    sns.barplot(
        data=context,
        x="dataset",
        y="percent",
        hue="dataset",
        legend=False,
        palette=["#4C78A8", "#F58518", "#54A24B"],
        ax=axes[0, 0],
    )
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Views with positive context gain (%)")
    axes[0, 0].set_title("A  Direction across tested views")
    for patch, value in zip(axes[0, 0].patches, context["percent"]):
        axes[0, 0].text(
            patch.get_x() + patch.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # B: major views do not collapse to one partition.
    order = ["Starrydata", "HTEM", "NanoMine"]
    sns.boxplot(
        data=aris,
        x="dataset",
        y="ari",
        order=order,
        hue="dataset",
        legend=False,
        palette=["#4C78A8", "#F58518", "#54A24B"],
        width=0.55,
        ax=axes[0, 1],
    )
    sns.stripplot(
        data=aris,
        x="dataset",
        y="ari",
        order=order,
        color="#222222",
        size=5,
        jitter=0.12,
        alpha=0.7,
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(0.9, ls="--", lw=1.2, color="#777777")
    axes[0, 1].set_ylim(0, 1.03)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Pairwise adjusted Rand index")
    axes[0, 1].set_title("B  Disagreement among similarity views")

    # C: HTEM's global and local criteria point in different directions.
    htem = objectives.loc[objectives["dataset"] == "HTEM"]
    axes[1, 0].scatter(
        htem["global_gain"],
        htem["local_gain"],
        s=24,
        alpha=0.45,
        color="#F58518",
        edgecolors="none",
    )
    axes[1, 0].axhline(0, color="#333333", lw=1)
    axes[1, 0].axvline(0, color="#333333", lw=1)
    axes[1, 0].set_xlabel("Global context gain (source-adjusted excess)")
    axes[1, 0].set_ylabel("Local MRR gain")
    axes[1, 0].set_title("C  HTEM: global gain rarely improves retrieval")
    axes[1, 0].text(
        0.98,
        0.06,
        "Both positive: 4.6%",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#BBBBBB"},
    )

    # D: same metrics make NanoMine and PNCExtract task rankings directly visible.
    labels = {
        "material_only": "Material",
        "balanced": "Balanced",
        "sequence_focus": "Sequence",
        "settings_focus": "Settings",
        "numeric_settings_focus": "Numeric",
        "process_only": "Process",
        "strict_equal": "Identity",
        "strict_loading": "+ loading",
        "soft_loading": "Soft + loading",
    }
    offsets = {
        "balanced": (5, 6),
        "sequence_focus": (5, -13),
        "settings_focus": (5, 8),
        "numeric_settings_focus": (5, -2),
        "strict_loading": (6, -1),
        "soft_loading": (6, 8),
    }
    styles = {
        "NanoMine": ("#54A24B", "o"),
        "PNCExtract": ("#B279A2", "s"),
    }
    for dataset in ["NanoMine", "PNCExtract"]:
        subset = objectives.loc[objectives["dataset"] == dataset]
        color, marker = styles[dataset]
        axes[1, 1].scatter(
            subset["global_value"],
            subset["local_value"],
            s=75,
            label=dataset,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
        )
        for row in subset.itertuples():
            axes[1, 1].annotate(
                labels.get(row.view, row.view),
                (row.global_value, row.local_value),
                xytext=offsets.get(row.view, (5, 4)),
                textcoords="offset points",
                fontsize=8.5,
            )
    axes[1, 1].set_xlabel("Paper ARI (k=50)")
    axes[1, 1].set_ylabel("Same-paper MRR")
    axes[1, 1].set_title("D  Global and local rankings are task-dependent")
    axes[1, 1].legend(frameon=True, fontsize=10)

    fig.suptitle(
        "Phase 4 cross-dataset synthesis: multiple similarities remain necessary",
        fontsize=19,
        y=1.01,
    )
    fig.tight_layout()
    save(fig, "phase4_cross_dataset_overview")


def evidence_matrix() -> None:
    evidence = pd.read_csv(RESULTS / "hypothesis_evidence_matrix.csv")
    datasets = ["Starrydata", "HTEM", "NanoMine", "PNCExtract"]
    codes = {
        "not_tested": 0,
        "mixed": 1,
        "partial_support": 2,
        "support": 3,
    }
    matrix = np.asarray(
        [[codes[value] for value in evidence.loc[i, datasets]] for i in evidence.index]
    )
    labels = [
        "P1  Context change",
        "P2  Values beyond mask",
        "P3  Missingness confounding",
        "P4  Global/local difference",
        "P5  Non-identical views",
        "P6  Cross-source consensus",
    ]
    cmap = ListedColormap(["#D9D9D9", "#E45756", "#F2CF5B", "#54A24B"])
    fig, ax = plt.subplots(figsize=(10, 6.3))
    sns.heatmap(
        matrix,
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        xticklabels=datasets,
        yticklabels=labels,
        annot=np.vectorize(
            {
                0: "Not tested",
                1: "Mixed",
                2: "Partial",
                3: "Support",
            }.get
        )(matrix),
        fmt="",
        annot_kws={"fontsize": 10},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Phase 4 evidence matrix", pad=15)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    legend = [
        Patch(facecolor="#54A24B", label="Support"),
        Patch(facecolor="#F2CF5B", label="Partial support"),
        Patch(facecolor="#E45756", label="Mixed"),
        Patch(facecolor="#D9D9D9", label="Not tested"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "phase4_evidence_matrix")


if __name__ == "__main__":
    overview()
    evidence_matrix()
