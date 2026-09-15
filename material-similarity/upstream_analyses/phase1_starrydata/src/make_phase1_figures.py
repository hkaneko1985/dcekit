#!/usr/bin/env python3
"""Render publication-ready Phase 1 summary figures from frozen JSON results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURE_LABELS = {
    "F0": "F0\nComposition",
    "F1": "F1\n+ Form",
    "F2": "F2\n+ Fabrication",
    "F3": "F3\n+ State",
}
COLORS = {
    "F0": "#6B7280",
    "F1": "#4C78A8",
    "F2": "#F58518",
    "F3": "#54A24B",
}


def save_both(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def feature_lift_figure(primary: dict, output_dir: Path) -> None:
    features = ["F0", "F1", "F2", "F3"]
    values = [primary["evaluation"][feature]["selected_resolution"]["lift"] for feature in features]
    intervals = [
        [math.exp(value) for value in primary["evaluation"][feature]["log_lift_bootstrap_95_interval"]]
        for feature in features
    ]
    lower = np.asarray([value - interval[0] for value, interval in zip(values, intervals)])
    upper = np.asarray([interval[1] - value for value, interval in zip(values, intervals)])

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(features))
    bars = ax.bar(
        x,
        values,
        yerr=np.vstack([lower, upper]),
        capsize=5,
        color=[COLORS[feature] for feature in features],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.1, label="No enrichment")
    ax.set_xticks(x, [FEATURE_LABELS[feature] for feature in features])
    ax.set_ylabel("Same-SID co-assignment lift")
    fig.suptitle(
        "Process-aware metadata increases weak-context enrichment",
        x=0.09,
        y=0.98,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.89,
        "Confirmation partition; 95% paper-block bootstrap intervals",
        fontsize=9,
        color="#4B5563",
    )
    ax.set_ylim(0, max(interval[1] for interval in intervals) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.13, f"{value:.2f}×", ha="center", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.01,
        0.005,
        "Same SID is weak positive support; different-SID pairs are composition-matched unlabeled comparisons.",
        fontsize=8,
        color="#4B5563",
    )
    fig.subplots_adjust(top=0.80, bottom=0.22)
    save_both(fig, output_dir / "phase1_feature_lifts")


def resolution_figure(resolution: dict, output_dir: Path) -> None:
    features = ["F0", "F1", "F2", "F3"]
    counts = sorted(int(value) for value in resolution["evaluation"]["F0"]["fixed_resolution_lift"])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for feature in features:
        values = [resolution["evaluation"][feature]["fixed_resolution_lift"][str(count)] for count in counts]
        ax.plot(counts, values, marker="o", linewidth=2, markersize=4, label=feature, color=COLORS[feature])
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.axvline(150, color="#9CA3AF", linestyle=":", linewidth=1.2)
    ax.text(153, 0.45, "Primary grid boundary", fontsize=8, color="#6B7280", rotation=90, va="bottom")
    ax.set_xlabel("Requested number of clusters")
    ax.set_ylabel("Same-SID co-assignment lift")
    fig.suptitle(
        "F2 exceeds F0 at every tested resolution",
        x=0.09,
        y=0.98,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.09,
        0.89,
        "25–150 preregistered; 200–300 post-hoc boundary sensitivity",
        fontsize=9,
        color="#4B5563",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.text(
        0.01,
        0.005,
        "The positive increment is multiresolution-robust; no unique natural cluster count was identified.",
        fontsize=8,
        color="#4B5563",
    )
    fig.subplots_adjust(top=0.80, bottom=0.20)
    save_both(fig, output_dir / "phase1_resolution_sensitivity")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--resolution-sensitivity", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    resolution = json.loads(args.resolution_sensitivity.read_text(encoding="utf-8"))
    feature_lift_figure(primary, args.output_dir)
    resolution_figure(resolution, args.output_dir)
    print(json.dumps({
        "created": [
            str(args.output_dir / "phase1_feature_lifts.png"),
            str(args.output_dir / "phase1_feature_lifts.svg"),
            str(args.output_dir / "phase1_resolution_sensitivity.png"),
            str(args.output_dir / "phase1_resolution_sensitivity.svg"),
        ]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
