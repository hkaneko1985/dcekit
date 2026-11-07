"""Render the complete stage52 comparison from recorded aggregate values.

No model selection or predictive fitting is performed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np


CASES = ["polymer_modulus", "metals_volume", "oxides_formation"]
REGRESSORS = ["PLS", "LASSO", "KRR", "SVR", "LightGBM", "GBDT"]
FEATURES = ["composition", "basis", "NN", "YAE", "YVAE", "random", "fusion", "fusion_boruta"]
PLOT_FEATURES = [x for x in FEATURES if x != "basis"]
NAMES = {"polymer_modulus": "Acrylate Young's modulus",
         "metals_volume": "Metal atomic volume",
         "oxides_formation": "Oxide formation energy"}
SHORT = {"composition": "Composition", "NN": "NN", "YAE": "YAE", "YVAE": "YVAE",
         "random": "Untrained", "fusion": "Combined", "fusion_boruta": "Combined\n+ Boruta"}


def read_complete(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(c, r, f) for c in CASES for r in REGRESSORS for f in FEATURES}
    data = {}
    for row in rows:
        key = row["case"], row["regressor"], row["feature_set"]
        if key in data:
            raise ValueError(f"Duplicate aggregate: {key}")
        repeats = 1 if row["case"] == "polymer_modulus" else 2
        expected_count = repeats * (1 if row["feature_set"] in {"composition", "basis"} else 3)
        if int(row["n_config_repeats"]) != expected_count:
            raise ValueError(f"Incomplete seed/repeat aggregation: {key}")
        data[key] = float(row["rmse_mean"])
    if set(data) != expected:
        raise ValueError("The full 144-cell comparison is required; partial results are not plotted")
    if not all(np.isfinite(v) and v > 0 for v in data.values()):
        raise ValueError("Invalid RMSE value")
    return data


def derive(data):
    comparisons = []
    for case in CASES:
        for reg in REGRESSORS:
            vals = {feat: data[case, reg, feat] for feat in FEATURES}
            single = min(["NN", "YAE", "YVAE"], key=vals.get)
            best = min(FEATURES, key=vals.get)
            comparisons.append({
                "case": case, "regressor": reg,
                "composition_rmse": vals["composition"], "basis_rmse": vals["basis"],
                "best_single_architecture": single, "best_single_rmse": vals[single],
                "fusion_rmse": vals["fusion"], "boruta_rmse": vals["fusion_boruta"],
                "best_of_eight": best, "best_of_eight_rmse": vals[best],
                "fusion_reduction_vs_basis_pct": 100 * (1 - vals["fusion"] / vals["basis"]),
                "fusion_reduction_vs_best_single_pct": 100 * (1 - vals["fusion"] / vals[single]),
                "boruta_reduction_vs_basis_pct": 100 * (1 - vals["fusion_boruta"] / vals["basis"]),
                "boruta_reduction_vs_fusion_pct": 100 * (1 - vals["fusion_boruta"] / vals["fusion"]),
                "boruta_reduction_vs_best_single_pct": 100 * (1 - vals["fusion_boruta"] / vals[single]),
            })
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = read_complete(args.summary)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    comparisons = derive(data)
    with (out / "regressor_comparison_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    counts = {field: sum(row[field] > 1e-8 for row in comparisons) for field in [
        "fusion_reduction_vs_basis_pct", "fusion_reduction_vs_best_single_pct",
        "boruta_reduction_vs_basis_pct", "boruta_reduction_vs_fusion_pct",
        "boruta_reduction_vs_best_single_pct"]}
    report = {"summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
              "scope": "18 target-regressor settings; descriptive comparisons of separately averaged RMSEs, not independent replications or inferential tests",
              "definition": "positive reduction percentage denotes lower RMSE; best single is a post-hoc descriptive minimum over NN, YAE, YVAE",
              "comparisons": comparisons, "positive_comparison_counts": counts}
    (out / "regressor_comparison_analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(6.65, 7.05))
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.15, top=0.955, hspace=0.45)
    cmap = LinearSegmentedColormap.from_list("blue_white_orange", ["#2478A8", "#FAFAFA", "#CA6B26"])
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)
    observed = []
    for idx, (case, ax) in enumerate(zip(CASES, axes)):
        matrix = np.array([[100 * (data[case, reg, feat] / data[case, reg, "basis"] - 1)
                            for feat in PLOT_FEATURES] for reg in REGRESSORS])
        observed.extend(matrix.ravel().tolist())
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"{chr(97+idx)}) {NAMES[case]}", loc="left", pad=7)
        ax.set_yticks(range(len(REGRESSORS)), REGRESSORS)
        ax.set_xticks(range(len(PLOT_FEATURES)), [SHORT[x] for x in PLOT_FEATURES])
        ax.tick_params(axis="both", length=0, pad=4, labelsize=8)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                val = matrix[row, col]
                text_color = "white" if abs(val) >= 36 else "#17202A"
                label = f"{val:+.2f}" if 0 < abs(val) < 0.05 else (f"{val:+.1f}" if abs(val) < 100 else f"{val:+.0f}")
                ax.text(col, row, label, ha="center", va="center", color=text_color, fontsize=8.3)
        ax.set_xticks(np.arange(-0.5, len(PLOT_FEATURES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(REGRESSORS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values(): spine.set_visible(False)
    color_ax = fig.add_axes([0.25, 0.075, 0.60, 0.017])
    color = fig.colorbar(im, cax=color_ax, orientation="horizontal", extend="both", ticks=[-50, -25, 0, 25, 50])
    color.set_label("RMSE change relative to B (%)   |   negative is better", fontsize=8.5, labelpad=3)
    color.ax.tick_params(labelsize=8)
    for ext in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 300} if ext == "png" else {}
        if ext == "pdf": kwargs["metadata"] = {"Title": "Six-regressor descriptor comparison", "Author": "Hiromasa Kaneko", "CreationDate": None}
        fig.savefig(out / f"Figure_8_Regressor_Comparison_v52.{ext}", **kwargs)
    plt.close(fig)
    (out / "figure8_metadata.json").write_text(json.dumps({
        "summary_sha256": report["summary_sha256"], "cells": len(observed),
        "minimum_percent_change": min(observed), "maximum_percent_change": max(observed),
        "color_limits_percent": [-50, 50], "clipped_cells_color_only": sum(abs(v)>50 for v in observed),
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "source_data_transformed_only": True,
    }, indent=2) + "\n")
    print(json.dumps({"comparison_rows": len(comparisons), "plot_cells": len(observed), "output_dir": str(out)}))


if __name__ == "__main__":
    main()
