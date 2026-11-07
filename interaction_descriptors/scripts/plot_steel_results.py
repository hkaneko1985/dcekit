#!/usr/bin/env python3
"""Redraw the fixed steel comparison from its recorded aggregate values.

This is a data-equivalent redraw of Figure 7, not the unavailable original
rendering script, and does not perform model fitting or prediction ensembling.
"""
from pathlib import Path
import argparse
import csv
import json
import hashlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with args.summary.open(encoding="utf-8-sig", newline="") as stream:
        data = {(row["method"], row["kind"], row["mode"]): float(row["rmse_mean"])
                for row in csv.DictReader(stream)}
    methods = ["PLS", "LASSO", "KRR", "SVR", "LightGBM", "GBDT"]
    series = [("NN", "X", "Composition only", "o"),
              ("NN", "XD", "Composition + fixed descriptors", "s"),
              ("NN", "all_frozen", "+ NN latent", "^"),
              ("YAE", "all_frozen", "+ YAE latent", "D"),
              ("YVAE", "all_frozen", "+ YVAE latent", "v"),
              ("random", "all_frozen", "+ Untrained latent", "x")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.5, 4.9))
    coordinates = []
    for index, (kind, mode, label, marker) in enumerate(series):
        values = [data[method, kind, mode] for method in methods]
        x = np.arange(len(methods)) + (index - 2.5) * 0.055
        axis.plot(x, values, marker=marker, linestyle="none", label=label, markersize=6)
        coordinates.extend({"regressor": method, "kind": kind, "mode": mode,
                            "rmse_mean_MPa": value} for method, value in zip(methods, values))
    axis.set_xticks(np.arange(len(methods)), methods)
    axis.set_ylabel("Yield-strength RMSE (MPa)")
    axis.set_title("Fixed steel comparison: all prespecified regressors")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(args.output_dir / f"Figure_7_steel_redraw.{ext}", dpi=200)
    plt.close(fig)
    with (args.output_dir / "Figure_7_plot_data.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coordinates[0]))
        writer.writeheader()
        writer.writerows(coordinates)
    status = {"status": "PASS", "plotted_aggregate_values": 36, "new_model_fits": 0,
              "source_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
              "rendering": "new rendering of original aggregate data; pixel identity to original Figure7 not claimed",
              "matplotlib": matplotlib.__version__}
    (args.output_dir / "Figure_7_plot_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
