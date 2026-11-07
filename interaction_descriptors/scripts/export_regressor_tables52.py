#!/usr/bin/env python3
"""Export Table 9 and Supplementary Tables S4/S5 from recorded stage52 results.

This standard-library entry point performs no fitting and requires all 144 cells.
The grid table is derived from the frozen candidate dictionaries, not retyped.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/regressor_comparison/frozen_reference"
CASES = ["polymer_modulus", "metals_volume", "oxides_formation"]
REGRESSORS = ["PLS", "LASSO", "KRR", "SVR", "LightGBM", "GBDT"]
FEATURES = ["composition", "basis", "NN", "YAE", "YVAE", "random", "fusion", "fusion_boruta"]
LABELS = {"polymer_modulus": "Acrylate modulus\nMPa", "metals_volume": "Metal volume\nÅ³/atom", "oxides_formation": "Oxide formation\neV/atom"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=REFERENCE / "summary.csv")
    parser.add_argument("--protocol", type=Path, default=REFERENCE / "protocol.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    data = {}
    for row in rows:
        key = row["case"], row["regressor"], row["feature_set"]
        if key in data:
            raise ValueError(f"Duplicate summary cell: {key}")
        expected = (1 if key[0] == "polymer_modulus" else 2) * (1 if key[2] in {"composition", "basis"} else 3)
        if int(row["n_config_repeats"]) != expected:
            raise ValueError(f"Incomplete seed/repeat aggregation: {key}")
        data[key] = float(row["rmse_mean"])
    if set(data) != set(itertools.product(CASES, REGRESSORS, FEATURES)):
        raise ValueError("Exactly the complete 144-cell matrix is required.")
    if not all(math.isfinite(value) and value > 0 for value in data.values()):
        raise ValueError("Nonpositive or nonfinite error value.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Use a new or empty output directory.")

    def save(name, table):
        with (args.output_dir / name).open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(table)

    def fmt(value, case):
        return f"{value:.2f}" if case == "polymer_modulus" else f"{value:.5f}"

    table = [["Target", "Regressor", "C", "B", "Single", "Combined", "Combined\n+ Boruta"]]
    for case in CASES:
        for reg in REGRESSORS:
            single = min(["NN", "YAE", "YVAE"], key=lambda family: data[case, reg, family])
            table.append([LABELS[case], reg, fmt(data[case, reg, "composition"], case),
                          fmt(data[case, reg, "basis"], case), single + "\n" + fmt(data[case, reg, single], case),
                          fmt(data[case, reg, "fusion"], case), fmt(data[case, reg, "fusion_boruta"], case)])
    save("Table_9_regressor_comparison.csv", table)

    protocol = json.loads(args.protocol.read_text())
    grids = [["Regressor", "Count", "Grid"]]
    for reg in REGRESSORS:
        candidates = protocol["grids"][reg]
        keys = list(candidates[0])
        options = {key: list(dict.fromkeys(candidate[key] for candidate in candidates)) for key in keys}
        expected = {tuple(values) for values in itertools.product(*options.values())}
        observed = {tuple(candidate[key] for key in keys) for candidate in candidates}
        if expected != observed or len(candidates) != len(expected):
            raise ValueError(f"Grid is not an exhaustive unique Cartesian product: {reg}")
        description = "; ".join(key + " = " + ", ".join(format(value, "g") for value in values) for key, values in options.items())
        grids.append([reg + (" (reused)" if reg == "KRR" else ""), str(len(candidates)), description])
    save("Table_S4_regressor_grids.csv", grids)
    for index, case in enumerate(CASES):
        matrix = [["Regressor", "C", "B", "NN", "YAE", "YVAE", "U", "Combined", "+ Boruta"]]
        for reg in REGRESSORS:
            matrix.append([reg] + [fmt(data[case, reg, feat], case) for feat in FEATURES])
        save(f"Table_S5{chr(97 + index)}_{case}.csv", matrix)
    status = {"status": "PASS", "tables": 5, "table9_numeric_cells": 90,
              "supplement_matrix_cells": 144, "grid_candidate_counts": {reg: len(protocol["grids"][reg]) for reg in REGRESSORS},
              "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(), "new_model_fits": 0,
              "scope": "Recorded aggregate table exports; the best single family is a descriptive post-evaluation minimum."}
    (args.output_dir / "regressor_table_export_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
