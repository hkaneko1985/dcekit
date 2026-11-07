#!/usr/bin/env python3
"""Rebuild manuscript Tables 1–8 and SI Tables S1–S3 from archived aggregate CSVs.

This is table regeneration from recorded results, not model refitting. Static
scope/distribution tables remain authored statements. Each numerical cell is
also exported unrounded with its source key in numeric_map.csv.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "publication/inputs"


def read(name):
    with (INPUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new or empty table output directory.")
    numeric = []
    tables = {}

    def number(artifact, row, column, value, source, key, digits):
        numeric.append({"artifact": artifact, "row": row, "column": column,
                        "value": float(value), "source": source, "source_key": key})
        return f"{float(value):.{digits}f}"

    def save(name, rows):
        with (output / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(rows)
        tables[name] = rows

    static = json.loads((ROOT / "publication/static_tables.json").read_text())
    save("Table_1_scope", static["Table_1_scope"])
    save("Table_S2_distribution_notes", static["Table_S2_distribution_notes"])

    labels = {"modulus": "Young’s modulus", "stress": "Peak stress",
              "strain": "Strain", "hardness": "Shore A hardness", "Tc": "Metal Tc"}
    kinds = ["NN", "YAE", "YVAE"]
    source = "heldout_summary_v41.csv"
    lookup = {(r["source"], r["method"]): r for r in read(source)}
    rows = [["Source response", *kinds]]
    for prop in labels:
        label = labels[prop]
        rows.append([label] + [number("Table_2", label, kind, lookup[prop, kind]["delta_skill"],
                                    source, f"{prop}|{kind}|delta_skill", 3) for kind in kinds])
    save("Table_2_departure_skill", rows)

    source = "polymer_controls_v40.csv"
    lookup = {r["dataset"]: r for r in read(source) if r["method"] == "KRR"}
    independent = {(r["dataset"], r["view"]): float(r["rmse"])
                   for r in read("polymer_neural_v33.csv")
                   if r["method"] == "KRR" and r["encoder"] == "NN"}
    views = [("baseXD", "Baseline"), ("source_linear_predictions", "Linear (3)"),
             ("source_quadratic_predictions", "Quadratic (3)"),
             ("NN_boundary_pred", "NN prediction (3)"),
             ("NN_boundary_latent", "NN latent (24)"), ("NN_boundary_both", "NN both (27)")]
    rows = [["Target", *[label for _, label in views]]]
    units = {"modulus": "MPa", "stress": "MPa", "strain": "dimensionless", "hardness": "Shore A"}
    for prop in ["modulus", "stress", "strain", "hardness"]:
        dataset = "polymer_" + prop
        label = labels[prop] + f" ({units[prop]})"
        for view, _ in views[3:]:
            if abs(float(lookup[dataset][view]) - independent[dataset, view.removeprefix("NN_")]) > 1e-10:
                raise ValueError(f"v40 and v33 neural values differ for {dataset}/{view}")
        rows.append([label] + [number("Table_3", label, column, lookup[dataset][view], source,
                                    f"{dataset}|KRR|{view}", 2 if prop == "modulus" else 5 if prop == "strain" else 4)
                              for view, column in views])
    save("Table_3_polymer_transfer", rows)

    source = "transfer_summary_v42.csv"
    lookup = {(r["case"], r["kind"], r["method"]): r for r in read(source)}
    cases = [("metals_volume", "Metal volume (Å³/atom)", 4),
             ("oxides_formation", "Oxide formation (eV/atom)", 5)]
    modes = [("XD", "Composition + fixed descriptors"),
             ("single_source_fixed", "Fixed same-property source"),
             ("single_source_nested", "Inner-CV-selected source"),
             ("all_frozen", "All 24 sources")]
    rows = [["Target / encoder", *[label for _, label in modes]]]
    for case, label, digits in cases:
        for kind in kinds:
            row = label + " / " + kind
            values = []
            for mode, column in modes:
                source_kind = "NN" if mode == "XD" else kind
                values.append(number("Table_4", row, column, lookup[case, source_kind, mode]["rmse_mean"],
                                     source, f"{case}|{source_kind}|{mode}", digits))
            rows.append([row, *values])
    save("Table_4_fixed_transfer", rows)

    modes = [("frozen_concat_head", "Frozen concatenation"), ("gated_frozen_concat", "Gate only"),
             ("gated_lastlayer_concat", "Gate + last layer"), ("sum_frozen_head", "Frozen mean"),
             ("gated_lastlayer_sum", "Adaptive weighted sum")]
    rows = [["Target / encoder", *[label for _, label in modes]]]
    for case, label, digits in cases:
        for kind in kinds:
            row = label + " / " + kind
            rows.append([row] + [number("Table_5", row, column, lookup[case, kind, mode]["rmse_mean"],
                                      source, f"{case}|{kind}|{mode}", digits)
                                for mode, column in modes])
    save("Table_5_adaptive_heads", rows)

    source = "learning_summary_v44.csv"
    lookup = {(r["case"], int(r["budget"]), r["kind"], r["method"]): r for r in read(source)}
    rows = [["Target / labels", "Composition + fixed descriptors", "All NN sources", "All YAE sources", "All YVAE sources"]]
    for case, label, digits in cases:
        for budget in [16, 32, 64]:
            row = label + " / " + str(budget)
            modes = [("NN", "XD"), ("NN", "all_frozen"), ("YAE", "all_frozen"), ("YVAE", "all_frozen")]
            rows.append([row] + [number("Table_6", row, rows[0][i + 1], lookup[case, budget, kind, mode]["rmse_mean"],
                                      source, f"{case}|{budget}|{kind}|{mode}", digits)
                                for i, (kind, mode) in enumerate(modes)])
    save("Table_6_learning_curves", rows)

    source = "steel_summary_v46.csv"
    lookup = {(r["method"], r["kind"], r["mode"]): r for r in read(source)}
    regressors = ["PLS", "LASSO", "KRR", "SVR", "LightGBM", "GBDT"]
    rows = [["Regressor", "Composition X", "X + D", "+NN", "+YAE", "+YVAE", "+Untrained"]]
    modes = [("NN", "X"), ("NN", "XD"), ("NN", "all_frozen"),
             ("YAE", "all_frozen"), ("YVAE", "all_frozen"), ("random", "all_frozen")]
    for regressor in regressors:
        if float(lookup[regressor, "NN", "XD"]["rmse_mean"]) != float(lookup[regressor, "NN", "XDM"]["rmse_mean"]):
            raise ValueError("Steel XD and XDM comparator equality changed.")
        rows.append([regressor] + [number("Table_7", regressor, rows[0][i + 1], lookup[regressor, kind, mode]["rmse_mean"],
                                        source, f"{regressor}|{kind}|{mode}", 2)
                                  for i, (kind, mode) in enumerate(modes)])
    save("Table_7_steel", rows)

    source = "architecture_summary_v51.csv"
    lookup = {(r["case"], r["feature_set"]): r for r in read(source)}
    cases51 = ["polymer_modulus", "metals_volume", "oxides_formation"]
    modes = [("composition", "Composition only"), ("basis", "Common baseline B"),
             ("NN", "B + NN"), ("YAE", "B + YAE"), ("YVAE", "B + YVAE"),
             ("random", "B + untrained control"), ("fusion", "B + NN + YAE + YVAE"),
             ("fusion_boruta", "B + concatenation + Boruta")]
    rows = [["Representation", "Modulus MPa", "Metal volume Å³/atom", "Oxide formation eV/atom"]]
    for mode, label in modes:
        rows.append([label] + [number("Table_8", label, rows[0][i + 1], lookup[case, mode]["rmse_mean"], source,
                                    f"{case}|{mode}", 2 if i == 0 else 5) for i, case in enumerate(cases51)])
    save("Table_8_architecture_KRR", rows)

    source = "source_catalog_v38.csv"
    counts = {row["source"]: row for row in read("source_catalog_counts_verified52.csv")}
    rows = [["Source ID", "Property", "Compositions", "Maximum components"]]
    for row in read(source):
        derived = counts[row["source"]]
        if int(row["n_compositions"]) != int(derived["n_compositions_from_X"]):
            raise ValueError("Source catalogue composition count differs from archived X.")
        maximum = row["max_components"] or derived["max_components_from_X"]
        if int(maximum) != int(derived["max_components_from_X"]):
            raise ValueError("Source maximum-component metadata differs from archived X.")
        rows.append([row["source"], row["property"], row["n_compositions"], maximum])
    save("Table_S1_source_catalogue", rows)

    source = "selector_counts_v51.csv"
    selected = read(source)
    rows = [["Target", "Nominal learned coordinates", "Minimum", "Maximum", "Mean"]]
    for case, label, dimension in [("polymer_modulus", "Acrylate modulus", 72),
                                   ("metals_volume", "Metal volume", 576),
                                   ("oxides_formation", "Oxide formation", 576)]:
        values = [int(row["confirmed_learned"]) for row in selected if row["case"] == case and row["level"] == "outer"]
        rows.append([label, str(dimension), str(min(values)), str(max(values)), f"{statistics.mean(values):.2f}"])
    save("Table_S3_selected_coordinates", rows)

    with (output / "numeric_map.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(numeric[0]))
        writer.writeheader()
        writer.writerows(numeric)
    status = {
        "status": "PASS", "tables": len(tables), "main_numeric_cells": len(numeric),
        "new_model_fits": 0,
        "scope": "Rebuilt Tables 1–8/S1–S3 from archived aggregate data; Table 9/S4/S5 use the separate regressor-tables entry.",
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUT.glob("*.csv"))},
    }
    (output / "table_export_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    export(parser.parse_args().output_dir)
