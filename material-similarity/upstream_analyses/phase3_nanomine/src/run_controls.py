#!/usr/bin/env python3
"""Value-preserving controls for process order and process settings."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from phase3_core import (
    blend,
    build_feature_records,
    clustering_metrics,
    load_records,
    map_interval_distance,
    numeric_scales,
    pair_metadata,
    sequence_edit_distance,
    weighted_composite,
)
from run_phase3 import cluster_many


def load_facets(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    payload = np.load(path)
    facets = defaultdict(dict)
    for key in payload.files:
        if "__" in key:
            facet, field = key.split("__", 1)
            facets[facet][field] = payload[key]
    return dict(facets), payload["pair_i"], payload["pair_j"]


def shuffled_sequence_distance(features, pair_i, pair_j, rng):
    sequences = []
    for feature in features:
        sequence = list(feature.sequence)
        sequences.append([sequence[index] for index in rng.permutation(len(sequence))] if sequence else [])
    return np.fromiter(
        (sequence_edit_distance(sequences[i], sequences[j]) for i, j in zip(pair_i, pair_j)),
        dtype=np.float32,
        count=len(pair_i),
    )


def shuffled_setting_maps(features, rng):
    maps = [copy.deepcopy(feature.settings_all) for feature in features]
    locations = defaultdict(list)
    for sample_index, (feature, mapping) in enumerate(zip(features, maps)):
        family = tuple(feature.process_family)
        for key, values in mapping.items():
            for value_index, value in enumerate(values):
                stratum = (family, key, value["kind"], value.get("scale_key", ""))
                locations[stratum].append((sample_index, key, value_index))
    for positions in locations.values():
        if len(positions) < 2:
            continue
        values = [copy.deepcopy(maps[i][key][position]) for i, key, position in positions]
        order = rng.permutation(len(values))
        for target, source_index in zip(positions, order):
            i, key, position = target
            maps[i][key][position] = values[int(source_index)]
    return maps


def reported_setting_distance(maps, scales, pair_i, pair_j):
    def one(left, right):
        return map_interval_distance(left, right, scales)[1]

    return np.fromiter(
        (one(maps[i], maps[j]) for i, j in zip(pair_i, pair_j)),
        dtype=np.float32,
        count=len(pair_i),
    )


def evaluate(distance, cluster_counts, records, papers, pair_i, pair_j, pair_meta):
    rows = []
    for k, labels in cluster_many(distance, cluster_counts).items():
        rows.append({"k": k, **clustering_metrics(labels, papers, pair_i, pair_j, pair_meta)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase3_run_config.json"))
    parser.add_argument("--records", type=Path, default=Path("data/nanomine_phase3_records.jsonl.gz"))
    parser.add_argument("--facets", type=Path, default=Path("results/facet_distances.npz"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = load_records(args.records)
    features = build_feature_records(records)
    facets, pair_i, pair_j = load_facets(args.facets)
    pair_meta = pair_metadata(records, pair_i, pair_j)
    papers = np.asarray([record["paper_group"] for record in records], dtype=object)
    scales = numeric_scales(features)
    output = args.output_root / "results"
    rng = np.random.default_rng(config["random_seed"] + 1)
    cluster_counts = [20, 50]

    views = {view["name"]: view for view in config["sentinel_views"]}
    results = []
    for control_name, view_name in (("sequence_order_shuffle", "sequence_focus"), ("settings_value_shuffle", "settings_focus")):
        view = views[view_name]
        material = weighted_composite(facets, config["material_profiles"][view["material"]], "reported")
        original_process = weighted_composite(facets, config["process_profiles"][view["process"]], "reported")
        original_distance = blend(material, original_process, view["lambda"])
        for row in evaluate(original_distance, cluster_counts, records, papers, pair_i, pair_j, pair_meta):
            results.append({"control": control_name, "condition": "original", "repeat": -1, **row})

        for repeat in range(config["shuffle_repeats"]):
            modified = {name: {field: array for field, array in values.items()} for name, values in facets.items()}
            if control_name == "sequence_order_shuffle":
                shuffled = shuffled_sequence_distance(features, pair_i, pair_j, rng)
                modified["sequence"] = {**modified["sequence"], "reported": shuffled}
            else:
                maps = shuffled_setting_maps(features, rng)
                shuffled = reported_setting_distance(maps, scales, pair_i, pair_j)
                modified["settings_all"] = {**modified["settings_all"], "reported": shuffled}
            process = weighted_composite(modified, config["process_profiles"][view["process"]], "reported")
            distance = blend(material, process, view["lambda"])
            for row in evaluate(distance, cluster_counts, records, papers, pair_i, pair_j, pair_meta):
                results.append({"control": control_name, "condition": "shuffled", "repeat": repeat, **row})

    frame = pd.DataFrame(results)
    frame.to_csv(output / "process_shuffle_controls.csv", index=False)
    summaries = []
    for (control, k), group in frame.groupby(["control", "k"]):
        original = group[group.condition == "original"].iloc[0]
        shuffled = group[group.condition == "shuffled"]
        for metric in ("same_paper_cocluster", "same_paper_recall_lift", "paper_ari", "shared_author_lift"):
            summaries.append({
                "control": control,
                "k": int(k),
                "metric": metric,
                "original": float(original[metric]),
                "shuffled_mean": float(shuffled[metric].mean()),
                "shuffled_sd": float(shuffled[metric].std(ddof=1)),
                "original_minus_shuffled": float(original[metric] - shuffled[metric].mean()),
            })
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "process_shuffle_control_summary.csv", index=False)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()

