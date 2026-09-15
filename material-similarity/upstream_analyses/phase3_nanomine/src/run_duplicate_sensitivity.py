#!/usr/bin/env python3
"""Audit and collapse exact input duplicates within each source paper."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from phase3_core import (
    blend,
    clustering_metrics,
    load_records,
    pair_metadata,
    square_subset,
    weighted_composite,
)
from run_phase3 import cluster_many
from run_controls import load_facets


def signature(record: dict, mode: str) -> str:
    material = {
        "matrix": record.get("matrix_names", []),
        "filler": record.get("filler_names", []),
        "surface": record.get("surface_names", []),
        "component_attributes": [
            (
                component.get("role"),
                component.get("name"),
                [
                    (attr.get("type"), attr.get("value"), attr.get("unit_group"), attr.get("raw_value"))
                    for attr in component.get("attributes", [])
                    if attr.get("type") in {
                        "MassFraction", "VolumeFraction", "Density", "Width", "AspectRatio", "SpecificSurfaceArea"
                    }
                ],
            )
            for component in record.get("components", [])
        ],
    }
    process = {
        "families": record.get("process_families", []),
        "sequence": record.get("step_sequence", []),
        "settings": [
            (setting.get("comparison_key"), setting.get("kind"), setting.get("value"), setting.get("unit_group"))
            for setting in record.get("settings", [])
        ],
    }
    payload = material if mode == "material" else process if mode == "process" else {"material": material, "process": process}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit(records: list[dict]) -> pd.DataFrame:
    rows = []
    for mode in ("material", "process", "full"):
        groups = defaultdict(list)
        for index, record in enumerate(records):
            groups[signature(record, mode)].append(index)
        within = cross = 0
        for indices in groups.values():
            for i, j in itertools.combinations(indices, 2):
                if records[i]["paper_group"] == records[j]["paper_group"]:
                    within += 1
                else:
                    cross += 1
        rows.append({
            "signature": mode,
            "samples": len(records),
            "unique_signatures": len(groups),
            "duplicate_groups": sum(len(indices) > 1 for indices in groups.values()),
            "largest_group": max(map(len, groups.values())),
            "within_paper_duplicate_pairs": within,
            "cross_paper_duplicate_pairs": cross,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase3_run_config.json"))
    parser.add_argument("--records", type=Path, default=Path("data/nanomine_phase3_records.jsonl.gz"))
    parser.add_argument("--facets", type=Path, default=Path("results/facet_distances.npz"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = load_records(args.records)
    facets, pair_i, pair_j = load_facets(args.facets)
    output = args.output_root / "results"
    audit(records).to_csv(output / "duplicate_signature_audit.csv", index=False)

    seen = set()
    keep = []
    for index, record in enumerate(records):
        key = (record["paper_group"], signature(record, "full"))
        if key not in seen:
            seen.add(key)
            keep.append(index)
    keep = np.asarray(keep, dtype=int)
    subset_records = [records[index] for index in keep]
    sub_i, sub_j = np.triu_indices(len(keep), 1)
    meta = pair_metadata(subset_records, sub_i, sub_j)
    papers = np.asarray([record["paper_group"] for record in subset_records], dtype=object)

    rows = []
    for view in config["sentinel_views"]:
        material = weighted_composite(facets, config["material_profiles"][view["material"]], "reported")
        process = weighted_composite(facets, config["process_profiles"][view["process"]], "reported")
        full_distance = blend(material, process, view["lambda"])
        distance = square_subset(full_distance, keep)
        for k, labels in cluster_many(distance, [20, 50]).items():
            rows.append({
                "view": view["name"],
                "k": k,
                "original_samples": len(records),
                "deduplicated_samples": len(keep),
                "removed_samples": len(records) - len(keep),
                **clustering_metrics(labels, papers, sub_i, sub_j, meta),
            })
    pd.DataFrame(rows).to_csv(output / "deduplicated_sentinel_sensitivity.csv", index=False)
    print(pd.DataFrame(rows)[["view", "k", "deduplicated_samples", "same_paper_cocluster", "same_paper_recall_lift", "paper_ari"]].to_string(index=False))


if __name__ == "__main__":
    main()

