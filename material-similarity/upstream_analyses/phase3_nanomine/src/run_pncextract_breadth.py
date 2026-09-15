#!/usr/bin/env python3
"""Composition-only breadth check on the public PNCExtract NanoMine sample set."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from phase3_core import (
    BOUNDS,
    assign_tuple,
    clustering_metrics,
    empty_facet_arrays,
    known_or_missing_set,
    map_interval_distance,
    missing_mask_distance,
    pair_metadata,
    retrieval_metrics,
    weighted_composite,
)
from run_phase3 import cluster_many


PROFILES = {
    "strict_equal": {"matrix_strict": 0.50, "filler_strict": 0.50},
    "soft_equal": {"matrix_soft": 0.50, "filler_soft": 0.50},
    "matrix_focus": {"matrix_strict": 0.75, "filler_strict": 0.25},
    "filler_focus": {"matrix_strict": 0.25, "filler_strict": 0.75},
    "strict_loading": {"matrix_strict": 0.35, "filler_strict": 0.35, "loading": 0.30},
    "soft_loading": {"matrix_soft": 0.35, "filler_soft": 0.35, "loading": 0.30},
    "loading_focus": {"matrix_strict": 0.20, "filler_strict": 0.20, "loading": 0.60},
}


def article_metadata(path: Path) -> dict[str, dict]:
    output = {}
    for file in path.glob("*.json"):
        try:
            common = json.loads(file.read_text(encoding="utf-8"))["Citation"]["CommonFields"]
        except (KeyError, json.JSONDecodeError):
            continue
        output[file.stem.lower()] = {
            "authors": common.get("Author", []),
            "location": common.get("Location"),
            "title": common.get("Title"),
            "citation_doi": common.get("DOI"),
        }
    return output


def load_samples(root: Path, metadata_dir: Path) -> list[dict]:
    metadata = article_metadata(metadata_dir)
    records = []
    for file in sorted(root.glob("*/L*/*.json")):
        row = json.loads(file.read_text(encoding="utf-8"))
        article = file.parent.name.lower()
        matrix = row.get("Matrix Chemical Name")
        filler = row.get("Filler Chemical Name")
        loading = {}
        for field, key in (("Filler Composition Mass", "MassFraction"), ("Filler Composition Volume", "VolumeFraction")):
            raw = row.get(field)
            if raw is not None and str(raw).strip():
                try:
                    loading[key] = float(raw)
                except ValueError:
                    pass
        records.append({
            "sample_id": f"{article}/{file.stem}",
            "article_id": article,
            "paper_group": "article:" + article,
            "doi": (metadata.get(article) or {}).get("citation_doi"),
            "sample_label": f"{filler or 'unreported filler'} in {matrix or 'unreported matrix'}",
            "matrix_names": [matrix] if matrix else [],
            "filler_names": [filler] if filler else [],
            "surface_names": [],
            "process_families": [],
            "step_sequence": [],
            "citation_metadata": metadata.get(article),
            "loading": loading,
        })
    return records


def compute(records):
    pair_i, pair_j = np.triu_indices(len(records), 1)
    facets = {
        name: empty_facet_arrays(len(pair_i))
        for name in ("matrix_strict", "matrix_soft", "filler_strict", "filler_soft", "loading")
    }
    fraction_scale = {"fraction": 1.0}
    for index, (i, j) in enumerate(zip(pair_i, pair_j)):
        left, right = records[i], records[j]
        assign_tuple(facets["matrix_strict"], index, known_or_missing_set(
            left["matrix_names"], right["matrix_names"], bool(left["matrix_names"]), bool(right["matrix_names"]), False
        ))
        assign_tuple(facets["matrix_soft"], index, known_or_missing_set(
            left["matrix_names"], right["matrix_names"], bool(left["matrix_names"]), bool(right["matrix_names"]), True
        ))
        assign_tuple(facets["filler_strict"], index, known_or_missing_set(
            left["filler_names"], right["filler_names"], bool(left["filler_names"]), bool(right["filler_names"]), False
        ))
        assign_tuple(facets["filler_soft"], index, known_or_missing_set(
            left["filler_names"], right["filler_names"], bool(left["filler_names"]), bool(right["filler_names"]), True
        ))
        left_map = {key: [{"kind": "numeric", "value": value, "scale_key": "fraction"}] for key, value in left["loading"].items()}
        right_map = {key: [{"kind": "numeric", "value": value, "scale_key": "fraction"}] for key, value in right["loading"].items()}
        assign_tuple(facets["loading"], index, map_interval_distance(left_map, right_map, fraction_scale))
    return facets, pair_i, pair_j


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, help="Use the frozen normalized JSONL-GZIP cohort without downloading upstream files.")
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.records is not None:
        if args.sample_root is not None or args.metadata_dir is not None:
            parser.error("Use --records or the two source directories, not both.")
        with gzip.open(args.records, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        if args.sample_root is None or args.metadata_dir is None:
            parser.error("Provide --records, or both --sample-root and --metadata-dir.")
        records = load_samples(args.sample_root, args.metadata_dir)
    if not records:
        parser.error("No sample records were loaded.")
    facets, pair_i, pair_j = compute(records)
    pair_meta = pair_metadata(records, pair_i, pair_j)
    papers = np.asarray([record["paper_group"] for record in records], dtype=object)
    output = args.output_root / "results"
    output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(data_dir / "pncextract_breadth_records.jsonl.gz", "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    rows = []
    retrieval = []
    for profile, weights in PROFILES.items():
        for bound in BOUNDS:
            distance = weighted_composite(facets, weights, bound)
            for k, labels in cluster_many(distance, [8, 12, 20, 30, 50, 80]).items():
                rows.append({
                    "profile": profile,
                    "bound": bound,
                    "requested_clusters": k,
                    "mean_interval_width": float(np.mean(
                        weighted_composite(facets, weights, "pessimistic")
                        - weighted_composite(facets, weights, "optimistic")
                    )),
                    **clustering_metrics(labels, papers, pair_i, pair_j, pair_meta),
                })
            if profile in {"strict_equal", "strict_loading", "soft_loading"}:
                retrieval.append({"profile": profile, "bound": bound, **retrieval_metrics(distance, records)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pncextract_breadth_grid.csv", index=False)
    pd.DataFrame(retrieval).to_csv(output / "pncextract_breadth_retrieval.csv", index=False)
    summary = {
        "samples": len(records),
        "papers": len(set(papers)),
        "same_paper_pairs": int(pair_meta["same_paper"].sum()),
        "clusterings": len(frame),
        "loading_reporting": Counter(
            "both" if len(record["loading"]) == 2 else "one" if len(record["loading"]) == 1 else "neither"
            for record in records
        ),
        "reported_lift": {
            "min": float(frame.loc[frame.bound == "reported", "same_paper_recall_lift"].min()),
            "median": float(frame.loc[frame.bound == "reported", "same_paper_recall_lift"].median()),
            "max": float(frame.loc[frame.bound == "reported", "same_paper_recall_lift"].max()),
        },
    }
    (output / "pncextract_breadth_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=dict), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
