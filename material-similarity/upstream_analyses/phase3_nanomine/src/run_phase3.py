#!/usr/bin/env python3
"""Run the preregistered multi-view NanoMine Phase 3 clustering grid."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from phase3_core import (
    BOUNDS,
    blend,
    build_feature_records,
    clustering_metrics,
    compute_facets,
    load_records,
    missing_mask_distance,
    pair_metadata,
    retrieval_metrics,
    square_subset,
    weighted_composite,
)


def configuration_distance(
    facets: dict,
    config: dict,
    material_name: str,
    process_name: str,
    process_lambda: float,
    bound: str,
) -> np.ndarray:
    material = weighted_composite(facets, config["material_profiles"][material_name], bound)
    process = weighted_composite(facets, config["process_profiles"][process_name], bound)
    return blend(material, process, process_lambda)


def cluster_many(distance: np.ndarray, cluster_counts: list[int]) -> dict[int, np.ndarray]:
    matrix = np.asarray(distance, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Non-finite composite distance")
    tree = linkage(matrix, method="average")
    valid = sorted({int(k) for k in cluster_counts if 1 < k < len(tree) + 1})
    cuts = cut_tree(tree, n_clusters=valid)
    return {k: cuts[:, column].astype(int) for column, k in enumerate(valid)}


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(array, dtype=np.float32).tobytes()).hexdigest()


def audit_facets(facets: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for facet, values in facets.items():
        reported = values["reported"]
        width = values["pessimistic"] - values["optimistic"]
        rows.append({
            "facet": facet,
            "pair_count": len(reported),
            "reported_pair_fraction": float(np.mean(np.isfinite(reported))),
            "mean_common_coverage": float(np.mean(values["coverage"])),
            "median_common_coverage": float(np.median(values["coverage"])),
            "mean_interval_width": float(np.mean(width)),
            "median_interval_width": float(np.median(width)),
            "median_reported_distance": float(np.nanmedian(reported)),
        })
    return pd.DataFrame(rows)


def permutation_test(
    labels: np.ndarray,
    paper_labels: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> dict:
    co = labels[pair_i] == labels[pair_j]
    observed_mask = paper_labels[pair_i] == paper_labels[pair_j]
    observed = float(np.mean(co[observed_mask]))
    null = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        shuffled = rng.permutation(paper_labels)
        mask = shuffled[pair_i] == shuffled[pair_j]
        null[repeat] = np.mean(co[mask])
    sd = float(np.std(null, ddof=1))
    return {
        "observed_same_paper_cocluster": observed,
        "null_mean": float(np.mean(null)),
        "null_sd": sd,
        "z": float((observed - np.mean(null)) / sd) if sd else math.inf,
        "one_sided_p": float((1 + np.sum(null >= observed)) / (repeats + 1)),
        "repeats": repeats,
    }


def metadata_for_pair(record: dict) -> dict:
    citation = record.get("citation_metadata") or {}
    return {
        "sample_id": record["sample_id"],
        "paper_group": record["paper_group"],
        "doi": record.get("doi"),
        "sample_label": record.get("sample_label"),
        "matrix": " | ".join(record.get("matrix_names", [])),
        "filler": " | ".join(record.get("filler_names", [])),
        "surface_treatment": " | ".join(record.get("surface_names", [])),
        "process_family": " | ".join(record.get("process_families", [])),
        "sequence": " > ".join(record.get("step_sequence", [])),
        "authors": " | ".join(citation.get("authors", [])),
        "location": citation.get("location"),
    }


def consensus_pairs(
    records: list[dict],
    distances: dict[str, np.ndarray],
    k_neighbors: int,
) -> pd.DataFrame:
    papers = np.asarray([record["paper_group"] for record in records], dtype=object)
    support: dict[tuple[int, int], dict] = defaultdict(lambda: {"views": set(), "ranks": [], "distances": []})
    for view, condensed in distances.items():
        matrix = squareform(condensed.astype(float))
        matrix[papers[:, None] == papers[None, :]] = np.inf
        np.fill_diagonal(matrix, np.inf)
        for i in range(len(records)):
            finite = np.flatnonzero(np.isfinite(matrix[i]))
            if not len(finite):
                continue
            order = finite[np.argsort(matrix[i, finite], kind="stable")[:k_neighbors]]
            for rank, j in enumerate(order, start=1):
                pair = (i, int(j)) if i < j else (int(j), i)
                entry = support[pair]
                entry["views"].add(view)
                entry["ranks"].append(rank)
                entry["distances"].append(float(matrix[i, j]))
    rows = []
    for (i, j), entry in support.items():
        left, right = metadata_for_pair(records[i]), metadata_for_pair(records[j])
        left_authors = set(left["authors"].split(" | ")) if left["authors"] else set()
        right_authors = set(right["authors"].split(" | ")) if right["authors"] else set()
        row = {
            "view_support": len(entry["views"]),
            "directed_neighbor_hits": len(entry["ranks"]),
            "mean_neighbor_rank": float(np.mean(entry["ranks"])),
            "mean_distance": float(np.mean(entry["distances"])),
            "shared_author": bool(left_authors & right_authors),
            "same_location_exact": bool(left["location"] and left["location"] == right["location"]),
        }
        row.update({f"left_{key}": value for key, value in left.items()})
        row.update({f"right_{key}": value for key, value in right.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["view_support", "directed_neighbor_hits", "mean_neighbor_rank", "mean_distance"],
        ascending=[False, False, True, True],
    ).head(200)


def cluster_profiles(records: list[dict], labels: np.ndarray, view: str, k: int) -> list[dict]:
    rows = []
    for cluster in sorted(set(labels)):
        indices = np.flatnonzero(labels == cluster)
        subset = [records[index] for index in indices]
        paper_counts = Counter(record["paper_group"] for record in subset)
        matrices = Counter(name for record in subset for name in record.get("matrix_names", []))
        fillers = Counter(name for record in subset for name in record.get("filler_names", []))
        families = Counter(family for record in subset for family in record.get("process_families", []))
        sequences = Counter(" > ".join(record.get("step_sequence", [])) for record in subset)
        rows.append({
            "view": view,
            "k": k,
            "cluster": int(cluster),
            "size": len(subset),
            "paper_count": len(paper_counts),
            "largest_paper_fraction": max(paper_counts.values()) / len(subset),
            "top_matrix": matrices.most_common(1)[0][0] if matrices else "unreported",
            "top_matrix_fraction": matrices.most_common(1)[0][1] / len(subset) if matrices else 0,
            "top_filler": fillers.most_common(1)[0][0] if fillers else "unreported",
            "top_filler_fraction": fillers.most_common(1)[0][1] / len(subset) if fillers else 0,
            "top_process_family": families.most_common(1)[0][0] if families else "unreported",
            "top_process_family_fraction": families.most_common(1)[0][1] / len(subset) if families else 0,
            "top_sequence": sequences.most_common(1)[0][0] if sequences else "",
            "top_sequence_fraction": sequences.most_common(1)[0][1] / len(subset) if sequences else 0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase3_run_config.json"))
    parser.add_argument("--records", type=Path, default=Path("data/nanomine_phase3_records.jsonl.gz"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = load_records(args.records)
    features = build_feature_records(records)
    facets, pair_i, pair_j = compute_facets(features)
    pair_meta = pair_metadata(records, pair_i, pair_j)
    papers = np.asarray([record["paper_group"] for record in records], dtype=object)
    output = args.output_root / "results"
    output.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output / "facet_distances.npz",
        pair_i=pair_i,
        pair_j=pair_j,
        **{f"{facet}__{bound}": values[bound] for facet, values in facets.items() for bound in (*BOUNDS, "coverage")},
    )
    facet_audit = audit_facets(facets)
    facet_audit.to_csv(output / "facet_audit.csv", index=False)

    material_cache = {
        (name, bound): weighted_composite(facets, weights, bound)
        for name, weights in config["material_profiles"].items()
        for bound in BOUNDS
    }
    process_cache = {
        (name, bound): weighted_composite(facets, weights, bound)
        for name, weights in config["process_profiles"].items()
        for bound in BOUNDS
    }

    grid_rows = []
    assignments = {}
    clustering_cache: dict[str, dict[int, np.ndarray]] = {}
    for material_name in config["material_profiles"]:
        for process_name in config["process_profiles"]:
            for process_lambda in config["process_lambdas"]:
                for bound in BOUNDS:
                    distance = blend(
                        material_cache[(material_name, bound)],
                        process_cache[(process_name, bound)],
                        process_lambda,
                    )
                    key_hash = array_hash(distance)
                    if key_hash not in clustering_cache:
                        clustering_cache[key_hash] = cluster_many(distance, config["cluster_counts"])
                    cuts = clustering_cache[key_hash]
                    uncertainty = blend(
                        material_cache[(material_name, "pessimistic")] - material_cache[(material_name, "optimistic")],
                        process_cache[(process_name, "pessimistic")] - process_cache[(process_name, "optimistic")],
                        process_lambda,
                    )
                    for k, labels in cuts.items():
                        key = (material_name, process_name, float(process_lambda), bound, int(k))
                        assignments[key] = labels
                        metrics = clustering_metrics(labels, papers, pair_i, pair_j, pair_meta)
                        grid_rows.append({
                            "material_profile": material_name,
                            "process_profile": process_name,
                            "process_lambda": process_lambda,
                            "bound": bound,
                            "requested_clusters": k,
                            "mean_pair_distance": float(np.mean(distance)),
                            "mean_same_paper_distance": float(np.mean(distance[pair_meta["same_paper"]])),
                            "mean_cross_paper_distance": float(np.mean(distance[pair_meta["cross_paper"]])),
                            "mean_interval_width": float(np.mean(uncertainty)),
                            **metrics,
                        })
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(output / "phase3_grid.csv", index=False)

    stability_rows = []
    for material_name in config["material_profiles"]:
        for process_name in config["process_profiles"]:
            for process_lambda in config["process_lambdas"]:
                for k in config["cluster_counts"]:
                    opt = assignments[(material_name, process_name, float(process_lambda), "optimistic", k)]
                    rep = assignments[(material_name, process_name, float(process_lambda), "reported", k)]
                    pes = assignments[(material_name, process_name, float(process_lambda), "pessimistic", k)]
                    stability_rows.append({
                        "material_profile": material_name,
                        "process_profile": process_name,
                        "process_lambda": process_lambda,
                        "requested_clusters": k,
                        "ari_optimistic_reported": adjusted_rand_score(opt, rep),
                        "ari_reported_pessimistic": adjusted_rand_score(rep, pes),
                        "ari_optimistic_pessimistic": adjusted_rand_score(opt, pes),
                    })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output / "bound_stability.csv", index=False)

    sentinel_lookup = {view["name"]: view for view in config["sentinel_views"]}
    sentinel_distances = {}
    retrieval_rows = []
    assignment_rows = []
    rng = np.random.default_rng(config["random_seed"])
    permutation_results = []
    profile_rows = []
    for name, view in sentinel_lookup.items():
        for bound in BOUNDS:
            distance = blend(
                material_cache[(view["material"], bound)],
                process_cache[(view["process"], bound)],
                view["lambda"],
            )
            sentinel_distances[f"{name}:{bound}"] = distance
            retrieval_rows.append({"view": name, "bound": bound, **retrieval_metrics(distance, records)})
            for k in (20, 50):
                labels = assignments[(view["material"], view["process"], float(view["lambda"]), bound, k)]
                for record, label in zip(records, labels):
                    assignment_rows.append({
                        "view": name, "bound": bound, "k": k,
                        "sample_id": record["sample_id"], "cluster": int(label),
                    })
                if bound == "reported":
                    permutation_results.append({
                        "view": name,
                        "k": k,
                        **permutation_test(labels, papers, pair_i, pair_j, config["permutation_repeats"], rng),
                    })
                    profile_rows.extend(cluster_profiles(records, labels, name, k))
    pd.DataFrame(retrieval_rows).to_csv(output / "sentinel_retrieval.csv", index=False)
    pd.DataFrame(permutation_results).to_csv(output / "label_permutation_tests.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(output / "sentinel_cluster_profiles.csv", index=False)
    with gzip.open(output / "sentinel_assignments.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in assignment_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    missing_distance = missing_mask_distance(features)
    missing_cuts = cluster_many(missing_distance, [20, 50])
    missing_rows = []
    for k, labels in missing_cuts.items():
        missing_rows.append({
            "view": "missing_mask_only",
            "k": k,
            **clustering_metrics(labels, papers, pair_i, pair_j, pair_meta),
            **retrieval_metrics(missing_distance, records),
        })
    pd.DataFrame(missing_rows).to_csv(output / "missingness_control.csv", index=False)

    family_rows = []
    for family in sorted({tuple(record.get("process_families", [])) for record in records}):
        indices = np.asarray([i for i, record in enumerate(records) if tuple(record.get("process_families", [])) == family])
        if len(indices) < 10:
            continue
        family_name = " | ".join(family)
        family_records = [records[i] for i in indices]
        family_i, family_j = np.triu_indices(len(indices), 1)
        family_meta = pair_metadata(family_records, family_i, family_j)
        family_papers = np.asarray([record["paper_group"] for record in family_records], dtype=object)
        for name, view in sentinel_lookup.items():
            for bound in BOUNDS:
                full_distance = sentinel_distances[f"{name}:{bound}"]
                distance = square_subset(full_distance, indices)
                cuts = cluster_many(distance, config["family_cluster_counts"])
                for k, labels in cuts.items():
                    family_rows.append({
                        "process_family": family_name,
                        "samples": len(indices),
                        "view": name,
                        "bound": bound,
                        "requested_clusters": k,
                        **clustering_metrics(labels, family_papers, family_i, family_j, family_meta),
                    })
    pd.DataFrame(family_rows).to_csv(output / "phase3_family_grid.csv", index=False)

    candidates = consensus_pairs(records, sentinel_distances, config["consensus_k_neighbors"])
    candidates.to_csv(output / "consensus_cross_paper_pairs.csv", index=False)

    summary = {
        "samples": len(records),
        "papers": len(set(papers)),
        "same_paper_pairs": int(pair_meta["same_paper"].sum()),
        "shared_author_cross_paper_pairs": int(pair_meta["shared_author_cross_paper"].sum()),
        "same_location_cross_paper_pairs": int(pair_meta["same_location_cross_paper"].sum()),
        "global_clusterings": len(grid),
        "family_clusterings": len(family_rows),
        "distance_views": len(config["material_profiles"]) * len(config["process_profiles"]) * len(config["process_lambdas"]) * len(BOUNDS),
        "unique_distance_clusterings": len(clustering_cache),
        "reported_lift_distribution": {
            "min": float(grid.loc[grid.bound == "reported", "same_paper_recall_lift"].min()),
            "median": float(grid.loc[grid.bound == "reported", "same_paper_recall_lift"].median()),
            "max": float(grid.loc[grid.bound == "reported", "same_paper_recall_lift"].max()),
        },
        "bound_stability": {
            "median_optimistic_pessimistic_ari": float(stability["ari_optimistic_pessimistic"].median()),
            "min_optimistic_pessimistic_ari": float(stability["ari_optimistic_pessimistic"].min()),
            "max_optimistic_pessimistic_ari": float(stability["ari_optimistic_pessimistic"].max()),
        },
    }
    (output / "phase3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
