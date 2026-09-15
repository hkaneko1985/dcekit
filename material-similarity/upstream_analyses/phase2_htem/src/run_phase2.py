#!/usr/bin/env python3
"""Execute revised HTEM Phase 2 without property targets or imputation."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.metrics import adjusted_rand_score

from phase2_interval_core import (
    FACETS,
    FIELD_SPECS,
    build_field_matrices,
    cluster_grid,
    composition_distance,
    composition_summary,
    evaluate_labels,
    json_safe,
    material_distance,
    process_interval_distances,
    prepare_evaluation_design,
    random_unlabeled_pairs,
    reporting_mask_distance,
    retrieval_metrics,
    row_order_stability,
    study_positive_instances,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def load_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def config_id(profile: str, variant: str, process_weight: float) -> str:
    return f"{profile}__{variant}__L{int(round(process_weight * 100)):03d}"


def metric_value(evaluation: dict[str, Any], key: str) -> float:
    value = evaluation["all"].get(key)
    return float(value) if value is not None else float("nan")


def process_matrix_for(
    profile: str,
    variant: str,
    profiles: dict[str, list[float]],
    field_distances: dict[str, np.ndarray],
    field_coverage: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    distances, coverage, _ = process_interval_distances(
        field_distances, field_coverage, profiles[profile]
    )
    return distances[variant], coverage


def stratified_permutation(strata: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(strata):
        groups[str(key)].append(index)
    permutation = np.arange(len(strata))
    for indices in groups.values():
        if len(indices) >= 2:
            target = np.asarray(indices, dtype=int)
            permutation[target] = rng.permutation(target)
    return permutation, float(np.mean(permutation != np.arange(len(permutation))))


def pair_source_strata(instruments: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = []
    for first, second in zip(left, right):
        a, b = str(instruments[first]), str(instruments[second])
        if a == b:
            output.append(f"same:{a}")
        else:
            output.append("different:" + "|".join(sorted((a, b))))
    return np.asarray(output, dtype=object)


def pair_detail(
    left: int,
    right: int,
    ids: np.ndarray,
    element_keys: np.ndarray,
    instruments: np.ndarray,
    composition: np.ndarray,
    process_distance: np.ndarray,
    process_coverage: np.ndarray,
    field_distances: dict[str, np.ndarray],
    field_coverage: dict[str, np.ndarray],
) -> dict[str, Any]:
    shared = []
    for field in FIELD_SPECS:
        q = float(field_coverage[field][left, right])
        if q > 0:
            shared.append({
                "field": field,
                "slot_coverage": q,
                "conditional_similarity": 1.0 - float(field_distances[field][left, right]),
            })
    shared.sort(key=lambda row: (-row["slot_coverage"], -row["conditional_similarity"], row["field"]))
    return {
        "sample_library_id_1": int(ids[left]),
        "sample_library_id_2": int(ids[right]),
        "element_system_1": str(element_keys[left]),
        "element_system_2": str(element_keys[right]),
        "composition_distance": float(composition[left, right]),
        "process_reported_distance_EQ": float(process_distance[left, right]),
        "process_reporting_coverage_EQ": float(process_coverage[left, right]),
        "same_instrument_validation_only": bool(instruments[left] == instruments[right] and instruments[left] != "NOT_REPORTED"),
        "instrument_1_validation_only": str(instruments[left]),
        "instrument_2_validation_only": str(instruments[right]),
        "shared_process_fields": shared,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    config_path = args.config.resolve()
    root = config_path.parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = json.loads(resolve(root, config["paths"]["spec"]).read_text(encoding="utf-8"))
    records_path = resolve(root, config["paths"]["records"])
    studies_path = resolve(root, config["paths"]["studies"])
    manifest_path = resolve(root, config["paths"]["input_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_records = manifest["outputs"]["records"]["sha256"]
    expected_studies = manifest["outputs"]["studies"]["sha256"]
    if sha256(records_path) != expected_records or sha256(studies_path) != expected_studies:
        raise ValueError("Input checksum mismatch")

    records = load_records(records_path)
    records.sort(key=lambda row: int(row["sample_library_id"]))
    ids = np.asarray([int(row["sample_library_id"]) for row in records], dtype=np.int64)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("Duplicate sample library IDs")
    studies = json.loads(studies_path.read_text(encoding="utf-8"))
    id_to_index = {int(value): index for index, value in enumerate(ids)}
    instruments = np.asarray([
        str((row.get("validation_only") or {}).get("deposition_instrument") or "NOT_REPORTED")
        for row in records
    ], dtype=object)
    print(f"Loaded {len(records)} HTEM libraries; building composition distance", flush=True)
    summaries = [composition_summary(row) for row in records]
    composition, composition_info = composition_distance(summaries)
    element_keys = composition_info["element_keys"]

    print("Building 14 process-setting distance and reporting-coverage matrices", flush=True)
    field_distances, field_coverage, field_audit = build_field_matrices(records)
    mask_distance = reporting_mask_distance(field_coverage)
    profiles = {name: list(map(float, weights)) for name, weights in spec["process_weight_profiles"].items()}
    process_lambdas = [float(value) for value in spec["material_distance"]["composition_process_lambdas"]]
    positive_lambdas = [value for value in process_lambdas if value > 0]
    variants = list(spec["material_distance"]["process_variants"])
    cluster_counts = [int(value) for value in spec["clustering"]["cluster_resolutions"]]
    seed = int(config["seed"])
    evaluation_cfg = config["evaluation"]

    positives = study_positive_instances(studies, id_to_index)
    unlabeled_i, unlabeled_j = random_unlabeled_pairs(
        len(records), positives["unique_pairs"], int(evaluation_cfg["unlabeled_pair_cap"]), seed + 1
    )
    positive_c = composition[positives["pair_i"], positives["pair_j"]]
    base_subsets = {"nonidentical_composition": positive_c > float(evaluation_cfg["nonidentical_composition_tolerance"])}
    positive_source_strata = pair_source_strata(instruments, positives["pair_i"], positives["pair_j"])
    unlabeled_source_strata = pair_source_strata(instruments, unlabeled_i, unlabeled_j)
    evaluation_design = prepare_evaluation_design(
        positives, unlabeled_i, unlabeled_j, composition,
        int(evaluation_cfg["composition_distance_bins"]),
        positive_source_strata, unlabeled_source_strata,
    )

    def evaluate_current(labels: np.ndarray, subsets: dict[str, np.ndarray]) -> dict[str, Any]:
        return evaluate_labels(
            labels, positives, unlabeled_i, unlabeled_j, composition,
            int(evaluation_cfg["composition_distance_bins"]), subsets,
            positive_source_strata, unlabeled_source_strata,
            evaluation_design,
        )

    primary_seed = int(evaluation_cfg["row_order_seeds"][0])
    grid_rows: list[dict[str, Any]] = []
    labels_store: dict[tuple[str, str, float, int], np.ndarray] = {}
    distance_retrieval: dict[tuple[str, str, float], dict[str, Any]] = {}
    baseline_evaluations = {}
    print("Clustering composition baseline", flush=True)
    baseline_labels = cluster_grid(composition, cluster_counts, ids, primary_seed)
    baseline_retrieval = retrieval_metrics(composition, positives, int(evaluation_cfg["retrieval_top_k"]))
    for clusters, labels in baseline_labels.items():
        evaluation = evaluate_current(labels, base_subsets)
        baseline_evaluations[clusters] = evaluation
        grid_rows.append({
            "config_id": "COMPOSITION_ONLY", "profile": "COMPOSITION_ONLY", "variant": "none",
            "process_lambda": 0.0, "clusters": clusters, "process_positive_coverage_mean": 0.0,
            "positive_coassignment": metric_value(evaluation, "positive_coassignment"),
            "composition_matched_expected": metric_value(evaluation, "expected"),
            "excess": metric_value(evaluation, "excess"),
            "component_equal_excess": metric_value(evaluation, "component_equal_excess"),
            "source_matched_component_equal_excess": float(evaluation["all_source_matched"]["component_equal_excess"]),
            "delta_vs_composition_same_k": 0.0, "mask_control_component_excess": float("nan"),
            "delta_vs_mask_control": float("nan"), "mrr": baseline_retrieval["mrr"],
            "source_matched_delta_vs_composition_same_k": 0.0,
            "source_matched_mask_control_component_excess": float("nan"),
            "source_matched_delta_vs_mask_control": float("nan"),
            "mrr_delta_vs_composition": 0.0, "hit_at_10": baseline_retrieval["hit_at_10"],
            "recall_at_10": baseline_retrieval["recall_at_10"],
            "nonidentical_component_equal_excess": evaluation["subsets"]["nonidentical_composition"]["component_equal_excess"],
            "source_matched_nonidentical_component_equal_excess": evaluation["subsets_source_matched"]["nonidentical_composition"]["component_equal_excess"],
            "largest_cluster_fraction": evaluation["clusters"]["largest_fraction"],
            "singleton_fraction": evaluation["clusters"]["singleton_fraction"],
        })

    print("Clustering reporting-mask controls", flush=True)
    mask_control = {}
    mask_only_labels = cluster_grid(mask_distance, cluster_counts, ids, primary_seed)
    mask_only_evaluations = {clusters: evaluate_current(labels, base_subsets) for clusters, labels in mask_only_labels.items()}
    for process_lambda in positive_lambdas:
        distance = material_distance(composition, mask_distance, process_lambda)
        labels_by_k = cluster_grid(distance, cluster_counts, ids, primary_seed)
        for clusters, labels in labels_by_k.items():
            mask_control[(process_lambda, clusters)] = evaluate_current(labels, base_subsets)
        del distance

    total_configs = len(profiles) * len(variants) * len(positive_lambdas)
    completed = 0
    coverage_by_profile = {}
    for profile, weights in profiles.items():
        process_distances, process_coverage, field_weights = process_interval_distances(field_distances, field_coverage, weights)
        positive_q = process_coverage[positives["pair_i"], positives["pair_j"]]
        all_triangle = process_coverage[np.triu_indices(len(records), 1)]
        coverage_by_profile[profile] = {
            "facet_weights": dict(zip(FACETS, weights)),
            "field_weights": field_weights,
            "all_pair_mean": float(np.mean(all_triangle)),
            "all_pair_median": float(np.median(all_triangle)),
            "positive_pair_mean": float(np.mean(positive_q)),
            "positive_pair_median": float(np.median(positive_q)),
            "positive_pairs_zero": int(np.sum(positive_q == 0)),
        }
        subsets = {**base_subsets, "high_process_coverage": positive_q >= 0.5}
        for variant in variants:
            process_distance = process_distances[variant]
            for process_lambda in positive_lambdas:
                distance = material_distance(composition, process_distance, process_lambda)
                labels_by_k = cluster_grid(distance, cluster_counts, ids, primary_seed)
                retrieval = retrieval_metrics(distance, positives, int(evaluation_cfg["retrieval_top_k"]))
                distance_retrieval[(profile, variant, process_lambda)] = retrieval
                cid = config_id(profile, variant, process_lambda)
                for clusters, labels in labels_by_k.items():
                    labels_store[(profile, variant, process_lambda, clusters)] = labels
                    evaluation = evaluate_current(labels, subsets)
                    baseline_component = metric_value(baseline_evaluations[clusters], "component_equal_excess")
                    mask_eval = mask_control[(process_lambda, clusters)]
                    mask_component = metric_value(mask_eval, "component_equal_excess")
                    current_component = metric_value(evaluation, "component_equal_excess")
                    baseline_source_component = float(baseline_evaluations[clusters]["all_source_matched"]["component_equal_excess"])
                    mask_source_component = float(mask_eval["all_source_matched"]["component_equal_excess"])
                    current_source_component = float(evaluation["all_source_matched"]["component_equal_excess"])
                    grid_rows.append({
                        "config_id": cid, "profile": profile, "variant": variant,
                        "process_lambda": process_lambda, "clusters": clusters,
                        "process_positive_coverage_mean": float(np.mean(positive_q)),
                        "positive_coassignment": metric_value(evaluation, "positive_coassignment"),
                        "composition_matched_expected": metric_value(evaluation, "expected"),
                        "excess": metric_value(evaluation, "excess"),
                        "component_equal_excess": current_component,
                        "source_matched_component_equal_excess": current_source_component,
                        "delta_vs_composition_same_k": current_component - baseline_component,
                        "source_matched_delta_vs_composition_same_k": current_source_component - baseline_source_component,
                        "mask_control_component_excess": mask_component,
                        "delta_vs_mask_control": current_component - mask_component,
                        "source_matched_mask_control_component_excess": mask_source_component,
                        "source_matched_delta_vs_mask_control": current_source_component - mask_source_component,
                        "mrr": retrieval["mrr"],
                        "mrr_delta_vs_composition": retrieval["mrr"] - baseline_retrieval["mrr"],
                        "hit_at_10": retrieval["hit_at_10"],
                        "recall_at_10": retrieval["recall_at_10"],
                        "nonidentical_component_equal_excess": evaluation["subsets"]["nonidentical_composition"]["component_equal_excess"],
                        "source_matched_nonidentical_component_equal_excess": evaluation["subsets_source_matched"]["nonidentical_composition"]["component_equal_excess"],
                        "largest_cluster_fraction": evaluation["clusters"]["largest_fraction"],
                        "singleton_fraction": evaluation["clusters"]["singleton_fraction"],
                    })
                completed += 1
                if completed % 25 == 0 or completed == total_configs:
                    print(f"Completed {completed}/{total_configs} weighted distance configurations", flush=True)
                del distance, labels_by_k
        del process_distances, process_coverage
        gc.collect()

    # Rank configurations descriptively. This ranking is post-hoc and does not
    # redefine the frozen grid or declare a universal best metric.
    candidates = [row for row in grid_rows if row["config_id"] != "COMPOSITION_ONLY"]
    candidates.sort(key=lambda row: (
        -row["source_matched_delta_vs_composition_same_k"],
        -row["source_matched_delta_vs_mask_control"],
        -row["delta_vs_composition_same_k"],
        -row["mrr_delta_vs_composition"],
        row["largest_cluster_fraction"],
    ))
    top_rows = []
    used_profiles = set()
    for row in candidates:
        if row["profile"] in used_profiles or row["largest_cluster_fraction"] > 0.8:
            continue
        top_rows.append(row)
        used_profiles.add(row["profile"])
        if len(top_rows) >= int(evaluation_cfg["top_configurations_for_diagnostics"]):
            break
    if len(top_rows) < int(evaluation_cfg["top_configurations_for_diagnostics"]):
        for row in candidates:
            key = (row["profile"], row["variant"], row["process_lambda"], row["clusters"])
            if any((x["profile"], x["variant"], x["process_lambda"], x["clusters"]) == key for x in top_rows):
                continue
            top_rows.append(row)
            if len(top_rows) >= int(evaluation_cfg["top_configurations_for_diagnostics"]):
                break

    print("Running row-order, bound, and process-shuffle diagnostics", flush=True)
    diagnostics = []
    rng = np.random.default_rng(seed + 100)
    repetitions = int(evaluation_cfg["shuffle_repetitions"])
    element_permutations = [stratified_permutation(element_keys, rng) for _ in range(repetitions)]
    element_instrument_strata = np.asarray(
        [f"{element_keys[index]}||{instruments[index]}" for index in range(len(records))],
        dtype=object,
    )
    source_preserving_permutations = [
        stratified_permutation(element_instrument_strata, rng) for _ in range(repetitions)
    ]
    process_cache = {}
    for rank, row in enumerate(top_rows, start=1):
        profile, variant = row["profile"], row["variant"]
        process_lambda, clusters = float(row["process_lambda"]), int(row["clusters"])
        cache_key = (profile, variant)
        if cache_key not in process_cache:
            process_cache[cache_key] = process_matrix_for(profile, variant, profiles, field_distances, field_coverage)[0]
        process_distance = process_cache[cache_key]
        distance = material_distance(composition, process_distance, process_lambda)
        order_stability = row_order_stability(distance, ids, clusters, [int(value) for value in evaluation_cfg["row_order_seeds"]])
        bound_ari = {}
        base_labels = labels_store[(profile, variant, process_lambda, clusters)]
        for other in variants:
            other_labels = labels_store[(profile, other, process_lambda, clusters)]
            bound_ari[other] = float(adjusted_rand_score(base_labels, other_labels))
        lambda_ari = {}
        for other_lambda in positive_lambdas:
            other_labels = labels_store[(profile, variant, other_lambda, clusters)]
            lambda_ari[f"{other_lambda:g}"] = float(adjusted_rand_score(base_labels, other_labels))
        def run_shuffle(permutations: list[tuple[np.ndarray, float]]) -> dict[str, Any]:
            raw_values, source_values, moved_fractions = [], [], []
            for permutation, moved_fraction in permutations:
                shuffled_process = process_distance[np.ix_(permutation, permutation)]
                shuffled_distance = material_distance(composition, shuffled_process, process_lambda)
                shuffled_labels = cluster_grid(shuffled_distance, [clusters], ids, primary_seed)[clusters]
                evaluation = evaluate_current(shuffled_labels, base_subsets)
                raw_component = metric_value(evaluation, "component_equal_excess")
                source_component = float(evaluation["all_source_matched"]["component_equal_excess"])
                baseline_component = metric_value(baseline_evaluations[clusters], "component_equal_excess")
                baseline_source = float(baseline_evaluations[clusters]["all_source_matched"]["component_equal_excess"])
                raw_values.append(raw_component - baseline_component)
                source_values.append(source_component - baseline_source)
                moved_fractions.append(moved_fraction)
            raw_array = np.asarray(raw_values, dtype=float)
            source_array = np.asarray(source_values, dtype=float)
            observed_raw = float(row["delta_vs_composition_same_k"])
            observed_source = float(row["source_matched_delta_vs_composition_same_k"])
            return {
                "repetitions": len(raw_values),
                "mean_moved_fraction": float(np.mean(moved_fractions)),
                "raw_delta": {
                    "mean": float(np.mean(raw_array)),
                    "percentile_95": float(np.quantile(raw_array, 0.95)),
                    "maximum": float(np.max(raw_array)),
                    "plus_one_p_one_sided": float((1 + np.sum(raw_array >= observed_raw)) / (1 + len(raw_array))),
                },
                "source_matched_delta": {
                    "mean": float(np.mean(source_array)),
                    "percentile_95": float(np.quantile(source_array, 0.95)),
                    "maximum": float(np.max(source_array)),
                    "plus_one_p_one_sided": float((1 + np.sum(source_array >= observed_source)) / (1 + len(source_array))),
                },
            }
        element_shuffle = run_shuffle(element_permutations)
        source_preserving_shuffle = run_shuffle(source_preserving_permutations)
        diagnostics.append({
            "descriptive_rank": rank,
            "configuration": {key: row[key] for key in ["config_id", "profile", "variant", "process_lambda", "clusters"]},
            "observed": {key: row[key] for key in [
                "component_equal_excess", "delta_vs_composition_same_k", "delta_vs_mask_control",
                "source_matched_component_equal_excess", "source_matched_delta_vs_composition_same_k",
                "source_matched_delta_vs_mask_control", "mrr_delta_vs_composition"
            ]},
            "row_order_stability": order_stability,
            "ari_against_uncertainty_variants": bound_ari,
            "ari_across_process_lambdas": lambda_ari,
            "process_shuffle_within_element_system": element_shuffle,
            "process_shuffle_within_element_system_and_instrument": source_preserving_shuffle,
        })
        del distance

    # Instrument-only and mask-only diagnostics are deliberately not scientific
    # similarities. They estimate source/reporting-pattern confounding.
    _, instrument_labels = np.unique(instruments, return_inverse=True)
    instrument_evaluation = evaluate_current(instrument_labels, base_subsets)

    # Consensus material pairs across every prespecified nonzero-lambda grid
    # configuration at the fixed descriptive resolution k=150.
    consensus_k = 150
    label_vectors = [
        labels for (profile, variant, process_lambda, clusters), labels in labels_store.items()
        if clusters == consensus_k
    ]
    eq_reported, eq_coverage = process_matrix_for("EQ", "reported", profiles, field_distances, field_coverage)
    pi, pj = positives["pair_i"], positives["pair_j"]
    positive_consensus = np.zeros(len(pi), dtype=np.int32)
    for labels in label_vectors:
        positive_consensus += labels[pi] == labels[pj]
    positive_order = np.argsort(-positive_consensus)
    robust_positive_pairs = []
    seen_positive = set()
    for index in positive_order:
        pair = (int(min(pi[index], pj[index])), int(max(pi[index], pj[index])))
        if pair in seen_positive:
            continue
        seen_positive.add(pair)
        detail = pair_detail(*pair, ids, element_keys, instruments, composition, eq_reported, eq_coverage, field_distances, field_coverage)
        detail["coassignment_fraction_across_grid_at_k150"] = float(positive_consensus[index] / len(label_vectors))
        detail["support"] = "same HTEM study"
        robust_positive_pairs.append(detail)
        if len(robust_positive_pairs) >= int(evaluation_cfg["interesting_pairs_per_type"]):
            break

    triangle_i, triangle_j = np.triu_indices(len(records), 1)
    candidate_mask = (
        (composition[triangle_i, triangle_j] > float(evaluation_cfg["nonidentical_composition_tolerance"]))
        & (eq_coverage[triangle_i, triangle_j] >= 0.5)
    )
    candidate_i = triangle_i[candidate_mask]
    candidate_j = triangle_j[candidate_mask]
    if len(candidate_i) > 20000:
        base_score = material_distance(composition, eq_reported, 0.5)[candidate_i, candidate_j]
        keep = np.argpartition(base_score, 20000)[:20000]
        candidate_i, candidate_j = candidate_i[keep], candidate_j[keep]
    positive_unique = positives["unique_pairs"]
    consensus_counts = np.zeros(len(candidate_i), dtype=np.int32)
    for labels in label_vectors:
        consensus_counts += labels[candidate_i] == labels[candidate_j]
    ranking = np.lexsort((composition[candidate_i, candidate_j], -eq_coverage[candidate_i, candidate_j], -consensus_counts))
    cross_study_pairs = []
    cross_study_cross_instrument_pairs = []
    for index in ranking:
        pair = (int(candidate_i[index]), int(candidate_j[index]))
        if pair in positive_unique:
            continue
        detail = pair_detail(*pair, ids, element_keys, instruments, composition, eq_reported, eq_coverage, field_distances, field_coverage)
        detail["coassignment_fraction_across_grid_at_k150"] = float(consensus_counts[index] / len(label_vectors))
        detail["support"] = "cross-study hypothesis; not a validated positive"
        limit = int(evaluation_cfg["interesting_pairs_per_type"])
        if len(cross_study_pairs) < limit:
            cross_study_pairs.append(detail)
        if (
            len(cross_study_cross_instrument_pairs) < limit
            and detail["instrument_1_validation_only"] != detail["instrument_2_validation_only"]
            and "NOT_REPORTED" not in {detail["instrument_1_validation_only"], detail["instrument_2_validation_only"]}
        ):
            cross_study_cross_instrument_pairs.append(detail)
        if len(cross_study_pairs) >= limit and len(cross_study_cross_instrument_pairs) >= limit:
            break

    # Mechanically summarize clusters for the highest-ranked configuration.
    primary_row = top_rows[0]
    primary_key = (
        primary_row["profile"], primary_row["variant"],
        float(primary_row["process_lambda"]), int(primary_row["clusters"]),
    )
    primary_labels = labels_store[primary_key]
    positive_pair_set = positives["unique_pairs"]
    cluster_profiles = []
    for cluster in np.unique(primary_labels):
        members = np.where(primary_labels == cluster)[0]
        if len(members) < 3:
            continue
        member_set = set(map(int, members))
        supported_pairs = sum(left in member_set and right in member_set for left, right in positive_pair_set)
        element_counts = Counter(map(str, element_keys[members]))
        instrument_counts = Counter(map(str, instruments[members]))
        cluster_profiles.append({
            "cluster": int(cluster),
            "size": int(len(members)),
            "same_study_unique_pairs": int(supported_pairs),
            "top_element_systems": element_counts.most_common(5),
            "instrument_distribution_validation_only": instrument_counts.most_common(5),
            "sample_library_ids_first_30": [int(ids[index]) for index in members[:30]],
        })
    cluster_profiles.sort(key=lambda row: (-row["same_study_unique_pairs"], -row["size"], row["cluster"]))
    cluster_profiles = cluster_profiles[:30]

    # Context-supported patterns require stability and controls. The threshold
    # is descriptive, not a claim that the metric is universally valid.
    supported_diagnostics = []
    for row in diagnostics:
        observed = row["observed"]
        shuffle = row["process_shuffle_within_element_system_and_instrument"]["source_matched_delta"]
        if (
            observed["source_matched_delta_vs_composition_same_k"] > 0
            and observed["source_matched_delta_vs_mask_control"] > 0
            and row["row_order_stability"]["median_ari"] >= 0.70
            and observed["source_matched_delta_vs_composition_same_k"] > shuffle["percentile_95"]
        ):
            supported_diagnostics.append(row["descriptive_rank"])
    robust_cross_study = [row for row in cross_study_pairs if row["coassignment_fraction_across_grid_at_k150"] >= 0.8]
    robust_cross_instrument = [row for row in cross_study_cross_instrument_pairs if row["coassignment_fraction_across_grid_at_k150"] >= 0.8]
    if supported_diagnostics:
        status = "INTERESTING_SOURCE_ADJUSTED_CONTEXT_PATTERNS_FOUND"
    elif robust_cross_instrument:
        status = "INTERESTING_EXPLORATORY_PAIR_HYPOTHESES_FOUND"
    elif robust_cross_study:
        status = "SOURCE_CONFOUNDED_PAIR_HYPOTHESES_ONLY"
    else:
        status = "NO_CLEAR_INTERESTING_PATTERN_UNDER_FROZEN_CRITERIA"

    # Save complete grid.
    grid_path = resolve(root, config["paths"]["grid"])
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(grid_rows[0])
    with grid_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(grid_rows)

    assignment_rows = []
    selected_keys = []
    for row in top_rows:
        key = (row["profile"], row["variant"], float(row["process_lambda"]), int(row["clusters"]))
        selected_keys.append((row["config_id"], int(row["clusters"]), labels_store[key]))
    for index, library_id in enumerate(ids):
        assignment_rows.append({
            "sample_library_id": int(library_id),
            "element_system": str(element_keys[index]),
            "assignments": {f"{cid}__K{clusters}": int(labels[index]) for cid, clusters, labels in selected_keys},
        })
    assignment_payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in assignment_rows
    )
    assignments_path = resolve(root, config["paths"]["assignments"])
    assignments_path.write_bytes(deterministic_gzip(assignment_payload))

    # Top tables for easier inspection.
    pairs_path = root / "results" / "interesting_pairs.json"
    pairs_path.write_text(json.dumps({
        "same_study": robust_positive_pairs,
        "cross_study_hypotheses": cross_study_pairs,
        "cross_study_cross_instrument_hypotheses": cross_study_cross_instrument_pairs,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profiles_path = root / "results" / "selected_cluster_profiles.json"
    profiles_path.write_text(json.dumps({"configuration": {key: primary_row[key] for key in ["config_id", "profile", "variant", "process_lambda", "clusters"]}, "clusters": cluster_profiles}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "version": "3.1.0",
        "status": status,
        "scope": spec["scope"],
        "counts": {
            "sample_libraries": len(records),
            "quantitative_composition_libraries": composition_info["quantitative_count"],
            "studies_with_multiple_known_libraries": len(positives["members_by_study"]),
            "study_components": len(positives["components"]),
            "known_positive_pair_instances": len(positives["pair_i"]),
            "known_positive_unique_pairs": len(positives["unique_pairs"]),
            "known_positive_pair_instances_same_instrument": int(np.sum(np.char.startswith(positive_source_strata.astype(str), "same:"))),
            "known_positive_pair_instances_different_instrument": int(np.sum(np.char.startswith(positive_source_strata.astype(str), "different:"))),
            "unlabeled_comparison_pairs": len(unlabeled_i),
            "grid_distance_configurations": 1 + total_configs,
            "grid_clusterings": len(grid_rows),
        },
        "input_checksums": {"records": sha256(records_path), "studies": sha256(studies_path), "spec": sha256(resolve(root, config["paths"]["spec"]))},
        "field_audit": field_audit,
        "coverage_by_profile": coverage_by_profile,
        "composition_baseline": {
            "retrieval": baseline_retrieval,
            "by_resolution": {str(k): baseline_evaluations[k] for k in cluster_counts},
        },
        "reporting_mask_only_by_resolution": {str(k): mask_only_evaluations[k] for k in cluster_counts},
        "instrument_only_control": instrument_evaluation,
        "descriptively_ranked_configurations": top_rows,
        "diagnostics": diagnostics,
        "context_supported_diagnostic_ranks": supported_diagnostics,
        "consensus": {
            "resolution": consensus_k,
            "configurations": len(label_vectors),
            "robust_cross_study_hypotheses_at_least_80_percent": len(robust_cross_study),
            "robust_cross_instrument_hypotheses_at_least_80_percent": len(robust_cross_instrument),
            "same_study_pairs_file": str(pairs_path.relative_to(root)),
            "cross_study_pairs_file": str(pairs_path.relative_to(root)),
        },
        "decision_interpretation": {
            "universal_best_similarity_selected": False,
            "weight_or_resolution_optimized": False,
            "property_target_used": False,
            "expert_label_used": False,
            "missing_values_imputed": False,
            "different_study_pairs_called_negative": False,
            "meaning": spec["exploratory_judgment"]["rule"],
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "runtime_seconds": time.time() - started,
    }
    results_path = resolve(root, config["paths"]["results"])
    results_path.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Completed Phase 2: {status}; runtime {result['runtime_seconds']:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
