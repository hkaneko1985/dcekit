#!/usr/bin/env python3
"""Run the preregistered property-free clustering pilot on Starrydata samples."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from phase1_core import (
    bootstrap_log_lift,
    categorical_distance,
    category_from_entry,
    cluster_grid,
    composition_bin_edges,
    composition_distance_matrix,
    composition_matrices,
    evaluate_coassignment,
    internal_cluster_metrics,
    mix_distances,
    normalize_key,
    numeric_distance,
    paper_equal_log_lift,
    parse_formula,
    parse_grain_size_um,
    parse_relative_density,
    percentile_interval,
    random_unlabeled_pairs,
    same_group_pairs,
    select_cluster_count,
    stable_hash,
    stratified_permutation_pvalue,
    subsample_stability,
    leave_one_group_out_delta,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_category(info: dict[str, Any], normalized_key: str) -> str | None:
    for key, entry in info.items():
        if normalize_key(key) == normalized_key:
            return category_from_entry(entry)
    return None


def get_prefix_categories(info: dict[str, Any], prefix: str) -> set[str]:
    output: set[str] = set()
    for key, entry in info.items():
        if normalize_key(key).startswith(prefix):
            value = category_from_entry(entry)
            if value:
                for token in value.split("|"):
                    normalized = normalize_key(token)
                    if normalized and normalized not in {"unknown", "other"}:
                        output.add(normalized)
    return output


def normalized_singleton(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = normalize_key(value)
    aliases = {
        "pellets": "pellet",
        "discs": "disc",
        "films": "film",
        "powders": "powder",
        "singlecrystalline": "singlecrystal",
        "polycrystalline": "polycrystal",
    }
    normalized = aliases.get(normalized, normalized)
    return {normalized} if normalized not in {"unknown", "other"} else set()


def extract_rows(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    counters = {
        "source_rows": int(len(frame)),
        "json_error": 0,
        "formula_rejected": 0,
        "explicit_nonexperiment_rejected": 0,
        "missing_form_and_fabrication": 0,
    }
    for row in frame.itertuples(index=False):
        try:
            info = json.loads(row.sample_info) if row.sample_info else {}
        except Exception:
            counters["json_error"] += 1
            continue
        if not isinstance(info, dict):
            counters["json_error"] += 1
            continue
        composition = parse_formula(row.composition)
        if composition is None:
            counters["formula_rejected"] += 1
            continue
        data_type = get_category(info, "datatype")
        if data_type and any(term in data_type.casefold() for term in ("reference", "calculation", "theory")):
            counters["explicit_nonexperiment_rejected"] += 1
            continue
        form = normalized_singleton(get_category(info, "form"))
        fabrication = get_prefix_categories(info, "fabricationprocess")
        if not form and not fabrication:
            counters["missing_form_and_fabrication"] += 1
            continue
        purity = normalized_singleton(get_category(info, "purity"))
        density_category = get_category(info, "relativedensity")
        grain_category = get_category(info, "grainsize")
        records.append({
            "SID": str(row.SID),
            "sample_id": str(row.sample_id),
            "DOI": str(row.DOI),
            "composition_raw": str(row.composition),
            "composition": composition,
            "form": form,
            "fabrication": fabrication,
            "purity": purity,
            "relative_density": parse_relative_density(density_category),
            "grain_size_log10_um": parse_grain_size_um(grain_category),
        })
    counters["eligible_before_sid_filter"] = len(records)
    return records, counters


def select_pilot(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(config["seed"])
    cohort = config["cohort"]
    minimum = int(cohort["minimum_samples_per_sid_before_sampling"])
    cap = int(cohort["maximum_samples_per_sid"])
    target = int(cohort["pilot_target_samples"])
    by_sid: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_sid.setdefault(record["SID"], []).append(record)
    eligible_groups = {sid: values for sid, values in by_sid.items() if len(values) >= minimum}
    selected: list[dict[str, Any]] = []
    selected_sids = []
    for sid in sorted(eligible_groups, key=lambda x: stable_hash(x, seed)):
        group = sorted(
            eligible_groups[sid],
            key=lambda x: stable_hash(f"{x['SID']}|{x['sample_id']}", seed),
        )[:cap]
        selected.extend(group)
        selected_sids.append(sid)
        if len(selected) >= target:
            break
    # Linkage implementations can break exact-distance ties by input order.
    # Randomize row order by an independent cryptographic hash so paper-contiguous
    # source order cannot masquerade as scientific similarity.
    selected = sorted(
        selected,
        key=lambda x: stable_hash(f"row-order|{x['SID']}|{x['sample_id']}", seed + 991),
    )
    return selected, {
        "eligible_sid_groups_with_minimum": len(eligible_groups),
        "selected_sid_groups": len(selected_sids),
        "selected_samples": len(selected),
        "per_sid_cap": cap,
        "target_samples": target,
        "row_order": "source-independent SHA-256 hash of (SID, sample_id)",
    }


def clean_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"positive_same", "expected_per_positive"}}


def nearest_neighbor_lift(distance: np.ndarray, sids: np.ndarray, indices: np.ndarray, ks=(1, 5, 10)) -> dict[str, Any]:
    local = distance[np.ix_(indices, indices)].copy()
    np.fill_diagonal(local, np.inf)
    order = np.argsort(local, axis=1)
    local_sids = sids[indices]
    counts = pd.Series(local_sids).value_counts()
    random_rate = float(np.mean([(counts[sid] - 1) / (len(local_sids) - 1) for sid in local_sids]))
    output = {}
    for k in ks:
        neighbors = order[:, :k]
        hits = local_sids[neighbors] == local_sids[:, None]
        rate = float(hits.mean())
        output[str(k)] = {
            "known_positive_neighbor_rate": rate,
            "random_pair_rate": random_rate,
            "lift": rate / max(random_rate, 1e-12),
        }
    return output


def paper_dominance(pair_groups: np.ndarray, flags_a: np.ndarray, flags_b: np.ndarray) -> dict[str, float]:
    contributions = []
    for group in np.unique(pair_groups):
        mask = pair_groups == group
        contributions.append(float((flags_a[mask].astype(float) - flags_b[mask].astype(float)).sum()))
    absolute = np.abs(contributions)
    total = float(absolute.sum())
    return {
        "maximum_absolute_increment_share": float(absolute.max() / total) if total else 0.0,
        "papers_with_nonzero_increment": int((absolute > 0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--seed-override", type=int, help="Post-hoc robustness runs only; the primary run uses the frozen config seed.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed_override is not None:
        config["seed"] = int(args.seed_override)
    seed = int(config["seed"])
    input_path = Path(config["input"]["samples_path"])
    digest = sha256_file(input_path)
    if digest != config["input"]["expected_sha256"]:
        raise ValueError(f"Input SHA-256 mismatch: {digest}")

    start = time.time()
    usecols = ["SID", "sample_id", "DOI", "composition", "sample_info"]
    frame = pd.read_csv(input_path, compression="gzip", dtype=str, keep_default_na=False, usecols=usecols)
    if set(frame.columns) != set(usecols):
        raise ValueError("Unexpected samples schema")
    records, exclusions = extract_rows(frame)
    pilot, selection = select_pilot(records, config)
    del frame, records

    n = len(pilot)
    sids = np.asarray([record["SID"] for record in pilot], dtype=object)
    sid_hash_fraction = np.asarray([(stable_hash(sid, seed + 1) % 10_000_000) / 10_000_000 for sid in sids])
    development = np.flatnonzero(sid_hash_fraction < float(config["split"]["development_fraction_by_sid_hash"]))
    confirmation = np.flatnonzero(sid_hash_fraction >= float(config["split"]["development_fraction_by_sid_hash"]))
    if len(np.unique(sids[development])) + len(np.unique(sids[confirmation])) != len(np.unique(sids)):
        raise AssertionError("SID leakage across development and confirmation partitions")

    exact, groups, periods, element_binary = composition_matrices([record["composition"] for record in pilot])
    composition_distance = composition_distance_matrix(exact, groups, periods, element_binary)
    form_distance = categorical_distance([record["form"] for record in pilot])
    fabrication_distance = categorical_distance([record["fabrication"] for record in pilot])
    purity_distance = categorical_distance([record["purity"] for record in pilot])
    density_distance = numeric_distance(np.asarray([record["relative_density"] for record in pilot]))
    grain_distance = numeric_distance(np.asarray([record["grain_size_log10_um"] for record in pilot]))

    distances = {
        "F0": composition_distance,
        "F1": mix_distances(composition_distance, [form_distance]),
        "F2": mix_distances(composition_distance, [form_distance, fabrication_distance]),
        "F3": mix_distances(composition_distance, [form_distance, fabrication_distance, purity_distance, density_distance, grain_distance]),
    }

    grid = [int(value) for value in config["clustering"]["cluster_count_grid"]]
    cluster_results: dict[str, Any] = {}
    labels_all: dict[str, dict[int, np.ndarray]] = {}
    for feature_set, distance in distances.items():
        _, labels_grid = cluster_grid(distance, grid)
        metrics = internal_cluster_metrics(distance, labels_grid, development, seed)
        selected_k = select_cluster_count(metrics)
        labels_all[feature_set] = labels_grid
        cluster_results[feature_set] = {
            "internal_grid": metrics,
            "selected_clusters": selected_k,
            "selection_used_proxy_labels": False,
        }

    local_sid = sids[confirmation]
    pos_i_local, pos_j_local, pair_groups = same_group_pairs(local_sid)
    pos_i = confirmation[pos_i_local]
    pos_j = confirmation[pos_j_local]
    pair_target = min(
        int(config["evaluation"]["unlabeled_pair_cap"]),
        int(config["evaluation"]["unlabeled_pair_multiplier"]) * len(pos_i_local),
    )
    unl_i_local, unl_j_local = random_unlabeled_pairs(local_sid, pair_target, seed + 2)
    unl_i = confirmation[unl_i_local]
    unl_j = confirmation[unl_j_local]
    pos_comp = composition_distance[pos_i, pos_j]
    unl_comp = composition_distance[unl_i, unl_j]
    edges = composition_bin_edges(unl_comp, int(config["evaluation"]["composition_distance_bins"]))
    exact_tolerance = float(config["evaluation"]["exact_composition_distance_tolerance"])
    nonexact = pos_comp > exact_tolerance

    composition_embedding = np.concatenate([groups, periods, exact], axis=1)
    stratum_count = min(30, max(2, len(confirmation) // 40))
    strata = KMeans(n_clusters=stratum_count, random_state=seed, n_init=10).fit_predict(composition_embedding[confirmation])

    evaluation: dict[str, Any] = {}
    bootstrap_samples: dict[str, np.ndarray] = {}
    resolution_lifts: dict[str, dict[str, float]] = {}
    for feature_set, by_k in labels_all.items():
        selected_k = cluster_results[feature_set]["selected_clusters"]
        selected_labels = by_k[selected_k]
        selected_eval = evaluate_coassignment(
            selected_labels,
            pos_i,
            pos_j,
            unl_i,
            unl_j,
            pos_comp,
            unl_comp,
            edges,
        )
        boot = bootstrap_log_lift(
            selected_eval["positive_same"],
            selected_eval["expected_per_positive"],
            pair_groups,
            int(config["evaluation"]["paper_block_bootstrap_repetitions"]),
            seed + 100,
        )
        bootstrap_samples[feature_set] = boot
        nonexact_eval = evaluate_coassignment(
            selected_labels,
            pos_i[nonexact],
            pos_j[nonexact],
            unl_i,
            unl_j,
            pos_comp[nonexact],
            unl_comp,
            edges,
        )
        permutation_p, permutation_interval = stratified_permutation_pvalue(
            selected_labels[confirmation],
            strata,
            pos_i_local,
            pos_j_local,
            int(config["evaluation"]["stratified_permutation_repetitions"]),
            seed + 200 + selected_k,
        )
        stability_n = min(int(config["evaluation"]["stability_subset_samples"]), len(development))
        stability_indices = development[np.argsort([stable_hash(f"stab|{i}", seed) for i in development])[:stability_n]]
        stability_k = max(5, int(round(selected_k * stability_n / n)))
        stability = subsample_stability(
            distance=distances[feature_set],
            reference_labels=selected_labels,
            cluster_count=stability_k,
            subset_indices=stability_indices,
            repetitions=int(config["evaluation"]["stability_repetitions"]),
            seed=seed + 300 + selected_k,
        )
        resolution_lifts[feature_set] = {}
        for k, labels in by_k.items():
            at_k = evaluate_coassignment(labels, pos_i, pos_j, unl_i, unl_j, pos_comp, unl_comp, edges)
            resolution_lifts[feature_set][str(k)] = float(at_k["lift"])
        evaluation[feature_set] = {
            "selected_resolution": clean_evaluation(selected_eval),
            "log_lift_bootstrap_95_interval": percentile_interval(boot),
            "nonexact_composition": clean_evaluation(nonexact_eval),
            "composition_stratified_permutation_pvalue": permutation_p,
            "permutation_positive_rate_95_interval": permutation_interval,
            "nearest_neighbor": nearest_neighbor_lift(distances[feature_set], sids, confirmation),
            "stability": {
                "subset_samples": stability_n,
                "clusters": stability_k,
                "repetitions": len(stability),
                "ari_values": stability,
                "median_ari": float(np.median(stability)),
                "ari_95_interval": percentile_interval(np.asarray(stability)),
            },
            "fixed_resolution_lift": resolution_lifts[feature_set],
        }

    comparisons = {}
    for feature_set in ("F1", "F2", "F3"):
        delta = bootstrap_samples[feature_set] - bootstrap_samples["F0"]
        comparisons[f"{feature_set}_minus_F0"] = {
            "observed_delta_log_lift": float(
                evaluation[feature_set]["selected_resolution"]["log_lift"]
                - evaluation["F0"]["selected_resolution"]["log_lift"]
            ),
            "bootstrap_95_interval": percentile_interval(delta),
            "bootstrap_probability_above_zero": float((delta > 0).mean()),
            "fixed_resolutions_with_positive_delta": int(sum(
                resolution_lifts[feature_set][str(k)] > resolution_lifts["F0"][str(k)] for k in grid
            )),
            "fixed_resolutions_total": len(grid),
        }

    for feature_set, reference in (("F2", "F1"), ("F3", "F2")):
        delta = bootstrap_samples[feature_set] - bootstrap_samples[reference]
        comparisons[f"{feature_set}_minus_{reference}"] = {
            "observed_delta_log_lift": float(
                evaluation[feature_set]["selected_resolution"]["log_lift"]
                - evaluation[reference]["selected_resolution"]["log_lift"]
            ),
            "bootstrap_95_interval": percentile_interval(delta),
            "bootstrap_probability_above_zero": float((delta > 0).mean()),
            "interpretation": "Increment attributable to the newly added metadata block under the selected resolutions.",
        }

    selected_f2_labels = labels_all["F2"][cluster_results["F2"]["selected_clusters"]]
    selected_f0_labels = labels_all["F0"][cluster_results["F0"]["selected_clusters"]]
    f2_eval_raw = evaluate_coassignment(selected_f2_labels, pos_i, pos_j, unl_i, unl_j, pos_comp, unl_comp, edges)
    f0_eval_raw = evaluate_coassignment(selected_f0_labels, pos_i, pos_j, unl_i, unl_j, pos_comp, unl_comp, edges)
    dominance = paper_dominance(pair_groups, f2_eval_raw["positive_same"], f0_eval_raw["positive_same"])
    paper_equal = {
        "F0": paper_equal_log_lift(f0_eval_raw["positive_same"], f0_eval_raw["expected_per_positive"], pair_groups),
        "F2": paper_equal_log_lift(f2_eval_raw["positive_same"], f2_eval_raw["expected_per_positive"], pair_groups),
    }
    paper_equal["F2_minus_F0_delta_log_lift"] = (
        paper_equal["F2"]["paper_equal_log_lift"] - paper_equal["F0"]["paper_equal_log_lift"]
    )
    leave_one_out = leave_one_group_out_delta(
        f2_eval_raw["positive_same"],
        f2_eval_raw["expected_per_positive"],
        f0_eval_raw["positive_same"],
        f0_eval_raw["expected_per_positive"],
        pair_groups,
    )

    masks = np.column_stack([
        np.asarray([bool(record["form"]) for record in pilot], dtype=float),
        np.asarray([bool(record["fabrication"]) for record in pilot], dtype=float),
        np.asarray([bool(record["purity"]) for record in pilot], dtype=float),
        np.asarray([np.isfinite(record["relative_density"]) for record in pilot], dtype=float),
        np.asarray([np.isfinite(record["grain_size_log10_um"]) for record in pilot], dtype=float),
    ])
    complete_global = confirmation[(masks[confirmation, 0] == 1) & (masks[confirmation, 1] == 1)]
    complete_sids = sids[complete_global]
    cc_i_local, cc_j_local, cc_groups = same_group_pairs(complete_sids)
    cc_i = complete_global[cc_i_local]
    cc_j = complete_global[cc_j_local]
    cc_unl_target = min(
        int(config["evaluation"]["unlabeled_pair_cap"]),
        int(config["evaluation"]["unlabeled_pair_multiplier"]) * len(cc_i),
    )
    cc_unl_i_local, cc_unl_j_local = random_unlabeled_pairs(complete_sids, cc_unl_target, seed + 600)
    cc_unl_i = complete_global[cc_unl_i_local]
    cc_unl_j = complete_global[cc_unl_j_local]
    cc_pos_comp = composition_distance[cc_i, cc_j]
    cc_unl_comp = composition_distance[cc_unl_i, cc_unl_j]
    cc_edges = composition_bin_edges(cc_unl_comp, int(config["evaluation"]["composition_distance_bins"]))
    complete_case = {
        "samples": int(len(complete_global)),
        "papers": int(len(np.unique(complete_sids))),
        "same_paper_pairs": int(len(cc_i)),
        "unlabeled_pairs": int(len(cc_unl_i)),
        "restriction": "Every sample has both Form and FabricationProcess, so F2 block missingness is constant.",
        "features": {},
    }
    cc_boot = {}
    for feature_set in ("F0", "F1", "F2", "F3"):
        labels = labels_all[feature_set][cluster_results[feature_set]["selected_clusters"]]
        cc_eval = evaluate_coassignment(
            labels,
            cc_i,
            cc_j,
            cc_unl_i,
            cc_unl_j,
            cc_pos_comp,
            cc_unl_comp,
            cc_edges,
        )
        cc_boot[feature_set] = bootstrap_log_lift(
            cc_eval["positive_same"],
            cc_eval["expected_per_positive"],
            cc_groups,
            int(config["evaluation"]["paper_block_bootstrap_repetitions"]),
            seed + 700,
        )
        complete_case["features"][feature_set] = {
            **clean_evaluation(cc_eval),
            "log_lift_bootstrap_95_interval": percentile_interval(cc_boot[feature_set]),
        }
    cc_delta = cc_boot["F2"] - cc_boot["F0"]
    complete_case["F2_minus_F0"] = {
        "observed_delta_log_lift": float(
            complete_case["features"]["F2"]["log_lift"] - complete_case["features"]["F0"]["log_lift"]
        ),
        "bootstrap_95_interval": percentile_interval(cc_delta),
        "bootstrap_probability_above_zero": float((cc_delta > 0).mean()),
    }

    exact_positive_mask = pos_comp <= exact_tolerance
    exact_groups: dict[tuple[tuple[str, float], ...], list[int]] = {}
    for global_index in confirmation:
        key = tuple(sorted((symbol, round(float(fraction), 10)) for symbol, fraction in pilot[global_index]["composition"].items()))
        exact_groups.setdefault(key, []).append(int(global_index))
    exact_unlabeled_i: list[int] = []
    exact_unlabeled_j: list[int] = []
    for indices in exact_groups.values():
        if len(indices) < 2:
            continue
        for i, j in itertools.combinations(indices, 2):
            if sids[i] != sids[j]:
                exact_unlabeled_i.append(i)
                exact_unlabeled_j.append(j)
    exact_unl_i = np.asarray(exact_unlabeled_i, dtype=np.int32)
    exact_unl_j = np.asarray(exact_unlabeled_j, dtype=np.int32)
    exact_unl_comp = composition_distance[exact_unl_i, exact_unl_j]
    exact_composition = {
        "same_paper_pairs": int(exact_positive_mask.sum()),
        "unlabeled_pairs": int(len(exact_unl_i)),
        "restriction": "Both members have identical normalized elemental fractions; different-SID pairs remain unlabeled, not negative.",
        "features": {},
    }
    exact_boot = {}
    exact_edges = np.asarray([-np.inf, np.inf])
    for feature_set in ("F0", "F1", "F2", "F3"):
        labels = labels_all[feature_set][cluster_results[feature_set]["selected_clusters"]]
        exact_eval = evaluate_coassignment(
            labels,
            pos_i[exact_positive_mask],
            pos_j[exact_positive_mask],
            exact_unl_i,
            exact_unl_j,
            pos_comp[exact_positive_mask],
            exact_unl_comp,
            exact_edges,
        )
        exact_boot[feature_set] = bootstrap_log_lift(
            exact_eval["positive_same"],
            exact_eval["expected_per_positive"],
            pair_groups[exact_positive_mask],
            int(config["evaluation"]["paper_block_bootstrap_repetitions"]),
            seed + 800,
        )
        exact_composition["features"][feature_set] = {
            **clean_evaluation(exact_eval),
            "log_lift_bootstrap_95_interval": percentile_interval(exact_boot[feature_set]),
        }
    exact_delta = exact_boot["F2"] - exact_boot["F0"]
    exact_composition["F2_minus_F0"] = {
        "observed_delta_log_lift": float(
            exact_composition["features"]["F2"]["log_lift"]
            - exact_composition["features"]["F0"]["log_lift"]
        ),
        "bootstrap_95_interval": percentile_interval(exact_delta),
        "bootstrap_probability_above_zero": float((exact_delta > 0).mean()),
    }

    # There are at most 2^5 missingness signatures. Asking hierarchical
    # clustering for more clusters would split zero-distance ties arbitrarily;
    # use each exact observation signature as its own diagnostic cluster.
    _, missingness_labels = np.unique(masks, axis=0, return_inverse=True)
    missingness_eval = evaluate_coassignment(missingness_labels, pos_i, pos_j, unl_i, unl_j, pos_comp, unl_comp, edges)

    rng = np.random.default_rng(seed + 500)
    shuffled_process = [set(record["fabrication"]) for record in pilot]
    development_strata_full = KMeans(n_clusters=30, random_state=seed + 1, n_init=10).fit_predict(composition_embedding)
    for stratum in np.unique(development_strata_full):
        indices = np.flatnonzero(development_strata_full == stratum)
        permutation = rng.permutation(indices)
        original = [shuffled_process[index] for index in permutation]
        for index, value in zip(indices, original):
            shuffled_process[index] = value
    shuffled_fabrication_distance = categorical_distance(shuffled_process)
    shuffled_f2_distance = mix_distances(composition_distance, [form_distance, shuffled_fabrication_distance])
    _, shuffled_grid = cluster_grid(shuffled_f2_distance, [cluster_results["F2"]["selected_clusters"]])
    shuffled_labels = shuffled_grid[cluster_results["F2"]["selected_clusters"]]
    shuffled_eval = evaluate_coassignment(shuffled_labels, pos_i, pos_j, unl_i, unl_j, pos_comp, unl_comp, edges)

    criteria = config["go_criteria"]
    increment_pass = any(
        comparisons[f"{feature}_minus_F0"]["bootstrap_95_interval"][0] > 0
        for feature in ("F2", "F3")
    )
    nonexact_pass = any(
        evaluation[feature]["nonexact_composition"]["log_lift"]
        > evaluation["F0"]["nonexact_composition"]["log_lift"]
        for feature in ("F2", "F3")
    )
    resolution_pass = any(
        comparisons[f"{feature}_minus_F0"]["fixed_resolutions_with_positive_delta"] >= 4
        for feature in ("F2", "F3")
    )
    stability_pass = any(
        evaluation[feature]["stability"]["median_ari"] >= float(criteria["median_subsample_ari_at_least"])
        for feature in ("F2", "F3")
    )
    dominance_pass = dominance["maximum_absolute_increment_share"] < float(criteria["maximum_single_paper_absolute_increment_share_below"])
    missingness_pass = missingness_eval["log_lift"] < evaluation["F2"]["selected_resolution"]["log_lift"]
    gate_results = {
        "increment_ci": increment_pass,
        "nonexact_composition_direction": nonexact_pass,
        "four_of_five_fixed_resolutions": resolution_pass,
        "stability": stability_pass,
        "paper_dominance": dominance_pass,
        "missingness_only_below_F2": missingness_pass,
    }
    decision = "GO" if all(gate_results.values()) else "CONDITIONAL_GO" if increment_pass and nonexact_pass else "NO_GO"
    supplemental_resolution = {
        "complete_case_increment_ci_positive": complete_case["F2_minus_F0"]["bootstrap_95_interval"][0] > 0,
        "leave_one_paper_out_never_reverses_direction": leave_one_out["papers_whose_omission_reverses_direction"] == 0,
        "paper_equal_increment_positive": paper_equal["F2_minus_F0_delta_log_lift"] > 0,
    }
    phase2_recommendation = (
        "GO_TO_EXTERNAL_REPLICATION_WITH_CAVEAT"
        if decision == "CONDITIONAL_GO" and all(supplemental_resolution.values())
        else "GO_TO_EXTERNAL_REPLICATION" if decision == "GO"
        else "HOLD"
    )

    result = {
        "run_version": "1.0.0",
        "seed": seed,
        "seed_override_used": args.seed_override is not None,
        "decision_for_phase2": decision,
        "phase2_recommendation": phase2_recommendation,
        "claim_boundary": "Weak-context construct validation only; no property prediction, transferability, or expert-ground-truth claim.",
        "input": {
            "path": str(input_path),
            "sha256": digest,
            "rows": exclusions["source_rows"],
            "opened_columns": usecols,
            "opened_tables": ["samples"],
            "forbidden_tables_opened": [],
            "source_mode": config["input"]["source_mode"],
        },
        "cohort_exclusions": exclusions,
        "pilot_selection": selection,
        "partition": {
            "development_samples": int(len(development)),
            "confirmation_samples": int(len(confirmation)),
            "development_sids": int(len(np.unique(sids[development]))),
            "confirmation_sids": int(len(np.unique(sids[confirmation]))),
            "sid_overlap": int(len(set(sids[development]).intersection(set(sids[confirmation])))),
        },
        "feature_coverage_pilot": {
            "form": float(masks[:, 0].mean()),
            "fabrication": float(masks[:, 1].mean()),
            "purity": float(masks[:, 2].mean()),
            "relative_density_numeric": float(masks[:, 3].mean()),
            "grain_size_numeric": float(masks[:, 4].mean()),
        },
        "known_positive_evaluation": {
            "confirmation_same_sid_pairs": int(len(pos_i)),
            "confirmation_nonexact_composition_same_sid_pairs": int(nonexact.sum()),
            "matched_unlabeled_pairs": int(len(unl_i)),
            "interpretation": "Different-SID pairs are an unlabeled comparison pool, not negatives.",
        },
        "clustering": cluster_results,
        "evaluation": evaluation,
        "incremental_comparisons": comparisons,
        "falsification_controls": {
            "paper_dominance_F2_minus_F0": dominance,
            "paper_equal_weighting": paper_equal,
            "leave_one_paper_out_F2_minus_F0": leave_one_out,
            "complete_form_and_fabrication_case": complete_case,
            "exact_composition_pairs": exact_composition,
            "missingness_only": clean_evaluation(missingness_eval),
            "composition_stratified_process_shuffle_F2": clean_evaluation(shuffled_eval),
            "institution_support": {
                "status": "NOT_RUN_NONBLOCKING",
                "reason": "Institution metadata are absent from the samples table; DOI enrichment is deferred until after the primary model is frozen.",
            },
        },
        "phase2_gate_results": gate_results,
        "supplemental_resolution_of_dominance_caveat": supplemental_resolution,
        "runtime_seconds": float(time.time() - start),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.assignments.parent.mkdir(parents=True, exist_ok=True)
    development_set = set(development.tolist())
    with gzip.open(args.assignments, "wt", encoding="utf-8") as handle:
        for index, record in enumerate(pilot):
            row = {
                "SID": record["SID"],
                "sample_id": record["sample_id"],
                "composition": record["composition_raw"],
                "form_categories": sorted(record["form"]),
                "fabrication_categories": sorted(record["fabrication"]),
                "purity_categories": sorted(record["purity"]),
                "relative_density": record["relative_density"] if np.isfinite(record["relative_density"]) else None,
                "grain_size_log10_um": record["grain_size_log10_um"] if np.isfinite(record["grain_size_log10_um"]) else None,
                "partition": "development" if index in development_set else "confirmation",
            }
            for feature_set in ("F0", "F1", "F2", "F3"):
                k = cluster_results[feature_set]["selected_clusters"]
                row[f"{feature_set}_cluster"] = int(labels_all[feature_set][k][index])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "decision_for_phase2": decision,
        "phase2_recommendation": phase2_recommendation,
        "pilot_samples": n,
        "confirmation_pairs": len(pos_i),
        "F0_lift": evaluation["F0"]["selected_resolution"]["lift"],
        "F2_lift": evaluation["F2"]["selected_resolution"]["lift"],
        "F3_lift": evaluation["F3"]["selected_resolution"]["lift"],
        "gate_results": gate_results,
        "runtime_seconds": result["runtime_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
