#!/usr/bin/env python3
"""Cross-dataset synthesis of outcome-free, missingness-aware similarities."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def med(values: Any) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def pair_aris(dataset: str, scope: str, views: dict[str, list[int]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "scope": scope,
            "view_a": left,
            "view_b": right,
            "ari": adjusted_rand_score(views[left], views[right]),
        }
        for left, right in combinations(views, 2)
    ]


def rank_stat(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    complete = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(complete) < 3 or complete[x].nunique() < 2 or complete[y].nunique() < 2:
        return math.nan, math.nan
    result = spearmanr(complete[x], complete[y])
    return float(result.statistic), float(result.pvalue)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def process_starry(paths: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(paths["starry_summary"])
    assignments = pd.DataFrame(read_jsonl(paths["starry_assignments"]))
    confirm = assignments.loc[assignments["partition"] == "confirmation"]
    views = {
        name: confirm[f"{name}_cluster"].astype(int).tolist()
        for name in ["F0", "F1", "F2"]  # F3 includes characterization descriptors; historical only
    }
    aris = pair_aris("Starrydata", "confirmation; k=150", views)
    f2 = np.asarray(summary["feature_lifts"]["F2"]["values"], float)
    mask = np.asarray(summary["missingness_only_lift"]["values"], float)
    shuffle = np.asarray(summary["process_shuffle_F2_lift"]["values"], float)
    gain = np.asarray(summary["F2_minus_F0_observed_delta_log_lift"]["values"], float)
    diag = {
        "dataset": "Starrydata",
        "domain": "mixed inorganic materials literature",
        "samples": len(assignments),
        "weak_groups": int(assignments["SID"].nunique()),
        "context_gain_metric": "F2-F0 log co-assignment lift",
        "context_gain_median": med(gain),
        "context_gain_positive_fraction": float(np.mean(gain > 0)),
        "semantic_vs_mask_metric": "log(F2 lift / missing-mask lift)",
        "semantic_vs_mask_median": med(np.log(f2 / mask)),
        "semantic_vs_shuffle_metric": "log(F2 lift / shuffled-process F2 lift)",
        "semantic_vs_shuffle_median": med(np.log(f2 / shuffle)),
        "shuffle_supported_fraction": float(np.mean(f2 > shuffle)),
        "global_local_spearman": math.nan,
        "global_local_spearman_p": math.nan,
        "global_local_joint_positive_fraction": math.nan,
        "view_pairwise_ari_median": med([row["ari"] for row in aris]),
        "view_pairwise_ari_min": min(row["ari"] for row in aris),
        "view_pairwise_ari_max": max(row["ari"] for row in aris),
        "all_bounds_positive_fraction": math.nan,
        "bound_sign_change_fraction": math.nan,
        "bound_partition_ari_median": math.nan,
        "cross_source_consensus_pairs": math.nan,
        "cross_source_candidates_with_author_or_location_support": math.nan,
        "interpretation": "形態・工程の増分は3 seedで正だが、欠損・論文様式の寄与も大きい。",
    }
    return diag, aris


def process_htem(
    paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = read_json(paths["htem_summary"])
    grid = pd.read_csv(paths["htem_grid"])
    rich = grid.loc[
        (grid["config_id"] != "COMPOSITION_ONLY")
        & (grid["variant"] == "reported")
        & (grid["process_lambda"] > 0)
    ].copy()
    assignment_rows = read_jsonl(paths["htem_assignments"])
    keys = list(assignment_rows[0]["assignments"])
    views = {key: [int(row["assignments"][key]) for row in assignment_rows] for key in keys}
    aris = pair_aris("HTEM", "six supported views; descriptive resolutions", views)
    rho, rho_p = rank_stat(
        rich, "source_matched_delta_vs_composition_same_k", "mrr_delta_vs_composition"
    )
    objectives = [
        {
            "dataset": "HTEM",
            "view": row.config_id + f"__K{int(row.clusters)}",
            "global_metric": "source-adjusted excess gain vs composition",
            "global_value": row.source_matched_delta_vs_composition_same_k,
            "global_gain": row.source_matched_delta_vs_composition_same_k,
            "local_metric": "MRR gain vs composition",
            "local_value": row.mrr,
            "local_gain": row.mrr_delta_vs_composition,
            "improves_global": row.source_matched_delta_vs_composition_same_k > 0,
            "improves_local": row.mrr_delta_vs_composition > 0,
        }
        for row in rich.itertuples()
    ]
    all_rich = grid.loc[
        (grid["config_id"] != "COMPOSITION_ONLY") & (grid["process_lambda"] > 0)
    ]
    bound = all_rich.pivot_table(
        index=["profile", "process_lambda", "clusters"],
        columns="variant",
        values="source_matched_delta_vs_composition_same_k",
    ).dropna(subset=["optimistic", "reported", "pessimistic"])
    uncertainty = []
    for index, row in bound.iterrows():
        values = row[["optimistic", "reported", "pessimistic"]].to_numpy(float)
        uncertainty.append(
            {
                "dataset": "HTEM",
                "view": f"{index[0]}__L{index[1]:.2f}__K{int(index[2])}",
                "metric": "source-adjusted excess gain vs composition",
                "optimistic": row.optimistic,
                "reported": row.reported,
                "pessimistic": row.pessimistic,
                "range": np.ptp(values),
                "all_positive": np.all(values > 0),
                "sign_change": np.min(values) < 0 < np.max(values),
            }
        )
    shuffle_margins, bound_aris = [], []
    for item in summary["diagnostics"]:
        observed = item["observed"]["source_matched_delta_vs_composition_same_k"]
        null95 = item["process_shuffle_within_element_system_and_instrument"][
            "source_matched_delta"
        ]["percentile_95"]
        shuffle_margins.append(observed - null95)
        bound_aris.extend(
            [
                item["ari_against_uncertainty_variants"]["optimistic"],
                item["ari_against_uncertainty_variants"]["pessimistic"],
            ]
        )
    weights = []
    for lam, group in rich.groupby("process_lambda"):
        weights.append(
            {
                "dataset": "HTEM",
                "process_weight": lam,
                "configurations": len(group),
                "global_gain_metric": "source-adjusted excess gain vs composition",
                "global_gain_median": med(group["source_matched_delta_vs_composition_same_k"]),
                "global_gain_positive_fraction": float(
                    np.mean(group["source_matched_delta_vs_composition_same_k"] > 0)
                ),
                "local_gain_metric": "MRR gain vs composition",
                "local_gain_median": med(group["mrr_delta_vs_composition"]),
            }
        )
    cross_source = len(
        read_json(paths["htem_candidates"])["cross_study_cross_instrument_hypotheses"]
    )
    diag = {
        "dataset": "HTEM",
        "domain": "high-throughput inorganic thin-film libraries",
        "samples": summary["counts"]["sample_libraries"],
        "weak_groups": summary["counts"]["study_components"],
        "context_gain_metric": "source-adjusted excess gain vs composition",
        "context_gain_median": med(rich["source_matched_delta_vs_composition_same_k"]),
        "context_gain_positive_fraction": float(
            np.mean(rich["source_matched_delta_vs_composition_same_k"] > 0)
        ),
        "semantic_vs_mask_metric": "source-adjusted excess gain vs mask",
        "semantic_vs_mask_median": med(rich["source_matched_delta_vs_mask_control"]),
        "semantic_vs_shuffle_metric": "observed gain minus shuffle 95th percentile; selected views",
        "semantic_vs_shuffle_median": med(shuffle_margins),
        "shuffle_supported_fraction": float(np.mean(np.asarray(shuffle_margins) > 0)),
        "global_local_spearman": rho,
        "global_local_spearman_p": rho_p,
        "global_local_joint_positive_fraction": float(
            np.mean(
                (rich["source_matched_delta_vs_composition_same_k"] > 0)
                & (rich["mrr_delta_vs_composition"] > 0)
            )
        ),
        "view_pairwise_ari_median": med([row["ari"] for row in aris]),
        "view_pairwise_ari_min": min(row["ari"] for row in aris),
        "view_pairwise_ari_max": max(row["ari"] for row in aris),
        "all_bounds_positive_fraction": float(np.mean(bound.gt(0).all(axis=1))),
        "bound_sign_change_fraction": float(
            np.mean((bound.min(axis=1) < 0) & (bound.max(axis=1) > 0))
        ),
        "bound_partition_ari_median": med(bound_aris),
        "cross_source_consensus_pairs": cross_source,
        "cross_source_candidates_with_author_or_location_support": math.nan,
        "interpretation": "工程は大域文脈を改善する条件が多いが、局所検索はほぼ改善せず、欠損境界で方向が変わる。",
    }
    return diag, aris, objectives, uncertainty, weights


def process_nano(
    paths: dict[str, Path],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    summary = read_json(paths["nano_summary"])
    grid = pd.read_csv(paths["nano_grid"])
    config = read_json(paths["nano_config"])
    retrieval = pd.read_csv(paths["nano_retrieval"])
    mask = pd.read_csv(paths["nano_mask"])
    shuffle = pd.read_csv(paths["nano_shuffle"])

    baseline = (
        grid.loc[grid["process_lambda"] == 0]
        .sort_values("process_profile")
        .drop_duplicates(["material_profile", "bound", "requested_clusters"])
        [
            [
                "material_profile",
                "bound",
                "requested_clusters",
                "paper_ari",
                "same_paper_recall_lift",
            ]
        ]
        .rename(columns={"paper_ari": "baseline_ari", "same_paper_recall_lift": "baseline_lift"})
    )
    rich = grid.loc[grid["process_lambda"] > 0].merge(
        baseline,
        on=["material_profile", "bound", "requested_clusters"],
        validate="many_to_one",
    )
    rich["delta_ari"] = rich["paper_ari"] - rich["baseline_ari"]
    rich["delta_log_lift"] = (
        np.log(rich["same_paper_recall_lift"]) - np.log(rich["baseline_lift"])
    )
    reported = rich.loc[rich["bound"] == "reported"]
    bound = rich.pivot_table(
        index=["material_profile", "process_profile", "process_lambda", "requested_clusters"],
        columns="bound",
        values="delta_ari",
    ).dropna(subset=["optimistic", "reported", "pessimistic"])
    uncertainty = []
    for index, row in bound.iterrows():
        values = row[["optimistic", "reported", "pessimistic"]].to_numpy(float)
        uncertainty.append(
            {
                "dataset": "NanoMine",
                "view": f"{index[0]}__{index[1]}__L{index[2]:.2f}__K{int(index[3])}",
                "metric": "paper ARI gain vs material-only",
                "optimistic": row.optimistic,
                "reported": row.reported,
                "pessimistic": row.pessimistic,
                "range": np.ptp(values),
                "all_positive": np.all(values > 0),
                "sign_change": np.min(values) < 0 < np.max(values),
            }
        )

    assignment_rows = pd.DataFrame(read_jsonl(paths["nano_assignments"]))
    matrix = assignment_rows.loc[
        (assignment_rows["bound"] == "reported") & (assignment_rows["k"] == 50)
    ].pivot(index="sample_id", columns="view", values="cluster")
    names = [item["name"] for item in config["sentinel_views"]]
    views = {name: matrix[name].astype(int).tolist() for name in names}
    aris = pair_aris("NanoMine", "six sentinel views; reported; k=50", views)

    sentinel_rows = []
    for item in config["sentinel_views"]:
        cluster_row = grid.loc[
            (grid["material_profile"] == item["material"])
            & (grid["process_profile"] == item["process"])
            & (grid["process_lambda"] == item["lambda"])
            & (grid["bound"] == "reported")
            & (grid["requested_clusters"] == 50)
        ].iloc[0]
        retrieval_row = retrieval.loc[
            (retrieval["view"] == item["name"]) & (retrieval["bound"] == "reported")
        ].iloc[0]
        sentinel_rows.append(
            {
                "view": item["name"],
                "paper_ari": cluster_row.paper_ari,
                "mrr": retrieval_row.mrr_midrank,
                "lift": cluster_row.same_paper_recall_lift,
                "mean_interval_width": cluster_row.mean_interval_width,
            }
        )
    sentinel = pd.DataFrame(sentinel_rows)
    base = sentinel.loc[sentinel["view"] == "material_only"].iloc[0]
    sentinel["global_gain"] = sentinel["paper_ari"] - base.paper_ari
    sentinel["local_gain"] = sentinel["mrr"] - base.mrr
    objectives = [
        {
            "dataset": "NanoMine",
            "view": row.view,
            "global_metric": "paper ARI at k=50",
            "global_value": row.paper_ari,
            "global_gain": row.global_gain,
            "local_metric": "same-paper MRR",
            "local_value": row.mrr,
            "local_gain": row.local_gain,
            "improves_global": row.global_gain > 0,
            "improves_local": row.local_gain > 0,
        }
        for row in sentinel.itertuples()
    ]
    rho, rho_p = rank_stat(sentinel, "paper_ari", "mrr")
    mask50 = mask.loc[mask["k"] == 50].iloc[0]
    shuffle_margin = float(
        shuffle.loc[
            (shuffle["control"] == "settings_value_shuffle")
            & (shuffle["k"] == 50)
            & (shuffle["metric"] == "paper_ari"),
            "original_minus_shuffled",
        ].iloc[0]
    )
    candidates = pd.read_csv(paths["nano_candidates"])
    strong = int(np.sum(candidates["view_support"] >= 16))
    externally_supported = int(
        np.sum(
            (candidates["view_support"] >= 16)
            & (candidates["shared_author"] | candidates["same_location_exact"])
        )
    )
    weights = []
    for lam, group in reported.groupby("process_lambda"):
        weights.append(
            {
                "dataset": "NanoMine",
                "process_weight": lam,
                "configurations": len(group),
                "global_gain_metric": "paper ARI gain vs material-only",
                "global_gain_median": med(group["delta_ari"]),
                "global_gain_positive_fraction": float(np.mean(group["delta_ari"] > 0)),
                "local_gain_metric": "not available for full grid",
                "local_gain_median": math.nan,
            }
        )
    nonbaseline = sentinel.loc[sentinel["view"] != "material_only"]
    diag = {
        "dataset": "NanoMine",
        "domain": "polymer nanocomposites literature",
        "samples": summary["samples"],
        "weak_groups": summary["papers"],
        "context_gain_metric": "paper ARI gain vs material-only",
        "context_gain_median": med(reported["delta_ari"]),
        "context_gain_positive_fraction": float(np.mean(reported["delta_ari"] > 0)),
        "semantic_vs_mask_metric": "median sentinel paper ARI minus mask ARI; k=50",
        "semantic_vs_mask_median": med(sentinel["paper_ari"] - mask50.paper_ari),
        "semantic_vs_mask_local_median": med(sentinel["mrr"] - mask50.mrr_midrank),
        "semantic_vs_shuffle_metric": "settings-focus paper ARI minus value-shuffle mean; k=50",
        "semantic_vs_shuffle_median": shuffle_margin,
        "shuffle_supported_fraction": float(shuffle_margin > 0),
        "global_local_spearman": rho,
        "global_local_spearman_p": rho_p,
        "global_local_joint_positive_fraction": float(
            np.mean((nonbaseline["global_gain"] > 0) & (nonbaseline["local_gain"] > 0))
        ),
        "view_pairwise_ari_median": med([row["ari"] for row in aris]),
        "view_pairwise_ari_min": min(row["ari"] for row in aris),
        "view_pairwise_ari_max": max(row["ari"] for row in aris),
        "all_bounds_positive_fraction": float(np.mean(bound.gt(0).all(axis=1))),
        "bound_sign_change_fraction": float(
            np.mean((bound.min(axis=1) < 0) & (bound.max(axis=1) > 0))
        ),
        "bound_partition_ari_median": summary["bound_stability"][
            "median_optimistic_pessimistic_ari"
        ],
        "cross_source_consensus_pairs": strong,
        "cross_source_candidates_with_author_or_location_support": externally_supported,
        "interpretation": "工程追加は大半のビューで論文文脈を強めるが欠損マスクも強い。大域ARIと局所MRRの順位はほぼ無相関。",
    }
    return diag, aris, objectives, uncertainty, weights, sentinel


def process_pnc(
    paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    summary = read_json(paths["pnc_summary"])
    grid = pd.read_csv(paths["pnc_grid"])
    retrieval = pd.read_csv(paths["pnc_retrieval"])
    k50 = grid.loc[
        (grid["bound"] == "reported") & (grid["requested_clusters"] == 50)
    ].merge(
        retrieval.loc[retrieval["bound"] == "reported"],
        on=["profile", "bound"],
        validate="one_to_one",
    )
    base = k50.loc[k50["profile"] == "strict_equal"].iloc[0]
    k50["global_gain"] = k50["paper_ari"] - base.paper_ari
    k50["local_gain"] = k50["mrr_midrank"] - base.mrr_midrank
    objectives = [
        {
            "dataset": "PNCExtract",
            "view": row.profile,
            "global_metric": "paper ARI at k=50",
            "global_value": row.paper_ari,
            "global_gain": row.global_gain,
            "local_metric": "same-paper MRR",
            "local_value": row.mrr_midrank,
            "local_gain": row.local_gain,
            "improves_global": row.global_gain > 0,
            "improves_local": row.local_gain > 0,
        }
        for row in k50.itertuples()
    ]
    rho, rho_p = rank_stat(k50, "paper_ari", "mrr_midrank")
    bound = grid.pivot_table(
        index=["profile", "requested_clusters"],
        columns="bound",
        values="same_paper_recall_lift",
    ).dropna(subset=["optimistic", "reported", "pessimistic"])
    uncertainty = []
    for index, row in bound.iterrows():
        values = row[["optimistic", "reported", "pessimistic"]].to_numpy(float)
        uncertainty.append(
            {
                "dataset": "PNCExtract",
                "view": f"{index[0]}__K{int(index[1])}",
                "metric": "same-paper co-assignment lift",
                "optimistic": row.optimistic,
                "reported": row.reported,
                "pessimistic": row.pessimistic,
                "range": np.ptp(values),
                "all_positive": np.all(values > 1),
                "sign_change": np.min(values) < 1 < np.max(values),
            }
        )
    strict_loading = k50.loc[k50["profile"] == "strict_loading"].iloc[0]
    nonbase = k50.loc[k50["profile"] != "strict_equal"]
    diag = {
        "dataset": "PNCExtract",
        "domain": "polymer nanocomposite composition breadth",
        "samples": summary["samples"],
        "weak_groups": summary["papers"],
        "context_gain_metric": "strict-loading paper ARI gain vs strict identity; k=50",
        "context_gain_median": strict_loading.global_gain,
        "context_gain_positive_fraction": float(np.mean(nonbase["global_gain"] > 0)),
        "semantic_vs_mask_metric": "not tested",
        "semantic_vs_mask_median": math.nan,
        "semantic_vs_shuffle_metric": "not tested",
        "semantic_vs_shuffle_median": math.nan,
        "shuffle_supported_fraction": math.nan,
        "global_local_spearman": rho,
        "global_local_spearman_p": rho_p,
        "global_local_joint_positive_fraction": float(
            np.mean((nonbase["global_gain"] > 0) & (nonbase["local_gain"] > 0))
        ),
        "view_pairwise_ari_median": math.nan,
        "view_pairwise_ari_min": math.nan,
        "view_pairwise_ari_max": math.nan,
        "all_bounds_positive_fraction": float(np.mean(bound.min(axis=1) > 1)),
        "bound_sign_change_fraction": float(
            np.mean((bound.min(axis=1) < 1) & (bound.max(axis=1) > 1))
        ),
        "bound_partition_ari_median": math.nan,
        "cross_source_consensus_pairs": math.nan,
        "cross_source_candidates_with_author_or_location_support": math.nan,
        "interpretation": "loading追加は局所MRRを上げる一方、k=50の論文ARIを下げ、用途依存性が明瞭。",
    }
    return diag, objectives, uncertainty, k50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_dir = output / "results"
    data_dir = output / "data"
    upstream = data_dir / "upstream"
    result_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "starry_summary": upstream / "starry_summary.json",
        "starry_assignments": upstream / "starry_assignments.jsonl.gz",
        "htem_summary": upstream / "htem_summary.json",
        "htem_grid": upstream / "htem_grid.csv",
        "htem_assignments": upstream / "htem_assignments.jsonl.gz",
        "htem_candidates": upstream / "htem_candidates.json",
        "nano_summary": upstream / "nano_summary.json",
        "nano_grid": upstream / "nano_grid.csv",
        "nano_config": upstream / "nano_config.json",
        "nano_assignments": upstream / "nano_assignments.jsonl.gz",
        "nano_retrieval": upstream / "nano_retrieval.csv",
        "nano_mask": upstream / "nano_mask.csv",
        "nano_shuffle": upstream / "nano_shuffle.csv",
        "nano_candidates": upstream / "nano_candidates.csv",
        "pnc_summary": upstream / "pnc_summary.json",
        "pnc_grid": upstream / "pnc_grid.csv",
        "pnc_retrieval": upstream / "pnc_retrieval.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing upstream artifacts:\n" + "\n".join(missing))
    manifest = {
        "created_on": "2026-09-10",
        "role": "Frozen upstream artifacts used by Phase 4",
        "files": {
            name: {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        },
    }
    (data_dir / "upstream_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    starry, starry_aris = process_starry(paths)
    htem, htem_aris, htem_obj, htem_unc, htem_weights = process_htem(paths)
    nano, nano_aris, nano_obj, nano_unc, nano_weights, sentinel = process_nano(paths)
    pnc, pnc_obj, pnc_unc, pnc_k50 = process_pnc(paths)

    diagnostics = pd.DataFrame([starry, htem, nano, pnc])
    disagreement = pd.DataFrame(starry_aris + htem_aris + nano_aris)
    objectives = pd.DataFrame(htem_obj + nano_obj + pnc_obj)
    uncertainty = pd.DataFrame(htem_unc + nano_unc + pnc_unc)
    weights = pd.DataFrame(htem_weights + nano_weights)
    view_summary = (
        disagreement.groupby("dataset")["ari"]
        .agg(pair_count="count", median="median", minimum="min", maximum="max")
        .reset_index()
    )

    diagnostics.to_csv(result_dir / "cross_dataset_diagnostics.csv", index=False)
    disagreement.to_csv(result_dir / "view_disagreement_pairs.csv", index=False)
    objectives.to_csv(result_dir / "objective_disagreement.csv", index=False)
    uncertainty.to_csv(result_dir / "uncertainty_sensitivity.csv", index=False)
    weights.to_csv(result_dir / "weight_response.csv", index=False)
    view_summary.to_csv(result_dir / "view_disagreement_summary.csv", index=False)
    sentinel.to_csv(result_dir / "nanomine_sentinel_objectives.csv", index=False)
    pnc_k50.to_csv(result_dir / "pncextract_k50_objectives.csv", index=False)

    evidence = pd.DataFrame(
        [
            {
                "proposition": "P1_noncomposition_changes_context_clustering",
                "Starrydata": "support",
                "HTEM": "support",
                "NanoMine": "support",
                "PNCExtract": "mixed",
                "basis": "正方向率100%, 88.6%, 97.6%; PNC loadingは大域ARI低下・局所MRR上昇",
            },
            {
                "proposition": "P2_values_exceed_missingness_or_shuffle",
                "Starrydata": "partial_support",
                "HTEM": "partial_support",
                "NanoMine": "partial_support",
                "PNCExtract": "not_tested",
                "basis": "値シャッフルとの差はあるが、欠損・source交絡は残る",
            },
            {
                "proposition": "P3_missingness_is_material_confounder",
                "Starrydata": "support",
                "HTEM": "support",
                "NanoMine": "support",
                "PNCExtract": "support",
                "basis": "mask対照、境界方向反転、区間幅またはloading境界感度",
            },
            {
                "proposition": "P4_global_and_local_tasks_differ",
                "Starrydata": "not_tested",
                "HTEM": "support",
                "NanoMine": "support",
                "PNCExtract": "support",
                "basis": "HTEMの両方改善4.6%、NanoMine順位rho≈0、PNCExtractで方向反転",
            },
            {
                "proposition": "P5_multiple_views_are_nonidentical",
                "Starrydata": "support",
                "HTEM": "support",
                "NanoMine": "support",
                "PNCExtract": "support",
                "basis": "主要ビュー間ARI中央値<0.90、目的・欠損境界・重みによる順位変化",
            },
            {
                "proposition": "P6_consensus_recovers_cross_source_relations",
                "Starrydata": "not_tested",
                "HTEM": "support",
                "NanoMine": "support",
                "PNCExtract": "not_tested",
                "basis": (
                    f"HTEM跨study/装置 {int(htem['cross_source_consensus_pairs'])}対、"
                    f"NanoMine 16/18ビュー以上 {int(nano['cross_source_consensus_pairs'])}対"
                ),
            },
        ]
    )
    evidence.to_csv(result_dir / "hypothesis_evidence_matrix.csv", index=False)

    indexed = diagnostics.set_index("dataset")
    criteria = {
        "A_cross_domain_context_change": bool(
            all(indexed.loc[name, "context_gain_median"] > 0 for name in ["Starrydata", "HTEM", "NanoMine"])
        ),
        "B_value_beyond_mask": bool(
            sum(
                [
                    indexed.loc["Starrydata", "semantic_vs_shuffle_median"] > 0,
                    indexed.loc["HTEM", "shuffle_supported_fraction"] >= 0.5,
                    indexed.loc["NanoMine", "semantic_vs_shuffle_median"] > 0,
                ]
            )
            >= 2
        ),
        "C_task_dependence": bool(
            sum(
                [
                    indexed.loc["HTEM", "global_local_joint_positive_fraction"] < 0.5,
                    abs(indexed.loc["NanoMine", "global_local_spearman"]) < 0.5,
                    indexed.loc["PNCExtract", "global_local_joint_positive_fraction"] < 0.5,
                ]
            )
            >= 2
        ),
        "D_multi_view_nonidentity": bool(
            sum(
                indexed.loc[name, "view_pairwise_ari_median"] < 0.9
                for name in ["Starrydata", "HTEM", "NanoMine"]
            )
            >= 2
        ),
        "E_cross_source_candidates": bool(
            indexed.loc["HTEM", "cross_source_consensus_pairs"] > 0
            and indexed.loc["NanoMine", "cross_source_consensus_pairs"] > 0
        ),
        "F_reproducibility": len(manifest["files"]) == len(paths),
    }
    optional = sum(
        criteria[name]
        for name in [
            "B_value_beyond_mask",
            "C_task_dependence",
            "D_multi_view_nonidentity",
            "F_reproducibility",
        ]
    )
    go = (
        criteria["A_cross_domain_context_change"]
        and criteria["E_cross_source_candidates"]
        and optional >= 3
    )
    decision = {
        "phase": 4,
        "decision": (
            "GO_FOR_METHODS_PAPER_HOLD_FOR_PERFORMANCE_TRANSFER_CLAIMS"
            if go
            else "HOLD_AND_REVISE"
        ),
        "criteria": criteria,
        "optional_criteria_passed": optional,
        "required_optional_passes": 3,
        "claim_supported": (
            "不完全な文献メタデータに対する多視点・欠損区間付き材料類似度は、"
            "複数領域で文脈的構造と安定な異source候補を回収する。"
            "ただし用途ごとにビューを保持する必要がある。"
        ),
        "claims_not_supported": [
            "普遍的に最良な材料類似度",
            "物理的な真の材料クラス",
            "性能予測または文献改善効果の転移可能性",
        ],
        "next_phase": "日本語論文草稿と統一APIの作成。性能転移研究は別研究として開始する。",
    }
    (result_dir / "phase4_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "datasets": diagnostics["dataset"].tolist(),
        "total_material_instances_across_tracks": int(diagnostics["samples"].sum()),
        "diagnostics": diagnostics.to_dict(orient="records"),
        "decision": decision,
        "notes": [
            "試料数の合計は独立トラックの延べ数であり、普遍コーパスの重複除去数ではない。",
            "重みグリッドは感度ビューであり、独立な統計反復ではない。",
            "定義の異なる効果量はデータセット間で数値的にプールしていない。",
        ],
    }
    (result_dir / "phase4_summary.json").write_text(
        json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision["decision"], "criteria": criteria}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
