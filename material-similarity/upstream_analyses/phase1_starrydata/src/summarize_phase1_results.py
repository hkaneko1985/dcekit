#!/usr/bin/env python3
"""Summarize Phase 1 replicate runs and property-free F2 cluster profiles."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


def numeric_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "values": [float(value) for value in array],
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def top(counter: collections.Counter[str], total: int, limit: int = 6) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": int(count), "fraction": float(count / total)}
        for value, count in counter.most_common(limit)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--replicate", action="append", default=[], type=Path)
    parser.add_argument("--resolution-sensitivity", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profiles-output", required=True, type=Path)
    args = parser.parse_args()

    runs = [json.loads(args.primary.read_text(encoding="utf-8"))]
    runs.extend(json.loads(path.read_text(encoding="utf-8")) for path in args.replicate)
    resolution = json.loads(args.resolution_sensitivity.read_text(encoding="utf-8"))

    feature_lifts = {
        feature: numeric_summary([
            run["evaluation"][feature]["selected_resolution"]["lift"] for run in runs
        ])
        for feature in ("F0", "F1", "F2", "F3")
    }
    summary = {
        "summary_version": "1.0.0",
        "primary_seed": runs[0]["seed"],
        "replicate_seeds": [run["seed"] for run in runs[1:]],
        "runs": len(runs),
        "feature_lifts": feature_lifts,
        "F2_minus_F0_observed_delta_log_lift": numeric_summary([
            run["incremental_comparisons"]["F2_minus_F0"]["observed_delta_log_lift"] for run in runs
        ]),
        "F2_minus_F1_observed_delta_log_lift": numeric_summary([
            run["incremental_comparisons"]["F2_minus_F1"]["observed_delta_log_lift"] for run in runs
        ]),
        "complete_case_F2_minus_F0_delta_log_lift": numeric_summary([
            run["falsification_controls"]["complete_form_and_fabrication_case"]["F2_minus_F0"]["observed_delta_log_lift"]
            for run in runs
        ]),
        "exact_composition_F2_minus_F0_delta_log_lift": numeric_summary([
            run["falsification_controls"]["exact_composition_pairs"]["F2_minus_F0"]["observed_delta_log_lift"]
            for run in runs
        ]),
        "missingness_only_lift": numeric_summary([
            run["falsification_controls"]["missingness_only"]["lift"] for run in runs
        ]),
        "process_shuffle_F2_lift": numeric_summary([
            run["falsification_controls"]["composition_stratified_process_shuffle_F2"]["lift"] for run in runs
        ]),
        "paper_dominance": numeric_summary([
            run["falsification_controls"]["paper_dominance_F2_minus_F0"]["maximum_absolute_increment_share"]
            for run in runs
        ]),
        "all_leave_one_paper_out_directions_positive": all(
            run["falsification_controls"]["leave_one_paper_out_F2_minus_F0"]["papers_whose_omission_reverses_direction"] == 0
            for run in runs
        ),
        "all_complete_case_ci_lower_bounds_positive": all(
            run["falsification_controls"]["complete_form_and_fabrication_case"]["F2_minus_F0"]["bootstrap_95_interval"][0] > 0
            for run in runs
        ),
        "all_exact_composition_ci_lower_bounds_positive": all(
            run["falsification_controls"]["exact_composition_pairs"]["F2_minus_F0"]["bootstrap_95_interval"][0] > 0
            for run in runs
        ),
        "resolution_sensitivity": {
            "cluster_counts": sorted(int(k) for k in resolution["evaluation"]["F2"]["fixed_resolution_lift"]),
            "F2_minus_F0_positive_resolutions": resolution["incremental_comparisons"]["F2_minus_F0"]["fixed_resolutions_with_positive_delta"],
            "total_resolutions": resolution["incremental_comparisons"]["F2_minus_F0"]["fixed_resolutions_total"],
            "silhouette_reached_upper_boundary": all(
                resolution["clustering"][feature]["selected_clusters"]
                == max(int(k) for k in resolution["evaluation"][feature]["fixed_resolution_lift"])
                for feature in ("F0", "F1", "F2", "F3")
            ),
            "interpretation": "The incremental result is multiresolution-robust, but the data do not identify a unique natural cluster count within the tested range.",
        },
        "strict_preregistered_decision": runs[0]["decision_for_phase2"],
        "recommended_next_action": "Proceed to HTEM external replication while retaining the paper-dominance and source-missingness caveats.",
    }

    rows = []
    with gzip.open(args.assignments, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    by_cluster: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_cluster[int(row["F2_cluster"])].append(row)
    profiles = []
    for cluster, members in sorted(by_cluster.items()):
        compositions = collections.Counter(row["composition"] for row in members)
        forms = collections.Counter(value for row in members for value in row["form_categories"])
        processes = collections.Counter(value for row in members for value in row["fabrication_categories"])
        profiles.append({
            "F2_cluster": cluster,
            "samples": len(members),
            "unique_papers": len({row["SID"] for row in members}),
            "top_compositions": top(compositions, len(members)),
            "top_forms": top(forms, len(members)),
            "top_fabrication_categories": top(processes, len(members)),
        })
    profile_output = {
        "profile_version": "1.0.0",
        "feature_set": "F2",
        "clusters": len(profiles),
        "samples": len(rows),
        "profiles": sorted(profiles, key=lambda item: (-item["samples"], item["F2_cluster"])),
        "warning": "Profiles are descriptive and were not used to select or validate the clusters.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.profiles_output.parent.mkdir(parents=True, exist_ok=True)
    args.profiles_output.write_text(json.dumps(profile_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "runs": len(runs),
        "F2_lift_median": feature_lifts["F2"]["median"],
        "paper_dominance_range": [summary["paper_dominance"]["minimum"], summary["paper_dominance"]["maximum"]],
        "resolution_positive": [summary["resolution_sensitivity"]["F2_minus_F0_positive_resolutions"], summary["resolution_sensitivity"]["total_resolutions"]],
        "cluster_profiles": len(profiles),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
