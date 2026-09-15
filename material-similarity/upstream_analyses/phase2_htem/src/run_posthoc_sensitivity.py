#!/usr/bin/env python3
"""Audit selected Phase 2 patterns on quantitative-composition and source controls."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from phase2_interval_core import (
    build_field_matrices,
    cluster_grid,
    composition_distance,
    composition_summary,
    evaluate_labels,
    material_distance,
    prepare_evaluation_design,
    random_unlabeled_pairs,
    reporting_mask_distance,
    retrieval_metrics,
    study_positive_instances,
)
from run_phase2 import pair_source_strata


def load_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_delta(values: list[float], repetitions: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)
    absolute = np.abs(array)
    return {
        "components": len(array),
        "mean": float(np.mean(array)),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "positive_component_fraction": float(np.mean(array > 0)),
        "maximum_absolute_component_share": float(np.max(absolute) / np.sum(absolute)) if np.sum(absolute) else 0.0,
        "component_deltas": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "config" / "phase2_run_config.json").read_text(encoding="utf-8"))
    results = json.loads((root / "results" / "phase2_results.json").read_text(encoding="utf-8"))
    records = load_records(root / "data" / "htem_phase2_records.jsonl.gz")
    records.sort(key=lambda row: int(row["sample_library_id"]))
    ids = np.asarray([int(row["sample_library_id"]) for row in records], dtype=np.int64)
    id_to_index = {int(value): index for index, value in enumerate(ids)}
    studies = json.loads((root / "data" / "htem_study_links.json").read_text(encoding="utf-8"))
    instruments = np.asarray([str((row.get("validation_only") or {}).get("deposition_instrument") or "NOT_REPORTED") for row in records], dtype=object)
    summaries = [composition_summary(row) for row in records]
    composition, composition_info = composition_distance(summaries)
    field_distances, field_coverage, _ = build_field_matrices(records)
    mask_distance = reporting_mask_distance(field_coverage)
    seed = int(config["seed"])
    e_cfg = config["evaluation"]

    positives = study_positive_instances(studies, id_to_index)
    ui, uj = random_unlabeled_pairs(len(records), positives["unique_pairs"], int(e_cfg["unlabeled_pair_cap"]), seed + 1)
    ps = pair_source_strata(instruments, positives["pair_i"], positives["pair_j"])
    us = pair_source_strata(instruments, ui, uj)
    design = prepare_evaluation_design(positives, ui, uj, composition, int(e_cfg["composition_distance_bins"]), ps, us)

    with gzip.open(root / "results" / "selected_assignments.jsonl.gz", "rt", encoding="utf-8") as handle:
        assignment_rows = [json.loads(line) for line in handle]
    if [row["sample_library_id"] for row in assignment_rows] != ids.tolist():
        raise ValueError("Assignment order does not match records")
    assignment_names = list(assignment_rows[0]["assignments"])
    assignment_labels = {
        name: np.asarray([row["assignments"][name] for row in assignment_rows], dtype=np.int32)
        for name in assignment_names
    }
    selected = results["descriptively_ranked_configurations"]
    needed_k = sorted({int(row["clusters"]) for row in selected})
    baseline_labels = cluster_grid(composition, needed_k, ids, int(e_cfg["row_order_seeds"][0]))

    quantitative = composition_info["quantitative"]
    quantitative_ids = {int(ids[index]) for index in np.where(quantitative)[0]}
    quantitative_studies = [
        {"study_id": study["study_id"], "sample_library": [value for value in study.get("sample_library") or [] if int(value) in quantitative_ids]}
        for study in studies
    ]
    q_positives = study_positive_instances(quantitative_studies, id_to_index)
    q_ui_mask = quantitative[ui] & quantitative[uj]
    q_ui, q_uj = ui[q_ui_mask], uj[q_ui_mask]
    q_ps = pair_source_strata(instruments, q_positives["pair_i"], q_positives["pair_j"])
    q_us = pair_source_strata(instruments, q_ui, q_uj)
    q_design = prepare_evaluation_design(q_positives, q_ui, q_uj, composition, int(e_cfg["composition_distance_bins"]), q_ps, q_us)

    selected_sensitivity = []
    for rank, row in enumerate(selected, start=1):
        name = f"{row['config_id']}__K{int(row['clusters'])}"
        labels = assignment_labels[name]
        base = baseline_labels[int(row["clusters"])]
        current_eval = evaluate_labels(labels, positives, ui, uj, composition, int(e_cfg["composition_distance_bins"]), {}, ps, us, design)
        base_eval = evaluate_labels(base, positives, ui, uj, composition, int(e_cfg["composition_distance_bins"]), {}, ps, us, design)
        current_q = evaluate_labels(labels, q_positives, q_ui, q_uj, composition, int(e_cfg["composition_distance_bins"]), {}, q_ps, q_us, q_design)
        base_q = evaluate_labels(base, q_positives, q_ui, q_uj, composition, int(e_cfg["composition_distance_bins"]), {}, q_ps, q_us, q_design)

        def deltas(current: dict[str, Any], baseline: dict[str, Any], key: str) -> list[float]:
            c = current[key]["component_excesses"]
            b = baseline[key]["component_excesses"]
            return [float(c[name] - b[name]) for name in sorted(set(c) & set(b))]

        selected_sensitivity.append({
            "descriptive_rank": rank,
            "configuration": {key: row[key] for key in ["config_id", "profile", "variant", "process_lambda", "clusters"]},
            "all_libraries_source_matched_component_bootstrap": bootstrap_delta(deltas(current_eval, base_eval, "all_source_matched"), 20000, seed + 200 + rank),
            "quantitative_composition_only": {
                "libraries": int(np.sum(quantitative)),
                "positive_pair_instances": len(q_positives["pair_i"]),
                "study_components": len(q_positives["components"]),
                "unlabeled_pairs": len(q_ui),
                "current_source_matched_component_excess": current_q["all_source_matched"]["component_equal_excess"],
                "baseline_source_matched_component_excess": base_q["all_source_matched"]["component_equal_excess"],
                "delta_bootstrap": bootstrap_delta(deltas(current_q, base_q, "all_source_matched"), 20000, seed + 300 + rank),
            },
        })

    top_k = int(e_cfg["retrieval_top_k"])
    mask_retrieval = {"mask_only": retrieval_metrics(mask_distance, positives, top_k), "composition_anchored": {}}
    for process_lambda in [0.1, 0.25, 0.5, 0.75, 1.0]:
        distance = material_distance(composition, mask_distance, process_lambda)
        mask_retrieval["composition_anchored"][f"{process_lambda:g}"] = retrieval_metrics(distance, positives, top_k)

    output = {
        "version": "1.0.0",
        "role": "post-hoc robustness audit; does not alter the frozen grid",
        "selected_configuration_sensitivity": selected_sensitivity,
        "reporting_mask_retrieval_control": mask_retrieval,
    }
    (root / "results" / "phase2_posthoc_sensitivity.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
