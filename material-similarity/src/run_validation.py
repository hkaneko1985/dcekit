#!/usr/bin/env python3
"""Run API validation against frozen NanoMine and Phase 4 artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from material_similarity.distance import fit_numeric_scales

RESOURCE_SCALES = None

from material_similarity import (  # noqa: E402
    DEFAULT_VIEWS,
    DocumentedValue as V,
    MaterialInstance,
    MaterialSimilarityEngine,
    ProcessStep,
    ReportingStatus,
    ViewSpec,
    agglomerative_cluster,
    cluster_consensus,
    consensus_pairs,
    nearest_neighbors,
)


DATA = PACKAGE / "data" / "upstream" / "nanomine_phase3_records.jsonl.gz"
HTEM_DATA = PACKAGE / "data" / "upstream" / "htem_phase2_records.jsonl.gz"
PHASE4 = PACKAGE / "data" / "upstream" / "phase4_summary.json"
RESULTS = PACKAGE / "outputs" / "reproduced"
TARGET_IDS = {"l157-s2-zhao-2008", "l238-s2-zhao-2008"}

MATERIAL_SYSTEM_ALIASES = {
    "Poly(methyl methacrylate)": "PMMA",
    "DGEBA Epoxy Resin": "DGEBA epoxy",
    "Poly(bisphenol A carbonate)": "PC",
    "Polystyrene": "PS",
    "Silicon dioxide": "SiO2",
    "Aluminium oxide": "Al2O3",
    "Multi-wall carbon nanotubes": "MWCNT",
}

PROCESS_METHODS = (
    "Solution Processing",
    "Melt Mixing",
    "In-Situ Polymerization",
)

PROCESS_METHOD_LABELS = {
    "Solution Processing": "Solution processing",
    "Melt Mixing": "Melt mixing",
    "In-Situ Polymerization": "In-situ polymerization",
}


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_records() -> list[dict]:
    with gzip.open(DATA, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_records(rows: list[dict], maximum: int = 180) -> list[dict]:
    """Select complete paper blocks without using material values or outcomes."""

    groups = defaultdict(list)
    for row in rows:
        groups[row["paper_group"]].append(row)
    selected = []
    for group in sorted(groups, key=stable_key):
        candidates = sorted(groups[group], key=lambda row: stable_key(row["sample_id"]))
        if len(candidates) < 2:
            continue
        selected.extend(candidates[: min(8, len(candidates))])
        if len(selected) >= maximum:
            break
    by_id = {row["sample_id"]: row for row in rows}
    selected_by_id = {row["sample_id"]: row for row in selected}
    for record_id in TARGET_IDS:
        selected_by_id[record_id] = by_id[record_id]
    return [selected_by_id[key] for key in sorted(selected_by_id, key=stable_key)]


def role_value(row: dict, role: str, field: str) -> V:
    reported = role in row.get("reported_roles", [])
    names = row.get(field, [])
    if reported:
        return V.reported(names, kind="set")
    return V.unknown(kind="set")


def adapt_nanomine(row: dict) -> MaterialInstance:
    from material_similarity.adapters import adapt_nanomine as audited_adapter
    return audited_adapter(row)


HTEM_CATEGORICAL = {
    "deposition_compounds",
    "deposition_gases",
    "deposition_substrate_material",
}


def parse_jsonish(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.casefold() in {"none", "null", "nan", "n/a", "unknown"}:
        return None
    if text[:1] in {"[", "{"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def adapt_htem(row: dict) -> MaterialInstance:
    measurements = []
    for measurement in row.get("composition_measurements", []):
        current = {str(key): float(value) for key, value in measurement.items() if float(value) > 0}
        total = sum(current.values())
        if total > 0:
            measurements.append({key: value / total for key, value in current.items()})
    if measurements:
        elements = sorted(set().union(*(set(item) for item in measurements)))
        composition = {
            element: float(np.mean([item.get(element, 0.0) for item in measurements]))
            for element in elements
        }
        composition_value = V.reported(composition, kind="composition")
    else:
        composition_value = V.unknown(kind="composition")

    settings = {}
    for field, raw in row.get("process", {}).items():
        parsed = parse_jsonish(raw)
        if parsed is None:
            continue
        if field in HTEM_CATEGORICAL:
            values = parsed if isinstance(parsed, list) else [parsed]
            values = [str(value) for value in values if value is not None]
            if values:
                settings[field] = V.reported(values, kind="set")
            continue
        values = parsed if isinstance(parsed, list) else [parsed]
        for index, value in enumerate(values):
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            key = field if len(values) == 1 else f"{field}#{index}"
            settings[key] = V.reported(number, kind="numeric", unit_key=field)

    return MaterialInstance(
        record_id=f"htem-{row['sample_library_id']}",
        composition=composition_value,
        process_method=V.unknown(kind="set"),
        process_settings=settings,
        process_sequence_status=ReportingStatus.UNREPORTED_OR_UNSET,
        context={"sample_library_id": row["sample_library_id"]},
    )


def run_htem_validation() -> dict:
    all_rows = read_gzip_jsonl(HTEM_DATA)
    eligible = [row for row in all_rows if row.get("composition_measurements")]
    selected = sorted(eligible, key=lambda row: stable_key(str(row["sample_library_id"])))[:183]
    records = [adapt_htem(row) for row in selected]
    engine = MaterialSimilarityEngine(records, numeric_scales=fit_numeric_scales([adapt_htem(row) for row in eligible]))
    views = [
        ViewSpec("htem_composition", {"composition": 1.0}),
        ViewSpec("htem_composition_settings", {"composition": 0.5, "settings": 0.5}),
    ]
    pairwise = {view.name: engine.pairwise(view) for view in views}
    cluster_count = 20
    baseline = agglomerative_cluster(
        pairwise["htem_composition"], n_clusters=cluster_count, bound="reported"
    ).labels
    output_rows = []
    for view in views:
        result = pairwise[view.name]
        for bound in ("optimistic", "reported", "pessimistic"):
            clustering = agglomerative_cluster(
                result,
                n_clusters=cluster_count,
                bound=bound,
                undefined="pessimistic",
            )
            upper_triangle = np.triu_indices(len(records), 1)
            output_rows.append(
                {
                    "view": view.name,
                    "bound": bound,
                    "n_records": len(records),
                    "n_clusters": cluster_count,
                    "ari_vs_reported_composition": adjusted_rand_score(baseline, clustering.labels),
                    "mean_uncertainty_width": float(result.uncertainty_width[upper_triangle].mean()),
                    "mean_common_reported_fraction": float(result.common_reported_fraction[upper_triangle].mean()),
                }
            )
    with (RESULTS / "api_htem_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "source_records_available": len(all_rows),
        "validation_records": len(records),
        "reported_composite_ari_vs_composition": next(
            row["ari_vs_reported_composition"]
            for row in output_rows
            if row["view"] == "htem_composition_settings" and row["bound"] == "reported"
        ),
    }


def reciprocal_rank(matrix: np.ndarray, labels: list[str]) -> float:
    values = []
    for row, label in enumerate(labels):
        if labels.count(label) < 2:
            continue
        order = [idx for idx in np.argsort(matrix[row], kind="stable") if idx != row]
        rank = next((rank for rank, idx in enumerate(order, start=1) if labels[idx] == label), None)
        if rank is not None:
            values.append(1.0 / rank)
    return float(np.mean(values)) if values else math.nan


def material_system_key(row: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Exact matrix--filler identity used only for the post-hoc face-validity panel."""

    return tuple(row.get("matrix_names", [])), tuple(row.get("filler_names", []))


def material_system_label(key: tuple[tuple[str, ...], tuple[str, ...]]) -> str:
    matrix, filler = key
    matrix_label = " + ".join(MATERIAL_SYSTEM_ALIASES.get(value, value) for value in matrix)
    filler_label = " + ".join(MATERIAL_SYSTEM_ALIASES.get(value, value) for value in filler)
    return f"{matrix_label} / {filler_label}"


def select_material_system_subset(
    rows: list[dict],
    *,
    n_systems: int = 6,
    samples_per_system: int = 12,
    minimum_samples: int = 20,
    minimum_papers: int = 2,
) -> tuple[list[dict], list[dict]]:
    """Select frequent exact systems without inspecting distances or cluster outcomes.

    Papers are used only to balance the illustrative subset so that a single paper
    cannot dominate a system. They remain excluded from every distance calculation.
    """

    counts: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = defaultdict(int)
    papers: dict[tuple[tuple[str, ...], tuple[str, ...]], set[str]] = defaultdict(set)
    by_system: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict]] = defaultdict(list)
    for row in rows:
        key = material_system_key(row)
        counts[key] += 1
        papers[key].add(row["paper_group"])
        by_system[key].append(row)

    eligible = [
        key
        for key, count in counts.items()
        if count >= minimum_samples
        and len(papers[key]) >= minimum_papers
        and key[0]
        and key[1]
    ]
    eligible.sort(key=lambda key: (-counts[key], key))
    selected_keys = eligible[:n_systems]
    if len(selected_keys) != n_systems:
        raise ValueError("Not enough material systems satisfy the visualization rule")

    selected: list[dict] = []
    system_rows: list[dict] = []
    for system_index, key in enumerate(selected_keys, start=1):
        by_paper: dict[str, list[dict]] = defaultdict(list)
        for row in by_system[key]:
            by_paper[row["paper_group"]].append(row)
        paper_order = sorted(by_paper, key=stable_key)
        for paper in paper_order:
            by_paper[paper].sort(key=lambda row: stable_key(row["sample_id"]))

        paper_positions = {paper: 0 for paper in paper_order}
        current: list[dict] = []
        while len(current) < samples_per_system:
            added = False
            for paper in paper_order:
                position = paper_positions[paper]
                if position < len(by_paper[paper]):
                    current.append(by_paper[paper][position])
                    paper_positions[paper] += 1
                    added = True
                    if len(current) == samples_per_system:
                        break
            if not added:
                break
        if len(current) != samples_per_system:
            raise ValueError(f"Insufficient records for material system {key}")
        selected.extend(current)

        matrix, filler = key
        system_rows.append(
            {
                "system_id": f"S{system_index}",
                "system_label": material_system_label(key),
                "matrix": " + ".join(matrix),
                "filler": " + ".join(filler),
                "available_samples": counts[key],
                "available_papers": len(papers[key]),
                "selected_samples": len(current),
                "selected_papers": len({row["paper_group"] for row in current}),
            }
        )
    selected.sort(key=lambda row: stable_key(row["sample_id"]))
    return selected, system_rows


def principal_coordinates(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Deterministic two-axis classical MDS/PCoA for visualization only."""

    n_records = len(matrix)
    centering = np.eye(n_records) - np.ones((n_records, n_records)) / n_records
    gram = -0.5 * centering @ (matrix ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if np.count_nonzero(eigenvalues > 1e-10) < 2:
        raise ValueError("The distance matrix has fewer than two positive PCoA axes")
    coordinates = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    # Eigenvector signs are arbitrary. Canonicalize them for byte-stable outputs.
    for axis in range(2):
        pivot = int(np.argmax(np.abs(coordinates[:, axis])))
        if coordinates[pivot, axis] < 0:
            coordinates[:, axis] *= -1
    positive_inertia = float(eigenvalues[eigenvalues > 1e-10].sum())
    negative_inertia = float(abs(eigenvalues[eigenvalues < -1e-10].sum()))
    two_axis_fraction = float(eigenvalues[:2].sum() / positive_inertia)
    negative_fraction = float(negative_inertia / (positive_inertia + negative_inertia))
    return coordinates, two_axis_fraction, negative_fraction


def neighbor_agreement(
    matrix: np.ndarray,
    labels: list[int],
    *,
    k: int,
    papers: list[str] | None = None,
) -> float:
    agreements: list[bool] = []
    for row in range(len(labels)):
        order = [index for index in np.argsort(matrix[row], kind="stable") if index != row]
        if papers is not None:
            order = [index for index in order if papers[index] != papers[row]]
        neighbors = order[:k]
        if len(neighbors) != k:
            raise ValueError("Not enough eligible neighbors for the requested agreement")
        agreements.extend(labels[index] == labels[row] for index in neighbors)
    return float(np.mean(agreements))


def neighbor_orders(
    matrix: np.ndarray,
    *,
    k: int,
    papers: list[str] | None = None,
) -> list[list[int]]:
    """Return deterministic eligible neighbor indices for repeated label tests."""

    output: list[list[int]] = []
    for row in range(len(matrix)):
        order = [index for index in np.argsort(matrix[row], kind="stable") if index != row]
        if papers is not None:
            order = [index for index in order if papers[index] != papers[row]]
        neighbors = order[:k]
        if len(neighbors) != k:
            raise ValueError("Not enough eligible neighbors for the requested agreement")
        output.append(neighbors)
    return output


def agreement_from_orders(labels: list[int], orders: list[list[int]]) -> float:
    return float(
        np.mean(
            [
                labels[neighbor] == labels[row]
                for row, neighbors in enumerate(orders)
                for neighbor in neighbors
            ]
        )
    )


def select_process_method_subset(
    rows: list[dict],
    *,
    samples_per_method: int = 40,
) -> tuple[list[dict], list[dict]]:
    """Select three named single-method groups without inspecting their distances.

    A round-robin across papers limits paper-specific replication. Catch-all and
    multi-label process-family records are excluded by construction.
    """

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        methods = row.get("process_families", [])
        if len(methods) == 1 and methods[0] in PROCESS_METHODS:
            by_method[methods[0]].append(row)

    selected: list[dict] = []
    method_rows: list[dict] = []
    for method_index, method in enumerate(PROCESS_METHODS, start=1):
        by_paper: dict[str, list[dict]] = defaultdict(list)
        for row in by_method[method]:
            by_paper[row["paper_group"]].append(row)
        paper_order = sorted(by_paper, key=stable_key)
        for paper in paper_order:
            by_paper[paper].sort(key=lambda row: stable_key(row["sample_id"]))

        positions = {paper: 0 for paper in paper_order}
        current: list[dict] = []
        while len(current) < samples_per_method:
            added = False
            for paper in paper_order:
                position = positions[paper]
                if position < len(by_paper[paper]):
                    current.append(by_paper[paper][position])
                    positions[paper] += 1
                    added = True
                    if len(current) == samples_per_method:
                        break
            if not added:
                break
        if len(current) != samples_per_method:
            raise ValueError(f"Insufficient records for process method {method}")
        selected.extend(current)
        method_rows.append(
            {
                "method_id": f"M{method_index}",
                "process_method": method,
                "display_label": PROCESS_METHOD_LABELS[method],
                "available_samples": len(by_method[method]),
                "available_papers": len(by_paper),
                "selected_samples": len(current),
                "selected_papers": len({row["paper_group"] for row in current}),
            }
        )
    selected.sort(key=lambda row: stable_key(row["sample_id"]))
    return selected, method_rows


def run_process_method_visualization(all_rows: list[dict]) -> dict:
    """Create a presentation subset and quantify synthesis-method separation."""

    selected, methods = select_process_method_subset(all_rows)
    method_to_index = {
        row["process_method"]: index for index, row in enumerate(methods)
    }
    labels = [method_to_index[row["process_families"][0]] for row in selected]
    papers = [row["paper_group"] for row in selected]
    records = [adapt_nanomine(row) for row in selected]
    engine = MaterialSimilarityEngine(records, numeric_scales=RESOURCE_SCALES)
    views = (
        ViewSpec(
            "method_blind_process",
            {"step_type": 0.30, "sequence": 0.45, "settings": 0.25},
            soft_identity=True,
        ),
        ViewSpec(
            "method_aware_process",
            {
                "process_method": 0.25,
                "step_type": 0.20,
                "sequence": 0.30,
                "settings": 0.25,
            },
            soft_identity=True,
        ),
    )

    rng = np.random.default_rng(20260911)
    permutations = [rng.permutation(labels).tolist() for _ in range(1000)]
    embedding_rows: list[dict] = []
    metric_rows: list[dict] = []
    per_method_rows: list[dict] = []
    for view in views:
        pairwise = engine.pairwise(view)
        matrix = pairwise.matrix("reported", undefined="pessimistic")
        clustering = agglomerative_cluster(
            pairwise,
            n_clusters=len(methods),
            bound="reported",
            undefined="pessimistic",
        )
        coordinates, inertia_2d, negative_inertia = principal_coordinates(matrix)
        upper = np.triu(np.ones_like(matrix, dtype=bool), 1)
        same = np.equal.outer(labels, labels) & upper
        different = (~np.equal.outer(labels, labels)) & upper
        within = float(matrix[same].mean())
        between = float(matrix[different].mean())
        cross_orders = neighbor_orders(matrix, k=5, papers=papers)
        observed = agreement_from_orders(labels, cross_orders)
        null = np.asarray(
            [agreement_from_orders(permuted, cross_orders) for permuted in permutations],
            dtype=float,
        )
        weights = view.facet_weights
        metric_rows.append(
            {
                "view": view.name,
                "bound": "reported",
                "n_records": len(selected),
                "n_methods": len(methods),
                "n_unique_papers": len(set(papers)),
                "n_clusters": len(methods),
                "process_method_weight": weights.get("process_method", 0.0),
                "step_type_weight": weights.get("step_type", 0.0),
                "sequence_weight": weights.get("sequence", 0.0),
                "settings_weight": weights.get("settings", 0.0),
                "ari_vs_process_method": adjusted_rand_score(labels, clustering.labels),
                "nmi_vs_process_method": normalized_mutual_info_score(labels, clustering.labels),
                "silhouette_process_method": silhouette_score(matrix, labels, metric="precomputed"),
                "knn5_process_method_agreement": neighbor_agreement(matrix, labels, k=5),
                "cross_paper_knn5_process_method_agreement": observed,
                "permutation_cross_paper_knn5_mean": float(null.mean()),
                "permutation_cross_paper_knn5_q025": float(np.quantile(null, 0.025)),
                "permutation_cross_paper_knn5_q975": float(np.quantile(null, 0.975)),
                "permutation_one_sided_p": float((1 + np.sum(null >= observed)) / (1 + len(null))),
                "mean_within_method_distance": within,
                "mean_between_method_distance": between,
                "between_within_distance_ratio": between / within,
                "pcoa_two_axis_positive_inertia_fraction": inertia_2d,
                "pcoa_negative_inertia_fraction": negative_inertia,
                "mean_uncertainty_width": float(pairwise.uncertainty_width[upper].mean()),
            }
        )
        for method_index, method in enumerate(methods):
            focal = [
                labels[neighbor] == labels[row]
                for row, neighbors in enumerate(cross_orders)
                if labels[row] == method_index
                for neighbor in neighbors
            ]
            per_method_rows.append(
                {
                    "view": view.name,
                    "method_id": method["method_id"],
                    "process_method": method["process_method"],
                    "cross_paper_knn5_agreement": float(np.mean(focal)),
                }
            )
        for index, row in enumerate(selected):
            method = methods[labels[index]]
            embedding_rows.append(
                {
                    "record_id": row["sample_id"],
                    "paper_group": row["paper_group"],
                    "method_id": method["method_id"],
                    "process_method": method["process_method"],
                    "display_label": method["display_label"],
                    "view": view.name,
                    "cluster": int(clustering.labels[index]) + 1,
                    "pcoa1": float(coordinates[index, 0]),
                    "pcoa2": float(coordinates[index, 1]),
                }
            )

    for filename, output_rows in (
        ("nanomine_process_methods.csv", methods),
        ("nanomine_process_method_embedding.csv", embedding_rows),
        ("nanomine_process_method_separation.csv", metric_rows),
        ("nanomine_process_method_neighbors.csv", per_method_rows),
    ):
        with (RESULTS / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)

    blind = next(row for row in metric_rows if row["view"] == "method_blind_process")
    aware = next(row for row in metric_rows if row["view"] == "method_aware_process")
    return {
        "selection_rule": {
            "process_methods": list(PROCESS_METHODS),
            "single_method_records_only": True,
            "samples_per_method": 40,
            "within_method_sampling": "deterministic round-robin across papers",
            "selection_uses_distances_or_outcomes": False,
        },
        "selected_records": len(selected),
        "selected_unique_papers": len(set(papers)),
        "methods": methods,
        "method_blind_process": blind,
        "method_aware_process": aware,
        "per_method_cross_paper_neighbors": per_method_rows,
        "interpretation": "construct_validity_ablation_not_independent_ground_truth",
    }


def run_material_system_visualization(all_rows: list[dict]) -> dict:
    selected, systems = select_material_system_subset(all_rows)
    label_to_index = {row["system_label"]: index for index, row in enumerate(systems)}
    labels = [label_to_index[material_system_label(material_system_key(row))] for row in selected]
    papers = [row["paper_group"] for row in selected]
    records = [adapt_nanomine(row) for row in selected]
    engine = MaterialSimilarityEngine(records, numeric_scales=RESOURCE_SCALES)

    embedding_rows: list[dict] = []
    metric_rows: list[dict] = []
    for view_name in ("material_identity", "balanced_instance"):
        pairwise = engine.pairwise(DEFAULT_VIEWS[view_name])
        matrix = pairwise.matrix("reported", undefined="pessimistic")
        clustering = agglomerative_cluster(
            pairwise,
            n_clusters=len(systems),
            bound="reported",
            undefined="pessimistic",
        )
        coordinates, inertia_2d, negative_inertia = principal_coordinates(matrix)
        upper = np.triu(np.ones_like(matrix, dtype=bool), 1)
        same = np.equal.outer(labels, labels) & upper
        different = (~np.equal.outer(labels, labels)) & upper
        within = float(matrix[same].mean())
        between = float(matrix[different].mean())
        metric_rows.append(
            {
                "view": view_name,
                "bound": "reported",
                "n_records": len(selected),
                "n_systems": len(systems),
                "n_unique_papers": len(set(papers)),
                "n_clusters": len(systems),
                "ari_vs_exact_material_system": adjusted_rand_score(labels, clustering.labels),
                "nmi_vs_exact_material_system": normalized_mutual_info_score(labels, clustering.labels),
                "silhouette_exact_material_system": silhouette_score(
                    matrix, labels, metric="precomputed"
                ),
                "knn5_material_system_agreement": neighbor_agreement(matrix, labels, k=5),
                "cross_paper_knn5_material_system_agreement": neighbor_agreement(
                    matrix, labels, k=5, papers=papers
                ),
                "mean_within_system_distance": within,
                "mean_between_system_distance": between,
                "between_within_distance_ratio": between / within,
                "pcoa_two_axis_positive_inertia_fraction": inertia_2d,
                "pcoa_negative_inertia_fraction": negative_inertia,
                "mean_uncertainty_width": float(pairwise.uncertainty_width[upper].mean()),
            }
        )
        for index, row in enumerate(selected):
            system = systems[labels[index]]
            embedding_rows.append(
                {
                    "record_id": row["sample_id"],
                    "paper_group": row["paper_group"],
                    "system_id": system["system_id"],
                    "system_label": system["system_label"],
                    "view": view_name,
                    "cluster": int(clustering.labels[index]) + 1,
                    "pcoa1": float(coordinates[index, 0]),
                    "pcoa2": float(coordinates[index, 1]),
                }
            )

    for filename, output_rows in (
        ("nanomine_material_systems.csv", systems),
        ("nanomine_material_system_embedding.csv", embedding_rows),
        ("nanomine_material_system_separation.csv", metric_rows),
    ):
        with (RESULTS / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)

    identity = next(row for row in metric_rows if row["view"] == "material_identity")
    balanced = next(row for row in metric_rows if row["view"] == "balanced_instance")
    return {
        "selection_rule": {
            "n_systems": 6,
            "samples_per_system": 12,
            "minimum_available_samples": 20,
            "minimum_available_papers": 2,
            "system_ranking": "descending available sample count, lexical tie-break",
            "within_system_sampling": "deterministic round-robin across papers",
        },
        "selected_records": len(selected),
        "selected_unique_papers": len(set(papers)),
        "systems": systems,
        "material_identity": identity,
        "balanced_instance": balanced,
        "interpretation": "face_validity_only_not_independent_ground_truth",
    }


def main() -> None:
    global RESOURCE_SCALES
    RESOURCE_SCALES = fit_numeric_scales([adapt_nanomine(row) for row in read_records()])
    global RESULTS
    parser = argparse.ArgumentParser(description="Reproduce the common HTEM and NanoMine examples from frozen normalized inputs.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS, help="Output directory (default: outputs/reproduced under the repository).")
    args = parser.parse_args()
    RESULTS = args.output_dir.resolve()
    print("Starting common-interface examples", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_nanomine_rows = read_records()
    raw = select_records(all_nanomine_rows)
    records = [adapt_nanomine(row) for row in raw]
    engine = MaterialSimilarityEngine(
        records,
        numeric_scales=RESOURCE_SCALES,
    )
    paper_labels = [record.context["paper_group"] for record in records]
    evaluated_views = [
        "material_identity",
        "material_identity_soft",
        "synthesis_pathway",
        "experimental_protocol",
        "balanced_instance",
    ]

    rows = []
    pairwise_results = {}
    clusterings = []
    cluster_count = min(20, max(2, len(set(paper_labels))))
    for view_name in evaluated_views:
        print(f"NanoMine view: {view_name}", flush=True)
        pairwise = engine.pairwise(DEFAULT_VIEWS[view_name])
        pairwise_results[view_name] = pairwise
        for bound in ("optimistic", "reported", "pessimistic"):
            matrix = pairwise.matrix(bound, undefined="pessimistic")
            clustering = agglomerative_cluster(
                pairwise,
                n_clusters=cluster_count,
                bound=bound,
                undefined="pessimistic",
            )
            if bound == "reported":
                clusterings.append(clustering)
            rows.append(
                {
                    "view": view_name,
                    "bound": bound,
                    "n_records": len(records),
                    "n_clusters": cluster_count,
                    "paper_ari": adjusted_rand_score(paper_labels, clustering.labels),
                    "paper_mrr": reciprocal_rank(matrix, paper_labels),
                    "mean_uncertainty_width": float(
                        pairwise.uncertainty_width[np.triu_indices(len(records), 1)].mean()
                    ),
                    "mean_common_reported_fraction": float(
                        pairwise.common_reported_fraction[np.triu_indices(len(records), 1)].mean()
                    ),
                }
            )

    with (RESULTS / "api_nanomine_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    target_details = []
    for view_name in evaluated_views:
        comparison = engine.compare(
            "l157-s2-zhao-2008", "l238-s2-zhao-2008", DEFAULT_VIEWS[view_name]
        )
        matrix = pairwise_results[view_name].matrix("reported", undefined="pessimistic")
        index = {record_id: i for i, record_id in enumerate(pairwise_results[view_name].ids)}
        i, j = index["l157-s2-zhao-2008"], index["l238-s2-zhao-2008"]
        rank_i = int(np.where(np.asarray([x for x in np.argsort(matrix[i]) if x != i]) == j)[0][0] + 1)
        rank_j = int(np.where(np.asarray([x for x in np.argsort(matrix[j]) if x != j]) == i)[0][0] + 1)
        target_details.append(
            {
                "view": view_name,
                "rank_left_to_right": rank_i,
                "rank_right_to_left": rank_j,
                **comparison.composite.as_dict(),
            }
        )
    with (RESULTS / "representative_pair_views.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_details[0]))
        writer.writeheader()
        writer.writerows(target_details)

    consensus = cluster_consensus(clusterings)
    candidates = consensus_pairs(
        consensus,
        tuple(record.record_id for record in records),
        minimum_support=0.8,
    )
    different_paper = [
        item
        for item in candidates
        if records[next(i for i, x in enumerate(records) if x.record_id == item["left_id"])].context["paper_group"]
        != records[next(i for i, x in enumerate(records) if x.record_id == item["right_id"])].context["paper_group"]
    ]
    with (RESULTS / "api_consensus_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["left_id", "right_id", "support"])
        writer.writeheader()
        writer.writerows(different_paper)

    # Synthetic invariants make the intended treatment of missing settings
    # explicit and independent of the reporting patterns in NanoMine.
    probe_a = MaterialInstance(
        "probe-a",
        composition=V.reported({"Al": 2, "O": 3}, kind="composition"),
        process_method=V.reported(["solution"], kind="set"),
        process_steps=(
            ProcessStep(
                "heating",
                {
                    "temperature": V.reported(100, kind="numeric", unit_key="temperature_c"),
                    "time": V.reported(60, kind="numeric", unit_key="time_min"),
                },
            ),
        ),
        process_sequence_status=ReportingStatus.REPORTED,
    )
    probe_b = MaterialInstance(
        "probe-b",
        composition=V.reported({"Al": 2, "O": 3}, kind="composition"),
        process_method=V.reported(["solution"], kind="set"),
        process_steps=(
            ProcessStep(
                "heating",
                {
                    "temperature": V.reported(100, kind="numeric", unit_key="temperature_c"),
                    "time": V.unknown(kind="numeric", unit_key="time_min"),
                },
            ),
        ),
        process_sequence_status=ReportingStatus.REPORTED,
    )
    probe = MaterialSimilarityEngine([probe_a, probe_b]).facet_distances("probe-a", "probe-b")["settings"]

    print("HTEM example (183 libraries)", flush=True)
    htem_validation = run_htem_validation()
    print("NanoMine material systems (72 records)", flush=True)
    material_system_visualization = run_material_system_visualization(all_nanomine_rows)
    print("NanoMine synthesis methods (120 records)", flush=True)
    process_method_visualization = run_process_method_visualization(all_nanomine_rows)

    phase4 = json.loads(PHASE4.read_text(encoding="utf-8"))
    locked = {
        "decision": phase4["decision"]["decision"],
        "total_material_instances_across_tracks": phase4["total_material_instances_across_tracks"],
        "dataset_counts": {
            row["dataset"]: {"samples": row["samples"], "weak_groups": row["weak_groups"]}
            for row in phase4["diagnostics"]
        },
        "all_decision_criteria_pass": all(phase4["decision"]["criteria"].values()),
    }
    (RESULTS / "phase4_claims_lock.json").write_text(
        json.dumps(locked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    payload = {
        "phase": 5,
        "api_version": "0.2.2",
        "source_records_available": len(all_nanomine_rows),
        "validation_records": len(records),
        "validation_papers": len(set(paper_labels)),
        "htem_validation": htem_validation,
        "material_system_visualization": material_system_visualization,
        "process_method_visualization": process_method_visualization,
        "views": evaluated_views,
        "bounds": ["optimistic", "reported", "pessimistic"],
        "cluster_count": cluster_count,
        "cross_paper_consensus_pairs_at_0_8": len(different_paper),
        "representative_pair": target_details,
        "missingness_probe": probe.as_dict(),
        "invariants": {
            "no_target_property_field_in_schema": "target" not in MaterialInstance.__dataclass_fields__,
            "context_ignored_by_distance": True,
            "no_missing_value_imputation": True,
            "reported_distance_uses_common_reported_settings": probe.reported == 0.0,
            "missing_setting_widens_interval": probe.width > 0.0,
            "phase4_claim_lock_passed": locked["all_decision_criteria_pass"],
        },
    }
    (RESULTS / "api_validation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Completed: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
