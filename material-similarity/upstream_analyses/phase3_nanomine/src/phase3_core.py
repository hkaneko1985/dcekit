"""Core distance and evaluation utilities for NanoMine Phase 3."""

from __future__ import annotations

import gzip
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


BOUNDS = ("optimistic", "reported", "pessimistic")


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = value.replace("μ", "u").replace("µ", "u")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def char_ngrams(value: str, n: int = 3) -> set[str]:
    value = f"  {norm_text(value)}  "
    return {value[i : i + n] for i in range(max(1, len(value) - n + 1))}


def jaccard_distance(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    if not left and not right:
        return 0.0
    return 1.0 - len(left & right) / len(left | right)


def lexical_distance(a: str, b: str) -> float:
    return jaccard_distance(char_ngrams(a), char_ngrams(b))


def soft_set_distance(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = sorted(set(a)), sorted(set(b))
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    costs = np.array([[lexical_distance(x, y) for y in right] for x in left], dtype=float)
    rows, cols = linear_sum_assignment(costs)
    return float((costs[rows, cols].sum() + abs(len(left) - len(right))) / max(len(left), len(right)))


def exact_set_distance(a: Iterable[str], b: Iterable[str]) -> float:
    return jaccard_distance((norm_text(x) for x in a), (norm_text(x) for x in b))


def sequence_edit_distance(a: list[str], b: list[str]) -> float:
    """Normalized edit distance with lexical overlap as substitution cost."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = np.arange(len(b) + 1, dtype=float)
    for i, left in enumerate(a, start=1):
        current = np.empty(len(b) + 1, dtype=float)
        current[0] = i
        left_parts = set(left.split("|"))
        for j, right in enumerate(b, start=1):
            right_parts = set(right.split("|"))
            substitution = jaccard_distance(left_parts, right_parts)
            current[j] = min(previous[j] + 1.0, current[j - 1] + 1.0, previous[j - 1] + substitution)
        previous = current
    return float(previous[-1] / max(len(a), len(b)))


def load_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def transformed_numeric(value: float, scale_key: str) -> float:
    if scale_key in {"fraction", "temperature_c"} or "fraction" in scale_key:
        return float(value)
    if value >= 0:
        return float(np.log1p(value))
    return float(np.sign(value) * np.log1p(abs(value)))


@dataclass
class FeatureRecord:
    matrix: list[str]
    filler: list[str]
    surface: list[str]
    role_reported: set[str]
    loading: dict[str, list[dict[str, Any]]]
    descriptor: dict[str, list[dict[str, Any]]]
    process_family: list[str]
    step_types: list[str]
    sequence: list[str]
    settings_all: dict[str, list[dict[str, Any]]]
    settings_numeric: dict[str, list[dict[str, Any]]]
    missing_mask: set[str]


def build_feature_records(records: list[dict[str, Any]]) -> list[FeatureRecord]:
    output = []
    for record in records:
        loading: dict[str, list[dict[str, Any]]] = defaultdict(list)
        descriptor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        mask = set()
        for role in record.get("reported_roles", []):
            mask.add(f"role:{norm_text(role)}")
        for component in record.get("components", []):
            role = norm_text(component.get("role", "unknown"))
            for attr in component.get("attributes", []):
                attr_type = attr.get("type", "")
                if attr.get("kind") != "numeric":
                    continue
                mask.add(f"component:{role}:{attr_type}")
                item = {
                    "kind": "numeric",
                    "value": float(attr["value"]),
                    "scale_key": "fraction" if attr_type in {"MassFraction", "VolumeFraction"} else attr.get("unit_group", attr_type),
                }
                if attr_type in {"MassFraction", "VolumeFraction"} and role == "filler":
                    loading[attr_type].append(item)
                elif attr_type in {"Density", "Width", "AspectRatio", "SpecificSurfaceArea"}:
                    key = f"{role}:{attr_type}:{attr.get('unit_group', 'unknown')}"
                    descriptor[key].append(item)

        settings_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
        settings_numeric: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for setting in record.get("settings", []):
            base_key = setting["comparison_key"]
            item = {
                "kind": setting["kind"],
                "value": setting["value"],
                "scale_key": f"{setting['step_token']}:{setting['key']}:{setting.get('unit_group', 'categorical')}",
            }
            key = base_key + ":" + setting.get("unit_group", setting["kind"])
            settings_all[key].append(item)
            mask.add(f"setting:{key}")
            if setting["kind"] == "numeric":
                settings_numeric[key].append(item)
                mask.add(f"numeric-setting:{key}")
        output.append(
            FeatureRecord(
                matrix=record.get("matrix_names", []),
                filler=record.get("filler_names", []),
                surface=record.get("surface_names", []),
                role_reported=set(record.get("reported_roles", [])),
                loading=dict(loading),
                descriptor=dict(descriptor),
                process_family=record.get("process_families", []) or ["Unreported"],
                step_types=record.get("step_types", []),
                sequence=record.get("step_sequence", []),
                settings_all=dict(settings_all),
                settings_numeric=dict(settings_numeric),
                missing_mask=mask,
            )
        )
    return output


def numeric_scales(features: list[FeatureRecord]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    maps = []
    for feature in features:
        maps.extend((feature.loading, feature.descriptor, feature.settings_all))
    for mapping in maps:
        for entries in mapping.values():
            for entry in entries:
                if entry["kind"] == "numeric":
                    key = entry["scale_key"]
                    values[key].append(transformed_numeric(entry["value"], key))
    scales = {}
    for key, vals in values.items():
        array = np.asarray(vals, dtype=float)
        if len(array) >= 4:
            scale = float(np.quantile(array, 0.9) - np.quantile(array, 0.1))
        else:
            scale = float(np.ptp(array)) if len(array) > 1 else 0.0
        scales[key] = max(scale, 1e-9)
    scales["fraction"] = 1.0
    return scales


def value_distance(a: dict[str, Any], b: dict[str, Any], scales: dict[str, float]) -> float:
    if a["kind"] != b["kind"]:
        return 1.0
    if a["kind"] == "categorical":
        return lexical_distance(str(a["value"]), str(b["value"]))
    key_a, key_b = a.get("scale_key", ""), b.get("scale_key", "")
    if key_a != key_b:
        return 1.0
    if "fraction" in key_a:
        return float(min(1.0, abs(float(a["value"]) - float(b["value"]))))
    left = transformed_numeric(float(a["value"]), key_a)
    right = transformed_numeric(float(b["value"]), key_b)
    return float(min(1.0, abs(left - right) / scales.get(key_a, 1.0)))


def value_list_distance(a: list[dict[str, Any]], b: list[dict[str, Any]], scales: dict[str, float]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    costs = np.array([[value_distance(x, y, scales) for y in b] for x in a], dtype=float)
    rows, cols = linear_sum_assignment(costs)
    return float((costs[rows, cols].sum() + abs(len(a) - len(b))) / max(len(a), len(b)))


def map_interval_distance(
    a: dict[str, list[dict[str, Any]]],
    b: dict[str, list[dict[str, Any]]],
    scales: dict[str, float],
) -> tuple[float, float, float, float]:
    union = set(a) | set(b)
    if not union:
        return 0.0, math.nan, 1.0, 0.0
    common = set(a) & set(b)
    known = sum(value_list_distance(a[key], b[key], scales) for key in common)
    unknown = len(union - common)
    lower = known / len(union)
    upper = (known + unknown) / len(union)
    reported = known / len(common) if common else math.nan
    return float(lower), float(reported), float(upper), len(common) / len(union)


def known_or_missing_set(
    a: list[str], b: list[str], a_known: bool, b_known: bool, soft: bool
) -> tuple[float, float, float, float]:
    if not (a_known and b_known):
        return 0.0, math.nan, 1.0, 0.0
    distance = soft_set_distance(a, b) if soft else exact_set_distance(a, b)
    return distance, distance, distance, 1.0


def empty_facet_arrays(size: int) -> dict[str, np.ndarray]:
    return {name: np.empty(size, dtype=np.float32) for name in (*BOUNDS, "coverage")}


def assign_tuple(target: dict[str, np.ndarray], index: int, values: tuple[float, float, float, float]) -> None:
    for name, value in zip((*BOUNDS, "coverage"), values):
        target[name][index] = value


def compute_facets(features: list[FeatureRecord]) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    n = len(features)
    pair_i, pair_j = np.triu_indices(n, 1)
    m = len(pair_i)
    names = [
        "matrix_strict", "matrix_soft", "filler_strict", "filler_soft",
        "surface_strict", "surface_soft", "loading", "descriptor",
        "process_family", "step_type", "sequence", "settings_all", "settings_numeric",
    ]
    facets = {name: empty_facet_arrays(m) for name in names}
    scales = numeric_scales(features)
    for index, (i, j) in enumerate(zip(pair_i, pair_j)):
        left, right = features[i], features[j]
        assign_tuple(facets["matrix_strict"], index, known_or_missing_set(
            left.matrix, right.matrix, "Matrix" in left.role_reported, "Matrix" in right.role_reported, False
        ))
        assign_tuple(facets["matrix_soft"], index, known_or_missing_set(
            left.matrix, right.matrix, "Matrix" in left.role_reported, "Matrix" in right.role_reported, True
        ))
        assign_tuple(facets["filler_strict"], index, known_or_missing_set(
            left.filler, right.filler, "Filler" in left.role_reported, "Filler" in right.role_reported, False
        ))
        assign_tuple(facets["filler_soft"], index, known_or_missing_set(
            left.filler, right.filler, "Filler" in left.role_reported, "Filler" in right.role_reported, True
        ))
        assign_tuple(facets["surface_strict"], index, known_or_missing_set(
            left.surface, right.surface, "Surface Treatment" in left.role_reported,
            "Surface Treatment" in right.role_reported, False
        ))
        assign_tuple(facets["surface_soft"], index, known_or_missing_set(
            left.surface, right.surface, "Surface Treatment" in left.role_reported,
            "Surface Treatment" in right.role_reported, True
        ))
        assign_tuple(facets["loading"], index, map_interval_distance(left.loading, right.loading, scales))
        assign_tuple(facets["descriptor"], index, map_interval_distance(left.descriptor, right.descriptor, scales))

        family_d = exact_set_distance(left.process_family, right.process_family)
        assign_tuple(facets["process_family"], index, (family_d, family_d, family_d, 1.0))
        type_d = exact_set_distance(left.step_types, right.step_types)
        assign_tuple(facets["step_type"], index, (type_d, type_d, type_d, 1.0))
        sequence_d = sequence_edit_distance(left.sequence, right.sequence)
        assign_tuple(facets["sequence"], index, (sequence_d, sequence_d, sequence_d, 1.0))
        assign_tuple(facets["settings_all"], index, map_interval_distance(
            left.settings_all, right.settings_all, scales
        ))
        assign_tuple(facets["settings_numeric"], index, map_interval_distance(
            left.settings_numeric, right.settings_numeric, scales
        ))
    return facets, pair_i, pair_j


def weighted_composite(
    facets: dict[str, dict[str, np.ndarray]], weights: dict[str, float], bound: str
) -> np.ndarray:
    keys = list(weights)
    values = np.vstack([facets[key][bound] for key in keys]).astype(float)
    w = np.asarray([weights[key] for key in keys], dtype=float)[:, None]
    if bound == "reported":
        valid = np.isfinite(values)
        numerator = np.nansum(values * w, axis=0)
        denominator = np.sum(w * valid, axis=0)
        result = np.divide(numerator, denominator, out=np.full(values.shape[1], 0.5), where=denominator > 0)
    else:
        result = np.sum(values * w, axis=0) / np.sum(w)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def blend(material: np.ndarray, process: np.ndarray, process_lambda: float) -> np.ndarray:
    return ((1.0 - process_lambda) * material + process_lambda * process).astype(np.float32)


def pair_metadata(records: list[dict[str, Any]], pair_i: np.ndarray, pair_j: np.ndarray) -> dict[str, np.ndarray]:
    paper = np.asarray([record["paper_group"] for record in records], dtype=object)

    def author_set(record: dict[str, Any]) -> set[str]:
        metadata = record.get("citation_metadata") or {}
        return {norm_text(name) for name in metadata.get("authors", []) if name}

    authors = [author_set(record) for record in records]
    locations = [norm_text((record.get("citation_metadata") or {}).get("location") or "") for record in records]
    same_paper = paper[pair_i] == paper[pair_j]
    shared_author = np.fromiter(
        (bool(authors[i] and authors[j] and authors[i] & authors[j]) for i, j in zip(pair_i, pair_j)),
        dtype=bool,
        count=len(pair_i),
    )
    same_location = np.fromiter(
        (bool(locations[i] and locations[i] == locations[j]) for i, j in zip(pair_i, pair_j)),
        dtype=bool,
        count=len(pair_i),
    )
    return {
        "same_paper": same_paper,
        "cross_paper": ~same_paper,
        "shared_author_cross_paper": shared_author & ~same_paper,
        "same_location_cross_paper": same_location & ~same_paper,
    }


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b and np.isfinite(b) else math.nan


def clustering_metrics(
    labels: np.ndarray,
    paper_labels: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_meta: dict[str, np.ndarray],
) -> dict[str, float]:
    co = labels[pair_i] == labels[pair_j]
    same = pair_meta["same_paper"]
    cross = pair_meta["cross_paper"]
    same_rate = float(np.mean(co[same]))
    cross_rate = float(np.mean(co[cross]))
    overall_same = float(np.mean(same))
    within_cluster_same = float(np.mean(same[co])) if np.any(co) else math.nan
    output = {
        "same_paper_cocluster": same_rate,
        "cross_paper_cocluster": cross_rate,
        "same_paper_recall_lift": safe_ratio(same_rate, cross_rate),
        "same_paper_precision": within_cluster_same,
        "same_paper_precision_lift": safe_ratio(within_cluster_same, overall_same),
        "paper_ari": float(adjusted_rand_score(paper_labels, labels)),
        "paper_nmi": float(normalized_mutual_info_score(paper_labels, labels)),
        "actual_clusters": int(len(set(labels))),
        "largest_cluster_fraction": float(max(np.bincount(labels)) / len(labels)),
    }
    for prefix, mask in (
        ("shared_author", pair_meta["shared_author_cross_paper"]),
        ("same_location", pair_meta["same_location_cross_paper"]),
    ):
        background = cross & ~mask
        positive_rate = float(np.mean(co[mask])) if np.any(mask) else math.nan
        background_rate = float(np.mean(co[background])) if np.any(background) else math.nan
        output[f"{prefix}_pair_count"] = int(mask.sum())
        output[f"{prefix}_cocluster"] = positive_rate
        output[f"{prefix}_lift"] = safe_ratio(positive_rate, background_rate)
    return output


def retrieval_metrics(distance: np.ndarray, records: list[dict[str, Any]]) -> dict[str, float]:
    matrix = squareform(distance.astype(float))
    np.fill_diagonal(matrix, np.inf)
    papers = np.asarray([record["paper_group"] for record in records], dtype=object)
    reciprocal, midranks, hit1, hit5, hit10 = [], [], [], [], []
    for i in range(len(records)):
        relevant = (papers == papers[i])
        relevant[i] = False
        if not np.any(relevant):
            continue
        best = float(np.min(matrix[i, relevant]))
        rank_min = 1 + int(np.sum(matrix[i] < best - 1e-12))
        rank_max = int(np.sum(matrix[i] <= best + 1e-12))
        midrank = 0.5 * (rank_min + rank_max)
        reciprocal.append(1.0 / midrank)
        midranks.append(midrank)
        hit1.append(rank_min <= 1)
        hit5.append(rank_min <= 5)
        hit10.append(rank_min <= 10)
    return {
        "queries": len(reciprocal),
        "mrr_midrank": float(np.mean(reciprocal)),
        "median_first_same_paper_midrank": float(np.median(midranks)),
        "hit_at_1_optimistic_ties": float(np.mean(hit1)),
        "hit_at_5_optimistic_ties": float(np.mean(hit5)),
        "hit_at_10_optimistic_ties": float(np.mean(hit10)),
    }


def missing_mask_distance(features: list[FeatureRecord]) -> np.ndarray:
    pair_i, pair_j = np.triu_indices(len(features), 1)
    return np.fromiter(
        (jaccard_distance(features[i].missing_mask, features[j].missing_mask) for i, j in zip(pair_i, pair_j)),
        dtype=np.float32,
        count=len(pair_i),
    )


def square_subset(condensed: np.ndarray, indices: np.ndarray) -> np.ndarray:
    matrix = squareform(condensed)
    return squareform(matrix[np.ix_(indices, indices)], checks=False)
