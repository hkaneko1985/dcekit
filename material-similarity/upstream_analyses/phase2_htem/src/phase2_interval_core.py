#!/usr/bin/env python3
"""Core methods for property-free HTEM interval similarities."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.metrics import adjusted_rand_score


ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
ELEMENT_INDEX = {symbol: index for index, symbol in enumerate(ELEMENTS)}
SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def periodic_positions() -> dict[str, tuple[int, int]]:
    rows = {
        1: {1: "H", 18: "He"},
        2: {1: "Li", 2: "Be", 13: "B", 14: "C", 15: "N", 16: "O", 17: "F", 18: "Ne"},
        3: {1: "Na", 2: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar"},
        4: dict(zip(range(1, 19), ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"])),
        5: dict(zip(range(1, 19), ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"])),
        6: {1: "Cs", 2: "Ba", 4: "Hf", 5: "Ta", 6: "W", 7: "Re", 8: "Os", 9: "Ir", 10: "Pt", 11: "Au", 12: "Hg", 13: "Tl", 14: "Pb", 15: "Bi", 16: "Po", 17: "At", 18: "Rn"},
        7: {1: "Fr", 2: "Ra", 4: "Rf", 5: "Db", 6: "Sg", 7: "Bh", 8: "Hs", 9: "Mt", 10: "Ds", 11: "Rg", 12: "Cn", 13: "Nh", 14: "Fl", 15: "Mc", 16: "Lv", 17: "Ts", 18: "Og"},
    }
    result = {}
    for period, row in rows.items():
        for group, symbol in row.items():
            result[symbol] = (period, group)
    for symbol in ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]:
        result[symbol] = (6, 3)
    for symbol in ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]:
        result[symbol] = (7, 3)
    return result


PERIOD_GROUP = periodic_positions()


FIELD_SPECS = {
    "deposition_compounds": ("source_energy", "categorical_array", False),
    "deposition_power": ("source_energy", "numeric_array", True),
    "deposition_target_pulses": ("source_energy", "numeric_array", True),
    "deposition_rep_rate": ("source_energy", "numeric_array", True),
    "deposition_energy": ("source_energy", "numeric", True),
    "deposition_base_pressure_mtorr": ("atmosphere", "numeric", True),
    "deposition_growth_pressure_mtorr": ("atmosphere", "numeric", True),
    "deposition_gases": ("atmosphere", "categorical_array", False),
    "deposition_gas_flow_sccm": ("atmosphere", "numeric_array", True),
    "deposition_initial_temp_c": ("thermal_temporal", "numeric", False),
    "deposition_sample_time_min": ("thermal_temporal", "numeric", True),
    "deposition_cycles": ("thermal_temporal", "numeric", True),
    "deposition_substrate_material": ("substrate_geometry", "categorical", False),
    "deposition_ts_distance": ("substrate_geometry", "numeric", True),
}
FACETS = ["source_energy", "atmosphere", "thermal_temporal", "substrate_geometry"]


def stable_hash(value: str, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.casefold() in {"null", "none", "nan", "n/a", "unknown"}:
        return None
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return value


def raw_slots(value: Any) -> list[Any]:
    value = parse_jsonish(value)
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_category(value: Any) -> str | None:
    value = parse_jsonish(value)
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    if not text or text in {"null", "none", "nan", "n/a", "unknown"}:
        return None
    return re.sub(r"\s+", " ", text)


def numeric_or_none(value: Any) -> float | None:
    value = parse_jsonish(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_number(text: str, position: int) -> tuple[float, int]:
    match = re.match(r"(?:\d+(?:\.\d*)?|\.\d+)", text[position:])
    if not match:
        return 1.0, position
    return float(match.group(0)), position + len(match.group(0))


def parse_formula(formula: Any) -> dict[str, float] | None:
    if formula is None:
        return None
    text = unicodedata.normalize("NFKC", str(formula)).translate(SUBSCRIPT_TRANSLATION)
    text = re.sub(r"\s+", "", text).replace("∙", "·").replace("•", "·")
    if not text or any(marker in text for marker in ("±", "~", "/", "|", ":", ",", ";", "=", "_", "−", "–", "—", "+", "-")):
        return None

    def parse_group(segment: str, start: int = 0, closing: str | None = None):
        counts: defaultdict[str, float] = defaultdict(float)
        pairs = {"(": ")", "[": "]", "{": "}"}
        index = start
        while index < len(segment):
            char = segment[index]
            if closing and char == closing:
                return counts, index + 1
            if char in pairs:
                inner, index = parse_group(segment, index + 1, pairs[char])
                multiplier, index = _read_number(segment, index)
                for symbol, amount in inner.items():
                    counts[symbol] += amount * multiplier
                continue
            if char.isupper():
                symbol = char
                index += 1
                if index < len(segment) and segment[index].islower():
                    symbol += segment[index]
                    index += 1
                if symbol not in ELEMENT_INDEX:
                    raise ValueError("unknown element")
                amount, index = _read_number(segment, index)
                counts[symbol] += amount
                continue
            raise ValueError("unsupported token")
        if closing:
            raise ValueError("unclosed group")
        return counts, index

    total: defaultdict[str, float] = defaultdict(float)
    try:
        for segment in text.split("·"):
            coefficient = 1.0
            match = re.match(r"(?:\d+(?:\.\d*)?|\.\d+)", segment)
            if match:
                coefficient = float(match.group(0))
                segment = segment[len(match.group(0)):]
            parsed, end = parse_group(segment)
            if end != len(segment) or not parsed:
                return None
            for symbol, amount in parsed.items():
                total[symbol] += coefficient * amount
    except (ValueError, OverflowError):
        return None
    scale = sum(total.values())
    if scale <= 0:
        return None
    return {symbol: amount / scale for symbol, amount in total.items()}


def vector_from_composition(composition: dict[str, float]) -> np.ndarray:
    vector = np.zeros(len(ELEMENTS), dtype=np.float32)
    for symbol, fraction in composition.items():
        if symbol in ELEMENT_INDEX:
            vector[ELEMENT_INDEX[symbol]] = float(fraction)
    if vector.sum() > 0:
        vector /= vector.sum()
    return vector


def composition_summary(record: dict[str, Any]) -> dict[str, Any]:
    measurements = []
    elements = {symbol for symbol in record.get("elements") or [] if symbol in ELEMENT_INDEX}
    for components in record.get("composition_measurements") or []:
        accumulator = np.zeros(len(ELEMENTS), dtype=np.float64)
        total_weight = 0.0
        for formula, raw_weight in components.items():
            parsed = parse_formula(formula)
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                continue
            if parsed is None or not math.isfinite(weight) or weight <= 0:
                continue
            accumulator += weight * vector_from_composition(parsed)
            total_weight += weight
            elements.update(parsed)
        if total_weight > 0 and accumulator.sum() > 0:
            accumulator /= accumulator.sum()
            measurements.append(accumulator.astype(np.float32))
    if measurements:
        matrix = np.vstack(measurements)
        mean = matrix.mean(axis=0).astype(np.float32)
        spread = matrix.std(axis=0).astype(np.float32)
        quantitative = True
    else:
        mean = np.zeros(len(ELEMENTS), dtype=np.float32)
        spread = np.zeros(len(ELEMENTS), dtype=np.float32)
        quantitative = False
    binary = np.zeros(len(ELEMENTS), dtype=np.float32)
    for symbol in elements:
        binary[ELEMENT_INDEX[symbol]] = 1.0
    return {
        "mean": mean,
        "spread": spread,
        "binary": binary,
        "quantitative": quantitative,
        "element_key": "-".join(sorted(elements)),
    }


def jaccard_distance(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(binary, dtype=np.float32)
    counts = binary.sum(axis=1)
    intersection = binary @ binary.T
    union = counts[:, None] + counts[None, :] - intersection
    available = (counts[:, None] > 0) & (counts[None, :] > 0)
    similarity = np.zeros_like(intersection, dtype=np.float32)
    np.divide(intersection, union, out=similarity, where=union > 0)
    distance = 1.0 - similarity
    distance[~available] = 0.0
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32), available


def composition_distance(summaries: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    mean = np.vstack([row["mean"] for row in summaries])
    spread = np.vstack([row["spread"] for row in summaries])
    binary = np.vstack([row["binary"] for row in summaries])
    quantitative = np.asarray([row["quantitative"] for row in summaries], dtype=bool)
    q_available = quantitative[:, None] & quantitative[None, :]
    groups = np.zeros((len(mean), 18), dtype=np.float32)
    periods = np.zeros((len(mean), 7), dtype=np.float32)
    for symbol, index in ELEMENT_INDEX.items():
        period, group = PERIOD_GROUP[symbol]
        groups[:, group - 1] += mean[:, index]
        periods[:, period - 1] += mean[:, index]
    set_distance, set_available = jaccard_distance(binary)
    exact = (0.5 * cdist(mean, mean, metric="cityblock")).astype(np.float32)
    group_d = (0.5 * cdist(groups, groups, metric="cityblock")).astype(np.float32)
    period_d = (0.5 * cdist(periods, periods, metric="cityblock")).astype(np.float32)
    spread_d = np.clip(cdist(spread, spread, metric="cityblock"), 0, 1).astype(np.float32)
    blocks = [
        (set_distance, set_available, 0.20),
        (exact, q_available, 0.40),
        (group_d, q_available, 0.20),
        (period_d, q_available, 0.10),
        (spread_d, q_available, 0.10),
    ]
    numerator = np.zeros_like(set_distance)
    denominator = np.zeros_like(set_distance)
    for distance, available, weight in blocks:
        numerator += np.where(available, distance * weight, 0).astype(np.float32)
        denominator += available.astype(np.float32) * weight
    output = np.zeros_like(set_distance)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    output = np.clip((output + output.T) / 2, 0, 1)
    np.fill_diagonal(output, 0)
    return output.astype(np.float32), {
        "quantitative_count": int(quantitative.sum()),
        "element_set_count": int(sum(bool(row["element_key"]) for row in summaries)),
        "quantitative": quantitative,
        "element_keys": np.asarray([row["element_key"] for row in summaries], dtype=object),
        "binary": binary,
    }


def field_pair_matrices(raw_values: list[Any], kind: str, log_transform: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(raw_values)
    declared = np.asarray([len(raw_slots(value)) for value in raw_values], dtype=np.float32)
    if kind.startswith("numeric"):
        parsed = [[numeric_or_none(value) for value in raw_slots(raw)] for raw in raw_values]
        reported_values = [[value for value in row if value is not None] for row in parsed]
        transformed = []
        for row in reported_values:
            current = []
            for value in row:
                if log_transform and value >= 0:
                    current.append(math.log1p(value))
                else:
                    current.append(value)
            transformed.append(sorted(current))
        flat = np.asarray([value for row in transformed for value in row], dtype=float)
        if len(flat) >= 2:
            low, high = np.quantile(flat, [0.05, 0.95])
            scale = max(float(high - low), 1e-12)
        else:
            scale = 1.0
        maximum = max((len(row) for row in transformed), default=0)
        matrix = np.full((n, maximum), np.nan, dtype=np.float32)
        for index, row in enumerate(transformed):
            if row:
                matrix[index, :len(row)] = row
        counts = np.asarray([len(row) for row in transformed], dtype=np.float32)
        common = np.minimum(counts[:, None], counts[None, :])
        numerator = np.zeros((n, n), dtype=np.float32)
        for column in range(maximum):
            values = matrix[:, column]
            available = np.isfinite(values)
            difference = np.clip(np.abs(values[:, None] - values[None, :]) / scale, 0, 1)
            numerator += np.where(available[:, None] & available[None, :], difference, 0).astype(np.float32)
        distance = np.zeros((n, n), dtype=np.float32)
        np.divide(numerator, common, out=distance, where=common > 0)
        reported_count = counts
        scale_value = scale
    else:
        normalized = [[value for raw in raw_slots(item) if (value := normalize_category(raw)) is not None] for item in raw_values]
        sets = [set(row) for row in normalized]
        vocabulary = sorted(set().union(*sets)) if sets else []
        index = {value: column for column, value in enumerate(vocabulary)}
        binary = np.zeros((n, len(vocabulary)), dtype=np.float32)
        for row_index, values in enumerate(sets):
            for value in values:
                binary[row_index, index[value]] = 1
        distance, _ = jaccard_distance(binary)
        reported_count = np.asarray([len(row) for row in normalized], dtype=np.float32)
        common = np.minimum(reported_count[:, None], reported_count[None, :])
        scale_value = None
    denominator = np.maximum(declared[:, None], declared[None, :])
    coverage = np.zeros((n, n), dtype=np.float32)
    np.divide(common, denominator, out=coverage, where=denominator > 0)
    coverage = np.clip((coverage + coverage.T) / 2, 0, 1)
    distance = np.clip((distance + distance.T) / 2, 0, 1)
    np.fill_diagonal(distance, 0)
    return distance.astype(np.float32), coverage.astype(np.float32), {
        "records_reported": int(np.sum(reported_count > 0)),
        "records_partially_reported": int(np.sum((reported_count > 0) & (reported_count < declared))),
        "declared_slots": int(np.sum(declared)),
        "reported_slots": int(np.sum(reported_count)),
        "robust_scale_after_transform": scale_value,
    }


def build_field_matrices(records: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    distances = {}
    coverage = {}
    audit = {}
    for field, (_, kind, log_transform) in FIELD_SPECS.items():
        values = [(row.get("process") or {}).get(field) for row in records]
        distance, q, info = field_pair_matrices(values, kind, log_transform)
        distances[field] = distance
        coverage[field] = q
        audit[field] = info
    return distances, coverage, audit


def process_interval_distances(
    field_distances: dict[str, np.ndarray],
    field_coverage: dict[str, np.ndarray],
    facet_weights: Iterable[float],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    facet_weights = list(map(float, facet_weights))
    if len(facet_weights) != len(FACETS) or any(value < 0 for value in facet_weights) or not np.isclose(sum(facet_weights), 1):
        raise ValueError("Facet weights must be four nonnegative values summing to one")
    field_weights = {}
    for facet, facet_weight in zip(FACETS, facet_weights):
        fields = [field for field, (owner, _, _) in FIELD_SPECS.items() if owner == facet]
        for field in fields:
            field_weights[field] = facet_weight / len(fields)
    shape = next(iter(field_distances.values())).shape
    supported = np.zeros(shape, dtype=np.float32)
    q_total = np.zeros(shape, dtype=np.float32)
    for field, weight in field_weights.items():
        q = field_coverage[field]
        supported += (weight * q * (1.0 - field_distances[field])).astype(np.float32)
        q_total += (weight * q).astype(np.float32)
    possible = np.clip(supported + (1.0 - q_total), 0, 1)
    reported_similarity = np.ones(shape, dtype=np.float32)
    np.divide(supported, q_total, out=reported_similarity, where=q_total > 0)
    distances = {
        "optimistic": np.clip(1.0 - possible, 0, 1).astype(np.float32),
        "reported": np.clip(1.0 - reported_similarity, 0, 1).astype(np.float32),
        "pessimistic": np.clip(1.0 - supported, 0, 1).astype(np.float32),
    }
    for matrix in distances.values():
        matrix[:] = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
    if np.max(distances["optimistic"] - distances["pessimistic"]) > 1e-6:
        raise AssertionError("Interval ordering failed")
    return distances, q_total, field_weights


def material_distance(composition: np.ndarray, process: np.ndarray, process_weight: float) -> np.ndarray:
    result = composition + (1.0 - composition) * float(process_weight) * process
    result = np.clip((result + result.T) / 2, 0, 1).astype(np.float32)
    np.fill_diagonal(result, 0)
    return result


def reporting_mask_distance(field_coverage: dict[str, np.ndarray]) -> np.ndarray:
    fields = list(FIELD_SPECS)
    # Self-coverage > 0 means at least one reported slot. Joint absences are
    # excluded by Jaccard, so they never add similarity.
    reported = np.column_stack([np.diag(field_coverage[field]) > 0 for field in fields]).astype(np.float32)
    distance, _ = jaccard_distance(reported)
    return distance


def cluster_grid(distance: np.ndarray, cluster_counts: list[int], ids: np.ndarray, order_seed: int) -> dict[int, np.ndarray]:
    order = np.argsort([stable_hash(f"row|{value}", order_seed) for value in ids])
    ordered = distance[np.ix_(order, order)]
    tree = linkage(squareform(ordered, checks=False), method="average", optimal_ordering=False)
    cuts = cut_tree(tree, n_clusters=cluster_counts)
    output = {}
    for column, count in enumerate(cluster_counts):
        labels = np.empty(len(ids), dtype=np.int32)
        labels[order] = cuts[:, column].astype(np.int32)
        output[int(count)] = labels
    return output


def study_positive_instances(studies: list[dict[str, Any]], id_to_index: dict[int, int]) -> dict[str, Any]:
    members = {}
    for study in studies:
        study_id = str(study.get("study_id", study.get("id")))
        values = sorted({id_to_index[int(value)] for value in study.get("sample_library") or [] if int(value) in id_to_index})
        if len(values) >= 2:
            members[study_id] = values
    parent = {key: key for key in members}

    def find(value: str) -> str:
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_library = defaultdict(list)
    for study, values in members.items():
        for value in values:
            by_library[value].append(study)
    for values in by_library.values():
        for value in values[1:]:
            union(values[0], value)
    groups = defaultdict(list)
    for study in members:
        groups[find(study)].append(study)
    component = {}
    for values in groups.values():
        name = "component:" + "+".join(sorted(values, key=lambda value: int(value)))
        for study in values:
            component[study] = name

    pair_i, pair_j, pair_study, pair_component = [], [], [], []
    for study in sorted(members, key=int):
        for left, right in itertools.combinations(members[study], 2):
            pair_i.append(left); pair_j.append(right)
            pair_study.append(study); pair_component.append(component[study])
    unique_pairs = {(min(i, j), max(i, j)) for i, j in zip(pair_i, pair_j)}
    return {
        "pair_i": np.asarray(pair_i, dtype=np.int32),
        "pair_j": np.asarray(pair_j, dtype=np.int32),
        "pair_study": np.asarray(pair_study, dtype=object),
        "pair_component": np.asarray(pair_component, dtype=object),
        "members_by_study": members,
        "components": sorted(set(component.values())),
        "unique_pairs": unique_pairs,
    }


def random_unlabeled_pairs(n: int, positives: set[tuple[int, int]], target: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected = set()
    while len(selected) < target:
        batch = max(10000, (target - len(selected)) * 2)
        left = rng.integers(0, n, batch)
        right = rng.integers(0, n, batch)
        for a, b in zip(left, right):
            if a == b:
                continue
            pair = (int(min(a, b)), int(max(a, b)))
            if pair not in positives:
                selected.add(pair)
            if len(selected) >= target:
                break
    ordered = sorted(selected)
    return np.asarray([x[0] for x in ordered], dtype=np.int32), np.asarray([x[1] for x in ordered], dtype=np.int32)


def composition_bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], np.inf)
    return edges


def prepare_evaluation_design(
    positives: dict[str, Any],
    unlabeled_i: np.ndarray,
    unlabeled_j: np.ndarray,
    composition: np.ndarray,
    bins: int,
    positive_source_strata: np.ndarray | None = None,
    unlabeled_source_strata: np.ndarray | None = None,
) -> dict[str, Any]:
    pi, pj = positives["pair_i"], positives["pair_j"]
    positive_c = composition[pi, pj]
    unlabeled_c = composition[unlabeled_i, unlabeled_j]
    edges = composition_bin_edges(unlabeled_c, bins)
    positive_bins = np.digitize(positive_c, edges[1:-1]).astype(np.int32)
    unlabeled_bins = np.digitize(unlabeled_c, edges[1:-1]).astype(np.int32)
    output = {
        "positive_bins": positive_bins,
        "unlabeled_bins": unlabeled_bins,
        "bins": int(bins),
    }
    if positive_source_strata is not None and unlabeled_source_strata is not None:
        positive_source_strata = np.asarray(positive_source_strata, dtype=object)
        unlabeled_source_strata = np.asarray(unlabeled_source_strata, dtype=object)
        levels = sorted(set(map(str, positive_source_strata)) | set(map(str, unlabeled_source_strata)))
        index = {value: code for code, value in enumerate(levels)}
        positive_source = np.asarray([index[str(value)] for value in positive_source_strata], dtype=np.int32)
        unlabeled_source = np.asarray([index[str(value)] for value in unlabeled_source_strata], dtype=np.int32)
        output.update({
            "positive_source": positive_source,
            "unlabeled_source": unlabeled_source,
            "positive_source_bin": positive_source * bins + positive_bins,
            "unlabeled_source_bin": unlabeled_source * bins + unlabeled_bins,
            "source_levels": levels,
        })
    return output


def evaluate_labels(
    labels: np.ndarray,
    positives: dict[str, Any],
    unlabeled_i: np.ndarray,
    unlabeled_j: np.ndarray,
    composition: np.ndarray,
    bins: int,
    positive_subsets: dict[str, np.ndarray] | None = None,
    positive_source_strata: np.ndarray | None = None,
    unlabeled_source_strata: np.ndarray | None = None,
    prepared_design: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pi, pj = positives["pair_i"], positives["pair_j"]
    positive_same = labels[pi] == labels[pj]
    unlabeled_same = labels[unlabeled_i] == labels[unlabeled_j]
    design = prepared_design or prepare_evaluation_design(
        positives, unlabeled_i, unlabeled_j, composition, bins,
        positive_source_strata, unlabeled_source_strata,
    )
    pb = design["positive_bins"]
    ub = design["unlabeled_bins"]
    global_rate = float(np.mean(unlabeled_same))
    bin_counts = np.bincount(ub, minlength=bins)
    bin_sums = np.bincount(ub, weights=unlabeled_same.astype(float), minlength=bins)
    bin_rates = np.full(bins, global_rate, dtype=float)
    np.divide(bin_sums, bin_counts, out=bin_rates, where=bin_counts >= 50)
    expected = bin_rates[pb]

    expected_source = expected.copy()
    source_fallback = {"exact_source_and_bin": 0, "source_only": 0, "composition_bin_only": 0}
    if "positive_source" in design:
        ps = design["positive_source"]
        us = design["unlabeled_source"]
        pse = design["positive_source_bin"]
        use = design["unlabeled_source_bin"]
        source_count = len(design["source_levels"])
        exact_size = source_count * bins
        exact_counts = np.bincount(use, minlength=exact_size)
        exact_sums = np.bincount(use, weights=unlabeled_same.astype(float), minlength=exact_size)
        exact_rates = np.zeros(exact_size, dtype=float)
        np.divide(exact_sums, exact_counts, out=exact_rates, where=exact_counts > 0)
        source_counts = np.bincount(us, minlength=source_count)
        source_sums = np.bincount(us, weights=unlabeled_same.astype(float), minlength=source_count)
        source_rates = np.zeros(source_count, dtype=float)
        np.divide(source_sums, source_counts, out=source_rates, where=source_counts > 0)
        exact_ok = exact_counts[pse] >= 50
        source_ok = (~exact_ok) & (source_counts[ps] >= 50)
        bin_only = ~(exact_ok | source_ok)
        expected_source[exact_ok] = exact_rates[pse[exact_ok]]
        expected_source[source_ok] = source_rates[ps[source_ok]]
        expected_source[bin_only] = bin_rates[pb[bin_only]]
        source_fallback = {
            "exact_source_and_bin": int(np.sum(exact_ok)),
            "source_only": int(np.sum(source_ok)),
            "composition_bin_only": int(np.sum(bin_only)),
        }

    def summary(mask: np.ndarray, expectation: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {"pairs": 0, "positive_coassignment": None, "expected": None, "excess": None, "component_equal_excess": None, "component_excesses": {}}
        component_scores = []
        component_score_map = {}
        for component in np.unique(positives["pair_component"][mask]):
            use_component = mask & (positives["pair_component"] == component)
            study_scores = []
            for study in np.unique(positives["pair_study"][use_component]):
                use = use_component & (positives["pair_study"] == study)
                study_scores.append(float(np.mean(positive_same[use]) - np.mean(expectation[use])))
            if study_scores:
                value = float(np.mean(study_scores))
                component_scores.append(value)
                component_score_map[str(component)] = value
        observed = float(np.mean(positive_same[mask]))
        baseline = float(np.mean(expectation[mask]))
        return {
            "pairs": int(np.sum(mask)),
            "positive_coassignment": observed,
            "expected": baseline,
            "excess": observed - baseline,
            "lift": observed / max(baseline, 1e-12),
            "component_equal_excess": float(np.mean(component_scores)) if component_scores else None,
            "component_count": len(component_scores),
            "component_excesses": component_score_map,
        }

    all_mask = np.ones(len(pi), dtype=bool)
    output = {
        "all": summary(all_mask, expected),
        "all_source_matched": summary(all_mask, expected_source),
        "subsets": {},
        "subsets_source_matched": {},
        "source_matching_fallback_counts": source_fallback,
    }
    for name, mask in (positive_subsets or {}).items():
        mask = np.asarray(mask, dtype=bool)
        output["subsets"][name] = summary(mask, expected)
        output["subsets_source_matched"][name] = summary(mask, expected_source)
    unique, counts = np.unique(labels, return_counts=True)
    output["clusters"] = {
        "actual": int(len(unique)),
        "largest_fraction": float(counts.max() / len(labels)),
        "singleton_fraction": float(counts[counts == 1].sum() / len(labels)),
    }
    return output


def retrieval_metrics(distance: np.ndarray, positives: dict[str, Any], top_k: int) -> dict[str, Any]:
    relevant: dict[int, set[int]] = defaultdict(set)
    for values in positives["members_by_study"].values():
        for anchor in values:
            relevant[anchor].update(value for value in values if value != anchor)
    reciprocal, hits, recalls = [], [], []
    for anchor, targets in relevant.items():
        order = np.argsort(distance[anchor], kind="stable")
        order = [int(value) for value in order if int(value) != anchor]
        ranks = [rank for rank, value in enumerate(order, start=1) if value in targets]
        first = min(ranks) if ranks else len(order) + 1
        reciprocal.append(1.0 / first)
        top = set(order[:top_k])
        hits.append(float(bool(top & targets)))
        recalls.append(len(top & targets) / len(targets))
    return {
        "anchors": len(relevant),
        "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        f"hit_at_{top_k}": float(np.mean(hits)) if hits else None,
        f"recall_at_{top_k}": float(np.mean(recalls)) if recalls else None,
    }


def row_order_stability(distance: np.ndarray, ids: np.ndarray, clusters: int, seeds: list[int]) -> dict[str, Any]:
    label_sets = [cluster_grid(distance, [clusters], ids, seed)[clusters] for seed in seeds]
    values = [adjusted_rand_score(label_sets[i], label_sets[j]) for i in range(len(label_sets)) for j in range(i + 1, len(label_sets))]
    return {"pairwise_ari": values, "median_ari": float(np.median(values)) if values else 1.0}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
