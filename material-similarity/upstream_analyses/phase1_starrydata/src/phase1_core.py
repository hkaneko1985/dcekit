#!/usr/bin/env python3
"""Core utilities for property-free materials similarity Phase 1."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score


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
ELEMENT_INDEX = {symbol: i for i, symbol in enumerate(ELEMENTS)}


def _periodic_positions() -> dict[str, tuple[int, int]]:
    rows = {
        1: {1: "H", 18: "He"},
        2: {1: "Li", 2: "Be", 13: "B", 14: "C", 15: "N", 16: "O", 17: "F", 18: "Ne"},
        3: {1: "Na", 2: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar"},
        4: dict(zip(range(1, 19), ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"])),
        5: dict(zip(range(1, 19), ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"])),
        6: {1: "Cs", 2: "Ba", 4: "Hf", 5: "Ta", 6: "W", 7: "Re", 8: "Os", 9: "Ir", 10: "Pt", 11: "Au", 12: "Hg", 13: "Tl", 14: "Pb", 15: "Bi", 16: "Po", 17: "At", 18: "Rn"},
        7: {1: "Fr", 2: "Ra", 4: "Rf", 5: "Db", 6: "Sg", 7: "Bh", 8: "Hs", 9: "Mt", 10: "Ds", 11: "Rg", 12: "Cn", 13: "Nh", 14: "Fl", 15: "Mc", 16: "Lv", 17: "Ts", 18: "Og"},
    }
    result: dict[str, tuple[int, int]] = {}
    for period, row in rows.items():
        for group, symbol in row.items():
            result[symbol] = (period, group)
    for symbol in ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]:
        result[symbol] = (6, 3)
    for symbol in ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]:
        result[symbol] = (7, 3)
    return result


PERIOD_GROUP = _periodic_positions()
SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
NULL_VALUES = {"", "none", "null", "nan", "na", "n/a", "unknown", "not reported", "-"}


def stable_hash(value: str, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def normalize_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def normalize_category(value: Any) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if text.casefold() in NULL_VALUES:
        return None
    text = re.sub(r"\s+", " ", text)
    return text


def category_from_entry(entry: Any) -> str | None:
    if isinstance(entry, dict):
        return normalize_category(entry.get("category"))
    return normalize_category(entry)


def _read_number(text: str, position: int) -> tuple[float, int]:
    match = re.match(r"(?:\d+(?:\.\d*)?|\.\d+)", text[position:])
    if not match:
        return 1.0, position
    return float(match.group(0)), position + len(match.group(0))


def parse_formula(formula: Any) -> dict[str, float] | None:
    """Parse a closed numeric chemical formula, including nested groups and hydrates.

    Variable compositions, charges, ranges, free text, and mixtures joined by
    plus/minus signs are rejected rather than guessed.
    """
    if formula is None:
        return None
    text = unicodedata.normalize("NFKC", str(formula)).translate(SUBSCRIPT_TRANSLATION)
    text = re.sub(r"\s+", "", text)
    if not text or any(marker in text for marker in ("±", "~", "/", "|", ":", ",", ";", "=", "_", "−", "–", "—", "+", "-")):
        return None
    text = text.replace("∙", "·").replace("•", "·")

    def parse_group(segment: str, start: int = 0, closing: str | None = None):
        counts: defaultdict[str, float] = defaultdict(float)
        i = start
        pairs = {"(": ")", "[": "]", "{": "}"}
        while i < len(segment):
            char = segment[i]
            if closing and char == closing:
                return counts, i + 1
            if char in pairs:
                inner, i = parse_group(segment, i + 1, pairs[char])
                multiplier, i = _read_number(segment, i)
                for symbol, amount in inner.items():
                    counts[symbol] += amount * multiplier
                continue
            if char.isupper():
                symbol = char
                i += 1
                if i < len(segment) and segment[i].islower():
                    symbol += segment[i]
                    i += 1
                if symbol not in ELEMENT_INDEX:
                    raise ValueError("unknown element")
                amount, i = _read_number(segment, i)
                counts[symbol] += amount
                continue
            raise ValueError("unsupported token")
        if closing:
            raise ValueError("unclosed group")
        return counts, i

    total: defaultdict[str, float] = defaultdict(float)
    try:
        for segment in text.split("·"):
            if not segment:
                return None
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
    if not total or any(not np.isfinite(value) or value <= 0 for value in total.values()):
        return None
    scale = sum(total.values())
    return {symbol: amount / scale for symbol, amount in total.items()}


def composition_matrices(compositions: Iterable[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    compositions = list(compositions)
    n = len(compositions)
    exact = np.zeros((n, len(ELEMENTS)), dtype=np.float32)
    groups = np.zeros((n, 18), dtype=np.float32)
    periods = np.zeros((n, 7), dtype=np.float32)
    binary = np.zeros((n, len(ELEMENTS)), dtype=np.float32)
    for row, composition in enumerate(compositions):
        for symbol, fraction in composition.items():
            index = ELEMENT_INDEX[symbol]
            exact[row, index] = fraction
            binary[row, index] = 1.0
            period, group = PERIOD_GROUP[symbol]
            groups[row, group - 1] += fraction
            periods[row, period - 1] += fraction
    return exact, groups, periods, binary


def jaccard_distance_matrix(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    present = binary.astype(np.float32, copy=False)
    counts = present.sum(axis=1)
    intersection = present @ present.T
    union = counts[:, None] + counts[None, :] - intersection
    available = (counts[:, None] > 0) & (counts[None, :] > 0)
    distance = np.zeros_like(intersection, dtype=np.float32)
    np.divide(intersection, union, out=distance, where=union > 0)
    distance = 1.0 - distance
    distance[~available] = 0.0
    np.fill_diagonal(distance, 0.0)
    return distance, available


def composition_distance_matrix(exact: np.ndarray, groups: np.ndarray, periods: np.ndarray, binary: np.ndarray) -> np.ndarray:
    exact_l1 = cdist(exact, exact, metric="cityblock").astype(np.float32) * 0.5
    group_l1 = cdist(groups, groups, metric="cityblock").astype(np.float32) * 0.5
    period_l1 = cdist(periods, periods, metric="cityblock").astype(np.float32) * 0.5
    jaccard, _ = jaccard_distance_matrix(binary)
    distance = 0.50 * exact_l1 + 0.25 * group_l1 + 0.15 * period_l1 + 0.10 * jaccard
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32, copy=False)


def multilabel_matrix(values: list[set[str]]) -> np.ndarray:
    vocabulary = sorted(set().union(*values)) if values else []
    index = {value: i for i, value in enumerate(vocabulary)}
    matrix = np.zeros((len(values), len(vocabulary)), dtype=np.float32)
    for row, labels in enumerate(values):
        for label in labels:
            matrix[row, index[label]] = 1.0
    return matrix


def categorical_distance(values: list[set[str]]) -> tuple[np.ndarray, np.ndarray]:
    return jaccard_distance_matrix(multilabel_matrix(values))


def numeric_distance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(values)
    available = observed[:, None] & observed[None, :]
    finite = values[observed]
    if finite.size < 2:
        return np.zeros((len(values), len(values)), dtype=np.float32), available
    low, high = np.quantile(finite, [0.05, 0.95])
    scale = max(float(high - low), 1e-12)
    safe = np.where(observed, values, 0.0)
    distance = np.abs(safe[:, None] - safe[None, :]) / scale
    distance = np.clip(distance, 0.0, 1.0).astype(np.float32)
    distance[~available] = 0.0
    np.fill_diagonal(distance, 0.0)
    return distance, available


def mix_distances(composition: np.ndarray, blocks: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    total = composition.astype(np.float32, copy=True)
    denominator = np.ones_like(total, dtype=np.float32)
    for distance, available in blocks:
        total += np.where(available, distance, 0.0).astype(np.float32)
        denominator += available.astype(np.float32)
    total /= denominator
    total = (total + total.T) * 0.5
    np.fill_diagonal(total, 0.0)
    return total


def cluster_grid(distance: np.ndarray, cluster_counts: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average", optimal_ordering=False)
    cuts = cut_tree(tree, n_clusters=cluster_counts)
    labels = {int(k): cuts[:, column].astype(np.int32) for column, k in enumerate(cluster_counts)}
    return tree, labels


def internal_cluster_metrics(
    distance: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    development_indices: np.ndarray,
    seed: int,
    silhouette_sample: int = 1500,
) -> list[dict[str, float | int]]:
    output = []
    dev = np.asarray(development_indices, dtype=int)
    ddev = distance[np.ix_(dev, dev)]
    for k, labels_all in labels_by_k.items():
        labels = labels_all[dev]
        counts = np.bincount(labels)
        score = silhouette_score(
            ddev,
            labels,
            metric="precomputed",
            sample_size=min(silhouette_sample, len(dev)),
            random_state=seed,
        )
        output.append({
            "requested_clusters": int(k),
            "actual_clusters": int(len(counts)),
            "silhouette": float(score),
            "largest_cluster_fraction": float(counts.max() / len(labels)),
            "singleton_sample_fraction": float(counts[counts == 1].sum() / len(labels)),
        })
    return output


def select_cluster_count(metrics: list[dict[str, float | int]]) -> int:
    eligible = [m for m in metrics if float(m["largest_cluster_fraction"]) <= 0.35]
    candidates = eligible or metrics
    selected = max(candidates, key=lambda m: (float(m["silhouette"]), -int(m["requested_clusters"])))
    return int(selected["requested_clusters"])


def same_group_pairs(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_group: defaultdict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        by_group[str(group)].append(index)
    first: list[int] = []
    second: list[int] = []
    pair_group: list[str] = []
    for group, indices in by_group.items():
        if len(indices) < 2:
            continue
        for i, j in itertools.combinations(indices, 2):
            first.append(i)
            second.append(j)
            pair_group.append(group)
    return np.asarray(first, dtype=np.int32), np.asarray(second, dtype=np.int32), np.asarray(pair_group, dtype=object)


def random_unlabeled_pairs(group_ids: np.ndarray, target: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(group_ids)
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    gathered = 0
    while gathered < target:
        batch = min(max((target - gathered) * 3, 10000), 500000)
        i = rng.integers(0, n, size=batch, dtype=np.int32)
        j = rng.integers(0, n, size=batch, dtype=np.int32)
        keep = (i != j) & (group_ids[i] != group_ids[j])
        i, j = i[keep], j[keep]
        swap = i > j
        ii = np.where(swap, j, i)
        jj = np.where(swap, i, j)
        take = min(target - gathered, len(ii))
        first.append(ii[:take])
        second.append(jj[:take])
        gathered += take
    return np.concatenate(first), np.concatenate(second)


def composition_bin_edges(unlabeled_distances: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(unlabeled_distances, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def evaluate_coassignment(
    labels: np.ndarray,
    positive_i: np.ndarray,
    positive_j: np.ndarray,
    unlabeled_i: np.ndarray,
    unlabeled_j: np.ndarray,
    positive_comp_distance: np.ndarray,
    unlabeled_comp_distance: np.ndarray,
    edges: np.ndarray,
    min_unlabeled_per_bin: int = 30,
) -> dict[str, Any]:
    positive_same = labels[positive_i] == labels[positive_j]
    unlabeled_same = labels[unlabeled_i] == labels[unlabeled_j]
    positive_bins = np.digitize(positive_comp_distance, edges[1:-1], right=False)
    unlabeled_bins = np.digitize(unlabeled_comp_distance, edges[1:-1], right=False)
    global_baseline = float(unlabeled_same.mean())
    expected = np.zeros(len(positive_i), dtype=float)
    bin_records = []
    for bin_id in range(len(edges) - 1):
        p_mask = positive_bins == bin_id
        u_mask = unlabeled_bins == bin_id
        count = int(u_mask.sum())
        rate = float(unlabeled_same[u_mask].mean()) if count >= min_unlabeled_per_bin else global_baseline
        expected[p_mask] = rate
        bin_records.append({
            "bin": int(bin_id),
            "positive_pairs": int(p_mask.sum()),
            "unlabeled_pairs": count,
            "unlabeled_coassignment_rate": rate,
        })
    observed = float(positive_same.mean())
    baseline = float(expected.mean())
    lift = observed / max(baseline, 1e-12)
    return {
        "positive_pairs": int(len(positive_i)),
        "unlabeled_pairs": int(len(unlabeled_i)),
        "positive_coassignment_rate": observed,
        "composition_matched_unlabeled_rate": baseline,
        "lift": float(lift),
        "log_lift": float(math.log(max(lift, 1e-12))),
        "positive_same": positive_same,
        "expected_per_positive": expected,
        "bins": bin_records,
    }


def bootstrap_log_lift(
    positive_same: np.ndarray,
    expected: np.ndarray,
    pair_groups: np.ndarray,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(pair_groups)
    by_group = {group: np.flatnonzero(pair_groups == group) for group in unique_groups}
    values = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        observed = float(positive_same[indices].mean())
        baseline = float(expected[indices].mean())
        values[repetition] = math.log(max(observed / max(baseline, 1e-12), 1e-12))
    return values


def paper_equal_log_lift(
    positive_same: np.ndarray,
    expected: np.ndarray,
    pair_groups: np.ndarray,
) -> dict[str, Any]:
    """Give every paper equal weight regardless of its number of sample pairs."""
    observations = []
    baselines = []
    for group in np.unique(pair_groups):
        mask = pair_groups == group
        observations.append(float(positive_same[mask].mean()))
        baselines.append(float(expected[mask].mean()))
    observed = float(np.mean(observations))
    baseline = float(np.mean(baselines))
    lift = observed / max(baseline, 1e-12)
    return {
        "papers": int(len(observations)),
        "paper_equal_positive_coassignment_rate": observed,
        "paper_equal_expected_rate": baseline,
        "paper_equal_lift": float(lift),
        "paper_equal_log_lift": float(math.log(max(lift, 1e-12))),
    }


def leave_one_group_out_delta(
    same_a: np.ndarray,
    expected_a: np.ndarray,
    same_b: np.ndarray,
    expected_b: np.ndarray,
    pair_groups: np.ndarray,
) -> dict[str, Any]:
    """Compute paper-equal log-lift A minus B after omitting each paper."""
    unique_groups = np.unique(pair_groups)
    obs_a = []
    exp_a = []
    obs_b = []
    exp_b = []
    for group in unique_groups:
        mask = pair_groups == group
        obs_a.append(float(same_a[mask].mean()))
        exp_a.append(float(expected_a[mask].mean()))
        obs_b.append(float(same_b[mask].mean()))
        exp_b.append(float(expected_b[mask].mean()))
    obs_a = np.asarray(obs_a)
    exp_a = np.asarray(exp_a)
    obs_b = np.asarray(obs_b)
    exp_b = np.asarray(exp_b)
    deltas = []
    keep_count = max(1, len(unique_groups) - 1)
    for index in range(len(unique_groups)):
        a_lift = ((obs_a.sum() - obs_a[index]) / keep_count) / max((exp_a.sum() - exp_a[index]) / keep_count, 1e-12)
        b_lift = ((obs_b.sum() - obs_b[index]) / keep_count) / max((exp_b.sum() - exp_b[index]) / keep_count, 1e-12)
        deltas.append(math.log(max(a_lift, 1e-12)) - math.log(max(b_lift, 1e-12)))
    values = np.asarray(deltas)
    return {
        "papers": int(len(unique_groups)),
        "minimum_delta_log_lift": float(values.min()),
        "maximum_delta_log_lift": float(values.max()),
        "median_delta_log_lift": float(np.median(values)),
        "papers_whose_omission_reverses_direction": int((values <= 0).sum()),
    }


def percentile_interval(values: np.ndarray, level: float = 0.95) -> list[float]:
    alpha = (1.0 - level) / 2.0
    return [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))]


def stratified_permutation_pvalue(
    labels: np.ndarray,
    strata: np.ndarray,
    positive_i: np.ndarray,
    positive_j: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    observed = float((labels[positive_i] == labels[positive_j]).mean())
    stratum_indices = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    rates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        permuted = labels.copy()
        for indices in stratum_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        rates[repetition] = float((permuted[positive_i] == permuted[positive_j]).mean())
    pvalue = (1.0 + float((rates >= observed).sum())) / (repetitions + 1.0)
    return pvalue, [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]


def subsample_stability(
    distance: np.ndarray,
    reference_labels: np.ndarray,
    cluster_count: int,
    subset_indices: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    subset_indices = np.asarray(subset_indices, dtype=int)
    base_distance = distance[np.ix_(subset_indices, subset_indices)]
    _, base_by_k = cluster_grid(base_distance, [cluster_count])
    base = base_by_k[cluster_count]
    scores = []
    take = max(cluster_count + 2, int(round(0.8 * len(subset_indices))))
    for _ in range(repetitions):
        local = np.sort(rng.choice(len(subset_indices), size=take, replace=False))
        replicate_distance = base_distance[np.ix_(local, local)]
        _, replicate = cluster_grid(replicate_distance, [cluster_count])
        scores.append(float(adjusted_rand_score(base[local], replicate[cluster_count])))
    return scores


def parse_relative_density(category: str | None) -> float:
    if not category:
        return float("nan")
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", category)
    if not match:
        return float("nan")
    value = float(match.group(1)) / 100.0
    return value if 0.0 < value <= 1.2 else float("nan")


def parse_grain_size_um(category: str | None) -> float:
    if not category:
        return float("nan")
    text = category.replace("μ", "u").replace("µ", "u").casefold()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(nm|um|mm|m)\b", text)
    if not match:
        return float("nan")
    value = float(match.group(1))
    factor = {"nm": 1e-3, "um": 1.0, "mm": 1e3, "m": 1e6}[match.group(2)]
    value_um = value * factor
    return math.log10(value_um) if value_um > 0 else float("nan")
