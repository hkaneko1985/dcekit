"""Coordinate-wise Kohler-type extension of a fixed vector-valued function.

All unordered pairs in the declared composition schema are retained, including
zero blocks for absent pairs. A source-vocabulary failure is never zero-filled.
Empirical pair support is supplied separately; it is not inferred from vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np

from .composition import _names, audit_and_align


@dataclass
class DescriptorResult:
    components: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    composition: np.ndarray
    whole: np.ndarray
    vertices: np.ndarray
    unary: np.ndarray
    pair_raw: np.ndarray
    pair_excess: np.ndarray
    boundary: np.ndarray
    pair_indices: np.ndarray
    pair_active: np.ndarray
    pair_vocabulary_supported: np.ndarray
    pair_empirical_support: np.ndarray
    row_vocabulary_supported: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        """Arrays suitable for np.savez, with explicit string/numeric dtypes."""
        return {k: np.asarray(v) for k, v in vars(self).items()}


def boundary_extension(
    X: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    *,
    components: Sequence[str],
    coordinate_names: Sequence[str],
    available_components: Iterable[str] | None = None,
    observed_pairs: Iterable[Sequence[str]] | None = None,
) -> DescriptorResult:
    """Evaluate whole compositions and every normalized active binary pair.

    ``evaluate`` accepts compositions in ``components`` order and returns an
    (n, q) matrix on one explicitly chosen scale. Endpoints are predictions from
    that same function, never measured pure-component properties.

    ``observed_pairs=None`` means empirical support has not been audited (-1).
    Supplying a complete set marks listed pairs 1 and other pairs 0. This records
    occurrence only, not pair-ratio coverage. Unsupported active pair values are
    NaN; absent pairs are exact zero with a separate activity mask. Whole/unary/
    boundary values are NaN whenever any positive component is outside source
    vocabulary, even if some known subpairs can be evaluated.
    """
    names = _names(components, "components")
    audit = audit_and_align(X, input_components=names, common_components=names,
                            source_components=names, input_basis="atomic_fraction",
                            source_basis="atomic_fraction")
    X = np.asarray(audit.common_fractions, dtype=float)
    coordinates = _names(coordinate_names, "coordinate_names")
    available = set(names if available_components is None else available_components)
    n, p = X.shape
    q = len(coordinates)
    component_ok = np.array([c in available for c in names], dtype=bool)
    row_ok = ~np.any((X > 0) & ~component_ok, axis=1)

    def call(A):
        out = np.asarray(evaluate(A), dtype=float)
        if out.shape != (len(A), q) or not np.isfinite(out).all():
            raise ValueError("evaluate must return finite (n, q) values on the declared scale")
        return out

    vertices = np.full((p, q), np.nan)
    if component_ok.any():
        vertices[component_ok] = call(np.eye(p)[component_ok])
    whole = np.full((n, q), np.nan)
    unary = np.full((n, q), np.nan)
    if row_ok.any():
        whole[row_ok] = call(X[row_ok])
        unary[row_ok] = X[row_ok][:, component_ok] @ vertices[component_ok]
    pairs = np.array(list(combinations(range(p), 2)), dtype=int).reshape(-1, 2)
    raw = np.zeros((n, len(pairs), q))
    excess = np.zeros_like(raw)
    active = np.zeros((n, len(pairs)), dtype=bool)
    pair_ok = np.zeros(len(pairs), dtype=bool)
    empirical = np.full(len(pairs), -1, dtype=np.int8)
    observed = None
    if observed_pairs is not None:
        observed = set()
        for pair in observed_pairs:
            pair = tuple(pair)
            if len(pair) != 2 or pair[0] == pair[1] or any(c not in available for c in pair):
                raise ValueError("Observed pairs must contain two distinct source components")
            observed.add(frozenset(pair))
    for k, (i, j) in enumerate(pairs):
        active[:, k] = (X[:, i] > 0) & (X[:, j] > 0)
        pair_ok[k] = component_ok[i] and component_ok[j]
        if observed is not None:
            empirical[k] = int(frozenset((names[i], names[j])) in observed)
        rows = np.flatnonzero(active[:, k])
        if not len(rows):
            continue  # s=0 (or a one-component edge) has exactly zero excess.
        if not pair_ok[k]:
            raw[rows, k] = np.nan
            excess[rows, k] = np.nan
            continue
        s = X[rows, i] + X[rows, j]
        # Compute both fractions directly. Subtracting one from the other can
        # erase a strictly positive trace constituent when the larger fraction
        # rounds to 1.0 (for example, [1.0, 1e-20]).
        ti = X[rows, i] / s
        tj = X[rows, j] / s
        u = np.zeros((len(rows), p))
        u[:, i] = ti
        u[:, j] = tj
        g = call(u)
        raw[rows, k] = g
        excess[rows, k] = s[:, None] ** 2 * (
            g - ti[:, None] * vertices[i] - tj[:, None] * vertices[j])
    boundary = unary + excess.sum(axis=1)
    boundary[~row_ok] = np.nan
    return DescriptorResult(names, coordinates, X, whole, vertices, unary, raw,
                            excess, boundary, pairs, active, pair_ok, empirical, row_ok)
