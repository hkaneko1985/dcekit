"""Clustering, neighborhood retrieval, and cross-view consensus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

from .distance import PairwiseDistanceResult


@dataclass(frozen=True)
class ClusteringResult:
    ids: tuple[str, ...]
    labels: np.ndarray
    n_clusters: int
    view_name: str
    bound: str


def agglomerative_cluster(
    result: PairwiseDistanceResult,
    *,
    n_clusters: int,
    bound: str = "reported",
    undefined: str = "raise",
) -> ClusteringResult:
    if n_clusters < 2 or n_clusters > len(result.ids):
        raise ValueError("n_clusters must be between 2 and the number of records")
    matrix = result.matrix(bound, undefined=undefined)
    if not np.allclose(matrix, matrix.T, atol=1e-7):
        raise ValueError("Distance matrix must be symmetric")
    tree = linkage(squareform(matrix, checks=False), method="average")
    labels = cut_tree(tree, n_clusters=[n_clusters])[:, 0].astype(np.int32)
    return ClusteringResult(result.ids, labels, n_clusters, result.view.name, bound)


def nearest_neighbors(
    result: PairwiseDistanceResult,
    *,
    k: int = 10,
    bound: str = "reported",
    undefined: str = "raise",
) -> dict[str, list[tuple[str, float]]]:
    if k < 1 or k >= len(result.ids):
        raise ValueError("k must be in [1, n_records - 1]")
    matrix = result.matrix(bound, undefined=undefined)
    output = {}
    for row, record_id in enumerate(result.ids):
        order = np.argsort(matrix[row], kind="stable")
        order = [index for index in order if index != row][:k]
        output[record_id] = [(result.ids[index], float(matrix[row, index])) for index in order]
    return output


def cluster_consensus(clusterings: Sequence[ClusteringResult]) -> np.ndarray:
    if not clusterings:
        raise ValueError("At least one clustering is required")
    reference = clusterings[0].ids
    if any(item.ids != reference for item in clusterings):
        raise ValueError("All clusterings must use the same record order")
    n = len(reference)
    consensus = np.zeros((n, n), dtype=np.float32)
    for item in clusterings:
        consensus += (item.labels[:, None] == item.labels[None, :]).astype(np.float32)
    consensus /= len(clusterings)
    np.fill_diagonal(consensus, 1.0)
    return consensus


def mutual_neighbor_consensus(
    results: Sequence[PairwiseDistanceResult],
    *,
    k: int = 10,
    bound: str = "reported",
    undefined: str = "raise",
) -> np.ndarray:
    if not results:
        raise ValueError("At least one pairwise result is required")
    reference = results[0].ids
    if any(item.ids != reference for item in results):
        raise ValueError("All views must use the same record order")
    n = len(reference)
    if k < 1 or k >= n:
        raise ValueError("k must be in [1, n_records - 1]")
    consensus = np.zeros((n, n), dtype=np.float32)
    for result in results:
        matrix = result.matrix(bound, undefined=undefined)
        neighbors = np.zeros((n, n), dtype=bool)
        for row in range(n):
            order = [index for index in np.argsort(matrix[row], kind="stable") if index != row][:k]
            neighbors[row, order] = True
        consensus += (neighbors & neighbors.T).astype(np.float32)
    consensus /= len(results)
    np.fill_diagonal(consensus, 1.0)
    return consensus


def consensus_pairs(
    consensus: np.ndarray,
    ids: Sequence[str],
    *,
    minimum_support: float,
) -> list[dict[str, float | str]]:
    if consensus.shape != (len(ids), len(ids)):
        raise ValueError("Consensus matrix shape does not match ids")
    output = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            support = float(consensus[i, j])
            if support >= minimum_support:
                output.append({"left_id": ids[i], "right_id": ids[j], "support": support})
    return sorted(output, key=lambda row: (-float(row["support"]), str(row["left_id"]), str(row["right_id"])))
