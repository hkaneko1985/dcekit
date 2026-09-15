"""Unified API for target-free, missingness-aware material similarity."""

from .analysis import (
    ClusteringResult,
    agglomerative_cluster,
    cluster_consensus,
    consensus_pairs,
    mutual_neighbor_consensus,
    nearest_neighbors,
)
from .distance import (
    FACET_NAMES,
    DistanceInterval,
    MaterialSimilarityEngine,
    NumericScale,
    PairDistance,
    PairwiseDistanceResult,
    ViewSpec,
    fit_numeric_scales,
    normalized_sequence_distance,
)
from .schema import DocumentedValue, MaterialInstance, ProcessStep, ReportingStatus
from .views import DEFAULT_VIEWS

__all__ = [
    "ClusteringResult",
    "DEFAULT_VIEWS",
    "DocumentedValue",
    "DistanceInterval",
    "FACET_NAMES",
    "MaterialInstance",
    "MaterialSimilarityEngine",
    "NumericScale",
    "PairDistance",
    "PairwiseDistanceResult",
    "ProcessStep",
    "ReportingStatus",
    "ViewSpec",
    "agglomerative_cluster",
    "cluster_consensus",
    "consensus_pairs",
    "fit_numeric_scales",
    "mutual_neighbor_consensus",
    "nearest_neighbors",
    "normalized_sequence_distance",
]

__version__ = "0.2.2"
