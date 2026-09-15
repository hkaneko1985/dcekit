from __future__ import annotations

import unittest

import numpy as np

from material_similarity import (
    DEFAULT_VIEWS,
    DocumentedValue as V,
    MaterialInstance,
    MaterialSimilarityEngine,
    ProcessStep,
    ReportingStatus,
    agglomerative_cluster,
    cluster_consensus,
    consensus_pairs,
    mutual_neighbor_consensus,
    nearest_neighbors,
)


def sample(record_id, element, method, step):
    return MaterialInstance(
        record_id,
        composition=V.reported({element: 1}, kind="composition"),
        material_identity={"form": V.reported("film")},
        process_method=V.reported([method], kind="set"),
        process_steps=(ProcessStep(step),),
        process_sequence_status=ReportingStatus.REPORTED,
    )


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            sample("a1", "Al", "solution", "mix"),
            sample("a2", "Al", "solution", "mix"),
            sample("b1", "Si", "melt", "extrude"),
            sample("b2", "Si", "melt", "extrude"),
        ]
        self.engine = MaterialSimilarityEngine(self.records)

    def test_clustering_and_neighbors_are_deterministic(self):
        pairwise = self.engine.pairwise(DEFAULT_VIEWS["balanced_instance"])
        first = agglomerative_cluster(pairwise, n_clusters=2)
        second = agglomerative_cluster(pairwise, n_clusters=2)
        self.assertTrue(np.array_equal(first.labels, second.labels))
        self.assertEqual(first.labels[0], first.labels[1])
        self.assertEqual(first.labels[2], first.labels[3])
        neighbors = nearest_neighbors(pairwise, k=1)
        self.assertEqual(neighbors["a1"][0][0], "a2")

    def test_cluster_consensus(self):
        material = agglomerative_cluster(
            self.engine.pairwise(DEFAULT_VIEWS["material_identity"]), n_clusters=2
        )
        process = agglomerative_cluster(
            self.engine.pairwise(DEFAULT_VIEWS["synthesis_pathway"]), n_clusters=2
        )
        matrix = cluster_consensus([material, process])
        self.assertEqual(matrix[0, 1], 1.0)
        pairs = consensus_pairs(matrix, material.ids, minimum_support=1.0)
        self.assertEqual(len(pairs), 2)

    def test_mutual_neighbor_consensus(self):
        views = [
            self.engine.pairwise(DEFAULT_VIEWS["material_identity"]),
            self.engine.pairwise(DEFAULT_VIEWS["synthesis_pathway"]),
        ]
        matrix = mutual_neighbor_consensus(views, k=1)
        self.assertEqual(matrix[0, 1], 1.0)
        self.assertEqual(matrix[2, 3], 1.0)


if __name__ == "__main__":
    unittest.main()
