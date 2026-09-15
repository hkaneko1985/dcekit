from __future__ import annotations

import math
import unittest

import numpy as np

from material_similarity import (
    DEFAULT_VIEWS,
    DocumentedValue as V,
    MaterialInstance,
    MaterialSimilarityEngine,
    NumericScale,
    ProcessStep,
    ReportingStatus,
    ViewSpec,
    normalized_sequence_distance,
)


def record(
    record_id: str,
    *,
    temp=100.0,
    time=10.0,
    steps=("heat", "cool"),
    sequence_status=ReportingStatus.REPORTED,
    context=None,
):
    process_steps = []
    for token in steps:
        settings = {}
        if token == "heat":
            settings = {
                "temperature": temp if isinstance(temp, V) else V.reported(temp, kind="numeric", unit_key="temperature_c"),
                "time": time if isinstance(time, V) else V.reported(time, kind="numeric", unit_key="time_min"),
            }
        process_steps.append(ProcessStep(token, settings))
    return MaterialInstance(
        record_id=record_id,
        composition=V.reported({"Al": 2, "O": 3}, kind="composition"),
        material_identity={"form": V.reported("powder")},
        process_method=V.reported(["solid state"], kind="set"),
        process_steps=tuple(process_steps),
        process_sequence_status=sequence_status,
        context=context or {},
    )


class SchemaAndDistanceTests(unittest.TestCase):
    def setUp(self):
        self.scales = {
            "temperature_c": NumericScale(100.0),
            "time_min": NumericScale(100.0),
        }

    def test_reported_value_requires_value(self):
        with self.assertRaises(ValueError):
            V(ReportingStatus.REPORTED, None, "numeric")

    def test_context_is_not_a_distance_feature(self):
        a = record("a", context={"paper": "P1", "property": 999})
        b = record("b", context={"paper": "P2", "property": -999})
        engine = MaterialSimilarityEngine([a, b], numeric_scales=self.scales)
        self.assertEqual(engine.compare("a", "b", DEFAULT_VIEWS["balanced_instance"]).composite.reported, 0.0)

    def test_missing_setting_does_not_change_common_reported_distance(self):
        a = record("a")
        b = record("b", time=V.unknown(kind="numeric", unit_key="time_min"))
        interval = MaterialSimilarityEngine([a, b], numeric_scales=self.scales).facet_distances("a", "b")["settings"]
        self.assertEqual(interval.reported, 0.0)
        self.assertAlmostEqual(interval.lower, 0.0)
        self.assertAlmostEqual(interval.upper, 0.5)
        self.assertAlmostEqual(interval.one_sided_fraction, 0.5)

    def test_known_absent_step_and_unreported_sequence_are_different(self):
        a = record("a", steps=("heat",))
        b = record("b", steps=())
        c = MaterialInstance(
            record_id="c",
            composition=V.reported({"Al": 2, "O": 3}, kind="composition"),
            process_method=V.reported(["solid state"], kind="set"),
            process_sequence_status=ReportingStatus.UNREPORTED_OR_UNSET,
        )
        engine = MaterialSimilarityEngine([a, b, c], numeric_scales=self.scales)
        absent = engine.facet_distances("a", "b")["settings"]
        unknown = engine.facet_distances("a", "c")["settings"]
        self.assertEqual(absent.reported, 1.0)
        self.assertEqual(absent.width, 0.0)
        self.assertIsNone(unknown.reported)
        self.assertEqual(unknown.lower, 0.0)
        self.assertEqual(unknown.upper, 1.0)

    def test_joint_nonapplicability_does_not_create_similarity(self):
        a = MaterialInstance(
            "a",
            composition=V.reported({"Si": 1}, kind="composition"),
            material_identity={"coating": V.not_applicable()},
        )
        b = MaterialInstance(
            "b",
            composition=V.reported({"Si": 1}, kind="composition"),
            material_identity={"coating": V.not_applicable()},
        )
        interval = MaterialSimilarityEngine([a, b]).facet_distances("a", "b")["material_identity"]
        self.assertIsNone(interval.reported)
        self.assertEqual((interval.lower, interval.upper), (0.0, 1.0))

    def test_sequence_preserves_order_and_repetition(self):
        reference = ["heat", "cool", "heat"]
        self.assertEqual(normalized_sequence_distance(reference, reference), 0.0)
        self.assertGreater(normalized_sequence_distance(reference, ["heat", "heat", "cool"]), 0.0)
        self.assertGreater(normalized_sequence_distance(reference, ["heat", "cool"]), 0.0)

    def test_weight_views_are_sensitivity_views_not_optimized(self):
        a = record("a")
        b = record("b", steps=("mix", "cool"))
        engine = MaterialSimilarityEngine([a, b], numeric_scales=self.scales)
        material = engine.compare("a", "b", DEFAULT_VIEWS["material_identity"]).composite.reported
        pathway = engine.compare("a", "b", DEFAULT_VIEWS["synthesis_pathway"]).composite.reported
        self.assertEqual(material, 0.0)
        self.assertGreater(pathway, material)

    def test_pairwise_matrices_are_symmetric_and_bounded(self):
        engine = MaterialSimilarityEngine([record("a"), record("b", temp=150), record("c", steps=("mix",))], numeric_scales=self.scales)
        result = engine.pairwise(DEFAULT_VIEWS["balanced_instance"])
        for matrix in (result.optimistic, result.pessimistic):
            self.assertTrue(np.allclose(matrix, matrix.T))
            self.assertGreaterEqual(float(matrix.min()), 0.0)
            self.assertLessEqual(float(matrix.max()), 1.0)
            self.assertTrue(np.allclose(np.diag(matrix), 0.0))

    def test_undefined_reported_requires_explicit_policy(self):
        a = MaterialInstance("a")
        b = MaterialInstance("b")
        result = MaterialSimilarityEngine([a, b]).pairwise(DEFAULT_VIEWS["composition"])
        with self.assertRaises(ValueError):
            result.matrix("reported")
        self.assertEqual(result.matrix("reported", undefined="pessimistic")[0, 1], 1.0)

    def test_composition_is_scale_invariant(self):
        a = MaterialInstance("a", composition=V.reported({"Al": 2, "O": 3}, kind="composition"))
        b = MaterialInstance("b", composition=V.reported({"Al": 4, "O": 6}, kind="composition"))
        interval = MaterialSimilarityEngine([a, b]).facet_distances("a", "b")["composition"]
        self.assertAlmostEqual(interval.reported, 0.0)

    def test_nonsequential_process_settings_are_supported(self):
        a = MaterialInstance(
            "a",
            process_settings={
                "pressure": V.reported(1.0, kind="numeric", unit_key="pressure")
            },
        )
        b = MaterialInstance(
            "b",
            process_settings={
                "pressure": V.reported(2.0, kind="numeric", unit_key="pressure")
            },
        )
        engine = MaterialSimilarityEngine([a, b], numeric_scales={"pressure": NumericScale(2.0)})
        interval = engine.facet_distances("a", "b")["settings"]
        self.assertAlmostEqual(interval.reported, 0.5)


if __name__ == "__main__":
    unittest.main()
