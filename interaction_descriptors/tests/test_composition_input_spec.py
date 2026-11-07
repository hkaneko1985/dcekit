"""Meaningful synthetic boundary checks for the C11 input contract.

Run: python test_composition_input_spec.py
The report explicitly excludes a real-dataset or source-model audit.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from interaction_descriptors.composition import (
    CompositionInputError,
    audit_and_align,
    common_component_schema,
)


def align(rows, **kwargs):
    options = dict(
        input_components=("Fe", "Ni"),
        common_components=("Fe", "Ni", "C", "Cr"),
        source_components=("Ni", "Fe"),
        input_basis="atomic_fraction",
        source_basis="atomic_fraction",
    )
    options.update(kwargs)
    return audit_and_align(rows, **options)


class InputContractTests(unittest.TestCase):
    def test_column_permutation_and_absent_component_zero(self):
        rows = [[0.25, 0.75]]
        out = align(
            rows,
            input_components=("Ni", "Fe"),
            common_components=("C", "Fe", "Ni", "Cr"),
            source_components=("Cr", "Ni", "Fe"),
        )
        self.assertEqual(out.common_fractions, ((0.0, 0.75, 0.25, 0.0),))
        self.assertEqual(out.source_inputs, ((0.0, 0.25, 0.75),))
        self.assertEqual(out.supported_mask, (True,))
        self.assertEqual(out.absent_components, (("C", "Cr"),))
        self.assertEqual(rows, [[0.25, 0.75]])

    def test_unsupported_positive_is_blocked_including_trace_amount(self):
        rows = [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [1 - 1e-14, 1e-14, 0.0]]
        out = align(rows, input_components=("Fe", "C", "Ni"))
        self.assertEqual(out.supported_mask, (True, False, False))
        self.assertEqual(out.source_inputs, ((0.0, 1.0), None, None))
        self.assertEqual(out.unsupported_present, ((), ("C",), ("C",)))
        self.assertEqual(out.unsupported_fractions, ((), (0.01,), (1e-14,)))
        self.assertEqual(out.common_fractions[1], (0.99, 0.0, 0.01, 0.0))
        self.assertEqual(out.common_fractions[2][2], 1e-14)

    def test_fraction_basis_requires_explicit_conversion(self):
        out = align([[0.8, 0.2]], input_basis="mass_fraction", source_basis="mass_fraction")
        self.assertEqual(out.fraction_basis, "mass_fraction")
        with self.assertRaises(CompositionInputError):
            align([[0.8, 0.2]], input_basis="mass_fraction")
        with self.assertRaises(CompositionInputError):
            align([[80.0, 20.0]], input_basis="mass_percent", source_basis="mass_percent")

    def test_sum_tolerance_does_not_normalize_values(self):
        out = align([[0.6, 0.4000000001]])
        self.assertEqual(out.common_fractions[0][:2], (0.6, 0.4000000001))
        self.assertEqual(out.source_inputs, ((0.4000000001, 0.6),))
        self.assertNotEqual(out.input_sums[0], 1.0)
        self.assertFalse(out.automatically_normalized)

    def test_invalid_numeric_compositions_are_rejected(self):
        cases = {
            "nan": [float("nan"), 1.0],
            "infinity": [float("inf"), 0.0],
            "negative_trace": [-1e-14, 1 + 1e-14],
            "invalid_sum": [0.4, 0.4],
            "all_zero": [0.0, 0.0],
            "finite_overflow": [1e308, 1e308],
            "wrong_width": [1.0],
        }
        for name, row in cases.items():
            with self.subTest(case=name), self.assertRaises(CompositionInputError):
                align([row])

    def test_schema_must_preserve_named_columns(self):
        union = common_component_schema(("Fe", "Ni"), ("C", "Fe"), ("Cr", "Ni"))
        self.assertEqual(union, ("Fe", "Ni", "C", "Cr"))
        with self.assertRaises(CompositionInputError):
            align([[0.8, 0.2]], common_components=("Fe",))
        with self.assertRaises(CompositionInputError):
            align([[0.8, 0.2]], input_components=("Fe", "Fe"))
        with self.assertRaises(CompositionInputError):
            common_component_schema("Fe")

    def test_empty_batch_is_not_reported_as_audit_success(self):
        with self.assertRaises(CompositionInputError):
            align([])

    def test_json_records_input_compatibility_not_empirical_support(self):
        out = align([[0.99, 0.01]], input_components=("Fe", "C"))
        record = json.loads(json.dumps(out.to_dict(), allow_nan=False))
        self.assertEqual(record["source_inputs"], [None])
        self.assertFalse(record["empirical_support_audited"])
        self.assertEqual(record["support_definition"], "declared_input_vocabulary_compatibility_only")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InputContractTests)
    test_names = [test.id().split(".")[-1] for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    example = align(
        [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [1 - 1e-14, 1e-14, 0.0]],
        input_components=("Fe", "C", "Ni"),
    )
    report = {
        "scope": "synthetic_input_contract_boundary_checks",
        "tests_run": result.testsRun,
        "passed": result.wasSuccessful(),
        "test_cases": test_names,
        "failures": [{"test": str(t), "traceback": detail} for t, detail in result.failures],
        "errors": [{"test": str(t), "traceback": detail} for t, detail in result.errors],
        "real_dataset_audit": "not_run_real_inputs_not_supplied_to_this_module",
        "source_models_retrained": False,
        "source_weights_modified": False,
        "empirical_pair_or_ratio_support_audited": False,
        "example_alignment": example.to_dict(),
    }
    path = Path(__file__).with_name("input_spec_validation.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
