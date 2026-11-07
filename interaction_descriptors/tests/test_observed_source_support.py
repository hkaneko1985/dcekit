"""Boundary checks specific to falsely declared support; no model fitting."""
import unittest

from interaction_descriptors.composition import CompositionInputError
from interaction_descriptors.observed_support import inspect_source_observations, audit_with_source_observations


class ObservedSupportTests(unittest.TestCase):
    def test_vocabulary_is_not_observed_support_and_traces_are_preserved(self):
        source = inspect_source_observations([[1, 0, 0], [.9, .1, 0]],
            source_components=['Fe', 'C', 'N'], source_basis='atomic_fraction')
        result = audit_with_source_observations(
            [[.01, .99, 0, 0], [0, 1 - 1e-14, 1e-14, 0], [0, .99, 0, .01]],
            input_components=['C', 'Fe', 'N', 'Si'],
            common_components=['Fe', 'C', 'N', 'Si'], input_basis='atomic_fraction',
            source_observations=source)
        self.assertEqual(source.positive_row_counts, (2, 1, 0))
        self.assertEqual(source.unobserved_components, ('N',))
        self.assertEqual(result.vocabulary_alignment.supported_mask, (True, True, False))
        self.assertEqual(result.observed_supported_mask, (True, False, False))
        self.assertEqual(result.unobserved_present, ((), ('N',), ()))
        self.assertEqual(result.observed_source_inputs, ((.99, .01, 0.), None, None))
        self.assertEqual(result.vocabulary_alignment.common_fractions[1][2], 1e-14)
        self.assertEqual(result.vocabulary_alignment.unsupported_present[2], ('Si',))

    def test_invalid_source_rows_are_not_accepted_as_evidence(self):
        for rows in [[], [[.9, 0]], [[float('nan'), 1]], [[1.1, -.1]], [[1, 0, 0]]]:
            with self.subTest(rows=rows), self.assertRaises(CompositionInputError):
                inspect_source_observations(rows, source_components=['Fe', 'C'],
                                            source_basis='atomic_fraction')


if __name__ == '__main__':
    unittest.main()
