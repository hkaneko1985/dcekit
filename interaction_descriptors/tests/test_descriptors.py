"""Focused mathematical and missing-component contract checks."""
import unittest

import numpy as np

from interaction_descriptors import CompositionInputError, boundary_extension
from interaction_descriptors.cli import synthetic_model


class DescriptorTests(unittest.TestCase):
    def test_pure_and_binary_agreement_for_nonlinear_network(self):
        model = synthetic_model()
        X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.3, .7, 0], [0, .8, .2]])
        r = model.describe(X, ["A", "B", "C"])
        np.testing.assert_allclose(r.whole, r.boundary, rtol=0, atol=1e-14)
        self.assertTrue(np.isfinite(r.boundary).all())
        self.assertEqual(r.pair_raw.shape, (5, 3, 3))
        self.assertTrue((r.pair_raw[~r.pair_active] == 0).all())

    def test_quadratic_exactness(self):
        rng = np.random.default_rng(4)
        X = rng.dirichlet(np.ones(4), 25)
        linear = rng.normal(size=(4, 3))
        coefficients = rng.normal(size=(4, 4, 3))
        def f(A):
            return A @ linear + np.einsum("ni,nj,ijq->nq", A, A, coefficients)
        r = boundary_extension(X, f, components=["A", "B", "C", "D"],
                               coordinate_names=["z1", "z2", "y"])
        np.testing.assert_allclose(r.boundary, f(X), rtol=0, atol=2e-15)

    def test_trace_fraction_survives_pair_normalization(self):
        # A positive trace can disappear in 1 - (x_i / s), even when it was
        # preserved by the full-composition input audit. Check both column
        # orientations and use a rescaled continuous quadratic response so the
        # lost trace causes an observable failure of the binary identity.
        X = np.array([[1.0, 1e-20], [1e-20, 1.0]])
        evaluated = []

        def f(A):
            evaluated.append(A.copy())
            return (1e20 * A[:, 0] * A[:, 1])[:, None]

        r = boundary_extension(X, f, components=["A", "B"], coordinate_names=["q"])
        np.testing.assert_array_equal(evaluated[-1], X)
        np.testing.assert_array_equal(r.composition, X)
        np.testing.assert_array_equal(r.whole, np.ones((2, 1)))
        np.testing.assert_array_equal(r.pair_raw[:, 0, :], r.whole)
        np.testing.assert_array_equal(r.boundary, r.whole)
        self.assertTrue(r.pair_active.all())

    def test_component_and_coordinate_schemas_reject_bare_strings(self):
        X = np.array([[0.4, 0.6]])
        for components, coordinates in [("Fe", ["z"]), (["Fe", "Ni"], "z")]:
            with self.subTest(components=components, coordinate_names=coordinates):
                with self.assertRaises(CompositionInputError):
                    boundary_extension(X, lambda A: A[:, :1],
                                       components=components, coordinate_names=coordinates)

    def test_unsupported_is_preserved_and_distinct_from_absent(self):
        model = synthetic_model()
        X = np.array([[.4, .6, 0, 0], [.4, 0, .6, 0], [.4, 0, 0, .6], [1 - 1e-12, 0, 0, 1e-12]])
        r = model.describe(X, ["A", "B", "C", "D"], observed_pairs=[("A", "B")])
        np.testing.assert_array_equal(r.composition, X)
        np.testing.assert_array_equal(r.row_vocabulary_supported, [True, True, False, False])
        self.assertTrue(np.isnan(r.boundary[2:]).all())
        self.assertTrue(np.isnan(r.whole[2:]).all())
        self.assertEqual(len(r.pair_indices), 6)
        self.assertFalse(r.pair_active[0, 2])  # A-D absent here.
        self.assertTrue((r.pair_raw[0, 2] == 0).all())
        self.assertTrue(np.isnan(r.pair_raw[2, 2]).all())  # A-D active but unsupported.
        self.assertEqual(r.pair_empirical_support[0], 1)  # A-B observed.
        self.assertEqual(r.pair_empirical_support[1], 0)  # A-C source-valid, unobserved.
        self.assertTrue(np.isfinite(r.pair_raw[1, 1]).all())
        unaudited = model.describe(X[:1], ["A", "B", "C", "D"])
        self.assertTrue((unaudited.pair_empirical_support == -1).all())

    def test_frozen_source_order_and_fraction_contract(self):
        model = synthetic_model()
        a, _, _ = model.infer([[.3, .2, .5]], ["A", "B", "C"])
        b, _, audit = model.infer([[.5, .3, .2]], ["C", "A", "B"])
        np.testing.assert_array_equal(a, b)
        self.assertFalse(audit.automatically_normalized)
        self.assertFalse(model.weights["x_mean"].flags.writeable)
        with self.assertRaises(CompositionInputError):
            model.infer([[.3, .2, .5]], ["A", "B", "C"], fraction_basis="mass_fraction")
        with self.assertRaises(CompositionInputError):
            model.infer([[.3, .2, .6]], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
