# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Semi-supervised Gaussian Mixture Regression with x-only GMM pretraining
# (ssGMR-xGMM)

import numpy as np
from sklearn.mixture import GaussianMixture

from .gmr import GMR


_VALID_COVARIANCE_TYPES = ('full', 'diag', 'tied', 'spherical')


def _as_2d_float_array(values, name, allow_one_dimensional=False):
    """Convert input data to a finite two-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1 and allow_one_dimensional:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError('{0} must be a two-dimensional array.'.format(name))
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError('{0} must not be empty.'.format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError('{0} contains NaN or infinite values.'.format(name))
    return array


def _validate_variable_numbers(numbers, number_of_variables, name):
    """Validate variable-number lists used by DCEKit GMR."""
    values = np.asarray(numbers, dtype=int).reshape(-1)
    if values.size == 0:
        raise ValueError('{0} must contain at least one variable number.'.format(name))
    if np.unique(values).size != values.size:
        raise ValueError('{0} contains duplicated variable numbers.'.format(name))
    if np.any(values < 0) or np.any(values >= number_of_variables):
        raise ValueError(
            '{0} contains a variable number outside [0, {1}].'.format(
                name, number_of_variables - 1
            )
        )
    return values


def _make_symmetric_positive_definite(matrix, minimum_eigenvalue):
    """Symmetrize a matrix and clip its eigenvalues to a positive floor."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, minimum_eigenvalue)
    matrix = (eigenvectors * eigenvalues).dot(eigenvectors.T)
    return 0.5 * (matrix + matrix.T)


def _stabilize_joint_covariance(
        covariance_x, covariance_xy, covariance_y, minimum_eigenvalue):
    """
    Construct a positive-definite block covariance while preserving the x and
    y marginal covariance matrices. The cross-covariance is shrunk only when
    required by the positive-definite constraint.
    """
    covariance_x = _make_symmetric_positive_definite(
        covariance_x, minimum_eigenvalue
    )
    covariance_y = _make_symmetric_positive_definite(
        covariance_y, minimum_eigenvalue
    )
    covariance_xy = np.asarray(covariance_xy, dtype=float)

    cholesky_x = np.linalg.cholesky(covariance_x)
    cholesky_y = np.linalg.cholesky(covariance_y)

    # The block matrix is positive definite if all singular values of
    # Lx^-1 Sxy Ly^-T are smaller than one.
    normalized_cross_covariance = np.linalg.solve(
        cholesky_x, covariance_xy
    )
    normalized_cross_covariance = np.linalg.solve(
        cholesky_y, normalized_cross_covariance.T
    ).T
    singular_values = np.linalg.svd(
        normalized_cross_covariance, compute_uv=False
    )
    largest_singular_value = (
        singular_values[0] if singular_values.size else 0.0
    )

    cross_covariance_scaling = 1.0
    maximum_allowed_singular_value = 1.0 - 1e-8
    if largest_singular_value >= maximum_allowed_singular_value:
        cross_covariance_scaling = (
            maximum_allowed_singular_value / largest_singular_value
        )
        covariance_xy = covariance_xy * cross_covariance_scaling

    number_of_x_variables = covariance_x.shape[0]
    number_of_y_variables = covariance_y.shape[0]
    covariance = np.zeros(
        (number_of_x_variables + number_of_y_variables,
         number_of_x_variables + number_of_y_variables),
        dtype=float,
    )
    covariance[:number_of_x_variables, :number_of_x_variables] = covariance_x
    covariance[:number_of_x_variables, number_of_x_variables:] = covariance_xy
    covariance[number_of_x_variables:, :number_of_x_variables] = covariance_xy.T
    covariance[number_of_x_variables:, number_of_x_variables:] = covariance_y
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, cross_covariance_scaling


def _covariances_to_precisions(covariances, covariance_type):
    """Convert covariance parameters to the precisions_init representation."""
    if covariance_type == 'full':
        number_of_components, number_of_variables, _ = covariances.shape
        identity = np.eye(number_of_variables)
        precisions = np.empty_like(covariances)
        for component_number in range(number_of_components):
            precision = np.linalg.solve(
                covariances[component_number], identity
            )
            precisions[component_number] = 0.5 * (
                precision + precision.T
            )
        return precisions
    if covariance_type == 'tied':
        identity = np.eye(covariances.shape[0])
        precision = np.linalg.solve(covariances, identity)
        return 0.5 * (precision + precision.T)
    if covariance_type in ('diag', 'spherical'):
        return 1.0 / covariances
    raise ValueError('Invalid covariance_type: {0}'.format(covariance_type))


def _estimate_covariances_given_means(
        dataset, responsibilities, effective_sample_sizes, means,
        reg_covar, covariance_type):
    """
    Estimate covariance parameters with means supplied externally.

    This helper is also used by ssGMR-xGMM-xMA after replacing the x-side
    means with their anchors.
    """
    dataset = np.asarray(dataset, dtype=float)
    responsibilities = np.asarray(responsibilities, dtype=float)
    means = np.asarray(means, dtype=float)
    number_of_samples, number_of_variables = dataset.shape
    number_of_components = responsibilities.shape[1]

    if covariance_type == 'full':
        covariances = np.empty(
            (number_of_components, number_of_variables, number_of_variables),
            dtype=float,
        )
        for component_number in range(number_of_components):
            centered_dataset = dataset - means[component_number]
            covariance = (
                centered_dataset.T * responsibilities[:, component_number]
            ).dot(centered_dataset) / effective_sample_sizes[component_number]
            covariance = 0.5 * (covariance + covariance.T)
            covariance.flat[::number_of_variables + 1] += reg_covar
            covariances[component_number] = covariance
        return covariances

    if covariance_type == 'tied':
        covariance = np.zeros(
            (number_of_variables, number_of_variables), dtype=float
        )
        for component_number in range(number_of_components):
            centered_dataset = dataset - means[component_number]
            covariance += (
                centered_dataset.T * responsibilities[:, component_number]
            ).dot(centered_dataset)
        covariance /= responsibilities.sum()
        covariance = 0.5 * (covariance + covariance.T)
        covariance.flat[::number_of_variables + 1] += reg_covar
        return covariance

    if covariance_type == 'diag':
        covariances = np.empty(
            (number_of_components, number_of_variables), dtype=float
        )
        for component_number in range(number_of_components):
            centered_dataset = dataset - means[component_number]
            covariances[component_number] = (
                responsibilities[:, component_number, np.newaxis]
                * centered_dataset ** 2
            ).sum(axis=0) / effective_sample_sizes[component_number]
        covariances += reg_covar
        return covariances

    if covariance_type == 'spherical':
        covariances = np.empty(number_of_components, dtype=float)
        for component_number in range(number_of_components):
            centered_dataset = dataset - means[component_number]
            covariances[component_number] = (
                responsibilities[:, component_number]
                * np.sum(centered_dataset ** 2, axis=1)
            ).sum() / (
                effective_sample_sizes[component_number] * number_of_variables
            )
        covariances += reg_covar
        return covariances

    raise ValueError('Invalid covariance_type: {0}'.format(covariance_type))


def _compute_precision_cholesky(covariances, covariance_type):
    """
    Compute precision Cholesky factors using scikit-learn's convention.

    For full and tied covariances, the returned matrix is inv(L).T where
    covariance = L L.T.
    """
    if covariance_type == 'full':
        number_of_components, number_of_variables, _ = covariances.shape
        identity = np.eye(number_of_variables)
        precision_cholesky = np.empty_like(covariances)
        for component_number in range(number_of_components):
            cholesky = np.linalg.cholesky(covariances[component_number])
            precision_cholesky[component_number] = np.linalg.solve(
                cholesky, identity
            ).T
        return precision_cholesky

    if covariance_type == 'tied':
        number_of_variables = covariances.shape[0]
        cholesky = np.linalg.cholesky(covariances)
        return np.linalg.solve(cholesky, np.eye(number_of_variables)).T

    if covariance_type in ('diag', 'spherical'):
        if np.any(covariances <= 0):
            raise ValueError('Covariances must be positive.')
        return 1.0 / np.sqrt(covariances)

    raise ValueError('Invalid covariance_type: {0}'.format(covariance_type))


class SSGMRXGMM(GMR):
    """
    Semi-supervised GMR with x-only GMM pretraining (ssGMR-xGMM).

    A GMM is first fitted to x data that do not require corresponding y
    measurements. Its weights, x-side means, and x-side covariance structure
    are then used to initialize a joint GMM fitted to the labeled [x, y]
    samples. After fitting, all prediction methods inherited from ``GMR`` can
    be used for forward prediction and direct inverse analysis.

    Parameters
    ----------
    n_components : int, default=1
        Number of Gaussian components.
    covariance_type : {'full', 'diag', 'tied', 'spherical'}, default='full'
        Covariance representation used in both the x-only and joint GMMs.
    rep : {'mean', 'mode'}, default='mean'
        Representative value returned by ``predict_rep``. ``'mode'`` selects
        the conditional mean of the component with the largest responsibility;
        ``'mean'`` returns the responsibility-weighted conditional mean.
    tol : float, default=1e-3
        EM convergence tolerance.
    reg_covar : float, default=1e-6
        Non-negative regularization added to covariance diagonals.
    max_iter : int, default=100
        Maximum number of EM iterations for both GMM stages.
    x_gmm_n_init : int, default=1
        Number of initializations for the x-only GMM.
    init_params : {'kmeans', 'k-means++', 'random', 'random_from_data'},
            default='kmeans'
        Initialization method for the x-only GMM.
    random_state : int, RandomState or None, default=None
        Random seed.
    display_flag : bool, default=False
        DCEKit-compatible display flag used by inherited utility methods.
    responsibility_floor : float, default=1e-12
        Small floor applied to x-only responsibilities before calculating
        y-side initial values.
    verbose : int, default=0
        Verbosity passed to scikit-learn's GMM implementation.
    verbose_interval : int, default=10
        Interval between verbose messages.

    Notes
    -----
    ``labeled_dataset`` and ``x_for_pretraining`` must be expressed in exactly
    the same x-variable scale and column order. The class intentionally does
    not autoscale data, matching DCEKit's ``GMR`` behavior.
    """

    def __init__(
            self, n_components=1, covariance_type='full', rep='mean',
            tol=1e-3, reg_covar=1e-6, max_iter=100, x_gmm_n_init=1,
            init_params='kmeans', random_state=None, display_flag=False,
            responsibility_floor=1e-12, verbose=0, verbose_interval=10):
        # GMR.__init__ currently uses super(self.__class__, self), which is not
        # subclass-safe. Initialize GaussianMixture directly and inherit only
        # GMR's regression/prediction methods.
        GaussianMixture.__init__(
            self,
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=1,
            init_params=init_params,
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=random_state,
            warm_start=False,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self.rep = rep
        self.x_gmm_n_init = x_gmm_n_init
        self.display_flag = display_flag
        self.responsibility_floor = responsibility_floor

    def _check_method_parameters(self):
        if self.covariance_type not in _VALID_COVARIANCE_TYPES:
            raise ValueError(
                'covariance_type must be one of {0}.'.format(
                    _VALID_COVARIANCE_TYPES
                )
            )
        if self.rep not in ('mean', 'mode'):
            raise ValueError("rep must be either 'mean' or 'mode'.")
        if not isinstance(self.x_gmm_n_init, (int, np.integer)) \
                or self.x_gmm_n_init < 1:
            raise ValueError('x_gmm_n_init must be a positive integer.')
        if self.responsibility_floor < 0:
            raise ValueError('responsibility_floor must be non-negative.')
        if self.reg_covar < 0:
            raise ValueError('reg_covar must be non-negative.')

    def _prepare_joint_fit(self):
        """Hook used by ssGMR-xGMM-xMA before joint EM starts."""
        return None

    def _build_initial_parameters(
            self, labeled_dataset, x_labeled, y_labeled,
            numbers_of_x, numbers_of_y):
        number_of_components = self.n_components
        number_of_variables = labeled_dataset.shape[1]
        number_of_x_variables = len(numbers_of_x)
        number_of_y_variables = len(numbers_of_y)
        responsibilities = self.x_gmm_.predict_proba(x_labeled)
        if self.responsibility_floor > 0:
            responsibilities = np.maximum(
                responsibilities, self.responsibility_floor
            )
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        effective_sample_sizes = responsibilities.sum(axis=0)
        minimum_effective_sample_size = np.finfo(float).eps * 100
        effective_sample_sizes = np.maximum(
            effective_sample_sizes, minimum_effective_sample_size
        )

        initial_weights = np.asarray(self.x_gmm_.weights_, dtype=float).copy()
        initial_weights /= initial_weights.sum()

        initial_means = np.zeros(
            (number_of_components, number_of_variables), dtype=float
        )
        initial_means[:, numbers_of_x] = self.x_gmm_.means_
        initial_y_means = responsibilities.T.dot(y_labeled) \
            / effective_sample_sizes[:, np.newaxis]
        initial_means[:, numbers_of_y] = initial_y_means

        minimum_eigenvalue = max(
            float(self.reg_covar), np.finfo(float).eps * 100
        )
        cross_covariance_scaling = np.ones(
            number_of_components, dtype=float
        )

        if self.covariance_type == 'full':
            initial_covariances = np.zeros(
                (number_of_components, number_of_variables,
                 number_of_variables),
                dtype=float,
            )
            for component_number in range(number_of_components):
                sample_weights = responsibilities[:, component_number]
                centered_x = (
                    x_labeled - self.x_gmm_.means_[component_number]
                )
                centered_y = (
                    y_labeled - initial_y_means[component_number]
                )
                covariance_y = (
                    centered_y.T * sample_weights
                ).dot(centered_y) / effective_sample_sizes[component_number]
                covariance_y = _make_symmetric_positive_definite(
                    covariance_y, minimum_eigenvalue
                )
                covariance_xy = (
                    centered_x.T * sample_weights
                ).dot(centered_y) / effective_sample_sizes[component_number]
                covariance, scaling = _stabilize_joint_covariance(
                    self.x_gmm_.covariances_[component_number],
                    covariance_xy,
                    covariance_y,
                    minimum_eigenvalue,
                )
                cross_covariance_scaling[component_number] = scaling
                initial_covariances[component_number][
                    np.ix_(numbers_of_x, numbers_of_x)
                ] = covariance[
                    :number_of_x_variables, :number_of_x_variables
                ]
                initial_covariances[component_number][
                    np.ix_(numbers_of_x, numbers_of_y)
                ] = covariance[
                    :number_of_x_variables, number_of_x_variables:
                ]
                initial_covariances[component_number][
                    np.ix_(numbers_of_y, numbers_of_x)
                ] = covariance[
                    number_of_x_variables:, :number_of_x_variables
                ]
                initial_covariances[component_number][
                    np.ix_(numbers_of_y, numbers_of_y)
                ] = covariance[
                    number_of_x_variables:, number_of_x_variables:
                ]

        elif self.covariance_type == 'diag':
            initial_covariances = np.zeros(
                (number_of_components, number_of_variables), dtype=float
            )
            initial_covariances[:, numbers_of_x] = self.x_gmm_.covariances_
            for component_number in range(number_of_components):
                centered_y = (
                    y_labeled - initial_y_means[component_number]
                )
                y_variances = (
                    responsibilities[:, component_number, np.newaxis]
                    * centered_y ** 2
                ).sum(axis=0) / effective_sample_sizes[component_number]
                initial_covariances[
                    component_number, numbers_of_y
                ] = np.maximum(y_variances, minimum_eigenvalue)
            initial_covariances = np.maximum(
                initial_covariances, minimum_eigenvalue
            )

        elif self.covariance_type == 'tied':
            covariance_xy_all = np.zeros(
                (number_of_components, number_of_x_variables,
                 number_of_y_variables),
                dtype=float,
            )
            covariance_y_all = np.zeros(
                (number_of_components, number_of_y_variables,
                 number_of_y_variables),
                dtype=float,
            )
            for component_number in range(number_of_components):
                sample_weights = responsibilities[:, component_number]
                centered_x = (
                    x_labeled - self.x_gmm_.means_[component_number]
                )
                centered_y = (
                    y_labeled - initial_y_means[component_number]
                )
                covariance_xy_all[component_number] = (
                    centered_x.T * sample_weights
                ).dot(centered_y) / effective_sample_sizes[component_number]
                covariance_y_all[component_number] = (
                    centered_y.T * sample_weights
                ).dot(centered_y) / effective_sample_sizes[component_number]

            component_weights = effective_sample_sizes \
                / effective_sample_sizes.sum()
            covariance_xy = np.tensordot(
                component_weights, covariance_xy_all, axes=(0, 0)
            )
            covariance_y = np.tensordot(
                component_weights, covariance_y_all, axes=(0, 0)
            )
            covariance, scaling = _stabilize_joint_covariance(
                self.x_gmm_.covariances_, covariance_xy, covariance_y,
                minimum_eigenvalue,
            )
            cross_covariance_scaling[:] = scaling
            initial_covariances = np.zeros(
                (number_of_variables, number_of_variables), dtype=float
            )
            initial_covariances[np.ix_(numbers_of_x, numbers_of_x)] = \
                covariance[:number_of_x_variables, :number_of_x_variables]
            initial_covariances[np.ix_(numbers_of_x, numbers_of_y)] = \
                covariance[:number_of_x_variables, number_of_x_variables:]
            initial_covariances[np.ix_(numbers_of_y, numbers_of_x)] = \
                covariance[number_of_x_variables:, :number_of_x_variables]
            initial_covariances[np.ix_(numbers_of_y, numbers_of_y)] = \
                covariance[number_of_x_variables:, number_of_x_variables:]

        else:  # spherical
            initial_covariances = np.zeros(number_of_components, dtype=float)
            for component_number in range(number_of_components):
                centered_y = (
                    y_labeled - initial_y_means[component_number]
                )
                y_variances = (
                    responsibilities[:, component_number, np.newaxis]
                    * centered_y ** 2
                ).sum(axis=0) / effective_sample_sizes[component_number]
                initial_covariances[component_number] = (
                    number_of_x_variables
                    * self.x_gmm_.covariances_[component_number]
                    + np.sum(y_variances)
                ) / number_of_variables
            initial_covariances = np.maximum(
                initial_covariances, minimum_eigenvalue
            )

        initial_precisions = _covariances_to_precisions(
            initial_covariances, self.covariance_type
        )
        return (
            initial_weights,
            initial_means,
            initial_covariances,
            initial_precisions,
            responsibilities,
            effective_sample_sizes,
            cross_covariance_scaling,
        )

    def fit(
            self, labeled_dataset, x_for_pretraining=None,
            numbers_of_x=None, numbers_of_y=None):
        """
        Fit ssGMR-xGMM.

        Parameters
        ----------
        labeled_dataset : array-like of shape (n_labeled, n_x + n_y)
            Paired, consistently scaled [x, y] samples used for the joint GMM.
        x_for_pretraining : array-like of shape (n_pretraining, n_x), optional
            x samples used by the x-only GMM. They may include the labeled x
            samples, additional unlabeled x samples, and, in a transductive
            protocol, target-domain test x samples. If omitted, the labeled x
            samples are used.
        numbers_of_x : array-like of int
            Column numbers of x in ``labeled_dataset``.
        numbers_of_y : array-like of int
            Column numbers of y in ``labeled_dataset``.

        Returns
        -------
        self : SSGMRXGMM
            Fitted model.
        """
        self._check_method_parameters()
        labeled_dataset = _as_2d_float_array(
            labeled_dataset, 'labeled_dataset'
        )
        number_of_variables = labeled_dataset.shape[1]
        if numbers_of_x is None or numbers_of_y is None:
            raise ValueError(
                'numbers_of_x and numbers_of_y must both be specified.'
            )
        numbers_of_x = _validate_variable_numbers(
            numbers_of_x, number_of_variables, 'numbers_of_x'
        )
        numbers_of_y = _validate_variable_numbers(
            numbers_of_y, number_of_variables, 'numbers_of_y'
        )
        if np.intersect1d(numbers_of_x, numbers_of_y).size:
            raise ValueError('numbers_of_x and numbers_of_y must not overlap.')
        if not np.array_equal(
                np.sort(np.r_[numbers_of_x, numbers_of_y]),
                np.arange(number_of_variables)):
            raise ValueError(
                'numbers_of_x and numbers_of_y must jointly cover every '
                'column of labeled_dataset.'
            )

        x_labeled = labeled_dataset[:, numbers_of_x]
        y_labeled = labeled_dataset[:, numbers_of_y]
        if x_for_pretraining is None:
            x_for_pretraining = x_labeled
        else:
            x_for_pretraining = np.asarray(
                x_for_pretraining, dtype=float
            )
            if x_for_pretraining.ndim == 1 and len(numbers_of_x) == 1:
                x_for_pretraining = x_for_pretraining.reshape(-1, 1)
            if x_for_pretraining.ndim != 2:
                raise ValueError(
                    'x_for_pretraining must be a two-dimensional array.'
                )
            if x_for_pretraining.shape[1] == number_of_variables:
                # Select x before the finite-value check so that a joint-style
                # array may contain missing y values.
                x_for_pretraining = x_for_pretraining[:, numbers_of_x]
            elif x_for_pretraining.shape[1] != len(numbers_of_x):
                raise ValueError(
                    'x_for_pretraining must have either {0} x columns or {1} '
                    'joint-data columns.'.format(
                        len(numbers_of_x), number_of_variables
                    )
                )
            x_for_pretraining = _as_2d_float_array(
                x_for_pretraining, 'x_for_pretraining'
            )

        if labeled_dataset.shape[0] < self.n_components:
            raise ValueError(
                'The number of labeled samples must be at least '
                'n_components.'
            )
        if x_for_pretraining.shape[0] < self.n_components:
            raise ValueError(
                'The number of x pretraining samples must be at least '
                'n_components.'
            )

        self.numbers_of_x_ = numbers_of_x.copy()
        self.numbers_of_y_ = numbers_of_y.copy()
        self.number_of_x_variables_ = len(numbers_of_x)
        self.number_of_y_variables_ = len(numbers_of_y)

        self.x_gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.x_gmm_n_init,
            init_params=self.init_params,
            random_state=self.random_state,
            warm_start=False,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )
        self.x_gmm_.fit(x_for_pretraining)

        (
            self.initial_weights_,
            self.initial_means_,
            self.initial_covariances_,
            self.initial_precisions_,
            self.x_responsibilities_for_labeled_samples_,
            self.x_effective_sample_sizes_,
            self.cross_covariance_scaling_,
        ) = self._build_initial_parameters(
            labeled_dataset,
            x_labeled,
            y_labeled,
            numbers_of_x,
            numbers_of_y,
        )

        # GaussianMixture consumes these public initialization attributes.
        self.weights_init = self.initial_weights_.copy()
        self.means_init = self.initial_means_.copy()
        self.precisions_init = np.array(
            self.initial_precisions_, copy=True
        )
        self._prepare_joint_fit()

        # Call GaussianMixture.fit directly to retain GMR's prediction methods
        # while avoiding GMR.__init__'s subclassing issue.
        GaussianMixture.fit(self, labeled_dataset)
        return self

    def fit_from_xy(self, x_labeled, y_labeled, x_for_pretraining=None):
        """
        Convenience interface when x and y are supplied as separate arrays.
        """
        x_labeled = _as_2d_float_array(
            x_labeled, 'x_labeled', allow_one_dimensional=True
        )
        y_labeled = _as_2d_float_array(
            y_labeled, 'y_labeled', allow_one_dimensional=True
        )
        if x_labeled.shape[0] != y_labeled.shape[0]:
            raise ValueError(
                'x_labeled and y_labeled must have the same number of samples.'
            )
        labeled_dataset = np.c_[x_labeled, y_labeled]
        numbers_of_x = np.arange(x_labeled.shape[1])
        numbers_of_y = np.arange(
            x_labeled.shape[1], labeled_dataset.shape[1]
        )
        return self.fit(
            labeled_dataset,
            x_for_pretraining=x_for_pretraining,
            numbers_of_x=numbers_of_x,
            numbers_of_y=numbers_of_y,
        )
