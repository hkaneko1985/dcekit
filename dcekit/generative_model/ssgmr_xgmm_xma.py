# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

# Semi-supervised Gaussian Mixture Regression with x-only GMM pretraining
# and x-mean anchoring (ssGMR-xGMM-xMA)

import numpy as np

from .ssgmr_xgmm import (
    SSGMRXGMM,
    _compute_precision_cholesky,
    _estimate_covariances_given_means,
)


class SSGMRXGMMXMA(SSGMRXGMM):
    """
    Semi-supervised GMR with x-only GMM pretraining and x-mean anchoring.

    Initialization is identical to :class:`SSGMRXGMM`. During every M-step of
    the joint-GMM EM algorithm, the x-side means are replaced by the x-only GMM
    means. Mixture weights, y-side means, and all covariance parameters remain
    trainable. The covariance update is calculated around the anchored means,
    as specified by ssGMR-xGMM-xMA.

    The constructor and ``fit`` interface are the same as ``SSGMRXGMM``.
    After fitting, all prediction methods inherited from DCEKit's ``GMR`` can
    be used for forward prediction and direct inverse analysis.
    """

    def _prepare_joint_fit(self):
        self.x_mean_anchors_ = self.initial_means_[
            :, self.numbers_of_x_
        ].copy()

    def _m_step(self, dataset, log_responsibilities, xp=None):
        """M-step with hard anchoring of the x-side component means."""
        dataset = np.asarray(dataset, dtype=float)
        log_responsibilities = np.asarray(
            log_responsibilities, dtype=float
        )
        responsibilities = np.exp(log_responsibilities)
        effective_sample_sizes = responsibilities.sum(axis=0)
        effective_sample_sizes += 10 * np.finfo(float).eps

        self.weights_ = effective_sample_sizes \
            / effective_sample_sizes.sum()
        means = responsibilities.T.dot(dataset) \
            / effective_sample_sizes[:, np.newaxis]
        means[:, self.numbers_of_x_] = self.x_mean_anchors_
        self.means_ = means

        self.covariances_ = _estimate_covariances_given_means(
            dataset,
            responsibilities,
            effective_sample_sizes,
            self.means_,
            self.reg_covar,
            self.covariance_type,
        )
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
