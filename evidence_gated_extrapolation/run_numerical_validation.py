#!/usr/bin/env python3
"""Numerical validation for standardization-only Anchor--Delta methods.

The script intentionally keeps the representation z componentwise standardized:
no PCA, PLS, or nonlinear variable transformation is used.  It evaluates

1. RBF-GPR / RBF-SVR versus kNN-AD-guided ADE variants for forward
   extrapolation, and
2. absolute-coordinate GMR, transition-GMR, and evidence-gated competitive
   direct inverse analysis (EGC-GMR).

All reported test responses are noise-free simulator values.  Training responses
contain a small amount of Gaussian noise.  Hyperparameters that affect the
distance law are selected only from an outer shell of the training data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-anchor-delta")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import expit, logsumexp
from scipy.stats import multivariate_normal, pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

_JP_FONT = Path(__file__).resolve().parent / "qa_fonts" / "NotoSansCJKjp-Regular.otf"
if _JP_FONT.exists():
    font_manager.fontManager.addfont(str(_JP_FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_JP_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


OUTER_BLUE = "#0B5FA5"
TEAL = "#0E7C86"
ORANGE = "#D97706"
RED = "#B42318"
GRAY = "#667085"
LIGHT = "#EEF4F8"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
        return 0.0
    return float(pearsonr(yt, yp).statistic)


def uniform_disk(rng: np.random.Generator, n: int, r_min: float, r_max: float) -> np.ndarray:
    theta = rng.uniform(-np.pi, np.pi, n)
    radius = np.sqrt(rng.uniform(r_min**2, r_max**2, n))
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def directional_features(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Odd-in-direction feature map for a directional derivative.

    Omitting direction-independent terms enforces h(z, -u) = -h(z, u) for
    the global component, which is the physically natural constraint for a
    differentiable local response surface.
    """
    z = np.atleast_2d(z)
    u = np.atleast_2d(u)
    bilinear = np.einsum("ni,nj->nij", z, u).reshape(len(z), -1)
    return np.column_stack([u, bilinear])


def transition_pairs(z: np.ndarray, ranks: tuple[int, ...] = (1, 2, 4, 8, 16)) -> tuple[np.ndarray, np.ndarray]:
    """Directed, multiscale neighbor pairs; each origin contributes equally."""
    n = len(z)
    k = min(max(ranks) + 1, n)
    nbr = NearestNeighbors(n_neighbors=k).fit(z)
    _, ind = nbr.kneighbors(z)
    origins: list[int] = []
    targets: list[int] = []
    for i in range(n):
        for rank in ranks:
            if rank < ind.shape[1]:
                j = int(ind[i, rank])
                if j != i:
                    origins.append(i)
                    targets.append(j)
    return np.asarray(origins, dtype=int), np.asarray(targets, dtype=int)


def knn_leave_one_out_scores(z: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean leave-one-out kNN distance in a standardized input space."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    if len(z) < 2:
        return np.zeros(len(z), dtype=float)
    k_eff = min(max(1, int(k)), len(z) - 1)
    neighbors = NearestNeighbors(n_neighbors=k_eff + 1).fit(z)
    distances, _ = neighbors.kneighbors(z)
    return np.mean(distances[:, 1:], axis=1)


class KNNApplicabilityDomain:
    """Training-calibrated kNN applicability domain.

    The boundary is the selected quantile of leave-one-out mean kNN
    distances.  Query scores are measured against the fitted training set.
    This deliberately remains independent of GMM/GMR likelihood and therefore
    avoids reusing the GMR-specific Index of Extrapolation (IoE).
    """

    def __init__(self, k: int = 5, boundary_quantile: float = 0.90) -> None:
        self.k = int(k)
        self.boundary_quantile = float(boundary_quantile)

    def fit(self, z: np.ndarray) -> "KNNApplicabilityDomain":
        z = np.atleast_2d(np.asarray(z, dtype=float))
        if len(z) < 2:
            raise ValueError("kNN applicability domain requires at least two samples")
        self.z_train_ = z
        self.k_ = min(max(1, self.k), len(z) - 1)
        self.training_scores_ = knn_leave_one_out_scores(z, self.k_)
        self.threshold_ = max(
            float(np.quantile(self.training_scores_, self.boundary_quantile)),
            1e-10,
        )
        core = np.flatnonzero(self.training_scores_ <= self.threshold_ + 1e-12)
        if not len(core):
            core = np.asarray([int(np.argmin(self.training_scores_))], dtype=int)
        self.core_indices_ = core
        self.query_neighbors_ = NearestNeighbors(n_neighbors=self.k_).fit(z)
        self.core_neighbors_ = NearestNeighbors(n_neighbors=1).fit(z[core])
        nearest_core = int(self.core_neighbors_.kneighbors(np.zeros((1, z.shape[1])), return_distance=False)[0, 0])
        self.center_ = z[core[nearest_core]].copy()
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        distances, _ = self.query_neighbors_.kneighbors(z)
        return np.mean(distances, axis=1)

    def ratio(self, z: np.ndarray) -> np.ndarray:
        return self.score(z) / max(self.threshold_, 1e-12)

    def anchors(self, z: np.ndarray) -> np.ndarray:
        """Project queries to the last supported point on a local line segment."""
        z = np.atleast_2d(np.asarray(z, dtype=float))
        ratios = self.ratio(z)
        anchors = z.copy()
        outer = ratios > 1.0 + 1e-12
        if not np.any(outer):
            return anchors

        points = z[outer]
        local_core_index = self.core_neighbors_.kneighbors(points, return_distance=False)[:, 0]
        starts = self.z_train_[self.core_indices_[local_core_index]]
        delta = points - starts

        # kNN level sets need not be globally spherical.  A short grid finds
        # the last supported interval on each local core-to-query segment;
        # vectorized bisection then locates its boundary accurately.
        grid = np.linspace(0.0, 1.0, 33)
        grid_points = starts[:, None, :] + grid[None, :, None] * delta[:, None, :]
        grid_ratio = self.ratio(grid_points.reshape(-1, z.shape[1])).reshape(len(points), -1)
        supported = grid_ratio <= 1.0
        grid_indices = np.arange(len(grid))[None, :]
        last_supported = np.max(np.where(supported, grid_indices, -1), axis=1)
        last_supported = np.clip(last_supported, 0, len(grid) - 2)
        low = grid[last_supported]
        high = grid[last_supported + 1]
        for _ in range(28):
            mid = 0.5 * (low + high)
            mid_points = starts + mid[:, None] * delta
            mid_supported = self.ratio(mid_points) <= 1.0
            low = np.where(mid_supported, mid, low)
            high = np.where(mid_supported, high, mid)
        anchors[outer] = starts + low[:, None] * delta
        return anchors

    def points_at_ratio(self, directions: np.ndarray, target_ratio: float) -> np.ndarray:
        """Construct points on a requested kNN-AD ratio in given directions."""
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("All shell directions must be nonzero")
        unit = directions / norms[:, None]
        target = float(target_ratio)
        if target <= 0:
            raise ValueError("target_ratio must be positive")
        center = np.repeat(self.center_[None, :], len(unit), axis=0)
        low = np.zeros(len(unit), dtype=float)
        high = np.full(len(unit), max(1.0, target * self.threshold_), dtype=float)
        for _ in range(40):
            high_points = center + high[:, None] * unit
            insufficient = self.ratio(high_points) < target
            if not np.any(insufficient):
                break
            high[insufficient] *= 2.0
        for _ in range(45):
            mid = 0.5 * (low + high)
            mid_points = center + mid[:, None] * unit
            below = self.ratio(mid_points) < target
            low = np.where(below, mid, low)
            high = np.where(below, high, mid)
        distance = 0.5 * (low + high)
        return center + distance[:, None] * unit


class AnchorDeltaRegressor:
    """kNN-AD-guided Anchor-Delta Extrapolation regression."""

    def __init__(
        self,
        kind: str,
        q_kind: str = "linear",
        local_weight: float = 0.5,
        boundary_quantile: float = 0.90,
        ad_k: int = 5,
        random_state: int = 0,
    ) -> None:
        self.kind = kind
        self.q_kind = q_kind
        self.local_weight = float(local_weight)
        self.boundary_quantile = float(boundary_quantile)
        self.ad_k = int(ad_k)
        self.random_state = int(random_state)

    def _make_base(self, p: int):
        if self.kind == "GPR":
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
                length_scale=np.full(p, 0.85), length_scale_bounds="fixed"
            ) + WhiteKernel(noise_level=0.015, noise_level_bounds="fixed")
            return GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                optimizer=None,
                random_state=self.random_state,
                alpha=1e-8,
            )
        if self.kind == "SVR":
            return SVR(kernel="rbf", C=12.0, gamma=0.55 / max(p, 1), epsilon=0.035)
        raise ValueError(self.kind)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AnchorDeltaRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.x_scaler_ = StandardScaler().fit(x)
        z = self.x_scaler_.transform(x)
        self.p_ = z.shape[1]
        self.base_ = self._make_base(self.p_)
        self.base_.fit(z, y)
        self.ad_ = KNNApplicabilityDomain(
            k=self.ad_k,
            boundary_quantile=self.boundary_quantile,
        ).fit(z)
        # ``a0_`` is retained as a compatibility alias for archived analysis
        # code; it now stores the kNN-distance threshold, not a radial radius.
        self.a0_ = self.ad_.threshold_
        self.delta_ratio_ = 0.12

        ranks = (1, 2, 4, 8, min(16, len(z) - 1))
        ranks = tuple(sorted(set(r for r in ranks if r > 0)))
        i_idx, j_idx = transition_pairs(z, ranks)
        dz = z[j_idx] - z[i_idx]
        dist = np.linalg.norm(dz, axis=1)
        # Differencing two noisy observations makes very short-pair slopes
        # unstable (Var[Delta y / r] is proportional to 1/r^2).  Remove only
        # the shortest fifth and retain multiscale pairs for local structure.
        min_dist = max(0.10, float(np.quantile(dist[dist > 1e-8], 0.20)))
        good = dist >= min_dist
        i_idx, j_idx, dz, dist = i_idx[good], j_idx[good], dz[good], dist[good]
        u = dz / dist[:, None]
        slope = (y[j_idx] - y[i_idx]) / dist
        phi = directional_features(z[i_idx], u)

        self.global_ = Ridge(alpha=0.08, fit_intercept=False)
        self.global_.fit(phi, slope, sample_weight=dist**2)
        residual = slope - self.global_.predict(phi)
        local_x = np.column_stack([z[i_idx], u])

        # Antisymmetric augmentation also constrains the local residual:
        # the response-rate sign reverses when only the direction is reversed.
        local_x = np.vstack([local_x, np.column_stack([z[i_idx], -u])])
        residual = np.concatenate([residual, -residual])

        rng = np.random.default_rng(self.random_state + 941)
        max_local = 260 if self.kind == "GPR" else 900
        if len(local_x) > max_local:
            keep = rng.choice(len(local_x), size=max_local, replace=False)
            local_x = local_x[keep]
            residual = residual[keep]

        self.local_scaler_ = StandardScaler().fit(local_x)
        local_z = self.local_scaler_.transform(local_x)
        if self.kind == "GPR":
            local_kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
                length_scale=np.full(local_z.shape[1], 1.05), length_scale_bounds="fixed"
            ) + WhiteKernel(noise_level=0.08, noise_level_bounds="fixed")
            self.local_ = GaussianProcessRegressor(
                kernel=local_kernel,
                normalize_y=True,
                optimizer=None,
                alpha=1e-6,
                random_state=self.random_state,
            )
        else:
            self.local_ = SVR(kernel="rbf", C=6.0, gamma="scale", epsilon=0.04)
        self.local_.fit(local_z, residual)
        return self

    def _q(self, r: np.ndarray) -> np.ndarray:
        tau = 0.90
        if self.q_kind == "linear":
            return r
        if self.q_kind == "log":
            return tau * np.log1p(r / tau)
        if self.q_kind == "saturation":
            return tau * (1.0 - np.exp(-r / tau))
        raise ValueError(self.q_kind)

    def predict_base(self, x: np.ndarray) -> np.ndarray:
        z = self.x_scaler_.transform(np.asarray(x, dtype=float))
        return np.asarray(self.base_.predict(z)).ravel()

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.x_scaler_.transform(np.asarray(x, dtype=float))
        base = np.asarray(self.base_.predict(z)).ravel()
        ratio, za, distance, direction, outer = self._geometry_from_z(z)
        if not np.any(outer):
            return base

        za_outer = za[outer]
        r = distance[outer]
        u = direction[outer]
        y_anchor = np.asarray(self.base_.predict(za_outer)).ravel()
        phi = directional_features(za_outer, u)
        h_global = self.global_.predict(phi)
        local_x = self.local_scaler_.transform(np.column_stack([za_outer, u]))
        h_local = self.local_.predict(local_x)
        h = h_global + self.local_weight * h_local
        ext = y_anchor + self._q(r) * h
        omega = expit((ratio[outer] - 1.0) / self.delta_ratio_)
        pred = base.copy()
        pred[outer] = (1.0 - omega) * base[outer] + omega * ext
        return pred

    def extrapolation_ratio(self, x: np.ndarray) -> np.ndarray:
        """Return mean kNN distance relative to the training-calibrated AD."""
        z = self.x_scaler_.transform(np.asarray(x, dtype=float))
        return self.ad_.ratio(z)

    def _geometry_from_z(
        self,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        ratio = self.ad_.ratio(z)
        outer = ratio > 1.0 + 1e-12
        anchors = z.copy()
        if np.any(outer):
            anchors[outer] = self.ad_.anchors(z[outer])
        delta = z - anchors
        distance = np.linalg.norm(delta, axis=1)
        direction = np.zeros_like(delta)
        if np.any(outer):
            direction[outer] = delta[outer] / np.maximum(distance[outer, None], 1e-12)
        return ratio, anchors, distance, direction, outer

    def extrapolation_geometry(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return AD ratio, input-space anchor, distance, and direction."""
        z = self.x_scaler_.transform(np.asarray(x, dtype=float))
        ratio, anchors_z, distance, direction, _ = self._geometry_from_z(z)
        return ratio, self.x_scaler_.inverse_transform(anchors_z), distance, direction

    def points_at_ad_ratio(self, directions: np.ndarray, target_ratio: float) -> np.ndarray:
        """Return original-scale points on a requested kNN-AD shell."""
        shell_z = self.ad_.points_at_ratio(directions, target_ratio)
        return self.x_scaler_.inverse_transform(shell_z)

    def predict_distribution(
        self,
        x: np.ndarray,
        calibration_error_q90: float = 0.0,
        calibration_distance: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate predictive mean and standard deviation for ADE-GPR.

        The existing ``predict`` method is intentionally unchanged.  This
        distributional extension is used by Bayesian optimization.  It
        propagates the base-GPR uncertainty at the anchor and the local
        direction-rate GP uncertainty.  A training-only outer-shell absolute
        error quantile supplies a model-discrepancy term that grows with the
        chosen distance continuation law.  The base and rate models are not a
        single joint GP, so the otherwise unknown cross-covariance is handled
        conservatively by a disagreement variance in the transition band.
        """
        if self.kind != "GPR":
            raise ValueError("predict_distribution is defined for GPR only")

        x = np.asarray(x, dtype=float)
        z = self.x_scaler_.transform(x)
        base_mean, base_std = self.base_.predict(z, return_std=True)
        base_mean = np.asarray(base_mean, dtype=float).ravel()
        base_std = np.asarray(base_std, dtype=float).ravel()
        mean = base_mean.copy()
        variance = np.maximum(base_std**2, 1e-12)

        ratio, za, distance, direction, outer = self._geometry_from_z(z)
        if not np.any(outer):
            return mean, np.sqrt(variance)

        za_outer = za[outer]
        distance_outer = distance[outer]
        direction_outer = direction[outer]

        anchor_mean, anchor_std = self.base_.predict(za_outer, return_std=True)
        anchor_mean = np.asarray(anchor_mean, dtype=float).ravel()
        anchor_std = np.asarray(anchor_std, dtype=float).ravel()
        phi = directional_features(za_outer, direction_outer)
        global_rate = np.asarray(self.global_.predict(phi), dtype=float).ravel()
        local_features = self.local_scaler_.transform(np.column_stack([za_outer, direction_outer]))
        local_rate, local_std = self.local_.predict(local_features, return_std=True)
        local_rate = np.asarray(local_rate, dtype=float).ravel()
        local_std = np.asarray(local_std, dtype=float).ravel()
        rate_mean = global_rate + self.local_weight * local_rate
        rate_std = abs(self.local_weight) * local_std

        q_distance = np.asarray(self._q(distance_outer), dtype=float)
        ext_mean = anchor_mean + q_distance * rate_mean
        ext_variance = anchor_std**2 + (q_distance * rate_std) ** 2

        if calibration_error_q90 > 0:
            ref_q = float(abs(self._q(np.asarray([max(calibration_distance, 1e-8)]))[0]))
            growth = np.maximum(1.0, np.abs(q_distance) / max(ref_q, 1e-8))
            # q90 / 1.645 converts the empirical 90% absolute-error radius to
            # a Gaussian-equivalent one-standard-deviation discrepancy.
            discrepancy_std = float(calibration_error_q90) * growth / 1.645
            ext_variance = ext_variance + discrepancy_std**2

        omega = expit((ratio[outer] - 1.0) / self.delta_ratio_)
        blended_mean = (1.0 - omega) * base_mean[outer] + omega * ext_mean
        blended_variance = (
            (1.0 - omega) ** 2 * base_std[outer] ** 2
            + omega**2 * ext_variance
            + omega * (1.0 - omega) * (base_mean[outer] - ext_mean) ** 2
        )
        mean[outer] = blended_mean
        variance[outer] = np.maximum(blended_variance, 1e-12)
        return mean, np.sqrt(variance)


def fit_tuned_ade(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    seed: int,
) -> tuple[AnchorDeltaRegressor, dict[str, object]]:
    """Choose the continuation law and local weight by kNN outer-shell validation."""
    scaler = StandardScaler().fit(x)
    ad_score = knn_leave_one_out_scores(scaler.transform(x), k=5)
    cut = np.quantile(ad_score, 0.70)
    core = ad_score <= cut
    shell = ~core
    choices: list[tuple[float, str, float]] = []
    for q_kind in ("linear", "log", "saturation"):
        inner = AnchorDeltaRegressor(kind, q_kind=q_kind, local_weight=1.0, random_state=seed)
        inner.fit(x[core], y[core])
        base_shell = inner.predict_base(x[shell])
        for local_weight in (0.0, 0.5, 1.0):
            inner.local_weight = local_weight
            pred = inner.predict(x[shell])
            score = rmse(y[shell], pred)
            # A tiny stability preference avoids choosing complexity for numerical ties.
            score += 1e-6 * local_weight
            choices.append((score, q_kind, local_weight))
        inner.local_weight = 1.0
    choices.sort(key=lambda row: row[0])
    _, q_best, w_best = choices[0]
    model = AnchorDeltaRegressor(kind, q_kind=q_best, local_weight=w_best, random_state=seed)
    model.fit(x, y)
    return model, {
        "q_kind": q_best,
        "local_weight": w_best,
        "validation_rmse": choices[0][0],
        "ad_metric": "mean_kNN_distance",
        "ad_k": 5,
        "ad_boundary_quantile": 0.90,
    }


@dataclass
class ForwardScenario:
    key: str
    label: str
    dim: int
    n_train: int
    simulator: Callable[[np.ndarray], np.ndarray]
    train_sampler: Callable[[np.random.Generator, int], np.ndarray]
    test_sampler: Callable[[np.random.Generator, int], np.ndarray]
    noise_sd: float


def f_trend(x: np.ndarray) -> np.ndarray:
    t = x[:, 0]
    return 0.80 * t + 0.35 * np.sin(2.2 * t)


def f_direction(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[:, 0], x[:, 1]
    return 1.05 * x1 - 0.75 * x2 + 0.48 * x1 * x2 + 0.22 * np.sin(1.7 * x1)


def f_saturation(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[:, 0], x[:, 1]
    return 2.45 * np.tanh(0.95 * x1) + 0.62 * x2


def f_switch(x: np.ndarray) -> np.ndarray:
    t = x[:, 0]
    at = np.abs(t)
    sign = np.sign(t)
    inside = 0.95 * t + 0.08 * np.sin(2.0 * t)
    boundary = 0.95 * 1.35 + 0.08 * np.sin(2.0 * 1.35)
    outside_abs = boundary - 0.55 * (at - 1.35)
    return np.where(at <= 1.35, inside, sign * outside_abs)


def make_forward_scenarios() -> list[ForwardScenario]:
    return [
        ForwardScenario(
            "F1",
            "One-dimensional trend continuation",
            1,
            86,
            f_trend,
            lambda rng, n: rng.uniform(-1.0, 1.0, size=(n, 1)),
            lambda rng, n: np.concatenate(
                [rng.uniform(-3.0, -1.25, size=n // 2), rng.uniform(1.25, 3.0, size=n - n // 2)]
            )[:, None],
            0.045,
        ),
        ForwardScenario(
            "F2",
            "Two-dimensional direction and interaction",
            2,
            145,
            f_direction,
            lambda rng, n: uniform_disk(rng, n, 0.0, 1.0),
            lambda rng, n: uniform_disk(rng, n, 1.30, 2.65),
            0.055,
        ),
        ForwardScenario(
            "F3",
            "Two-dimensional saturating response",
            2,
            145,
            f_saturation,
            lambda rng, n: uniform_disk(rng, n, 0.0, 1.05),
            lambda rng, n: uniform_disk(rng, n, 1.30, 2.75),
            0.055,
        ),
        ForwardScenario(
            "F4",
            "Unobserved mechanism change",
            1,
            86,
            f_switch,
            lambda rng, n: rng.uniform(-1.0, 1.0, size=(n, 1)),
            lambda rng, n: np.concatenate(
                [rng.uniform(-3.0, -1.25, size=n // 2), rng.uniform(1.25, 3.0, size=n - n // 2)]
            )[:, None],
            0.045,
        ),
    ]


def forward_once(scenario: ForwardScenario, rep: int, n_test: int = 800):
    rng = np.random.default_rng(10_000 + 101 * rep + int(scenario.key[1:]))
    x_train = scenario.train_sampler(rng, scenario.n_train)
    y_true_train = scenario.simulator(x_train)
    y_train = y_true_train + rng.normal(0.0, scenario.noise_sd, len(x_train))
    x_test = scenario.test_sampler(rng, n_test)
    y_test = scenario.simulator(x_test)

    pred: dict[str, np.ndarray] = {}
    settings: list[dict[str, object]] = []
    fitted: dict[str, AnchorDeltaRegressor] = {}
    for kind in ("GPR", "SVR"):
        model, choice = fit_tuned_ade(x_train, y_train, kind, seed=rep + 17)
        fitted[kind] = model
        pred[kind] = model.predict_base(x_test)
        pred[f"AD-ADE-{kind}"] = model.predict(x_test)
        settings.append({"scenario": scenario.key, "rep": rep, "kind": kind, **choice})

    ridge = Pipeline(
        [
            ("z", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=0.20)),
        ]
    )
    ridge.fit(x_train, y_train)
    pred["Quadratic Ridge"] = ridge.predict(x_test)

    rows = []
    true_sd = float(np.std(y_test, ddof=1))
    for name, yp in pred.items():
        rows.append(
            {
                "scenario": scenario.key,
                "scenario_label": scenario.label,
                "rep": rep,
                "model": name,
                "rmse": rmse(y_test, yp),
                "mae": float(mean_absolute_error(y_test, yp)),
                "r2": float(r2_score(y_test, yp)),
                "correlation": safe_corr(y_test, yp),
                "spread_ratio": float(np.std(yp, ddof=1) / max(true_sd, 1e-12)),
            }
        )
    payload = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "pred": pred,
        "fitted": fitted,
    }
    return rows, settings, payload


class BlockScaler:
    def __init__(self) -> None:
        self.scalers: list[StandardScaler] = []

    def fit(self, blocks: Iterable[np.ndarray]) -> "BlockScaler":
        self.scalers = [StandardScaler().fit(np.asarray(b)) for b in blocks]
        return self

    def transform(self, blocks: Iterable[np.ndarray]) -> np.ndarray:
        return np.column_stack([s.transform(np.asarray(b)) for s, b in zip(self.scalers, blocks)])


def choose_gmm(z: np.ndarray, max_components: int, seed: int) -> GaussianMixture:
    upper = min(max_components, max(1, len(z) // 35))
    candidates: list[tuple[float, GaussianMixture]] = []
    for k in range(1, upper + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=2e-4,
            n_init=2,
            max_iter=400,
            random_state=seed + k * 13,
        )
        gmm.fit(z)
        candidates.append((float(gmm.bic(z)), gmm))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def conditional_components(
    gmm: GaussianMixture,
    c: np.ndarray,
    unknown_idx: np.ndarray,
    condition_idx: np.ndarray,
    ridge: float = 1e-6,
) -> list[dict[str, object]]:
    c = np.asarray(c, dtype=float).ravel()
    out: list[dict[str, object]] = []
    for k in range(gmm.n_components):
        mu = gmm.means_[k]
        cov = gmm.covariances_[k]
        mu_u = mu[unknown_idx]
        mu_c = mu[condition_idx]
        suu = cov[np.ix_(unknown_idx, unknown_idx)]
        suc = cov[np.ix_(unknown_idx, condition_idx)]
        scc = cov[np.ix_(condition_idx, condition_idx)] + ridge * np.eye(len(condition_idx))
        inv_term = np.linalg.solve(scc, c - mu_c)
        cond_mu = mu_u + suc @ inv_term
        cond_cov = suu - suc @ np.linalg.solve(scc, suc.T)
        cond_cov = 0.5 * (cond_cov + cond_cov.T) + ridge * np.eye(len(unknown_idx))
        log_evidence = math.log(max(gmm.weights_[k], 1e-300)) + float(
            multivariate_normal.logpdf(c, mean=mu_c, cov=scc, allow_singular=True)
        )
        out.append(
            {
                "component": k,
                "mean": cond_mu,
                "cov": cond_cov,
                "log_evidence": log_evidence,
            }
        )
    return out


def grouped_multiscale_pairs(
    x: np.ndarray,
    groups: np.ndarray | None,
    ranks: tuple[int, ...] = (1, 3, 7, 15, 31),
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if groups is None:
        groups = np.zeros(n, dtype=int)
    origins: list[int] = []
    targets: list[int] = []
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) < 3:
            continue
        local_ranks = tuple(r for r in ranks if r < len(idx))
        if not local_ranks:
            local_ranks = (1,)
        nbr = NearestNeighbors(n_neighbors=max(local_ranks) + 1).fit(z[idx])
        _, ind = nbr.kneighbors(z[idx])
        for local_i, global_i in enumerate(idx):
            for rank in local_ranks:
                global_j = idx[ind[local_i, rank]]
                if global_j != global_i:
                    origins.append(int(global_i))
                    targets.append(int(global_j))
    return np.asarray(origins, dtype=int), np.asarray(targets, dtype=int)


class TransitionGMR:
    def __init__(self, max_components: int = 7, random_state: int = 0) -> None:
        self.max_components = max_components
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> "TransitionGMR":
        self.x_ = np.asarray(x, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        if self.y_.ndim == 1:
            self.y_ = self.y_[:, None]
        i_idx, j_idx = grouped_multiscale_pairs(self.x_, groups)
        xa = self.x_[i_idx]
        ya = self.y_[i_idx]
        dx = self.x_[j_idx] - self.x_[i_idx]
        dy = self.y_[j_idx] - self.y_[i_idx]
        self.block_scaler_ = BlockScaler().fit([xa, ya, dx, dy])
        z = self.block_scaler_.transform([xa, ya, dx, dy])
        self.px_ = self.x_.shape[1]
        self.py_ = self.y_.shape[1]
        self.gmm_ = choose_gmm(z, self.max_components, self.random_state)
        return self

    def candidates(self, anchor_indices: np.ndarray, target_y: np.ndarray) -> pd.DataFrame:
        target_y = np.asarray(target_y, dtype=float).reshape(1, -1)
        sx, sy, sdx, sdy = self.block_scaler_.scalers
        unknown_idx = np.arange(self.px_ + self.py_, 2 * self.px_ + self.py_)
        condition_idx = np.r_[
            np.arange(0, self.px_ + self.py_),
            np.arange(2 * self.px_ + self.py_, 2 * self.px_ + 2 * self.py_),
        ]
        rows: list[dict[str, object]] = []
        for a in np.asarray(anchor_indices, dtype=int):
            xa = self.x_[a : a + 1]
            ya = self.y_[a : a + 1]
            dy = target_y - ya
            c = np.r_[sx.transform(xa).ravel(), sy.transform(ya).ravel(), sdy.transform(dy).ravel()]
            comps = conditional_components(self.gmm_, c, unknown_idx, condition_idx)
            for comp in comps:
                dx_z = np.asarray(comp["mean"]).reshape(1, -1)
                dx = sdx.inverse_transform(dx_z).ravel()
                x_candidate = xa.ravel() + dx
                scale = sdx.scale_
                cov_raw = np.diag(scale) @ np.asarray(comp["cov"]) @ np.diag(scale)
                rows.append(
                    {
                        "anchor": int(a),
                        "component": int(comp["component"]),
                        "log_score": float(comp["log_evidence"] - math.log(len(anchor_indices))),
                        "cov_trace": float(np.trace(cov_raw)),
                        **{f"x{j+1}": float(v) for j, v in enumerate(x_candidate)},
                    }
                )
        frame = pd.DataFrame(rows)
        if len(frame):
            frame["weight"] = np.exp(frame["log_score"] - logsumexp(frame["log_score"]))
            frame = frame.sort_values("log_score", ascending=False).reset_index(drop=True)
        return frame

    @property
    def n_components_(self) -> int:
        return int(self.gmm_.n_components)


class AbsoluteGMR:
    def __init__(self, max_components: int = 7, random_state: int = 0) -> None:
        self.max_components = max_components
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AbsoluteGMR":
        self.x_ = np.asarray(x, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        if self.y_.ndim == 1:
            self.y_ = self.y_[:, None]
        self.sx_ = StandardScaler().fit(self.x_)
        self.sy_ = StandardScaler().fit(self.y_)
        z = np.column_stack([self.sx_.transform(self.x_), self.sy_.transform(self.y_)])
        self.px_ = self.x_.shape[1]
        self.py_ = self.y_.shape[1]
        self.gmm_ = choose_gmm(z, self.max_components, self.random_state)
        return self

    def candidates(self, target_y: np.ndarray) -> pd.DataFrame:
        target_y = np.asarray(target_y, dtype=float).reshape(1, -1)
        c = self.sy_.transform(target_y).ravel()
        unknown_idx = np.arange(self.px_)
        condition_idx = np.arange(self.px_, self.px_ + self.py_)
        comps = conditional_components(self.gmm_, c, unknown_idx, condition_idx)
        rows: list[dict[str, object]] = []
        for comp in comps:
            x_z = np.asarray(comp["mean"]).reshape(1, -1)
            x_candidate = self.sx_.inverse_transform(x_z).ravel()
            cov_raw = np.diag(self.sx_.scale_) @ np.asarray(comp["cov"]) @ np.diag(self.sx_.scale_)
            rows.append(
                {
                    "component": int(comp["component"]),
                    "log_score": float(comp["log_evidence"]),
                    "cov_trace": float(np.trace(cov_raw)),
                    **{f"x{j+1}": float(v) for j, v in enumerate(x_candidate)},
                }
            )
        frame = pd.DataFrame(rows)
        frame["weight"] = np.exp(frame["log_score"] - logsumexp(frame["log_score"]))
        return frame.sort_values("log_score", ascending=False).reset_index(drop=True)

    @property
    def n_components_(self) -> int:
        return int(self.gmm_.n_components)


def f_quadratic(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x)[:, 0] ** 2)[:, None]


def f_coupled(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x1, x2 = x[:, 0], x[:, 1]
    y1 = 1.10 * x1 + 0.45 * x2 + 0.25 * x1 * x2
    y2 = -0.25 * x1 + 1.00 * x2 + 0.20 * x1**2
    return np.column_stack([y1, y2])


def nearest_y_anchors(y_train: np.ndarray, target: np.ndarray, n_anchor: int) -> np.ndarray:
    scaler = StandardScaler().fit(y_train)
    d = np.linalg.norm(scaler.transform(y_train) - scaler.transform(np.asarray(target).reshape(1, -1)), axis=1)
    return np.argsort(d)[:n_anchor]


def maximin_unit_directions(z: np.ndarray, maximum: int) -> np.ndarray:
    """Return a deterministic, directionally diverse subset of nonzero rows."""
    z = np.asarray(z, dtype=float)
    radius = np.linalg.norm(z, axis=1)
    keep = np.isfinite(radius) & (radius > 1e-10)
    if not np.any(keep):
        raise ValueError("No nonzero directions are available")
    z = z[keep]
    radius = radius[keep]
    unit = z / radius[:, None]
    target = min(maximum, len(unit))
    chosen = [int(np.argmax(radius))]
    min_distance = np.linalg.norm(unit - unit[chosen[0]], axis=1)
    while len(chosen) < target:
        index = int(np.argmax(min_distance))
        if index in chosen or min_distance[index] < 1e-8:
            break
        chosen.append(index)
        min_distance = np.minimum(min_distance, np.linalg.norm(unit - unit[index], axis=1))
    return unit[np.asarray(chosen, dtype=int)]


@dataclass
class CompetitiveGMRResult:
    candidates: pd.DataFrame
    supported: bool
    reason: str
    absolute_log_evidence: float
    transition_log_evidence: float
    absolute_support_margin: float
    transition_support_margin: float


class EvidenceGatedCompetitiveGMR:
    """Direct inverse analysis using evidence-gated competition between two GMRs.

    Absolute-GMR and Transition-GMR remain analytic conditional-Gaussian inverse
    models.  Their raw log evidences are not directly comparable because their
    conditioning dimensions differ.  Each model is therefore calibrated on a
    training-only response shell, and a model-specific support margin is used
    for gating and model order.  Feasible candidates are then interleaved by
    calibrated support and filtered for diversity.  No numerical optimization
    of x is performed.
    """

    def __init__(
        self,
        max_components: int = 7,
        random_state: int = 0,
        support_shell_ratio: float = 2.0,
        support_quantile: float = 0.05,
        max_support_directions: int = 24,
        uncertainty_penalty: float = 0.05,
        minimum_candidate_distance: float = 0.025,
    ) -> None:
        self.max_components = max_components
        self.random_state = random_state
        self.support_shell_ratio = support_shell_ratio
        self.support_quantile = support_quantile
        self.max_support_directions = max_support_directions
        self.uncertainty_penalty = uncertainty_penalty
        self.minimum_candidate_distance = minimum_candidate_distance

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> "EvidenceGatedCompetitiveGMR":
        self.x_ = np.asarray(x, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        if self.y_.ndim == 1:
            self.y_ = self.y_[:, None]
        self.x_scaler_ = StandardScaler().fit(self.x_)
        self.y_scaler_ = StandardScaler().fit(self.y_)
        self.absolute_ = AbsoluteGMR(
            max_components=self.max_components,
            random_state=self.random_state + 101,
        ).fit(self.x_, self.y_)
        self.transition_ = TransitionGMR(
            max_components=self.max_components,
            random_state=self.random_state + 1101,
        ).fit(self.x_, self.y_, groups)

        train_y_z = self.y_scaler_.transform(self.y_)
        self.support_radius_ = float(np.quantile(np.linalg.norm(train_y_z, axis=1), 0.90))
        directions = maximin_unit_directions(train_y_z, self.max_support_directions)
        shell_z = directions * (self.support_shell_ratio * self.support_radius_)
        shell_y = self.y_scaler_.inverse_transform(shell_z)
        calibration: dict[str, list[float]] = {"Absolute-GMR": [], "Transition-GMR": []}
        for target_y in shell_y:
            anchors = nearest_y_anchors(self.y_, target_y, n_anchor=min(12, len(self.y_)))
            absolute = self.absolute_.candidates(target_y)
            transition = self.transition_.candidates(anchors, target_y)
            calibration["Absolute-GMR"].append(float(absolute["log_score"].max()))
            calibration["Transition-GMR"].append(float(transition["log_score"].max()))

        self.support_threshold_: dict[str, float] = {}
        self.support_scale_: dict[str, float] = {}
        for method, values in calibration.items():
            array = np.asarray(values, dtype=float)
            self.support_threshold_[method] = float(np.quantile(array, self.support_quantile))
            iqr = float(np.quantile(array, 0.75) - np.quantile(array, 0.25))
            self.support_scale_[method] = max(iqr, 1.0)
        self.support_calibration_ = calibration
        return self

    def _support_margin(self, method: str, log_evidence: float) -> float:
        return float(
            (log_evidence - self.support_threshold_[method])
            / self.support_scale_[method]
        )

    def _prepare_candidates(
        self,
        frame: pd.DataFrame,
        bounds: np.ndarray,
        source_method: str,
        support_margin: float,
    ) -> pd.DataFrame:
        x_cols = [f"x{j + 1}" for j in range(len(bounds))]
        feasible = frame.copy()
        for j, column in enumerate(x_cols):
            feasible = feasible[feasible[column].between(bounds[j, 0], bounds[j, 1])]
        if len(feasible) == 0:
            return feasible
        log_weight = feasible["log_score"].to_numpy(dtype=float)
        log_weight = log_weight - logsumexp(log_weight)
        variance_scale = max(float(np.sum(self.x_scaler_.scale_**2)), 1e-12)
        uncertainty = feasible["cov_trace"].to_numpy(dtype=float) / variance_scale
        feasible = feasible.copy()
        feasible["source_method"] = source_method
        feasible["model_support_margin"] = support_margin
        feasible["candidate_quality"] = log_weight - self.uncertainty_penalty * np.log1p(uncertainty)
        feasible = feasible.sort_values("candidate_quality", ascending=False).reset_index(drop=True)

        selected_rows: list[pd.Series] = []
        selected_x: list[np.ndarray] = []
        for _, row in feasible.iterrows():
            candidate = row[x_cols].to_numpy(dtype=float)
            candidate_z = self.x_scaler_.transform(candidate.reshape(1, -1)).ravel()
            if not selected_x or min(np.linalg.norm(candidate_z - prior) for prior in selected_x) > self.minimum_candidate_distance:
                selected_rows.append(row)
                selected_x.append(candidate_z)
        if not selected_rows:
            return feasible.iloc[0:0].copy()
        out = pd.DataFrame(selected_rows).reset_index(drop=True)
        out["source_rank"] = np.arange(1, len(out) + 1)
        return out

    def candidates(
        self,
        target_y: np.ndarray,
        bounds: np.ndarray,
        top_k: int = 5,
        n_anchor: int = 12,
    ) -> CompetitiveGMRResult:
        target_y = np.asarray(target_y, dtype=float).ravel()
        anchors = nearest_y_anchors(self.y_, target_y, n_anchor=min(n_anchor, len(self.y_)))
        absolute = self.absolute_.candidates(target_y)
        transition = self.transition_.candidates(anchors, target_y)
        absolute_log = float(absolute["log_score"].max())
        transition_log = float(transition["log_score"].max())
        margins = {
            "Absolute-GMR": self._support_margin("Absolute-GMR", absolute_log),
            "Transition-GMR": self._support_margin("Transition-GMR", transition_log),
        }
        raw_frames = {"Absolute-GMR": absolute, "Transition-GMR": transition}
        eligible: dict[str, pd.DataFrame] = {}
        for method, frame in raw_frames.items():
            if margins[method] < 0:
                continue
            prepared = self._prepare_candidates(frame, bounds, method, margins[method])
            if len(prepared):
                eligible[method] = prepared

        empty_columns = list(absolute.columns) + [
            "source_method",
            "model_support_margin",
            "candidate_quality",
            "source_rank",
            "hybrid_rank",
        ]
        if not eligible:
            reason = "unsupported_evidence"
            if any(value >= 0 for value in margins.values()):
                reason = "no_feasible_candidate"
            return CompetitiveGMRResult(
                candidates=pd.DataFrame(columns=list(dict.fromkeys(empty_columns))),
                supported=False,
                reason=reason,
                absolute_log_evidence=absolute_log,
                transition_log_evidence=transition_log,
                absolute_support_margin=margins["Absolute-GMR"],
                transition_support_margin=margins["Transition-GMR"],
            )

        method_order = sorted(eligible, key=lambda method: margins[method], reverse=True)
        selected_rows: list[pd.Series] = []
        selected_z: list[np.ndarray] = []
        positions = {method: 0 for method in method_order}
        x_cols = [f"x{j + 1}" for j in range(len(bounds))]
        while len(selected_rows) < top_k:
            added = False
            for method in method_order:
                frame = eligible[method]
                while positions[method] < len(frame):
                    row = frame.iloc[positions[method]]
                    positions[method] += 1
                    candidate = row[x_cols].to_numpy(dtype=float)
                    candidate_z = self.x_scaler_.transform(candidate.reshape(1, -1)).ravel()
                    if selected_z and min(np.linalg.norm(candidate_z - prior) for prior in selected_z) <= self.minimum_candidate_distance:
                        continue
                    selected_rows.append(row)
                    selected_z.append(candidate_z)
                    added = True
                    break
                if len(selected_rows) >= top_k:
                    break
            if not added:
                break

        candidates = pd.DataFrame(selected_rows).reset_index(drop=True)
        candidates["hybrid_rank"] = np.arange(1, len(candidates) + 1)
        return CompetitiveGMRResult(
            candidates=candidates,
            supported=bool(len(candidates)),
            reason="supported",
            absolute_log_evidence=absolute_log,
            transition_log_evidence=transition_log,
            absolute_support_margin=margins["Absolute-GMR"],
            transition_support_margin=margins["Transition-GMR"],
        )


def inverse_quadratic_once(rep: int, n_targets: int = 22):
    rng = np.random.default_rng(30_000 + rep * 131)
    n_side = 95
    x_neg = rng.uniform(-1.42, -0.20, n_side)
    x_pos = rng.uniform(0.20, 1.42, n_side)
    x = np.concatenate([x_neg, x_pos])[:, None]
    groups = np.concatenate([np.zeros(n_side, dtype=int), np.ones(n_side, dtype=int)])
    y_true = f_quadratic(x)
    y = y_true + rng.normal(0.0, 0.018, y_true.shape)

    tgmr = TransitionGMR(max_components=7, random_state=rep + 101).fit(x, y, groups)
    agmr = AbsoluteGMR(max_components=7, random_state=rep + 201).fit(x, y)
    x_abs = np.linspace(1.48, 2.05, n_targets)
    targets = x_abs**2
    rows: list[dict[str, object]] = []
    example: dict[str, object] | None = None
    for ti, (xa, target) in enumerate(zip(x_abs, targets)):
        target_arr = np.array([target])
        branch_anchors = []
        for g in (0, 1):
            idx = np.flatnonzero(groups == g)
            order = idx[np.argsort(np.abs(y[idx, 0] - target))[:4]]
            branch_anchors.extend(order.tolist())
        candidate_sets = {
            "Absolute-coordinate GMR": agmr.candidates(target_arr),
            "Transition-GMR direct inverse": tgmr.candidates(np.asarray(branch_anchors), target_arr),
        }
        roots = np.array([-xa, xa])
        for method, cand in candidate_sets.items():
            cand = cand[(cand["x1"] >= -2.5) & (cand["x1"] <= 2.5)].copy()
            if len(cand) == 0:
                rows.append(
                    {
                        "scenario": "I1",
                        "scenario_label": "Non-monotonic inverse problem with two solutions",
                        "rep": rep,
                        "target_index": ti,
                        "method": method,
                        "response_error": np.nan,
                        "input_set_error": np.nan,
                        "both_branch_success": 0.0,
                    }
                )
                continue
            top = cand.iloc[0]
            x_top = np.array([[top["x1"]]])
            response_error = float(abs(f_quadratic(x_top)[0, 0] - target))
            cand_top = cand.head(min(14, len(cand)))["x1"].to_numpy()
            branch_errors = np.array([np.min(np.abs(cand_top - root)) for root in roots])
            rows.append(
                {
                    "scenario": "I1",
                    "scenario_label": "Non-monotonic inverse problem with two solutions",
                    "rep": rep,
                    "target_index": ti,
                    "method": method,
                    "response_error": response_error,
                    "input_set_error": float(np.mean(branch_errors)),
                    "both_branch_success": float(np.all(branch_errors < 0.28)),
                }
            )
        if rep == 0 and ti == n_targets // 2:
            example = {
                "x_train": x,
                "y_train": y,
                "target": target,
                "roots": roots,
                "absolute": candidate_sets["Absolute-coordinate GMR"],
                "transition": candidate_sets["Transition-GMR direct inverse"],
            }
    settings = {
        "scenario": "I1",
        "rep": rep,
        "absolute_components": agmr.n_components_,
        "transition_components": tgmr.n_components_,
    }
    return rows, settings, example


def inverse_coupled_once(rep: int, n_targets: int = 28):
    rng = np.random.default_rng(40_000 + rep * 149)
    x = uniform_disk(rng, 230, 0.0, 1.0)
    y_true = f_coupled(x)
    y = y_true + rng.normal(0.0, 0.020, y_true.shape)
    tgmr = TransitionGMR(max_components=8, random_state=rep + 301).fit(x, y)
    agmr = AbsoluteGMR(max_components=8, random_state=rep + 401).fit(x, y)
    x_target = uniform_disk(rng, n_targets, 1.18, 1.82)
    y_target = f_coupled(x_target)

    rows: list[dict[str, object]] = []
    example: dict[str, object] | None = None
    for ti in range(n_targets):
        target = y_target[ti]
        anchors = nearest_y_anchors(y, target, n_anchor=12)
        candidate_sets = {
            "Absolute-coordinate GMR": agmr.candidates(target),
            "Transition-GMR direct inverse": tgmr.candidates(anchors, target),
        }
        for method, cand in candidate_sets.items():
            feasible = cand[
                (cand["x1"].between(-2.2, 2.2)) & (cand["x2"].between(-2.2, 2.2))
            ].copy()
            if len(feasible) == 0:
                rows.append(
                    {
                        "scenario": "I2",
                        "scenario_label": "Coupled two-input, two-response inverse problem",
                        "rep": rep,
                        "target_index": ti,
                        "method": method,
                        "response_error": np.nan,
                        "input_error": np.nan,
                        "success": 0.0,
                    }
                )
                continue
            top = feasible.iloc[0]
            x_hat = np.array([[top["x1"], top["x2"]]])
            y_hat = f_coupled(x_hat)[0]
            response_error = float(np.linalg.norm(y_hat - target))
            input_error = float(np.linalg.norm(x_hat[0] - x_target[ti]))
            rows.append(
                {
                    "scenario": "I2",
                    "scenario_label": "Coupled two-input, two-response inverse problem",
                    "rep": rep,
                    "target_index": ti,
                    "method": method,
                    "response_error": response_error,
                    "input_error": input_error,
                    "success": float(response_error < 0.25),
                }
            )
        if rep == 0 and ti == 0:
            example = {
                "x_train": x,
                "y_train": y,
                "x_target": x_target[ti],
                "y_target": target,
                "absolute": candidate_sets["Absolute-coordinate GMR"],
                "transition": candidate_sets["Transition-GMR direct inverse"],
            }

    # Unsupported-target diagnostic: compare transition evidence for reachable and remote targets.
    remote_x = uniform_disk(rng, n_targets, 2.7, 3.1)
    remote_y = f_coupled(remote_x)
    evidence_rows = []
    for label, ys in ((0, y_target), (1, remote_y)):
        for target in ys:
            anchors = nearest_y_anchors(y, target, n_anchor=12)
            cand = tgmr.candidates(anchors, target)
            evidence_rows.append({"unsupported": label, "max_log_evidence": float(cand["log_score"].max())})
    ev = pd.DataFrame(evidence_rows)
    # Higher IoE = lower evidence. AUC near 1 means the warning statistic separates unsupported targets.
    auc = float(roc_auc_score(ev["unsupported"], -ev["max_log_evidence"]))
    settings = {
        "scenario": "I2",
        "rep": rep,
        "absolute_components": agmr.n_components_,
        "transition_components": tgmr.n_components_,
        "unsupported_target_auc": auc,
    }
    return rows, settings, example


def quantile_summary(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{metric}_q25"] = float(np.quantile(values, 0.25)) if len(values) else np.nan
            row[f"{metric}_q75"] = float(np.quantile(values, 0.75)) if len(values) else np.nan
        records.append(row)
    return pd.DataFrame(records)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D0D5DD", alpha=0.55, linewidth=0.7)
    ax.tick_params(labelsize=8)


def make_forward_example_figure(examples: dict[str, dict[str, object]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    for ax, scenario_key, title in zip(
        axes,
        ("F1", "F4"),
        ("(a) Trend continues outside", "(b) Mechanism reverses outside training"),
    ):
        ex = examples[scenario_key]
        order = np.argsort(ex["x_test"][:, 0])
        xt = ex["x_test"][order, 0]
        ax.scatter(ex["x_train"][:, 0], ex["y_train"], s=15, color="#344054", alpha=0.60, label="Training data")
        ax.plot(xt, ex["y_test"][order], color="black", linewidth=2.2, label="Ground truth")
        ax.plot(xt, ex["pred"]["GPR"][order], color=GRAY, linestyle="--", linewidth=1.8, label="RBF-GPR")
        ax.plot(xt, ex["pred"]["AD-ADE-GPR"][order], color=OUTER_BLUE, linewidth=2.0, label="AD-ADE-GPR")
        ax.plot(xt, ex["pred"]["SVR"][order], color="#98A2B3", linestyle=":", linewidth=1.8, label="RBF-SVR")
        ax.plot(xt, ex["pred"]["AD-ADE-SVR"][order], color=ORANGE, linewidth=1.7, label="AD-ADE-SVR")
        ax.axvspan(-1.0, 1.0, color=LIGHT, alpha=0.65, zorder=-5)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("x (training range shaded light blue)", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_direction_figure(example: dict[str, object], out: Path) -> None:
    x = example["x_test"]
    radius = np.linalg.norm(x, axis=1)
    band = np.abs(radius - 2.2) < 0.08
    theta = np.arctan2(x[band, 1], x[band, 0])
    order = np.argsort(theta)
    fig, ax = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
    ax.plot(theta[order], example["y_test"][band][order], color="black", linewidth=2.2, label="Ground truth")
    ax.plot(theta[order], example["pred"]["GPR"][band][order], color=GRAY, linestyle="--", linewidth=1.8, label="RBF-GPR")
    ax.plot(theta[order], example["pred"]["AD-ADE-GPR"][band][order], color=OUTER_BLUE, linewidth=2.0, label="AD-ADE-GPR")
    ax.plot(theta[order], example["pred"]["AD-ADE-SVR"][band][order], color=ORANGE, linewidth=1.7, label="AD-ADE-SVR")
    ax.set_xlabel("Extrapolation direction theta [rad] (radius about 2.2)", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_title("Predictions by direction at a fixed extrapolation distance", fontsize=11, fontweight="bold", loc="left")
    style_axes(ax)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_forward_summary_figure(summary: pd.DataFrame, out: Path) -> None:
    order_models = ["GPR", "AD-ADE-GPR", "SVR", "AD-ADE-SVR", "Quadratic Ridge"]
    colors = [GRAY, OUTER_BLUE, "#98A2B3", ORANGE, TEAL]
    scenarios = ["F1", "F2", "F3", "F4"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    width = 0.15
    xloc = np.arange(len(scenarios))
    for mi, (model, color) in enumerate(zip(order_models, colors)):
        data = summary[summary["model"] == model].set_index("scenario").reindex(scenarios)
        y = data["rmse_median"].to_numpy()
        lo = y - data["rmse_q25"].to_numpy()
        hi = data["rmse_q75"].to_numpy() - y
        axes[0].bar(xloc + (mi - 2) * width, y, width, color=color, label=model)
        axes[0].errorbar(xloc + (mi - 2) * width, y, yerr=np.vstack([lo, hi]), fmt="none", ecolor="#344054", capsize=2, linewidth=0.7)
        spread = data["spread_ratio_median"].to_numpy()
        axes[1].bar(xloc + (mi - 2) * width, spread, width, color=color, label=model)
    axes[0].set_title("(a) Extrapolation RMSE (median and IQR)", fontsize=11, fontweight="bold", loc="left")
    axes[0].set_ylabel("RMSE", fontsize=9)
    axes[1].set_title("(b) Predicted spread / reference spread", fontsize=11, fontweight="bold", loc="left")
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Spread ratio (1 is ideal)", fontsize=9)
    for ax in axes:
        ax.set_xticks(xloc, scenarios)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_inverse_figure(q_example: dict[str, object], c_example: dict[str, object], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    ax = axes[0]
    x_grid = np.linspace(-2.45, 2.45, 500)
    ax.plot(x_grid, x_grid**2, color="black", linewidth=2.0, label="True response curve")
    ax.scatter(q_example["x_train"][:, 0], q_example["y_train"][:, 0], s=11, color="#98A2B3", alpha=0.45, label="Training data")
    ax.axhline(q_example["target"], color=RED, linestyle="--", linewidth=1.5, label="Target y")
    for method, frame, color, marker in (
        ("Absolute-coordinate GMR", q_example["absolute"], GRAY, "x"),
        ("Transition-GMR direct inverse", q_example["transition"], OUTER_BLUE, "o"),
    ):
        top = frame.head(12)
        sizes = 18 + 130 * top["weight"].to_numpy() / max(top["weight"].max(), 1e-12)
        ax.scatter(top["x1"], np.full(len(top), q_example["target"]), s=sizes, color=color, marker=marker, alpha=0.78, label=method)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.1, 5.3)
    ax.set_xlabel("Candidate x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_title("(a) Non-monotonic system: retaining two inverse solutions", fontsize=11, fontweight="bold", loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)

    ax = axes[1]
    train = c_example["x_train"]
    ax.scatter(train[:, 0], train[:, 1], s=10, color="#98A2B3", alpha=0.32, label="Training inputs")
    for method, frame, color, marker in (
        ("Absolute-coordinate GMR", c_example["absolute"], GRAY, "x"),
        ("Transition-GMR direct inverse", c_example["transition"], OUTER_BLUE, "o"),
    ):
        top = frame.head(12)
        sizes = 18 + 110 * top["weight"].to_numpy() / max(top["weight"].max(), 1e-12)
        ax.scatter(top["x1"], top["x2"], s=sizes, color=color, marker=marker, alpha=0.75, label=method)
    ax.scatter(
        c_example["x_target"][0],
        c_example["x_target"][1],
        s=150,
        color=RED,
        edgecolor="white",
        linewidth=0.8,
        marker="*",
        zorder=10,
        label="True external input",
    )
    circle = plt.Circle((0, 0), 1.0, color=LIGHT, fill=False, linewidth=1.5, linestyle="--")
    ax.add_patch(circle)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1", fontsize=9)
    ax.set_ylabel("x2", fontsize=9)
    ax.set_title("(b) Coupled system: direct candidate generation from a target", fontsize=11, fontweight="bold", loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_inverse_summary_figure(i1: pd.DataFrame, i2: pd.DataFrame, out: Path) -> None:
    s1 = i1.groupby(["rep", "method"]).agg(
        response_error=("response_error", "mean"),
        set_error=("input_set_error", "mean"),
        branch_success=("both_branch_success", "mean"),
    ).reset_index()
    s2 = i2.groupby(["rep", "method"]).agg(
        response_error=("response_error", "mean"),
        input_error=("input_error", "mean"),
        success=("success", "mean"),
    ).reset_index()
    methods = ["Absolute-coordinate GMR", "Transition-GMR direct inverse"]
    colors = [GRAY, OUTER_BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for ax, data, metric, title, ylabel in (
        (axes[0], s1, "set_error", "(a) Input error for the two-solution set", "Mean nearest-solution error"),
        (axes[1], s2, "response_error", "(b) Target-response error in the coupled system", "||f(x_hat) - y*||2"),
    ):
        med = data.groupby("method")[metric].median().reindex(methods)
        q25 = data.groupby("method")[metric].quantile(0.25).reindex(methods)
        q75 = data.groupby("method")[metric].quantile(0.75).reindex(methods)
        pos = np.arange(len(methods))
        ax.bar(pos, med, color=colors, width=0.58)
        ax.errorbar(pos, med, yerr=np.vstack([med - q25, q75 - med]), fmt="none", ecolor="#344054", capsize=4)
        ax.set_xticks(pos, ["Absolute-coordinate\nGMR", "Transition-GMR\ndirect inverse"])
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel(ylabel.replace("x̂", "x_pred"), fontsize=9)
        style_axes(ax)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-reps", type=int, default=20)
    parser.add_argument("--inverse-reps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("validation_output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    forward_rows: list[dict[str, object]] = []
    forward_settings: list[dict[str, object]] = []
    forward_examples: dict[str, dict[str, object]] = {}
    for scenario in make_forward_scenarios():
        for rep in range(args.forward_reps):
            rows, settings, payload = forward_once(scenario, rep)
            forward_rows.extend(rows)
            forward_settings.extend(settings)
            if rep == 0:
                forward_examples[scenario.key] = payload
            print(f"forward {scenario.key} rep {rep + 1}/{args.forward_reps}", flush=True)

    forward = pd.DataFrame(forward_rows)
    forward_setting_df = pd.DataFrame(forward_settings)
    forward_summary = quantile_summary(
        forward,
        ["scenario", "scenario_label", "model"],
        ["rmse", "mae", "r2", "correlation", "spread_ratio"],
    )
    forward.to_csv(args.output / "forward_raw.csv", index=False)
    forward_setting_df.to_csv(args.output / "forward_settings.csv", index=False)
    forward_summary.to_csv(args.output / "forward_summary.csv", index=False)

    i1_rows: list[dict[str, object]] = []
    i2_rows: list[dict[str, object]] = []
    inverse_settings: list[dict[str, object]] = []
    q_example = None
    c_example = None
    for rep in range(args.inverse_reps):
        rows, settings, example = inverse_quadratic_once(rep)
        i1_rows.extend(rows)
        inverse_settings.append(settings)
        if example is not None:
            q_example = example
        print(f"inverse I1 rep {rep + 1}/{args.inverse_reps}", flush=True)
        rows, settings, example = inverse_coupled_once(rep)
        i2_rows.extend(rows)
        inverse_settings.append(settings)
        if example is not None:
            c_example = example
        print(f"inverse I2 rep {rep + 1}/{args.inverse_reps}", flush=True)

    i1 = pd.DataFrame(i1_rows)
    i2 = pd.DataFrame(i2_rows)
    inv_setting_df = pd.DataFrame(inverse_settings)
    i1.to_csv(args.output / "inverse_quadratic_raw.csv", index=False)
    i2.to_csv(args.output / "inverse_coupled_raw.csv", index=False)
    inv_setting_df.to_csv(args.output / "inverse_settings.csv", index=False)

    i1_rep = i1.groupby(["rep", "method"], as_index=False).agg(
        response_error=("response_error", "mean"),
        input_set_error=("input_set_error", "mean"),
        both_branch_success=("both_branch_success", "mean"),
    )
    i2_rep = i2.groupby(["rep", "method"], as_index=False).agg(
        response_error=("response_error", "mean"),
        input_error=("input_error", "mean"),
        success=("success", "mean"),
    )
    i1_summary = quantile_summary(
        i1_rep,
        ["method"],
        ["response_error", "input_set_error", "both_branch_success"],
    )
    i2_summary = quantile_summary(
        i2_rep,
        ["method"],
        ["response_error", "input_error", "success"],
    )
    i1_summary.to_csv(args.output / "inverse_quadratic_summary.csv", index=False)
    i2_summary.to_csv(args.output / "inverse_coupled_summary.csv", index=False)

    make_forward_example_figure(forward_examples, args.output / "fig_forward_1d.png")
    make_direction_figure(forward_examples["F2"], args.output / "fig_forward_direction.png")
    make_forward_summary_figure(forward_summary, args.output / "fig_forward_summary.png")
    if q_example is not None and c_example is not None:
        make_inverse_figure(q_example, c_example, args.output / "fig_inverse_examples.png")
        make_inverse_summary_figure(i1, i2, args.output / "fig_inverse_summary.png")

    manifest = {
        "forward_reps": args.forward_reps,
        "inverse_reps": args.inverse_reps,
        "random_seed_scheme": "deterministic scenario/replicate offsets; see script",
        "representation": "componentwise StandardScaler only; no PCA/PLS/nonlinear transform",
        "forward_scenarios": [s.__dict__ | {"simulator": s.simulator.__name__, "train_sampler": "callable", "test_sampler": "callable"} for s in make_forward_scenarios()],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("completed", flush=True)


if __name__ == "__main__":
    main()
