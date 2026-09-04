#!/usr/bin/env python3
"""Pool-based Bayesian-optimization validation on the seven study systems.

The benchmark deliberately starts from the existing inner training region and
searches only the held-out outer candidate pool.  It therefore asks the exact
question motivating AD-guided ADE: can a model rank *different extrapolation
regions*, rather than collapsing to the same far-field prediction?

The proposed method combines

* the kNN-AD-guided ADE predictive mean,
* an ADE uncertainty made from anchor-GPR, direction-rate-GP and a
  training-only outer-shell discrepancy term,
* a training-calibrated kNN applicability-domain support gate,
* a UCB acquisition with a mild extrapolation-risk penalty, and
* optional three-point maximin batching without Kriging-believer fantasies.

Only componentwise standardization is used.  The simulator/experimental
outputs in the cached D1--D7 datasets are revealed only after a candidate is
selected; they are otherwise used solely for offline evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib import font_manager
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from run_numerical_validation import (
    AnchorDeltaRegressor,
    KNNApplicabilityDomain,
    fit_tuned_ade,
    knn_leave_one_out_scores,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "physics_validation_results"
DATASETS = RESULTS / "datasets"

BLUE = "#0B5FA5"
ORANGE = "#D97706"
GRAY = "#667085"
LIGHT_GRAY = "#D0D5DD"
TEAL = "#0E7C86"
RED = "#B42318"
PURPLE = "#7A5AF8"

FONT_PATH = ROOT / "qa_fonts" / "NotoSansCJKjp-Regular.otf"
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


def exact_wilcoxon_signed_rank(values: np.ndarray) -> tuple[float, float]:
    """Exact two-sided signed-rank test after discarding numerical zeros.

    Explicit sign enumeration keeps the six-dataset inference unambiguous,
    including when two or more absolute differences have tied ranks.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.abs(array) > 1e-12]
    if len(array) == 0:
        return 0.0, 1.0
    ranks = rankdata(np.abs(array), method="average")
    total = float(np.sum(ranks))
    positive = float(np.sum(ranks[array > 0]))
    statistic = min(positive, total - positive)
    extreme = 0
    permutations = 2 ** len(array)
    for signs in itertools.product((False, True), repeat=len(array)):
        candidate_positive = float(np.sum(ranks[np.asarray(signs, dtype=bool)]))
        candidate_statistic = min(candidate_positive, total - candidate_positive)
        extreme += int(candidate_statistic <= statistic + 1e-12)
    return statistic, extreme / permutations


@dataclass(frozen=True)
class ObjectiveDefinition:
    weights: tuple[float, ...]
    label_ja: str
    formula_ja: str


OBJECTIVES: dict[str, ObjectiveDefinition] = {
    "D1": ObjectiveDefinition(
        (1.0, -0.10, -0.05),
        "High conversion with low temperature and heat-removal burdens",
        "+1.00 conversion -0.10 reactor_temperature -0.05 heat_removal",
    ),
    "D2": ObjectiveDefinition(
        (0.0, 0.0, 1.0, 0.20),
        "Liquid-phase stability and tin chemical potential",
        "+1.00 liquid_fraction +0.20 mu_sn",
    ),
    "D3": ObjectiveDefinition(
        (0.0, 1.0, -0.25, 0.0),
        "High discharge energy with low temperature rise",
        "+1.00 discharge_energy -0.25 peak_temperature_rise",
    ),
    "D4": ObjectiveDefinition(
        (0.0, 1.0, 0.75, -0.25),
        "High regeneration recovery and carbon-dioxide purity with low breakthrough",
        "+1.00 regeneration_recovery +0.75 desorption_co2_purity -0.25 breakthrough_ratio",
    ),
    "D5": ObjectiveDefinition(
        (1.0, 0.25, 0.0, -0.20),
        "High methanol production and recovery with low utility demand",
        "+1.00 methanol_product +0.25 methanol_recovery -0.20 total_utility",
    ),
    "D6": ObjectiveDefinition(
        (1.0, -0.10, -0.20, -0.10),
        "Large uniform grains with low boundary density",
        "+1.00 mean_grain_diameter -0.10 grain_count -0.20 boundary_density -0.10 grain_size_cv",
    ),
    "D7": ObjectiveDefinition(
        (0.10, 0.05, 1.0),
        "Strength-led composite workability objective",
        "+1.00 compressive_strength_28d +0.10 slump +0.05 flow",
    ),
}


SEQUENTIAL_METHODS = (
    "Random",
    "RBF-GPR-UCB",
    "AD-ADE-mean/base-std-UCB",
    "EG-AD-ADE-GPR-UCB",
)

BATCH_METHODS = (
    "Random batch-3",
    "RBF-GPR top-3",
    "EG-AD-ADE top-3",
    "EG-AD-ADE diverse-3",
)


def parse_names(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(";")]


def initial_and_outer_indices(
    x_train: np.ndarray,
    dataset_number: int,
    rep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select an input-central initial set and an input-outer candidate subset.

    Responses are never used for this split.  The initial observations are
    sampled from the innermost 70% of the original training inputs, while the
    outer 30% joins the pre-existing extrapolation test set as the BO pool.
    """
    scaler = StandardScaler().fit(x_train)
    ad_score = knn_leave_one_out_scores(scaler.transform(x_train), k=5)
    cut = float(np.quantile(ad_score, 0.70))
    core_indices = np.flatnonzero(ad_score <= cut)
    outer_indices = np.flatnonzero(ad_score > cut)
    rng = np.random.default_rng(910_000 + 10_007 * dataset_number + 1_009 * rep)
    count = min(len(core_indices), max(24, int(round(0.55 * len(x_train)))))
    initial = np.sort(rng.choice(core_indices, size=count, replace=False))
    return initial, np.sort(outer_indices)


def objective_transform(
    y_initial: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a training-defined dimensionless desirability.

    The 5% mean-magnitude floor prevents a nearly constant training response
    (notably D4 purity) from receiving an arbitrarily large numerical weight.
    No held-out response is used to define the center or scale.
    """
    center = np.mean(y_initial, axis=0)
    sample_sd = np.std(y_initial, axis=0, ddof=1)
    scale = np.maximum.reduce(
        [sample_sd, 0.05 * np.abs(center), np.full_like(center, 1e-8)]
    )
    utility = ((np.asarray(y_values, dtype=float) - center) / scale) @ weights
    return np.asarray(utility, dtype=float), center, scale


def standardize_objective(initial_utility: np.ndarray, utility: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(np.mean(initial_utility))
    scale = float(np.std(initial_utility, ddof=1))
    scale = max(scale, 1e-8)
    return (np.asarray(utility, dtype=float) - center) / scale, center, scale


def fit_base_gpr(x: np.ndarray, y: np.ndarray, seed: int):
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    factory = AnchorDeltaRegressor("GPR", random_state=seed)
    model = factory._make_base(z.shape[1])
    model.fit(z, y)
    return scaler, model


def base_distribution(scaler: StandardScaler, model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = model.predict(scaler.transform(x), return_std=True)
    return np.asarray(mean, dtype=float).ravel(), np.asarray(std, dtype=float).ravel()


@dataclass
class CalibratedADE:
    model: AnchorDeltaRegressor
    calibration_error_q90: float
    calibration_distance: float
    support_ratio: float
    calibration_rmse: float
    shell_size: int
    exploration_weight: float
    discrepancy_penalty: float


def fit_calibrated_ade(
    x: np.ndarray,
    y: np.ndarray,
    q_kind: str,
    local_weight: float,
    seed: int,
    prefit_model: AnchorDeltaRegressor | None = None,
) -> CalibratedADE:
    """Fit ADE and calibrate discrepancy/support on a training-only shell."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    ad_scaler = StandardScaler().fit(x)
    ad_score = knn_leave_one_out_scores(ad_scaler.transform(x), k=5)
    cut = float(np.quantile(ad_score, 0.70))
    core = ad_score <= cut
    if int(np.sum(core)) < max(18, x.shape[1] + 6):
        order = np.argsort(ad_score)
        core = np.zeros(len(x), dtype=bool)
        core[order[: max(18, x.shape[1] + 6)]] = True
    shell = ~core
    if int(np.sum(shell)) < 4:
        order = np.argsort(ad_score)
        shell = np.zeros(len(x), dtype=bool)
        shell[order[-4:]] = True
        core = ~shell

    inner = AnchorDeltaRegressor(
        "GPR",
        q_kind=q_kind,
        local_weight=local_weight,
        random_state=seed + 101,
    ).fit(x[core], y[core])
    shell_prediction = inner.predict(x[shell])
    shell_error = np.abs(y[shell] - shell_prediction)
    calibration_error_q90 = float(np.quantile(shell_error, 0.90))
    calibration_rmse = float(np.sqrt(np.mean((y[shell] - shell_prediction) ** 2)))

    shell_ratio, _, shell_distance, _ = inner.extrapolation_geometry(x[shell])
    positive_distance = shell_distance[shell_distance > 1e-8]
    calibration_distance = float(
        np.median(positive_distance) if len(positive_distance) else max(0.10, 0.10 * inner.a0_)
    )

    # The 1.50 expansion is fixed before seeing held-out responses.  The lower
    # bound permits one modest frontier step even for compact training shells;
    # the upper bound forbids an unsupported jump to arbitrarily remote points.
    support_ratio = float(
        np.clip(max(2.00, 1.50 * np.quantile(shell_ratio, 0.95)), 2.00, 2.50)
    )

    shell_learn_mean, shell_learn_std = inner.predict_distribution(
        x[shell], calibration_error_q90=0.0, calibration_distance=calibration_distance
    )
    _, shell_total_std = inner.predict_distribution(
        x[shell],
        calibration_error_q90=max(calibration_error_q90, 1e-6),
        calibration_distance=calibration_distance,
    )
    shell_discrepancy = np.sqrt(np.maximum(shell_total_std**2 - shell_learn_std**2, 0.0))
    shell_risk = 0.05 * np.maximum(shell_ratio - 1.0, 0.0) ** 2
    top_count = max(2, int(math.ceil(0.25 * len(shell_ratio))))
    policy_choices: list[tuple[float, float, float]] = []
    for exploration_weight in (0.25, 0.50, 1.00):
        for discrepancy_penalty in (0.25, 0.50, 1.00):
            shell_acquisition = (
                shell_learn_mean
                + exploration_weight * shell_learn_std
                - discrepancy_penalty * shell_discrepancy
                - shell_risk
            )
            order = np.argsort(shell_acquisition)[::-1][:top_count]
            # Select the policy whose top acquisition quartile has the highest
            # observed shell utility.  Ties favor lower exploration and higher
            # discrepancy penalty through the secondary sort keys below.
            validation_value = float(np.mean(y[shell][order]))
            policy_choices.append(
                (validation_value, -exploration_weight, discrepancy_penalty)
            )
    policy_choices.sort(reverse=True)
    _, negative_exploration, discrepancy_penalty = policy_choices[0]
    exploration_weight = -negative_exploration
    model = prefit_model
    if model is None:
        model = AnchorDeltaRegressor(
            "GPR",
            q_kind=q_kind,
            local_weight=local_weight,
            random_state=seed,
        ).fit(x, y)
    return CalibratedADE(
        model=model,
        calibration_error_q90=max(calibration_error_q90, 1e-6),
        calibration_distance=max(calibration_distance, 1e-6),
        support_ratio=support_ratio,
        calibration_rmse=calibration_rmse,
        shell_size=int(np.sum(shell)),
        exploration_weight=float(exploration_weight),
        discrepancy_penalty=float(discrepancy_penalty),
    )


def robust_distance_threshold(z_observed: np.ndarray) -> float:
    if len(z_observed) < 3:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=2).fit(z_observed)
    distances, _ = nbrs.kneighbors(z_observed)
    positive = distances[:, 1][distances[:, 1] > 1e-10]
    if not len(positive):
        return 0.0
    return float(np.quantile(positive, 0.25))


def acquisition_state(
    method: str,
    x_observed: np.ndarray,
    y_observed: np.ndarray,
    x_candidates: np.ndarray,
    q_kind: str,
    local_weight: float,
    seed: int,
    cycle: int,
) -> dict[str, object]:
    kappa = 1.50 + 0.05 * math.sqrt(math.log1p(cycle + 1))
    if method.startswith("Random"):
        scaler = StandardScaler().fit(x_observed)
        z_observed = scaler.transform(x_observed)
        z_candidates = scaler.transform(x_candidates)
        ad = KNNApplicabilityDomain(k=5, boundary_quantile=0.90).fit(z_observed)
        rho = ad.ratio(z_candidates)
        return {
            "mean": np.full(len(x_candidates), np.nan),
            "std": np.full(len(x_candidates), np.nan),
            "learnable_std": np.full(len(x_candidates), np.nan),
            "discrepancy_std": np.full(len(x_candidates), np.nan),
            "acquisition": np.zeros(len(x_candidates)),
            "eligible": np.ones(len(x_candidates), dtype=bool),
            "rho": rho,
            "support_ratio": np.nan,
            "scaler": scaler,
            "z_observed": z_observed,
            "z_candidates": z_candidates,
            "calibration_rmse": np.nan,
            "exploration_weight": np.nan,
            "discrepancy_penalty": np.nan,
        }

    if method.startswith("RBF-GPR"):
        scaler, model = fit_base_gpr(x_observed, y_observed, seed + cycle)
        mean, std = base_distribution(scaler, model, x_candidates)
        z_observed = scaler.transform(x_observed)
        z_candidates = scaler.transform(x_candidates)
        ad = KNNApplicabilityDomain(k=5, boundary_quantile=0.90).fit(z_observed)
        rho = ad.ratio(z_candidates)
        return {
            "mean": mean,
            "std": std,
            "learnable_std": std,
            "discrepancy_std": np.zeros(len(x_candidates)),
            "acquisition": mean + kappa * std,
            "eligible": np.ones(len(x_candidates), dtype=bool),
            "rho": rho,
            "support_ratio": np.nan,
            "scaler": scaler,
            "z_observed": z_observed,
            "z_candidates": z_candidates,
            "calibration_rmse": np.nan,
            "exploration_weight": kappa,
            "discrepancy_penalty": 0.0,
        }

    if method == "AD-ADE-mean/base-std-UCB":
        model = AnchorDeltaRegressor(
            "GPR",
            q_kind=q_kind,
            local_weight=local_weight,
            random_state=seed + cycle,
        ).fit(x_observed, y_observed)
        mean = model.predict(x_candidates)
        _, std = base_distribution(model.x_scaler_, model.base_, x_candidates)
        rho = model.extrapolation_ratio(x_candidates)
        z_observed = model.x_scaler_.transform(x_observed)
        return {
            "mean": mean,
            "std": std,
            "learnable_std": std,
            "discrepancy_std": np.zeros(len(x_candidates)),
            "acquisition": mean + kappa * std,
            "eligible": np.ones(len(x_candidates), dtype=bool),
            "rho": rho,
            "support_ratio": np.nan,
            "scaler": model.x_scaler_,
            "z_observed": z_observed,
            "z_candidates": model.x_scaler_.transform(x_candidates),
            "calibration_rmse": np.nan,
            "exploration_weight": kappa,
            "discrepancy_penalty": 0.0,
        }

    if method in {"EG-AD-ADE-GPR-UCB", "EG-AD-ADE top-3", "EG-AD-ADE diverse-3"}:
        calibrated = fit_calibrated_ade(
            x_observed,
            y_observed,
            q_kind,
            local_weight,
            seed + cycle,
        )
        mean, std = calibrated.model.predict_distribution(
            x_candidates,
            calibration_error_q90=calibrated.calibration_error_q90,
            calibration_distance=calibrated.calibration_distance,
        )
        _, learnable_std = calibrated.model.predict_distribution(
            x_candidates,
            calibration_error_q90=0.0,
            calibration_distance=calibrated.calibration_distance,
        )
        discrepancy_std = np.sqrt(np.maximum(std**2 - learnable_std**2, 0.0))
        rho = calibrated.model.extrapolation_ratio(x_candidates)
        eligible = rho <= calibrated.support_ratio + 1e-12
        # Learnable GP uncertainty supports exploration; shell-calibrated
        # extrapolation discrepancy is model risk and is therefore penalized.
        risk = (
            calibrated.discrepancy_penalty * discrepancy_std
            + 0.05 * np.maximum(rho - 1.0, 0.0) ** 2
        )
        acquisition = mean + calibrated.exploration_weight * learnable_std - risk
        acquisition = np.where(eligible, acquisition, -np.inf)
        return {
            "mean": mean,
            "std": std,
            "learnable_std": learnable_std,
            "discrepancy_std": discrepancy_std,
            "acquisition": acquisition,
            "eligible": eligible,
            "rho": rho,
            "support_ratio": calibrated.support_ratio,
            "scaler": calibrated.model.x_scaler_,
            "z_observed": calibrated.model.x_scaler_.transform(x_observed),
            "z_candidates": calibrated.model.x_scaler_.transform(x_candidates),
            "calibration_rmse": calibrated.calibration_rmse,
            "exploration_weight": calibrated.exploration_weight,
            "discrepancy_penalty": calibrated.discrepancy_penalty,
        }
    raise ValueError(method)


def choose_batch(
    method: str,
    state: dict[str, object],
    batch_size: int,
    rng: np.random.Generator,
) -> list[int]:
    eligible = np.asarray(state["eligible"], dtype=bool)
    available = np.flatnonzero(eligible)
    if not len(available):
        return []
    take = min(batch_size, len(available))
    if method.startswith("Random"):
        return [int(v) for v in rng.choice(available, size=take, replace=False)]

    acquisition = np.asarray(state["acquisition"], dtype=float)
    if method != "EG-AD-ADE diverse-3":
        order = available[np.argsort(acquisition[available])[::-1]]
        return [int(v) for v in order[:take]]

    first = int(available[np.argmax(acquisition[available])])
    selected = [first]
    if take == 1:
        return selected

    finite_acq = acquisition[available]
    quality_cut = float(np.quantile(finite_acq[np.isfinite(finite_acq)], 0.60))
    quality = available[acquisition[available] >= quality_cut]
    z_candidates = np.asarray(state["z_candidates"], dtype=float)
    z_observed = np.asarray(state["z_observed"], dtype=float)
    minimum_distance = robust_distance_threshold(z_observed)

    while len(selected) < take:
        candidates = [int(v) for v in quality if int(v) not in selected]
        if not candidates:
            break
        prior = np.vstack([z_observed, z_candidates[selected]])
        distances = np.asarray(
            [np.min(np.linalg.norm(prior - z_candidates[v], axis=1)) for v in candidates],
            dtype=float,
        )
        feasible = distances >= minimum_distance - 1e-12
        if not np.any(feasible):
            break
        feasible_candidates = np.asarray(candidates, dtype=int)[feasible]
        feasible_distances = distances[feasible]
        # Maximin is primary; acquisition breaks numerical ties.
        best_distance = float(np.max(feasible_distances))
        tied = feasible_candidates[np.isclose(feasible_distances, best_distance, rtol=1e-10, atol=1e-12)]
        chosen = int(tied[np.argmax(acquisition[tied])])
        selected.append(chosen)
    return selected


def run_method(
    dataset: str,
    dataset_label: str,
    rep: int,
    regime: str,
    method: str,
    x_initial: np.ndarray,
    utility_initial: np.ndarray,
    x_pool: np.ndarray,
    utility_pool: np.ndarray,
    q_kind: str,
    local_weight: float,
    budget: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model_initial, model_center, model_scale = standardize_objective(
        utility_initial, utility_initial
    )
    x_observed = np.asarray(x_initial, dtype=float).copy()
    y_observed = np.asarray(model_initial, dtype=float).copy()
    remaining = np.arange(len(x_pool), dtype=int)
    selected_global: list[int] = []
    rows: list[dict[str, object]] = []
    cycle = 0
    best_utility = -np.inf
    pool_min = float(np.min(utility_pool))
    pool_max = float(np.max(utility_pool))
    pool_range = max(pool_max - pool_min, 1e-12)
    top10_cut = float(np.quantile(utility_pool, 0.90))
    stop_reason = "budget_reached"
    rng = np.random.default_rng(seed + 97 * rep + sum(ord(c) for c in method))

    while len(selected_global) < budget and len(remaining):
        x_candidates = x_pool[remaining]
        state = acquisition_state(
            method,
            x_observed,
            y_observed,
            x_candidates,
            q_kind,
            local_weight,
            seed,
            cycle,
        )
        current_batch = min(batch_size, budget - len(selected_global), len(remaining))
        local_selected = choose_batch(method, state, current_batch, rng)
        if not local_selected:
            stop_reason = "unsupported_frontier"
            break

        z_candidates = np.asarray(state["z_candidates"], dtype=float)
        z_observed = np.asarray(state["z_observed"], dtype=float)
        batch_prior = z_observed.copy()
        batch_global_indices: list[int] = []
        for batch_rank, local_index in enumerate(local_selected, start=1):
            global_index = int(remaining[local_index])
            batch_global_indices.append(global_index)
            candidate_z = z_candidates[local_index]
            distance_to_prior = float(np.min(np.linalg.norm(batch_prior - candidate_z, axis=1)))
            batch_prior = np.vstack([batch_prior, candidate_z])
            selected_global.append(global_index)
            value = float(utility_pool[global_index])
            best_utility = max(best_utility, value)
            quality = float(np.clip((best_utility - pool_min) / pool_range, 0.0, 1.0))
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label,
                    "rep": rep,
                    "regime": regime,
                    "method": method,
                    "evaluation": len(selected_global),
                    "cycle": cycle + 1,
                    "rank_in_batch": batch_rank,
                    "candidate_pool_index": global_index,
                    "utility": value,
                    "best_utility": best_utility,
                    "quality": quality,
                    "normalized_regret": 1.0 - quality,
                    "top10_hit": float(value >= top10_cut),
                    "predicted_mean": float(np.asarray(state["mean"])[local_index]),
                    "predicted_std": float(np.asarray(state["std"])[local_index]),
                    "learnable_std": float(np.asarray(state["learnable_std"])[local_index]),
                    "discrepancy_std": float(np.asarray(state["discrepancy_std"])[local_index]),
                    "acquisition": float(np.asarray(state["acquisition"])[local_index]),
                    "rho": float(np.asarray(state["rho"])[local_index]),
                    "support_ratio": float(state["support_ratio"]),
                    "supported_when_selected": float(np.asarray(state["eligible"])[local_index]),
                    "distance_to_prior": distance_to_prior,
                    "calibration_rmse": float(state["calibration_rmse"]),
                    "exploration_weight": float(state["exploration_weight"]),
                    "discrepancy_penalty": float(state["discrepancy_penalty"]),
                }
            )

        x_new = x_pool[batch_global_indices]
        utility_new = utility_pool[batch_global_indices]
        y_new = (utility_new - model_center) / model_scale
        x_observed = np.vstack([x_observed, x_new])
        y_observed = np.concatenate([y_observed, y_new])
        keep = np.ones(len(remaining), dtype=bool)
        keep[np.asarray(local_selected, dtype=int)] = False
        remaining = remaining[keep]
        cycle += 1
        if cycle > budget + 2:
            stop_reason = "cycle_guard"
            break

    status = {
        "dataset": dataset,
        "dataset_label": dataset_label,
        "rep": rep,
        "regime": regime,
        "method": method,
        "selected_count": len(selected_global),
        "cycle_count": cycle,
        "stop_reason": stop_reason,
        "final_quality": float(rows[-1]["quality"]) if rows else 0.0,
        "top10_hit": float(any(row["top10_hit"] > 0.5 for row in rows)),
    }
    return rows, status


def initial_uncertainty_evaluation(
    dataset: str,
    dataset_label: str,
    rep: int,
    x_initial: np.ndarray,
    utility_initial: np.ndarray,
    x_pool: np.ndarray,
    utility_pool: np.ndarray,
    tuned_model: AnchorDeltaRegressor,
    q_kind: str,
    local_weight: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y_initial, center, scale = standardize_objective(utility_initial, utility_initial)
    y_pool = (utility_pool - center) / scale
    rows: list[dict[str, object]] = []

    base_scaler, base_model = fit_base_gpr(x_initial, y_initial, seed)
    base_mean, base_std = base_distribution(base_scaler, base_model, x_pool)
    methods: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("RBF-GPR", base_mean, base_std, np.ones(len(x_pool), dtype=bool))
    ]

    ade_mean = tuned_model.predict(x_pool)
    _, ade_base_std = base_distribution(tuned_model.x_scaler_, tuned_model.base_, x_pool)
    methods.append(
        ("AD-ADE mean + base std", ade_mean, ade_base_std, np.ones(len(x_pool), dtype=bool))
    )

    calibrated = fit_calibrated_ade(
        x_initial,
        y_initial,
        q_kind,
        local_weight,
        seed + 211,
        prefit_model=tuned_model,
    )
    proposed_mean, proposed_std = calibrated.model.predict_distribution(
        x_pool,
        calibration_error_q90=calibrated.calibration_error_q90,
        calibration_distance=calibrated.calibration_distance,
    )
    proposed_rho = calibrated.model.extrapolation_ratio(x_pool)
    proposed_support = proposed_rho <= calibrated.support_ratio + 1e-12
    methods.append(("EG-AD-ADE uncertainty", proposed_mean, proposed_std, proposed_support))

    for method, mean, std, support in methods:
        half_width = 1.645 * np.maximum(std, 1e-12)
        lower = mean - half_width
        upper = mean + half_width
        interval_hit = (y_pool >= lower) & (y_pool <= upper)
        # Proper 90% interval score: narrower calibrated intervals score lower.
        alpha = 0.10
        interval_score = (
            upper
            - lower
            + (2.0 / alpha) * (lower - y_pool) * (y_pool < lower)
            + (2.0 / alpha) * (y_pool - upper) * (y_pool > upper)
        )
        for candidate_index in range(len(x_pool)):
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label,
                    "rep": rep,
                    "method": method,
                    "candidate_pool_index": candidate_index,
                    "supported": float(support[candidate_index]),
                    "actual": float(y_pool[candidate_index]),
                    "predicted_mean": float(mean[candidate_index]),
                    "predicted_std": float(std[candidate_index]),
                    "absolute_error": float(abs(y_pool[candidate_index] - mean[candidate_index])),
                    "interval_lower_90": float(lower[candidate_index]),
                    "interval_upper_90": float(upper[candidate_index]),
                    "interval_width_90": float(upper[candidate_index] - lower[candidate_index]),
                    "covered_90": float(interval_hit[candidate_index]),
                    "interval_score_90": float(interval_score[candidate_index]),
                    "support_ratio": float(calibrated.support_ratio)
                    if method == "EG-AD-ADE uncertainty"
                    else np.nan,
                }
            )

    rng = np.random.default_rng(seed + 701)
    raw_directions = rng.normal(size=(256, x_initial.shape[1]))
    raw_directions /= np.maximum(np.linalg.norm(raw_directions, axis=1, keepdims=True), 1e-12)
    selected_direction_indices = [0]
    while len(selected_direction_indices) < 32:
        selected_directions = raw_directions[selected_direction_indices]
        distances = np.min(
            np.linalg.norm(
                raw_directions[:, None, :] - selected_directions[None, :, :], axis=2
            ),
            axis=1,
        )
        distances[selected_direction_indices] = -np.inf
        selected_direction_indices.append(int(np.argmax(distances)))
    directions = raw_directions[selected_direction_indices]

    variation_rows: list[dict[str, object]] = []
    for rho_value in (1.5, 4.0, 32.0):
        base_z_initial = base_scaler.transform(x_initial)
        base_ad = KNNApplicabilityDomain(k=5, boundary_quantile=0.90).fit(base_z_initial)
        base_shell = base_scaler.inverse_transform(base_ad.points_at_ratio(directions, rho_value))
        shell_mean, shell_std = base_distribution(base_scaler, base_model, base_shell)
        shell_acquisition = shell_mean + 1.5 * shell_std
        variation_rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label,
                "rep": rep,
                "method": "RBF-GPR-UCB",
                "rho": rho_value,
                "support_rate": 1.0,
                "acquisition_directional_sd": float(np.std(shell_acquisition, ddof=1)),
                "mean_directional_sd": float(np.std(shell_mean, ddof=1)),
            }
        )

        ade_shell = tuned_model.points_at_ad_ratio(directions, rho_value)
        ade_shell_mean = tuned_model.predict(ade_shell)
        _, ade_shell_base_std = base_distribution(
            tuned_model.x_scaler_, tuned_model.base_, ade_shell
        )
        ade_shell_acquisition = ade_shell_mean + 1.5 * ade_shell_base_std
        variation_rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label,
                "rep": rep,
                "method": "AD-ADE mean + base std UCB",
                "rho": rho_value,
                "support_rate": 1.0,
                "acquisition_directional_sd": float(
                    np.std(ade_shell_acquisition, ddof=1)
                ),
                "mean_directional_sd": float(np.std(ade_shell_mean, ddof=1)),
            }
        )

        proposed_shell = calibrated.model.points_at_ad_ratio(directions, rho_value)
        proposed_mean, proposed_total_std = calibrated.model.predict_distribution(
            proposed_shell,
            calibration_error_q90=calibrated.calibration_error_q90,
            calibration_distance=calibrated.calibration_distance,
        )
        _, proposed_learn_std = calibrated.model.predict_distribution(
            proposed_shell,
            calibration_error_q90=0.0,
            calibration_distance=calibrated.calibration_distance,
        )
        proposed_discrepancy = np.sqrt(
            np.maximum(proposed_total_std**2 - proposed_learn_std**2, 0.0)
        )
        proposed_support = rho_value <= calibrated.support_ratio + 1e-12
        proposed_acquisition = (
            proposed_mean
            + calibrated.exploration_weight * proposed_learn_std
            - calibrated.discrepancy_penalty * proposed_discrepancy
            - 0.05 * max(rho_value - 1.0, 0.0) ** 2
        )
        variation_rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label,
                "rep": rep,
                "method": "EG-AD-ADE risk-adjusted UCB",
                "rho": rho_value,
                "support_rate": float(proposed_support),
                "acquisition_directional_sd": float(
                    np.std(proposed_acquisition, ddof=1)
                )
                if proposed_support
                else np.nan,
                "mean_directional_sd": float(np.std(proposed_mean, ddof=1))
                if proposed_support
                else np.nan,
            }
        )
    return rows, variation_rows


def complete_curves(
    raw: pd.DataFrame,
    statuses: pd.DataFrame,
    budget: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for _, status in statuses.iterrows():
        mask = (
            (raw["dataset"] == status["dataset"])
            & (raw["rep"] == status["rep"])
            & (raw["regime"] == status["regime"])
            & (raw["method"] == status["method"])
        )
        run = raw[mask].sort_values("evaluation")
        quality_lookup = dict(zip(run["evaluation"].astype(int), run["quality"].astype(float)))
        last = 0.0
        first_top10 = budget + 1
        for evaluation in range(1, budget + 1):
            if evaluation in quality_lookup:
                last = quality_lookup[evaluation]
            if first_top10 == budget + 1 and len(run):
                hits = run[(run["evaluation"] <= evaluation) & (run["top10_hit"] > 0.5)]
                if len(hits):
                    first_top10 = int(hits["evaluation"].min())
            curve_rows.append(
                {
                    "dataset": status["dataset"],
                    "dataset_label": status["dataset_label"],
                    "rep": int(status["rep"]),
                    "regime": status["regime"],
                    "method": status["method"],
                    "evaluation": evaluation,
                    "quality": last,
                }
            )
        run_distances = run["distance_to_prior"].to_numpy(dtype=float) if len(run) else np.asarray([])
        summary_rows.append(
            {
                **status.to_dict(),
                "final_quality": last,
                "normalized_simple_regret": 1.0 - last,
                "auc_quality": float(
                    np.mean(
                        [
                            row["quality"]
                            for row in curve_rows[-budget:]
                        ]
                    )
                ),
                "steps_to_top10": first_top10,
                "mean_selected_rho": float(run["rho"].mean()) if len(run) else np.nan,
                "deep_extrapolation_rate": float(np.mean(run["rho"] > 2.0)) if len(run) else np.nan,
                "mean_distance_to_prior": float(np.mean(run_distances)) if len(run_distances) else np.nan,
                "minimum_distance_to_prior": float(np.min(run_distances)) if len(run_distances) else np.nan,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def summarize_results(
    curve_raw: pd.DataFrame,
    run_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_summary = curve_raw.groupby(
        ["regime", "method", "evaluation"], as_index=False
    ).agg(
        quality_median=("quality", "median"),
        quality_q25=("quality", lambda s: s.quantile(0.25)),
        quality_q75=("quality", lambda s: s.quantile(0.75)),
    )
    dataset_summary = run_summary.groupby(
        ["dataset", "dataset_label", "regime", "method"], as_index=False
    ).agg(
        final_quality_median=("final_quality", "median"),
        final_quality_q25=("final_quality", lambda s: s.quantile(0.25)),
        final_quality_q75=("final_quality", lambda s: s.quantile(0.75)),
        auc_quality_median=("auc_quality", "median"),
        top10_hit_rate=("top10_hit", "mean"),
        steps_to_top10_median=("steps_to_top10", "median"),
        stop_rate=("selected_count", lambda s: float(np.mean(s < s.max())) if len(s) else np.nan),
        selected_count_mean=("selected_count", "mean"),
        mean_selected_rho=("mean_selected_rho", "mean"),
        deep_extrapolation_rate=("deep_extrapolation_rate", "mean"),
        mean_distance_to_prior=("mean_distance_to_prior", "mean"),
        minimum_distance_to_prior=("minimum_distance_to_prior", "median"),
    )
    return curve_summary, dataset_summary


def summarize_uncertainty(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, label, rep, method), group in raw.groupby(
        ["dataset", "dataset_label", "rep", "method"], sort=False
    ):
        supported = group[group["supported"] > 0.5]
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": label,
                "rep": rep,
                "method": method,
                "support_rate": float(group["supported"].mean()),
                "coverage_90_supported": float(supported["covered_90"].mean())
                if len(supported)
                else np.nan,
                "interval_width_90_supported": float(supported["interval_width_90"].mean())
                if len(supported)
                else np.nan,
                "interval_score_90_supported": float(supported["interval_score_90"].mean())
                if len(supported)
                else np.nan,
                "rmse_supported": float(
                    np.sqrt(np.mean((supported["actual"] - supported["predicted_mean"]) ** 2))
                )
                if len(supported)
                else np.nan,
            }
        )
    rep_summary = pd.DataFrame(rows)
    return rep_summary.groupby(["dataset", "dataset_label", "method"], as_index=False).agg(
        support_rate=("support_rate", "mean"),
        coverage_90_supported_median=("coverage_90_supported", "median"),
        interval_width_90_supported_median=("interval_width_90_supported", "median"),
        interval_score_90_supported_median=("interval_score_90_supported", "median"),
        rmse_supported_median=("rmse_supported", "median"),
    )


def pairwise_summary(run_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return dataset-clustered inference and repetition-level descriptives.

    Repetitions from the same engineering system share the external candidate
    pool and are not independent experimental units.  Inference is therefore
    performed on one median paired difference per dataset. Repetition-level
    pairs are retained only as descriptive counts.
    """
    comparisons = [
        ("sequential", "EG-AD-ADE-GPR-UCB", "RBF-GPR-UCB"),
        ("sequential", "EG-AD-ADE-GPR-UCB", "AD-ADE-mean/base-std-UCB"),
        ("batch", "EG-AD-ADE diverse-3", "RBF-GPR top-3"),
        ("batch", "EG-AD-ADE diverse-3", "EG-AD-ADE top-3"),
    ]
    rows: list[dict[str, object]] = []
    repetition_rows: list[dict[str, object]] = []
    for regime, proposed, baseline in comparisons:
        subset = run_summary[run_summary["regime"] == regime]
        pivot = subset.pivot_table(
            index=["dataset", "rep"], columns="method", values="final_quality", aggfunc="first"
        ).dropna(subset=[proposed, baseline])
        repetition_difference = pivot[proposed] - pivot[baseline]
        dataset_difference = repetition_difference.groupby(level="dataset").median()
        statistic, p_value = exact_wilcoxon_signed_rank(
            dataset_difference.to_numpy(dtype=float)
        )
        rng = np.random.default_rng(20260902 + len(rows) * 1009)
        if len(dataset_difference):
            bootstrap = np.median(
                rng.choice(
                    dataset_difference.to_numpy(dtype=float),
                    size=(100_000, len(dataset_difference)),
                    replace=True,
                ),
                axis=1,
            )
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
        else:
            ci_low, ci_high = np.nan, np.nan
        rows.append(
            {
                "regime": regime,
                "proposed": proposed,
                "baseline": baseline,
                "independent_unit": "dataset",
                "dataset_count": len(dataset_difference),
                "repetition_pair_count": len(repetition_difference),
                "dataset_win_count": int(np.sum(dataset_difference > 1e-12)),
                "dataset_tie_count": int(np.sum(np.abs(dataset_difference) <= 1e-12)),
                "dataset_loss_count": int(np.sum(dataset_difference < -1e-12)),
                "median_dataset_difference": float(np.median(dataset_difference))
                if len(dataset_difference)
                else np.nan,
                "cluster_bootstrap_ci_low": float(ci_low),
                "cluster_bootstrap_ci_high": float(ci_high),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
            }
        )
        repetition_rows.append(
            {
                "regime": regime,
                "proposed": proposed,
                "baseline": baseline,
                "pair_count": len(repetition_difference),
                "win_count": int(np.sum(repetition_difference > 1e-12)),
                "tie_count": int(np.sum(np.abs(repetition_difference) <= 1e-12)),
                "loss_count": int(np.sum(repetition_difference < -1e-12)),
                "median_quality_difference": float(np.median(repetition_difference))
                if len(repetition_difference)
                else np.nan,
                "interpretation": "descriptive only; repetitions within a dataset are clustered",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(repetition_rows)


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, alpha=0.55, linewidth=0.7)
    ax.tick_params(labelsize=8)


def make_convergence_figure(curve_summary: pd.DataFrame, output: Path) -> None:
    styles = {
        "Random": ("#98A2B3", ":"),
        "RBF-GPR-UCB": (GRAY, "--"),
        "AD-ADE-mean/base-std-UCB": (ORANGE, "-."),
        "EG-AD-ADE-GPR-UCB": (BLUE, "-"),
        "Random batch-3": ("#98A2B3", ":"),
        "RBF-GPR top-3": (GRAY, "--"),
        "EG-AD-ADE top-3": (TEAL, "-."),
        "EG-AD-ADE diverse-3": (BLUE, "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True, sharey=True)
    for ax, regime, title in zip(
        axes,
        ("sequential", "batch"),
        ("(a) Sequential Bayesian optimization", "(b) Three-point batch search"),
    ):
        methods = SEQUENTIAL_METHODS if regime == "sequential" else BATCH_METHODS
        for method in methods:
            data = curve_summary[
                (curve_summary["regime"] == regime) & (curve_summary["method"] == method)
            ].sort_values("evaluation")
            if not len(data):
                continue
            color, linestyle = styles[method]
            x = data["evaluation"].to_numpy(dtype=float)
            median = data["quality_median"].to_numpy(dtype=float)
            q25 = data["quality_q25"].to_numpy(dtype=float)
            q75 = data["quality_q75"].to_numpy(dtype=float)
            ax.plot(x, median, color=color, linestyle=linestyle, linewidth=2.0, label=method)
            ax.fill_between(x, q25, q75, color=color, alpha=0.12, linewidth=0)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Cumulative outer-candidate evaluations", fontsize=9)
        ax.set_ylim(0, 1.04)
        style_axis(ax)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    axes[0].set_ylabel("Best-attainment score in candidate pool", fontsize=9)
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_final_quality_figure(dataset_summary: pd.DataFrame, output: Path) -> None:
    data = dataset_summary[dataset_summary["regime"] == "sequential"]
    datasets = [f"D{i}" for i in range(1, 8)]
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.8), constrained_layout=True, sharey=True)
    colors = ["#98A2B3", GRAY, ORANGE, BLUE]
    labels = ["Random", "RBF", "AD-ADE mean", "EG-AD-ADE"]
    for index, dataset in enumerate(datasets):
        ax = axes.flat[index]
        values = []
        low = []
        high = []
        for method in SEQUENTIAL_METHODS:
            row = data[(data["dataset"] == dataset) & (data["method"] == method)].iloc[0]
            values.append(float(row["final_quality_median"]))
            low.append(float(row["final_quality_q25"]))
            high.append(float(row["final_quality_q75"]))
        pos = np.arange(len(values))
        ax.bar(pos, values, color=colors, width=0.72)
        ax.errorbar(
            pos,
            values,
            yerr=np.vstack([np.asarray(values) - low, np.asarray(high) - values]),
            fmt="none",
            ecolor="#344054",
            linewidth=0.7,
            capsize=2,
        )
        ax.set_xticks(pos, labels, rotation=28, ha="right", fontsize=7)
        ax.set_title(dataset, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        style_axis(ax)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("Final attainment score", fontsize=9)
    axes[1, 0].set_ylabel("Final attainment score", fontsize=9)
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_uncertainty_batch_figure(
    uncertainty_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), constrained_layout=True)
    axes = axes.ravel()
    # D5 is displayed in dataset-specific outputs but excluded from pooled
    # summaries because simulator failures were removed non-uniformly.
    uncertainty_primary = uncertainty_summary[uncertainty_summary["dataset"] != "D5"]
    uncertainty_methods = ["RBF-GPR", "AD-ADE mean + base std", "EG-AD-ADE uncertainty"]
    uncertainty_labels = ["RBF", "AD-ADE mean + base SD", "EG-AD-ADE"]
    coverage = [
        float(
            uncertainty_primary[uncertainty_primary["method"] == method][
                "coverage_90_supported_median"
            ].median()
        )
        for method in uncertainty_methods
    ]
    support = [
        float(uncertainty_primary[uncertainty_primary["method"] == method]["support_rate"].mean())
        for method in uncertainty_methods
    ]
    axes[0].bar(np.arange(3), coverage, color=[GRAY, ORANGE, BLUE])
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(3), uncertainty_labels, rotation=22, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) 90% interval coverage", loc="left", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Supported outer candidates", fontsize=8)
    style_axis(axes[0])

    interval_width = [
        float(
            uncertainty_primary[uncertainty_primary["method"] == method][
                "interval_width_90_supported_median"
            ].median()
        )
        for method in uncertainty_methods
    ]
    axes[1].bar(np.arange(3), interval_width, color=[GRAY, ORANGE, BLUE])
    axes[1].set_xticks(np.arange(3), uncertainty_labels, rotation=22, ha="right")
    axes[1].set_title("(b) 90% interval width", loc="left", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Standardized objective units", fontsize=8)
    style_axis(axes[1])

    axes[2].bar(np.arange(3), support, color=[GRAY, ORANGE, BLUE])
    axes[2].set_xticks(np.arange(3), uncertainty_labels, rotation=22, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("(c) Initial support rate", loc="left", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Fraction of outer candidates", fontsize=8)
    style_axis(axes[2])

    batch = dataset_summary[
        (dataset_summary["regime"] == "batch") & (dataset_summary["dataset"] != "D5")
    ]
    batch_methods = ["RBF-GPR top-3", "EG-AD-ADE top-3", "EG-AD-ADE diverse-3"]
    labels = ["RBF top-3", "EG-AD-ADE top-3", "EG-AD-ADE diverse"]
    distances = [
        float(batch[batch["method"] == method]["mean_distance_to_prior"].median())
        for method in batch_methods
    ]
    axes[3].bar(np.arange(3), distances, color=[GRAY, TEAL, BLUE])
    axes[3].set_xticks(np.arange(3), labels, rotation=22, ha="right")
    axes[3].set_title("(d) Batch-candidate separation", loc="left", fontsize=10, fontweight="bold")
    axes[3].set_ylabel("Dataset mean minimum distance", fontsize=8)
    style_axis(axes[3])
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["bayesian_optimization"] = {
        "script": Path(__file__).name,
        "repetitions": args.reps,
        "external_evaluation_budget": args.budget,
        "batch_size": args.batch_size,
        "acquisition": "risk-adjusted UCB; exploration and discrepancy weights selected on the training-only outer shell",
        "ad_definition": "mean k=5 nearest-neighbor distance in the training-standardized x space; boundary=Q90 of leave-one-out training scores",
        "support_gate": "kNN-AD ratio rho <= clip(max(2.00, 1.50*Q95(training-shell rho)), 2.00, 2.50)",
        "risk_penalty": "lambda*discrepancy_std + 0.05*max(rho-1,0)^2; kappa and lambda selected on training-only outer shell",
        "batch_rule": "first acquisition maximum; subsequent maximin among top-40% acquisition candidates and above robust distance threshold; no fantasy observations",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "outputs": [
            "bo_selection_raw.csv",
            "bo_run_status.csv",
            "bo_curve_raw.csv",
            "bo_curve_summary.csv",
            "bo_run_summary.csv",
            "bo_dataset_summary.csv",
            "bo_uncertainty_raw.csv",
            "bo_uncertainty_summary.csv",
            "bo_acquisition_variation_raw.csv",
            "bo_acquisition_variation_summary.csv",
            "bo_pairwise_summary.csv",
            "bo_pairwise_all_datasets_exploratory.csv",
            "bo_pairwise_repetition_descriptive.csv",
            "bo_model_choices.csv",
            "bo_objective_definitions.csv",
            "fig_bo_convergence.png",
            "fig_bo_final_quality.png",
            "fig_bo_uncertainty_batch.png",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(input_dir / "dataset_manifest.csv")
    selection_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    uncertainty_rows: list[dict[str, object]] = []
    acquisition_variation_rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []
    objective_rows: list[dict[str, object]] = []

    for _, metadata in manifest.iterrows():
        dataset = str(metadata["dataset"])
        dataset_number = int(dataset[1:])
        dataset_label = str(metadata["label"])
        x_names = parse_names(metadata["input_variables"])
        y_names = parse_names(metadata["output_variables"])
        frame = pd.read_csv(input_dir / "datasets" / f"{dataset}_data.csv")
        train = frame[frame["split"] == "train"].reset_index(drop=True)
        pool = frame[frame["split"] == "test"].reset_index(drop=True)
        x_train = train[x_names].to_numpy(dtype=float)
        y_train = train[y_names].to_numpy(dtype=float)
        x_pool = pool[x_names].to_numpy(dtype=float)
        y_pool = pool[y_names].to_numpy(dtype=float)
        definition = OBJECTIVES[dataset]
        weights = np.asarray(definition.weights, dtype=float)
        objective_rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label,
                "objective_label": definition.label_ja,
                "formula": definition.formula_ja,
                "normalization": "initial-training mean; max(SD, 0.05*abs(mean), 1e-8)",
            }
        )

        for rep in range(args.reps):
            indices, outer_train_indices = initial_and_outer_indices(x_train, dataset_number, rep)
            x_initial = x_train[indices]
            y_initial_raw = y_train[indices]
            x_bo_pool = np.vstack([x_train[outer_train_indices], x_pool])
            y_bo_pool = np.vstack([y_train[outer_train_indices], y_pool])
            utility_initial, objective_center, objective_scale = objective_transform(
                y_initial_raw, y_initial_raw, weights
            )
            utility_pool, _, _ = objective_transform(y_initial_raw, y_bo_pool, weights)
            model_y_initial, _, _ = standardize_objective(utility_initial, utility_initial)
            tune_seed = 930_000 + 10_019 * dataset_number + 101 * rep
            tuned_model, choice = fit_tuned_ade(
                x_initial,
                model_y_initial,
                "GPR",
                seed=tune_seed,
            )
            q_kind = str(choice["q_kind"])
            local_weight = float(choice["local_weight"])
            choice_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label,
                    "rep": rep,
                    "initial_count": len(x_initial),
                    "pool_count": len(x_bo_pool),
                    "outer_training_pool_count": len(outer_train_indices),
                    "original_test_pool_count": len(x_pool),
                    "q_kind": q_kind,
                    "local_weight": local_weight,
                    "outer_shell_validation_rmse": float(choice["validation_rmse"]),
                    **{
                        f"objective_center_{name}": float(value)
                        for name, value in zip(y_names, objective_center)
                    },
                    **{
                        f"objective_scale_{name}": float(value)
                        for name, value in zip(y_names, objective_scale)
                    },
                }
            )
            uncertainty_result, variation_result = initial_uncertainty_evaluation(
                dataset,
                dataset_label,
                rep,
                x_initial,
                utility_initial,
                x_bo_pool,
                utility_pool,
                tuned_model,
                q_kind,
                local_weight,
                tune_seed + 500,
            )
            uncertainty_rows.extend(uncertainty_result)
            acquisition_variation_rows.extend(variation_result)

            for method in SEQUENTIAL_METHODS:
                rows, status = run_method(
                    dataset,
                    dataset_label,
                    rep,
                    "sequential",
                    method,
                    x_initial,
                    utility_initial,
                    x_bo_pool,
                    utility_pool,
                    q_kind,
                    local_weight,
                    args.budget,
                    1,
                    tune_seed + 1_000,
                )
                selection_rows.extend(rows)
                status_rows.append(status)

            for method in BATCH_METHODS:
                rows, status = run_method(
                    dataset,
                    dataset_label,
                    rep,
                    "batch",
                    method,
                    x_initial,
                    utility_initial,
                    x_bo_pool,
                    utility_pool,
                    q_kind,
                    local_weight,
                    args.budget,
                    args.batch_size,
                    tune_seed + 2_000,
                )
                selection_rows.extend(rows)
                status_rows.append(status)
            print(f"BO {dataset} rep {rep + 1}/{args.reps}", flush=True)

    raw = pd.DataFrame(selection_rows)
    statuses = pd.DataFrame(status_rows)
    uncertainty_raw = pd.DataFrame(uncertainty_rows)
    acquisition_variation_raw = pd.DataFrame(acquisition_variation_rows)
    choices = pd.DataFrame(choice_rows)
    objectives = pd.DataFrame(objective_rows).drop_duplicates("dataset")
    curve_raw, run_summary = complete_curves(raw, statuses, args.budget)
    curve_summary, dataset_summary = summarize_results(curve_raw, run_summary)
    uncertainty_summary = summarize_uncertainty(uncertainty_raw)
    acquisition_variation_summary = acquisition_variation_raw.groupby(
        ["dataset", "dataset_label", "method", "rho"], as_index=False
    ).agg(
        support_rate=("support_rate", "mean"),
        acquisition_directional_sd_median=("acquisition_directional_sd", "median"),
        mean_directional_sd_median=("mean_directional_sd", "median"),
    )
    if "primary_analysis" in manifest.columns:
        primary_datasets = set(
            manifest.loc[manifest["primary_analysis"].astype(bool), "dataset"].astype(str)
        )
    else:
        primary_datasets = set(manifest["dataset"].astype(str)) - {"D5"}
    paired, paired_repetition = pairwise_summary(
        run_summary[run_summary["dataset"].isin(primary_datasets)]
    )
    paired_all, _ = pairwise_summary(run_summary)

    raw.to_csv(output_dir / "bo_selection_raw.csv", index=False)
    statuses.to_csv(output_dir / "bo_run_status.csv", index=False)
    curve_raw.to_csv(output_dir / "bo_curve_raw.csv", index=False)
    curve_summary.to_csv(output_dir / "bo_curve_summary.csv", index=False)
    run_summary.to_csv(output_dir / "bo_run_summary.csv", index=False)
    dataset_summary.to_csv(output_dir / "bo_dataset_summary.csv", index=False)
    uncertainty_raw.to_csv(output_dir / "bo_uncertainty_raw.csv", index=False)
    uncertainty_summary.to_csv(output_dir / "bo_uncertainty_summary.csv", index=False)
    acquisition_variation_raw.to_csv(
        output_dir / "bo_acquisition_variation_raw.csv", index=False
    )
    acquisition_variation_summary.to_csv(
        output_dir / "bo_acquisition_variation_summary.csv", index=False
    )
    paired.to_csv(output_dir / "bo_pairwise_summary.csv", index=False)
    paired_all.to_csv(
        output_dir / "bo_pairwise_all_datasets_exploratory.csv", index=False
    )
    paired_repetition.to_csv(
        output_dir / "bo_pairwise_repetition_descriptive.csv", index=False
    )
    choices.to_csv(output_dir / "bo_model_choices.csv", index=False)
    objectives.to_csv(output_dir / "bo_objective_definitions.csv", index=False)

    make_convergence_figure(curve_summary, output_dir / "fig_bo_convergence.png")
    make_final_quality_figure(dataset_summary, output_dir / "fig_bo_final_quality.png")
    make_uncertainty_batch_figure(
        uncertainty_summary,
        dataset_summary,
        output_dir / "fig_bo_uncertainty_batch.png",
    )
    update_manifest(output_dir, args)

    # Assertions encode the paper's central safety and reproducibility claims.
    proposed = raw[raw["method"].isin(["EG-AD-ADE-GPR-UCB", "EG-AD-ADE top-3", "EG-AD-ADE diverse-3"])]
    assert len(raw), "No BO selections were produced"
    assert np.all(raw["quality"].between(-1e-12, 1 + 1e-12))
    assert np.all(proposed["supported_when_selected"] > 0.5)
    assert not raw.duplicated(["dataset", "rep", "regime", "method", "candidate_pool_index"]).any()
    assert set(objectives["dataset"]) == {f"D{i}" for i in range(1, 8)}
    print("ALL_BO_ASSERTIONS_PASSED", flush=True)
    print(paired.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--budget", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument(
        "--input-dir",
        default=str(RESULTS),
        help="Directory containing dataset_manifest.csv and the datasets subdirectory.",
    )
    parser.add_argument("--output-dir", default=str(RESULTS))
    args = parser.parse_args()
    if args.reps < 1 or args.budget < 1 or args.batch_size < 1:
        raise ValueError("reps, budget and batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
