#!/usr/bin/env python3
"""Diagnose far-field prediction variation for the seven-system validation.

This is a model-behaviour stress test, separate from the physical-accuracy
test in ``run_physics_validation.py``.  Standardized directions taken from the
held-out outer set are continued to progressively deeper kNN-AD shells.

Forward test
------------
For RBF-GPR/SVR and their AD-guided ADE variants, measure the standard
deviation of predictions across extrapolation directions at a fixed kNN-AD
ratio.
A constant far-field predictor has directional SD -> 0.

Inverse test
------------
For Absolute-GMR and Transition-GMR, place target responses on standardized
output shells and measure the spread of the bounded top-1 input candidates
across target directions.  This explicitly checks -- rather than assumes --
whether an inverse method returns the same x for different external targets.

Only componentwise standardization is used.  PCA, PLS and nonlinear variable
transforms are not used.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler

from physics_simulators import DatasetSpec, ROOT, dataset_specs
from run_numerical_validation import (
    AnchorDeltaRegressor,
    AbsoluteGMR,
    EvidenceGatedCompetitiveGMR,
    TransitionGMR,
    nearest_y_anchors,
)
from run_physics_validation import _candidate_bounds, select_candidate_set, split_arrays, train_indices


SHELL_RATIOS = np.array([1.20, 2.00, 4.00, 8.00, 16.00, 32.00], dtype=float)
BLUE = "#0B5FA5"
ORANGE = "#D97706"
GRAY = "#667085"
LIGHT_GRAY = "#D0D5DD"
TEAL = "#0E7C86"

FONT_PATH = ROOT / "qa_fonts" / "NotoSansCJKjp-Regular.otf"
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def maximin_directions(z: np.ndarray, maximum: int) -> np.ndarray:
    """Select directionally diverse unit vectors from held-out outer points."""
    z = np.asarray(z, dtype=float)
    radius = np.linalg.norm(z, axis=1)
    keep = np.isfinite(radius) & (radius > 1e-10)
    z = z[keep]
    radius = radius[keep]
    if len(z) == 0:
        raise ValueError("No nonzero held-out directions")
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


def rbf_kernel_max(kind: str, shell_z: np.ndarray, train_z: np.ndarray) -> np.ndarray:
    distances2 = np.sum((shell_z[:, None, :] - train_z[None, :, :]) ** 2, axis=2)
    if kind == "GPR":
        similarity = np.exp(-0.5 * distances2 / (0.85**2))
    elif kind == "SVR":
        similarity = np.exp(-(0.55 / train_z.shape[1]) * distances2)
    else:
        raise ValueError(kind)
    return np.max(similarity, axis=1)


def base_limit(model: AnchorDeltaRegressor) -> float:
    if model.kind == "GPR":
        return float(np.asarray(model.base_._y_train_mean).reshape(-1)[0])
    return float(np.asarray(model.base_.intercept_).reshape(-1)[0])


def load_frames(result_dir: Path, specs: list[DatasetSpec]) -> dict[str, pd.DataFrame]:
    return {spec.key: pd.read_csv(result_dir / "datasets" / f"{spec.key}_data.csv") for spec in specs}


def forward_variation(
    specs: list[DatasetSpec],
    frames: dict[str, pd.DataFrame],
    choices: pd.DataFrame,
    reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        x_train_full, y_train_full, x_test, _ = split_arrays(spec, frames[spec.key])
        for rep in range(reps):
            idx = train_indices(len(x_train_full), rep, 10000 + int(spec.key[1:]) * 211)
            x_train = x_train_full[idx]
            y_train = y_train_full[idx]
            x_scaler = StandardScaler().fit(x_train)
            train_z = x_scaler.transform(x_train)
            directions = maximin_directions(x_scaler.transform(x_test), maximum=32)
            for output_index, output_name in enumerate(spec.y_names):
                y_center = float(np.mean(y_train[:, output_index]))
                y_scale = float(np.std(y_train[:, output_index], ddof=1))
                if y_scale < 1e-10:
                    y_scale = 1.0
                y_standard = (y_train[:, output_index] - y_center) / y_scale
                for kind in ("GPR", "SVR"):
                    selected = choices[
                        (choices["dataset"] == spec.key)
                        & (choices["rep"] == rep)
                        & (choices["output"] == output_name)
                        & (choices["kind"] == kind)
                    ]
                    if len(selected) != 1:
                        raise ValueError(f"Missing model choice for {spec.key} rep={rep} {output_name} {kind}")
                    choice = selected.iloc[0]
                    model = AnchorDeltaRegressor(
                        kind,
                        q_kind=str(choice["q_kind"]),
                        local_weight=float(choice["local_weight"]),
                        random_state=7000 + rep * 31 + output_index * 7 + int(spec.key[1:]),
                    ).fit(x_train, y_standard)
                    # x_scaler and model.x_scaler_ are fitted to the same x_train.
                    train_z_model = model.x_scaler_.transform(x_train)
                    directions_model = maximin_directions(
                        model.x_scaler_.transform(x_test) - model.ad_.center_[None, :],
                        maximum=32,
                    )
                    limit = base_limit(model)
                    for rho in SHELL_RATIOS:
                        shell_x = model.points_at_ad_ratio(directions_model, rho)
                        shell_z = model.x_scaler_.transform(shell_x)
                        kernel_max = rbf_kernel_max(kind, shell_z, train_z_model)
                        predictions = {
                            f"RBF-{kind}": model.predict_base(shell_x),
                            f"AD-ADE-{kind}": model.predict(shell_x),
                        }
                        for method, pred in predictions.items():
                            for direction_index, value in enumerate(np.asarray(pred).ravel()):
                                rows.append(
                                    {
                                        "dataset": spec.key,
                                        "dataset_label": spec.label_ja,
                                        "rep": rep,
                                        "output": output_name,
                                        "kind": kind,
                                        "method": method,
                                        "rho": float(rho),
                                        "direction": direction_index,
                                        "prediction_standardized": float(value),
                                        "kernel_max": float(kernel_max[direction_index]),
                                        "base_limit": limit,
                                    }
                                )
            print(f"forward variation {spec.key} rep {rep + 1}/{reps}", flush=True)

    raw = pd.DataFrame(rows)
    output_stats = raw.groupby(
        ["dataset", "dataset_label", "rep", "output", "kind", "method", "rho"], as_index=False
    ).agg(
        directional_sd=("prediction_standardized", "std"),
        directional_range=("prediction_standardized", lambda s: float(np.ptp(s))),
        limit_rmse=(
            "prediction_standardized",
            lambda s: float(
                np.sqrt(
                    np.mean(
                        (
                            s.to_numpy(dtype=float)
                            - raw.loc[s.index, "base_limit"].to_numpy(dtype=float)
                        )
                        ** 2
                    )
                )
            ),
        ),
        median_kernel_max=("kernel_max", "median"),
    )
    rep_stats = output_stats.groupby(
        ["dataset", "dataset_label", "rep", "kind", "method", "rho"], as_index=False
    ).agg(
        directional_sd=("directional_sd", "mean"),
        directional_range=("directional_range", "mean"),
        limit_rmse=("limit_rmse", "mean"),
        median_kernel_max=("median_kernel_max", "mean"),
    )
    summary = rep_stats.groupby(
        ["dataset", "dataset_label", "kind", "method", "rho"], as_index=False
    ).agg(
        directional_sd_median=("directional_sd", "median"),
        directional_sd_q25=("directional_sd", lambda s: float(s.quantile(0.25))),
        directional_sd_q75=("directional_sd", lambda s: float(s.quantile(0.75))),
        directional_range_median=("directional_range", "median"),
        limit_rmse_median=("limit_rmse", "median"),
        median_kernel_max=("median_kernel_max", "median"),
    )
    return raw, summary


def pairwise_mean_distance(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    if len(z) < 2:
        return 0.0
    distances = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    upper = distances[np.triu_indices(len(z), k=1)]
    return float(np.mean(upper)) if len(upper) else 0.0


def naive_absolute_candidate(
    candidates: pd.DataFrame,
    bounds: np.ndarray,
    fallback_x: np.ndarray,
) -> tuple[np.ndarray, bool, float, bool]:
    """Probability-space GMR mean with a constant fallback after underflow.

    This deliberately reproduces the numerical failure motivating the user's
    original question.  It is not the recommended implementation: stable GMR
    should normalize log evidence with log-sum-exp.
    """
    x_cols = [f"x{j + 1}" for j in range(len(bounds))]
    log_score = candidates["log_score"].to_numpy(dtype=float)
    density = np.exp(log_score)
    total = float(np.sum(density))
    underflow = not np.isfinite(total) or total <= 0.0
    if underflow:
        raw_candidate = np.asarray(fallback_x, dtype=float).copy()
    else:
        weights = density / total
        raw_candidate = np.sum(
            weights[:, None] * candidates[x_cols].to_numpy(dtype=float), axis=0
        )
    candidate = np.clip(raw_candidate, bounds[:, 0], bounds[:, 1])
    projected = bool(np.any(np.abs(candidate - raw_candidate) > 1e-12))
    return candidate, projected, float(np.max(log_score)), underflow


def inverse_variation(
    specs: list[DatasetSpec],
    frames: dict[str, pd.DataFrame],
    reps: int,
    top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    rep_rows: list[dict[str, object]] = []
    for spec in specs:
        x_train_full, y_train_full, x_test, y_test = split_arrays(spec, frames[spec.key])
        bounds = _candidate_bounds(spec, x_train_full, x_test)
        for rep in range(reps):
            idx = train_indices(len(x_train_full), rep, 50000 + int(spec.key[1:]) * 313)
            x_train = x_train_full[idx]
            y_train = y_train_full[idx]
            x_scaler = StandardScaler().fit(x_train)
            y_scaler = StandardScaler().fit(y_train)
            train_y_z = y_scaler.transform(y_train)
            b0 = float(np.quantile(np.linalg.norm(train_y_z, axis=1), 0.90))
            directions = maximin_directions(y_scaler.transform(y_test), maximum=24)
            competitive = EvidenceGatedCompetitiveGMR(
                max_components=7,
                random_state=799 + rep,
                support_shell_ratio=2.0,
                support_quantile=0.05,
            ).fit(x_train, y_train)
            absolute = competitive.absolute_
            transition = competitive.transition_
            for rho in SHELL_RATIOS:
                target_z = directions * (rho * b0)
                targets = y_scaler.inverse_transform(target_z)
                by_method: dict[str, list[np.ndarray]] = {
                    "Absolute-GMR (naive density)": [],
                    "Absolute-GMR": [],
                    "Transition-GMR": [],
                    "EGC-GMR": [],
                }
                meta: dict[str, list[tuple[float, float, float, int, float, float]]] = {
                    "Absolute-GMR (naive density)": [],
                    "Absolute-GMR": [],
                    "Transition-GMR": [],
                    "EGC-GMR": [],
                }
                for direction_index, target_y in enumerate(targets):
                    anchors = nearest_y_anchors(y_train, target_y, n_anchor=min(12, len(y_train)))
                    frames_by_method = {
                        "Absolute-GMR": absolute.candidates(target_y),
                        "Transition-GMR": transition.candidates(anchors, target_y),
                    }
                    for method, candidates in frames_by_method.items():
                        candidate_set, projected, log_score = select_candidate_set(candidates, bounds, top_k)
                        candidate = candidate_set[0]
                        z_set = x_scaler.transform(candidate_set)
                        diversity = pairwise_mean_distance(z_set)
                        by_method[method].append(candidate)
                        meta[method].append(
                            (float(projected), float(log_score), diversity, len(candidate_set), 0.0, 1.0)
                        )
                        raw_rows.append(
                            {
                                "dataset": spec.key,
                                "dataset_label": spec.label_ja,
                                "rep": rep,
                                "method": method,
                                "rho": float(rho),
                                "direction": direction_index,
                                "projected_to_bounds": float(projected),
                                "log_score": float(log_score),
                                "topk_diversity": diversity,
                                "candidate_count": len(candidate_set),
                                "underflow_fallback": 0.0,
                                "supported_target": 1.0,
                                "rejection_reason": "not_gated",
                                **{
                                    f"target_{name}": float(value)
                                    for name, value in zip(spec.y_names, target_y)
                                },
                                **{
                                    f"candidate_{name}": float(value)
                                    for name, value in zip(spec.x_names, candidate)
                                },
                            }
                        )
                    competitive_result = competitive.candidates(target_y, bounds=bounds, top_k=5)
                    competitive_margin = max(
                        competitive_result.absolute_support_margin,
                        competitive_result.transition_support_margin,
                    )
                    if competitive_result.supported:
                        x_cols = [f"x{j + 1}" for j in range(len(bounds))]
                        candidate_set = competitive_result.candidates[x_cols].to_numpy(dtype=float)
                        candidate = candidate_set[0]
                        z_set = x_scaler.transform(candidate_set)
                        diversity = pairwise_mean_distance(z_set)
                        by_method["EGC-GMR"].append(candidate)
                        meta["EGC-GMR"].append(
                            (0.0, competitive_margin, diversity, len(candidate_set), 0.0, 1.0)
                        )
                    else:
                        candidate_set = np.empty((0, len(bounds)), dtype=float)
                        candidate = np.full(len(bounds), np.nan, dtype=float)
                        diversity = np.nan
                        meta["EGC-GMR"].append(
                            (0.0, competitive_margin, np.nan, 0, 0.0, 0.0)
                        )
                    raw_rows.append(
                        {
                            "dataset": spec.key,
                            "dataset_label": spec.label_ja,
                            "rep": rep,
                            "method": "EGC-GMR",
                            "rho": float(rho),
                            "direction": direction_index,
                            "projected_to_bounds": 0.0,
                            "log_score": competitive_margin,
                            "topk_diversity": diversity,
                            "candidate_count": len(candidate_set),
                            "underflow_fallback": 0.0,
                            "supported_target": float(competitive_result.supported),
                            "rejection_reason": competitive_result.reason,
                            **{
                                f"target_{name}": float(value)
                                for name, value in zip(spec.y_names, target_y)
                            },
                            **{
                                f"candidate_{name}": float(value)
                                for name, value in zip(spec.x_names, candidate)
                            },
                        }
                    )
                    naive_candidate, naive_projected, naive_log_score, underflow = naive_absolute_candidate(
                        frames_by_method["Absolute-GMR"], bounds, np.mean(x_train, axis=0)
                    )
                    naive_method = "Absolute-GMR (naive density)"
                    by_method[naive_method].append(naive_candidate)
                    meta[naive_method].append(
                        (float(naive_projected), naive_log_score, 0.0, 1, float(underflow), 1.0)
                    )
                    raw_rows.append(
                        {
                            "dataset": spec.key,
                            "dataset_label": spec.label_ja,
                            "rep": rep,
                            "method": naive_method,
                            "rho": float(rho),
                            "direction": direction_index,
                            "projected_to_bounds": float(naive_projected),
                            "log_score": naive_log_score,
                            "topk_diversity": 0.0,
                            "candidate_count": 1,
                            "underflow_fallback": float(underflow),
                            "supported_target": 1.0,
                            "rejection_reason": "not_gated",
                            **{
                                f"target_{name}": float(value)
                                for name, value in zip(spec.y_names, target_y)
                            },
                            **{
                                f"candidate_{name}": float(value)
                                for name, value in zip(spec.x_names, naive_candidate)
                            },
                        }
                    )
                for method, candidate_list in by_method.items():
                    metadata = np.asarray(meta[method], dtype=float)
                    if len(candidate_list) >= 2:
                        candidate_z = x_scaler.transform(np.vstack(candidate_list))
                        candidate_spread = float(
                            np.sqrt(np.mean(np.var(candidate_z, axis=0, ddof=1)))
                        )
                        candidate_range = float(np.sqrt(np.mean(np.ptp(candidate_z, axis=0) ** 2)))
                    elif len(candidate_list) == 1:
                        candidate_spread = np.nan
                        candidate_range = np.nan
                    else:
                        candidate_spread = np.nan
                        candidate_range = np.nan
                    rep_rows.append(
                        {
                            "dataset": spec.key,
                            "dataset_label": spec.label_ja,
                            "rep": rep,
                            "method": method,
                            "rho": float(rho),
                            "candidate_spread": candidate_spread,
                            "candidate_range": candidate_range,
                            "projection_rate": float(np.mean(metadata[:, 0])),
                            "median_log_score": float(np.median(metadata[:, 1])),
                            "topk_diversity": float(np.nanmean(metadata[:, 2])) if np.any(np.isfinite(metadata[:, 2])) else np.nan,
                            "mean_candidate_count": float(np.mean(metadata[:, 3])),
                            "underflow_fallback_rate": float(np.mean(metadata[:, 4])),
                            "supported_rate": float(np.mean(metadata[:, 5])),
                        }
                    )
            print(f"inverse variation {spec.key} rep {rep + 1}/{reps}", flush=True)

    raw = pd.DataFrame(raw_rows)
    rep_stats = pd.DataFrame(rep_rows)
    baseline = rep_stats[rep_stats["rho"] == SHELL_RATIOS[0]][
        ["dataset", "rep", "method", "median_log_score"]
    ].rename(columns={"median_log_score": "near_log_score"})
    rep_stats = rep_stats.merge(baseline, on=["dataset", "rep", "method"], how="left")
    rep_stats["log_evidence_drop"] = rep_stats["near_log_score"] - rep_stats["median_log_score"]
    summary = rep_stats.groupby(
        ["dataset", "dataset_label", "method", "rho"], as_index=False
    ).agg(
        candidate_spread_median=("candidate_spread", "median"),
        candidate_spread_q25=("candidate_spread", lambda s: float(s.quantile(0.25))),
        candidate_spread_q75=("candidate_spread", lambda s: float(s.quantile(0.75))),
        candidate_range_median=("candidate_range", "median"),
        projection_rate=("projection_rate", "mean"),
        topk_diversity_median=("topk_diversity", "median"),
        mean_candidate_count=("mean_candidate_count", "mean"),
        underflow_fallback_rate=("underflow_fallback_rate", "mean"),
        supported_rate=("supported_rate", "mean"),
        log_evidence_drop_median=("log_evidence_drop", "median"),
    )
    return raw, summary


def observed_inverse_variation(
    specs: list[DatasetSpec],
    frames: dict[str, pd.DataFrame],
    inverse_raw_path: Path,
) -> pd.DataFrame:
    raw = pd.read_csv(inverse_raw_path)
    rows: list[dict[str, object]] = []
    for spec in specs:
        train = frames[spec.key][frames[spec.key]["split"] == "train"]
        x_scale = train[spec.x_names].std(ddof=1).to_numpy(dtype=float)
        x_scale = np.where(x_scale < 1e-10, 1.0, x_scale)
        candidate_cols = [f"candidate_{name}" for name in spec.x_names]
        subset = raw[raw["dataset"] == spec.key]
        for (rep, method), group in subset.groupby(["rep", "method"]):
            valid = group[candidate_cols].dropna(axis=0, how="any")
            if len(valid) >= 2:
                z = valid.to_numpy(dtype=float) / x_scale
                spread = float(np.sqrt(np.mean(np.var(z, axis=0, ddof=1))))
            else:
                spread = np.nan
            support_rate = (
                float(group["supported_target"].mean())
                if "supported_target" in group
                else 1.0
            )
            rows.append(
                {
                    "dataset": spec.key,
                    "dataset_label": spec.label_ja,
                    "rep": int(rep),
                    "method": method,
                    "observed_target_candidate_spread": spread,
                    "observed_target_support_rate": support_rate,
                }
            )
    rep_frame = pd.DataFrame(rows)
    return rep_frame.groupby(["dataset", "dataset_label", "method"], as_index=False).agg(
        observed_target_candidate_spread_median=("observed_target_candidate_spread", "median"),
        observed_target_candidate_spread_q25=(
            "observed_target_candidate_spread", lambda s: float(s.quantile(0.25))
        ),
        observed_target_candidate_spread_q75=(
            "observed_target_candidate_spread", lambda s: float(s.quantile(0.75))
        ),
        observed_target_support_rate=("observed_target_support_rate", "mean"),
    )


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, alpha=0.55, linewidth=0.7)
    ax.tick_params(labelsize=8)


def make_forward_figure(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.7), sharex=True, constrained_layout=True)
    methods = ["RBF-GPR", "AD-ADE-GPR", "RBF-SVR", "AD-ADE-SVR"]
    styles = {
        "RBF-GPR": (GRAY, "--", "o"),
        "AD-ADE-GPR": (BLUE, "-", "o"),
        "RBF-SVR": ("#98A2B3", "--", "s"),
        "AD-ADE-SVR": (ORANGE, "-", "s"),
    }
    for index, dataset in enumerate([f"D{i}" for i in range(1, 8)]):
        ax = axes.flat[index]
        for method in methods:
            data = summary[(summary["dataset"] == dataset) & (summary["method"] == method)].sort_values("rho")
            color, linestyle, marker = styles[method]
            ax.plot(
                data["rho"],
                np.maximum(data["directional_sd_median"], 1e-7),
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=3.5,
                linewidth=1.5,
                label=method,
            )
        ax.set_title(dataset, loc="left", fontsize=10, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SHELL_RATIOS)
        ax.set_xticklabels(["1.2", "2", "4", "8", "16", "32"])
        ax.set_ylim(1e-7, 1e3)
        style_axis(ax)
        if index % 4 == 0:
            ax.set_ylabel("Directional SD (training-y SD units)", fontsize=8)
        if index >= 4:
            ax.set_xlabel(r"kNN-AD ratio $\rho_{AD}$", fontsize=8)
    axes.flat[7].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=8)
    fig.suptitle("Forward-prediction variation across extrapolation directions", fontsize=12, fontweight="bold")
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_inverse_figure(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.7), sharex=True, constrained_layout=True)
    styles = {
        "Absolute-GMR (naive density)": ("#98A2B3", ":", "s"),
        "Absolute-GMR": (GRAY, "--", "o"),
        "Transition-GMR": (BLUE, "-", "o"),
        "EGC-GMR": (TEAL, "-", "D"),
    }
    display_labels = {
        "Absolute-GMR (naive density)": "Absolute-GMR (naive density)",
        "Absolute-GMR": "Absolute-GMR (log stable)",
        "Transition-GMR": "Transition-GMR",
        "EGC-GMR": "EGC-GMR (evidence gated)",
    }
    for index, dataset in enumerate([f"D{i}" for i in range(1, 8)]):
        ax = axes.flat[index]
        for method, (color, linestyle, marker) in styles.items():
            data = summary[(summary["dataset"] == dataset) & (summary["method"] == method)].sort_values("rho")
            ax.plot(
                data["rho"],
                np.maximum(data["candidate_spread_median"], 1e-5),
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=3.5,
                linewidth=1.6,
                label=display_labels[method],
            )
        ax.set_title(dataset, loc="left", fontsize=10, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SHELL_RATIOS)
        ax.set_xticklabels(["1.2", "2", "4", "8", "16", "32"])
        style_axis(ax)
        if index % 4 == 0:
            ax.set_ylabel(r"Candidate $x$ SD across target directions", fontsize=8)
        if index >= 4:
            ax.set_xlabel(r"Target-response shell ratio $\rho_y$", fontsize=8)
    axes.flat[7].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=8)
    fig.suptitle(r"Direct-inverse candidate variation across external targets", fontsize=12, fontweight="bold")
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_inverse_support_figure(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.7), sharex=True, sharey=True, constrained_layout=True)
    for index, dataset in enumerate([f"D{i}" for i in range(1, 8)]):
        ax = axes.flat[index]
        data = summary[
            (summary["dataset"] == dataset) & (summary["method"] == "EGC-GMR")
        ].sort_values("rho")
        ax.plot(
            data["rho"],
            100.0 * data["supported_rate"],
            color=TEAL,
            marker="D",
            markersize=3.5,
            linewidth=1.6,
        )
        ax.set_title(dataset, loc="left", fontsize=10, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(SHELL_RATIOS)
        ax.set_xticklabels(["1.2", "2", "4", "8", "16", "32"])
        ax.set_ylim(-5, 105)
        style_axis(ax)
        if index % 4 == 0:
            ax.set_ylabel("Evidence-gate pass rate [%]", fontsize=8)
        if index >= 4:
            ax.set_xlabel(r"Target-response shell ratio $\rho_y$", fontsize=8)
    axes.flat[7].axis("off")
    fig.suptitle("Supported range for EGC-GMR direct inverse candidates", fontsize=12, fontweight="bold")
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("physics_validation_results"),
        help="Directory containing cached datasets and forward-model choices.",
    )
    args = parser.parse_args()
    result_dir = args.result_dir
    specs = dataset_specs()
    frames = load_frames(result_dir, specs)
    choices = pd.read_csv(result_dir / "forward_model_choices.csv")

    forward_raw, forward_summary = forward_variation(specs, frames, choices, reps=5)
    forward_raw.to_csv(result_dir / "forward_variation_raw.csv", index=False)
    forward_summary.to_csv(result_dir / "forward_variation_summary.csv", index=False)
    make_forward_figure(forward_summary, result_dir / "fig_forward_variation_shells.png")
    if args.forward_only:
        print("forward prediction variation validation completed", flush=True)
        return

    inverse_raw, inverse_summary = inverse_variation(specs, frames, reps=3, top_k=3)
    observed = observed_inverse_variation(specs, frames, result_dir / "inverse_metrics_raw.csv")

    inverse_raw.to_csv(result_dir / "inverse_variation_raw.csv", index=False)
    inverse_summary.to_csv(result_dir / "inverse_variation_summary.csv", index=False)
    observed.to_csv(result_dir / "inverse_observed_target_variation.csv", index=False)
    make_inverse_figure(inverse_summary, result_dir / "fig_inverse_variation_shells.png")
    make_inverse_support_figure(inverse_summary, result_dir / "fig_inverse_support_shells.png")
    print("prediction variation validation completed", flush=True)


if __name__ == "__main__":
    main()
