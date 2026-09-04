#!/usr/bin/env python3
"""End-to-end validation on seven chemical/materials systems.

Forward extrapolation compares conventional radial-kernel GPR/SVR against the
kNN-applicability-domain-guided Anchor--Delta extrapolation (AD-ADE). Direct inverse
analysis compares absolute-coordinate GMR, Transition-GMR, and the proposed
evidence-gated competitive direct inverse framework (EGC-GMR).

The only learned representation is componentwise standardization fitted to
the current training split.  PCA, PLS and nonlinear variable transforms are
not used anywhere in the learning pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("IDAES_DATA", str(Path(__file__).resolve().parent / ".idaes"))
os.environ.setdefault("PYOMO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".pyomo"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib import font_manager
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from physics_simulators import (
    DatasetSpec,
    ROOT,
    configure_runtime_environment,
    dataset_specs,
    generate_dataset,
)
from run_numerical_validation import (
    AbsoluteGMR,
    EvidenceGatedCompetitiveGMR,
    KNNApplicabilityDomain,
    TransitionGMR,
    fit_tuned_ade,
    nearest_y_anchors,
)


configure_runtime_environment()
warnings.filterwarnings("ignore", category=RuntimeWarning)

BLUE = "#0B5FA5"
ORANGE = "#D97706"
GRAY = "#667085"
LIGHT_GRAY = "#D0D5DD"
TEAL = "#0E7C86"
RED = "#B42318"
PALE = "#EEF4F8"

FONT_PATH = ROOT / "qa_fonts" / "NotoSansCJKjp-Regular.otf"
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if len(yt) < 3 or np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
        return 0.0
    return float(pearsonr(yt, yp).statistic)


def package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for module_name in ("pycalphad", "pybamm", "idaes", "pyomo"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            versions[module_name] = f"unavailable: {exc}"
    return versions


def save_or_load_datasets(
    specs: list[DatasetSpec],
    data_dir: Path,
    regenerate: bool,
) -> dict[str, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for spec in specs:
        path = data_dir / f"{spec.key}_data.csv"
        if path.exists() and not regenerate:
            frame = pd.read_csv(path)
            print(f"dataset {spec.key}: cache {len(frame)} rows", flush=True)
        else:
            started = time.time()
            print(f"dataset {spec.key}: generating with {spec.engine}", flush=True)
            frame = generate_dataset(
                spec,
                failure_log_path=data_dir / f"{spec.key}_failed_simulations.csv",
            )
            frame.to_csv(path, index=False)
            print(
                f"dataset {spec.key}: generated {len(frame)} rows in {time.time() - started:.1f} s",
                flush=True,
            )
        frames[spec.key] = frame
    return frames


def split_arrays(spec: DatasetSpec, frame: pd.DataFrame):
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    x_train = train[spec.x_names].to_numpy(dtype=float)
    y_train = train[spec.y_names].to_numpy(dtype=float)
    x_test = test[spec.x_names].to_numpy(dtype=float)
    y_test = test[spec.y_names].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test


def train_indices(n: int, rep: int, seed: int, fraction: float = 0.90) -> np.ndarray:
    rng = np.random.default_rng(seed + rep * 1009)
    count = max(28, min(n, int(round(fraction * n))))
    return np.sort(rng.choice(n, size=count, replace=False))


def fit_forward_models(
    specs: list[DatasetSpec],
    frames: dict[str, pd.DataFrame],
    reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    choices: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for spec in specs:
        x_train_full, y_train_full, x_test, y_test = split_arrays(spec, frames[spec.key])
        for rep in range(reps):
            idx = train_indices(len(x_train_full), rep, 10000 + int(spec.key[1:]) * 211)
            x_train = x_train_full[idx]
            y_train = y_train_full[idx]
            y_center = np.mean(y_train, axis=0)
            y_scale = np.std(y_train, axis=0, ddof=1)
            y_scale = np.where(y_scale < 1e-10, 1.0, y_scale)
            predictions: dict[str, np.ndarray] = {
                "GPR": np.zeros_like(y_test),
                "AD-ADE-GPR": np.zeros_like(y_test),
                "SVR": np.zeros_like(y_test),
                "AD-ADE-SVR": np.zeros_like(y_test),
            }
            for output_index, output_name in enumerate(spec.y_names):
                y_standard = (y_train[:, output_index] - y_center[output_index]) / y_scale[output_index]
                for kind in ("GPR", "SVR"):
                    fitted, choice = fit_tuned_ade(
                        x_train,
                        y_standard,
                        kind,
                        seed=7000 + rep * 31 + output_index * 7 + int(spec.key[1:]),
                    )
                    predictions[kind][:, output_index] = (
                        fitted.predict_base(x_test) * y_scale[output_index] + y_center[output_index]
                    )
                    predictions[f"AD-ADE-{kind}"][:, output_index] = (
                        fitted.predict(x_test) * y_scale[output_index] + y_center[output_index]
                    )
                    choices.append(
                        {
                            "dataset": spec.key,
                            "rep": rep,
                            "output": output_name,
                            "kind": kind,
                            **choice,
                        }
                    )

            for model_name, pred in predictions.items():
                output_nrmse: list[float] = []
                output_mae_scaled: list[float] = []
                output_corr: list[float] = []
                output_spread: list[float] = []
                for j, output_name in enumerate(spec.y_names):
                    scale = y_scale[j]
                    value_rmse = rmse(y_test[:, j], pred[:, j])
                    nrmse = value_rmse / scale
                    mae_scaled = float(mean_absolute_error(y_test[:, j], pred[:, j]) / scale)
                    corr = safe_corr(y_test[:, j], pred[:, j])
                    spread = float(
                        np.std(pred[:, j], ddof=1) / max(np.std(y_test[:, j], ddof=1), 1e-12)
                    )
                    test_nrmse = value_rmse / max(np.std(y_test[:, j], ddof=1), 1e-12)
                    test_range_nrmse = value_rmse / max(np.ptp(y_test[:, j]), 1e-12)
                    output_nrmse.append(nrmse)
                    output_mae_scaled.append(mae_scaled)
                    output_corr.append(corr)
                    output_spread.append(spread)
                    rows.append(
                        {
                            "dataset": spec.key,
                            "dataset_label": spec.label_ja,
                            "rep": rep,
                            "model": model_name,
                            "level": "output",
                            "output": output_name,
                            "nrmse": nrmse,
                            "scaled_mae": mae_scaled,
                            "correlation": corr,
                            "spread_ratio": spread,
                            "raw_rmse": value_rmse,
                            "r2": float(r2_score(y_test[:, j], pred[:, j])),
                            "test_sd_nrmse": test_nrmse,
                            "test_range_nrmse": test_range_nrmse,
                        }
                    )
                rows.append(
                    {
                        "dataset": spec.key,
                        "dataset_label": spec.label_ja,
                        "rep": rep,
                        "model": model_name,
                        "level": "aggregate",
                        "output": "mean",
                        "nrmse": float(np.mean(output_nrmse)),
                        "scaled_mae": float(np.mean(output_mae_scaled)),
                        "correlation": float(np.mean(output_corr)),
                        "spread_ratio": float(np.mean(output_spread)),
                        "raw_rmse": np.nan,
                        "r2": np.nan,
                        "test_sd_nrmse": float(
                            np.mean(
                                [
                                    rmse(y_test[:, j], pred[:, j])
                                    / max(np.std(y_test[:, j], ddof=1), 1e-12)
                                    for j in range(len(spec.y_names))
                                ]
                            )
                        ),
                        "test_range_nrmse": float(
                            np.mean(
                                [
                                    rmse(y_test[:, j], pred[:, j])
                                    / max(np.ptp(y_test[:, j]), 1e-12)
                                    for j in range(len(spec.y_names))
                                ]
                            )
                        ),
                    }
                )

            if rep == 0:
                ad_scaler = StandardScaler().fit(x_train)
                ad = KNNApplicabilityDomain(k=5, boundary_quantile=0.90).fit(
                    ad_scaler.transform(x_train)
                )
                ad_ratio = ad.ratio(ad_scaler.transform(x_test))
                for test_index in range(len(x_test)):
                    for model_name, pred in predictions.items():
                        prediction_rows.append(
                            {
                                "dataset": spec.key,
                                "test_index": test_index,
                                "model": model_name,
                                "ad_ratio": ad_ratio[test_index],
                                "output": spec.y_names[0],
                                "actual": y_test[test_index, 0],
                                "predicted": pred[test_index, 0],
                            }
                        )
            print(f"forward {spec.key} rep {rep + 1}/{reps}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(choices), pd.DataFrame(prediction_rows)


def summarize_forward(raw: pd.DataFrame) -> pd.DataFrame:
    aggregate = raw[raw["level"] == "aggregate"]
    grouped = aggregate.groupby(["dataset", "dataset_label", "model"], sort=False)
    summary = grouped.agg(
        nrmse_median=("nrmse", "median"),
        nrmse_q25=("nrmse", lambda s: s.quantile(0.25)),
        nrmse_q75=("nrmse", lambda s: s.quantile(0.75)),
        scaled_mae_median=("scaled_mae", "median"),
        correlation_median=("correlation", "median"),
        spread_ratio_median=("spread_ratio", "median"),
        test_sd_nrmse_median=("test_sd_nrmse", "median"),
        test_range_nrmse_median=("test_range_nrmse", "median"),
    ).reset_index()
    lookup = summary.set_index(["dataset", "model"])["nrmse_median"]
    improvements = []
    for _, row in summary.iterrows():
        model = str(row["model"])
        if model.startswith("AD-ADE-"):
            baseline = model.replace("AD-ADE-", "")
            base_value = float(lookup.loc[(row["dataset"], baseline)])
            improvements.append(100.0 * (base_value - row["nrmse_median"]) / max(base_value, 1e-12))
        else:
            improvements.append(np.nan)
    summary["nrmse_improvement_percent"] = improvements
    return summary


def _candidate_bounds(spec: DatasetSpec, x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if spec.key != "D7":
        return np.asarray(spec.full_bounds, dtype=float)
    # D7 has no independently declared physical bounds.  Use the training
    # inputs only so held-out test inputs cannot influence inverse feasibility.
    lo = np.min(x_train, axis=0)
    hi = np.max(x_train, axis=0)
    margin = 0.03 * np.maximum(hi - lo, 1e-12)
    return np.column_stack([lo - margin, hi + margin])


def select_candidate(frame: pd.DataFrame, bounds: np.ndarray) -> tuple[np.ndarray, bool, float]:
    x_cols = [f"x{j + 1}" for j in range(len(bounds))]
    feasible = frame.copy()
    for j, col in enumerate(x_cols):
        feasible = feasible[feasible[col].between(bounds[j, 0], bounds[j, 1])]
    if len(feasible):
        top = feasible.iloc[0]
        return top[x_cols].to_numpy(dtype=float), False, float(top["log_score"])
    return np.full(len(bounds), np.nan, dtype=float), False, float(frame.iloc[0]["log_score"])


def select_candidate_set(
    frame: pd.DataFrame,
    bounds: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, bool, float]:
    """Return diverse feasible candidates in posterior-score order."""
    x_cols = [f"x{j + 1}" for j in range(len(bounds))]
    feasible = frame.copy()
    for j, col in enumerate(x_cols):
        feasible = feasible[feasible[col].between(bounds[j, 0], bounds[j, 1])]
    if len(feasible) == 0:
        # Apply the same no-clipping feasibility rule used by EGC-GMR.
        return np.empty((0, len(bounds)), dtype=float), False, float(frame.iloc[0]["log_score"])
    scale = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-12)
    selected: list[np.ndarray] = []
    for _, row in feasible.iterrows():
        candidate = row[x_cols].to_numpy(dtype=float)
        if not selected or min(np.linalg.norm((candidate - prior) / scale) for prior in selected) > 0.025:
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return np.vstack(selected), False, float(feasible.iloc[0]["log_score"])


def fit_inverse_models(
    specs: list[DatasetSpec],
    frames: dict[str, pd.DataFrame],
    reps: int,
    targets_per_rep: int,
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        x_train_full, y_train_full, x_test, y_test = split_arrays(spec, frames[spec.key])
        bounds = _candidate_bounds(spec, x_train_full, x_test)
        for rep in range(reps):
            idx = train_indices(len(x_train_full), rep, 50000 + int(spec.key[1:]) * 313)
            x_train = x_train_full[idx]
            y_train = y_train_full[idx]
            x_scale = np.std(x_train, axis=0, ddof=1)
            y_scale = np.std(y_train, axis=0, ddof=1)
            x_scale = np.where(x_scale < 1e-10, 1.0, x_scale)
            y_scale = np.where(y_scale < 1e-10, 1.0, y_scale)
            competitive = EvidenceGatedCompetitiveGMR(
                max_components=7,
                random_state=799 + rep,
                support_shell_ratio=2.0,
                support_quantile=0.05,
            ).fit(x_train, y_train)
            absolute = competitive.absolute_
            transition = competitive.transition_
            rng = np.random.default_rng(70000 + int(spec.key[1:]) * 701 + rep)
            count = min(targets_per_rep, len(x_test))
            target_indices = np.sort(rng.choice(len(x_test), size=count, replace=False))
            for target_index in target_indices:
                target_y = y_test[target_index]
                anchors = nearest_y_anchors(y_train, target_y, n_anchor=min(12, len(y_train)))
                candidate_frames = {
                    "Absolute-GMR": absolute.candidates(target_y),
                    "Transition-GMR": transition.candidates(anchors, target_y),
                }
                evaluations: list[dict[str, object]] = []
                for method, candidates in candidate_frames.items():
                    candidate_set, projected, log_score = select_candidate_set(candidates, bounds, top_k)
                    feasible_candidate = bool(len(candidate_set))
                    evaluations.append(
                        {
                            "method": method,
                            "candidate_set": candidate_set,
                            "projected": projected,
                            "log_score": log_score,
                            "supported": feasible_candidate,
                            "reason": "not_gated" if feasible_candidate else "no_feasible_candidate",
                            "absolute_log_evidence": np.nan,
                            "transition_log_evidence": np.nan,
                            "absolute_support_margin": np.nan,
                            "transition_support_margin": np.nan,
                            "primary_source": method,
                            "absolute_candidate_fraction": float(method == "Absolute-GMR"),
                            "transition_candidate_fraction": float(method == "Transition-GMR"),
                        }
                    )

                competitive_result = competitive.candidates(target_y, bounds=bounds, top_k=top_k)
                if competitive_result.supported:
                    x_cols = [f"x{j + 1}" for j in range(len(bounds))]
                    competitive_set = competitive_result.candidates[x_cols].to_numpy(dtype=float)
                    source = competitive_result.candidates["source_method"].astype(str)
                    primary_source = str(source.iloc[0])
                    absolute_fraction = float(np.mean(source == "Absolute-GMR"))
                    transition_fraction = float(np.mean(source == "Transition-GMR"))
                else:
                    competitive_set = np.empty((0, len(bounds)), dtype=float)
                    primary_source = "none"
                    absolute_fraction = 0.0
                    transition_fraction = 0.0
                evaluations.append(
                    {
                        "method": "EGC-GMR",
                        "candidate_set": competitive_set,
                        "projected": False,
                        "log_score": np.nan,
                        "supported": competitive_result.supported,
                        "reason": competitive_result.reason,
                        "absolute_log_evidence": competitive_result.absolute_log_evidence,
                        "transition_log_evidence": competitive_result.transition_log_evidence,
                        "absolute_support_margin": competitive_result.absolute_support_margin,
                        "transition_support_margin": competitive_result.transition_support_margin,
                        "primary_source": primary_source,
                        "absolute_candidate_fraction": absolute_fraction,
                        "transition_candidate_fraction": transition_fraction,
                    }
                )

                for evaluation in evaluations:
                    method = str(evaluation["method"])
                    candidate_set = np.asarray(evaluation["candidate_set"], dtype=float)
                    projected = bool(evaluation["projected"])
                    log_score = float(evaluation["log_score"])
                    supported = bool(evaluation["supported"])
                    candidate = (
                        candidate_set[0]
                        if len(candidate_set)
                        else np.full(len(spec.x_names), np.nan, dtype=float)
                    )
                    if len(candidate_set):
                        input_errors = np.linalg.norm(
                            (candidate_set - x_test[target_index]) / x_scale,
                            axis=1,
                        ) / math.sqrt(len(spec.x_names))
                        input_error = float(input_errors[0])
                        input_top3_error = float(np.min(input_errors[: min(3, len(input_errors))]))
                        input_top5_error = float(np.min(input_errors[: min(5, len(input_errors))]))
                    else:
                        input_errors = np.asarray([], dtype=float)
                        input_error = np.nan
                        input_top3_error = np.nan
                        input_top5_error = np.nan
                    response_top3_error = np.nan
                    response_top5_error = np.nan
                    response_errors: list[float] = []
                    simulator_ok = False
                    if len(candidate_set) and spec.forward_verifiable and spec.simulator is not None:
                        try:
                            simulated_set = spec.simulator(candidate_set)
                            for simulated in simulated_set:
                                if np.all(np.isfinite(simulated)):
                                    response_errors.append(
                                        float(
                                            np.linalg.norm((simulated - target_y) / y_scale)
                                            / math.sqrt(len(spec.y_names))
                                        )
                                    )
                                else:
                                    response_errors.append(np.nan)
                            simulator_ok = bool(
                                len(response_errors) and np.any(np.isfinite(response_errors))
                            )
                        except Exception:
                            response_errors = []
                    response_error = response_errors[0] if response_errors else np.nan
                    if response_errors and np.any(np.isfinite(response_errors)):
                        response_array = np.asarray(response_errors, dtype=float)
                        response_top3_error = float(
                            np.nanmin(response_array[: min(3, len(response_array))])
                        )
                        response_top5_error = float(
                            np.nanmin(response_array[: min(5, len(response_array))])
                        )
                    rows.append(
                        {
                            "dataset": spec.key,
                            "dataset_label": spec.label_ja,
                            "rep": rep,
                            "target_index": int(target_index),
                            "method": method,
                            "response_nrmse": response_error,
                            "topk_response_nrmse": response_top3_error,
                            "top5_response_nrmse": response_top5_error,
                            "input_standardized_error": input_error,
                            "topk_input_standardized_error": input_top3_error,
                            "top5_input_standardized_error": input_top5_error,
                            "candidate_count": len(candidate_set),
                            "projected_to_bounds": float(projected),
                            "simulator_verified": float(simulator_ok),
                            "log_score": log_score,
                            "supported_target": float(supported),
                            "rejection_reason": str(evaluation["reason"]),
                            "unsupported_evidence": float(evaluation["reason"] == "unsupported_evidence"),
                            "no_feasible_candidate": float(evaluation["reason"] == "no_feasible_candidate"),
                            "absolute_log_evidence": evaluation["absolute_log_evidence"],
                            "transition_log_evidence": evaluation["transition_log_evidence"],
                            "absolute_support_margin": evaluation["absolute_support_margin"],
                            "transition_support_margin": evaluation["transition_support_margin"],
                            "primary_source": evaluation["primary_source"],
                            "absolute_candidate_fraction": evaluation["absolute_candidate_fraction"],
                            "transition_candidate_fraction": evaluation["transition_candidate_fraction"],
                            "absolute_components": absolute.n_components_,
                            "transition_components": transition.n_components_,
                            **{f"candidate_{name}": value for name, value in zip(spec.x_names, candidate)},
                        }
                    )
            print(f"inverse {spec.key} rep {rep + 1}/{reps}", flush=True)
    return pd.DataFrame(rows)


def summarize_inverse(raw: pd.DataFrame) -> pd.DataFrame:
    rep_level = raw.groupby(["dataset", "dataset_label", "rep", "method"], as_index=False).agg(
        response_nrmse=("response_nrmse", "mean"),
        topk_response_nrmse=("topk_response_nrmse", "mean"),
        top5_response_nrmse=("top5_response_nrmse", "mean"),
        input_standardized_error=("input_standardized_error", "mean"),
        topk_input_standardized_error=("topk_input_standardized_error", "mean"),
        top5_input_standardized_error=("top5_input_standardized_error", "mean"),
        projected_rate=("projected_to_bounds", "mean"),
        verification_rate=("simulator_verified", "mean"),
        acceptance_rate=("supported_target", "mean"),
        unsupported_evidence_rate=("unsupported_evidence", "mean"),
        no_feasible_candidate_rate=("no_feasible_candidate", "mean"),
        candidate_count=("candidate_count", "mean"),
        absolute_candidate_fraction=("absolute_candidate_fraction", "mean"),
        transition_candidate_fraction=("transition_candidate_fraction", "mean"),
    )
    return rep_level.groupby(["dataset", "dataset_label", "method"], as_index=False).agg(
        response_nrmse_median=("response_nrmse", "median"),
        response_nrmse_q25=("response_nrmse", lambda s: s.quantile(0.25)),
        response_nrmse_q75=("response_nrmse", lambda s: s.quantile(0.75)),
        input_error_median=("input_standardized_error", "median"),
        topk_response_nrmse_median=("topk_response_nrmse", "median"),
        topk_input_error_median=("topk_input_standardized_error", "median"),
        top5_response_nrmse_median=("top5_response_nrmse", "median"),
        top5_input_error_median=("top5_input_standardized_error", "median"),
        projected_rate=("projected_rate", "mean"),
        verification_rate=("verification_rate", "mean"),
        acceptance_rate=("acceptance_rate", "mean"),
        unsupported_evidence_rate=("unsupported_evidence_rate", "mean"),
        no_feasible_candidate_rate=("no_feasible_candidate_rate", "mean"),
        mean_candidate_count=("candidate_count", "mean"),
        absolute_candidate_fraction=("absolute_candidate_fraction", "mean"),
        transition_candidate_fraction=("transition_candidate_fraction", "mean"),
    )


def summarize_inverse_matched(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare all methods on exactly the targets verified for EGC-GMR.

    D1--D6 require a simulator-verified EGC candidate.  D7 has no unperformed
    experiment with which to verify the response, so an accepted EGC target is
    retained and only input-space errors are interpreted.
    """
    key_columns = ["dataset", "dataset_label", "rep", "target_index"]
    egc = raw[raw["method"] == "EGC-GMR"].copy()
    eligible = (egc["supported_target"] > 0.5) & (
        (egc["dataset"] == "D7") | (egc["simulator_verified"] > 0.5)
    )
    keys = egc.loc[eligible, key_columns].drop_duplicates()
    matched = raw.merge(keys, on=key_columns, how="inner", validate="many_to_one")

    total_counts = (
        raw[key_columns]
        .drop_duplicates()
        .groupby(["dataset", "dataset_label"], as_index=False)
        .size()
        .rename(columns={"size": "total_target_count"})
    )
    matched_counts = (
        keys.groupby(["dataset", "dataset_label"], as_index=False)
        .size()
        .rename(columns={"size": "matched_target_count"})
        .merge(total_counts, on=["dataset", "dataset_label"], how="left")
    )
    matched_counts["matched_target_fraction"] = (
        matched_counts["matched_target_count"] / matched_counts["total_target_count"]
    )

    rep_level = matched.groupby(
        ["dataset", "dataset_label", "rep", "method"], as_index=False
    ).agg(
        response_nrmse=("response_nrmse", "mean"),
        topk_response_nrmse=("topk_response_nrmse", "mean"),
        top5_response_nrmse=("top5_response_nrmse", "mean"),
        input_standardized_error=("input_standardized_error", "mean"),
        topk_input_standardized_error=("topk_input_standardized_error", "mean"),
        top5_input_standardized_error=("top5_input_standardized_error", "mean"),
    )
    summary = rep_level.groupby(
        ["dataset", "dataset_label", "method"], as_index=False
    ).agg(
        response_nrmse_median=("response_nrmse", "median"),
        topk_response_nrmse_median=("topk_response_nrmse", "median"),
        top5_response_nrmse_median=("top5_response_nrmse", "median"),
        input_error_median=("input_standardized_error", "median"),
        topk_input_error_median=("topk_input_standardized_error", "median"),
        top5_input_error_median=("top5_input_standardized_error", "median"),
    )
    return summary.merge(
        matched_counts,
        on=["dataset", "dataset_label"],
        how="left",
        validate="many_to_one",
    )


def summarize_inverse_decision_utility(
    raw: pd.DataFrame,
    rejection_penalty: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate inverse decisions on every target, counting rejection as failure.

    D5 is excluded from the primary analysis because failed simulator runs were
    removed non-uniformly from its training and external designs.  D7 is
    excluded because no new experiment verifies whether an inverse candidate
    reaches the target response.  Both remain available in the raw outputs as
    explicitly exploratory analyses.
    """
    evaluated = raw[~raw["dataset"].isin(["D5", "D7"])].copy()
    evaluated["verified_answer"] = (
        (evaluated["supported_target"] > 0.5)
        & (evaluated["candidate_count"] > 0)
        & (evaluated["simulator_verified"] > 0.5)
        & evaluated["top5_response_nrmse"].notna()
    )
    evaluated["success_nrmse_le_0_5"] = (
        evaluated["verified_answer"]
        & (evaluated["top5_response_nrmse"] <= 0.5)
    )
    evaluated["success_nrmse_le_1_0"] = (
        evaluated["verified_answer"]
        & (evaluated["top5_response_nrmse"] <= 1.0)
    )
    evaluated["penalized_top5_nrmse"] = np.where(
        evaluated["verified_answer"],
        np.minimum(evaluated["top5_response_nrmse"], rejection_penalty),
        rejection_penalty,
    )

    per_dataset = evaluated.groupby(
        ["dataset", "dataset_label", "method"], as_index=False
    ).agg(
        target_count=("target_index", "size"),
        verified_answer_rate=("verified_answer", "mean"),
        success_rate_nrmse_le_0_5=("success_nrmse_le_0_5", "mean"),
        success_rate_nrmse_le_1_0=("success_nrmse_le_1_0", "mean"),
        conditional_top5_nrmse_median=(
            "top5_response_nrmse",
            lambda values: float(
                values[evaluated.loc[values.index, "verified_answer"]].median()
            ),
        ),
        penalized_top5_nrmse_mean=("penalized_top5_nrmse", "mean"),
    )
    overall = evaluated.groupby("method", as_index=False).agg(
        target_count=("target_index", "size"),
        verified_answer_rate=("verified_answer", "mean"),
        success_rate_nrmse_le_0_5=("success_nrmse_le_0_5", "mean"),
        success_rate_nrmse_le_1_0=("success_nrmse_le_1_0", "mean"),
        conditional_top5_nrmse_median=(
            "top5_response_nrmse",
            lambda values: float(
                values[evaluated.loc[values.index, "verified_answer"]].median()
            ),
        ),
        penalized_top5_nrmse_mean=("penalized_top5_nrmse", "mean"),
    )
    overall["datasets"] = "D1-D4,D6"
    overall["rejection_penalty"] = rejection_penalty
    return per_dataset, overall


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, alpha=0.55, linewidth=0.7)
    ax.tick_params(labelsize=8)


def make_forward_summary_figure(summary: pd.DataFrame, output: Path) -> None:
    datasets = [f"D{i}" for i in range(1, 8)]
    models = ["GPR", "AD-ADE-GPR", "SVR", "AD-ADE-SVR"]
    colors = [GRAY, BLUE, "#98A2B3", ORANGE]
    hatches = ["", "//", "", "//"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), constrained_layout=True)
    width = 0.19
    pos = np.arange(len(datasets))
    for mi, (model, color, hatch) in enumerate(zip(models, colors, hatches)):
        data = summary[summary["model"] == model].set_index("dataset").reindex(datasets)
        y = data["nrmse_median"].to_numpy(dtype=float)
        q25 = data["nrmse_q25"].to_numpy(dtype=float)
        q75 = data["nrmse_q75"].to_numpy(dtype=float)
        xpos = pos + (mi - 1.5) * width
        axes[0].bar(xpos, y, width, color=color, hatch=hatch, label=model, edgecolor="white")
        axes[0].errorbar(
            xpos,
            y,
            yerr=np.vstack([y - q25, q75 - y]),
            fmt="none",
            ecolor="#344054",
            linewidth=0.7,
            capsize=2,
        )
        spread = data["spread_ratio_median"].to_numpy(dtype=float)
        axes[1].bar(xpos, spread, width, color=color, hatch=hatch, edgecolor="white")
    axes[0].set_title("(a) Forward extrapolation NRMSE (median and IQR)", loc="left", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mean NRMSE", fontsize=9)
    axes[0].set_yscale("log")
    axes[1].set_title("(b) Predicted spread / reference spread", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Spread ratio (1 = ideal)", fontsize=9)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    for ax in axes:
        ax.set_xticks(pos, datasets)
        style_axis(ax)
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_improvement_figure(summary: pd.DataFrame, output: Path) -> None:
    data = summary[summary["model"].isin(["AD-ADE-GPR", "AD-ADE-SVR"])].copy()
    pivot = data.pivot(index="dataset", columns="model", values="nrmse_improvement_percent").reindex(
        [f"D{i}" for i in range(1, 8)]
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.3), constrained_layout=True)
    pos = np.arange(len(pivot))
    width = 0.34
    ax.bar(pos - width / 2, pivot["AD-ADE-GPR"], width, color=BLUE, label="AD-ADE-GPR")
    ax.bar(pos + width / 2, pivot["AD-ADE-SVR"], width, color=ORANGE, label="AD-ADE-SVR")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(pos, pivot.index)
    ax.set_ylabel("NRMSE improvement (%)", fontsize=9)
    ax.set_title("Change in extrapolation error relative to conventional RBF models", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax)
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_prediction_examples_figure(predictions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12.0), constrained_layout=True)
    axes = axes.ravel()
    for idx, dataset in enumerate([f"D{i}" for i in range(1, 8)]):
        ax = axes[idx]
        data = predictions[predictions["dataset"] == dataset]
        actual = data[data["model"] == "GPR"].sort_values("ad_ratio")
        for model, color, marker in (
            ("GPR", GRAY, "x"),
            ("AD-ADE-GPR", BLUE, "o"),
            ("SVR", "#98A2B3", "+"),
            ("AD-ADE-SVR", ORANGE, "s"),
        ):
            subset = data[data["model"] == model].sort_values("ad_ratio")
            ax.scatter(
                subset["actual"],
                subset["predicted"],
                s=16,
                alpha=0.55,
                color=color,
                marker=marker,
                label=model,
            )
        lo = min(float(actual["actual"].min()), float(data["predicted"].min()))
        hi = max(float(actual["actual"].max()), float(data["predicted"].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
        ax.set_title(dataset, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Ground truth in extrapolation domain", fontsize=8)
        ax.set_ylabel("Prediction", fontsize=8)
        style_axis(ax)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, ncol=2, frameon=False, loc="center", fontsize=10)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_inverse_figure(summary: pd.DataFrame, output: Path) -> None:
    datasets = [f"D{i}" for i in range(1, 8)]
    methods = ["Absolute-GMR", "Transition-GMR", "EGC-GMR"]
    colors = [GRAY, BLUE, TEAL]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), constrained_layout=True)
    width = 0.25
    pos = np.arange(len(datasets))
    for mi, (method, color) in enumerate(zip(methods, colors)):
        data = summary[summary["method"] == method].set_index("dataset").reindex(datasets)
        xerr = data["input_error_median"].to_numpy(dtype=float)
        axes[0].bar(pos + (mi - 1.0) * width, xerr, width, color=color, label=method)
        rerr = data["response_nrmse_median"].to_numpy(dtype=float)
        axes[1].bar(pos + (mi - 1.0) * width, rerr, width, color=color)
    axes[0].set_title("(a) Direct inverse analysis: standardized input recovery error", loc="left", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("standardized input error", fontsize=9)
    axes[1].set_title("(b) Target-response error after simulator verification", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("response NRMSE", fontsize=9)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(pos, datasets)
        style_axis(ax)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].text(6, 0.02, "D7 requires new experiments", ha="center", va="bottom", fontsize=8, color=RED)
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_inverse_topk_figure(summary: pd.DataFrame, output: Path) -> None:
    datasets = [f"D{i}" for i in range(1, 8)]
    methods = ["Absolute-GMR", "Transition-GMR", "EGC-GMR"]
    colors = [GRAY, BLUE, TEAL]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), constrained_layout=True)
    width = 0.25
    pos = np.arange(len(datasets))
    for mi, (method, color) in enumerate(zip(methods, colors)):
        data = summary[summary["method"] == method].set_index("dataset").reindex(datasets)
        xerr = data["topk_input_error_median"].to_numpy(dtype=float)
        rerr = data["topk_response_nrmse_median"].to_numpy(dtype=float)
        axes[0].bar(pos + (mi - 1.0) * width, xerr, width, color=color, label=method)
        axes[1].bar(pos + (mi - 1.0) * width, rerr, width, color=color)
    axes[0].set_title("(a) External-input coverage error of the top three candidates", loc="left", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("minimum standardized input error", fontsize=9)
    axes[1].set_title("(b) Minimum simulator-verified response error among the top three", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("minimum response NRMSE", fontsize=9)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(pos, datasets)
        style_axis(ax)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].text(6, 0.02, "D7 requires new experiments", ha="center", va="bottom", fontsize=8, color=RED)
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_inverse_support_figure(
    summary: pd.DataFrame, matched_summary: pd.DataFrame, output: Path
) -> None:
    datasets = [f"D{i}" for i in range(1, 8)]
    methods = ["Absolute-GMR", "Transition-GMR", "EGC-GMR"]
    colors = [GRAY, BLUE, TEAL]
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.0), constrained_layout=True)
    pos = np.arange(len(datasets))
    egc = summary[summary["method"] == "EGC-GMR"].set_index("dataset").reindex(datasets)
    axes[0].bar(
        pos,
        100.0 * egc["acceptance_rate"].to_numpy(dtype=float),
        color=TEAL,
        width=0.55,
    )
    axes[0].set_title("(a) EGC-GMR evidence-gate pass rate", loc="left", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("accepted targets [%]", fontsize=9)
    axes[0].set_ylim(0, 105)

    width = 0.25
    for mi, (method, color) in enumerate(zip(methods, colors)):
        data = (
            matched_summary[matched_summary["method"] == method]
            .set_index("dataset")
            .reindex(datasets)
        )
        values = data["top5_response_nrmse_median"].to_numpy(dtype=float)
        axes[1].bar(pos + (mi - 1.0) * width, values, width, color=color, label=method)
    axes[1].set_title(
        "(b) Minimum top-five response error on identical EGC-accepted verified targets",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_ylabel("minimum response NRMSE", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, ncol=3)
    axes[1].text(6, 0.02, "D7 requires new experiments", ha="center", va="bottom", fontsize=8, color=RED)
    for ax in axes:
        ax.set_xticks(pos, datasets)
        style_axis(ax)
    fig.savefig(output, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dataset_manifest(specs: list[DatasetSpec], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for spec in specs:
        frame = frames[spec.key]
        records.append(
            {
                "dataset": spec.key,
                "label": spec.label_ja,
                "engine": spec.engine,
                "engine_status": spec.engine_status,
                "n_train": int(np.sum(frame["split"] == "train")),
                "n_test": int(np.sum(frame["split"] == "test")),
                "n_train_requested": spec.n_core,
                "n_test_requested": spec.n_outer,
                "train_success_rate": float(np.sum(frame["split"] == "train") / spec.n_core),
                "test_success_rate": float(np.sum(frame["split"] == "test") / spec.n_outer),
                "primary_analysis": spec.key != "D5",
                "n_inputs": len(spec.x_names),
                "n_outputs": len(spec.y_names),
                "input_variables": "; ".join(spec.x_names),
                "output_variables": "; ".join(spec.y_names),
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("physics_validation_results"))
    parser.add_argument("--forward-reps", type=int, default=5)
    parser.add_argument("--inverse-reps", type=int, default=3)
    parser.add_argument("--inverse-targets", type=int, default=8)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = dataset_specs()
    frames = save_or_load_datasets(specs, args.output / "datasets", args.regenerate)
    manifest_df = dataset_manifest(specs, frames)
    manifest_df.to_csv(args.output / "dataset_manifest.csv", index=False)
    if args.data_only:
        return

    forward_raw, choices, predictions = fit_forward_models(specs, frames, args.forward_reps)
    forward_summary = summarize_forward(forward_raw)
    forward_raw.to_csv(args.output / "forward_metrics_raw.csv", index=False)
    forward_summary.to_csv(args.output / "forward_summary.csv", index=False)
    choices.to_csv(args.output / "forward_model_choices.csv", index=False)
    predictions.to_csv(args.output / "forward_example_predictions.csv", index=False)

    make_forward_summary_figure(forward_summary, args.output / "fig_forward_summary.png")
    make_improvement_figure(forward_summary, args.output / "fig_forward_improvement.png")
    make_prediction_examples_figure(predictions, args.output / "fig_forward_examples.png")

    if args.forward_only:
        manifest_path = args.output / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest.update(
            {
                "created_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
                "representation": "training-split componentwise StandardScaler only; no PCA, no PLS, no nonlinear feature transform",
                "forward_methods": ["RBF-GPR", "AD-ADE-GPR", "RBF-SVR", "AD-ADE-SVR"],
                "forward_applicability_domain": {
                    "metric": "mean k=5 nearest-neighbor distance in training-standardized x space",
                    "boundary": "Q90 of leave-one-out training kNN distances",
                },
                "forward_repetitions": args.forward_reps,
                "versions": package_versions(),
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("forward physics validation completed", flush=True)
        return

    inverse_raw = fit_inverse_models(specs, frames, args.inverse_reps, args.inverse_targets)
    inverse_summary = summarize_inverse(inverse_raw)
    inverse_matched_summary = summarize_inverse_matched(inverse_raw)
    inverse_decision_by_dataset, inverse_decision_overall = summarize_inverse_decision_utility(
        inverse_raw
    )
    inverse_raw.to_csv(args.output / "inverse_metrics_raw.csv", index=False)
    inverse_summary.to_csv(args.output / "inverse_summary.csv", index=False)
    inverse_matched_summary.to_csv(args.output / "inverse_matched_summary.csv", index=False)
    inverse_decision_by_dataset.to_csv(
        args.output / "inverse_decision_utility_by_dataset.csv", index=False
    )
    inverse_decision_overall.to_csv(
        args.output / "inverse_decision_utility_overall.csv", index=False
    )

    make_inverse_figure(inverse_matched_summary, args.output / "fig_inverse_summary.png")
    make_inverse_topk_figure(inverse_matched_summary, args.output / "fig_inverse_topk.png")
    make_inverse_support_figure(
        inverse_summary,
        inverse_matched_summary,
        args.output / "fig_inverse_support.png",
    )

    manifest = {
        "created_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "representation": "training-split componentwise StandardScaler only; no PCA, no PLS, no nonlinear feature transform",
        "forward_methods": ["RBF-GPR", "AD-ADE-GPR", "RBF-SVR", "AD-ADE-SVR"],
        "forward_applicability_domain": {
            "metric": "mean k=5 nearest-neighbor distance in training-standardized x space",
            "boundary": "Q90 of leave-one-out training kNN distances",
        },
        "inverse_methods": [
            "absolute-coordinate GMR (log-stable baseline)",
            "Transition-GMR direct conditioning",
            "EGC-GMR: evidence-gated competitive direct inverse analysis",
        ],
        "egc_gmr": {
            "support_shell_ratio": 2.0,
            "support_quantile": 0.05,
            "support_radius_quantile": 0.90,
            "maximum_support_directions": 24,
            "candidate_budget": 5,
            "candidate_sources": ["Absolute-GMR", "Transition-GMR"],
            "feasibility_rule": "discard candidates outside declared physical bounds; do not clip",
            "selection_rule": "interleave model-specific candidates by calibrated support after covariance and diversity filtering",
            "rejection_rule": "return unsupported_evidence if neither model exceeds its training-only shell threshold",
        },
        "inverse_matched_comparison": {
            "rule": "compare all three methods only on targets accepted and simulator-verified for EGC-GMR; D7 retains accepted targets for input-space comparison only",
            "output": "inverse_matched_summary.csv",
        },
        "inverse_all_target_comparison": {
            "primary_datasets": "D1-D4,D6",
            "excluded_from_primary": {
                "D5": "non-uniform simulator-failure filtering",
                "D7": "inverse responses require new experiments",
            },
            "rule": "count every rejected, infeasible, or unverified target as an unsuccessful decision",
            "rejection_penalty_nrmse": 2.0,
            "outputs": [
                "inverse_decision_utility_by_dataset.csv",
                "inverse_decision_utility_overall.csv",
            ],
        },
        "forward_repetitions": args.forward_reps,
        "inverse_repetitions": args.inverse_reps,
        "inverse_targets_per_repetition": args.inverse_targets,
        "variation_diagnostic": {
            "shell_ratios": [1.2, 2.0, 4.0, 8.0, 16.0, 32.0],
            "forward_repetitions": 5,
            "inverse_repetitions": 3,
            "maximum_directions": 24,
            "purpose": (
                "verify that conventional local-kernel predictions collapse to a constant, "
                "whereas the proposed forward model remains direction-dependent and EGC-GMR "
                "either returns direction-dependent direct inverse candidates or explicitly rejects "
                "targets outside its training-calibrated evidence support"
            ),
        },
        "versions": package_versions(),
        "random_seed": 20260823,
        "limitations": [
            "Outer-domain truth is limited to the declared physical bounds.",
            "EGC-GMR errors are conditional on targets accepted by the support gate; acceptance is reported separately.",
            "D5 is exploratory because 21/125 training and 23/65 external simulator runs failed and were removed.",
            "IDAES CO2 uses five spatial elements and short adsorption/desorption horizons for tractability.",
            "Concrete Slump candidates cannot be confirmed without new experiments; inverse response error is not reported.",
            "Phase-field is an in-house finite-difference implementation, not MOOSE.",
        ],
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("physics validation completed", flush=True)


if __name__ == "__main__":
    main()
