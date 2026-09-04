#!/usr/bin/env python3
"""Training-only tuning of stronger extrapolation baselines.

The original fixed RBF models are retained as mechanistic controls for the
far-field kernel-collapse diagnostic.  This script adds tuned RBF-GPR,
tuned RBF-SVR, and global ridge regression baselines for predictive accuracy.
Hyperparameters are selected only on the outer 30% shell of each training
split, then refitted on the complete training split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from physics_simulators import dataset_specs
from run_numerical_validation import KNNApplicabilityDomain
from run_physics_validation import rmse, split_arrays, train_indices


RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
GPR_LENGTH_SCALES = (0.35, 0.55, 0.85, 1.30, 2.00)
GPR_NOISE_LEVELS = (0.005, 0.015, 0.05, 0.15)
SVR_C_VALUES = (1.0, 4.0, 12.0, 36.0)
SVR_GAMMA_MULTIPLIERS = (0.20, 0.55, 1.20)
SVR_EPSILONS = (0.02, 0.05, 0.10)


def split_training_shell(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ad = KNNApplicabilityDomain(k=5, boundary_quantile=0.90).fit(z)
    cutoff = float(np.quantile(ad.training_scores_, 0.70))
    core = ad.training_scores_ <= cutoff
    shell = ~core
    if int(np.sum(shell)) < 4:
        order = np.argsort(ad.training_scores_)
        shell = np.zeros(len(z), dtype=bool)
        shell[order[-4:]] = True
        core = ~shell
    return core, shell


def make_gpr(length_scale: float, noise_level: float) -> GaussianProcessRegressor:
    kernel = RBF(length_scale, length_scale_bounds="fixed") + WhiteKernel(
        noise_level, noise_level_bounds="fixed"
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=False,
        optimizer=None,
    )


def tune_models(
    z: np.ndarray,
    w: np.ndarray,
    core: np.ndarray,
    shell: np.ndarray,
) -> dict[str, tuple[object, dict[str, float]]]:
    best_ridge: tuple[float, float] | None = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(z[core], w[core])
        error = rmse(w[shell], model.predict(z[shell]))
        candidate = (error, alpha)
        if best_ridge is None or candidate < best_ridge:
            best_ridge = candidate

    best_gpr: tuple[float, float, float] | None = None
    for length_scale in GPR_LENGTH_SCALES:
        for noise_level in GPR_NOISE_LEVELS:
            model = make_gpr(length_scale, noise_level).fit(z[core], w[core])
            error = rmse(w[shell], model.predict(z[shell]))
            candidate = (error, length_scale, noise_level)
            if best_gpr is None or candidate < best_gpr:
                best_gpr = candidate

    best_svr: tuple[float, float, float, float] | None = None
    p = max(z.shape[1], 1)
    for c_value in SVR_C_VALUES:
        for gamma_multiplier in SVR_GAMMA_MULTIPLIERS:
            for epsilon in SVR_EPSILONS:
                model = SVR(
                    kernel="rbf",
                    C=c_value,
                    gamma=gamma_multiplier / p,
                    epsilon=epsilon,
                ).fit(z[core], w[core])
                error = rmse(w[shell], model.predict(z[shell]))
                candidate = (error, c_value, gamma_multiplier, epsilon)
                if best_svr is None or candidate < best_svr:
                    best_svr = candidate

    assert best_ridge is not None and best_gpr is not None and best_svr is not None
    models = {
        "Tuned-Ridge": (
            Ridge(alpha=best_ridge[1]).fit(z, w),
            {"alpha": best_ridge[1], "shell_rmse": best_ridge[0]},
        ),
        "Tuned-RBF-GPR": (
            make_gpr(best_gpr[1], best_gpr[2]).fit(z, w),
            {
                "length_scale": best_gpr[1],
                "noise_level": best_gpr[2],
                "shell_rmse": best_gpr[0],
            },
        ),
        "Tuned-RBF-SVR": (
            SVR(
                kernel="rbf",
                C=best_svr[1],
                gamma=best_svr[2] / p,
                epsilon=best_svr[3],
            ).fit(z, w),
            {
                "C": best_svr[1],
                "gamma_multiplier": best_svr[2],
                "epsilon": best_svr[3],
                "shell_rmse": best_svr[0],
            },
        ),
    }
    return models


def run(input_dir: Path, output_dir: Path, reps: int) -> None:
    rows: list[dict[str, object]] = []
    choices: list[dict[str, object]] = []
    for spec in dataset_specs():
        frame = pd.read_csv(input_dir / "datasets" / f"{spec.key}_data.csv")
        x_train_full, y_train_full, x_test, y_test = split_arrays(spec, frame)
        for rep in range(reps):
            index = train_indices(
                len(x_train_full), rep, 10000 + int(spec.key[1:]) * 211
            )
            x_train = x_train_full[index]
            y_train = y_train_full[index]
            x_scaler = StandardScaler().fit(x_train)
            z_train = x_scaler.transform(x_train)
            z_test = x_scaler.transform(x_test)
            core, shell = split_training_shell(z_train)
            for output_index, output_name in enumerate(spec.y_names):
                y_scaler = StandardScaler().fit(y_train[:, [output_index]])
                w_train = y_scaler.transform(y_train[:, [output_index]]).ravel()
                w_test = y_scaler.transform(y_test[:, [output_index]]).ravel()
                models = tune_models(z_train, w_train, core, shell)
                test_sd = max(float(np.std(y_test[:, output_index], ddof=1)), 1e-12)
                test_range = max(
                    float(np.ptp(y_test[:, output_index])), 1e-12
                )
                train_sd = max(
                    float(np.std(y_train[:, output_index], ddof=1)), 1e-12
                )
                for model_name, (model, hyperparameters) in models.items():
                    prediction_z = np.asarray(model.predict(z_test), dtype=float)
                    prediction = y_scaler.inverse_transform(
                        prediction_z.reshape(-1, 1)
                    ).ravel()
                    raw_rmse = rmse(y_test[:, output_index], prediction)
                    rows.append(
                        {
                            "dataset": spec.key,
                            "dataset_label": spec.label_ja,
                            "primary_analysis": spec.key != "D5",
                            "rep": rep,
                            "model": model_name,
                            "level": "output",
                            "output": output_name,
                            "raw_rmse": raw_rmse,
                            "train_sd_nrmse": raw_rmse / train_sd,
                            "test_sd_nrmse": raw_rmse / test_sd,
                            "test_range_nrmse": raw_rmse / test_range,
                        }
                    )
                    choices.append(
                        {
                            "dataset": spec.key,
                            "rep": rep,
                            "output": output_name,
                            "model": model_name,
                            "training_core_count": int(np.sum(core)),
                            "training_shell_count": int(np.sum(shell)),
                            **hyperparameters,
                        }
                    )
            print(f"strong baselines {spec.key} rep {rep + 1}/{reps}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_raw = pd.DataFrame(rows)
    aggregate = output_raw.groupby(
        ["dataset", "dataset_label", "primary_analysis", "rep", "model"],
        as_index=False,
    ).agg(
        train_sd_nrmse=("train_sd_nrmse", "mean"),
        test_sd_nrmse=("test_sd_nrmse", "mean"),
        test_range_nrmse=("test_range_nrmse", "mean"),
    )
    aggregate["level"] = "aggregate"
    aggregate["output"] = "mean"
    output_raw = pd.concat([output_raw, aggregate], ignore_index=True, sort=False)
    summary = aggregate.groupby(
        ["dataset", "dataset_label", "primary_analysis", "model"], as_index=False
    ).agg(
        test_sd_nrmse_median=("test_sd_nrmse", "median"),
        test_sd_nrmse_q25=("test_sd_nrmse", lambda s: s.quantile(0.25)),
        test_sd_nrmse_q75=("test_sd_nrmse", lambda s: s.quantile(0.75)),
        train_sd_nrmse_median=("train_sd_nrmse", "median"),
        test_range_nrmse_median=("test_range_nrmse", "median"),
    )
    output_raw.to_csv(output_dir / "strong_baseline_metrics_raw.csv", index=False)
    summary.to_csv(output_dir / "strong_baseline_summary.csv", index=False)
    pd.DataFrame(choices).to_csv(
        output_dir / "strong_baseline_model_choices.csv", index=False
    )
    metadata = {
        "independent_test_use": False,
        "selection_set": "outer 30% kNN-distance shell of each training split",
        "primary_metric": "mean response-wise RMSE normalized by external-test response SD",
        "D5_status": "exploratory because failed simulations were removed non-uniformly",
        "grids": {
            "ridge_alpha": RIDGE_ALPHAS,
            "gpr_length_scale": GPR_LENGTH_SCALES,
            "gpr_noise_level": GPR_NOISE_LEVELS,
            "svr_C": SVR_C_VALUES,
            "svr_gamma_multiplier_divided_by_p": SVR_GAMMA_MULTIPLIERS,
            "svr_epsilon": SVR_EPSILONS,
        },
    }
    (output_dir / "strong_baseline_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("physics_validation_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("physics_validation_results"))
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.reps)


if __name__ == "__main__":
    main()
