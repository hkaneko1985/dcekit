#!/usr/bin/env python3
"""Verify stage52 result integrity, nested-loop parity, and historical controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_regressors52 as r
import numpy as np


def integrity():
    r.verify_inputs()
    counts = dict(jobs=0, outer_fits=0, inner_partitions=0, prediction_events=0,
                  reused_selector_applications=0, retained_base_checks=0,
                  finite_predictions=True, test_rows_and_labels_match=True,
                  split_order_preserved=True, training_only_selection=True)
    diagnostics = dict(max_lasso_dual_gap=0.0, lasso_dual_qp_uses=0,
                       candidate_or_outer_fits_with_warnings=0, total_warning_messages=0)
    for job in r.jobs():
        case, feature, seed, repeat, regressor = job
        data = dict(np.load(r.ROOT / f"inputs/{case}__{seed or 101}.npz", allow_pickle=False))
        matrix, nbase, names = r.design(data, feature)
        expected_folds = json.loads((r.ROOT / f"inputs/{case}_splits.json").read_text())[repeat]
        path = r.ROOT / "results/jobs" / (r.stem(job) + ".json")
        obj = json.loads(path.read_text())
        array_path = path.with_suffix(".npz")
        assert r.sha(array_path) == obj["prediction_file_sha256"]
        assert obj["protocol_sha256"] == r.sha(r.ROOT / "protocol.json")
        assert obj["feature_names"] == names
        saved = dict(np.load(array_path, allow_pickle=False))
        assert np.isfinite(saved["prediction"]).all()
        expected_rows = np.concatenate([fold["test"] for fold in expected_folds])
        expected_labels = np.concatenate([[i] * len(fold["test"]) for i, fold in enumerate(expected_folds)])
        assert np.array_equal(expected_rows, saved["indices"])
        assert np.array_equal(expected_labels, saved["fold"])
        assert np.array_equal(data["y"][expected_rows], saved["y"])
        assert np.array_equal(data["ids"][expected_rows], saved["ids"])
        assert np.array_equal(data["groups"][expected_rows], saved["groups"])
        checks = []
        for fi, (fold, fit) in enumerate(zip(expected_folds, obj["fits"])):
            r.split_check(data, fold)
            assert fold["train"] == fit["train"] and fold["test"] == fit["test"]
            assert len(fold["inner"]) == len(fit["inner"])
            counts["outer_fits"] += 1
            infos = [fit["final_info"]]
            selections = [(fit["selection"], fit["train"], fit["test"])]
            for expected, actual in zip(fold["inner"], fit["inner"]):
                assert expected["train"] == actual["train"] and expected["valid"] == actual["valid"]
                counts["inner_partitions"] += 1
                infos.extend(actual["grid_fit_info"])
                selections.append((actual["selection"], actual["train"], actual["valid"] + fit["test"]))
            for info in infos:
                diagnostics["max_lasso_dual_gap"] = max(diagnostics["max_lasso_dual_gap"], float(info.get("dual_gap", 0)))
                diagnostics["lasso_dual_qp_uses"] += info.get("solver") == "LARS+dualQP"
                diagnostics["candidate_or_outer_fits_with_warnings"] += bool(info.get("warnings"))
                diagnostics["total_warning_messages"] += len(info.get("warnings", []))
            for selection, train, heldout in selections:
                if feature != "fusion_boruta":
                    assert selection is None
                    continue
                assert selection["reused"] and selection["training_indices"] == train
                assert not set(train) & set(heldout)
                assert selection["training_matrix_sha256"] == r.array_sha(matrix[train])
                assert selection["training_target_sha256"] == r.array_sha(data["y"][train])
                assert np.isin(np.arange(nbase), selection["kept_indices"]).all()
                counts["reused_selector_applications"] += 1
                counts["retained_base_checks"] += 1
            pred = saved["prediction"][saved["fold"] == fi]
            checks.append(r.metrics(data["y"][fold["test"]], pred))
        if case == "polymer_modulus":
            expected_metrics = {key: float(np.mean([item[key] for item in checks])) for key in ("rmse", "mae", "q2")}
            assert len(saved["prediction"]) == 101 and len(np.unique(saved["indices"])) == 47
        else:
            assert np.array_equal(np.sort(saved["indices"]), np.arange(len(data["y"])))
            expected_metrics = r.metrics(saved["y"], saved["prediction"])
        assert all(expected_metrics[key] == obj["summary"][key] for key in expected_metrics)
        counts["jobs"] += 1
        counts["prediction_events"] += len(saved["prediction"])
    assert counts["jobs"] == 500 and counts["outer_fits"] == 1800
    assert counts["inner_partitions"] == 4800 and counts["prediction_events"] == 62700
    assert counts["reused_selector_applications"] == 990
    assert diagnostics["max_lasso_dual_gap"] <= 1e-6
    return dict(status="PASS", **counts, numerical_diagnostics=diagnostics)


def original_loop_parity():
    data = dict(np.load(r.ROOT / "inputs/metals_volume__101.npz", allow_pickle=False))
    matrix, _, _ = r.design(data, "fusion")
    fold = json.loads((r.ROOT / "inputs/metals_volume_splits.json").read_text())[0][0]
    inner = [(np.asarray(item["train"], int), np.asarray(item["valid"], int)) for item in fold["inner"]]
    tr, te = np.asarray(fold["train"], int), np.asarray(fold["test"], int)
    numerics = r.load_numerics()
    checks = []
    for regressor in r.REGRESSORS:
        job = ("metals_volume", "fusion", 101, 0, regressor)
        obj = json.loads((r.ROOT / "results/jobs" / (r.stem(job) + ".json")).read_text())
        saved = dict(np.load(r.ROOT / "results/jobs" / (r.stem(job) + ".npz"), allow_pickle=False))
        prediction, info = numerics.nested(matrix, data["y"], tr, te, inner, regressor)
        actual = saved["prediction"][saved["fold"] == 0]
        difference = float(np.max(abs(prediction - actual)))
        expected_mse = np.asarray(info["grid_sse"]) / sum(len(b) for _, b in inner)
        grid_difference = float(np.max(abs(expected_mse - np.asarray(obj["fits"][0]["inner_grid_mse"]))))
        assert difference < 1e-7 and grid_difference < 1e-7
        assert info["config"] == obj["fits"][0]["config"]
        checks.append(dict(regressor=regressor, candidate_grid_size=len(numerics.GRIDS[regressor]),
                           predictions=len(actual), max_prediction_difference=difference,
                           max_inner_grid_mse_difference=grid_difference, config_identical=True))
    return dict(status="PASS", scope="separate original nested() control flow for one full-concatenation outer fold per new regressor",
                extra_verification_outer_refits=5, extra_verification_prediction_events=sum(item["predictions"] for item in checks),
                tolerance=1e-7, checks=checks)


def historical_polymer_controls(archive_root):
    source = Path(archive_root) / "Interaction_Descriptors_Output_Forms_v33/reference/v32/results/jobs"
    checks = []
    for job in r.jobs():
        case, feature, seed, repeat, regressor = job
        if case != "polymer_modulus" or feature.startswith("fusion"):
            continue
        tag = {"composition": "none__0__baseX", "basis": "none__0__baseXD"}.get(feature, f"{feature}__{seed}__boundary_pair")
        path = source / f"{case}__{regressor}__r{repeat}" / f"{tag}.npz"
        original = dict(np.load(path, allow_pickle=False))
        saved = dict(np.load(r.ROOT / "results/jobs" / (r.stem(job) + ".npz"), allow_pickle=False))
        for name in ("indices", "fold", "y"):
            assert np.array_equal(original[name], saved[name]), (r.stem(job), name)
        difference = float(np.max(abs(original["prediction"] - saved["prediction"])))
        checks.append(dict(regressor=regressor, feature_set=feature, source_seed=seed,
                           prediction_events=len(saved["prediction"]), max_abs_difference=difference,
                           original_path=str(path), original_sha256=r.sha(path)))
    r.write_csv(r.ROOT / "results/historical_polymer_parity.csv", checks)
    maximum = max(item["max_abs_difference"] for item in checks)
    assert len(checks) == 70
    assert maximum < 1e-7, maximum
    return dict(status="PASS", jobs=70, prediction_events=sum(item["prediction_events"] for item in checks),
                max_absolute_difference=maximum, tolerance=1e-7,
                scope="all available polymer baseline and single-family conditions; inorganic five-regressor stage52 conditions are newly fitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=r.ROOT.parents[1] / "stage51/archives")
    parser.add_argument("--skip-historical", action="store_true")
    args = parser.parse_args()
    results = dict(integrity=integrity(), original_nested_loop_parity=original_loop_parity())
    if not args.skip_historical:
        results["historical_polymer_controls"] = historical_polymer_controls(args.archive_root)
    r.verify_inputs()
    results["input_hashes_unchanged_after_verification"] = True
    r.dump(r.ROOT / "results/validation.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
