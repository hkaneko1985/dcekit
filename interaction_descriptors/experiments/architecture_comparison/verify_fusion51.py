#!/usr/bin/env python3
"""Focused checks for the added nested selector and inherited predictions."""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np

import run_fusion51 as r


def numerical_parity(source_root):
    p = source_root / "Interaction_Descriptors_Learning_Curves_v44_20261106/code/regression.py"
    spec = importlib.util.spec_from_file_location("historical_regression", p)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    d = dict(np.load(r.ROOT/"inputs/metals_volume__101.npz"))
    f = json.loads((r.ROOT/"inputs/metals_volume_splits.json").read_text())[0][0]
    F, _, _ = r.design(d, "fusion")
    a, b = np.asarray(f["inner"][0]["train"]), np.asarray(f["inner"][0]["valid"])
    A, B, keep = r.prep(F[a], F[b])
    Ar, Br, kr = ref.prep(F[a], F[b])
    assert np.array_equal(A, Ar) and np.array_equal(B, Br) and np.array_equal(keep, kr)
    p1 = r.predict_grid(A, d["y"][a], B)
    p2, _ = ref.predict_grid(A, d["y"][a], B, "KRR")
    max_grid = float(np.max(abs(p1-p2)))
    assert max_grid <= 1e-12
    one = r.predict_one(A, d["y"][a], B, r.GRID[13])
    one_ref, _ = ref.run_one(A, d["y"][a], B, "KRR", r.GRID[13])
    max_one = float(np.max(abs(one-one_ref)))
    assert max_one <= 1e-12
    return dict(grid_max_abs_difference=max_grid, final_fit_max_abs_difference=max_one,
                inherited_regression_sha256=r.sha(p))


def heldout_perturbation():
    d = dict(np.load(r.ROOT/"inputs/polymer_modulus__101.npz"))
    F, nbase, names = r.design(d, "fusion")
    fold = json.loads((r.ROOT/"inputs/polymer_modulus_splits.json").read_text())[0][0]
    train = np.asarray(fold["inner"][0]["train"], int)
    outside = np.setdiff1d(np.arange(len(F)), train)
    y = d["y"].copy()
    selected1, audit1 = r.select_features(F, y, train, nbase, names, 519999)
    modified_F = F.copy()
    modified_F[outside] = 1e6 + np.random.default_rng(23).normal(size=modified_F[outside].shape)
    modified_y = y.copy()
    modified_y[outside] = -1e8 + np.arange(len(outside)) * 1e7
    selected2, audit2 = r.select_features(modified_F, modified_y, train, nbase, names, 519999)
    assert np.array_equal(selected1, selected2)
    for key in ("decisions", "hit_counts", "weak_rough_fix", "kept_indices", "iterations"):
        assert audit1[key] == audit2[key], key
    assert selected1[:nbase].all()
    assert not set(audit1["training_indices"]) & set(outside)
    return dict(training_rows=len(train), changed_validation_and_test_rows=len(outside),
                selector_decisions_identical=True, hit_counts_identical=True,
                protected_base_preserved=True)


def audit_completed_results():
    count = 0
    zero = 0
    for path in sorted((r.ROOT/"results/jobs").glob("*fusion_boruta*.json")):
        obj = json.loads(path.read_text())
        nbase = None
        for fold in obj["fits"]:
            test = set(fold["test"])
            outer = fold["selection"]
            assert set(outer["training_indices"]) == set(fold["train"])
            assert not test & set(outer["training_indices"])
            nbase = outer["protected_base_dim"]
            audit = [outer]
            for inner in fold["inner"]:
                ss = inner["selection"]
                assert set(ss["training_indices"]) == set(inner["train"])
                assert not set(ss["training_indices"]) & (set(inner["valid"]) | test)
                audit.append(ss)
            for ss in audit:
                decisions = np.array(ss["decisions"])
                kept = np.array(ss["kept_indices"], int)
                assert np.isin(np.arange(nbase), kept).all()
                assert np.array_equal(kept[kept >= nbase], np.flatnonzero(decisions[nbase:] == 1)+nbase)
                assert ss["confirmed_learned"] == len(kept[kept >= nbase])
                zero += int(ss["confirmed_learned"] == 0)
                count += 1
    return dict(selector_fits_audited=count, zero_confirmed_base_only_fits=zero,
                no_heldout_rows_used=True, only_confirmed_learned_coordinates=True,
                base_always_retained=True)


def inherited_predictions(source_root):
    comparisons = []
    p33 = source_root/"Interaction_Descriptors_Output_Forms_v33/reference/v32/results/jobs/polymer_modulus__KRR__r0"
    p44 = source_root/"Interaction_Descriptors_Learning_Curves_v44_20261106/results/jobs"
    for case, method, seed, repeat in r.jobs():
        if method.startswith("fusion"):
            continue
        now = r.ROOT/f"results/jobs/{case}__{method}__{seed}__r{repeat}.npz"
        if not now.exists():
            continue
        if case == "polymer_modulus":
            tag = {"composition":"none__0__baseX", "basis":"none__0__baseXD"}.get(method, f"{method}__{seed}__boundary_pair")
            ref = p33/f"{tag}.npz"
        else:
            meth = {"composition":"X", "basis":"XDM"}.get(method, "all_frozen")
            kind = method if method in r.KINDS else "NN"
            ref = p44/f"{case}__64__{kind}__{seed or 101}__{meth}__{repeat}.npz"
        if not ref.exists():
            raise FileNotFoundError(ref)
        a, b = dict(np.load(now)), dict(np.load(ref))
        if case == "polymer_modulus":
            assert np.array_equal(a["indices"], b["indices"])
            assert np.array_equal(a["fold"], b["fold"])
            assert np.array_equal(a["y"], b["y"])
            err = float(np.max(abs(a["prediction"]-b["prediction"])))
        else:
            order = np.argsort(a["indices"])
            assert np.array_equal(a["y"][order], b["y"])
            err = float(np.max(abs(a["prediction"][order]-b["pred"])))
        comparisons.append(dict(case=case, feature_set=method, seed=seed, repeat=repeat,
                                prediction_events=len(a["prediction"]), max_abs_difference=err,
                                historical_path=str(ref.relative_to(source_root)), historical_sha256=r.sha(ref)))
    r.write_csv(r.ROOT/"results/historical_prediction_replay.csv", comparisons)
    maxerr = max(v["max_abs_difference"]for v in comparisons)
    assert maxerr < 1e-7, maxerr
    return dict(configuration_repeats=len(comparisons), prediction_events=sum(v["prediction_events"]for v in comparisons),
                max_abs_difference=maxerr, tolerance=1e-7,
                scope="saved historical predictions with current Python3.12, same NumPy/SciPy/scikit-learn versions; not full source-model re-execution")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, default=r.ROOT.parent/"archives")
    args = ap.parse_args()
    r.verify_inputs()
    results = dict(krr_numerical_parity=numerical_parity(args.source_root),
                   heldout_perturbation=heldout_perturbation(),
                   selection_audit=audit_completed_results(),
                   historical_prediction_replay=inherited_predictions(args.source_root))
    r.dump(r.ROOT/"results/validation.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
