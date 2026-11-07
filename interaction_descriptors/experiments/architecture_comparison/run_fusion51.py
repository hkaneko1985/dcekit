#!/usr/bin/env python3
"""Stage 51 frozen-feature concatenation experiment.

This is an additional development-cohort analysis. Existing benchmark files are
never changed. Source encoders are not fitted here. Boruta is fitted separately
inside every saved inner-training partition and on each outer-training set.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

ROOT = Path(__file__).resolve().parent
SEEDS = (101, 202, 303)
KINDS = ("NN", "YAE", "YVAE", "random")
CASES = ("polymer_modulus", "metals_volume", "oxides_formation")
ELLS = (.25, .5, 1., 2., 4., 8.)
ALPHAS = (1e-4, .01, .1, 1., 10.)
GRID = [{"ell": e, "alpha": a} for e in ELLS for a in ALPHAS]
BORUTA = dict(n_estimators=128, max_iter=100, perc=100, alpha=.05, two_step=True,
              early_stopping=False)
RF = dict(n_estimators=128, max_depth=5, max_features="sqrt", n_jobs=1,
          criterion="squared_error", min_samples_leaf=1, bootstrap=True)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class AuditedBoruta(BorutaPy):
    """Record statistical decisions without changing BorutaPy's tests."""

    def _do_tests(self, dec_reg, hit_reg, iteration):
        result = super()._do_tests(dec_reg, hit_reg, iteration)
        self.statistical_decision_ = result.copy()
        self.hit_counts_ = hit_reg.copy()
        self.iterations_ = int(iteration)
        return result


def prepare(source_root):
    """Copy only the predefined saved inputs, then lock protocol and hashes."""
    source_root = Path(source_root).resolve()
    p33 = source_root / "Interaction_Descriptors_Output_Forms_v33"
    p44 = source_root / "Interaction_Descriptors_Learning_Curves_v44_20261106"
    out = ROOT / "inputs"
    out.mkdir(exist_ok=True)
    if (ROOT / "protocol.json").exists():
        verify_inputs()
        return

    used = {}
    metadata = {}

    def used_file(path):
        used[str(Path(path).relative_to(source_root))] = sha(path)
        return Path(path)

    catpath = used_file(p44 / "data/source_catalog.json")
    cat = json.loads(catpath.read_text())
    sids = [c["source"] for c in cat]
    assert len(sids) == 24

    for case in CASES:
        if case == "polymer_modulus":
            dp = used_file(p33 / "data/polymer_modulus.npz")
            t = dict(np.load(dp))
            sp = used_file(p33 / "data/polymer_modulus_splits_0.json")
            splits = [json.loads(sp.read_text())]
            meta = dict(n=len(t["y"]), unit="MPa", representation="boundary_latent",
                        base="X + D", composition_dim=t["X"].shape[1],
                        fixed_dim=t["D"].shape[1], mask_dim=0,
                        learned_dim_per_kind=24, combined_learned_dim=72,
                        repeats=1, outer_folds_per_repeat=6,
                        aggregation="unweighted mean of six region-held-out fold metrics",
                        source_exclusion="other source properties only; modulus source omitted in inherited v33 features",
                        source_properties=["stress", "strain", "hardness"])
        else:
            dp = used_file(p44 / f"data/target_{case}.npz")
            t = dict(np.load(dp))
            sp = used_file(p44 / f"data/learning_splits_{case}.json")
            splits = json.loads(sp.read_text())["64"]
            meta = dict(n=len(t["y"]), unit=str(t["unit"]), representation="whole_composition_latent",
                        base="X + D + 24 source applicability masks", composition_dim=t["X"].shape[1],
                        fixed_dim=t["D"].shape[1], mask_dim=24,
                        learned_dim_per_kind=192, combined_learned_dim=576,
                        repeats=2, outer_folds_per_repeat=3,
                        aggregation="pooled held-out predictions within each repeat, then mean across repeats and source seeds",
                        source_exclusion="inherited v38/v44 bank excludes target-cohort compositions according to source catalog; no source refit",
                        source_tasks=sids)
        meta["train_sizes"] = sorted({len(f["train"]) for repeat in splits for f in repeat})
        meta["test_sizes"] = sorted({len(f["test"]) for repeat in splits for f in repeat})
        for repeat in splits:
            for fold in repeat:
                split_check(t, fold)
        dump(out / f"{case}_splits.json", splits)
        metadata[case] = meta

        masks = None
        for seed in SEEDS:
            a = {k: t[k] for k in ("X", "D", "y", "ids")}
            a["groups"] = t.get("groups", np.array([str(i) for i in range(len(t["y"]))]))
            if "input_names" in t:
                a["input_names"] = t["input_names"]
            else:
                a["input_names"] = np.array([f"component_{i}" for i in range(t["X"].shape[1])])
            for kind in KINDS:
                if case == "polymer_modulus":
                    fp = used_file(p33 / f"features/{case}__{kind}__{seed}.npz")
                    f = dict(np.load(fp))
                    z = f["boundary_latent"]
                    names = f["latent_names"].astype(str)
                    assert not any("modulus" in n.lower() for n in names)
                    a[f"names_{kind}"] = np.array([f"{kind}:{n}" for n in names])
                else:
                    folder = "metals" if case == "metals_volume" else "mixed_oxides"
                    ff = []
                    for sid in sids:
                        fp = used_file(p44 / f"features/{folder}/{kind}/seed{seed}/{sid}.npz")
                        ff.append(dict(np.load(fp)))
                    mask = np.column_stack([f["mask"] for f in ff])
                    if masks is None:
                        masks = mask
                    else:
                        assert np.array_equal(masks, mask), "applicability masks differ across encoders/seeds"
                    assert np.isin(mask, [0, 1]).all()
                    assert all(np.all(f["z"][f["mask"] == 0] == 0) for f in ff)
                    z = np.concatenate([f["z"] for f in ff], axis=1)
                    a[f"names_{kind}"] = np.array([f"{kind}:{sid}:z{i}" for sid in sids for i in range(8)])
                assert z.shape == (len(t["y"]), meta["learned_dim_per_kind"])
                assert np.isfinite(z).all()
                a[f"Z_{kind}"] = z
            a["M"] = np.zeros((len(t["y"]), 0)) if masks is None else masks
            assert all(np.isfinite(a[k]).all() for k in ("X", "D", "M", "y"))
            assert np.all(a["X"] >= 0)
            np.savez_compressed(out / f"{case}__{seed}.npz", **a)

    dump(out / "metadata.json", metadata)
    dump(ROOT / "source_hashes.json", used)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in sorted(out.glob("*")) if p.is_file()}
    dump(ROOT / "input_hashes.json", inputs)
    proto = dict(
        stage="51", status="additional development-cohort analysis, not independent confirmation",
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        timing_pilot="One training-only selector timing pilot with 19 iterations; no prediction errors were used to choose the protocol",
        cases=metadata, source_seeds=list(SEEDS), final_regressor="KRR",
        krr_grid=GRID, krr_kernel="exp(-sum((a-b)^2)/(p * 2 * ell^2)) after training-only standardization",
        krr_fit="training y centered/scaled; alpha added to kernel diagonal",
        comparison_sets=["composition", "basis", "NN", "YAE", "YVAE", "random", "fusion", "fusion_boruta"],
        baseline_seeds="not duplicated across source seeds",
        concatenation="append NN, YAE, YVAE latent coordinates in this order at a matched source seed; no latent averaging",
        protected_columns="all original X, D and applicability mask columns remain in the candidate basis; numerical constant columns are filtered within each fit as in saved prep",
        boruta=dict(implementation="BorutaPy 0.4.3", parameters=BORUTA, random_forest=RF,
                    selection="statistically confirmed learned coordinates only; unresolved and rejected learned coordinates excluded",
                    fitting="full base+latent candidate matrix, training-only constant removal/standardization; fit separately inside every inner fold and once on outer training",
                    zero_selected="use base only; no forced coordinate fallback",
                    logging="statistical confirmed/unresolved/rejected, weak rough-fix flag, hit counts, full coordinate decisions",
                    random_seed="510000 + case_index*10000 + source_seed*10 + repeat*100 + outer_fold*10 + inner_index; outer refit inner_index=9"),
        selection_order="configuration fixed before target prediction errors from this stage were computed",
        source_overlap="retain historical target-specific exclusions; do not apply latest bank to original Tc/gap targets",
        scope_limits=["KRR-only additional comparison; original six-regressor benchmark remains unchanged",
                      "source models and full source-training data were not retrained/re-audited",
                      "same saved development cohorts and splits are reused",
                      "polymer boundary route and inorganic whole-composition route are evaluated separately",
                      "no claim that latent coordinates identify causal microscopic interactions"],
    )
    dump(ROOT / "protocol.json", proto)
    dump(ROOT / "environment.json", dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), dependency_environment="venv inherits system numpy/scipy/sklearn; not a clean dependency reinstall",
        packages={p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "Boruta", "lightgbm", "joblib", "threadpoolctl"]}))
    print("PREPARED", len(inputs), "inputs", len(used), "historical files", flush=True)


def verify_inputs():
    manifest = json.loads((ROOT / "input_hashes.json").read_text())
    mismatches = [p for p, h in manifest.items() if sha(ROOT / p) != h]
    if mismatches:
        raise RuntimeError(f"Input hashes changed: {mismatches}")


def split_check(t, fold):
    tr, te = set(fold["train"]), set(fold["test"])
    assert tr and te and not tr & te
    groups = t.get("groups")
    if groups is not None:
        assert not set(groups[list(tr)]) & set(groups[list(te)])
    for f in fold["inner"]:
        a, b = set(f["train"]), set(f["valid"])
        assert a and b and not a & b and (a | b) <= tr
        if groups is not None:
            assert not set(groups[list(a)]) & set(groups[list(b)])


def prep(A, B):
    sd = A.std(0)
    keep = sd > 1e-10
    if not keep.any():
        return np.zeros((len(A), 1)), np.zeros((len(B), 1)), keep
    mean = A[:, keep].mean(0)
    scale = sd[keep]
    return (A[:, keep] - mean) / scale, (B[:, keep] - mean) / scale, keep


def predict_grid(A, y, B):
    # Numerically identical KRR grid routine to the saved regression.py.
    mu, sd = y.mean(), max(y.std(), 1e-10)
    v = (y - mu) / sd
    D = cdist(A, A, "sqeuclidean") / A.shape[1]
    Q = cdist(B, A, "sqeuclidean") / A.shape[1]
    out = []
    for ell in ELLS:
        K = np.exp(-D / (2 * ell ** 2))
        T = np.exp(-Q / (2 * ell ** 2))
        ev, U = eigh((K + K.T) / 2, check_finite=False)
        ev = np.maximum(ev, 0)
        pred = (T @ U) @ ((U.T @ v)[:, None] / (ev[:, None] + np.asarray(ALPHAS)[None, :])) * sd + mu
        out.append(pred)
    return np.concatenate(out, axis=1)


def predict_one(A, y, B, config):
    mu, sd = float(y.mean()), max(float(y.std()), 1e-10)
    D = cdist(A, A, "sqeuclidean") / A.shape[1]
    Q = cdist(B, A, "sqeuclidean") / A.shape[1]
    K = np.exp(-D / (2 * config["ell"] ** 2))
    Q = np.exp(-Q / (2 * config["ell"] ** 2))
    return Q @ np.linalg.solve(K + config["alpha"] * np.eye(len(A)), (y - mu) / sd) * sd + mu


def select_features(F, y, train, nbase, names, random_seed):
    start = time.perf_counter()
    train = np.asarray(train, int)
    A = F[train]
    variable = A.std(0) > 1e-10
    indices = np.flatnonzero(variable)
    selection = np.zeros(F.shape[1], dtype=bool)
    selection[:nbase] = True
    decision = np.full(F.shape[1], -2, dtype=int)  # -2 = training-constant, not tested
    weak = np.zeros(F.shape[1], dtype=bool)
    hits = np.zeros(F.shape[1], dtype=int)
    if indices.size:
        aa = A[:, variable]
        aa = (aa - aa.mean(0)) / aa.std(0)
        selector = AuditedBoruta(RandomForestRegressor(**RF), **BORUTA, random_state=random_seed, verbose=0)
        selector.fit(aa, y[train])
        d = selector.statistical_decision_
        decision[indices] = d
        weak[indices] = selector.support_weak_
        hits[indices] = selector.hit_counts_
        learned = indices >= nbase
        selection[indices[learned]] = d[learned] == 1
        iterations = selector.iterations_
    else:
        iterations = 0
    learned = np.arange(F.shape[1]) >= nbase
    info = dict(training_indices=train.tolist(), random_seed=int(random_seed),
                total_candidate_dim=int(F.shape[1]), protected_base_dim=int(nbase),
                variable_candidate_dim=int(variable.sum()), variable_learned_dim=int((variable & learned).sum()),
                confirmed_learned=int(((decision == 1) & learned).sum()),
                unresolved_learned=int(((decision == 0) & learned).sum()),
                rejected_learned=int(((decision == -1) & learned).sum()),
                constant_learned=int(((decision == -2) & learned).sum()),
                weak_rough_fix_learned=int((weak & learned).sum()),
                confirmed_all=int((decision == 1).sum()), unresolved_all=int((decision == 0).sum()),
                rejected_all=int((decision == -1).sum()),
                iterations=iterations, elapsed_seconds=time.perf_counter()-start,
                decisions=decision.tolist(), weak_rough_fix=weak.astype(int).tolist(),
                hit_counts=hits.tolist(), kept_indices=np.flatnonzero(selection).tolist(),
                selected_learned_names=[names[i] for i in np.flatnonzero(selection & learned)])
    assert info["confirmed_learned"] + info["unresolved_learned"] + info["rejected_learned"] + info["constant_learned"] == F.shape[1] - nbase
    return selection, info


def metrics(y, p):
    sse = float(np.sum((p-y)**2))
    sst = float(np.sum((y-y.mean())**2))
    return dict(rmse=float(np.sqrt(np.mean((p-y)**2))), mae=float(np.mean(abs(p-y))),
                q2=1-sse/sst if sst > 1e-15 else None)


def design(d, feature_set):
    base = np.c_[d["X"], d["D"], d["M"]]
    bn = ([f"X:{x}" for x in d["input_names"].astype(str)]
          + [f"D:{i}" for i in range(d["D"].shape[1])]
          + [f"M:{i}" for i in range(d["M"].shape[1])])
    if feature_set == "composition":
        return d["X"], d["X"].shape[1], bn[:d["X"].shape[1]]
    if feature_set == "basis":
        return base, base.shape[1], bn
    kinds = ("NN", "YAE", "YVAE") if feature_set.startswith("fusion") else (feature_set,)
    F = np.c_[base, *(d[f"Z_{k}"] for k in kinds)]
    names = bn + [str(n) for k in kinds for n in d[f"names_{k}"]]
    assert len(names) == F.shape[1]
    return F, base.shape[1], names


def work(job):
    case, feature_set, seed, repeat = job
    stem = f"{case}__{feature_set}__{seed}__r{repeat}"
    dest = ROOT / "results/jobs"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/f"{stem}.json").exists():
        return json.loads((dest/f"{stem}.json").read_text())["summary"]
    start = time.perf_counter()
    d = dict(np.load(ROOT / f"inputs/{case}__{seed or 101}.npz"))
    folds = json.loads((ROOT / f"inputs/{case}_splits.json").read_text())[repeat]
    F, nbase, names = design(d, feature_set)
    y = d["y"]
    pred, rows, folds_idx, infos, foldmetrics = [], [], [], [], []
    selector_fits = 0
    for fi, fold in enumerate(folds):
        split_check(d, fold)
        tr, te = np.asarray(fold["train"], int), np.asarray(fold["test"], int)
        losses, nv, inninfos = np.zeros(len(GRID)), 0, []
        seedbase = 510000 + CASES.index(case)*10000 + seed*10 + repeat*100 + fi*10
        for ii, inner in enumerate(fold["inner"]):
            a, b = np.asarray(inner["train"], int), np.asarray(inner["valid"], int)
            if feature_set == "fusion_boruta":
                keep, audit = select_features(F, y, a, nbase, names, seedbase+ii)
                selector_fits += 1
                assert not set(a) & set(b) and not set(a) & set(te)
            else:
                keep, audit = np.ones(F.shape[1], bool), None
            A, B, variable = prep(F[a][:, keep], F[b][:, keep])
            pp = predict_grid(A, y[a], B)
            losses += ((pp-y[b, None])**2).sum(0)
            nv += len(b)
            inninfos.append(dict(train=a.tolist(), valid=b.tolist(), fitted_dim=int(variable.sum()), selection=audit))
        best = int(np.argmin(losses))
        if feature_set == "fusion_boruta":
            keep, audit = select_features(F, y, tr, nbase, names, seedbase+9)
            selector_fits += 1
        else:
            keep, audit = np.ones(F.shape[1], bool), None
        A, B, variable = prep(F[tr][:, keep], F[te][:, keep])
        p = predict_one(A, y[tr], B, GRID[best])
        assert np.isfinite(p).all()
        fm = metrics(y[te], p)
        foldmetrics.append(dict(fold=fi, key=fold.get("key", str(fi)), n_train=len(tr), n_test=len(te), **fm))
        infos.append(dict(fold=fi, train=tr.tolist(), test=te.tolist(), config=GRID[best],
                          inner_mse=float(losses[best]/nv), inner_grid_mse=(losses/nv).tolist(),
                          fitted_dim=int(variable.sum()), inner=inninfos, selection=audit))
        pred.extend(p)
        rows.extend(te)
        folds_idx.extend([fi]*len(te))
    pred, rows, folds_idx = np.array(pred), np.array(rows, int), np.array(folds_idx, int)
    if case == "polymer_modulus":
        m = {k: float(np.mean([f[k] for f in foldmetrics])) for k in ("rmse", "mae", "q2")}
    else:
        assert np.array_equal(np.sort(rows), np.arange(len(y))), "outer test folds must cover each row exactly once"
        m = metrics(y[rows], pred)
    summary = dict(case=case, feature_set=feature_set, source_seed=seed, repeat=repeat, regressor="KRR",
                   n_samples=len(y), outer_fits=len(folds), inner_fits=sum(len(f["inner"])for f in folds),
                   selector_fits=selector_fits, n_prediction_events=len(pred), candidate_dim=F.shape[1],
                   elapsed_seconds=time.perf_counter()-start, **m)
    np.savez_compressed(dest/f"{stem}.npz", prediction=pred, y=y[rows], indices=rows, fold=folds_idx, ids=d["ids"][rows], groups=d["groups"][rows])
    dump(dest/f"{stem}.json", dict(summary=summary, fold_metrics=foldmetrics, fits=infos, feature_names=names))
    return summary


def jobs():
    out = []
    for case in CASES:
        for repeat in range(1 if case == "polymer_modulus" else 2):
            for meth in ("composition", "basis"):
                out.append((case, meth, 0, repeat))
            for seed in SEEDS:
                for meth in (*KINDS, "fusion", "fusion_boruta"):
                    out.append((case, meth, seed, repeat))
    return out


def analyze():
    expected = jobs()
    found = []
    folds, selections = [], []
    for case, method, seed, repeat in expected:
        path = ROOT / f"results/jobs/{case}__{method}__{seed}__r{repeat}.json"
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        found.append(obj["summary"])
        context = {k: obj["summary"][k] for k in ("case", "feature_set", "source_seed", "repeat")}
        for fm, fit in zip(obj["fold_metrics"], obj["fits"]):
            folds.append(dict(**context, **fm, fitted_dim=fit["fitted_dim"], ell=fit["config"]["ell"], alpha=fit["config"]["alpha"]))
            selected = [("outer", -1, fit["selection"])] + [("inner", i, v["selection"]) for i, v in enumerate(fit["inner"])]
            for level, ii, se in selected:
                if se is None:
                    continue
                selections.append(dict(**context, fold=fit["fold"], level=level, inner_index=ii,
                    **{k:v for k,v in se.items() if k not in ("training_indices", "decisions", "weak_rough_fix", "hit_counts", "kept_indices", "selected_learned_names")}))
    write_csv(ROOT/"results/configuration_metrics.csv", found)
    write_csv(ROOT/"results/fold_metrics.csv", folds)
    write_csv(ROOT/"results/selection_summary.csv", selections)
    summaries = []
    for case in CASES:
        for method in ("composition", "basis", "NN", "YAE", "YVAE", "random", "fusion", "fusion_boruta"):
            ss = [v for v in found if v["case"] == case and v["feature_set"] == method]
            if not ss:
                continue
            se = [v for v in selections if v["case"] == case and v["level"] == "outer"] if method == "fusion_boruta" else []
            summaries.append(dict(case=case, feature_set=method, n_config_repeats=len(ss),
                rmse_mean=float(np.mean([s["rmse"] for s in ss])),
                rmse_sd=float(np.std([s["rmse"]for s in ss],ddof=1))if len(ss)>1 else 0.,
                rmse_min=min(s["rmse"]for s in ss), rmse_max=max(s["rmse"]for s in ss),
                mae_mean=float(np.mean([s["mae"]for s in ss])),
                confirmed_learned_min=min((s["confirmed_learned"]for s in se),default=""),
                confirmed_learned_max=max((s["confirmed_learned"]for s in se),default=""),
                confirmed_learned_mean=float(np.mean([s["confirmed_learned"]for s in se]))if se else ""))
    write_csv(ROOT/"results/summary.csv", summaries)
    status = dict(expected_configuration_repeats=len(expected), completed_configuration_repeats=len(found),
                  outer_fits=sum(s["outer_fits"]for s in found), inner_partitions=sum(s["inner_fits"]for s in found),
                  inner_grid_candidate_predictions=30*sum(s["inner_fits"]for s in found),
                  selector_fits=sum(s["selector_fits"]for s in found),
                  prediction_events=sum(s["n_prediction_events"]for s in found),
                  complete=len(expected)==len(found),
                  protocol_sha256=sha(ROOT/"protocol.json"),
                  input_manifest_sha256=sha(ROOT/"input_hashes.json"))
    dump(ROOT/"results/status.json", status)
    return status, summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=ROOT.parent/"archives")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    prepare(args.source_root)
    if args.prepare_only:
        return
    if args.analyze_only:
        print(json.dumps(analyze()[0], indent=2))
        return
    verify_inputs()
    todo = jobs()
    dump(ROOT/"job_manifest.json", todo)
    t0 = time.perf_counter()
    print("RUN", len(todo), "configuration-repeats", "workers", args.workers, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(work, j):j for j in todo}
        for i, fut in enumerate(as_completed(pending), 1):
            s = fut.result()
            print("DONE", i, "/", len(todo), s["case"], s["feature_set"], s["source_seed"], s["repeat"],
                  "RMSE", round(s["rmse"], 6), "elapsed", round(time.perf_counter()-t0, 1), flush=True)
    verify_inputs()
    status, _ = analyze()
    status["run_wall_seconds"] = time.perf_counter()-t0
    dump(ROOT/"results/status.json", status)
    print("COMPLETE", json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
