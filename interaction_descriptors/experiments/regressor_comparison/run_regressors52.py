#!/usr/bin/env python3
"""Stage52: the fixed stage51 feature comparison across five additional regressors.

The source encoders and row splits are unchanged. Training-specific Boruta
decisions are reused after validating input and split identities. All original
numerical regression routines and hyperparameter grids are copied unchanged.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import ast
import csv
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REGRESSORS = ("PLS", "LASSO", "SVR", "LightGBM", "GBDT")
ALL_REGRESSORS = ("PLS", "LASSO", "KRR", "SVR", "LightGBM", "GBDT")
CASES = ("polymer_modulus", "metals_volume", "oxides_formation")
FEATURES = ("composition", "basis", "NN", "YAE", "YVAE", "random", "fusion", "fusion_boruta")
SEEDS = (101, 202, 303)
_NUMERICS = None


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_numerics():
    global _NUMERICS
    if _NUMERICS is None:
        spec = importlib.util.spec_from_file_location("archived_regression52", ROOT / "numerics/archived_regression.py")
        _NUMERICS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_NUMERICS)
    return _NUMERICS


def jobs(regressors=REGRESSORS):
    result = []
    # Interleave regressors so timing and completed checkpoints cover all families.
    for case in CASES:
        for repeat in range(1 if case == "polymer_modulus" else 2):
            for feature in ("composition", "basis"):
                for regressor in regressors:
                    result.append((case, feature, 0, repeat, regressor))
            for seed in SEEDS:
                for feature in FEATURES[2:]:
                    for regressor in regressors:
                        result.append((case, feature, seed, repeat, regressor))
    return result


def stem(job, include_regressor=True):
    case, feature, seed, repeat = job[:4]
    name = f"{case}__{feature}__{seed}__r{repeat}"
    return name + ("__" + job[4] if include_regressor else "")


def design(data, feature):
    base = np.c_[data["X"], data["D"], data["M"]]
    names = ([f"X:{name}" for name in data["input_names"].astype(str)]
             + [f"D:{i}" for i in range(data["D"].shape[1])]
             + [f"M:{i}" for i in range(data["M"].shape[1])])
    if feature == "composition":
        return data["X"], data["X"].shape[1], names[:data["X"].shape[1]]
    if feature == "basis":
        return base, base.shape[1], names
    kinds = ("NN", "YAE", "YVAE") if feature.startswith("fusion") else (feature,)
    matrix = np.c_[base, *(data[f"Z_{kind}"] for kind in kinds)]
    names += [str(name) for kind in kinds for name in data[f"names_{kind}"]]
    assert len(names) == matrix.shape[1]
    return matrix, base.shape[1], names


def split_check(data, fold):
    train, test = set(fold["train"]), set(fold["test"])
    assert train and test and not train & test
    assert max(train | test) < len(data["y"]) and min(train | test) >= 0
    assert not set(data["groups"][list(train)]) & set(data["groups"][list(test)])
    for inner in fold["inner"]:
        a, b = set(inner["train"]), set(inner["valid"])
        assert a and b and not a & b and (a | b) <= train
        assert not set(data["groups"][list(a)]) & set(data["groups"][list(b)])


def validate_selector(audit, matrix, y, train, nbase, names):
    train = np.asarray(train, int)
    assert audit["training_indices"] == train.tolist()
    assert audit["total_candidate_dim"] == matrix.shape[1]
    assert audit["protected_base_dim"] == nbase
    decisions = np.asarray(audit["decisions"], int)
    keep = np.asarray(audit["kept_indices"], int)
    assert len(decisions) == matrix.shape[1]
    variable = matrix[train].std(0) > 1e-10
    assert np.array_equal(decisions == -2, ~variable)
    expected = np.r_[np.arange(nbase), np.flatnonzero(decisions[nbase:] == 1) + nbase]
    assert np.array_equal(keep, expected)
    assert audit["confirmed_learned"] == len(keep) - nbase
    assert audit["selected_learned_names"] == [names[i] for i in keep if i >= nbase]
    return dict(training_indices=train.tolist(), training_matrix_sha256=array_sha(matrix[train]),
                training_target_sha256=array_sha(y[train]), feature_names_sha256=object_sha(names),
                total_candidate_dim=matrix.shape[1], protected_base_dim=nbase,
                decisions_sha256=object_sha(audit["decisions"]),
                confirmed_learned=audit["confirmed_learned"], kept_indices=keep.tolist())


def prepare(stage51, regression_source=None):
    stage51 = Path(stage51).resolve()
    if (ROOT / "protocol.json").exists():
        verify_inputs()
        return
    for directory in ("inputs", "reference", "selector_cache", "numerics", "results/jobs"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((stage51 / "fusion/protocol.json").read_text())
    source_inputs = json.loads((stage51 / "fusion/input_hashes.json").read_text())
    provenance = {}
    for relative, expected in source_inputs.items():
        source = stage51 / "fusion" / relative
        assert sha(source) == expected
        destination = ROOT / relative
        shutil.copyfile(source, destination)
        assert sha(destination) == expected
        provenance[relative] = dict(source=str(source), sha256=expected)
    for name in ("protocol.json", "input_hashes.json", "source_hashes.json", "release_annotations.json"):
        source = stage51 / "fusion" / name
        if source.exists():
            shutil.copyfile(source, ROOT / "reference" / ("stage51_" + name))
            provenance["reference/stage51_" + name] = dict(source=str(source), sha256=sha(source))
    for name in ("summary.csv", "configuration_metrics.csv", "status.json", "validation.json"):
        source = stage51 / "fusion/results" / name
        target = ROOT / "reference" / ("krr_" + name)
        shutil.copyfile(source, target)
        provenance[str(target.relative_to(ROOT))] = dict(source=str(source), sha256=sha(source))
    assert json.loads((ROOT / "reference/krr_status.json").read_text())["complete"]

    archival_numerics = stage51 / "archives/Interaction_Descriptors_Learning_Curves_v44_20261106/code/regression.py"
    bundled_numerics = ROOT / "numerics/archived_regression.py"
    regression_source = Path(regression_source or (archival_numerics if archival_numerics.exists() else bundled_numerics))
    source_text = regression_source.read_text()
    v33_source = stage51 / "archives/Interaction_Descriptors_Output_Forms_v33/code/regression.py"
    v33_ast_identical = None
    if v33_source.exists():
        a = [ast.dump(node, include_attributes=False) for node in ast.parse(source_text).body
             if isinstance(node, (ast.FunctionDef, ast.Assign))]
        b = [ast.dump(node, include_attributes=False) for node in ast.parse(v33_source.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.Assign))]
        assert a == b, "v33 and v44 numerical bodies differ"
        v33_ast_identical = True
    v49_tree = ast.parse((stage51 / "sources/v49_replay.py").read_text())
    block = next(ast.literal_eval(node.value) for node in v49_tree.body
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ARCHIVED_NUMERICAL_BLOCK" for target in node.targets))
    assert block == source_text, "v49 archive differs from v44 regression source"
    if regression_source.resolve() != bundled_numerics.resolve():
        shutil.copyfile(regression_source, bundled_numerics)
    provenance["numerics/archived_regression.py"] = dict(source=str(regression_source), sha256=sha(regression_source),
                                                        v33_numerical_ast_identical=v33_ast_identical, v49_block_byte_identical=True)
    numerics = load_numerics()
    assert numerics.GRIDS["KRR"] == source_protocol["krr_grid"]

    selector_records = []
    for job in jobs(("KRR",)):
        if job[1] != "fusion_boruta":
            continue
        case, _, seed, repeat, _ = job
        data = dict(np.load(ROOT / f"inputs/{case}__{seed}.npz", allow_pickle=False))
        matrix, nbase, names = design(data, "fusion_boruta")
        folds = json.loads((ROOT / f"inputs/{case}_splits.json").read_text())[repeat]
        source = stage51 / "fusion/results/jobs" / (stem(job, False) + ".json")
        old = json.loads(source.read_text())
        assert old["feature_names"] == names
        assert len(old["fits"]) == len(folds)
        records = []
        for fi, (fold, old_fit) in enumerate(zip(folds, old["fits"])):
            split_check(data, fold)
            assert old_fit["train"] == fold["train"] and old_fit["test"] == fold["test"]
            assert len(old_fit["inner"]) == len(fold["inner"])
            selectors = []
            for ii, (inner, old_inner) in enumerate(zip(fold["inner"], old_fit["inner"])):
                assert old_inner["train"] == inner["train"] and old_inner["valid"] == inner["valid"]
                checked = validate_selector(old_inner["selection"], matrix, data["y"], inner["train"], nbase, names)
                checked.update(level="inner", inner_index=ii)
                selectors.append(dict(identity=checked, original_selection=old_inner["selection"]))
                selector_records.append(dict(case=case, source_seed=seed, repeat=repeat, outer_fold=fi, **checked))
            checked = validate_selector(old_fit["selection"], matrix, data["y"], fold["train"], nbase, names)
            checked.update(level="outer", inner_index=-1)
            selector_records.append(dict(case=case, source_seed=seed, repeat=repeat, outer_fold=fi, **checked))
            records.append(dict(outer_fold=fi, train=fold["train"], test=fold["test"],
                                inner=selectors, outer=dict(identity=checked, original_selection=old_fit["selection"])))
        target = ROOT / "selector_cache" / (stem(job, False) + ".json")
        dump(target, dict(source_job_sha256=sha(source), source_job=str(source),
                         input_sha256=sha(ROOT / f"inputs/{case}__{seed}.npz"),
                         splits_sha256=sha(ROOT / f"inputs/{case}_splits.json"),
                         source_protocol_sha256=sha(stage51 / "fusion/protocol.json"),
                         feature_names=names, fits=records))
        provenance[str(target.relative_to(ROOT))] = dict(source=str(source), original_job_sha256=sha(source), sha256=sha(target))
    assert len(selector_records) == 198
    dump(ROOT / "selector_reuse_audit.json", dict(
        status="PASS", unique_reused_training_specific_selectors=len(selector_records),
        new_selector_fits=0, exact_ordered_train_rows=True, train_only=True,
        protected_base_preserved=True, confirmed_only=True,
        zero_confirmed_selector_fits=sum(item["confirmed_learned"] == 0 for item in selector_records),
        records=selector_records))
    dump(ROOT / "provenance.json", provenance)
    required_files = [p for directory in ("inputs", "reference", "selector_cache", "numerics")
                      for p in (ROOT / directory).rglob("*") if p.is_file()]
    required_files += [ROOT / "selector_reuse_audit.json", ROOT / "provenance.json"]
    manifest = {str(path.relative_to(ROOT)): sha(path) for path in sorted(required_files)}
    dump(ROOT / "input_hashes.json", manifest)
    dump(ROOT / "job_manifest.json", jobs())
    dump(ROOT / "protocol.json", dict(
        stage=52, frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="additional analysis on previously used development cohorts; no independent external cohort",
        stage51_protocol_sha256=sha(stage51 / "fusion/protocol.json"),
        input_manifest_sha256=sha(ROOT / "input_hashes.json"),
        job_manifest_sha256=sha(ROOT / "job_manifest.json"),
        regression_source_sha256=sha(ROOT / "numerics/archived_regression.py"),
        numerical_implementation=("byte-identical archived regression.py; v49 embedded block matches byte-for-byte; "
                                  + ("v33 and v44 function ASTs checked and identical" if v33_ast_identical
                                     else "v33 numerical AST not rechecked in this preparation")),
        new_regressors=list(REGRESSORS), reused_regressor="KRR",
        grids={method: numerics.GRIDS[method] for method in ALL_REGRESSORS},
        cases=source_protocol["cases"], source_seeds=list(SEEDS), feature_sets=list(FEATURES),
        selectors=dict(new_fits=0, unique_reused_fits=198,
                       source="fixed stage51 nested Boruta fits, independently of final regressor",
                       guards="exact full input hashes, feature order, ordered train/validation/test indices, train-matrix and train-target hashes, confirmed decisions, protected base"),
        preprocessing="training-only constant filtering and X/y standardization, independently inside each inner split and outer refit",
        candidate_selection="argmin of pooled inner-validation squared error; first grid entry on exact ties",
        aggregation=source_protocol["cases"],
        scheduled_new_configuration_repeat_jobs=500, scheduled_new_outer_fits=1800,
        scheduled_new_inner_partitions=4800, scheduled_new_prediction_events=62700,
        reused_krr_configuration_repeat_jobs=100, reused_krr_outer_fits=360,
        reused_krr_prediction_events=12540,
        scope_limits=["same development datasets and held-out splits; stage52 is not an independent confirmation",
                      "no new source training or source selection", "no dimension-matched three-family random concatenation control",
                      "all mean error differences are descriptive, not tests of significance",
                      "polymer and inorganic descriptor routes remain different and explicitly labeled",
                      "Boruta mask reuse is not a new selector fit or independent selection replication"],
    ))
    dump(ROOT / "environment.json", dict(
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        package_versions={package: importlib.metadata.version(package) for package in
                          ("numpy", "scipy", "scikit-learn", "lightgbm", "joblib", "threadpoolctl")},
        dependency_environment="existing stage51 Python3.12 venv with system dependencies; not a fresh package reinstall",
        blas_thread_environment={key: os.environ.get(key) for key in
                                 ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}))
    print("PREPARED", len(jobs()), "jobs; 198 verified selector records; protocol", sha(ROOT / "protocol.json"), flush=True)


def verify_inputs():
    manifest = json.loads((ROOT / "input_hashes.json").read_text())
    bad = [relative for relative, expected in manifest.items() if sha(ROOT / relative) != expected]
    assert not bad, f"Inputs changed: {bad}"
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert sha(ROOT / "input_hashes.json") == protocol["input_manifest_sha256"]
    assert sha(ROOT / "job_manifest.json") == protocol["job_manifest_sha256"]


def selected_columns(cached, matrix, y, train, nbase, names):
    checked = validate_selector(cached["original_selection"], matrix, y, train, nbase, names)
    identity = cached["identity"]
    assert all(checked[key] == identity[key] for key in checked), "Cached selector identity mismatch"
    return np.asarray(identity["kept_indices"], int), dict(
        reused=True, level=identity["level"], inner_index=identity["inner_index"],
        training_indices=identity["training_indices"],
        training_matrix_sha256=identity["training_matrix_sha256"],
        training_target_sha256=identity["training_target_sha256"],
        decisions_sha256=identity["decisions_sha256"],
        kept_indices=identity["kept_indices"], confirmed_learned=identity["confirmed_learned"],
        protected_base_dim=nbase)


def metrics(y, prediction):
    sse = float(np.sum((prediction - y) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return dict(rmse=float(np.sqrt(np.mean((prediction - y) ** 2))),
                mae=float(np.mean(abs(prediction - y))), q2=1 - sse / sst if sst > 1e-15 else None)


def work(job):
    case, feature, seed, repeat, regressor = job
    start = time.perf_counter()
    destination = ROOT / "results/jobs"
    json_path = destination / (stem(job) + ".json")
    npz_path = destination / (stem(job) + ".npz")
    protocol_hash = sha(ROOT / "protocol.json")
    input_hash = sha(ROOT / "input_hashes.json")
    if json_path.exists():
        existing = json.loads(json_path.read_text())
        assert existing["protocol_sha256"] == protocol_hash
        assert existing["input_manifest_sha256"] == input_hash
        assert npz_path.exists() and sha(npz_path) == existing["prediction_file_sha256"]
        return existing["summary"]
    data = dict(np.load(ROOT / f"inputs/{case}__{seed or 101}.npz", allow_pickle=False))
    folds = json.loads((ROOT / f"inputs/{case}_splits.json").read_text())[repeat]
    matrix, nbase, names = design(data, feature)
    y = data["y"]
    numerics = load_numerics()
    grid = numerics.GRIDS[regressor]
    cache_path = ROOT / "selector_cache" / (stem(job, False) + ".json")
    cache = json.loads(cache_path.read_text()) if feature == "fusion_boruta" else None
    if cache is not None:
        assert cache["feature_names"] == names
        assert cache["input_sha256"] == sha(ROOT / f"inputs/{case}__{seed}.npz")
        assert cache["splits_sha256"] == sha(ROOT / f"inputs/{case}_splits.json")
    all_prediction, all_rows, all_folds, fits, fold_metrics = [], [], [], [], []
    selector_reuses = 0
    for fi, fold in enumerate(folds):
        split_check(data, fold)
        train, test = np.asarray(fold["train"], int), np.asarray(fold["test"], int)
        losses, nvalid, inner_infos = np.zeros(len(grid)), 0, []
        for ii, inner in enumerate(fold["inner"]):
            a, b = np.asarray(inner["train"], int), np.asarray(inner["valid"], int)
            if cache is not None:
                keep, selection = selected_columns(cache["fits"][fi]["inner"][ii], matrix, y, a, nbase, names)
                selector_reuses += 1
            else:
                keep, selection = np.arange(matrix.shape[1]), None
            A, B, variable = numerics.prep(matrix[a][:, keep], matrix[b][:, keep])
            predictions, infos = numerics.predict_grid(A, y[a], B, regressor)
            assert predictions.shape == (len(b), len(grid)) and np.isfinite(predictions).all()
            sse = ((predictions - y[b, None]) ** 2).sum(0)
            losses += sse
            nvalid += len(b)
            inner_infos.append(dict(train=a.tolist(), valid=b.tolist(), fitted_dim=int(variable.sum()),
                                    selection=selection, grid_sse=sse.tolist(), grid_fit_info=infos))
        best = int(np.argmin(losses))
        if cache is not None:
            keep, selection = selected_columns(cache["fits"][fi]["outer"], matrix, y, train, nbase, names)
            selector_reuses += 1
        else:
            keep, selection = np.arange(matrix.shape[1]), None
        A, B, variable = numerics.prep(matrix[train][:, keep], matrix[test][:, keep])
        prediction, final_info = numerics.run_one(A, y[train], B, regressor, grid[best])
        assert np.isfinite(prediction).all()
        fold_metrics.append(dict(fold=fi, key=fold.get("key", str(fi)), n_train=len(train), n_test=len(test),
                                 **metrics(y[test], prediction)))
        fits.append(dict(fold=fi, train=train.tolist(), test=test.tolist(),
                         config=grid[best], selected_grid_index=best,
                         inner_mse=float(losses[best] / nvalid), inner_grid_mse=(losses / nvalid).tolist(),
                         fitted_dim=int(variable.sum()), selection=selection, inner=inner_infos,
                         final_info=final_info))
        all_prediction.extend(prediction)
        all_rows.extend(test)
        all_folds.extend([fi] * len(test))
    prediction, rows, fold_ids = np.asarray(all_prediction), np.asarray(all_rows, int), np.asarray(all_folds, int)
    if case == "polymer_modulus":
        values = {key: float(np.mean([item[key] for item in fold_metrics])) for key in ("rmse", "mae", "q2")}
    else:
        assert np.array_equal(np.sort(rows), np.arange(len(y)))
        values = metrics(y[rows], prediction)
    summary = dict(case=case, feature_set=feature, source_seed=seed, repeat=repeat, regressor=regressor,
                   provenance="new stage52 fit", n_samples=len(y), outer_fits=len(folds),
                   inner_fits=sum(len(fold["inner"]) for fold in folds),
                   inner_candidate_fits=len(grid) * sum(len(fold["inner"]) for fold in folds),
                   new_selector_fits=0, selector_reuses=selector_reuses,
                   n_prediction_events=len(prediction), candidate_dim=matrix.shape[1],
                   elapsed_seconds=time.perf_counter() - start, **values)
    temporary = destination / (stem(job) + ".tmp.npz")
    np.savez_compressed(temporary, prediction=prediction, y=y[rows], indices=rows,
                        fold=fold_ids, ids=data["ids"][rows], groups=data["groups"][rows])
    temporary.replace(npz_path)
    dump(json_path, dict(summary=summary, fold_metrics=fold_metrics, fits=fits, feature_names=names,
                         protocol_sha256=protocol_hash, input_manifest_sha256=input_hash,
                         prediction_file_sha256=sha(npz_path),
                         selector_cache_sha256=sha(cache_path) if cache else None))
    return summary


def analyze():
    previous_status = (json.loads((ROOT / "results/status.json").read_text())
                       if (ROOT / "results/status.json").exists() else {})
    new, folds, warnings = [], [], []
    for job in jobs():
        path = ROOT / "results/jobs" / (stem(job) + ".json")
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        new.append(obj["summary"])
        context = {key: obj["summary"][key] for key in ("case", "feature_set", "source_seed", "repeat", "regressor")}
        for metric, fit in zip(obj["fold_metrics"], obj["fits"]):
            folds.append(dict(**context, **metric, fitted_dim=fit["fitted_dim"],
                              selected_grid_index=fit["selected_grid_index"], config=json.dumps(fit["config"], sort_keys=True)))
            for level, ii, ci, info in [("outer", -1, fit["selected_grid_index"], fit["final_info"])] + [
                    ("inner", ii, ci, info) for ii, inner in enumerate(fit["inner"])
                    for ci, info in enumerate(inner["grid_fit_info"])]:
                if info.get("warnings") or info.get("solver") == "LARS+dualQP":
                    warnings.append(dict(**context, fold=fit["fold"], level=level, inner_index=ii,
                                         candidate=ci, warnings=json.dumps(info.get("warnings", [])),
                                         solver=info.get("solver", ""), dual_gap=info.get("dual_gap", ""),
                                         dual_qp_success=info.get("dual_qp_success", "")))
    reference = read_csv(ROOT / "reference/krr_configuration_metrics.csv")
    for row in reference:
        for key in ("source_seed", "repeat", "n_samples", "outer_fits", "inner_fits", "selector_fits", "n_prediction_events", "candidate_dim"):
            row[key] = int(row[key])
        for key in ("elapsed_seconds", "rmse", "mae", "q2"):
            row[key] = float(row[key]) if row[key] else None
        row["provenance"] = "reused stage51 KRR result"
    all_rows = new + reference
    write_csv(ROOT / "results/configuration_metrics.csv", all_rows)
    write_csv(ROOT / "results/fold_metrics_new.csv", folds)
    write_csv(ROOT / "results/numerical_diagnostics.csv", warnings)
    summaries = []
    for case in CASES:
        for regressor in ALL_REGRESSORS:
            for feature in FEATURES:
                selected = [row for row in all_rows if row["case"] == case and row["regressor"] == regressor and row["feature_set"] == feature]
                if not selected:
                    continue
                rmses = [row["rmse"] for row in selected]
                summaries.append(dict(case=case, regressor=regressor, feature_set=feature,
                                      provenance=selected[0]["provenance"], n_config_repeats=len(selected),
                                      rmse_mean=float(np.mean(rmses)), rmse_sd=float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
                                      rmse_min=min(rmses), rmse_max=max(rmses),
                                      mae_mean=float(np.mean([row["mae"] for row in selected])),
                                      q2_mean=float(np.mean([row["q2"] for row in selected]))))
    write_csv(ROOT / "results/summary.csv", summaries)
    comparisons = []
    for case in CASES:
        for regressor in ALL_REGRESSORS:
            table = {row["feature_set"]: row for row in summaries if row["case"] == case and row["regressor"] == regressor}
            if len(table) != 8:
                continue
            best_single = min(("NN", "YAE", "YVAE"), key=lambda key: table[key]["rmse_mean"])
            best = min(FEATURES, key=lambda key: table[key]["rmse_mean"])
            base = table["basis"]["rmse_mean"]
            concat = table["fusion"]["rmse_mean"]
            boruta = table["fusion_boruta"]["rmse_mean"]
            single = table[best_single]["rmse_mean"]
            comparisons.append(dict(case=case, regressor=regressor, best_feature_set=best,
                                    best_rmse=table[best]["rmse_mean"], best_single=best_single, best_single_rmse=single,
                                    composition_rmse=table["composition"]["rmse_mean"], basis_rmse=base,
                                    fusion_rmse=concat, fusion_boruta_rmse=boruta,
                                    fusion_improvement_percent_vs_B=100 * (base - concat) / base,
                                    fusion_improvement_percent_vs_best_single=100 * (single - concat) / single,
                                    boruta_improvement_percent_vs_B=100 * (base - boruta) / base,
                                    boruta_improvement_percent_vs_fusion=100 * (concat - boruta) / concat,
                                    boruta_improvement_percent_vs_best_single=100 * (single - boruta) / single))
    write_csv(ROOT / "results/comparisons.csv", comparisons)
    status = dict(expected_new_jobs=500, completed_new_jobs=len(new),
                  new_outer_fits=sum(row["outer_fits"] for row in new),
                  new_inner_partitions=sum(row["inner_fits"] for row in new),
                  new_inner_candidate_fits=sum(row["inner_candidate_fits"] for row in new),
                  new_prediction_events=sum(row["n_prediction_events"] for row in new),
                  new_selector_fits=0, unique_reused_selectors=198,
                  selector_applications_across_new_regressors=sum(row["selector_reuses"] for row in new),
                  reused_krr_jobs=len(reference), reused_krr_outer_fits=sum(row["outer_fits"] for row in reference),
                  reused_krr_prediction_events=sum(row["n_prediction_events"] for row in reference),
                  combined_summary_rows=len(summaries), complete=len(new) == 500,
                  protocol_sha256=sha(ROOT / "protocol.json"), input_manifest_sha256=sha(ROOT / "input_hashes.json"),
                  by_regressor={method: dict(jobs=sum(row["regressor"] == method for row in new),
                                            elapsed_job_seconds=sum(row["elapsed_seconds"] for row in new if row["regressor"] == method))
                                for method in REGRESSORS})
    # Reaggregation must retain the actual execution receipt rather than erase it.
    for key in ("invocation_wall_seconds", "failures_in_invocation", "cached_jobs_at_invocation_start",
                "new_jobs_completed_this_invocation", "finished_utc"):
        if key in previous_status:
            status[key] = previous_status[key]
    dump(ROOT / "results/status.json", status)
    return status, summaries, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage51-root", type=Path, default=ROOT.parents[1] / "stage51")
    parser.add_argument("--regression-source", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    prepare(args.stage51_root, args.regression_source)
    verify_inputs()
    if args.prepare_only:
        return
    if args.analyze_only:
        print(json.dumps(analyze()[0], indent=2))
        return
    started = time.perf_counter()
    schedule = jobs()
    cached_at_start = sum((ROOT / "results/jobs" / (stem(job) + ".json")).exists() for job in schedule)
    errors = []
    print("RUN", len(schedule), "jobs; workers", args.workers, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(work, job): job for job in schedule}
        for index, future in enumerate(as_completed(pending), 1):
            job = pending[future]
            try:
                result = future.result()
                print("DONE", index, "/", len(schedule), stem(job), "seconds", round(result["elapsed_seconds"], 2),
                      "wall", round(time.perf_counter() - started, 1), flush=True)
            except Exception as error:
                failure = dict(job=list(job), exception=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                errors.append(failure)
                dump(ROOT / "results" / "failures" / (stem(job) + ".json"), failure)
                print("FAILED", stem(job), type(error).__name__, str(error), flush=True)
            if index % 25 == 0:
                status = analyze()[0]
                dump(ROOT / "results/progress.json", dict(**status, elapsed_run_wall_seconds=time.perf_counter() - started,
                                                          failed_jobs_in_this_invocation=len(errors)))
    verify_inputs()
    status, _, _ = analyze()
    status["invocation_wall_seconds"] = time.perf_counter() - started
    status["failures_in_invocation"] = len(errors)
    status["cached_jobs_at_invocation_start"] = cached_at_start
    status["new_jobs_completed_this_invocation"] = status["completed_new_jobs"] - cached_at_start
    status["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    dump(ROOT / "results/status.json", status)
    print("FINISHED", json.dumps(status), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
