#!/usr/bin/env python3
"""Reproduce one archived 24-bank source fit in a separate output directory.

Uses the unchanged source_common.fit_at function and the original final-fit
seed/epoch/scaling rules. Does not modify the archived bank, tune a source model,
load target labels, or repeat source OOF diagnostics. Requires external v38 assets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_modeled_property(current, expected, compatible):
    """Report parity only where a source is applicable, including zero rows."""
    import numpy as np
    compatible = np.asarray(compatible, dtype=bool)
    count = int(np.count_nonzero(compatible))
    difference = None
    if count:
        difference = float(np.max(np.abs(np.asarray(current)[compatible] -
                                         np.asarray(expected)[compatible])))
    return {
        "modeled_property_comparison": "compared" if count else "not_applicable_no_compatible_rows",
        "modeled_property_comparison_rows": count,
        "max_modeled_property_difference": difference,
    }


def feature_parity_passed(check, tolerance=1e-8):
    """Keep mask/latent checks even when the property comparison has no rows."""
    if not check["mask_equal"]:
        return False
    for key, value in check.items():
        if not key.startswith("max_"):
            continue
        if value is None:
            if key != "max_modeled_property_difference" or check["modeled_property_comparison_rows"] != 0:
                return False
        elif not value <= tolerance:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source", default="mp_decomp")
    parser.add_argument("--kind", choices=["NN", "YAE", "YVAE"], default="NN")
    parser.add_argument("--seed", type=int, choices=[101, 202, 303], default=101)
    parser.add_argument("--target-features", choices=["metals", "mixed_oxides"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    archive = args.archive_root.resolve()
    output = args.output_dir.resolve()
    if output == archive or archive in output.parents:
        raise SystemExit("The output must be separate from the immutable archive.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit("Output directory must be new or empty.")
    manifest_path = root / "manifests/source_training_assets.json"
    manifest = json.loads(manifest_path.read_text())
    if args.source not in manifest["sources"]:
        raise SystemExit("Source is not in the archived 24-task catalogue.")
    item = manifest["sources"][args.source]
    source_path = archive / item["input_path"]
    model_path = archive / f"models/{args.source}/{args.kind}/seed{args.seed}.npz"
    code_path = root / "reference/source_bank/source_common.py"
    identities = {
        archive / "data/source_catalog.json": manifest["source_catalog_sha256"],
        source_path: item["input_sha256"],
        model_path: item["models"][args.kind][str(args.seed)],
        code_path: manifest["source_common_sha256"],
    }
    if args.target_features:
        target = args.target_features
        target_path = archive / f"data/target_{target}.npz"
        feature_path = archive / f"features/{target}/{args.kind}/seed{args.seed}/{args.source}.npz"
        identities[target_path] = manifest["targets"][target]["input_sha256"]
        identities[feature_path] = manifest["targets"][target]["features"][args.source][args.kind][str(args.seed)]
    missing = [str(path) for path in identities if not path.is_file()]
    if missing:
        raise SystemExit("Missing archived source-fit inputs: " + "; ".join(missing) +
                         ". See PIPELINE.md for the required external assets.")
    for path, expected in identities.items():
        if sha(path) != expected:
            raise SystemExit(f"Archived input/code identity mismatch: {path.name}")

    import numpy as np
    import scipy
    import sklearn
    import torch

    versions = {"python": sys.version, "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                "torch": torch.__version__}
    if (np.__version__, scipy.__version__, sklearn.__version__, torch.__version__) != (
            "2.3.5", "1.17.0", "1.8.0", "2.10.0+cpu"):
        raise SystemExit("Use the original pinned CPU source-fit dependencies; see PIPELINE.md.")
    spec = importlib.util.spec_from_file_location("archived_source_common52", code_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    torch.set_num_threads(1)

    with np.load(source_path, allow_pickle=False) as data:
        X, y, elements = data["X"], data["y"], list(map(str, data["elements"]))
    with np.load(model_path, allow_pickle=False) as data:
        original = dict(data)
    output.mkdir(parents=True, exist_ok=True)
    actual_seed = args.seed + int(item["index"]) * 10000 + 360000
    plan = {"source": args.source, "source_index": item["index"],
            "kind": args.kind, "source_seed": args.seed,
            "actual_seed": actual_seed, "epochs": 200,
            "rows": len(y), "columns": X.shape[1],
            "target_features": args.target_features,
            "training_uses_target_labels": False,
            "manifest_sha256": sha(manifest_path),
            "input_identities": {str(path): expected for path, expected in identities.items()},
            "environment": versions, "comparison_tolerance": 1e-8}
    (output / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    start = time.perf_counter()
    model, preprocessing, _ = module.fit_at(
        X, y, np.arange(len(y)), np.arange(min(7, len(y))),
        args.kind, actual_seed, True, [200])
    weights = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    weights.update(preprocessing)
    latent, prediction = module.apply_np(X, weights)
    weights.update(z_mean=latent.mean(0), z_sd=np.maximum(latent.std(0), 1e-10),
                   p_mean=float(prediction.mean()), p_sd=max(float(prediction.std()), 1e-10))
    np.savez_compressed(output / "retrained_source.npz", **weights)

    if set(weights) != set(original):
        raise RuntimeError("Regenerated model keys differ from archived keys.")
    checks = []
    for key in sorted(weights):
        new, old = np.asarray(weights[key]), np.asarray(original[key])
        if new.shape != old.shape:
            raise RuntimeError(f"Shape mismatch for {key}")
        delta = float(np.max(np.abs(new.astype(float) - old.astype(float))))
        checks.append({"array": key, "shape": list(new.shape),
                       "max_absolute_difference": delta,
                       "exact_array_equal": bool(np.array_equal(new, old))})
    feature_check = None
    if args.target_features:
        with np.load(target_path, allow_pickle=False) as target:
            # Deliberately access X and component names only, never target y.
            target_X = target["X"]
            target_elements = list(map(str, target["input_names"]))
        allowed = set(elements)
        compatible = np.array([all(e in allowed for e, fraction in zip(target_elements, row)
                                   if fraction > 0) for row in target_X])
        mapped = np.zeros((len(target_X), len(elements)))
        for column, element in enumerate(elements):
            if element in target_elements:
                mapped[:, column] = target_X[:, target_elements.index(element)]
        if not np.allclose(mapped[compatible].sum(1), 1, atol=1e-8, rtol=0):
            raise RuntimeError("Compatible complete compositions do not sum to one.")
        z = np.zeros((len(target_X), 8))
        p = np.zeros((len(target_X), 1))
        physical = np.full(len(target_X), np.nan)
        zz, pp = module.apply_np(mapped[compatible], weights)
        z[compatible] = (zz - weights["z_mean"]) / weights["z_sd"]
        p[compatible, 0] = (pp - weights["p_mean"]) / weights["p_sd"]
        physical[compatible] = pp
        np.savez_compressed(output / "regenerated_target_features.npz", z=z, p=p,
                            mask=compatible.astype(float), physical_pred=physical)
        with np.load(feature_path, allow_pickle=False) as old:
            feature_check = {
                "target": args.target_features, "rows": len(target_X),
                "compatible_rows": int(compatible.sum()),
                "latent_coordinates": 8, "mask_equal": bool(np.array_equal(compatible, old["mask"])),
                "max_latent_difference": float(np.max(np.abs(z - old["z"]))),
                "max_standardized_property_difference": float(np.max(np.abs(p - old["p"]))),
            }
            feature_check.update(compare_modeled_property(physical, old["physical_pred"], compatible))
    max_difference = max(check["max_absolute_difference"] for check in checks)
    passed = max_difference <= 1e-8
    if feature_check:
        passed = passed and feature_parity_passed(feature_check)
    unchanged = all(sha(path) == expected for path, expected in identities.items())
    result = {
        "status": "PASS" if passed and unchanged else "NUMERICAL_PARITY_FAILURE",
        "source_fit_count": 1, "source_oof_fits": 0,
        "source": args.source, "kind": args.kind, "source_seed": args.seed,
        "actual_training_seed": actual_seed, "epochs": 200,
        "parameter_and_scaling_arrays_checked": len(checks),
        "max_parameter_scaling_difference": max_difference,
        "all_parameter_arrays_exact": all(check["exact_array_equal"] for check in checks),
        "parameter_checks": checks, "target_feature_check": feature_check,
        "original_assets_unchanged": unchanged,
        "source_training_code_modified": False,
        "target_training_rerun": False,
        "fresh_environment_claim": "See separate environment installation evidence; this script alone does not attest installation.",
        "elapsed_seconds": time.perf_counter() - start, "environment": versions,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "parameter_checks"}, indent=2))
    return 0 if passed and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
