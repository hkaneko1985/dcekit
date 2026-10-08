# -*- coding: utf-8 -*-
"""
FTCP-VAE Training and Evaluation Pipeline

Author: Issa Onishi
Created: October 1, 2026
"""

import os
import json
import itertools
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
warnings.filterwarnings("ignore")

from module.utils import minmax_transform, inv_minmax
from module.result_analysis import (
    ensure_dir, MAPE, MAE_site_coor, elem_acc, elem_acc_valid, extract_lattice_coords,
    build_and_load, predict_recon, aggregate_results, find_best_cfg,
)


# ===========================================================================
# SECTION 0: SETTINGS (kept identical to 01_main.py)
# ===========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_NAME = "data_query_3_5_elements_property_nsites_below_112"
MAX_ELMS  = 5
MAX_SITES = 112
PROP      = ["formation_energy_per_atom", "band_gap"]

cnn_patterns_all        = ["0", "1", "2"]
superviseds_all         = ["SVAE", "USVAE"]
network_pattern_numbers = ["0", "1", "2", "3", "4", "5", "6"]

OUT_ROOT = os.path.join(BASE_DIR, "result_for_paper")
ensure_dir(OUT_ROOT)

# Column names/order match the reconstruction-accuracy table format used in
# the manuscript.
TABLE_COLUMNS = [
    "CNN", "VAE type", "network pattern",
    "MAE Ef (eV/atom)", "MAE Eg (eV)", "Element acc.",
    "Lattice const. MAPE (%)", "Lattice angle MAPE (%)", "Atom coord. MAE",
]


# ===========================================================================
# SECTION 1: DATA LOADING (validation + test subsets)
# ===========================================================================

print("=" * 60)
print("Loading data ...")
print("=" * 60)

elm_str     = joblib.load(os.path.join(BASE_DIR, "data/element.pkl"))
Ntotal_elms = len(elm_str)

df_full = pd.read_csv(os.path.join(BASE_DIR, f"./{DATA_NAME}.csv"), index_col=0)


def _cnt(s):
    return len(set(s.replace("[", "").replace("]", "").split(",")))


df_full["n_elements"] = df_full["elements"].apply(_cnt)

# Load the EXACT split indices and the scaler_X / scaler_y fitted on the
# training subset only, as saved by 01_main.py. This script must reuse these
# rather than re-derive them, to avoid normalizing val/test differently from
# how the models saw data during training.
split_path    = os.path.join(BASE_DIR, f"{DATA_NAME}_result", "split_indices.pkl")
scaler_X_path = os.path.join(BASE_DIR, f"{DATA_NAME}_result", "scaler_X.pkl")
scaler_y_path = os.path.join(BASE_DIR, f"{DATA_NAME}_result", "scaler_y.pkl")
for p in (split_path, scaler_X_path, scaler_y_path):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. Run 01_main.py first; it saves the train-only-fitted "
            "scalers and the train/val/test split indices used for training, "
            "which this script must reuse (not re-derive)."
        )

split_indices = joblib.load(split_path)
ind_val       = split_indices["ind_val"]
ind_test      = split_indices["ind_test"]
scaler_X      = joblib.load(scaler_X_path)
scaler_y      = joblib.load(scaler_y_path)

ftcp_full_path = os.path.join(BASE_DIR, f"{DATA_NAME}_result", "FTCP_representation_full.npz")
if not os.path.exists(ftcp_full_path):
    raise FileNotFoundError(
        f"{ftcp_full_path} not found. Run 01_main.py first; it caches the "
        "full-dataset (raw, padded) FTCP representation for reuse here."
    )
npz_full          = np.load(ftcp_full_path)
FTCP_rep_full_raw = npz_full["FTCP_representation"]
Nsites_full       = npz_full["Nsites"]


def _load_subset(ind_subset):
    """Build (X, y, Nsites, n_elements, df_subset) for a given index subset,
    normalized with the training-fitted scalers (transform only, no refit)."""
    df_subset = df_full.iloc[ind_subset].copy().reset_index(drop=True)
    ftcp_rep  = FTCP_rep_full_raw[ind_subset]
    nsites    = Nsites_full[ind_subset]
    X = minmax_transform(ftcp_rep.astype("float32"), scaler_X)
    y = scaler_y.transform(df_subset[PROP].values.astype("float32"))
    n_el = df_subset["n_elements"].values
    return X, y, nsites, n_el, df_subset


X_val,  y_val,  Nsites_val,  n_el_val,  df_val  = _load_subset(ind_val)
X_test, y_test, Nsites_test, n_el_test, df_test = _load_subset(ind_test)

print(f"  Validation: {len(df_val)} total  | 3-4 el: {np.isin(n_el_val, [3, 4]).sum()}  | 5 el: {(n_el_val == 5).sum()}")
print(f"  Test:       {len(df_test)} total | 3-4 el: {np.isin(n_el_test, [3, 4]).sum()} | 5 el: {(n_el_test == 5).sum()}")

X_orig_val  = inv_minmax(X_val,  scaler_X)
X_orig_test = inv_minmax(X_test, scaler_X)
abc_o_val,  ang_o_val,  coor_o_val  = extract_lattice_coords(X_orig_val,  Ntotal_elms, MAX_SITES)
abc_o_test, ang_o_test, coor_o_test = extract_lattice_coords(X_orig_test, Ntotal_elms, MAX_SITES)
y_true_val  = scaler_y.inverse_transform(y_val)
y_true_test = scaler_y.inverse_transform(y_test)

# Bundle everything needed per (val/test) dataset for the evaluation loop.
DATASETS = {
    "val": dict(
        X=X_val, y=y_val, Nsites=Nsites_val, n_el=n_el_val,
        abc_o=abc_o_val, ang_o=ang_o_val, coor_o=coor_o_val,
        X_orig=X_orig_val, y_true=y_true_val,
    ),
    "test": dict(
        X=X_test, y=y_test, Nsites=Nsites_test, n_el=n_el_test,
        abc_o=abc_o_test, ang_o=ang_o_test, coor_o=coor_o_test,
        X_orig=X_orig_test, y_true=y_true_test,
    ),
}

# The four output tables: (dataset key, element-subset label).
TABLE_KEYS = {
    "val_3to4":  ("val",  "3-4"),
    "val_5":     ("val",  "5"),
    "test_3to4": ("test", "3-4"),
    "test_5":    ("test", "5"),
}


# ===========================================================================
# SECTION 2: EVALUATE ALL 42 MODELS ON VAL + TEST, SPLIT BY N_ELEMENTS
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 2: Evaluating all 42 model combinations (val + test, by n_elements)")
print("=" * 60)


def _nan_row(cnn, sup, net):
    return {
        "CNN": cnn, "VAE type": sup, "network pattern": net,
        "MAE Ef (eV/atom)": np.nan, "MAE Eg (eV)": np.nan,
        "Element acc.": np.nan, "Element acc. (valid slots)": np.nan,
        "Lattice const. MAPE (%)": np.nan,
        "Lattice angle MAPE (%)": np.nan, "Atom coord. MAE": np.nan,
    }


rows = {key: [] for key in TABLE_KEYS}

for cnn, sup, net in itertools.product(cnn_patterns_all, superviseds_all, network_pattern_numbers):
    tag    = f"CNN{cnn} {sup} pattern{net}"
    cfg    = {"cnn": cnn, "vae_type": sup, "pattern": net, "label": tag}
    is_sup = (sup == "SVAE")
    print(f"  Evaluating: {tag}")

    computed_rows = None
    try:
        vae = build_and_load(cfg, X_test, y_test, BASE_DIR, DATA_NAME)

        recon = {}
        for dname, d in DATASETS.items():
            X_recon_norm  = predict_recon(vae, cfg, d["X"], d["y"])
            X_recon       = inv_minmax(X_recon_norm, scaler_X)
            y_hat         = scaler_y.inverse_transform(vae.predict_y(d["X"], verbose=0)) if is_sup else None
            recon[dname]  = {"X_recon": X_recon, "y_hat": y_hat}

        computed_rows = {}
        for key, (dname, subset) in TABLE_KEYS.items():
            d       = DATASETS[dname]
            mask    = np.isin(d["n_el"], [3, 4]) if subset == "3-4" else (d["n_el"] == 5)
            X_recon = recon[dname]["X_recon"]
            y_hat   = recon[dname]["y_hat"]
            abc_r, ang_r, coor_r = extract_lattice_coords(X_recon, Ntotal_elms, MAX_SITES)

            mae_ef = float(np.mean(np.abs(d["y_true"][mask, 0] - y_hat[mask, 0]))) if is_sup else np.nan
            mae_eg = float(np.mean(np.abs(d["y_true"][mask, 1] - y_hat[mask, 1]))) if is_sup else np.nan

            computed_rows[key] = {
                "CNN": cnn, "VAE type": sup, "network pattern": net,
                "MAE Ef (eV/atom)": mae_ef,
                "MAE Eg (eV)": mae_eg,
                "Element acc.": elem_acc(d["X_orig"][mask], X_recon[mask], MAX_ELMS, Ntotal_elms),
                "Element acc. (valid slots)": elem_acc_valid(d["X_orig"][mask], X_recon[mask], MAX_ELMS, Ntotal_elms),
                "Lattice const. MAPE (%)": MAPE(d["abc_o"][mask], abc_r[mask]),
                "Lattice angle MAPE (%)": MAPE(d["ang_o"][mask], ang_r[mask]),
                "Atom coord. MAE": MAE_site_coor(d["coor_o"][mask], coor_r[mask], d["Nsites"][mask]),
            }
    except FileNotFoundError as e:
        print(f"    SKIPPED (weights not found / training failed): {e}")
    except Exception as e:
        print(f"    ERROR during evaluation of {tag}: {e}")
    finally:
        tf.keras.backend.clear_session()

    for key in TABLE_KEYS:
        rows[key].append(computed_rows[key] if computed_rows is not None else _nan_row(cnn, sup, net))

TABLE_FILENAMES = {
    "val_3to4":  "TableS1_val_3to4elements.csv",
    "val_5":     "TableS2_val_5elements.csv",
    "test_3to4": "TableS3_test_3to4elements.csv",
    "test_5":    "TableS4_test_5elements.csv",
}

for key, fname in TABLE_FILENAMES.items():
    df_table = pd.DataFrame(rows[key])[TABLE_COLUMNS]
    out_path = os.path.join(OUT_ROOT, fname)
    df_table.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  (shape {df_table.shape})")
    # Supplementary: element accuracy with zero-padded slots excluded
    # (TABLE_COLUMNS / the table above are deliberately left unchanged).
    df_valid = pd.DataFrame(rows[key])[["CNN", "VAE type", "network pattern",
                                        "Element acc.", "Element acc. (valid slots)"]]
    valid_path = os.path.join(OUT_ROOT, fname.replace(".csv", "_element_acc_valid_slots.csv"))
    df_valid.to_csv(valid_path, index=False)
    print(f"  Saved: {valid_path}")


# ===========================================================================
# SECTION 3: BEST MODEL SELECTION (validation set, NOT split by n_elements)
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 3: Best model selection (validation set only)")
print("=" * 60)

result_all = aggregate_results(
    BASE_DIR, DATA_NAME, cnn_patterns_all, superviseds_all, network_pattern_numbers
)

agg_path = os.path.join(BASE_DIR, f"{DATA_NAME}_result_all.csv")
result_all.to_csv(agg_path)
print(f"  Saved: {agg_path}  (shape {result_all.shape})")

# Two independent selections, each ranked only within its own VAE type (see
# module docstring above). Neither is constrained to share the other's
# (CNN, network pattern).
result_svae  = result_all[result_all.index.str.contains(" SVAE ")]
result_usvae = result_all[result_all.index.str.contains(" USVAE ")]

best_svae_tag,  ranking_svae_df  = find_best_cfg(result_svae)
best_usvae_tag, ranking_usvae_df = find_best_cfg(result_usvae)

ranking_svae_df.to_csv(os.path.join(OUT_ROOT, "model_ranking_svae.csv"))
ranking_usvae_df.to_csv(os.path.join(OUT_ROOT, "model_ranking_usvae.csv"))

print(f"\n  Best SVAE model detected (= \"Proposed\", matches Table 3):  {best_svae_tag}")
print(f"  Best USVAE model detected (independent search, Tables 4/5): {best_usvae_tag}")

# best_tag kept as an alias of best_svae_tag: this is "Proposed" throughout
# the manuscript (Table 3 stays SVAE-only; Section 3.1's architecture
# selection is SVAE-based), and 03_compare_conventional_vs_best.py reads
# best_tag under that name for backward compatibility.
best_tag = best_svae_tag

# Conventional baseline configuration (fixed reference model), reused by
# 03_compare_conventional_vs_best.py.
CONV_CFG = {"cnn": "0", "vae_type": "SVAE", "pattern": "0"}
conv_tag = f"CNN{CONV_CFG['cnn']} {CONV_CFG['vae_type']} pattern{CONV_CFG['pattern']}"

best_model_info = {
    "best_tag": best_tag,
    "best_svae_tag": best_svae_tag,
    "best_usvae_tag": best_usvae_tag,
    "conv_tag": conv_tag,
    "conv_cfg": CONV_CFG,
}
best_model_path = os.path.join(OUT_ROOT, "best_model.json")
with open(best_model_path, "w", encoding="utf-8") as f:
    json.dump(best_model_info, f, indent=2, ensure_ascii=False)
print(f"  Saved: {best_model_path}")

print("\nDone.")
