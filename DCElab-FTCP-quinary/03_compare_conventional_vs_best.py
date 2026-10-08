# -*- coding: utf-8 -*-
"""
FTCP-VAE Training and Evaluation Pipeline

Author: Issa Onishi
Created: October 1, 2026
"""

import os, re, glob, warnings, joblib, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
warnings.filterwarnings("ignore")

from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

from module.utils import minmax_transform, inv_minmax
from module.result_analysis import (
    sanitize, ensure_dir, safe_mean, savefig, MAPE, MAE_site_coor, elem_acc, elem_acc_valid,
    extract_lattice_coords, slot_accs, element_level_accuracy,
    get_bond_lengths, bond_mae, compute_bond_errors_5el_all, tag_to_cfg,
    build_and_load, predict_recon, get_model_display_name,
    write_single_cif_from_ftcp, plot_metric_breakdown_bar,
    compute_per_sample_ftcp_errors, add_composite_score,
    generate_ranked_cif_pairs, plot_best_worst_tables,
    MODEL_COLOR,
)


# ===========================================================================
# SECTION 0: SETTINGS (kept identical to 01_main.py / 02_evaluate_all_models.py)
# ===========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_NAME    = "data_query_3_5_elements_property_nsites_below_112"
MAX_ELMS     = 5
MAX_SITES    = 112
RANDOM_STATE = 21
PROP         = ["formation_energy_per_atom", "band_gap"]

# Number of samples per n_elements for the 100-sample CIF set
N_PER_NELEM = {3: 35, 4: 35, 5: 30}

OUT_ROOT = os.path.join(BASE_DIR, "result_for_paper")
os.makedirs(OUT_ROOT, exist_ok=True)

# Target bond pairs for bond-length error analysis
TARGET_BOND_PAIRS_RAW = [
    ("Fe", "O"), ("Co", "O"), ("Ti", "O"), ("Mn", "O"),
    ("Ni", "O"), ("Cu", "O"), ("Li", "O"), ("Al", "O"),
    ("Fe", "Fe"), ("Co", "Co"), ("Mn", "Mn"),
]
TARGET_BOND_PAIRS = [tuple(sorted(p)) for p in TARGET_BOND_PAIRS_RAW]

METRICS_IDX = [
    "MAE Ef (eV)",
    "MAE Eg (eV)",
    "Element accuracy",
    "Lattice constant MAPE (%)",
    "Lattice angle MAPE (%)",
    "Site coordinate MAE (frac)",
]

# Global font size settings
FS = {"base": 22, "tick": 20, "legend": 20, "title": 22}
plt.rcParams.update({
    "font.size":        FS["base"],
    "axes.labelsize":   FS["base"],
    "xtick.labelsize":  FS["tick"],
    "ytick.labelsize":  FS["tick"],
    "legend.fontsize":  FS["legend"],
    "axes.titlesize":   FS["title"],
})

try:
    import seaborn as sns
    sns.set_style("ticks")
except ImportError:
    pass


# ===========================================================================
# HELPER: load CIF pairs from a directory tree (shared by Section 6)
# ===========================================================================

def _load_cif_pairs_from_dir(cif_root, run_tag):
    """Load original/reconstructed CIF pairs from the 100-sample directory."""
    pairs = []
    cif_dir = os.path.join(cif_root, sanitize(run_tag))
    for n_el in [3, 4, 5]:
        el_dir = os.path.join(cif_dir, f"{n_el}_elements")
        if not os.path.isdir(el_dir):
            continue
        for orig_path in sorted(glob.glob(os.path.join(el_dir, "original_*.cif"))):
            fname      = os.path.basename(orig_path)
            recon_path = os.path.join(el_dir, "reconstructed_" + fname.replace("original_", ""))
            if not os.path.exists(recon_path):
                continue
            try:
                so = Structure.from_file(orig_path)
                sr = Structure.from_file(recon_path)
            except Exception:
                continue
            idx_m   = re.search(r"idx(\d+)\.cif$", fname)
            idx_val = int(idx_m.group(1)) if idx_m else -1
            pairs.append({"idx": idx_val, "n_el": n_el, "struct_orig": so, "struct_recon": sr})
    return pairs


# ===========================================================================
# SECTION 1: DATA LOADING (test subset only) + BEST MODEL FROM 02's OUTPUT
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
ind_test      = split_indices["ind_test"]
scaler_X      = joblib.load(scaler_X_path)
scaler_y      = joblib.load(scaler_y_path)

df_test = df_full.iloc[ind_test].copy().reset_index(drop=True)

ftcp_full_path = os.path.join(BASE_DIR, f"{DATA_NAME}_result", "FTCP_representation_full.npz")
if not os.path.exists(ftcp_full_path):
    raise FileNotFoundError(
        f"{ftcp_full_path} not found. Run 01_main.py first; it caches the "
        "full-dataset (raw, padded) FTCP representation for reuse here."
    )
npz_full          = np.load(ftcp_full_path)
FTCP_rep_full_raw = npz_full["FTCP_representation"]
Nsites_full       = npz_full["Nsites"]

FTCP_rep    = FTCP_rep_full_raw[ind_test]
Nsites_test = Nsites_full[ind_test]
X_test = minmax_transform(FTCP_rep.astype("float32"), scaler_X)
y_test = scaler_y.transform(df_test[PROP].values.astype("float32"))

n_el_test = df_test["n_elements"].values
mask_34   = np.isin(n_el_test, [3, 4])
mask_5    = (n_el_test == 5)
mask_all  = np.ones(len(df_test), dtype=bool)

X_orig_global = inv_minmax(X_test, scaler_X)
abc_o_global, ang_o_global, coor_o_global = extract_lattice_coords(
    X_orig_global, Ntotal_elms, MAX_SITES
)
y_true_global = scaler_y.inverse_transform(y_test)

print(f"  Test: {len(df_test)} total  | 3-4 el: {mask_34.sum()}  | 5 el: {mask_5.sum()}")

# Load the best/conventional model tags selected by 02_evaluate_all_models.py
best_model_path = os.path.join(OUT_ROOT, "best_model.json")
if not os.path.exists(best_model_path):
    raise FileNotFoundError(
        f"{best_model_path} not found. Run 02_evaluate_all_models.py first; "
        "it selects the best model (from validation-set metrics) and saves "
        "the choice here for this script to reuse."
    )
import json
with open(best_model_path, "r", encoding="utf-8") as f:
    best_model_info = json.load(f)

best_tag       = best_model_info["best_tag"]         # == best_svae_tag; "Proposed" throughout
best_svae_tag  = best_model_info.get("best_svae_tag", best_tag)
best_usvae_tag = best_model_info.get("best_usvae_tag")
conv_tag       = best_model_info["conv_tag"]
print(f"\n  Conventional model: {conv_tag}")
print(f"  Best SVAE model (= \"Proposed\"): {best_svae_tag}")
print(f"  Best USVAE model (Tables 4/5 only): {best_usvae_tag}")

conv_cfg_key = conv_tag
# best_usvae_tag is included here too so Section 3's breakdown computes it
# alongside conv/best-SVAE, for the Table 4/5 summary further below.
all_eval_tags = list(dict.fromkeys([conv_cfg_key, best_tag, best_usvae_tag]))
all_eval_tags = [t for t in all_eval_tags if t]  # drop None if best_usvae_tag missing


# ===========================================================================
# SECTION 3: RECONSTRUCTION ACCURACY BY n_elements
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 3: Reconstruction accuracy breakdown (3-4 vs 5 elements)")
print("=" * 60)

out_bd = ensure_dir(os.path.join(OUT_ROOT, "fig_breakdown"))

print(f"  Models to evaluate ({len(all_eval_tags)}):")
for t in all_eval_tags:
    print(f"    {t}")

recon_cache  = {}
latent_cache = {}
pred_y_cache = {}
breakdown    = {}

for tag in all_eval_tags:
    cfg = tag_to_cfg(tag)
    print(f"\n  Evaluating: {tag}")
    try:
        vae          = build_and_load(cfg, X_test, y_test, BASE_DIR, DATA_NAME)
        X_recon_norm = predict_recon(vae, cfg, X_test, y_test)
        X_recon      = inv_minmax(X_recon_norm, scaler_X)
        abc_r, ang_r, coor_r = extract_lattice_coords(X_recon, Ntotal_elms, MAX_SITES)

        sup   = (cfg["vae_type"] == "SVAE")
        y_hat = scaler_y.inverse_transform(vae.predict_y(X_test, verbose=0)) if sup else None
        z     = vae.compress_to_latent(X_test, verbose=0)

        res = {}
        for mask, lbl in [(mask_34, "3-4"), (mask_5, "5")]:
            mae_ef = float(np.mean(np.abs(y_true_global[mask, 0] - y_hat[mask, 0]))) if sup else np.nan
            mae_eg = float(np.mean(np.abs(y_true_global[mask, 1] - y_hat[mask, 1]))) if sup else np.nan
            res[lbl] = {
                "MAE Ef (eV)":                mae_ef,
                "MAE Eg (eV)":                mae_eg,
                "Element accuracy":           elem_acc(X_orig_global[mask], X_recon[mask], MAX_ELMS, Ntotal_elms),
                "Element accuracy (valid slots)": elem_acc_valid(X_orig_global[mask], X_recon[mask], MAX_ELMS, Ntotal_elms),
                "Lattice constant MAPE (%)":  MAPE(abc_o_global[mask], abc_r[mask]),
                "Lattice angle MAPE (%)":     MAPE(ang_o_global[mask], ang_r[mask]),
                "Site coordinate MAE (frac)": MAE_site_coor(coor_o_global[mask], coor_r[mask], Nsites_test[mask]),
            }

        breakdown[tag]    = res
        recon_cache[tag]  = X_recon
        latent_cache[tag] = z
        pred_y_cache[tag] = y_hat
    except Exception as e:
        print(f"    Error on {tag}: {e}")
    finally:
        tf.keras.backend.clear_session()

pivot_34 = pd.DataFrame({tag: breakdown[tag]["3-4"] for tag in breakdown}).T
pivot_5  = pd.DataFrame({tag: breakdown[tag]["5"]   for tag in breakdown}).T
pivot_34.to_csv(os.path.join(OUT_ROOT, "evaluation_table_all_models_pivot_3-4.csv"))
pivot_5.to_csv(os.path.join(OUT_ROOT,  "evaluation_table_all_models_pivot_5.csv"))

rows = []
for tag, res in breakdown.items():
    for subset, vals in res.items():
        rows.append({"Model": tag, "Subset": subset + "_elements", **vals})
pd.DataFrame(rows).to_csv(os.path.join(OUT_ROOT, "breakdown_3-4_vs_5.csv"), index=False)
print(f"\n  Saved pivot CSVs and breakdown CSV to {OUT_ROOT}")

# Figure S1 (2 x 3 panels). The values are taken from the test-set tables
# written by 02_evaluate_all_models.py (TableS3/TableS4), i.e. the same
# source as Tables 4-6 of the manuscript, so that the figure and the tables
# show identical numbers. (The former per-metric PNGs bd_*.png are no longer
# generated because they are not used in the manuscript.)
_s3 = pd.read_csv(os.path.join(OUT_ROOT, "TableS3_test_3to4elements.csv"))
_s4 = pd.read_csv(os.path.join(OUT_ROOT, "TableS4_test_5elements.csv"))

def _table_row(df, tag):
    c = tag_to_cfg(tag)
    m = ((df["CNN"].astype(str) == str(c["cnn"]))
         & (df["VAE type"] == c["vae_type"])
         & (df["network pattern"].astype(str) == str(c["pattern"])))
    return df[m].iloc[0]

_S1_METRICS = [  # (column in TableS3/S4, y-axis label)
    ("Element acc.",            "Element accuracy"),
    ("Atom coord. MAE",         "Site coordinate MAE (frac)"),
    ("Lattice angle MAPE (%)",  "Lattice angle MAPE (%)"),
    ("Lattice const. MAPE (%)", "Lattice constant MAPE (%)"),
    ("MAE Ef (eV/atom)",        "MAE Ef (eV/atom)"),
    ("MAE Eg (eV)",             "MAE Eg (eV)"),
]
_conv = [_table_row(_s3, conv_tag), _table_row(_s4, conv_tag)]
_prop = [_table_row(_s3, best_tag), _table_row(_s4, best_tag)]
with plt.rc_context({"font.size": 10}):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.8))
    x, w = np.arange(2), 0.35
    for k, (ax, (col, ylabel)) in enumerate(zip(axs.ravel(), _S1_METRICS)):
        ax.bar(x - w / 2, [r[col] for r in _conv], w, color=MODEL_COLOR["Conventional"], edgecolor="black")
        ax.bar(x + w / 2, [r[col] for r in _prop], w, color=MODEL_COLOR["Proposed"],     edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(["3-4 elements", "5 elements"])
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(labelsize=14)
        ax.text(-0.3, 1.1, f"({'ABCDEF'[k]})", transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="bottom", ha="left")
        if col == "Element acc.":
            ax.set_ylim(0, 1.15)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black", label="Conventional"),
            plt.Rectangle((0, 0), 1, 1, facecolor="blue",  edgecolor="black", label="Proposed"),
        ],
        loc="lower right", fontsize=15,
    )
    fig.savefig(os.path.join(out_bd, "FigS1_metric_comparison.png"), dpi=300)
    plt.close(fig)
print(f"  Saved: FigS1_metric_comparison.png to {out_bd}")


# ===========================================================================
# SECTION 4: LATENT SPACE VISUALIZATION
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 4: Latent space visualization")
print("=" * 60)

out_lat = ensure_dir(os.path.join(OUT_ROOT, "fig_latent"))
y_dn    = scaler_y.inverse_transform(y_test)

# Latent space plots are computed on the held-out TEST subset only, for
# consistency with every other narrowed-claim metric (reconstruction, CN
# match rate, space group conservation, and the planned StructureMatcher
# match rate/RMSD) -- all of which are test-set-only.
# Only the proposed model (Figure 6) is plotted; the latent-space plots of
# the other models are not used in the manuscript.
for tag in [best_tag]:
    if tag not in latent_cache:
        print(f"  Skipped (not in cache): {tag}")
        continue
    print(f"  Latent space for: {tag}")
    z     = latent_cache[tag]
    tag_s = sanitize(tag)

    if z.shape[1] < 3:
        print(f"    Skipped: latent dimension < 3 for {tag}")
        continue

    font_size = 26
    plt.rcParams["axes.labelsize"]  = font_size
    plt.rcParams["xtick.labelsize"] = font_size - 2
    plt.rcParams["ytick.labelsize"] = font_size - 2

    fig, ax = plt.subplots(1, 2, figsize=(18, 7.3))
    fig.text(0.016, 0.92, "(A) $E_\\mathrm{f}$", fontsize=font_size)
    fig.text(0.533, 0.92, "(B) $E_\\mathrm{g}$", fontsize=font_size)

    s0 = ax[0].scatter(z[:, 0], z[:, 2], s=7, c=np.squeeze(y_dn[:, 0]), cmap="viridis")
    plt.colorbar(s0, ax=ax[0]).set_label("$E_f$ (eV/atom)")
    ax[0].set_xlabel("$z_1$")
    ax[0].set_ylabel("$z_3$")

    s1 = ax[1].scatter(z[:, 0], z[:, 2], s=7, c=np.squeeze(y_dn[:, 1]), cmap="viridis")
    plt.colorbar(s1, ax=ax[1]).set_label("$E_g$ (eV)")
    ax[1].set_xlabel("$z_1$")
    ax[1].set_ylabel("$z_3$")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, top=0.85)
    plt.savefig(
        os.path.join(out_lat, f"Ef_Eg_{tag_s}.png"),
        bbox_inches="tight", dpi=300,
    )
    plt.close()
    print(f"    Saved: {tag_s}")

# Ensure best model cache is loaded for downstream sections
if best_tag not in recon_cache:
    print(f"\n  Loading {best_tag} for downstream analyses ...")
    cfg_best_tmp = tag_to_cfg(best_tag)
    try:
        vae_tmp                = build_and_load(cfg_best_tmp, X_test, y_test, BASE_DIR, DATA_NAME)
        Xrn_tmp                = predict_recon(vae_tmp, cfg_best_tmp, X_test, y_test)
        recon_cache[best_tag]  = inv_minmax(Xrn_tmp, scaler_X)
        latent_cache[best_tag] = vae_tmp.compress_to_latent(X_test, verbose=0)
        pred_y_cache[best_tag] = scaler_y.inverse_transform(vae_tmp.predict_y(X_test, verbose=0))
    except Exception as e:
        print(f"  ERROR: could not load best model {best_tag}: {e}")
    finally:
        tf.keras.backend.clear_session()


# ===========================================================================
# SECTION 5: CIF GENERATION FOR PAPER FIGURES
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 5: CIF generation (100 samples)")
print("=" * 60)

CIF_ROOT = ensure_dir(os.path.join(OUT_ROOT, "cif_samples_100"))
rng      = np.random.RandomState(RANDOM_STATE)

selected_indices = {}
for n_el in [3, 4, 5]:
    idxs = np.where(n_el_test == n_el)[0]
    selected_indices[n_el] = rng.choice(
        idxs,
        size=min(N_PER_NELEM[n_el], len(idxs)),
        replace=False,
    )

cache_100 = {}
for run_tag in [conv_tag, best_tag]:
    if run_tag not in recon_cache:
        print(f"  Skipped CIF generation (not in cache): {run_tag}")
        continue
    tag_s   = sanitize(run_tag)
    X_recon = recon_cache[run_tag]
    cif_dir = ensure_dir(os.path.join(CIF_ROOT, tag_s))
    print(f"  Generating 100-sample CIFs for {run_tag} ...")

    status_records = []
    for n_el in [3, 4, 5]:
        el_dir = ensure_dir(os.path.join(cif_dir, f"{n_el}_elements"))
        sel    = selected_indices[n_el]

        for k, idx in enumerate(sel):
            orig_path  = os.path.join(el_dir, f"original_{k:04d}_idx{idx:05d}.cif")
            recon_path = os.path.join(el_dir, f"reconstructed_{k:04d}_idx{idx:05d}.cif")
            try:
                with open(orig_path, "w", encoding="utf-8") as f:
                    f.write(df_test.iloc[idx]["cif"])
                write_single_cif_from_ftcp(
                    X_recon[idx], recon_path, MAX_ELMS, MAX_SITES, elm_str
                )
                Structure.from_file(orig_path)
                Structure.from_file(recon_path)
                st = "ok"
            except Exception as e:
                st = f"error:{type(e).__name__}:{e}"
            status_records.append({
                "idx": int(idx), "n_el": n_el, "k": k, "status": st
            })

    df_status = pd.DataFrame(status_records)
    df_status.to_csv(os.path.join(cif_dir, "cif_status.csv"), index=False)
    cache_100[run_tag] = df_status
    n_ok = int((df_status["status"] == "ok").sum())
    print(f"    {n_ok}/{len(df_status)} valid CIF pairs generated")


# ===========================================================================
# SECTION 5.5: BEST- AND WORST-RECONSTRUCTED STRUCTURES
# (Conventional and Proposed, over the FULL test set)
# ===========================================================================
# For each of the Conventional and Proposed (best SVAE) models, ranks every
# test structure by a composite reconstruction-error score (weighted
# combination of lattice constant MAPE, lattice angle MAPE, atomic
# coordinate MAE, and element-identity error; lower = better reconstructed)
# and writes out the top-10 best- and worst-reconstructed structures as CIF
# pairs (original vs. reconstructed), plus the full per-sample error table
# for reference. Rank 1 in each *_best10*/*_worst10* table is the single
# best/worst-reconstructed structure for that model.

print("\n" + "=" * 60)
print("SECTION 5.5: Best/worst-reconstructed structures (Conventional & Proposed)")
print("=" * 60)

BW_ROOT = ensure_dir(os.path.join(OUT_ROOT, "best_worst_reconstructed"))

for run_tag in [conv_tag, best_tag]:
    label = "Conventional" if run_tag == conv_tag else "Proposed"
    if run_tag not in recon_cache:
        print(f"  Skipped ({label}, {run_tag} not in recon_cache).")
        continue

    print(f"  {label} ({run_tag}): scoring all {len(X_orig_global)} test structures ...")
    X_recon = recon_cache[run_tag]
    tag_s   = sanitize(run_tag)
    out_dir = ensure_dir(os.path.join(BW_ROOT, tag_s))

    df_err = compute_per_sample_ftcp_errors(
        X_orig_global, X_recon, Nsites_test, n_el_test, Ntotal_elms, MAX_ELMS, MAX_SITES
    )
    df_err = add_composite_score(df_err)
    df_err.to_csv(os.path.join(out_dir, "per_sample_ftcp_error.csv"), index=False)

    df_best10  = df_err.nsmallest(10, "composite_score").copy()
    df_worst10 = df_err.nlargest(10,  "composite_score").copy()
    plot_best_worst_tables(df_best10,  os.path.join(out_dir, "best10.csv"))
    plot_best_worst_tables(df_worst10, os.path.join(out_dir, "worst10.csv"))

    generate_ranked_cif_pairs(
        X_recon, df_best10, ensure_dir(os.path.join(out_dir, "best10_cifs")), "BEST",
        df_test, n_el_test, MAX_ELMS, MAX_SITES, elm_str,
    )
    generate_ranked_cif_pairs(
        X_recon, df_worst10, ensure_dir(os.path.join(out_dir, "worst10_cifs")), "WORST",
        df_test, n_el_test, MAX_ELMS, MAX_SITES, elm_str,
    )

    best1  = df_best10.iloc[0]
    worst1 = df_worst10.iloc[0]
    print(f"    Best-reconstructed:  idx={int(best1['idx'])}  "
          f"n_el={int(best1['n_elements'])}  composite_score={best1['composite_score']:.6f}")
    print(f"    Worst-reconstructed: idx={int(worst1['idx'])}  "
          f"n_el={int(worst1['n_elements'])}  composite_score={worst1['composite_score']:.6f}")
    print(f"    Saved: {out_dir}")


# ===========================================================================
# SECTION 6: BOND-TYPE RECONSTRUCTION ERROR (all 5-element test structures)
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 6: Bond-type reconstruction error")
print("=" * 60)

out_bond  = ensure_dir(os.path.join(OUT_ROOT, "fig_bond_error"))
nn_finder = CrystalNN()

bond_data_5el_all = {}
for run_tag in [conv_tag, best_tag]:
    if run_tag not in recon_cache:
        print(f"  Skipped bond error (not in cache): {run_tag}")
        continue
    tag_s = sanitize(run_tag)
    print(f"  Bond error (all {mask_5.sum()} 5-element structures) for {run_tag} ...")
    df_5el = compute_bond_errors_5el_all(
        run_tag, recon_cache[run_tag], df_test, n_el_test,
        TARGET_BOND_PAIRS, nn_finder, MAX_ELMS, MAX_SITES, elm_str,
    )
    bond_data_5el_all[run_tag] = df_5el
    df_5el.to_csv(os.path.join(out_bond, f"bond_error_5el_all_{tag_s}.csv"), index=False)

avail_tags_bond = [t for t in [conv_tag, best_tag] if t in bond_data_5el_all]
if len(avail_tags_bond) > 0:
    bond_labels = [f"{p[0]}-{p[1]}" for p in TARGET_BOND_PAIRS]
    avail_bonds = [
        bl for bl in bond_labels
        if any(
            bl in bond_data_5el_all[t].columns
            and bond_data_5el_all[t][bl].dropna().shape[0] >= 3
            for t in avail_tags_bond
        )
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x     = np.arange(len(avail_bonds))
    width = 0.34
    for run_tag, offset in [(conv_tag, -width / 2), (best_tag, width / 2)]:
        if run_tag not in bond_data_5el_all:
            continue
        model_name = get_model_display_name(run_tag, conv_tag, best_tag)
        vals = [safe_mean(bond_data_5el_all[run_tag][bl].dropna().values) for bl in avail_bonds]
        ax.bar(x + offset, vals, width,
               color=MODEL_COLOR[model_name], edgecolor="black", linewidth=1.0, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(avail_bonds, rotation=35, ha="right")
    ax.set_title("Per-bond-type reconstruction error (5 elements)")
    ax.set_ylabel("Bond length MAE (A)")
    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0, frameon=True)
    savefig(fig, os.path.join(out_bond, "bond_error_comparison_5el.png"))
    print("  Saved: bond_error_comparison_5el.png")


# ===========================================================================
# SECTION 7: ELEMENT SLOT ACCURACY
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 7: Element slot accuracy")
print("=" * 60)

out_slot = ensure_dir(os.path.join(OUT_ROOT, "fig_slot_accuracy"))

slot_res = {}
for run_tag in [conv_tag, best_tag]:
    if run_tag not in recon_cache:
        print(f"  Skipped slot accuracy (not in cache): {run_tag}")
        continue
    Xr = recon_cache[run_tag]
    ec_all, et_all = element_level_accuracy(X_orig_global, Xr, mask_all, MAX_ELMS, Ntotal_elms, elm_str)
    slot_res[run_tag] = {
        "3-4":    slot_accs(X_orig_global, Xr, mask_34, MAX_ELMS, Ntotal_elms),
        "5":      slot_accs(X_orig_global, Xr, mask_5,  MAX_ELMS, Ntotal_elms),
        "ec_all": ec_all,
        "et_all": et_all,
    }

if conv_tag in slot_res and best_tag in slot_res:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True)
    for ax_idx, (subset_key, title) in enumerate([("3-4", "3-4 elements"), ("5", "5 elements")]):
        ax    = axes[ax_idx]
        x     = np.arange(MAX_ELMS)
        width = 0.34
        for model_name, run_tag, offset in [
            ("Conventional", conv_tag, -width / 2),
            ("Proposed",     best_tag,  width / 2),
        ]:
            ax.bar(x + offset, slot_res[run_tag][subset_key], width,
                   color=MODEL_COLOR[model_name], edgecolor="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Slot {i}" for i in range(MAX_ELMS)], rotation=12, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        if ax_idx == 0:
            ax.set_ylabel("Accuracy")

    fig.suptitle("Per-slot element reconstruction accuracy", y=0.98)
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black", label="Conventional"),
            plt.Rectangle((0, 0), 1, 1, facecolor="blue",  edgecolor="black", label="Proposed"),
        ],
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        frameon=True,
    )
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.18, top=0.82, wspace=0.18)
    fig.savefig(os.path.join(out_slot, "slot_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Exact per-slot accuracy values, for citing precise numbers in the text
    # (the PNG above is not sufficient for that -- added because the body
    # text needs the underlying figures, not a plot to eyeball).
    slot_rows = []
    for model_name, run_tag in [("Conventional", conv_tag), ("Proposed", best_tag)]:
        for subset_key, subset_label in [("3-4", "3-4 elements"), ("5", "5 elements")]:
            accs = slot_res[run_tag][subset_key]
            for slot_idx, acc in enumerate(accs):
                slot_rows.append({
                    "model": model_name,
                    "tag": run_tag,
                    "subset": subset_label,
                    "slot": slot_idx,
                    "accuracy": float(acc),
                })
    df_slot = pd.DataFrame(slot_rows)
    df_slot.to_csv(os.path.join(out_slot, "slot_accuracy_all.csv"), index=False)
    print(f"  Saved: slot_accuracy_all.csv ({len(df_slot)} rows)")

if best_tag in slot_res:
    ec_all = slot_res[best_tag]["ec_all"]
    et_all = slot_res[best_tag]["et_all"]
    el_acc_map = {s: ec_all[s] / et_all[s] for s in et_all if et_all[s] >= 5}
    df_el = pd.DataFrame({
        "element":  list(el_acc_map.keys()),
        "accuracy": list(el_acc_map.values()),
        "count":    [et_all[s] for s in el_acc_map],
    }).sort_values("accuracy")
    df_el.to_csv(
        os.path.join(out_slot, f"element_accuracy_all_{sanitize(best_tag)}.csv"), index=False
    )
    # Figure S5: 10 lowest-accuracy species, hydrogen excluded (zero-padded
    # slots are assigned to H in the argmax-based evaluation).
    worst = df_el[df_el["element"] != "H"].head(10)
    if len(worst) > 0:
        # Font sizes: 1.5 times those of the previous manuscript version
        # (tick labels 13 -> 19.5, axis label 14 -> 21).
        with plt.rc_context({"font.size": 10}):
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            ax.barh(worst["element"], worst["accuracy"], color="red", edgecolor="black")
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("Accuracy", fontsize=21)
            ax.tick_params(labelsize=19.5)
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_slot, f"FigS5_element_worst10_{sanitize(best_tag)}.png"),
                dpi=300,
            )
            plt.close(fig)

print(f"  Saved slot accuracy figures to {out_slot}")


# ===========================================================================
# FINAL SUMMARY TABLE
# ===========================================================================

print("\n" + "=" * 60)
print("Summary table (main text)")
print("=" * 60)

rows = []
for run_tag in [conv_tag, best_tag]:
    if run_tag not in breakdown:
        continue
    res = breakdown[run_tag]
    for sk in ["3-4", "5"]:
        rows.append({"Model": run_tag, "Subset": f"{sk} elements", **res[sk]})
df_sum = pd.DataFrame(rows)
df_sum.to_csv(os.path.join(OUT_ROOT, "summary_main_table.csv"), index=False)
print(df_sum.to_string(index=False))

# ===========================================================================
# Table 4 / Table 5 SOURCE TABLE: Conventional SVAE / Best SVAE / Best USVAE
# ===========================================================================
# Table 4 (3-4-element subset) and Table 5 (5-element subset) in the
# manuscript each show three rows: Conventional SVAE (CNN 0, pattern 0),
# Best SVAE (= "Proposed", CNN 1... pattern 2), and Best USVAE (its own,
# independently selected architecture -- see 02_evaluate_all_models.py).
# USVAE has no property regressor, so its MAE Ef / MAE Eg columns are
# expected to be missing/NaN; that is not an error.

MODEL_LABELS = {
    conv_tag:       "Conventional SVAE",
    best_svae_tag:  "Best SVAE",
}
if best_usvae_tag:
    MODEL_LABELS[best_usvae_tag] = "Best USVAE"

rows_45 = []
for run_tag, label in MODEL_LABELS.items():
    if run_tag not in breakdown:
        print(f"  WARNING: {run_tag} ({label}) not in breakdown; skipping in Table 4/5 source.")
        continue
    res = breakdown[run_tag]
    for sk, table_name in [("3-4", "Table 4"), ("5", "Table 5")]:
        rows_45.append({
            "Table": table_name,
            "Model": label,
            "Tag": run_tag,
            "Subset": f"{sk} elements",
            **res[sk],
        })
df_45 = pd.DataFrame(rows_45)
df_45_path = os.path.join(OUT_ROOT, "summary_table4_table5_source.csv")
df_45.to_csv(df_45_path, index=False)
print(f"\n  Saved: {df_45_path}")
print(df_45.to_string(index=False))

print("\nDone.")
