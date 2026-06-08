# -*- coding: utf-8 -*-
"""
Unified analysis script for the FTCP-VAE 5-element extension paper.

Major updates in this revision
-------------------------------
1. Larger figure font sizes for paper-ready plots.
2. Breakdown plots use Conventional = black and Proposed = blue.
3. Latent histograms removed.
4. Latent Ef/Eg plots use the requested two-panel format.
5. Legends moved outside figures where needed.
6. Violin plots are added in addition to boxplots.
7. Detailed full-test-set analysis for the proposed model is preserved.
8. Best model configuration is auto-detected from saved evaluation CSVs.

Fixes applied
-------------
1. Docstring: corrected "53 different network configurations" to "42".
2. orig_cache redundancy removed: X_orig_global is used directly throughout.
3. all_eval_tags narrowed to [conv_cfg_key, best_tag] to avoid loading all
   per-metric best models unnecessarily.
4. KeyError guard added: recon_cache writes are skipped on error and
   downstream sections check for key existence before referencing.
5. Section 9e now guards against empty full_pairs before executing the loop.
6. Section 8 CIF-loading logic refactored into a single helper function
   _load_cif_pairs_from_dir to eliminate the three near-identical blocks.
"""

import os, re, glob, warnings, joblib, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

tf.compat.v1.enable_eager_execution()
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from module.data import FTCP_represent
from module.utils import pad, minmax, inv_minmax
from module.result_analysis import (
    sanitize, ensure_dir, safe_mean, savefig, MAPE, MAE_site_coor, elem_acc,
    extract_lattice_coords, compute_per_sample_ftcp_errors, add_composite_score,
    slot_accs, element_level_accuracy, get_bond_lengths, bond_mae,
    compute_bond_errors_5el_all, tag_to_cfg, build_and_load, predict_recon,
    get_model_display_name, write_single_cif_from_ftcp, generate_ranked_cif_pairs,
    build_full_reconstructed_cif_cache, load_cif_pairs_from_cache,
    plot_metric_breakdown_bar, plot_violin_with_box, plot_best_worst_tables,
    aggregate_results, find_best_cfg, MODEL_COLOR,
)


# ===========================================================================
# SECTION 0: SETTINGS
# ===========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_NAME    = "data_query_3_5_elements_property_nsites_below_112"
MAX_ELMS     = 5
MAX_SITES    = 112
RANDOM_STATE = 21
PROP         = ["formation_energy_per_atom", "band_gap"]

# Conventional baseline configuration (fixed reference model)
CONV_CFG = {
    "cnn":      "0",
    "vae_type": "SVAE",
    "pattern":  "0",
    "label":    "Conventional\n(CNN0 SVAE p0)",
}

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

# Metric index shared across sections
METRICS_IDX = [
    "MAE Ef (eV)",
    "MAE Eg (eV)",
    "Element accuracy",
    "Lattice constant MAPE (%)",
    "Lattice angle MAPE (%)",
    "Site coordinate MAE (frac)",
]

# True if higher value is better for the metric
HIGHER_IS_BETTER = {
    "MAE Ef (eV)":                False,
    "MAE Eg (eV)":                False,
    "Element accuracy":           True,
    "Lattice constant MAPE (%)":  False,
    "Lattice angle MAPE (%)":     False,
    "Site coordinate MAE (frac)": False,
}

SUBSET_COLOR = {"3-4": "black", "5": "red"}

# Global font size settings
FS = {
    "base":   22,
    "tick":   20,
    "legend": 20,
    "title":  22,
}

plt.rcParams.update({
    "font.size":        FS["base"],
    "axes.labelsize":   FS["base"],
    "xtick.labelsize":  FS["tick"],
    "ytick.labelsize":  FS["tick"],
    "legend.fontsize":  FS["legend"],
    "axes.titlesize":   FS["title"],
})

sns.set_style("ticks")

# Model/network pattern configurations for full sweep
cnn_patterns_all        = ["0", "1", "2"]
superviseds_all         = ["SVAE", "USVAE"]
network_pattern_numbers = ["0", "1", "2", "3", "4", "5", "6"]


# ===========================================================================
# HELPER: load CIF pairs from a directory tree (shared by Section 8a/8b/8c)
# ===========================================================================

def _load_cif_pairs_from_dir(cif_root, run_tag):
    """Load original/reconstructed CIF pairs from the 100-sample directory.

    Parameters
    ----------
    cif_root : str
        Root directory that contains per-model sub-directories.
    run_tag : str
        Model tag string used to locate the sub-directory via sanitize().

    Returns
    -------
    list of dict
        Each dict contains keys: idx (int), n_el (int),
        struct_orig (Structure), struct_recon (Structure).
    """
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
# SECTION 1: DATA LOADING
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
indices = np.arange(len(df_full))
_, ind_test = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)
df_test = df_full.iloc[ind_test].copy().reset_index(drop=True)

FTCP_rep, Nsites_test = FTCP_represent(df_test, MAX_ELMS, MAX_SITES, return_Nsites=True)
FTCP_rep = pad(FTCP_rep, 2)
X_test, scaler_X = minmax(FTCP_rep.astype("float32"))

scaler_y = MinMaxScaler()
scaler_y.fit(df_full[PROP].values.astype("float32"))
y_test = scaler_y.transform(df_test[PROP].values.astype("float32"))

n_el_test = df_test["n_elements"].values
mask_34   = np.isin(n_el_test, [3, 4])
mask_5    = (n_el_test == 5)
mask_all  = np.ones(len(df_test), dtype=bool)

# Precompute original arrays once; reused by all sections
X_orig_global = inv_minmax(X_test, scaler_X)
abc_o_global, ang_o_global, coor_o_global = extract_lattice_coords(
    X_orig_global, Ntotal_elms, MAX_SITES
)
y_true_global = scaler_y.inverse_transform(y_test)

print(f"  Test: {len(df_test)} total  | 3-4 el: {mask_34.sum()}  | 5 el: {mask_5.sum()}")


# ===========================================================================
# SECTION 2: RESULT AGGREGATION AND BEST MODEL DETECTION
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 2: Result aggregation and best model detection")
print("=" * 60)

result_all = aggregate_results(
    BASE_DIR, DATA_NAME, cnn_patterns_all, superviseds_all, network_pattern_numbers
)

agg_path = os.path.join(BASE_DIR, f"{DATA_NAME}_result_all.csv")
result_all.to_csv(agg_path)
print(f"  Saved: {agg_path}  (shape {result_all.shape})")

# Auto-detect best model from reconstruction metrics
best_tag, ranking_df = find_best_cfg(result_all)
ranking_df.to_csv(os.path.join(OUT_ROOT, "model_ranking.csv"))
print(f"\n  Best model detected: {best_tag}")

# Build BEST_CFG from auto-detected tag
BEST_CFG          = tag_to_cfg(best_tag)
BEST_CFG["label"] = f"Proposed\n({best_tag})"

conv_tag     = f"CNN{CONV_CFG['cnn']} {CONV_CFG['vae_type']} pattern{CONV_CFG['pattern']}"
conv_cfg_key = conv_tag

# Per-metric best model (used only for reporting; not added to all_eval_tags)
best_tag_for = {}
for metric_name, hib in HIGHER_IS_BETTER.items():
    if metric_name not in result_all.columns:
        continue
    col = result_all[metric_name].dropna()
    if len(col) == 0:
        continue
    best_tag_for[metric_name] = col.idxmax() if hib else col.idxmin()

best_df = pd.DataFrame([
    {"Metric": m, "BestModelTag": t, "BestValue": result_all.loc[t, m]}
    for m, t in best_tag_for.items()
])
best_df.to_csv(os.path.join(OUT_ROOT, "best_model_per_metric.csv"), index=False)
print(best_df.to_string(index=False))


# ===========================================================================
# SECTION 2.5: FULL BREAKDOWN FOR ALL MODEL CONFIGURATIONS
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 2.5: Full breakdown (all configurations, 3-4 vs 5 elements)")
print("=" * 60)

all_breakdown_rows = []

for cnn, sup, net in itertools.product(cnn_patterns_all, superviseds_all, network_pattern_numbers):
    tag = f"CNN{cnn} {sup} pattern{net}"
    cfg = {"cnn": cnn, "vae_type": sup, "pattern": net, "label": tag}
    try:
        vae          = build_and_load(cfg, X_test, y_test, BASE_DIR, DATA_NAME)
        X_recon_norm = predict_recon(vae, cfg, X_test, y_test)
        X_recon      = inv_minmax(X_recon_norm, scaler_X)
        abc_r, ang_r, coor_r = extract_lattice_coords(X_recon, Ntotal_elms, MAX_SITES)

        is_sup = (sup == "SVAE")
        y_hat  = scaler_y.inverse_transform(vae.predict_y(X_test, verbose=0)) if is_sup else None

        for mask, subset_label in [(mask_34, "3-4_elements"), (mask_5, "5_elements")]:
            mae_ef = float(np.mean(np.abs(y_true_global[mask, 0] - y_hat[mask, 0]))) if is_sup else np.nan
            mae_eg = float(np.mean(np.abs(y_true_global[mask, 1] - y_hat[mask, 1]))) if is_sup else np.nan
            all_breakdown_rows.append({
                "CNN":                   cnn,
                "VAE_type":              sup,
                "network_pattern_code":  net,
                "model_tag":             tag,
                "subset":                subset_label,
                "MAE_Ef":                mae_ef,
                "MAE_Eg":                mae_eg,
                "Accuracy_element":      elem_acc(X_orig_global[mask], X_recon[mask], MAX_ELMS, Ntotal_elms),
                "Lattice_constant_MAPE": MAPE(abc_o_global[mask], abc_r[mask]),
                "Lattice_angles_MAPE":   MAPE(ang_o_global[mask], ang_r[mask]),
                "Atom_coordinates_MAE":  MAE_site_coor(coor_o_global[mask], coor_r[mask], Nsites_test[mask]),
            })

        del X_recon, X_recon_norm, abc_r, ang_r, coor_r, y_hat
    except Exception as e:
        print(f"    Error on {tag}: {e}")
    finally:
        tf.keras.backend.clear_session()

df_all_breakdown = pd.DataFrame(all_breakdown_rows)

all_breakdown_path = os.path.join(OUT_ROOT, "all_models_breakdown_3-4_vs_5.csv")
df_all_breakdown.to_csv(all_breakdown_path, index=False)
print(f"  Saved: {all_breakdown_path}  (shape {df_all_breakdown.shape})")


# ===========================================================================
# SECTION 3: RECONSTRUCTION ACCURACY BY n_elements
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 3: Reconstruction accuracy breakdown (3-4 vs 5 elements)")
print("=" * 60)

out_bd = ensure_dir(os.path.join(OUT_ROOT, "fig_breakdown"))

# Evaluate only conventional and best model to avoid loading all per-metric models
all_eval_tags = list(dict.fromkeys([conv_cfg_key, best_tag]))

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

# Breakdown bar charts: use best_tag for all metrics (per-metric best no longer loaded)
if best_tag in breakdown:
    for metric_name in METRICS_IDX:
        if metric_name not in breakdown[conv_cfg_key]["3-4"]:
            continue
        conv_vals = [breakdown[conv_cfg_key]["3-4"][metric_name], breakdown[conv_cfg_key]["5"][metric_name]]
        best_vals = [breakdown[best_tag]["3-4"][metric_name],     breakdown[best_tag]["5"][metric_name]]
        plot_metric_breakdown_bar(
            metric_name,
            conv_vals,
            best_vals,
            os.path.join(out_bd, f"bd_{sanitize(metric_name)}_{sanitize(best_tag)}.png"),
            best_tag,
        )

print(f"  Saved breakdown figures to {out_bd}")


# ===========================================================================
# SECTION 4: LATENT SPACE VISUALIZATION
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 4: Latent space visualization")
print("=" * 60)

out_lat = ensure_dir(os.path.join(OUT_ROOT, "fig_latent"))
y_dn    = scaler_y.inverse_transform(y_test)

for tag in all_eval_tags:
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
    plt.colorbar(s0, ax=ax[0]).set_label("$E_f$ (eV)")
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
        vae_tmp              = build_and_load(cfg_best_tmp, X_test, y_test, BASE_DIR, DATA_NAME)
        Xrn_tmp              = predict_recon(vae_tmp, cfg_best_tmp, X_test, y_test)
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

# Fixed sample indices shared across models for fair comparison
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
    worst = df_el.head(20)
    if len(worst) > 0:
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        ax.barh(worst["element"], worst["accuracy"],
                color="red", edgecolor="black", linewidth=0.8)
        ax.set_xlabel("Reconstruction accuracy")
        ax.set_xlim(0, 1.05)
        ax.set_title("20 lowest-accuracy elements on all test structures: Proposed model")
        fig.subplots_adjust(left=0.20, right=0.97, bottom=0.12, top=0.90)
        fig.savefig(
            os.path.join(out_slot, f"element_worst20_all_{sanitize(best_tag)}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

print(f"  Saved slot accuracy figures to {out_slot}")


# ===========================================================================
# SECTION 8: ADDITIONAL ANALYSES ON 100-SAMPLE SET
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 8: Additional analyses on 100-sample CIF set")
print("=" * 60)

out_add = ensure_dir(os.path.join(OUT_ROOT, "fig_additional"))

# ----- 8a: Coordination number accuracy -----
print("  8a: Coordination number accuracy ...")
cn_data_all = {}
for run_tag in [conv_tag, best_tag]:
    tag_s = sanitize(run_tag)
    pairs = _load_cif_pairs_from_dir(CIF_ROOT, run_tag)

    recs = []
    for p in pairs:
        n_sites = min(len(p["struct_orig"]), len(p["struct_recon"]))
        for si in range(n_sites):
            try:
                cno = len(nn_finder.get_nn_info(p["struct_orig"],  si))
                cnr = len(nn_finder.get_nn_info(p["struct_recon"], si))
            except Exception:
                continue
            recs.append({
                "idx": p["idx"], "n_el": p["n_el"],
                "CN_orig": cno, "CN_recon": cnr, "match": int(cno == cnr),
            })
    df_cn = pd.DataFrame(recs)
    cn_data_all[run_tag] = df_cn
    df_cn.to_csv(os.path.join(out_add, f"cn_100_{tag_s}.csv"), index=False)

fig, ax = plt.subplots(figsize=(8.4, 5.8))
x     = np.arange(2)
width = 0.34
for subset_key, n_vals, xpos in [("3-4", [3, 4], 0), ("5", [5], 1)]:
    for model_name, run_tag, offset in [
        ("Conventional", conv_tag, -width / 2),
        ("Proposed",     best_tag,  width / 2),
    ]:
        if run_tag not in cn_data_all:
            continue
        sub = cn_data_all[run_tag]
        sub = sub[sub["n_el"].isin(n_vals)]
        val = float(sub["match"].mean()) if len(sub) > 0 else np.nan
        ax.bar(xpos + offset, val, width,
               color=MODEL_COLOR[model_name], edgecolor="black", linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels(["3-4 elements", "5 elements"])
ax.set_ylabel("Coordination number match rate")
ax.set_ylim(0, 1.08)
ax.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black", label="Conventional"),
        plt.Rectangle((0, 0), 1, 1, facecolor="blue",  edgecolor="black", label="Proposed"),
    ],
    loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0,
)
ax.set_title("Coordination number accuracy")
savefig(fig, os.path.join(out_add, "cn_match_rate_100.png"))

# ----- 8b: Unit cell volume reconstruction error -----
print("  8b: Unit cell volume reconstruction error ...")
vol_data_all = {}
for run_tag in [conv_tag, best_tag]:
    tag_s = sanitize(run_tag)
    pairs = _load_cif_pairs_from_dir(CIF_ROOT, run_tag)

    recs = []
    for p in pairs:
        vo = p["struct_orig"].volume
        vr = p["struct_recon"].volume
        recs.append({
            "idx":     p["idx"],
            "subset":  "5" if p["n_el"] == 5 else "3-4",
            "model":   get_model_display_name(run_tag, conv_tag, best_tag),
            "rel_err": abs(vo - vr) / max(vo, 1.0) * 100.0,
        })
    df_v = pd.DataFrame(recs)
    vol_data_all[run_tag] = df_v
    df_v.to_csv(os.path.join(out_add, f"volume_100_{tag_s}.csv"), index=False)

vol_plot_df = pd.concat(
    [vol_data_all[t] for t in [conv_tag, best_tag] if t in vol_data_all],
    ignore_index=True,
)
plot_violin_with_box(
    vol_plot_df,
    x_col="subset", y_col="rel_err", hue_col="model",
    title="Unit cell volume reconstruction error",
    y_label="Volume relative error (%)",
    out_prefix=os.path.join(out_add, "volume_error"),
    palette={"Conventional": "gray", "Proposed": "steelblue"},
)

# ----- 8c: Space group conservation rate -----
print("  8c: Space group conservation rate ...")
sg_data_all = {}
for run_tag in [conv_tag, best_tag]:
    tag_s = sanitize(run_tag)
    pairs = _load_cif_pairs_from_dir(CIF_ROOT, run_tag)

    recs = []
    for p in pairs:
        try:
            sgo = SpacegroupAnalyzer(p["struct_orig"],  symprec=0.1).get_space_group_number()
            sgr = SpacegroupAnalyzer(p["struct_recon"], symprec=0.1).get_space_group_number()
            recs.append({
                "idx":    p["idx"],
                "subset": "5" if p["n_el"] == 5 else "3-4",
                "model":  get_model_display_name(run_tag, conv_tag, best_tag),
                "match":  int(sgo == sgr),
            })
        except Exception:
            continue
    df_sg = pd.DataFrame(recs)
    sg_data_all[run_tag] = df_sg
    df_sg.to_csv(os.path.join(out_add, f"spacegroup_100_{tag_s}.csv"), index=False)

fig, ax = plt.subplots(figsize=(8.4, 5.8))
for subset_key, xpos in [("3-4", 0), ("5", 1)]:
    for model_name, run_tag, offset in [
        ("Conventional", conv_tag, -0.17),
        ("Proposed",     best_tag,  0.17),
    ]:
        if run_tag not in sg_data_all:
            continue
        sub = sg_data_all[run_tag]
        sub = sub[sub["subset"] == subset_key]
        val = float(sub["match"].mean()) if len(sub) > 0 else np.nan
        ax.bar(xpos + offset, val, 0.34,
               color=MODEL_COLOR[model_name], edgecolor="black", linewidth=1.0)
ax.set_xticks([0, 1])
ax.set_xticklabels(["3-4 elements", "5 elements"])
ax.set_ylabel("Space group conservation rate")
ax.set_ylim(0, 1.08)
ax.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black", label="Conventional"),
        plt.Rectangle((0, 0), 1, 1, facecolor="blue",  edgecolor="black", label="Proposed"),
    ],
    loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0,
)
ax.set_title("Space group conservation rate")
savefig(fig, os.path.join(out_add, "spacegroup_conservation_100.png"))


# ===========================================================================
# SECTION 9: FULL-TEST-SET DETAILED ANALYSIS FOR PROPOSED MODEL ONLY
# ===========================================================================

print("\n" + "=" * 60)
print("SECTION 9: Full-test-set detailed analysis for proposed model")
print("=" * 60)

if best_tag not in recon_cache:
    print(f"  WARNING: {best_tag} not in recon_cache; skipping Section 9.")
else:
    out_full    = ensure_dir(os.path.join(OUT_ROOT, "full_testset_proposed"))
    X_best      = recon_cache[best_tag]

    # ----- 9a: Per-sample FTCP error table -----
    print("  9a: Per-sample FTCP error table ...")
    df_err = compute_per_sample_ftcp_errors(
        X_orig_global, X_best, Nsites_test, n_el_test, Ntotal_elms, MAX_ELMS, MAX_SITES
    )
    df_err = add_composite_score(df_err)
    df_err.to_csv(os.path.join(out_full, "per_sample_ftcp_error_proposed.csv"), index=False)

    # ----- 9b: Error distribution plots -----
    print("  9b: Error distribution plots by number of elements ...")
    plot_violin_with_box(
        df_err.assign(
            subset=df_err["n_elements"].map(lambda x: "5" if x == 5 else "3-4"),
            model="Proposed"
        ),
        x_col="subset", y_col="coord_mae", hue_col="model",
        title="Atomic coordinate error on all test structures: Proposed model",
        y_label="Atomic coordinate MAE",
        out_prefix=os.path.join(out_full, "coord_error_proposed"),
        palette={"Proposed": "steelblue"},
    )

    plot_violin_with_box(
        df_err.assign(
            subset=df_err["n_elements"].map(lambda x: "5" if x == 5 else "3-4"),
            model="Proposed"
        ),
        x_col="subset", y_col="composite_score", hue_col="model",
        title="Composite reconstruction score on all test structures: Proposed model",
        y_label="Composite reconstruction score",
        out_prefix=os.path.join(out_full, "composite_score_proposed"),
        palette={"Proposed": "steelblue"},
    )

    # ----- 9c: Best 10 and worst 10 structures -----
    print("  9c: Best 10 and worst 10 structures by composite score ...")
    df_best10  = df_err.nsmallest(10, "composite_score").copy()
    df_worst10 = df_err.nlargest(10,  "composite_score").copy()
    plot_best_worst_tables(df_best10,  os.path.join(out_full, "best10_proposed.csv"))
    plot_best_worst_tables(df_worst10, os.path.join(out_full, "worst10_proposed.csv"))

    generate_ranked_cif_pairs(
        X_best, df_best10,
        ensure_dir(os.path.join(out_full, "best10_cifs")), "BEST",
        df_test, n_el_test, MAX_ELMS, MAX_SITES, elm_str,
    )
    generate_ranked_cif_pairs(
        X_best, df_worst10,
        ensure_dir(os.path.join(out_full, "worst10_cifs")), "WORST",
        df_test, n_el_test, MAX_ELMS, MAX_SITES, elm_str,
    )

    # ----- 9d: Full CIF cache -----
    print("  9d: Full CIF cache for all test structures of proposed model ...")
    full_cache_dir = ensure_dir(os.path.join(out_full, "full_cif_cache"))
    full_cache_df  = build_full_reconstructed_cif_cache(
        best_tag, X_best, full_cache_dir,
        df_test, n_el_test, MAX_ELMS, MAX_SITES, elm_str,
    )
    full_pairs = load_cif_pairs_from_cache(full_cache_df)
    print(f"    Valid full CIF pairs: {len(full_pairs)} / {len(full_cache_df)}")

    # ----- 9e: Bond reconstruction error across all valid structures -----
    print("  9e: Bond reconstruction error across all valid reconstructed structures ...")
    if len(full_pairs) == 0:
        print("    WARNING: no valid CIF pairs found; skipping 9e.")
    else:
        full_bond_records = []
        for p in full_pairs:
            bo  = get_bond_lengths(p["struct_orig"],  nn_finder)
            br  = get_bond_lengths(p["struct_recon"], nn_finder)
            row = {
                "idx":        p["idx"],
                "n_elements": p["n_el"],
                "subset":     "5" if p["n_el"] == 5 else "3-4",
            }
            for pair in TARGET_BOND_PAIRS:
                row[f"{pair[0]}-{pair[1]}"] = bond_mae(bo, br, pair)
            full_bond_records.append(row)

        df_bond_full = pd.DataFrame(full_bond_records)
        df_bond_full.to_csv(
            os.path.join(out_full, "bond_error_all_valid_structures_proposed.csv"), index=False
        )

        bond_long_rows = []
        for _, row in df_bond_full.iterrows():
            for label in [f"{p[0]}-{p[1]}" for p in TARGET_BOND_PAIRS]:
                if label in row.index and pd.notna(row[label]):
                    bond_long_rows.append({
                        "idx":       int(row["idx"]),
                        "subset":    row["subset"],
                        "bond_type": label,
                        "mae":       float(row[label]),
                    })

        df_bond_long = pd.DataFrame(bond_long_rows)
        df_bond_long.to_csv(
            os.path.join(out_full, "bond_error_all_valid_structures_proposed_long.csv"),
            index=False,
        )

        if len(df_bond_long) > 0:
            grouped = (
                df_bond_long.groupby(["subset", "bond_type"])["mae"]
                .mean()
                .reset_index()
            )
            fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
            for ax_idx, subset_key in enumerate(["3-4", "5"]):
                ax  = axes[ax_idx]
                sub = grouped[grouped["subset"] == subset_key].sort_values("bond_type")
                x   = np.arange(len(sub))
                ax.bar(x, sub["mae"], color="steelblue", edgecolor="black", linewidth=0.9)
                ax.set_xticks(x)
                ax.set_xticklabels(sub["bond_type"], rotation=35, ha="right")
                ax.set_title(f"{subset_key} elements")
                ax.set_ylabel("Bond length MAE (A)" if ax_idx == 0 else "")
            fig.suptitle(
                "Proposed model: bond reconstruction error on all valid reconstructed structures"
            )
            savefig(fig, os.path.join(out_full, "bond_error_all_valid_structures_proposed.png"))

    # ----- 9f: Element accuracy across all test structures -----
    print("  9f: Element accuracy across all test structures ...")
    elem_rows = []
    for slot in range(MAX_ELMS):
        t = np.argmax(X_orig_global[:, :Ntotal_elms, slot], axis=1)
        p = np.argmax(X_best[:, :Ntotal_elms, slot],        axis=1)
        for idx in range(len(t)):
            elem_rows.append({
                "idx":          idx,
                "slot":         slot,
                "subset":       "5" if n_el_test[idx] == 5 else "3-4",
                "true_element": elm_str[t[idx]],
                "pred_element": elm_str[p[idx]],
                "correct":      int(t[idx] == p[idx]),
            })

    df_elem_all = pd.DataFrame(elem_rows)
    df_elem_all.to_csv(
        os.path.join(out_full, "element_slot_detail_all_test_structures_proposed.csv"),
        index=False,
    )

    df_slot_summary = (
        df_elem_all.groupby(["subset", "slot"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    df_slot_summary.to_csv(
        os.path.join(out_full, "element_slot_accuracy_all_test_structures_proposed.csv"),
        index=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    for ax_idx, subset_key in enumerate(["3-4", "5"]):
        ax  = axes[ax_idx]
        sub = df_slot_summary[df_slot_summary["subset"] == subset_key].sort_values("slot")
        ax.bar(sub["slot"].astype(str), sub["accuracy"],
               color="steelblue", edgecolor="black", linewidth=1.0)
        ax.set_title(f"{subset_key} elements")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Accuracy" if ax_idx == 0 else "")
    fig.suptitle("Proposed model: slot accuracy on all test structures")
    savefig(fig, os.path.join(out_full, "element_slot_accuracy_all_test_structures_proposed.png"))

    el_detail = (
        df_elem_all.groupby(["subset", "true_element"])["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "count"})
    )
    el_detail = el_detail[el_detail["count"] >= 5].copy()
    el_detail.to_csv(
        os.path.join(
            out_full,
            "element_accuracy_by_true_element_all_test_structures_proposed.csv",
        ),
        index=False,
    )

    for subset_key in ["3-4", "5"]:
        worst = (
            el_detail[el_detail["subset"] == subset_key]
            .sort_values(["accuracy", "count"], ascending=[True, False])
            .head(10)
        )
        if len(worst) == 0:
            continue
        fig, ax = plt.subplots(figsize=(9.0, 5.8))
        ax.barh(worst["element"], worst["accuracy"],
                color="red", edgecolor="black", linewidth=0.9)
        ax.set_xlabel("Accuracy")
        ax.set_xlim(0, 1.05)
        ax.set_title(
            f"Proposed model: worst 10 true elements in {subset_key}-element structures"
        )
        savefig(fig, os.path.join(out_full, f"worst10_elements_{subset_key}_proposed.png"))

    # ----- 9g: Latent-error relationship -----
    print("  9g: Latent-error relationship for proposed model ...")
    z_best = latent_cache[best_tag]
    df_lat_err = pd.DataFrame({
        "idx":             np.arange(len(z_best)),
        "subset":          np.where(n_el_test == 5, "5", "3-4"),
        "z1":              z_best[:, 0],
        "z2":              z_best[:, 1],
        "coord_mae":       df_err["coord_mae"].values,
        "composite_score": df_err["composite_score"].values,
    })
    df_lat_err.to_csv(
        os.path.join(out_full, "latent_error_relation_proposed.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sc = ax.scatter(
        df_lat_err["z1"], df_lat_err["z2"],
        c=df_lat_err["composite_score"],
        cmap="viridis", s=14, alpha=0.75, linewidths=0,
    )
    fig.colorbar(sc, ax=ax).set_label("Composite reconstruction score")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_title(
        "Proposed model: latent space colored by composite reconstruction score"
    )
    savefig(fig, os.path.join(out_full, "latent_composite_score_proposed.png"))


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

print("\nDone.")