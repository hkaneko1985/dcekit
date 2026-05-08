<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Result analysis utility functions for FTCP-VAE experiments.

All functions are stateless; experiment-level variables (paths, arrays, etc.)
are passed explicitly as arguments so this module can be imported cleanly.
"""

import os
import re
import glob
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from itertools import product
from pymatgen.core import Structure
from sklearn import metrics as sk_metrics
from module.FTCP import FTCP_VAE
from module.sampling import get_info


# Module-level style constants
MODEL_COLOR = {"Conventional": "black", "Proposed": "blue"}


# ===========================================================================
# General utilities
# ===========================================================================

def sanitize(s):
    """Convert a string to a filesystem-safe identifier."""
    s = s.replace(" ", "_").replace("/", "_").replace("\n", "_")
    s = s.replace("(", "").replace(")", "").replace("%", "pct")
    return re.sub(r"[^A-Za-z0-9_\-\.]", "", s)


def ensure_dir(path):
    """Create directory if it does not exist and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def safe_mean(values):
    """Return mean of finite values; return NaN if no finite values exist."""
    values = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def savefig(fig, path):
    """Apply tight layout, save figure at 300 dpi, and close."""
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def legend_outside_bottom_right(ax, ncol=1):
    """Place legend outside the axes, anchored at lower-left of the bbox."""
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        frameon=True,
        ncol=ncol,
    )


# ===========================================================================
# Metric computation
# ===========================================================================

def MAPE(yt, yp):
    """Mean Absolute Percentage Error."""
    yt = np.array(yt, dtype=float) + 1e-12
    yp = np.array(yp, dtype=float) + 1e-12
    return float(np.mean(np.abs((yt - yp) / yt)) * 100.0)


def MAE_site_coor(coor, coor_r, nsites):
    """MAE for site coordinates, evaluated only over occupied sites."""
    src = []
    rec = []
    for i in range(len(coor)):
        src.append(coor[i, :nsites[i], :])
        rec.append(coor_r[i, :nsites[i], :])
    if len(src) == 0:
        return np.nan
    return float(np.mean(np.abs(np.vstack(src) - np.vstack(rec))))


def elem_acc(X_o, X_r, max_elms, n_elm):
    """Mean element classification accuracy across all element slots."""
    accs = []
    for i in range(max_elms):
        t = np.argmax(X_o[:, :n_elm, i], axis=1)
        p = np.argmax(X_r[:, :n_elm, i], axis=1)
        accs.append(sk_metrics.accuracy_score(t, p))
    return float(np.mean(accs))


def extract_lattice_coords(X_dn, n_elm, max_sites):
    """Extract lattice constants, angles, and fractional site coordinates from FTCP tensor."""
    abc  = X_dn[:, n_elm,           :3]
    ang  = X_dn[:, n_elm + 1,       :3]
    coor = X_dn[:, n_elm + 2:n_elm + 2 + max_sites, :3]
    return abc, ang, coor


def compute_per_sample_ftcp_errors(X_orig, X_recon, nsites, n_el_test,
                                    n_elm, max_elms, max_sites):
    """Compute per-sample reconstruction errors for all FTCP components."""
    abc_o, ang_o, coor_o = extract_lattice_coords(X_orig, n_elm, max_sites)
    abc_r, ang_r, coor_r = extract_lattice_coords(X_recon, n_elm, max_sites)
    records = []
    for i in range(len(X_orig)):
        elem_slot_accs = []
        for slot in range(max_elms):
            t = int(np.argmax(X_orig[i, :n_elm, slot]))
            p = int(np.argmax(X_recon[i, :n_elm, slot]))
            elem_slot_accs.append(int(t == p))
        elem_error = 1.0 - float(np.mean(elem_slot_accs))
        abc_mape  = MAPE(abc_o[i:i+1], abc_r[i:i+1])
        ang_mape  = MAPE(ang_o[i:i+1], ang_r[i:i+1])
        coord_mae = MAE_site_coor(coor_o[i:i+1], coor_r[i:i+1], np.array([nsites[i]]))
        records.append({
            "idx":        i,
            "n_elements": int(n_el_test[i]),
            "abc_mape":   abc_mape,
            "ang_mape":   ang_mape,
            "coord_mae":  coord_mae,
            "elem_error": elem_error,
        })
    return pd.DataFrame(records)


def add_composite_score(df_err):
    """Add normalized composite reconstruction score to per-sample error DataFrame."""
    df = df_err.copy()
    for col in ["abc_mape", "ang_mape", "coord_mae", "elem_error"]:
        vals = df[col].values.astype(float)
        lo   = np.nanmin(vals)
        hi   = np.nanmax(vals)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            df[col + "_scaled"] = (vals - lo) / (hi - lo)
        else:
            df[col + "_scaled"] = 0.0
    df["composite_score"] = (
        0.25 * df["abc_mape_scaled"]  +
        0.25 * df["ang_mape_scaled"]  +
        0.35 * df["coord_mae_scaled"] +
        0.15 * df["elem_error_scaled"]
    )
    return df


def slot_accs(X_o, X_r, mask, max_elms, n_elm):
    """Per-slot element reconstruction accuracy for a given data subset."""
    res = []
    for i in range(max_elms):
        t = np.argmax(X_o[mask, :n_elm, i], axis=1)
        p = np.argmax(X_r[mask, :n_elm, i], axis=1)
        res.append(sk_metrics.accuracy_score(t, p))
    return res


def element_level_accuracy(X_o, X_r, mask, max_elms, n_elm, elm_str):
    """Per-element correct count and total count for a given data subset."""
    ec, et = {}, {}
    for i in range(max_elms):
        t = np.argmax(X_o[mask, :n_elm, i], axis=1)
        p = np.argmax(X_r[mask, :n_elm, i], axis=1)
        for ti, pi in zip(t, p):
            sym     = elm_str[ti]
            ec[sym] = ec.get(sym, 0) + int(ti == pi)
            et[sym] = et.get(sym, 0) + 1
    return ec, et


# ===========================================================================
# Bond analysis
# ===========================================================================

def get_bond_lengths(structure, nn_finder):
    """Extract bond lengths per sorted element pair from a pymatgen Structure."""
    bond_map = {}
    for i, site in enumerate(structure):
        try:
            nbs = nn_finder.get_nn_info(structure, i)
        except Exception:
            continue
        for nb in nbs:
            j = nb["site_index"]
            if j <= i:
                continue
            key = tuple(sorted([
                str(site.specie.symbol),
                str(structure[j].specie.symbol),
            ]))
            bond_map.setdefault(key, []).append(site.distance(structure[j]))
    return bond_map


def bond_mae(bo, br, pair):
    """MAE between original and reconstructed bond lengths for a given element pair."""
    key  = tuple(sorted(pair))
    lo   = bo.get(key, [])
    lr   = br.get(key, [])
    if len(lo) == 0 or len(lr) == 0:
        return np.nan
    n    = min(len(lo), len(lr))
    lo_s = np.array(sorted(lo)[:n], dtype=float)
    lr_s = np.array(sorted(lr)[:n], dtype=float)
    return float(np.mean(np.abs(lo_s - lr_s)))


def compute_bond_errors_5el_all(run_tag, X_recon_all, df_test, n_el_test,
                                 target_bond_pairs, nn_finder,
                                 max_elms, max_sites, elm_str):
    """Compute bond reconstruction errors for all 5-element test structures using temporary CIFs."""
    idx_5el = np.where(n_el_test == 5)[0]
    records = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for count, idx in enumerate(idx_5el):
            orig_path  = os.path.join(tmp_dir, f"orig_{idx:05d}.cif")
            recon_path = os.path.join(tmp_dir, f"recon_{idx:05d}.cif")
            try:
                with open(orig_path, "w", encoding="utf-8") as f:
                    f.write(df_test.iloc[idx]["cif"])
                write_single_cif_from_ftcp(
                    X_recon_all[idx], recon_path, max_elms, max_sites, elm_str
                )
                so = Structure.from_file(orig_path)
                sr = Structure.from_file(recon_path)
            except Exception:
                continue
            bo  = get_bond_lengths(so, nn_finder)
            br  = get_bond_lengths(sr, nn_finder)
            row = {"idx": int(idx), "n_el": 5}
            for pair in target_bond_pairs:
                row[f"{pair[0]}-{pair[1]}"] = bond_mae(bo, br, pair)
            records.append(row)
            if (count + 1) % 200 == 0:
                print(f"    5-el bond progress: {count+1}/{len(idx_5el)}")
    return pd.DataFrame(records)


# ===========================================================================
# Model loading and inference
# ===========================================================================

def tag_to_cfg(tag):
    """Parse a model tag string (e.g. 'CNN1 SVAE pattern16') into a config dict."""
    m = re.match(r"CNN(\d+)\s+(SVAE|USVAE)\s+pattern(\d+)", tag)
    if m is None:
        raise ValueError(f"Cannot parse tag: {tag}")
    return {
        "cnn":      m.group(1),
        "vae_type": m.group(2),
        "pattern":  m.group(3),
        "label":    tag,
    }


def _weight_path(cfg, base_dir, data_name):
    """Resolve weight file path from model configuration and base directory."""
    vt      = cfg["vae_type"]
    pat     = cfg["pattern"]
    log_dir = os.path.join(
        base_dir,
        f"{data_name}_result",
        f"CNN_{cfg['cnn']}",
        f"{vt}_result",
        "learning_log",
    )
    candidates = [
        f"VAE_weight_pattern{pat}.keras",
        f"VAE_weight_pattern{pat}.h5",
    ]
    for name in candidates:
        path = os.path.join(log_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(log_dir, candidates[0])


def build_and_load(cfg, X_test, y_test, base_dir, data_name):
    """Build FTCP_VAE model structure and load saved weights."""
    wp         = _weight_path(cfg, base_dir, data_name)
    supervised = (cfg["vae_type"] == "SVAE")
    if not os.path.exists(wp):
        raise FileNotFoundError(
            f"Weight file not found.\n"
            f"  Searched under: {os.path.dirname(wp)}\n"
            f"  Pattern: {cfg['pattern']}, VAE type: {cfg['vae_type']}\n"
            f"  Last tried: {wp}"
        )
    print(f"    Loading weights: {os.path.basename(wp)}")
    vae = FTCP_VAE(
        X_train=X_test,
        y_train=y_test,
        supervised=supervised,
        coeff_KL=2,
        coeff_prop=10,
        restart=False,
        network_pattern=cfg["pattern"],
        cnn_pattern=cfg["cnn"],
        csv_path=None,
        model_prefix=wp,
    )
    vae.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=lambda yt, yp: 0.0,
    )
    vae.load_weights(wp, by_name=True)
    return vae


def predict_recon(vae, cfg, X_test, y_test):
    """Run reconstruction prediction on the test set."""
    sup = (cfg["vae_type"] == "SVAE")
    return vae.predict([X_test, y_test] if sup else X_test, verbose=0)


def get_model_display_name(run_tag, conv_tag, best_tag):
    """Map a model tag to its display name for figures."""
    if run_tag == conv_tag:
        return "Conventional"
    elif run_tag == best_tag:
        return "Proposed"
    else:
        return run_tag


# ===========================================================================
# CIF generation
# ===========================================================================

def write_single_cif_from_ftcp(ftcp_one, out_path, max_elms, max_sites, elm_str):
    """Generate a CIF file from a single FTCP design vector via get_info."""
    if os.path.exists("designed_CIFs"):
        shutil.rmtree("designed_CIFs")
    get_info(
        ftcp_designs=ftcp_one[np.newaxis, ...],
        max_elms=max_elms,
        max_sites=max_sites,
        elm_str=elm_str,
        to_CIF=True,
        check_uniqueness=False,
    )
    src = os.path.join("designed_CIFs", "0.cif")
    if not os.path.exists(src):
        raise FileNotFoundError("designed_CIFs/0.cif was not created")
    shutil.move(src, out_path)


def generate_ranked_cif_pairs(X_recon, score_df, out_dir, prefix,
                               df_test, n_el_test, max_elms, max_sites, elm_str):
    """Generate original/reconstructed CIF pairs for a ranked list of structures."""
    ensure_dir(out_dir)
    rows   = score_df.reset_index(drop=True)
    status = []
    for rank, row in rows.iterrows():
        idx   = int(row["idx"])
        n_el  = int(row["n_elements"])
        score = float(row["composite_score"])
        stem  = (
            f"{prefix}_rank{rank+1:02d}_idx{idx:05d}_"
            f"nel{n_el}_score{score:.6f}"
        )
        orig_path  = os.path.join(out_dir, stem + "_original.cif")
        recon_path = os.path.join(out_dir, stem + "_reconstructed.cif")
        try:
            with open(orig_path, "w", encoding="utf-8") as f:
                f.write(df_test.iloc[idx]["cif"])
            write_single_cif_from_ftcp(
                X_recon[idx], recon_path, max_elms, max_sites, elm_str
            )
            Structure.from_file(orig_path)
            Structure.from_file(recon_path)
            st = "ok"
        except Exception as e:
            st = f"error:{type(e).__name__}:{e}"
        status.append({
            "rank":               rank + 1,
            "idx":                idx,
            "n_elements":         n_el,
            "composite_score":    score,
            "status":             st,
            "original_path":      orig_path,
            "reconstructed_path": recon_path,
        })
    pd.DataFrame(status).to_csv(
        os.path.join(out_dir, f"{prefix.lower()}_status.csv"), index=False
    )


def build_full_reconstructed_cif_cache(run_tag, X_recon, out_root,
                                        df_test, n_el_test, max_elms, max_sites, elm_str):
    """Build and save a complete CIF cache for all test set structures."""
    ensure_dir(out_root)
    records = []
    for idx in range(len(X_recon)):
        n_el       = int(n_el_test[idx])
        subset_dir = ensure_dir(os.path.join(out_root, f"{n_el}_elements"))
        orig_path  = os.path.join(subset_dir, f"orig_idx{idx:05d}.cif")
        recon_path = os.path.join(subset_dir, f"recon_idx{idx:05d}.cif")
        rec = {
            "idx":        idx,
            "n_elements": n_el,
            "orig_path":  orig_path,
            "recon_path": recon_path,
            "status":     "not_run",
        }
        try:
            with open(orig_path, "w", encoding="utf-8") as f:
                f.write(df_test.iloc[idx]["cif"])
            write_single_cif_from_ftcp(
                X_recon[idx], recon_path, max_elms, max_sites, elm_str
            )
            Structure.from_file(orig_path)
            Structure.from_file(recon_path)
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = f"error:{type(e).__name__}:{e}"
        records.append(rec)
        if (idx + 1) % 250 == 0:
            print(f"    Full CIF cache progress: {idx + 1}/{len(X_recon)}")
    df_cache = pd.DataFrame(records)
    df_cache.to_csv(
        os.path.join(out_root, f"full_cif_cache_{sanitize(run_tag)}.csv"), index=False
    )
    return df_cache


def load_cif_pairs_from_cache(cache_df):
    """Load valid original/reconstructed Structure pairs from a CIF cache DataFrame."""
    pairs = []
    ok_df = cache_df[cache_df["status"] == "ok"].copy()
    for _, row in ok_df.iterrows():
        try:
            so = Structure.from_file(row["orig_path"])
            sr = Structure.from_file(row["recon_path"])
            pairs.append({
                "idx":          int(row["idx"]),
                "n_el":         int(row["n_elements"]),
                "struct_orig":  so,
                "struct_recon": sr,
            })
        except Exception:
            continue
    return pairs


# ===========================================================================
# Plotting
# ===========================================================================

def plot_metric_breakdown_bar(metric_name, conv_vals, best_vals, out_path,
                               best_tag_label, model_color=None):
    """Bar chart comparing conventional and proposed models across element subsets."""
    if model_color is None:
        model_color = MODEL_COLOR
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    x     = np.arange(2)
    width = 0.34

    ax.bar(x - width / 2, conv_vals, width,
           color=model_color["Conventional"], edgecolor="black",
           linewidth=1.0, label="Conventional")
    ax.bar(x + width / 2, best_vals, width,
           color=model_color["Proposed"], edgecolor="black",
           linewidth=1.0, label="Proposed")

    ax.set_xticks(x)
    ax.set_xticklabels(["3-4 elements", "5 elements"])
    ax.set_ylabel(metric_name)

    all_vals = [v for v in list(conv_vals) + list(best_vals) if np.isfinite(v)]
    if len(all_vals) > 0:
        ymax  = max(all_vals)
        upper = ymax * 1.18 if ymax > 0 else 1.0
        ax.set_ylim(0, upper)

    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0),
              borderaxespad=0.0, frameon=True)
    savefig(fig, out_path)


def plot_violin_with_box(df_plot, x_col, y_col, hue_col, title,
                          y_label, out_prefix, palette):
    """Save boxplot and violin plot for a given DataFrame grouping."""
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.0))
    sns.boxplot(data=df_plot, x=x_col, y=y_col, hue=hue_col,
                palette=palette, ax=ax1, showfliers=False)
    ax1.set_title(title)
    ax1.set_ylabel(y_label)
    legend_outside_bottom_right(ax1)
    savefig(fig1, out_prefix + "_box.png")

    fig2, ax2 = plt.subplots(figsize=(9.0, 6.0))
    sns.violinplot(data=df_plot, x=x_col, y=y_col, hue=hue_col,
                   palette=palette, ax=ax2, cut=0, inner="box")
    ax2.set_title(title)
    ax2.set_ylabel(y_label)
    legend_outside_bottom_right(ax2)
    savefig(fig2, out_prefix + "_violin.png")


def plot_best_worst_tables(df_ranked, out_csv):
    """Save a ranked structure table to CSV."""
    df_ranked.to_csv(out_csv, index=False)


# ===========================================================================
# Result aggregation and best model detection (from 結果まとめ.py)
# ===========================================================================

def aggregate_results(base_dir, data_name, cnn_patterns, superviseds, network_pattern_numbers):
    """
    Aggregate all evaluation CSVs produced by main.py into a single DataFrame.

    Returns
    -------
    result_all_df : pd.DataFrame
        Rows are model tags (e.g. 'CNN1 SVAE pattern16').
        Columns are evaluation metrics from main.py output CSVs.
    """
    result_all_df = pd.DataFrame()

    for cnn, sup, net in product(cnn_patterns, superviseds, network_pattern_numbers):
        fp = os.path.join(
            base_dir,
            f"{data_name}_result",
            f"CNN_{cnn}",
            f"{sup}_result",
            "recon_result",
            f"FTCP_evaluation_pattern{net}.csv",
        )
        try:
            rdf = pd.read_csv(fp, index_col=0)
        except FileNotFoundError:
            continue
        rdf       = rdf.select_dtypes(include="number")
        rdf.index = [f"CNN{cnn} {sup} pattern{net}"]
        result_all_df = pd.concat([result_all_df, rdf])

    return result_all_df


def find_best_cfg(result_all_df):
    """
    Find the best model configuration by ranking across reconstruction metrics.

    Ranking criteria
    ----------------
    - Element accuracy          : higher is better
    - Lattice constant MAPE (%) : lower is better
    - Lattice angle MAPE (%)    : lower is better
    - Site coordinate MAE (frac): lower is better

    Returns
    -------
    best_tag : str
        Model tag of the best-ranked model.
    ranking_df : pd.DataFrame
        Full ranking table including per-metric ranks and total rank.
    """
    rank_specs = {
        "Element accuracy":           False,  # ascending=False -> higher is better
        "Lattice constant MAPE (%)":  True,   # ascending=True  -> lower is better
        "Lattice angle MAPE (%)":     True,
        "Site coordinate MAE (frac)": True,
    }

    ranking_df = result_all_df[list(rank_specs.keys())].dropna().copy()
    for col, ascending in rank_specs.items():
        ranking_df[f"rank_{sanitize(col)}"] = ranking_df[col].rank(
            ascending=ascending, method="min"
        )

    rank_cols              = [c for c in ranking_df.columns if c.startswith("rank_")]
    ranking_df["total_rank"] = ranking_df[rank_cols].sum(axis=1)

    best_tag = ranking_df["total_rank"].idxmin()
=======
# -*- coding: utf-8 -*-
"""
Result analysis utility functions for FTCP-VAE experiments.

All functions are stateless; experiment-level variables (paths, arrays, etc.)
are passed explicitly as arguments so this module can be imported cleanly.
"""

import os
import re
import glob
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from itertools import product
from pymatgen.core import Structure
from sklearn import metrics as sk_metrics
from module.FTCP import FTCP_VAE
from module.sampling import get_info


# Module-level style constants
MODEL_COLOR = {"Conventional": "black", "Proposed": "blue"}


# ===========================================================================
# General utilities
# ===========================================================================

def sanitize(s):
    """Convert a string to a filesystem-safe identifier."""
    s = s.replace(" ", "_").replace("/", "_").replace("\n", "_")
    s = s.replace("(", "").replace(")", "").replace("%", "pct")
    return re.sub(r"[^A-Za-z0-9_\-\.]", "", s)


def ensure_dir(path):
    """Create directory if it does not exist and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def safe_mean(values):
    """Return mean of finite values; return NaN if no finite values exist."""
    values = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def savefig(fig, path):
    """Apply tight layout, save figure at 300 dpi, and close."""
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def legend_outside_bottom_right(ax, ncol=1):
    """Place legend outside the axes, anchored at lower-left of the bbox."""
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        frameon=True,
        ncol=ncol,
    )


# ===========================================================================
# Metric computation
# ===========================================================================

def MAPE(yt, yp):
    """Mean Absolute Percentage Error."""
    yt = np.array(yt, dtype=float) + 1e-12
    yp = np.array(yp, dtype=float) + 1e-12
    return float(np.mean(np.abs((yt - yp) / yt)) * 100.0)


def MAE_site_coor(coor, coor_r, nsites):
    """MAE for site coordinates, evaluated only over occupied sites."""
    src = []
    rec = []
    for i in range(len(coor)):
        src.append(coor[i, :nsites[i], :])
        rec.append(coor_r[i, :nsites[i], :])
    if len(src) == 0:
        return np.nan
    return float(np.mean(np.abs(np.vstack(src) - np.vstack(rec))))


def elem_acc(X_o, X_r, max_elms, n_elm):
    """Mean element classification accuracy across all element slots."""
    accs = []
    for i in range(max_elms):
        t = np.argmax(X_o[:, :n_elm, i], axis=1)
        p = np.argmax(X_r[:, :n_elm, i], axis=1)
        accs.append(sk_metrics.accuracy_score(t, p))
    return float(np.mean(accs))


def extract_lattice_coords(X_dn, n_elm, max_sites):
    """Extract lattice constants, angles, and fractional site coordinates from FTCP tensor."""
    abc  = X_dn[:, n_elm,           :3]
    ang  = X_dn[:, n_elm + 1,       :3]
    coor = X_dn[:, n_elm + 2:n_elm + 2 + max_sites, :3]
    return abc, ang, coor


def compute_per_sample_ftcp_errors(X_orig, X_recon, nsites, n_el_test,
                                    n_elm, max_elms, max_sites):
    """Compute per-sample reconstruction errors for all FTCP components."""
    abc_o, ang_o, coor_o = extract_lattice_coords(X_orig, n_elm, max_sites)
    abc_r, ang_r, coor_r = extract_lattice_coords(X_recon, n_elm, max_sites)
    records = []
    for i in range(len(X_orig)):
        elem_slot_accs = []
        for slot in range(max_elms):
            t = int(np.argmax(X_orig[i, :n_elm, slot]))
            p = int(np.argmax(X_recon[i, :n_elm, slot]))
            elem_slot_accs.append(int(t == p))
        elem_error = 1.0 - float(np.mean(elem_slot_accs))
        abc_mape  = MAPE(abc_o[i:i+1], abc_r[i:i+1])
        ang_mape  = MAPE(ang_o[i:i+1], ang_r[i:i+1])
        coord_mae = MAE_site_coor(coor_o[i:i+1], coor_r[i:i+1], np.array([nsites[i]]))
        records.append({
            "idx":        i,
            "n_elements": int(n_el_test[i]),
            "abc_mape":   abc_mape,
            "ang_mape":   ang_mape,
            "coord_mae":  coord_mae,
            "elem_error": elem_error,
        })
    return pd.DataFrame(records)


def add_composite_score(df_err):
    """Add normalized composite reconstruction score to per-sample error DataFrame."""
    df = df_err.copy()
    for col in ["abc_mape", "ang_mape", "coord_mae", "elem_error"]:
        vals = df[col].values.astype(float)
        lo   = np.nanmin(vals)
        hi   = np.nanmax(vals)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            df[col + "_scaled"] = (vals - lo) / (hi - lo)
        else:
            df[col + "_scaled"] = 0.0
    df["composite_score"] = (
        0.25 * df["abc_mape_scaled"]  +
        0.25 * df["ang_mape_scaled"]  +
        0.35 * df["coord_mae_scaled"] +
        0.15 * df["elem_error_scaled"]
    )
    return df


def slot_accs(X_o, X_r, mask, max_elms, n_elm):
    """Per-slot element reconstruction accuracy for a given data subset."""
    res = []
    for i in range(max_elms):
        t = np.argmax(X_o[mask, :n_elm, i], axis=1)
        p = np.argmax(X_r[mask, :n_elm, i], axis=1)
        res.append(sk_metrics.accuracy_score(t, p))
    return res


def element_level_accuracy(X_o, X_r, mask, max_elms, n_elm, elm_str):
    """Per-element correct count and total count for a given data subset."""
    ec, et = {}, {}
    for i in range(max_elms):
        t = np.argmax(X_o[mask, :n_elm, i], axis=1)
        p = np.argmax(X_r[mask, :n_elm, i], axis=1)
        for ti, pi in zip(t, p):
            sym     = elm_str[ti]
            ec[sym] = ec.get(sym, 0) + int(ti == pi)
            et[sym] = et.get(sym, 0) + 1
    return ec, et


# ===========================================================================
# Bond analysis
# ===========================================================================

def get_bond_lengths(structure, nn_finder):
    """Extract bond lengths per sorted element pair from a pymatgen Structure."""
    bond_map = {}
    for i, site in enumerate(structure):
        try:
            nbs = nn_finder.get_nn_info(structure, i)
        except Exception:
            continue
        for nb in nbs:
            j = nb["site_index"]
            if j <= i:
                continue
            key = tuple(sorted([
                str(site.specie.symbol),
                str(structure[j].specie.symbol),
            ]))
            bond_map.setdefault(key, []).append(site.distance(structure[j]))
    return bond_map


def bond_mae(bo, br, pair):
    """MAE between original and reconstructed bond lengths for a given element pair."""
    key  = tuple(sorted(pair))
    lo   = bo.get(key, [])
    lr   = br.get(key, [])
    if len(lo) == 0 or len(lr) == 0:
        return np.nan
    n    = min(len(lo), len(lr))
    lo_s = np.array(sorted(lo)[:n], dtype=float)
    lr_s = np.array(sorted(lr)[:n], dtype=float)
    return float(np.mean(np.abs(lo_s - lr_s)))


def compute_bond_errors_5el_all(run_tag, X_recon_all, df_test, n_el_test,
                                 target_bond_pairs, nn_finder,
                                 max_elms, max_sites, elm_str):
    """Compute bond reconstruction errors for all 5-element test structures using temporary CIFs."""
    idx_5el = np.where(n_el_test == 5)[0]
    records = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for count, idx in enumerate(idx_5el):
            orig_path  = os.path.join(tmp_dir, f"orig_{idx:05d}.cif")
            recon_path = os.path.join(tmp_dir, f"recon_{idx:05d}.cif")
            try:
                with open(orig_path, "w", encoding="utf-8") as f:
                    f.write(df_test.iloc[idx]["cif"])
                write_single_cif_from_ftcp(
                    X_recon_all[idx], recon_path, max_elms, max_sites, elm_str
                )
                so = Structure.from_file(orig_path)
                sr = Structure.from_file(recon_path)
            except Exception:
                continue
            bo  = get_bond_lengths(so, nn_finder)
            br  = get_bond_lengths(sr, nn_finder)
            row = {"idx": int(idx), "n_el": 5}
            for pair in target_bond_pairs:
                row[f"{pair[0]}-{pair[1]}"] = bond_mae(bo, br, pair)
            records.append(row)
            if (count + 1) % 200 == 0:
                print(f"    5-el bond progress: {count+1}/{len(idx_5el)}")
    return pd.DataFrame(records)


# ===========================================================================
# Model loading and inference
# ===========================================================================

def tag_to_cfg(tag):
    """Parse a model tag string (e.g. 'CNN1 SVAE pattern16') into a config dict."""
    m = re.match(r"CNN(\d+)\s+(SVAE|USVAE)\s+pattern(\d+)", tag)
    if m is None:
        raise ValueError(f"Cannot parse tag: {tag}")
    return {
        "cnn":      m.group(1),
        "vae_type": m.group(2),
        "pattern":  m.group(3),
        "label":    tag,
    }


def _weight_path(cfg, base_dir, data_name):
    """Resolve weight file path from model configuration and base directory."""
    vt      = cfg["vae_type"]
    pat     = cfg["pattern"]
    log_dir = os.path.join(
        base_dir,
        f"{data_name}_result",
        f"CNN_{cfg['cnn']}",
        f"{vt}_result",
        "learning_log",
    )
    candidates = [
        f"VAE_weight_pattern{pat}.keras",
        f"VAE_weight_pattern{pat}.h5",
    ]
    for name in candidates:
        path = os.path.join(log_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(log_dir, candidates[0])


def build_and_load(cfg, X_test, y_test, base_dir, data_name):
    """Build FTCP_VAE model structure and load saved weights."""
    wp         = _weight_path(cfg, base_dir, data_name)
    supervised = (cfg["vae_type"] == "SVAE")
    if not os.path.exists(wp):
        raise FileNotFoundError(
            f"Weight file not found.\n"
            f"  Searched under: {os.path.dirname(wp)}\n"
            f"  Pattern: {cfg['pattern']}, VAE type: {cfg['vae_type']}\n"
            f"  Last tried: {wp}"
        )
    print(f"    Loading weights: {os.path.basename(wp)}")
    vae = FTCP_VAE(
        X_train=X_test,
        y_train=y_test,
        supervised=supervised,
        coeff_KL=2,
        coeff_prop=10,
        restart=False,
        network_pattern=cfg["pattern"],
        cnn_pattern=cfg["cnn"],
        csv_path=None,
        model_prefix=wp,
    )
    vae.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=lambda yt, yp: 0.0,
    )
    vae.load_weights(wp, by_name=True)
    return vae


def predict_recon(vae, cfg, X_test, y_test):
    """Run reconstruction prediction on the test set."""
    sup = (cfg["vae_type"] == "SVAE")
    return vae.predict([X_test, y_test] if sup else X_test, verbose=0)


def get_model_display_name(run_tag, conv_tag, best_tag):
    """Map a model tag to its display name for figures."""
    if run_tag == conv_tag:
        return "Conventional"
    elif run_tag == best_tag:
        return "Proposed"
    else:
        return run_tag


# ===========================================================================
# CIF generation
# ===========================================================================

def write_single_cif_from_ftcp(ftcp_one, out_path, max_elms, max_sites, elm_str):
    """Generate a CIF file from a single FTCP design vector via get_info."""
    if os.path.exists("designed_CIFs"):
        shutil.rmtree("designed_CIFs")
    get_info(
        ftcp_designs=ftcp_one[np.newaxis, ...],
        max_elms=max_elms,
        max_sites=max_sites,
        elm_str=elm_str,
        to_CIF=True,
        check_uniqueness=False,
    )
    src = os.path.join("designed_CIFs", "0.cif")
    if not os.path.exists(src):
        raise FileNotFoundError("designed_CIFs/0.cif was not created")
    shutil.move(src, out_path)


def generate_ranked_cif_pairs(X_recon, score_df, out_dir, prefix,
                               df_test, n_el_test, max_elms, max_sites, elm_str):
    """Generate original/reconstructed CIF pairs for a ranked list of structures."""
    ensure_dir(out_dir)
    rows   = score_df.reset_index(drop=True)
    status = []
    for rank, row in rows.iterrows():
        idx   = int(row["idx"])
        n_el  = int(row["n_elements"])
        score = float(row["composite_score"])
        stem  = (
            f"{prefix}_rank{rank+1:02d}_idx{idx:05d}_"
            f"nel{n_el}_score{score:.6f}"
        )
        orig_path  = os.path.join(out_dir, stem + "_original.cif")
        recon_path = os.path.join(out_dir, stem + "_reconstructed.cif")
        try:
            with open(orig_path, "w", encoding="utf-8") as f:
                f.write(df_test.iloc[idx]["cif"])
            write_single_cif_from_ftcp(
                X_recon[idx], recon_path, max_elms, max_sites, elm_str
            )
            Structure.from_file(orig_path)
            Structure.from_file(recon_path)
            st = "ok"
        except Exception as e:
            st = f"error:{type(e).__name__}:{e}"
        status.append({
            "rank":               rank + 1,
            "idx":                idx,
            "n_elements":         n_el,
            "composite_score":    score,
            "status":             st,
            "original_path":      orig_path,
            "reconstructed_path": recon_path,
        })
    pd.DataFrame(status).to_csv(
        os.path.join(out_dir, f"{prefix.lower()}_status.csv"), index=False
    )


def build_full_reconstructed_cif_cache(run_tag, X_recon, out_root,
                                        df_test, n_el_test, max_elms, max_sites, elm_str):
    """Build and save a complete CIF cache for all test set structures."""
    ensure_dir(out_root)
    records = []
    for idx in range(len(X_recon)):
        n_el       = int(n_el_test[idx])
        subset_dir = ensure_dir(os.path.join(out_root, f"{n_el}_elements"))
        orig_path  = os.path.join(subset_dir, f"orig_idx{idx:05d}.cif")
        recon_path = os.path.join(subset_dir, f"recon_idx{idx:05d}.cif")
        rec = {
            "idx":        idx,
            "n_elements": n_el,
            "orig_path":  orig_path,
            "recon_path": recon_path,
            "status":     "not_run",
        }
        try:
            with open(orig_path, "w", encoding="utf-8") as f:
                f.write(df_test.iloc[idx]["cif"])
            write_single_cif_from_ftcp(
                X_recon[idx], recon_path, max_elms, max_sites, elm_str
            )
            Structure.from_file(orig_path)
            Structure.from_file(recon_path)
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = f"error:{type(e).__name__}:{e}"
        records.append(rec)
        if (idx + 1) % 250 == 0:
            print(f"    Full CIF cache progress: {idx + 1}/{len(X_recon)}")
    df_cache = pd.DataFrame(records)
    df_cache.to_csv(
        os.path.join(out_root, f"full_cif_cache_{sanitize(run_tag)}.csv"), index=False
    )
    return df_cache


def load_cif_pairs_from_cache(cache_df):
    """Load valid original/reconstructed Structure pairs from a CIF cache DataFrame."""
    pairs = []
    ok_df = cache_df[cache_df["status"] == "ok"].copy()
    for _, row in ok_df.iterrows():
        try:
            so = Structure.from_file(row["orig_path"])
            sr = Structure.from_file(row["recon_path"])
            pairs.append({
                "idx":          int(row["idx"]),
                "n_el":         int(row["n_elements"]),
                "struct_orig":  so,
                "struct_recon": sr,
            })
        except Exception:
            continue
    return pairs


# ===========================================================================
# Plotting
# ===========================================================================

def plot_metric_breakdown_bar(metric_name, conv_vals, best_vals, out_path,
                               best_tag_label, model_color=None):
    """Bar chart comparing conventional and proposed models across element subsets."""
    if model_color is None:
        model_color = MODEL_COLOR
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    x     = np.arange(2)
    width = 0.34

    ax.bar(x - width / 2, conv_vals, width,
           color=model_color["Conventional"], edgecolor="black",
           linewidth=1.0, label="Conventional")
    ax.bar(x + width / 2, best_vals, width,
           color=model_color["Proposed"], edgecolor="black",
           linewidth=1.0, label="Proposed")

    ax.set_xticks(x)
    ax.set_xticklabels(["3-4 elements", "5 elements"])
    ax.set_ylabel(metric_name)

    all_vals = [v for v in list(conv_vals) + list(best_vals) if np.isfinite(v)]
    if len(all_vals) > 0:
        ymax  = max(all_vals)
        upper = ymax * 1.18 if ymax > 0 else 1.0
        ax.set_ylim(0, upper)

    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0),
              borderaxespad=0.0, frameon=True)
    savefig(fig, out_path)


def plot_violin_with_box(df_plot, x_col, y_col, hue_col, title,
                          y_label, out_prefix, palette):
    """Save boxplot and violin plot for a given DataFrame grouping."""
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.0))
    sns.boxplot(data=df_plot, x=x_col, y=y_col, hue=hue_col,
                palette=palette, ax=ax1, showfliers=False)
    ax1.set_title(title)
    ax1.set_ylabel(y_label)
    legend_outside_bottom_right(ax1)
    savefig(fig1, out_prefix + "_box.png")

    fig2, ax2 = plt.subplots(figsize=(9.0, 6.0))
    sns.violinplot(data=df_plot, x=x_col, y=y_col, hue=hue_col,
                   palette=palette, ax=ax2, cut=0, inner="box")
    ax2.set_title(title)
    ax2.set_ylabel(y_label)
    legend_outside_bottom_right(ax2)
    savefig(fig2, out_prefix + "_violin.png")


def plot_best_worst_tables(df_ranked, out_csv):
    """Save a ranked structure table to CSV."""
    df_ranked.to_csv(out_csv, index=False)


# ===========================================================================
# Result aggregation and best model detection (from 結果まとめ.py)
# ===========================================================================

def aggregate_results(base_dir, data_name, cnn_patterns, superviseds, network_pattern_numbers):
    """
    Aggregate all evaluation CSVs produced by main.py into a single DataFrame.

    Returns
    -------
    result_all_df : pd.DataFrame
        Rows are model tags (e.g. 'CNN1 SVAE pattern16').
        Columns are evaluation metrics from main.py output CSVs.
    """
    result_all_df = pd.DataFrame()

    for cnn, sup, net in product(cnn_patterns, superviseds, network_pattern_numbers):
        fp = os.path.join(
            base_dir,
            f"{data_name}_result",
            f"CNN_{cnn}",
            f"{sup}_result",
            "recon_result",
            f"FTCP_evaluation_pattern{net}.csv",
        )
        try:
            rdf = pd.read_csv(fp, index_col=0)
        except FileNotFoundError:
            continue
        rdf       = rdf.select_dtypes(include="number")
        rdf.index = [f"CNN{cnn} {sup} pattern{net}"]
        result_all_df = pd.concat([result_all_df, rdf])

    return result_all_df


def find_best_cfg(result_all_df):
    """
    Find the best model configuration by ranking across reconstruction metrics.

    Ranking criteria
    ----------------
    - Element accuracy          : higher is better
    - Lattice constant MAPE (%) : lower is better
    - Lattice angle MAPE (%)    : lower is better
    - Site coordinate MAE (frac): lower is better

    Returns
    -------
    best_tag : str
        Model tag of the best-ranked model.
    ranking_df : pd.DataFrame
        Full ranking table including per-metric ranks and total rank.
    """
    rank_specs = {
        "Element accuracy":           False,  # ascending=False -> higher is better
        "Lattice constant MAPE (%)":  True,   # ascending=True  -> lower is better
        "Lattice angle MAPE (%)":     True,
        "Site coordinate MAE (frac)": True,
    }

    ranking_df = result_all_df[list(rank_specs.keys())].dropna().copy()
    for col, ascending in rank_specs.items():
        ranking_df[f"rank_{sanitize(col)}"] = ranking_df[col].rank(
            ascending=ascending, method="min"
        )

    rank_cols              = [c for c in ranking_df.columns if c.startswith("rank_")]
    ranking_df["total_rank"] = ranking_df[rank_cols].sum(axis=1)

    best_tag = ranking_df["total_rank"].idxmin()
>>>>>>> 2.15.X
    return best_tag, ranking_df