# -*- coding: utf-8 -*-
"""
FTCP-VAE Training and Evaluation Pipeline

Author: Issa Onishi
Created: October 1, 2026
"""

import os, re, glob, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from module.result_analysis import (
    sanitize, ensure_dir, savefig, get_model_display_name, MODEL_COLOR,
)
import seaborn as sns


# ===========================================================================
# SECTION 0: SETTINGS (kept identical to 01/02/03)
# ===========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(BASE_DIR, "result_for_paper")

CIF_ROOT = os.path.join(OUT_ROOT, "cif_samples_100")
if not os.path.isdir(CIF_ROOT):
    raise FileNotFoundError(
        f"{CIF_ROOT} not found. Run 03_compare_conventional_vs_best.py first; "
        "it generates the fixed 100-sample CIF set (Section 5) that this "
        "script reads."
    )

best_model_path = os.path.join(OUT_ROOT, "best_model.json")
if not os.path.exists(best_model_path):
    raise FileNotFoundError(
        f"{best_model_path} not found. Run 02_evaluate_all_models.py first; "
        "it selects the best model and saves the choice here for this "
        "script to reuse."
    )
with open(best_model_path, "r", encoding="utf-8") as f:
    best_model_info = json.load(f)

best_tag = best_model_info["best_tag"]   # "Proposed" throughout
conv_tag = best_model_info["conv_tag"]   # "Conventional" throughout
print(f"  Conventional model: {conv_tag}")
print(f"  Proposed model:     {best_tag}")

out_add = ensure_dir(os.path.join(OUT_ROOT, "fig_additional"))
nn_finder = CrystalNN()


# ===========================================================================
# HELPER: load CIF pairs from the 100-sample directory tree
# (identical to the former learning_result_analysis.py helper)
# ===========================================================================

def _load_cif_pairs_from_dir(cif_root, run_tag):
    """Load original/reconstructed CIF pairs from the 100-sample directory.

    Returns a list of dicts with keys: idx (int), n_el (int),
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


pairs_by_tag = {}
for run_tag in [conv_tag, best_tag]:
    pairs = _load_cif_pairs_from_dir(CIF_ROOT, run_tag)
    if len(pairs) == 0:
        print(f"  WARNING: no CIF pairs found for {run_tag} under {CIF_ROOT}")
    pairs_by_tag[run_tag] = pairs
    print(f"  Loaded {len(pairs)} CIF pairs for {run_tag}")


# ===========================================================================
# SECTION A: COORDINATION NUMBER MATCH RATE (Figure S3)
# ===========================================================================

print("\n" + "=" * 60)
print("Coordination number match rate")
print("=" * 60)

cn_data_all = {}
for run_tag in [conv_tag, best_tag]:
    tag_s = sanitize(run_tag)
    recs = []
    for p in pairs_by_tag[run_tag]:
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
    print(f"  {run_tag}: {len(df_cn)} site-level CN comparisons saved")

# Exact match-rate numbers by subset, for citing precise figures in the text
# (the PNG below is not sufficient for that).
cn_rate_rows = []
for run_tag in [conv_tag, best_tag]:
    model_name = get_model_display_name(run_tag, conv_tag, best_tag)
    df_cn = cn_data_all[run_tag]
    for subset_key, n_vals in [("3-4", [3, 4]), ("5", [5])]:
        sub = df_cn[df_cn["n_el"].isin(n_vals)] if len(df_cn) > 0 else df_cn
        rate = float(sub["match"].mean()) if len(sub) > 0 else np.nan
        cn_rate_rows.append({
            "model": model_name, "tag": run_tag, "subset": subset_key,
            "n_comparisons": len(sub), "match_rate": rate,
        })
df_cn_rate = pd.DataFrame(cn_rate_rows)
df_cn_rate.to_csv(os.path.join(out_add, "cn_match_rate_summary.csv"), index=False)
print("  Saved: cn_match_rate_summary.csv")
print(df_cn_rate.to_string(index=False))

# Font sizes: 2.5 times those of the previous manuscript version
# (default 10 pt -> 25 pt). Tick labels are broken into two lines and the
# legend is placed inside the axes so that the enlarged text does not overlap.
with plt.rc_context({"font.size": 25}):
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    for subset_key, xpos in [("3-4", 0), ("5", 1)]:
        for model_name, offset in [("Conventional", -0.17), ("Proposed", 0.17)]:
            sub = df_cn_rate[(df_cn_rate["model"] == model_name) & (df_cn_rate["subset"] == subset_key)]
            val = float(sub["match_rate"].iloc[0]) if len(sub) > 0 else np.nan
            ax.bar(xpos + offset, val, 0.34,
                   color=MODEL_COLOR[model_name], edgecolor="black", linewidth=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["3-4\nelements", "5\nelements"])
    ax.set_ylabel("Coordination number match rate")
    ax.set_ylim(0, 1.08)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black", label="Conventional"),
            plt.Rectangle((0, 0), 1, 1, facecolor="blue",  edgecolor="black", label="Proposed"),
        ],
        loc="upper right",
    )
    ax.set_title("Coordination number accuracy")
    savefig(fig, os.path.join(out_add, "cn_match_rate_100.png"))
print("  Saved: cn_match_rate_100.png (Figure S3)")


# ===========================================================================
# SECTION B: SPACE GROUP CONSERVATION RATE (Figure S4)
# ===========================================================================

print("\n" + "=" * 60)
print("Space group conservation rate")
print("=" * 60)

sg_data_all = {}
for run_tag in [conv_tag, best_tag]:
    tag_s = sanitize(run_tag)
    model_name = get_model_display_name(run_tag, conv_tag, best_tag)
    recs = []
    for p in pairs_by_tag[run_tag]:
        try:
            sgo = SpacegroupAnalyzer(p["struct_orig"],  symprec=0.1).get_space_group_number()
            sgr = SpacegroupAnalyzer(p["struct_recon"], symprec=0.1).get_space_group_number()
            recs.append({
                "idx":    p["idx"],
                "subset": "5" if p["n_el"] == 5 else "3-4",
                "model":  model_name,
                "sg_orig": sgo, "sg_recon": sgr,
                "match":  int(sgo == sgr),
            })
        except Exception:
            continue
    df_sg = pd.DataFrame(recs)
    sg_data_all[run_tag] = df_sg
    df_sg.to_csv(os.path.join(out_add, f"spacegroup_100_{tag_s}.csv"), index=False)
    print(f"  {run_tag}: {len(df_sg)} structure-level space group comparisons saved")

# Exact conservation-rate numbers by subset, for citing precise figures in
# the text.
sg_rate_rows = []
for run_tag in [conv_tag, best_tag]:
    model_name = get_model_display_name(run_tag, conv_tag, best_tag)
    df_sg = sg_data_all[run_tag]
    for subset_key in ["3-4", "5"]:
        sub = df_sg[df_sg["subset"] == subset_key] if len(df_sg) > 0 else df_sg
        rate = float(sub["match"].mean()) if len(sub) > 0 else np.nan
        sg_rate_rows.append({
            "model": model_name, "tag": run_tag, "subset": subset_key,
            "n_structures": len(sub), "conservation_rate": rate,
        })
df_sg_rate = pd.DataFrame(sg_rate_rows)
df_sg_rate.to_csv(os.path.join(out_add, "spacegroup_conservation_summary.csv"), index=False)
print("  Saved: spacegroup_conservation_summary.csv")
print(df_sg_rate.to_string(index=False))

# Font sizes: 2.5 times those of the previous manuscript version
# (default 10 pt -> 25 pt). Tick labels are broken into two lines and the
# legend is placed inside the axes so that the enlarged text does not overlap.
with plt.rc_context({"font.size": 25}):
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    for subset_key, xpos in [("3-4", 0), ("5", 1)]:
        for model_name, offset in [("Conventional", -0.17), ("Proposed", 0.17)]:
            sub = df_sg_rate[(df_sg_rate["model"] == model_name) & (df_sg_rate["subset"] == subset_key)]
            val = float(sub["conservation_rate"].iloc[0]) if len(sub) > 0 else np.nan
            ax.bar(xpos + offset, val, 0.34,
                   color=MODEL_COLOR[model_name], edgecolor="black", linewidth=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["3-4\nelements", "5\nelements"])
    ax.set_ylabel("Space group conservation rate")
    ax.set_ylim(0, 1.08)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black", label="Conventional"),
            plt.Rectangle((0, 0), 1, 1, facecolor="blue",  edgecolor="black", label="Proposed"),
        ],
        loc="upper right",
    )
    ax.set_title("Space group conservation rate")
    savefig(fig, os.path.join(out_add, "spacegroup_conservation_100.png"))
print("  Saved: spacegroup_conservation_100.png (Figure S4)")

# ===========================================================================
# SECTION C: UNIT CELL VOLUME RELATIVE ERROR (Figure S2)
# (former Section 8b of learning_result_analysis.py, computed on the same
#  fixed 100-sample CIF set as Figures S3/S4)
# ===========================================================================

print("\n" + "=" * 60)
print("Unit cell volume relative error")
print("=" * 60)

vol_data_all = {}
for run_tag in [conv_tag, best_tag]:
    tag_s = sanitize(run_tag)
    recs = []
    for p in pairs_by_tag[run_tag]:
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

vol_plot_df = pd.concat([vol_data_all[t] for t in [conv_tag, best_tag]], ignore_index=True)
vol_summary = (vol_plot_df.groupby(["model", "subset"])["rel_err"]
               .agg(n="count", mean="mean", median="median").reset_index())
vol_summary.to_csv(os.path.join(out_add, "volume_error_summary.csv"), index=False)
print(vol_summary.to_string(index=False))
# Figure S2: (A) box plot and (B) violin plot in one figure.
# Font sizes of the axis labels, tick labels, titles and legend are 2 times
# those of the previous manuscript version; the panel labels (A)/(B) are
# unchanged (22 pt).
_vol_pal = {"Conventional": "gray", "Proposed": "steelblue"}
_hue_order = ["Conventional", "Proposed"]
vol_plot_df["subset"] = vol_plot_df["subset"].astype(str)
with plt.rc_context({"font.size": 10}):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    sns.boxplot(data=vol_plot_df, x="subset", y="rel_err", hue="model",
                order=["3-4", "5"], hue_order=_hue_order, palette=_vol_pal,
                showfliers=False, ax=axs[0])
    sns.violinplot(data=vol_plot_df, x="subset", y="rel_err", hue="model",
                   order=["3-4", "5"], hue_order=_hue_order, palette=_vol_pal,
                   cut=0, inner="box", ax=axs[1])
    for ax, letter in zip(axs, "AB"):
        ax.set_title("Unit cell volume reconstruction error", fontsize=22)
        ax.set_ylabel("Volume relative error (%)", fontsize=20)
        ax.set_xlabel("subset", fontsize=20)
        ax.tick_params(labelsize=20)
        ax.text(-0.12, 1.12, f"({letter})", transform=ax.transAxes,
                fontsize=22, fontweight="bold")
    axs[0].get_legend().remove()
    axs[1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0),
                  borderaxespad=0.0, fontsize=20)
    savefig(fig, os.path.join(out_add, "volume_error_box_violin.png"))
print("  Saved: volume_error_box_violin.png (Figure S2)")

print("\nDone. Outputs written to:", out_add)
