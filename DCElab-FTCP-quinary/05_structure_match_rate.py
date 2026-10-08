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
from pymatgen.analysis.structure_matcher import StructureMatcher

from module.result_analysis import (
    sanitize, ensure_dir, savefig, get_model_display_name, MODEL_COLOR,
)


# ===========================================================================
# SECTION 0: SETTINGS (kept identical to 01/02/03/04)
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

out_sm = ensure_dir(os.path.join(OUT_ROOT, "fig_structure_match"))

# Tolerances identical to CDVAE's RecEval (cdvae/scripts/compute_metrics.py)
matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)


# ===========================================================================
# HELPER: load CIF pairs from the 100-sample directory tree
# (identical to the helper in 04_narrow_inverse_design_claims.py)
# ===========================================================================

def _load_cif_pairs_from_dir(cif_root, run_tag):
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
# StructureMatcher-based match rate and RMSD
# ===========================================================================

print("\n" + "=" * 60)
print("StructureMatcher match rate (CDVAE tolerances: ltol=0.3, stol=0.5, angle_tol=10)")
print("=" * 60)

sm_data_all = {}
for run_tag in [conv_tag, best_tag]:
    tag_s = sanitize(run_tag)
    model_name = get_model_display_name(run_tag, conv_tag, best_tag)
    recs = []
    for p in pairs_by_tag[run_tag]:
        try:
            res = matcher.get_rms_dist(p["struct_recon"], p["struct_orig"])
        except Exception as e:
            res = None
        match = res is not None
        rmsd  = float(res[0]) if match else np.nan
        max_d = float(res[1]) if match else np.nan
        recs.append({
            "idx":    p["idx"],
            "subset": "5" if p["n_el"] == 5 else "3-4",
            "model":  model_name,
            "match":  int(match),
            "rmsd":   rmsd,
            "max_dist": max_d,
        })
    df_sm = pd.DataFrame(recs)
    sm_data_all[run_tag] = df_sm
    df_sm.to_csv(os.path.join(out_sm, f"structure_match_{tag_s}.csv"), index=False)
    print(f"  {run_tag}: {len(df_sm)} structure pairs compared, "
          f"{int(df_sm['match'].sum())} matched")

# Exact match-rate and mean-RMSD numbers by subset, for citing precise
# figures in the text / rebuttal letter.
sm_rows = []
for run_tag in [conv_tag, best_tag]:
    model_name = get_model_display_name(run_tag, conv_tag, best_tag)
    df_sm = sm_data_all[run_tag]
    for subset_key in ["3-4", "5", "all"]:
        sub = df_sm if subset_key == "all" else df_sm[df_sm["subset"] == subset_key]
        n = len(sub)
        match_rate = float(sub["match"].mean()) if n > 0 else np.nan
        mean_rmsd  = float(sub.loc[sub["match"] == 1, "rmsd"].mean()) if sub["match"].sum() > 0 else np.nan
        sm_rows.append({
            "model": model_name, "tag": run_tag, "subset": subset_key,
            "n_structures": n, "n_matched": int(sub["match"].sum()),
            "match_rate": match_rate, "mean_rmsd_matched": mean_rmsd,
        })
df_sm_rate = pd.DataFrame(sm_rows)
df_sm_rate.to_csv(os.path.join(out_sm, "structure_match_rate_summary.csv"), index=False)
print("  Saved: structure_match_rate_summary.csv")
print(df_sm_rate.to_string(index=False))

# (No figure is generated: the StructureMatcher results are not shown as a
#  figure in the manuscript. The numbers are in structure_match_rate_summary.csv.)

print("\nDone. Outputs written to:", out_sm)
