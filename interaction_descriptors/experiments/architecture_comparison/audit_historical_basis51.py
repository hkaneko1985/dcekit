#!/usr/bin/env python3
"""Distinguish historical X+D from the protected X+D+M basis.

This is a numerical baseline audit, not an extension or modification of the
stage-51 prespecified comparison. It runs 12 additional outer X+D fits.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import run_fusion51 as r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root",type=Path,default=r.ROOT.parent/"archives")
    args = ap.parse_args()
    saved = args.source_root/"Interaction_Descriptors_Learning_Curves_v44_20261106/results/jobs"
    rows = []
    for case in ("metals_volume", "oxides_formation"):
        d = dict(np.load(r.ROOT/f"inputs/{case}__101.npz"))
        F, y = np.c_[d["X"],d["D"]], d["y"]
        splits = json.loads((r.ROOT/f"inputs/{case}_splits.json").read_text())
        for repeat,folds in enumerate(splits):
            pred = np.zeros(len(y))
            counts = np.zeros(len(y),int)
            for fold in folds:
                r.split_check(d,fold)
                sse = np.zeros(len(r.GRID))
                for inner in fold["inner"]:
                    a,b = np.array(inner["train"]),np.array(inner["valid"])
                    A,B,_ = r.prep(F[a],F[b])
                    pp = r.predict_grid(A,y[a],B)
                    sse += ((pp-y[b,None])**2).sum(0)
                best = int(np.argmin(sse))
                tr,te = np.array(fold["train"]),np.array(fold["test"])
                A,B,_ = r.prep(F[tr],F[te])
                pred[te] = r.predict_one(A,y[tr],B,r.GRID[best])
                counts[te] += 1
            assert (counts==1).all()
            for label, prediction, nfits in (("X+D",pred,3),("X+D+M",None,0)):
                key = "XD" if label=="X+D" else "XDM"
                history = dict(np.load(saved/f"{case}__64__NN__101__{key}__{repeat}.npz"))
                if prediction is None:
                    now = dict(np.load(r.ROOT/f"results/jobs/{case}__basis__0__r{repeat}.npz"))
                    prediction = now["prediction"][np.argsort(now["indices"])]
                assert np.array_equal(y,history["y"])
                diff = float(np.max(abs(prediction-history["pred"])))
                assert diff < 1e-9
                rows.append(dict(case=case,baseline=label,repeat=repeat,
                    current_rmse=r.metrics(y,prediction)["rmse"],
                    historical_rmse=r.metrics(y,history["pred"])["rmse"],
                    max_abs_prediction_difference=diff,
                    additional_outer_fits=nfits,
                    meaning="composition+fixed descriptors" if label=="X+D" else "composition+fixed descriptors+24 applicability masks"))
    r.write_csv(r.ROOT/"results/historical_baseline_recheck.csv",rows)
    summary = []
    for case in ("metals_volume","oxides_formation"):
        for label in ("X+D","X+D+M"):
            vv=[v for v in rows if v["case"]==case and v["baseline"]==label]
            summary.append(dict(case=case,baseline=label,
                rmse_mean=float(np.mean([v["current_rmse"]for v in vv])),
                historical_rmse_mean=float(np.mean([v["historical_rmse"]for v in vv])),
                max_abs_prediction_difference=max(v["max_abs_prediction_difference"]for v in vv)))
    r.dump(r.ROOT/"results/historical_baseline_recheck.json",dict(
        status="additional numerical audit only; fixed 100 configuration-repeat protocol unchanged",
        explanation="Inorganic B is X+D+M in stage51, matching the base of inherited all_frozen feature sets. Historical manuscript X+D omits M and is a different comparator.",
        extra_outer_fits=12, extra_prediction_events=2*(143+120), summary=summary))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
