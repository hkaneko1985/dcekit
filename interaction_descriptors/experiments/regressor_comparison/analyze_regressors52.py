#!/usr/bin/env python3
"""Produce descriptive, non-inferential comparisons from the completed stage52 grid."""
import json
from pathlib import Path
import run_regressors52 as r


def main():
    status, summaries, comparisons = r.analyze()
    assert status["complete"], "Do not summarize a partially completed comparison grid"
    table = {(row["case"], row["regressor"], row["feature_set"]): row for row in summaries}
    per_case = {}
    all_paired = []
    for case in r.CASES:
        matrix = []
        for feature in r.FEATURES:
            matrix.append(dict(feature_set=feature, **{
                method: table[(case, method, feature)]["rmse_mean"] for method in r.ALL_REGRESSORS}))
        r.write_csv(r.ROOT / f"results/rmse_matrix_{case}.csv", matrix)
        selected = [row for row in summaries if row["case"] == case]
        original = min((row for row in selected if not row["feature_set"].startswith("fusion")), key=lambda row: row["rmse_mean"])
        added = min((row for row in selected if row["feature_set"].startswith("fusion")), key=lambda row: row["rmse_mean"])
        overall = min(selected, key=lambda row: row["rmse_mean"])
        per_case[case] = dict(
            best_original_condition=original, best_added_condition=added, best_overall_condition=overall,
            improvement_percent_best_added_vs_best_original=100 * (original["rmse_mean"] - added["rmse_mean"]) / original["rmse_mean"],
            regressor_comparisons=[row for row in comparisons if row["case"] == case])
        for method in r.ALL_REGRESSORS:
            entries = {feature: table[(case, method, feature)] for feature in r.FEATURES}
            old_best = min(r.FEATURES[:6], key=lambda feature: entries[feature]["rmse_mean"])
            new_best = min(r.FEATURES[6:], key=lambda feature: entries[feature]["rmse_mean"])
            a, b = entries[old_best]["rmse_mean"], entries[new_best]["rmse_mean"]
            all_paired.append(dict(case=case, regressor=method, best_original_feature=old_best,
                                   best_original_rmse=a, best_added_feature=new_best, best_added_rmse=b,
                                   improvement_percent=100 * (a - b) / a, improvement=b < a))
    r.write_csv(r.ROOT / "results/added_vs_best_original.csv", all_paired)
    analysis = dict(
        status="complete descriptive comparison", combined_condition_rows=len(summaries),
        unique_target_regressor_cells=len(comparisons),
        new_regressors=list(r.REGRESSORS), reused_regressor="KRR",
        scope="same saved development datasets and splits; not independent statistical trials",
        counts={
            "fusion_below_B": sum(row["fusion_improvement_percent_vs_B"] > 0 for row in comparisons),
            "fusion_below_best_single_family": sum(row["fusion_improvement_percent_vs_best_single"] > 0 for row in comparisons),
            "boruta_below_B": sum(row["boruta_improvement_percent_vs_B"] > 0 for row in comparisons),
            "boruta_below_fusion": sum(row["boruta_improvement_percent_vs_fusion"] > 0 for row in comparisons),
            "boruta_below_best_single_family": sum(row["boruta_improvement_percent_vs_best_single"] > 0 for row in comparisons),
            "best_added_below_best_original": sum(row["improvement"] for row in all_paired),
            "denominator": len(comparisons),
        },
        per_case=per_case,
        caution="Counts summarize correlated task/regressor cells and are not binomial trials; minimum errors are descriptive model comparisons, not prospectively validated selected models.",
    )
    r.dump(r.ROOT / "results/analysis_summary.json", analysis)
    print(json.dumps({"counts": analysis["counts"], "per_case": {
        case: {key: value for key, value in results.items() if key != "regressor_comparisons"}
        for case, results in per_case.items()}}, indent=2))


if __name__ == "__main__":
    main()
