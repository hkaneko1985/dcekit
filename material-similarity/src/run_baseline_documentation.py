#!/usr/bin/env python3
"""Unmasked documentation baseline for the unchanged Figure 7 decision grid.

One deterministic pass; no masks, targets, fitted weights, or imputation.
The 25/50/75% experiments and their random streams remain unchanged.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from run_interval_utility import (
    ROOT, read_records, select_process_method_subset, setting_fields,
    fit_numeric_scales, adapt_nanomine, E, DEFAULT_VIEWS, write_csv,
    summarize_rows,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/revision')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT/'config/interval_utility.json').read_text())
    raw = read_records()
    chosen, _ = select_process_method_subset(raw)
    records = [adapt_nanomine(r) for r in chosen]
    full = [adapt_nanomine(r) for r in raw]
    groups = [r['paper_group'] for r in chosen]
    scales = {g: fit_numeric_scales([r for r in full if r.context['paper_group'] != g])
              for g in sorted(set(groups))}
    rows = []
    ranges = {v: {'minimum_upper_distance': 1., 'maximum_common_coverage': 0.}
              for v in cfg['views']}
    zero_settings = [r.record_id for r in records if not setting_fields(r)]
    for i, query in enumerate(records):
        if i % 20 == 0:
            print(f'Unmasked baseline: query {i+1}/{len(records)}', flush=True)
        engine = E(records, numeric_scales=scales[groups[i]], feature_policy='fixed_schema')
        values = {v: [] for v in cfg['views']}
        for j, candidate in enumerate(records):
            if groups[j] == groups[i]:
                continue
            facets = {flag: engine.facet_distances(query, candidate, soft_identity=flag)
                      for flag in {DEFAULT_VIEWS[v].soft_identity for v in values}}
            for view in values:
                d = engine._composite(facets[DEFAULT_VIEWS[view].soft_identity],
                                      DEFAULT_VIEWS[view].facet_weights)
                values[view].append((np.nan if d.reported is None else d.reported,
                                     d.lower, d.upper, d.common_reported_fraction))
        for view, array in values.items():
            point, lower, upper, coverage = np.asarray(array).T
            valid = np.isfinite(point)
            assert np.all(point[valid] >= lower[valid]-1e-10)
            assert np.all(point[valid] <= upper[valid]+1e-10)
            ranges[view]['minimum_upper_distance'] = min(
                ranges[view]['minimum_upper_distance'], float(upper.min()))
            ranges[view]['maximum_common_coverage'] = max(
                ranges[view]['maximum_common_coverage'], float(coverage.max()))
            for threshold in cfg['distance_thresholds']:
                reference_close = valid & (point <= threshold+1e-10)
                rules = {'reported_point': reference_close,
                         'interval_upper': valid & (upper <= threshold+1e-10)}
                for gate in cfg['coverage_thresholds']:
                    rules[f'coverage_{gate:g}'] = reference_close & (coverage >= gate-1e-10)
                for rule, accepted in rules.items():
                    rows.append({'sample_id': query.record_id, 'article_group': groups[i],
                                 'view': view, 'mask_fraction': 0., 'replicate': 0,
                                 'distance_threshold': threshold, 'rule': rule,
                                 'eligible_pairs': int(valid.sum()),
                                 'reference_close_pairs': int(reference_close.sum()),
                                 'accepted_pairs': int(accepted.sum()),
                                 'contradictory_pairs': int(np.sum(accepted & ~reference_close)),
                                 'retained_reference_close_pairs': int(np.sum(accepted & reference_close)),
                                 'point_defined_pairs': int(valid.sum())})
    assert ranges['material_identity']['minimum_upper_distance'] >= .4-1e-10
    assert ranges['material_identity']['maximum_common_coverage'] <= .6+1e-10
    assert ranges['balanced_instance']['minimum_upper_distance'] >= .2-1e-10
    assert not any(r['contradictory_pairs'] for r in rows)
    summary = summarize_rows(rows, cfg)
    write_csv(out/'baseline_documentation_queries.csv', rows)
    write_csv(out/'baseline_documentation_summary.csv', summary)
    audit = {'query_records': len(records), 'article_groups': len(set(groups)),
             'scale_source_records': len(full), 'mask_fraction': 0., 'query_repetitions': 1,
             'query_rows': len(rows), 'summary_conditions': len(summary),
             'eligible_directed_pairs_per_condition': sorted({r['eligible_pairs'] for r in summary}),
             'records_without_reported_settings': zero_settings,
             'empty_masks_in_existing_25_50_75_percent_experiment':
                 len(zero_settings)*len(cfg['query_mask_fractions'])*cfg['replicates'],
             'observed_baseline_ranges': ranges,
             'rules_and_views': 'unchanged config/interval_utility.json; R3 descriptive baseline',
             'reference': 'documented unmasked pair distance; no physical target',
             'scope': 'pairwise decisions only; no cluster-membership or stability guarantee'}
    (out/'baseline_documentation_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
