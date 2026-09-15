#!/usr/bin/env python3
"""Fixed-rule interpretation of close relations under incomplete settings.

No physical target, fitted weights, imputation or chemical ground truth is used.
The reference is each pair's distance before experimental withholding. This
measures the cost of conservative metadata interpretation, not superiority of
clustering or property prediction. See the prospectively frozen configuration.
"""
import argparse, json, sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from run_revision_evaluation import read_records,select_process_method_subset,setting_fields,mask,fit_numeric_scales
from material_similarity import MaterialSimilarityEngine as E,DEFAULT_VIEWS
from material_similarity.adapters import adapt_nanomine
from audit_sources import write_csv

def summarize_rows(rows, cfg):
    """Article-block bootstrap summaries shared with the unmasked baseline."""
    grouped=defaultdict(list)
    for row in rows:grouped[(row['view'],row['mask_fraction'],row['distance_threshold'],row['rule'])].append(row)
    summary=[]
    for key,values in sorted(grouped.items()):
        counts=defaultdict(lambda:np.zeros(5))
        for row in values:
            counts[row['article_group']]+=np.array([row[k] for k in ['eligible_pairs','accepted_pairs','contradictory_pairs','reference_close_pairs','retained_reference_close_pairs']])
        array=np.array([counts[g] for g in sorted(counts)]);totals=array.sum(axis=0)
        brng=np.random.default_rng(cfg['seed']+1);bs=array[brng.integers(0,len(array),(cfg['bootstrap']['replicates'],len(array)))].sum(axis=1)
        result=dict(zip(['view','mask_fraction','distance_threshold','rule'],key))
        result.update(dict(zip(['eligible_pairs','accepted_pairs','contradictory_pairs','reference_close_pairs','retained_reference_close_pairs'],map(int,totals))))
        result.update(n_article_groups=len(array),n_query_runs=len(values))
        for name,num,den in [('retained_fraction',1,0),('contradiction_fraction',2,1),('reference_close_recall',4,3)]:
            result[name]=float(totals[num]/totals[den]) if totals[den] else ''
            draws=bs[bs[:,den]>0];ratios=draws[:,num]/draws[:,den] if len(draws) else []
            result[name+'_ci_low']=float(np.quantile(ratios,.025)) if len(ratios) else ''
            result[name+'_ci_high']=float(np.quantile(ratios,.975)) if len(ratios) else ''
        summary.append(result)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/revision');args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/interval_utility.json').read_text());rng=np.random.default_rng(cfg['seed'])
    raw=read_records();chosen,_=select_process_method_subset(raw)
    records=[adapt_nanomine(r) for r in chosen];full=[adapt_nanomine(r) for r in raw]
    groups=[r['paper_group'] for r in chosen]
    scales={g:fit_numeric_scales([r for r in full if r.context['paper_group']!=g]) for g in sorted(set(groups))}
    rows=[];audit=[];outside=0
    for i,q in enumerate(records):
        if i%20==0:print(f'Interval interpretation: query {i+1}/{len(records)}',flush=True)
        gallery=[j for j in range(len(records)) if groups[j]!=groups[i]]
        engine=E(records,numeric_scales=scales[groups[i]],feature_policy='fixed_schema')
        base={v:[] for v in cfg['views']}
        for j in gallery:
            facets={flag:engine.facet_distances(q,records[j],soft_identity=flag) for flag in {DEFAULT_VIEWS[v].soft_identity for v in base}}
            for view in base:
                d=engine._composite(facets[DEFAULT_VIEWS[view].soft_identity],DEFAULT_VIEWS[view].facet_weights)
                base[view].append(np.nan if d.reported is None else d.reported)
        base={v:np.array(x) for v,x in base.items()}
        fields=setting_fields(q)
        for fraction in cfg['query_mask_fractions']:
            for rep in range(cfg['replicates']):
                count=math.ceil(fraction*len(fields))
                keys={fields[k] for k in rng.choice(len(fields),count,replace=False)} if count else set()
                mq=mask(q,keys)
                audit.append({'sample_id':q.record_id,'article_group':groups[i],'mask_fraction':fraction,'replicate':rep,
                              'reported_setting_count':len(fields),'masked_keys':json.dumps(sorted(keys))})
                masked={v:[] for v in base}
                for j in gallery:
                    facets={flag:engine.facet_distances(mq,records[j],soft_identity=flag) for flag in {DEFAULT_VIEWS[v].soft_identity for v in masked}}
                    for view in masked:
                        d=engine._composite(facets[DEFAULT_VIEWS[view].soft_identity],DEFAULT_VIEWS[view].facet_weights)
                        masked[view].append((np.nan if d.reported is None else d.reported,d.lower,d.upper,d.common_reported_fraction))
                for view,values in masked.items():
                    point,lower,upper,coverage=np.asarray(values).T
                    valid=np.isfinite(base[view]); defined=np.isfinite(point)
                    outside+=int(np.sum(valid & ((base[view]<lower-1e-10)|(base[view]>upper+1e-10))))
                    for threshold in cfg['distance_thresholds']:
                        reference_close=valid & (base[view]<=threshold+1e-10)
                        point_close=valid & defined & (point<=threshold+1e-10)
                        rules={'reported_point':point_close,'interval_upper':valid & (upper<=threshold+1e-10)}
                        for gate in cfg['coverage_thresholds']:
                            rules[f'coverage_{gate:g}']=point_close & (coverage>=gate-1e-10)
                        for rule,accepted in rules.items():
                            rows.append({'sample_id':q.record_id,'article_group':groups[i],'view':view,'mask_fraction':fraction,
                              'replicate':rep,'distance_threshold':threshold,'rule':rule,'eligible_pairs':int(valid.sum()),
                              'reference_close_pairs':int(reference_close.sum()),'accepted_pairs':int(accepted.sum()),
                              'contradictory_pairs':int(np.sum(accepted & ~reference_close)),
                              'retained_reference_close_pairs':int(np.sum(accepted & reference_close)),
                              'point_defined_pairs':int(np.sum(valid & defined))})
    write_csv(out/'interval_utility_queries.csv',rows);write_csv(out/'interval_utility_mask_audit.csv',audit)
    summary=summarize_rows(rows,cfg)
    write_csv(out/'interval_utility_summary.csv',summary)
    report={'config':cfg,'query_records':len(records),'article_groups':len(set(groups)),
            'reference_distance_outside_masked_interval':outside,'interpretation':'pre-mask documented distances only; no property or expert target',
            'guarantee_scope':'fixed feature universe and scales; masking only reported settings; interval decisions quantify an algebraic containment/retention trade-off'}
    (out/'interval_utility_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    assert outside==0,'Reference point is outside masked interval; inspect composite aggregation before interpreting decisions'
    # Paired source-group bootstrap: modest held-out metadata support, no best-view claim.
    import csv
    rr=[r for r in csv.DictReader((out/'retrieval_queries.csv').open()) if r['task']=='heldout_method']
    bygroup=defaultdict(lambda:defaultdict(list))
    for r in rr:bygroup[r['article_group']][r['method']].append(float(r['agreement_at_5']))
    paired=[]
    for comparator in ['random_gallery','equal_facets','gower_field_pool']:
        dif=np.array([np.mean(bygroup[g]['balanced_strict'])-np.mean(bygroup[g][comparator]) for g in sorted(bygroup)])
        brng=np.random.default_rng(cfg['seed']);boot=brng.choice(dif,(10000,len(dif)),replace=True).mean(axis=1)
        paired.append({'contrast':'balanced_strict - '+comparator,'article_macro_difference':dif.mean(),
                       'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),'n_article_groups':len(dif),'bootstrap_replicates':10000})
    write_csv(out/'heldout_paired_comparisons.csv',paired)
    print(json.dumps({'query_rows':len(rows),'summary_rows':len(summary),'outside_interval':outside}),flush=True)

if __name__=='__main__':main()
