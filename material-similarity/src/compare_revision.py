#!/usr/bin/env python3
"""Quantify revision impacts; never use the historical adapter for new analysis."""
import sys,csv,json
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'provenance')]
from run_validation import read_records,select_process_method_subset,select_records
from material_similarity import MaterialSimilarityEngine as E,DEFAULT_VIEWS
from material_similarity.adapters import adapt_nanomine
from material_similarity.distance import fit_numeric_scales
from legacy_adapter_v010 import adapt_nanomine as legacy
from audit_sources import write_csv

def main():
 raw=read_records();byid={x['sample_id']:x for x in raw};affected=sorted({x['sample_id'] for x in csv.DictReader((ROOT/'results/revision/duplicate_settings.csv').open())})
 display=select_process_method_subset(raw)[0]
 subset={x['sample_id']:x for x in display}
 for id in affected:subset[id]=byid[id]
 rows=list(subset.values());old=[legacy(x) for x in rows];new=[adapt_nanomine(x) for x in rows];repaired=[]
 # Isolate overwriting within the corrected representation: last-value collapse
 # is an audit comparator, not a supported default.
 for r in new:
  steps=[]
  for s in r.process_steps:
   values={}
   for k,v in s.settings.items():
    if v.kind in {'numeric_multiset','categorical_multiset'}:
     last=s.setting_provenance[k][-1]['normalization']['value']
     v=replace(v,value=last,kind=v.kind.replace('_multiset',''))
    values[k]=v
   steps.append(replace(s,settings=values))
  repaired.append(replace(r,process_steps=tuple(steps)))
 reference_scales=fit_numeric_scales([adapt_nanomine(x) for x in raw])
 # Previous display used scales fitted to its own records. The augmented
 # audit gallery below uses its own historical fit, disclosed in the output.
 eo=E(old);en=E(new,numeric_scales=reference_scales);el=E(repaired,numeric_scales=reference_scales)
 out=[]
 for id in affected:
  i=next(i for i,x in enumerate(rows) if x['sample_id']==id)
  for viewname in ['experimental_protocol','balanced_instance']:
   view=DEFAULT_VIEWS[viewname];ds=[]
   for j in range(len(rows)):
    if j==i:continue
    ds.append((j,eo.compare(old[i],old[j],view).composite.reported,en.compare(new[i],new[j],view).composite.reported,el.compare(repaired[i],repaired[j],view).composite.reported))
   a=np.array(ds);oo=[int(x) for x in a[np.argsort(a[:,1],kind='stable'),0][:5]];nn=[int(x) for x in a[np.argsort(a[:,2],kind='stable'),0][:5]];ll=[int(x) for x in a[np.argsort(a[:,3],kind='stable'),0][:5]]
   out.append({'sample_id':id,'view':viewname,'n_gallery':len(ds),'old_to_revised_mean_abs_distance_change':float(np.mean(abs(a[:,1]-a[:,2]))),'old_to_revised_top5_overlap':len(set(oo)&set(nn))/5,'overwrite_fix_only_mean_abs_distance_change':float(np.mean(abs(a[:,3]-a[:,2]))),'overwrite_fix_only_top5_overlap':len(set(ll)&set(nn))/5})
 write_csv(ROOT/'results/revision/affected_record_impact.csv',out)
 # Actual Figure 5--7 aggregates before/after revision from supplied artifacts.
 olddir=ROOT/'provenance/results_v010';newdir=ROOT/'results';changes=[]
 for filename,keys in [('api_nanomine_validation.csv',['view','bound']),('representative_pair_views.csv',['view']),('nanomine_material_system_separation.csv',['view']),('nanomine_process_method_separation.csv',['view']),('api_htem_validation.csv',['view','bound'])]:
  a=list(csv.DictReader((olddir/filename).open()));b=list(csv.DictReader((newdir/filename).open()));lookup={tuple(x[k] for k in keys):x for x in a}
  for row in b:
   key=tuple(row[k] for k in keys);previous=lookup[key]
   for metric,val in row.items():
    try:before=float(previous[metric]);after=float(val)
    except (ValueError,KeyError):continue
    if abs(before-after)>1e-9:changes.append({'file':filename,'condition':' / '.join(key),'metric':metric,'before':before,'after':after,'change':after-before})
 write_csv(ROOT/'results/revision/before_after_metrics.csv',changes)
 print('Impact audit:',len(affected),'affected records;',len(changes),'changed aggregate values')
if __name__=='__main__':main()
