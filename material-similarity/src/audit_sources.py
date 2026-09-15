#!/usr/bin/env python3
"""Source-level audit; preserves raw evidence, never imputes experimental settings."""
import sys,json,csv,re,hashlib
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from run_validation import read_records,select_records,select_material_system_subset,select_process_method_subset
from material_similarity.adapters import adapt_nanomine
from material_similarity.normalization import normalize_setting,normalize_numeric,canonical_name,QUANTITIES,UNITS
from material_similarity.distance import fit_numeric_scales

def write_csv(path,rows):
 if rows:
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 out=ROOT/'results/revision';out.mkdir(parents=True,exist_ok=True)
 rows=read_records();audit=[];duplicates=[];provenance=[]
 for row in rows:
  doi=str(row.get('doi') or '');valid=bool(re.fullmatch(r'10\.\d{4,9}/\S+',doi,re.I))
  provenance.append({'sample_id':row['sample_id'],'article_group':row['paper_group'],'article_id':row['article_id'],'doi':doi,'identifier_status':'doi_formatted_unverified' if valid else 'placeholder_or_missing','sample_uri':row.get('sample_uri',row.get('uri',''))})
  for s in row['steps']:
   counts=Counter(canonical_name(x['key']) for x in s['settings'])
   for key,n in counts.items():
    if n>1:duplicates.append({'sample_id':row['sample_id'],'step_uri':s['uri'],'key':key,'number_of_values':n,'handling':'unordered_multiset_no_zone_imputation'})
   for item in s['settings']:
    p=normalize_setting(item['key'],str(item.get('raw',item.get('value',''))),str(item.get('raw_unit','')))
    audit.append({'sample_id':row['sample_id'],'scope':'process','source_uri':s['uri'],'key':item['key'],'raw':p.get('raw'),'raw_unit':p.get('raw_unit',''),'old_kind':item.get('kind'),'old_value':item.get('value'),'old_unit_group':item.get('unit_group'),'new_kind':p['kind'],'new_value':p.get('value'),'new_unit_group':p.get('unit_group'),'status':p.get('normalization_status'),'reason':p.get('reason',''),'primary_input':True})
  for c in row['components']:
   for item in c.get('attributes',[]):
    p=normalize_numeric(item['type'],str(item.get('raw_value',item.get('raw',''))),str(item.get('raw_unit','')))
    if p is None:continue
    audit.append({'sample_id':row['sample_id'],'scope':'component','source_uri':c.get('uri',''),'key':item['type'],'raw':p.get('raw'),'raw_unit':p.get('raw_unit',''),'old_kind':item.get('kind'),'old_value':item.get('value'),'old_unit_group':item.get('unit_group'),'new_kind':p['kind'],'new_value':p.get('value'),'new_unit_group':p.get('unit_group'),'status':p.get('normalization_status'),'reason':p.get('reason',''),'primary_input':canonical_name(item['type']) in {'mass fraction','volume fraction'}})
 write_csv(out/'normalization_audit.csv',audit);write_csv(out/'duplicate_settings.csv',duplicates);write_csv(out/'source_identifiers.csv',provenance)
 duplicate_ids={x['sample_id'] for x in duplicates};bad={x['sample_id'] for x in provenance if x['identifier_status']!='doi_formatted_unverified'}
 subsets={'full':rows,'api':select_records(rows),'systems':select_material_system_subset(rows)[0],'methods':select_process_method_subset(rows)[0]}
 summary={'source_records':len(rows),'source_article_groups':len(set(x['paper_group'] for x in rows)),
   'subsets':{k:{'n':len(v),'article_groups':len(set(x['paper_group'] for x in v)),'duplicate_setting_records':sum(x['sample_id'] in duplicate_ids for x in v),'placeholder_records':sum(x['sample_id'] in bad for x in v),'placeholder_article_groups':len(set(x['paper_group'] for x in v if x['sample_id'] in bad))} for k,v in subsets.items()},
   'quarantined_entries':sum(x['status']=='quarantined' for x in audit),'quarantined_process_entries':sum(x['status']=='quarantined' and x['scope']=='process' for x in audit),
   'quarantine_by_reason':dict(Counter(x['reason'] for x in audit if x['status']=='quarantined')),
   'quarantine_by_key':dict(Counter(x['key'] for x in audit if x['status']=='quarantined')),
   'numeric_scale_scope':'quantity_kind x canonical_unit, fitted to all 832 normalized primary records',
   'primary_component_features':'constituent names, mass fraction, volume fraction; excludes density, width, aspect ratio and specific surface area'}
 (out/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 normalized=[adapt_nanomine(x) for x in rows]
 (ROOT/'data/processed').mkdir(exist_ok=True)
 with (ROOT/'data/processed/nanomine_audited_records.jsonl').open('w') as f:
  for r in normalized:f.write(json.dumps(r.to_mapping(),sort_keys=True)+'\n')
 scales={k:{'scale':v.scale,'transform':v.transform} for k,v in fit_numeric_scales(normalized).items()}
 (out/'fixed_numeric_scales.json').write_text(json.dumps(scales,indent=2)+'\n')
 (ROOT/'config/numeric_dictionary.json').write_text(json.dumps({'quantity_types':QUANTITIES,'units':[{'quantity':q,'normalized_source_unit':u,'canonical_unit':v[0],'factor':v[1],'offset':v[2]} for (q,u),v in sorted(UNITS.items())],'unknown_unit_policy':'quarantine; retain raw value; no inference'},indent=2)+'\n')
 (ROOT/'data/nanomine_selection_manifest.json').write_text(json.dumps({'normalized_snapshot_created_at':'2026-09-10T00:59:08.885129+00:00','subsets':{k:[r['sample_id'] for r in v] for k,v in subsets.items()},'source_record_semantic_hashes':{r['sample_id']:hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest() for r in rows}},indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
