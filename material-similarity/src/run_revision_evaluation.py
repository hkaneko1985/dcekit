#!/usr/bin/env python3
"""Target-free retrieval, held-out metadata, missingness and alignment evaluations.

All weight choices and selections are fixed in config/revision_evaluation.json.
These tasks assess metadata utility, not property transfer or chemical equivalence.
"""
from __future__ import annotations
import sys,json,csv,copy,re,math,argparse
from pathlib import Path
from dataclasses import replace
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from run_validation import read_records,select_process_method_subset,select_material_system_subset,stable_key,principal_coordinates
from material_similarity import DocumentedValue as V,MaterialInstance as M,ProcessStep as P,ReportingStatus as S,MaterialSimilarityEngine as E,ViewSpec,DEFAULT_VIEWS
from material_similarity.distance import FACET_NAMES,fit_numeric_scales,NumericScale,DistanceInterval,normalized_sequence_distance,jaccard_distance,normalize_text
from material_similarity.distance import step_setting_locations
from material_similarity.adapters import adapt_nanomine
from audit_sources import write_csv
CFG=json.loads((ROOT/'config/revision_evaluation.json').read_text());OUT=ROOT/'results/revision'
WEIGHTS=CFG['weights'];SEED=CFG['seed']

def signature(r):
 return repr((r.composition,r.material_identity,r.process_method,[(s.step_type,s.settings) for s in r.process_steps],r.process_settings))

def heldout(r):
 aliases=re.compile(r'\b(?:solution[ -]?process(?:ing)?|melt[ -]?mix(?:ing)?|in[ -]?situ[ -]?polymeri[sz]ation)\b',re.I)
 def clean_value(v):
  if v.status is S.REPORTED and v.kind in {'categorical','categorical_multiset','set'}:
   text=json.dumps(v.value)
   if aliases.search(text):return V.unknown(kind='set' if v.kind=='set' else 'categorical')
  return v
 steps=[]
 for s in r.process_steps:
  t=aliases.sub('',s.step_type).strip(' |-_') or 'unspecified operation'
  steps.append(P(t,{k:clean_value(v) for k,v in s.settings.items() if not aliases.search(k)}))
 out=replace(r,process_method=V.unknown(kind='set'),process_steps=tuple(steps),
   material_identity={k:clean_value(v) for k,v in r.material_identity.items() if not aliases.search(k)},context={})
 assert not aliases.search(signature(out))
 return out

def setting_fields(r):
 return [key for key,locations in step_setting_locations(r).items()
         if all(r.process_steps[i].settings[k].status is S.REPORTED for i,k in locations)]

def mask(r,keys,identity_keys=()):
 locations=step_setting_locations(r)
 hidden={location for key in keys for location in locations.get(key,[])}
 steps=[replace(s,settings={k:(V.unknown(kind=v.kind,unit_key=v.unit_key) if (i,k) in hidden else v)
                            for k,v in s.settings.items()}) for i,s in enumerate(r.process_steps)]
 return replace(r,process_steps=tuple(steps),material_identity={k:V.unknown(kind=v.kind,unit_key=v.unit_key) if k in identity_keys else v for k,v in r.material_identity.items()},context={})

def metrics_from_facets(engine,facets,left,right):
 results={}
 for name,weights in WEIGHTS.items():
  d=engine._composite(facets,weights);results[name]=1. if d.reported is None else d.reported
 # Flat Gower-style pooling: identical scalar/set/edit comparators and availability;
 # each identity/setting variable, rather than each facet, has equal weight.
 numer=denom=0.
 for f in facets.values():
  mass=f.active_weight*(f.common_reported_fraction+f.structural_difference_fraction)
  if f.reported is not None:numer+=mass*f.reported;denom+=mass
 results['gower_field_pool']=numer/denom if denom else 1.
 altered=dict(facets);altered['sequence']=facets['step_type']
 d=engine._composite(altered,WEIGHTS['balanced_strict']);results['sequence_as_set']=d.reported if d.reported is not None else 1.
 if left.process_sequence_status is S.REPORTED and right.process_sequence_status is S.REPORTED:
  ca=Counter(normalize_text(s.step_type) for s in left.process_steps);cb=Counter(normalize_text(s.step_type) for s in right.process_steps)
  total=sum((ca|cb).values());bag=1-sum((ca&cb).values())/total if total else 0.
  altered['sequence']=DistanceInterval(bag,bag,bag,1.,0.,0.,1.,0.)
 else:altered['sequence']=facets['sequence']
 d=engine._composite(altered,WEIGHTS['balanced_strict']);results['sequence_as_bag']=d.reported if d.reported is not None else 1.
 return results

def tie_rank_scores(dist,target):
 targetd=dist[target];a=int(np.sum(dist<targetd-1e-10));b=int(np.sum(np.abs(dist-targetd)<=1e-10))
 return {'mrr':float(np.mean(1/np.arange(a+1,a+b+1))), 'recall_at_1':min(max(1-a,0),b)/b,'recall_at_5':min(max(5-a,0),b)/b,'tie_size':b}

def knn_agreement(dist,labels,label,k=5):
 order=np.sort(dist);cut=order[min(k,len(order))-1]
 strict=dist<cut-1e-10;tied=np.abs(dist-cut)<=1e-10
 rem=min(k,len(dist))-strict.sum()
 return float(((labels[strict]==label).sum()+rem*np.mean(labels[tied]==label))/min(k,len(dist)))

def summarize(rows,task,metric):
 selected=[r for r in rows if r['task']==task]
 out=[]
 for method in sorted({r['method'] for r in selected}):
  rr=[r for r in selected if r['method']==method];bygroup=defaultdict(list)
  for r in rr:bygroup[r['article_group']].append(float(r[metric]))
  values=np.array([np.mean(bygroup[g]) for g in sorted(bygroup)])
  rng=np.random.default_rng(SEED);bs=np.mean(rng.choice(values,(1000,len(values)),replace=True),axis=1)
  out.append({'task':task,'method':method,'metric':metric,'article_macro_mean':float(values.mean()),'ci_low':float(np.quantile(bs,.025)),'ci_high':float(np.quantile(bs,.975)),'n_article_groups':len(values),'n_query_runs':len(rr)})
 return out

def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/revision');args=parser.parse_args();OUT=args.output_dir.resolve()
 OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(SEED)
 raw=read_records();full=[adapt_nanomine(x) for x in raw];selected,_=select_process_method_subset(raw);records=[adapt_nanomine(x) for x in selected]
 labels=np.array([x['process_families'][0] for x in selected]);groups=np.array([x['paper_group'] for x in selected]);scales=fit_numeric_scales(full)
 bygroup={g:fit_numeric_scales([r for r in full if r.context['paper_group']!=g]) for g in sorted(set(groups))}
 held=[heldout(r) for r in records];sigs=[signature(r) for r in held]
 output=[];exclusions=0;distance_cache={}
 print('Held-out method evaluation',flush=True)
 for i,q in enumerate(held):
  engine=E(held,numeric_scales=bygroup[groups[i]])
  gallery=[j for j in range(len(records)) if groups[j]!=groups[i] and sigs[j]!=sigs[i]]
  exclusions+=sum(groups[j]!=groups[i] and sigs[j]==sigs[i] for j in range(len(records)))
  distances=defaultdict(list)
  for j in gallery:
   for name,d in metrics_from_facets(engine,engine.facet_distances(q,held[j]),q,held[j]).items():distances[name].append(d)
  for name,values in distances.items():
   d=np.asarray(values);score=knn_agreement(d,labels[gallery],labels[i])
   if name not in distance_cache:distance_cache[name]=np.full((len(records),len(records)),np.nan)
   distance_cache[name][i,gallery]=d
   output.append({'task':'heldout_method','method':name,'sample_id':q.record_id,'article_group':groups[i],'replicate':0,'mrr':'','recall_at_1':'','recall_at_5':'','tie_size':'','agreement_at_5':score,'gallery_size':len(gallery)})
  chance=float(np.mean(labels[gallery]==labels[i]))
  output.append({'task':'heldout_method','method':'random_gallery','sample_id':q.record_id,'article_group':groups[i],'replicate':0,'mrr':'','recall_at_1':'','recall_at_5':'','tie_size':'','agreement_at_5':chance,'gallery_size':len(gallery)})
 print('Masked-record retrieval',flush=True)
 eligible=[i for i,r in enumerate(records) if len(setting_fields(r))>=CFG['masked_retrieval']['minimum_reported_setting_fields']]
 for rep in range(3):
  for i in eligible:
   keys=setting_fields(records[i]);nk=max(1,math.ceil(.4*len(keys)));keys=[keys[z] for z in rng.choice(len(keys),nk,replace=False)]
   ik=[k for k,v in records[i].material_identity.items() if v.status is S.REPORTED];ni=math.ceil(.4*len(ik));ik=list(rng.choice(ik,ni,replace=False))
   q=mask(records[i],set(keys),ik);engine=E(records,numeric_scales=bygroup[groups[i]])
   gallery=[j for j in range(len(records)) if j==i or groups[j]!=groups[i]];target=gallery.index(i);distances=defaultdict(list)
   for j in gallery:
    for name,d in metrics_from_facets(engine,engine.facet_distances(q,records[j]),q,records[j]).items():distances[name].append(d)
   for name,values in distances.items():
    scores=tie_rank_scores(np.asarray(values),target)
    output.append({'task':'masked_record','method':name,'sample_id':records[i].record_id,'article_group':groups[i],'replicate':rep,**scores,'agreement_at_5':'','gallery_size':len(gallery)})
 write_csv(OUT/'retrieval_queries.csv',output)
 summary=summarize(output,'heldout_method','agreement_at_5')
 for metric in ['mrr','recall_at_1','recall_at_5']:summary.extend(summarize(output,'masked_record',metric))
 write_csv(OUT/'retrieval_summary.csv',summary)
 # Source provenance sensitivity uses the identical selected records, minus
 # identifiers resolved neither as a DOI nor as a published source.
 doi_path=ROOT/'results/revision/doi_resolution.json';resolved={x['doi'] for x in json.loads(doi_path.read_text()) if x['status']=='crossref_resolved'} if doi_path.exists() else set()
 verified_ids={x['sample_id'] for x in selected if x.get('doi') in resolved}
 verified=[]
 for i,r in enumerate(records):
  if r.record_id not in verified_ids:continue
  for name,matrix in distance_cache.items():
   gallery=[j for j,x in enumerate(records) if x.record_id in verified_ids and np.isfinite(matrix[i,j])]
   if len(gallery)<5:continue
   verified.append({'task':'heldout_method','method':name,'sample_id':r.record_id,'article_group':groups[i],'agreement_at_5':knn_agreement(matrix[i,gallery],labels[gallery],labels[i])})
 write_csv(OUT/'retrieval_resolved_subset_sensitivity.csv',summarize(verified,'heldout_method','agreement_at_5'))
 print('Missingness and sequence sensitivity',flush=True)
 engine=E(records,numeric_scales=scales);fixed=E(records,numeric_scales=scales,feature_policy='fixed_schema')
 pairs=[(i,j) for i in range(len(records)) for j in range(i+1,len(records))]
 pairs=[pairs[z] for z in rng.choice(len(pairs),300,replace=False)]
 miss=[];v=ViewSpec('settings',{'settings':1})
 # Article-block mask: all records in a selected source group share which
 # setting names are withheld; cross-record correlation is retained.
 mask_rng=np.random.default_rng(CFG['random_streams']['missingness'])
 allnames=sorted({key.rsplit(':',1)[1] for r in records for key in setting_fields(r)})
 paper_masks={g:set(mask_rng.choice(allnames,max(1,len(allnames)//2),replace=False)) for g in sorted(set(groups))}
 mask_audit=[]
 for pi,(i,j) in enumerate(pairs):
  a,b=records[i],records[j];ka=set(setting_fields(a));kb=set(setting_fields(b))
  selected_a={x for x in sorted(ka) if mask_rng.random()<.5}
  joint_keys=sorted(ka&kb);shared={x for x in joint_keys if mask_rng.random()<.5}
  scenarios={'unilateral':(mask(a,selected_a),b),'bilateral':(mask(a,shared),mask(b,shared)),
   'article_block':(mask(a,{x for x in ka if x.rsplit(':',1)[1] in paper_masks[groups[i]]}),mask(b,{x for x in kb if x.rsplit(':',1)[1] in paper_masks[groups[j]]}))}
  for name,(ma,mb) in scenarios.items():
   removed_a=ka-set(setting_fields(ma));removed_b=kb-set(setting_fields(mb))
   if name=='bilateral':assert removed_a==removed_b==shared
   mask_audit.append({'pair':pi,'left':a.record_id,'right':b.record_id,'replicate':0,'mask':name,
     'eligible_joint_reported_keys':json.dumps(joint_keys),'masked_left_keys':json.dumps(sorted(removed_a)),
     'masked_right_keys':json.dumps(sorted(removed_b)),'bilateral_keys_match':removed_a==removed_b if name=='bilateral' else ''})
  for policy,eng in [('active',engine),('fixed_schema',fixed)]:
   original=eng.compare(a,b,v).composite
   for name,(ma,mb) in scenarios.items():
    d=eng.compare(ma,mb,v).composite
    contains=d.lower<=original.lower+1e-10 and d.upper>=original.upper-1e-10
    sparse_zero=d.reported==0 and d.common_reported_fraction<.5
    miss.append({'pair':pi,'mask':name,'policy':policy,'original_lower':original.lower,'original_upper':original.upper,'lower':d.lower,'upper':d.upper,'reported':d.reported,'interval_contains_original_interval':contains,'width_shrank':d.width<original.width-1e-10,'common_reported_fraction':d.common_reported_fraction,'structural_difference_fraction':d.structural_difference_fraction,'excluded_weight':d.excluded_weight,'sparse_zero':sparse_zero})
 write_csv(OUT/'missingness_pairs.csv',miss)
 write_csv(OUT/'mask_key_audit.csv',mask_audit)
 ms=[]
 for policy in ['active','fixed_schema']:
  for scenario in scenarios:
   rr=[r for r in miss if r['policy']==policy and r['mask']==scenario]
   ms.append({'policy':policy,'mask':scenario,'n_pairs':len(rr),'containment_fraction':np.mean([r['interval_contains_original_interval'] for r in rr]),'width_shrank_fraction':np.mean([r['width_shrank'] for r in rr]),'mean_width':np.mean([r['upper']-r['lower'] for r in rr]),'mean_common_fraction':np.mean([r['common_reported_fraction'] for r in rr]),'mean_structural_fraction':np.mean([r['structural_difference_fraction'] for r in rr]),'mean_excluded_variable_count':np.mean([r['excluded_weight'] for r in rr]),'sparse_zero_pairs':sum(r['sparse_zero'] for r in rr)})
   effective={r['pair'] for r in mask_audit if r['mask']==scenario and
              (json.loads(r['masked_left_keys']) or json.loads(r['masked_right_keys']))}
   effective_rows=[r for r in rr if r['pair'] in effective]
   ms[-1].update(n_effectively_masked_pairs=len(effective_rows),
                 containment_among_effectively_masked=np.mean([r['interval_contains_original_interval'] for r in effective_rows]) if effective_rows else '')
 write_csv(OUT/'missingness_summary.csv',ms)
 # Controlled counterexamples document mechanisms, not empirical task success.
 def sample(id,seq):return M(id,process_steps=tuple(P(t,{'time':V.reported(x,kind='numeric',unit_key='duration|min')}) for t,x in seq),process_sequence_status=S.REPORTED)
 a=sample('base',[('heat',1),('cool',2),('heat',3)])
 counter=[]
 for name,b in [('insert_same_type',sample('insert',[('heat',9),('heat',1),('cool',2),('heat',3)])),('delete_repeat',sample('delete',[('heat',1),('cool',2)])),('reorder',sample('reorder',[('heat',1),('heat',3),('cool',2)]))]:
  for align in ['occurrence','sequence']:
   eng=E([a,b],numeric_scales={'duration|min':NumericScale(10)},settings_alignment=align)
   f=eng.facet_distances(a,b)
   counter.append({'example':name,'settings_alignment':align,'step_type_distance':f['step_type'].reported,'sequence_distance':f['sequence'].reported,'settings_distance':f['settings'].reported})
 write_csv(OUT/'sequence_counterexamples.csv',counter)
 seq=E(records,numeric_scales=scales,settings_alignment='sequence');subset=E(records,numeric_scales=fit_numeric_scales(records));augmented=[adapt_nanomine(x,include_component_descriptors=True) for x in selected];augengine=E(augmented,numeric_scales=fit_numeric_scales([adapt_nanomine(x,include_component_descriptors=True) for x in raw]))
 sr=[]
 for i,j in pairs:
  sr.append({'left':records[i].record_id,'right':records[j].record_id,'occurrence_settings':engine.compare(records[i],records[j],v).composite.reported,'aligned_settings':seq.compare(records[i],records[j],v).composite.reported,'fixed_scale_balanced':engine.compare(records[i],records[j],DEFAULT_VIEWS['balanced_instance']).composite.reported,'subset_scale_balanced':subset.compare(records[i],records[j],DEFAULT_VIEWS['balanced_instance']).composite.reported,'augmented_descriptors_balanced':augengine.compare(augmented[i],augmented[j],DEFAULT_VIEWS['balanced_instance']).composite.reported})
 write_csv(OUT/'representation_sensitivity_pairs.csv',sr)
 bs=[];rawgroup=defaultdict(list)
 for r in full:rawgroup[r.context['paper_group']].append(r)
 groupnames=sorted(rawgroup)
 scale_rng=np.random.default_rng(CFG['random_streams']['scale_bootstrap'])
 for rep in range(20):
  fit=[r for g in scale_rng.choice(groupnames,len(groupnames),replace=True) for r in rawgroup[g]]
  eng=E(records,numeric_scales=fit_numeric_scales(fit))
  vals=[eng.compare(records[i],records[j],DEFAULT_VIEWS['balanced_instance']).composite.reported for i,j in pairs]
  ref=[r['fixed_scale_balanced'] for r in sr]
  bs.append({'replicate':rep,'spearman':spearmanr(ref,vals).statistic,'mean_absolute_change':np.mean(np.abs(np.asarray(ref)-vals)),'max_absolute_change':np.max(np.abs(np.asarray(ref)-vals))})
 write_csv(OUT/'scale_block_bootstrap.csv',bs)
 audit={'heldout_exact_duplicate_cross_group_pairs_excluded':exclusions,'masked_query_records':len(eligible),'masked_query_article_groups':len(set(groups[eligible])),'heldout_query_records':len(held),'heldout_article_groups':len(set(groups)),'crossref_resolved_query_records':len(verified_ids),'weights_optimized':False,'interpretation':'metadata retrieval and held-out metadata support; no property/transfer evidence','sequence_alignment_spearman':spearmanr([r['occurrence_settings'] for r in sr],[r['aligned_settings'] for r in sr]).statistic,'subset_scale_spearman':spearmanr([r['fixed_scale_balanced'] for r in sr],[r['subset_scale_balanced'] for r in sr]).statistic,'augmented_descriptor_spearman':spearmanr([r['fixed_scale_balanced'] for r in sr],[r['augmented_descriptors_balanced'] for r in sr]).statistic}
 audit.update({'bilateral_mask_universe':'intersection of jointly reported canonical operation-occurrence-setting keys',
               'bilateral_key_mismatch_pairs':sum(r['bilateral_keys_match'] is False for r in mask_audit if r['mask']=='bilateral'),
               'random_streams':CFG['random_streams']})
 (OUT/'evaluation_summary.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
