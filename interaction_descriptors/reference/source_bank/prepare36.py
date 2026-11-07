"""Build an elemental source bank using archived values; never select on target y."""
import csv,json,sys,shutil,hashlib,collections,datetime
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
OLD=R/'rebuild_input/legacy_v35'
sys.path.insert(0,str(OLD/'code'))
from prepare35 import formula,key,vector,physical

def write(path,rows):
 rows=list(rows)
 if not rows:return
 with open(path,'w',encoding='utf-8-sig',newline='')as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def read(path):return list(csv.DictReader(open(path,encoding='utf-8-sig')))

# All 865 stored JARVIS compositions (not only the two selected target cohorts).
excluded=set()
for row in read(OLD/'raw/jarvis_metadata.csv'):
 excluded.add(key(dict((k,float(v))for k,v in (a.split(':')for a in row['composition'].split()))))
for c in ['metals','mixed_oxides']:
 for ext in ['npz','csv']:shutil.copy2(OLD/f'data/target_{c}.{ext}',R/f'data/target_{c}.{ext}')
 shutil.copy2(OLD/f'data/splits_{c}.json',R/f'data/splits_{c}.json')
# Enumerate tasks before any new downstream results.
spec=[
 ('CritExam__Ed','mp_decomp','thermo_stability','identity',1),
 ('MatBench__mp_e_form','mp_formation','formation','identity',0),
 ('MatBench__mp_gap','mp_gap','electronic','identity',5),
 ('MatBench__elasticity_log10K_VRH','mp_bulk','mechanical','identity',3),
 ('MatBench__elasticity_log10G_VRH','mp_shear','mechanical','identity',3),
 ('mp_elastic_anisotropy','mp_anisotropy','mechanical','log1p',3),
 ('mp_e_hull','mp_hull','thermo_stability','identity',1),
 ('IRnet__OQMD_Bandgap','oqmd_gap','electronic','identity',5),
 ('IRnet__OQMD_Energy_per_atom','oqmd_energy','total_energy','identity',2),
 ('IRnet__OQMD_Formation_Enthalpy','oqmd_formation','formation','identity',0),
 ('IRnet__OQMD_Volume_per_atom','oqmd_volume','structural','log1p',3),
 ('aflow__Egap','aflow_gap','electronic','identity',5),
 ('aflow__energy_atom','aflow_energy','total_energy','identity',2),
 ('aflow__ael_bulk_modulus_vrh','aflow_bulk','mechanical','log1p',3),
 ('aflow__ael_shear_modulus_vrh','aflow_shear','mechanical','log1p',3),
 ('aflow__ael_debye_temperature','aflow_debye','thermal','log1p',4),
 ('aflow__agl_thermal_conductivity_300K','aflow_kappa','thermal','log1p',4),
 ('aflow__agl_thermal_expansion_300K','aflow_expansion','thermal','identity',4),
]
raw=read(R/'raw/simulation_composition_observations.csv')
source_records=[];catalog=[];audit=[]
def build(sid,rawrows,prop,family,transform,semantic,origin,url,scope,original_dataset,unit):
 g=collections.defaultdict(list);removed=set();bad=0
 for r in rawrows:
  try:
   k=key(formula(r['formula']));v=float(r['value'])
   if not np.isfinite(v):raise ValueError('nonfinite')
   if transform=='log1p'and v<0:raise ValueError('negative log input')
  except Exception:bad+=1;continue
  if k in excluded:removed.add(k);continue
  g[k].append((v,str(r.get('id',''))))
 ks=sorted(g);els=sorted({e for k in ks for e,v in k});ng=len(set('-'.join(e for e,v in k)for k in ks))
 if len(ks)<60 or ng<12:
  audit.append(dict(source=sid,status='excluded',reason='fewer_than_60_compositions_or_12_groups',n=len(ks)));return
 X=vector(ks,els);yv=np.array([np.median([v for v,i in g[k]])for k in ks]);y=np.log1p(yv)if transform=='log1p'else yv
 assert np.std(y)>1e-10
 ids=np.array([str(k)for k in ks]);groups=np.array(['-'.join(e for e,v in k)for k in ks])
 np.savez_compressed(R/f'sources/{sid}.npz',X=X,y=y,y_raw=yv,groups=groups,ids=ids,elements=els)
 for k,v in zip(ks,yv):source_records.append(dict(source=sid,composition_key=str(k),n_elements=len(k),value=float(v),raw_rows=len(g[k]),original_ids='|'.join(i for x,i in g[k]),formula=''.join(e+str(a)for e,a in k)))
 row=dict(source=sid,property=prop,family=family,transform=transform,semantic_rank=semantic,parent=origin,original_dataset=original_dataset,unit=unit,source_url=url,scope=scope,raw_rows=len(rawrows),n_compositions=len(ks),n_groups=ng,n_elements=len(els),excluded_target_compositions=len(removed),invalid_rows=bad,max_components=max(map(len,ks)))
 catalog.append(row)
for dataset,sid,fam,tr,se in spec:
 rs=[r for r in raw if r['dataset_name']==dataset]
 build(sid,[dict(formula=r['reduced_formula'],value=r['property_value_numeric'],id=r['observation_id'])for r in rs],rs[0]['property_name'],fam,tr,se,rs[0]['database'],rs[0]['upstream_source'],'archived pure-metals and single-metal-oxide subset; not full database',dataset,rs[0]['property_unit'])
rs=[r for r in read(OLD/'raw/dielectric_saved.csv')if r['domain']=='inorganic'and r['formula']]
for field,sid,fam,tr in [('electronic_dielectric','petousis_electronic','dielectric','log1p'),('static_dielectric_total','petousis_total','dielectric','log1p'),('band_gap_eV','petousis_gap','electronic','identity')]:
 build(sid,[dict(formula=r['formula'],value=r[field],id=r['source_id'])for r in rs],field,fam,tr,5,'Materials Project / Petousis','https://doi.org/10.1038/sdata.2016.134','archived full 1056-row composition table; structures collapsed to medians','matminer_dielectric_constant','eV'if sid.endswith('gap')else'dimensionless')
# Source v35 metal Tc is already cleaned of all JARVIS parent identities; revalidate.
a=np.load(OLD/'data/source_metals.npz');els=a['input_names'];r=[]
for x,v,cid in zip(a['X'],a['Y'][:,0],a['ids']):r.append(dict(formula=''.join(e+str(float(n))for e,n in zip(els,x)if n>0),value=v,id=cid))
build('supercon_Tc',r,'superconducting_Tc','magnetic_superconductivity','identity',6,'SuperCon / Hamidieh','https://doi.org/10.24432/C53P47','archived metal-only subset; arbitrary 1–5 components','SuperCon_Hamidieh_metal','K')
# Coverage is a feature-compatibility audit, not a target-label-driven filter.
cover=[]
for row in catalog:
 a=np.load(R/f'sources/{row["source"]}.npz');allowed=set(a['elements'].tolist())
 for c in ['metals','mixed_oxides']:
  t=np.load(R/f'data/target_{c}.npz');ok=np.array([all(e in allowed for e,v in zip(t['input_names'],x)if v>0)for x in t['X']])
  cover.append(dict(source=row['source'],case=c,supported=int(ok.sum()),total=len(ok),fraction=float(ok.mean())))
write(R/'data/source_catalog.csv',catalog);dump(R/'data/source_catalog.json',catalog);write(R/'data/source_observations.csv',source_records);write(R/'data/source_coverage.csv',cover)
audit += [dict(source=s,status='excluded',reason=why)for s,why in [('CritExam__Ef','avoids a second version of MP formation property'),('mp_bulk_modulus','avoids second MP bulk-modulus snapshot'),('mp_shear_modulus','avoids second MP shear-modulus snapshot'),('MatBench__dielectric','derived refractive-index redundancy and fewer than 60 after exclusion'),('mp_mu_b','cell-extensive target without cell-size normalization'),('MatBench__mp_is_metal','classification excluded from the scalar-regression bank'),('MatBench__castelli','fewer than 60 after exclusion'),('MatBench__jdft2d','fewer than 60 after exclusion'),('MatBench__phonons','fewer than 60 after exclusion')]]
write(R/'data/source_exclusions.csv',audit)
protocol={
 'version':'v0.36.0','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'main_question':'Does adding diverse elemental source tasks improve frozen-descriptor transfer, and do source semantics or input coverage help select?',
 'sources':len(catalog),'source_final_seeds':[101,202,303],'encoders':['NN','YAE','YVAE'],
 'source_epochs':200,'source_architecture':'p-32-SiLU-16-SiLU-8-16-SiLU-1; independent per dataset/property',
 'source_validation':'fixed hyperparameters; 3-fold element-set OOF at independent diagnostic seed; no target labels',
 'source_labels':'median of available source rows per normalized composition; transform as catalog',
 'target_cases':['metals','mixed_oxides'],'target_labels':'archived JARVIS structure-median formation energy, eV/atom',
 'target_budget_primary':64,'target_splits':'unchanged v35, 3 group folds x2 repeats, inner3',
 'source_exclusions':'all865 JARVIS parent compositions from all source properties; no source JARVIS tables',
 'source_missing_elements':'do not call source model; source-standardized neutral zeros plus explicit availability mask',
 'base':'X+D; also X+D+all22 applicability masks as a matched information control',
 'primary_representations':['all','prior_small','semantic4','coverage4','balanced4','strict_all'],
 'strict_all':'exclude two formation-energy/enthalpy tasks; close thermodynamic and total-energy tasks remain',
 'source_count_diagnostic':'NN and random NN, coverage k=1,2,4,8,22; random order k=1,2,4,8; KRR/LightGBM; 3 model seeds',
 'selection':'semantics from property definitions fixed; coverage calculated only on current train subset; balanced uses semantics/coverage/quality and representation nonredundancy; all rankings repeated in each inner split',
 'semantics':'formation0; stability1; total-energy2; elastic/structural3; thermal4; electronic/dielectric5; superconductivity6',
 'source_training_limit':'fixed 200 epochs, no source architecture tuning in this edition',
 'provenance':'saved filtered collection, not 22 full independent databases; 18 tasks rely on archived pure-metal/single-metal-oxide subset',
 'no_new_materials':True,'no_new_web_raw_download':True,
 'no_fine_tuning':True,'no_new_pair_expansion':True,
}
dump(R/'protocol.json',protocol)
print('SOURCE BANK',len(catalog),'source property-composition rows',len(source_records))
print([(r['source'],r['n_compositions'])for r in catalog])
