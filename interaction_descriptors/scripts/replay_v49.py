#!/usr/bin/env python3
"""v49 frozen-source steel replay. No source training or outcome-driven tuning.

Keep this file with steel_v46_replay_inputs.npz, steel_v46_oof_predictions.csv,
steel_v46_summary.csv and v49_frozen_bank.npz. No ZIP extraction is required.

  python v49_replay.py --mode preflight --output-dir preflight49
  python v49_replay.py --mode audit --output-dir audit49
  python v49_replay.py --mode representative --output-dir replay49
  python v49_replay.py --mode full --output-dir full49

The full mode replays the existing 180 jobs / 540 outer fits, not a new test.
Use --require-isolated only in a deliberately created independent venv.
Successful isolation is not itself proof of a fresh dependency installation.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, csv, hashlib, importlib.metadata, importlib.util, json, platform
import sys, time
from collections import defaultdict
from pathlib import Path

EXPECTED_INPUTS = {'steel_v46_replay_inputs.npz': 'd48e6ad2481e106cb11a2799f41fd07c3dc98c2dcd05beb7f5d175b595609950', 'steel_v46_oof_predictions.csv': 'b4d259535a1c4d6568f52889274191ec0e01e852e75dff23d86c93b8ea48ff44', 'steel_v46_summary.csv': 'ee045ab447d3dabb72ea9feebecec927e2f562a891ffd7dd57bfebc1eec4ee64', 'v49_frozen_bank.npz': 'e4a47d98becfdb408264f6ebc20abc8add9aa51b982944db1afce00683c3e859'}
EXPECTED_VERSIONS = {'numpy': '2.3.5', 'scipy': '1.17.0', 'scikit-learn': '1.8.0', 'lightgbm': '4.6.0', 'rdkit': '2025.9.4', 'joblib': '1.5.3', 'threadpoolctl': '3.6.0', 'Pillow': '13.0.0'}

MODULES={'numpy':'numpy','scipy':'scipy','scikit-learn':'sklearn','lightgbm':'lightgbm',
         'rdkit':'rdkit','joblib':'joblib','threadpoolctl':'threadpoolctl','Pillow':'PIL'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok: raise ValueError(message)

def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def preflight(input_dir, require_isolated):
    import site
    pkg=[]
    for name,expected in EXPECTED_VERSIONS.items():
        try:
            d=importlib.metadata.distribution(name);version=d.version
            path=str(Path(d.locate_file('')).resolve())
            mod=importlib.util.find_spec(MODULES[name]);origin=mod.origin if mod else None
        except importlib.metadata.PackageNotFoundError:
            version=None;path=None;origin=None
        pkg.append(dict(package=name,expected=expected,actual=version,exact_match=version==expected,
                        distribution_root=path,module_origin=origin))
    cfg=Path(sys.prefix)/'pyvenv.cfg'
    cfg_text=cfg.read_text() if cfg.is_file() else ''
    isolated=(sys.prefix!=sys.base_prefix and 'include-system-site-packages = false' in cfg_text.lower())
    local=all(p['module_origin'] and Path(p['module_origin']).resolve().is_relative_to(Path(sys.prefix).resolve()) for p in pkg)
    files=[]
    for name,expected in EXPECTED_INPUTS.items():
        path=input_dir/name;actual=sha(path) if path.is_file() else None
        files.append(dict(file=name,expected=expected,actual=actual,match=actual==expected))
    result=dict(python=sys.version,python_executable=sys.executable,platform=platform.platform(),
                prefix=sys.prefix,base_prefix=sys.base_prefix,venv_isolated_from_system_packages=isolated,
                dependencies_resolved_within_prefix=local,site_user_enabled=site.ENABLE_USER_SITE,
                python_minor_matches=sys.version_info[:2]==(3,13),packages=pkg,inputs=files,
                fresh_dependency_installation_attested=False,
                note='Package isolation and exact versions alone do not prove a freshly installed environment.')
    result['ready']=bool(result['python_minor_matches'] and all(p['exact_match'] for p in pkg)
                         and all(f['match'] for f in files)
                         and (not require_isolated or (isolated and local)))
    return result

ARCHIVED_NUMERICAL_BLOCK = '"""Unmodified numerical regression routines from v13; sklearn/lightgbm implementations."""\nimport numpy as np\nfrom scipy.linalg import eigh\nfrom scipy.spatial.distance import cdist\nfrom scipy.optimize import minimize,LinearConstraint,lsq_linear\nfrom sklearn.cross_decomposition import PLSRegression\nfrom sklearn.linear_model import LassoLars\nfrom sklearn.svm import SVR\nfrom sklearn.ensemble import GradientBoostingRegressor\nfrom lightgbm import LGBMRegressor\nimport warnings,time\nELLS=[.25,.5,1,2,4,8];ALPHAS=[1e-4,.01,.1,1,10]\nGRIDS={\n \'PLS\':[dict(n_components=n) for n in [1,2,4,8]],\n \'LASSO\':[dict(alpha=a) for a in [1e-4,.001,.01,.1,1]],\n \'KRR\':[dict(ell=l,alpha=a) for l in ELLS for a in ALPHAS],\n \'SVR\':[dict(ell=l,C=c,epsilon=e) for l in [.5,1,2,4] for c in [.1,1,10,100] for e in [.05,.1]],\n \'LightGBM\':[dict(num_leaves=l,min_child_samples=n) for l in [3,7,15] for n in [3,7]],\n \'GBDT\':[dict(max_depth=d,min_samples_leaf=n) for d in [1,2,3] for n in [2,5]]}\n\ndef prep(A,B):\n sd=A.std(0);ok=sd>1e-10\n if not ok.any():return np.zeros((len(A),1)),np.zeros((len(B),1)),ok\n mu=A[:,ok].mean(0);sd=sd[ok];return (A[:,ok]-mu)/sd,(B[:,ok]-mu)/sd,ok\n\ndef run_one(A,y,B,method,config):\n mu=float(y.mean());sd=max(float(y.std()),1e-10);ys=(y-mu)/sd;extra={}\n with warnings.catch_warnings(record=True) as ww:\n  warnings.simplefilter(\'always\')\n  if method==\'PLS\':\n   rank=int(np.linalg.matrix_rank(A));n=min(config[\'n_components\'],A.shape[1],len(A)-1,max(rank,1));m=PLSRegression(n_components=n,scale=False,max_iter=1000,tol=1e-8);m.fit(A,ys);p=m.predict(B).ravel();extra[\'effective_components\']=n\n  elif method==\'LASSO\':\n   # Same L1 objective as coordinate descent; LARS avoids slow convergence\n   # of highly correlated latent columns. Explicit primal-dual gap is audited.\n   alpha=config[\'alpha\'];m=LassoLars(alpha=alpha,fit_intercept=False,max_iter=5000,eps=1e-12);m.fit(A,ys)\n   coef=m.coef_.copy()\n   def dualgap(coef):\n    rr=ys-A@coef;dn=float(np.max(np.abs(A.T@rr)));c=min(1.,alpha*len(A)/dn) if dn>0 else 1.\n    return max(0.,float((.5*np.dot(rr,rr)*(1+c*c)+alpha*len(A)*np.abs(coef).sum()-c*np.dot(rr,ys))/len(A)))\n   gap=dualgap(coef);solver=\'LARS\'\n   if gap>1e-7:\n    # Numerically robust solution of the SAME LASSO objective via its dual:\n    # min_v 0.5 ||v-y||^2, subject to |A.T v| <= n*alpha.\n    # Recover a primal solution on the active signed columns by bounded LS.\n    bound=alpha*len(A)\n    dual=minimize(lambda v:.5*np.sum((v-ys)**2),np.zeros(len(A)),jac=lambda v:v-ys,\n      method=\'SLSQP\',constraints=[LinearConstraint(A.T,-bound,bound)],\n      options={\'maxiter\':1000,\'ftol\':1e-13})\n    grad=A.T@dual.x\n    for atol in [1e-6,1e-5,1e-4,1e-3]:\n     active=np.abs(grad)>=bound*(1-atol);cc=np.zeros(A.shape[1])\n     if active.any():\n      signs=np.sign(grad[active]);ls=lsq_linear(A[:,active]*signs,ys-dual.x,\n        bounds=(0,np.inf),method=\'bvls\',tol=1e-13,max_iter=1000);cc[active]=ls.x*signs\n     gg=dualgap(cc)\n     if gg<gap:coef=cc;gap=gg;solver=\'LARS+dualQP\'\n     if gap<1e-7:break\n    extra.update(dual_qp_iterations=int(dual.nit),dual_qp_success=bool(dual.success))\n   if gap>1e-6:raise RuntimeError(f\'LASSO optimality check failed: gap={gap}\')\n   p=B@coef;extra.update(n_iter=int(m.n_iter_),dual_gap=gap,solver=solver)\n  elif method==\'SVR\':\n   m=SVR(C=config[\'C\'],epsilon=config[\'epsilon\'],gamma=1/(2*config[\'ell\']**2*A.shape[1]),cache_size=100);m.fit(A,ys);p=m.predict(B)\n  elif method==\'LightGBM\':\n   m=LGBMRegressor(n_estimators=100,learning_rate=.05,reg_lambda=1,min_data_in_bin=1,max_bin=63,n_jobs=1,verbosity=-1,random_state=6101,deterministic=True,force_col_wise=True,**config);m.fit(A,ys);p=m.booster_.predict(B,num_threads=1)\n  elif method==\'GBDT\':\n   m=GradientBoostingRegressor(n_estimators=100,learning_rate=.05,loss=\'squared_error\',random_state=6101,**config);m.fit(A,ys);p=m.predict(B)\n  elif method==\'KRR\':\n   D=cdist(A,A,\'sqeuclidean\')/A.shape[1];Q=cdist(B,A,\'sqeuclidean\')/A.shape[1];K=np.exp(-D/(2*config[\'ell\']**2));Q=np.exp(-Q/(2*config[\'ell\']**2));p=Q@np.linalg.solve(K+config[\'alpha\']*np.eye(len(A)),ys)\n  else:raise ValueError(method)\n extra[\'warnings\']=[type(w.message).__name__+\': \'+str(w.message)[:180] for w in ww]\n return p*sd+mu,extra\n\ndef predict_grid(A,y,B,method):\n if method!=\'KRR\':\n  res=[run_one(A,y,B,method,c) for c in GRIDS[method]];return np.column_stack([r[0] for r in res]),[r[1] for r in res]\n mu=y.mean();sd=max(y.std(),1e-10);v=(y-mu)/sd;D=cdist(A,A,\'sqeuclidean\')/A.shape[1];Q=cdist(B,A,\'sqeuclidean\')/A.shape[1];out=[]\n for ell in ELLS:\n  K=np.exp(-D/(2*ell**2));T=np.exp(-Q/(2*ell**2));ev,U=eigh((K+K.T)/2,check_finite=False);ev=np.maximum(ev,0)\n  pred=(T@U)@((U.T@v)[:,None]/(ev[:,None]+np.array(ALPHAS)[None,:]))*sd+mu;out.append(pred)\n return np.concatenate(out,axis=1),[{} for _ in GRIDS[method]]\n\ndef nested(X,y,tr,te,inner,method):\n t=time.time();sse=np.zeros(len(GRIDS[method]));nvalid=0;warn=[];maxgap=0.\n for a,b in inner:\n  assert set(a)<=set(tr) and set(b)<=set(tr) and not set(a)&set(b)\n  A,B,ok=prep(X[a],X[b]);pp,infos=predict_grid(A,y[a],B,method)\n  sse+=((pp-y[b,None])**2).sum(0);nvalid+=len(b)\n  for k,i in enumerate(infos):\n   maxgap=max(maxgap,float(i.get(\'dual_gap\',0)))\n   if i.get(\'warnings\'):warn.append(dict(candidate=k,messages=i[\'warnings\']))\n k=int(np.argmin(sse));cfg=GRIDS[method][k];A,B,ok=prep(X[tr],X[te]);p,info=run_one(A,y[tr],B,method,cfg)\n if not np.isfinite(p).all():raise FloatingPointError(\'Nonfinite prediction\')\n return p,dict(config=cfg,inner_mse=float(sse[k]/nvalid),elapsed_seconds=time.time()-t,n_features=int(ok.sum()),inner_warning_count=len(warn),max_inner_lasso_dual_gap=maxgap,final_info=info,grid_sse=sse.tolist(),warnings=warn)\n'
ARCHIVED_NUMERICAL_BLOCK_SHA256 = '3b41f51ee849ffbbbf86c3e3303b74d8faee11955f10c06305e422f4fe28714c'

def load_numerics():
    exec(compile(ARCHIVED_NUMERICAL_BLOCK,'archived_v46_numerical_block','exec'),globals())
    from scipy.special import expit
    globals()['expit']=expit


def regenerate(data, bank_path, output):
    from rdkit import Chem
    X=data['X'];names=list(map(str,data['input_names']));pt=Chem.GetPeriodicTable()
    element_values=np.array([[pt.GetAtomicNumber(e),pt.GetAtomicWeight(e),pt.GetRcovalent(e),pt.GetNOuterElecs(e)] for e in names])
    columns=[]
    for j in range(4):
        v=element_values[:,j];mean=X@v
        std=np.sqrt(np.maximum(np.sum(X*(v[None,:]-mean[:,None])**2,axis=1),0))
        columns.extend([mean,std,np.min(np.where(X>0,v[None,:],np.inf),axis=1),np.max(np.where(X>0,v[None,:],-np.inf),axis=1)])
    columns.extend([-np.sum(X*np.log(np.maximum(X,1e-300)),axis=1),np.sum(X*X,axis=1)])
    D=np.column_stack(columns);de=float(np.max(abs(D-data['D'])))
    require(de<1e-10,f'Fixed-descriptor mismatch: {de}')
    arrays={};rows=[]
    with np.load(bank_path,allow_pickle=False) as bank:
        meta=json.loads(str(bank['__metadata_json__']));cat=meta['catalogue_original']
        require(len(cat)==24,'Expected 24 source tasks')
        for kind in ['NN','YAE','YVAE','random']:
            for seed in [101,202,303]:
                blocks=[];masks=[];nused=0
                for i,item in enumerate(cat):
                    elements=list(map(str,bank[f's{i:02d}__elements']))
                    missing=[j for j,e in enumerate(names) if e not in elements]
                    ok=np.ones(len(X),bool) if not missing else X[:,missing].sum(1)==0
                    out=np.zeros((len(X),8));masks.append(ok.astype(float))
                    if ok.any():
                        tag=f's{i:02d}__{kind}__{seed}__'
                        get=lambda k:bank[tag+k]
                        xx=np.zeros((int(ok.sum()),len(elements)))
                        for j,e in enumerate(elements):
                            if e in names:xx[:,j]=X[ok,names.index(e)]
                        a=(xx-get('x_mean'))/get('x_sd')
                        for layer in ['trunk.0','trunk.2']:
                            a=a@get(layer+'.weight').T+get(layer+'.bias');a=a*expit(a)
                        mu=a@get('mu.weight').T+get('mu.bias')
                        out[ok]=(mu-get('z_mean'))/get('z_sd');nused+=1
                    blocks.append(out)
                Z=np.concatenate(blocks,axis=1);M=np.column_stack(masks)
                delta=float(np.max(abs(Z-data[f'Z_{kind}_{seed}'])))
                require(delta<1e-10,f'Feature mismatch for {kind}/{seed}: {delta}')
                require(np.array_equal(M,data['M']),'Applicability mismatch')
                arrays[f'Z_{kind}_{seed}']=Z
                rows.append(dict(kind=kind,seed=seed,applicable_sources=nused,rows=len(X),columns=Z.shape[1],varying_columns=int((Z.std(0)>1e-10).sum()),max_difference=delta))
    arrays.update(X=X,D=D,M=M,y=data['y'])
    write_csv(output/'source_feature_checks.csv',rows)
    np.savez_compressed(output/'regenerated_features.npz',**arrays)
    return arrays,dict(models_evaluated=sum(r['applicable_sources'] for r in rows),feature_arrays=len(rows),max_feature_difference=max(r['max_difference'] for r in rows),fixed18_max_difference=de)


def metric_audit(data, records, summary, output):
    y=data['y'];group=defaultdict(list)
    for r in records:group[r['job']].append(r)
    scores=[]
    for name,rr in sorted(group.items()):
        rr.sort(key=lambda x:int(x['sample_id']))
        require([int(x['sample_id']) for x in rr]==list(range(len(y))),f'Evaluation ID coverage: {name}')
        yy=np.array([float(x['y_MPa']) for x in rr]);pr=np.array([float(x['prediction_MPa']) for x in rr])
        require(np.array_equal(yy,y),'Response identity mismatch');require(np.isfinite(pr).all(),'Non-finite saved prediction')
        scores.append(dict(job=name,kind=rr[0]['kind'],mode=rr[0]['mode'],method=rr[0]['method'],seed=int(rr[0]['seed']),repeat=int(rr[0]['repeat']),
                           rmse=float(np.sqrt(np.mean((pr-y)**2))),mae=float(np.mean(abs(pr-y))),r2=float(1-np.sum((pr-y)**2)/np.sum((y-y.mean())**2))))
    checks=[]
    for r in summary:
        s=[x for x in scores if all(x[k]==r[k] for k in ['kind','mode','method'])]
        require(len(s)==int(r['n_jobs']),'Summary job-count mismatch')
        for metric in ['rmse','mae','r2']:
            actual=float(np.mean([x[metric] for x in s]));expected=float(r[metric+'_mean'])
            require(abs(actual-expected)<1e-10,'Metric reaggregation mismatch')
            checks.append(dict(kind=r['kind'],mode=r['mode'],method=r['method'],metric=metric,recomputed=actual,saved=expected,difference=abs(actual-expected)))
    write_csv(output/'metric_checks.csv',checks)
    splits=json.loads(str(data['splits_json']))
    for r in [0,1]:
        ss=[s for s in splits if s['repeat']==r]
        require(sorted(i for s in ss for i in s['test'])==list(range(len(y))),'Outer coverage mismatch')
        for s in ss:
            tr=set(s['train']);te=set(s['test'])
            require(len(tr)==64 and len(te)==104 and not tr&te,'Outer budget/leakage check failed')
            inner_valid=[]
            for t in s['inner']:
                a=set(t['train']);b=set(t['valid']);require(a|b==tr and not a&b,'Inner partition mismatch');inner_valid+=t['valid']
            require(sorted(inner_valid)==sorted(tr),'Inner coverage mismatch')
    return splits,dict(saved_prediction_events=len(records),jobs=len(scores),metrics_checked=len(checks),max_metric_difference=max(c['difference'] for c in checks),outer_splits=len(splits))


def replay(arrays,data,records,splits,mode,output):
    y=data['y'];look={}
    for r in records:
        key=(r['kind'],int(r['seed']),r['mode'],r['method'],int(r['repeat']))
        if key not in look:look[key]=np.empty(len(y))
        look[key][int(r['sample_id'])]=float(r['prediction_MPa'])
    jobs=[]
    if mode=='representative':
        jobs=[('NN',101,'all_frozen',m,0) for m in ['PLS','LASSO','KRR','SVR','LightGBM','GBDT']]
    elif mode=='full':
        for kind in ['NN','YAE','YVAE','random']:
            for seed in [101,202,303]:
                modes=['all_frozen']+(['X','XD','XDM'] if kind=='NN' and seed==101 else [])
                for variant in modes:
                    for m in ['PLS','LASSO','KRR','SVR','LightGBM','GBDT']:
                        for rep in [0,1]:jobs.append((kind,seed,variant,m,rep))
    checks=[];predictions=[]
    for kind,seed,variant,method,rep in jobs:
        if variant=='X':F=arrays['X']
        elif variant=='XD':F=np.c_[arrays['X'],arrays['D']]
        elif variant=='XDM':F=np.c_[arrays['X'],arrays['D'],arrays['M']]
        else:F=np.c_[arrays['X'],arrays['D'],arrays['M'],arrays[f'Z_{kind}_{seed}']]
        pr=np.empty(len(y));selected=[];start=time.perf_counter()
        for s in splits:
            if s['repeat']!=rep:continue
            tr=np.array(s['train']);te=np.array(s['test']);inner=[(np.array(k['train']),np.array(k['valid'])) for k in s['inner']]
            pp,info=nested(F,y,tr,te,inner,method);pr[te]=pp;selected.append(info)
        key=(kind,seed,variant,method,rep);delta=float(np.max(abs(pr-look[key])))
        require(delta<1e-6,f'Prediction replay mismatch {key}: {delta}')
        stem=f'{kind}_{seed}_{variant}_{method}_{rep}'
        np.savez_compressed(output/(stem+'.npz'),pred=pr,y=y,ids=data['ids'])
        dump(output/(stem+'_selection.json'),selected)
        row=dict(kind=kind,seed=seed,mode=variant,method=method,repeat=rep,outer_fits=3,predictions=len(y),rmse=float(np.sqrt(np.mean((pr-y)**2))),max_prediction_difference=delta,elapsed_seconds=time.perf_counter()-start)
        checks.append(row)
        print(f"{len(checks)}/{len(jobs)} {stem} max_difference={delta:.3g}",flush=True)
    write_csv(output/'replay_checks.csv',checks)
    return dict(configuration_repeat_jobs=len(checks),outer_fits=3*len(checks),prediction_events=len(y)*len(checks),max_prediction_difference=max((r['max_prediction_difference'] for r in checks),default=None))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output-dir',type=Path,default=Path('v49_checks'))
    p.add_argument('--mode',choices=['preflight','audit','representative','full'],default='audit')
    p.add_argument('--require-isolated',action='store_true')
    args=p.parse_args();inp=args.input_dir.resolve();out=args.output_dir.resolve()
    require(not out.exists() or not any(out.iterdir()),'Use a new, empty output directory.')
    out.mkdir(parents=True,exist_ok=True)
    result=dict(release='v49',mode=args.mode,status='STARTED',new_materials=0,new_source_training=0,new_performance_conditions=0,fresh_environment_reproduction_verified=False)
    try:
        check=preflight(inp,args.require_isolated);dump(out/'preflight.json',check)
        require(check['ready'],'Preflight did not pass; see preflight.json. No package-version fallback is applied.')
        if args.mode!='preflight':
            load_numerics()
            with np.load(inp/'steel_v46_replay_inputs.npz',allow_pickle=False) as f:data=dict(f)
            records=read_csv(inp/'steel_v46_oof_predictions.csv');summary=read_csv(inp/'steel_v46_summary.csv')
            splits,metrics=metric_audit(data,records,summary,out)
            arrays,features=regenerate(data,inp/'v49_frozen_bank.npz',out)
            rep=replay(arrays,data,records,splits,args.mode,out)
            result.update(metric_audit=metrics,feature_regeneration=features,replay=rep)
        result['status']='PASS'
        result['input_files_unchanged']=all(sha(inp/n)==s for n,s in EXPECTED_INPUTS.items())
        require(result['input_files_unchanged'],'Input changed during execution')
        result['numerical_block_sha256']=ARCHIVED_NUMERICAL_BLOCK_SHA256
        dump(out/'audit.json',result);print(json.dumps(result,indent=2))
    except Exception as exc:
        result.update(status='BLOCKED_OR_FAILED',error_type=type(exc).__name__,error=str(exc))
        dump(out/'audit.json',result)
        raise

if __name__=='__main__':main()
