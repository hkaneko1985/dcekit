"""Unmodified numerical regression routines from v13; sklearn/lightgbm implementations."""
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.optimize import minimize,LinearConstraint,lsq_linear
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoLars
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import warnings,time
ELLS=[.25,.5,1,2,4,8];ALPHAS=[1e-4,.01,.1,1,10]
GRIDS={
 'PLS':[dict(n_components=n) for n in [1,2,4,8]],
 'LASSO':[dict(alpha=a) for a in [1e-4,.001,.01,.1,1]],
 'KRR':[dict(ell=l,alpha=a) for l in ELLS for a in ALPHAS],
 'SVR':[dict(ell=l,C=c,epsilon=e) for l in [.5,1,2,4] for c in [.1,1,10,100] for e in [.05,.1]],
 'LightGBM':[dict(num_leaves=l,min_child_samples=n) for l in [3,7,15] for n in [3,7]],
 'GBDT':[dict(max_depth=d,min_samples_leaf=n) for d in [1,2,3] for n in [2,5]]}

def prep(A,B):
 sd=A.std(0);ok=sd>1e-10
 if not ok.any():return np.zeros((len(A),1)),np.zeros((len(B),1)),ok
 mu=A[:,ok].mean(0);sd=sd[ok];return (A[:,ok]-mu)/sd,(B[:,ok]-mu)/sd,ok

def run_one(A,y,B,method,config):
 mu=float(y.mean());sd=max(float(y.std()),1e-10);ys=(y-mu)/sd;extra={}
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always')
  if method=='PLS':
   rank=int(np.linalg.matrix_rank(A));n=min(config['n_components'],A.shape[1],len(A)-1,max(rank,1));m=PLSRegression(n_components=n,scale=False,max_iter=1000,tol=1e-8);m.fit(A,ys);p=m.predict(B).ravel();extra['effective_components']=n
  elif method=='LASSO':
   # Same L1 objective as coordinate descent; LARS avoids slow convergence
   # of highly correlated latent columns. Explicit primal-dual gap is audited.
   alpha=config['alpha'];m=LassoLars(alpha=alpha,fit_intercept=False,max_iter=5000,eps=1e-12);m.fit(A,ys)
   coef=m.coef_.copy()
   def dualgap(coef):
    rr=ys-A@coef;dn=float(np.max(np.abs(A.T@rr)));c=min(1.,alpha*len(A)/dn) if dn>0 else 1.
    return max(0.,float((.5*np.dot(rr,rr)*(1+c*c)+alpha*len(A)*np.abs(coef).sum()-c*np.dot(rr,ys))/len(A)))
   gap=dualgap(coef);solver='LARS'
   if gap>1e-7:
    # Numerically robust solution of the SAME LASSO objective via its dual:
    # min_v 0.5 ||v-y||^2, subject to |A.T v| <= n*alpha.
    # Recover a primal solution on the active signed columns by bounded LS.
    bound=alpha*len(A)
    dual=minimize(lambda v:.5*np.sum((v-ys)**2),np.zeros(len(A)),jac=lambda v:v-ys,
      method='SLSQP',constraints=[LinearConstraint(A.T,-bound,bound)],
      options={'maxiter':1000,'ftol':1e-13})
    grad=A.T@dual.x
    for atol in [1e-6,1e-5,1e-4,1e-3]:
     active=np.abs(grad)>=bound*(1-atol);cc=np.zeros(A.shape[1])
     if active.any():
      signs=np.sign(grad[active]);ls=lsq_linear(A[:,active]*signs,ys-dual.x,
        bounds=(0,np.inf),method='bvls',tol=1e-13,max_iter=1000);cc[active]=ls.x*signs
     gg=dualgap(cc)
     if gg<gap:coef=cc;gap=gg;solver='LARS+dualQP'
     if gap<1e-7:break
    extra.update(dual_qp_iterations=int(dual.nit),dual_qp_success=bool(dual.success))
   if gap>1e-6:raise RuntimeError(f'LASSO optimality check failed: gap={gap}')
   p=B@coef;extra.update(n_iter=int(m.n_iter_),dual_gap=gap,solver=solver)
  elif method=='SVR':
   m=SVR(C=config['C'],epsilon=config['epsilon'],gamma=1/(2*config['ell']**2*A.shape[1]),cache_size=100);m.fit(A,ys);p=m.predict(B)
  elif method=='LightGBM':
   m=LGBMRegressor(n_estimators=100,learning_rate=.05,reg_lambda=1,min_data_in_bin=1,max_bin=63,n_jobs=1,verbosity=-1,random_state=6101,deterministic=True,force_col_wise=True,**config);m.fit(A,ys);p=m.booster_.predict(B,num_threads=1)
  elif method=='GBDT':
   m=GradientBoostingRegressor(n_estimators=100,learning_rate=.05,loss='squared_error',random_state=6101,**config);m.fit(A,ys);p=m.predict(B)
  elif method=='KRR':
   D=cdist(A,A,'sqeuclidean')/A.shape[1];Q=cdist(B,A,'sqeuclidean')/A.shape[1];K=np.exp(-D/(2*config['ell']**2));Q=np.exp(-Q/(2*config['ell']**2));p=Q@np.linalg.solve(K+config['alpha']*np.eye(len(A)),ys)
  else:raise ValueError(method)
 extra['warnings']=[type(w.message).__name__+': '+str(w.message)[:180] for w in ww]
 return p*sd+mu,extra

def predict_grid(A,y,B,method):
 if method!='KRR':
  res=[run_one(A,y,B,method,c) for c in GRIDS[method]];return np.column_stack([r[0] for r in res]),[r[1] for r in res]
 mu=y.mean();sd=max(y.std(),1e-10);v=(y-mu)/sd;D=cdist(A,A,'sqeuclidean')/A.shape[1];Q=cdist(B,A,'sqeuclidean')/A.shape[1];out=[]
 for ell in ELLS:
  K=np.exp(-D/(2*ell**2));T=np.exp(-Q/(2*ell**2));ev,U=eigh((K+K.T)/2,check_finite=False);ev=np.maximum(ev,0)
  pred=(T@U)@((U.T@v)[:,None]/(ev[:,None]+np.array(ALPHAS)[None,:]))*sd+mu;out.append(pred)
 return np.concatenate(out,axis=1),[{} for _ in GRIDS[method]]

def nested(X,y,tr,te,inner,method):
 t=time.time();sse=np.zeros(len(GRIDS[method]));nvalid=0;warn=[];maxgap=0.
 for a,b in inner:
  assert set(a)<=set(tr) and set(b)<=set(tr) and not set(a)&set(b)
  A,B,ok=prep(X[a],X[b]);pp,infos=predict_grid(A,y[a],B,method)
  sse+=((pp-y[b,None])**2).sum(0);nvalid+=len(b)
  for k,i in enumerate(infos):
   maxgap=max(maxgap,float(i.get('dual_gap',0)))
   if i.get('warnings'):warn.append(dict(candidate=k,messages=i['warnings']))
 k=int(np.argmin(sse));cfg=GRIDS[method][k];A,B,ok=prep(X[tr],X[te]);p,info=run_one(A,y[tr],B,method,cfg)
 if not np.isfinite(p).all():raise FloatingPointError('Nonfinite prediction')
 return p,dict(config=cfg,inner_mse=float(sse[k]/nvalid),elapsed_seconds=time.time()-t,n_features=int(ok.sum()),inner_warning_count=len(warn),max_inner_lasso_dual_gap=maxgap,final_info=info,grid_sse=sse.tolist(),warnings=warn)
