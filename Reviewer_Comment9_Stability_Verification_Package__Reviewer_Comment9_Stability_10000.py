import io,zipfile,time
from pathlib import Path
import numpy as np,pandas as pd
from statsmodels.stats.multitest import multipletests
R=Path('/mnt/data'); reps=10000; seed=20261031
with zipfile.ZipFile(R/'PhaseE89_complete.zip') as z:
 frames=[pd.read_csv(io.BytesIO(z.read(n))) for n in z.namelist() if n.endswith('_predictions.csv')]
 rec=pd.read_csv(io.BytesIO(z.read('nested_summary.csv'))).set_index('model')
long=pd.concat(frames,ignore_index=True)
base=long[long.model=='XGBoost'][['outer','source_KR_row','observed','weight','survey_cluster_id']].merge(pd.read_csv(R/'BDHS2022_VAS_FINAL_Source_Validated.csv',usecols=['source_KR_row','strata']),on='source_KR_row',validate='one_to_one').sort_values(['outer','source_KR_row']).reset_index(drop=True)
models=rec.index.tolist(); n=len(base); y=base.observed.to_numpy(dtype=int);w=base.weight.to_numpy();folds=sorted(base.outer.unique()); nf=len(folds); nm=len(models)
probs=[]
for m in models:
 t=base[['outer','source_KR_row']].merge(long[long.model==m][['outer','source_KR_row','probability']],on=['outer','source_KR_row'],validate='one_to_one')
 assert t.probability.notna().all();probs.append(t.probability.to_numpy())
# weighted AUC via cumulative negatives at distinct scores, using pre-sorted model/fold order
fold_data=[]
for f in folds:
 ix=np.flatnonzero(base.outer.to_numpy()==f); yy=y[ix]; ww=w[ix]
 groupmap={}
 for (st,cl),ids in base.iloc[ix].groupby(['strata','survey_cluster_id']).indices.items():groupmap.setdefault(st,[]).append(np.asarray(ids,dtype=int))
 ranks=[]
 for pp in probs:
  p=pp[ix]; order=np.argsort(p,kind='mergesort'); vals=p[order]; end=np.r_[np.flatnonzero(np.diff(vals)!=0),len(vals)-1];start=np.r_[0,end[:-1]+1];ranks.append((order,start,end))
 fold_data.append((ix,yy,ww,groupmap,ranks))
def auc_weighted(yy,ww,order,start):
 sy=yy[order];sw=ww[order];pos=sw*sy;neg=sw*(1-sy)
 ps=np.add.reduceat(pos,start);ns=np.add.reduceat(neg,start)
 cp=np.cumsum(ns)-ns
 den=ps.sum()*ns.sum()
 return (ps*(cp+.5*ns)).sum()/den if den>0 else np.nan
point=np.empty(nm)
for j in range(nm):point[j]=np.mean([auc_weighted(yy,ww,*ranks[j][:2]) for _,yy,ww,_,ranks in fold_data])
assert np.max(np.abs(point-rec.loc[models,'mean_auc'].to_numpy()))<1e-12
rng=np.random.default_rng(seed); boot=np.empty((reps,nm)); tic=time.time()
for r in range(reps):
 fa=np.empty((nf,nm))
 for fi,(ix,yy,ww,groupmap,ranks) in enumerate(fold_data):
  counts=np.zeros(len(ix),dtype=np.int32)
  for groups in groupmap.values():
   draw=rng.integers(len(groups),size=len(groups))
   for k in draw:counts[groups[k]]+=1
  neww=ww*counts
  for j,(order,start,end) in enumerate(ranks):fa[fi,j]=auc_weighted(yy,neww,order,start)
 boot[r]=fa.mean(axis=0)
 if (r+1)%1000==0:print('completed',r+1,'seconds',round(time.time()-tic,1),flush=True)
ref=models.index('XGBoost'); rows=[]
for j,m in enumerate(models):
 if m=='XGBoost':continue
 delta=point[ref]-point[j]; bd=boot[:,ref]-boot[:,j];ci=np.quantile(bd,[.025,.975]);centered=bd-delta
 p=(1+np.sum(np.abs(centered)>=abs(delta)))/(reps+1)
 rows.append(dict(comparator=m,xgboost_mean_auc=point[ref],comparator_mean_auc=point[j],difference=delta,ci_lower=ci[0],ci_upper=ci[1],bootstrap_p_two_sided=p))
out=pd.DataFrame(rows);out['holm_adjusted_p']=multipletests(out.bootstrap_p_two_sided,method='holm')[1]
out.to_csv(R/'Reviewer_Comment9_Paired_AUC_Comparison_10000.csv',index=False)
pd.DataFrame(boot,columns=models).assign(replicate=np.arange(1,reps+1)).to_csv(R/'Reviewer_Comment9_Bootstrap10000_AUC.csv',index=False)
print(out.to_string(index=False));print('elapsed',time.time()-tic)
