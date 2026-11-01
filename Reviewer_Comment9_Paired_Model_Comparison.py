"""Exploratory paired PSU bootstrap for original 3x3 nested-CV out-of-fold ROC-AUC.
Requires PhaseE89_complete.zip and BDHS2022_VAS_FINAL_Source_Validated.csv.
Bootstrap preserves outer folds and samples PSUs within each (outer fold, survey stratum).
Results are conditional on fitted outer-fold models; model-selection uncertainty is not captured.
"""
import io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parent
ZIP=ROOT/'PhaseE89_complete.zip'
SOURCE=ROOT/'BDHS2022_VAS_FINAL_Source_Validated.csv'
REPS=500
SEED=20261031
with zipfile.ZipFile(ZIP) as z:
    frames=[pd.read_csv(io.BytesIO(z.read(n))) for n in z.namelist() if n.endswith('_predictions.csv')]
    recorded=pd.read_csv(io.BytesIO(z.read('nested_summary.csv'))).set_index('model')
long=pd.concat(frames,ignore_index=True)
assert long.shape[0]==6092*8 and not long.duplicated(['outer','model','source_KR_row']).any()
source=pd.read_csv(SOURCE,usecols=['source_KR_row','strata'])
base=long[long.model=='XGBoost'][['outer','source_KR_row','observed','weight','survey_cluster_id']].copy()
base=base.merge(source,on='source_KR_row',how='left',validate='one_to_one').sort_values(['outer','source_KR_row']).reset_index(drop=True)
assert base.strata.notna().all() and base.source_KR_row.is_unique
models=recorded.index.tolist()
probs={}
for model in models:
    x=long[long.model==model][['outer','source_KR_row','observed','weight','survey_cluster_id','probability']]
    x=base[['outer','source_KR_row']].merge(x,on=['outer','source_KR_row'],how='left',validate='one_to_one')
    assert x.probability.notna().all()
    assert np.array_equal(x.observed.to_numpy(),base.observed.to_numpy())
    assert np.allclose(x.weight,base.weight)
    assert np.array_equal(x.survey_cluster_id.to_numpy(),base.survey_cluster_id.to_numpy())
    probs[model]=x.probability.to_numpy()
y=base.observed.to_numpy().astype(int)
w=base.weight.to_numpy()
folds=sorted(base.outer.unique())
fold_indices={f:np.flatnonzero(base.outer.to_numpy()==f) for f in folds}
def mean_fold_auc(indices):
    return np.array([np.mean([roc_auc_score(y[ix],probs[m][ix],sample_weight=w[ix]) for ix in indices]) for m in models])
point=mean_fold_auc([fold_indices[f] for f in folds])
assert np.max(np.abs(point-recorded.loc[models,'mean_auc'].to_numpy()))<1e-12
# Clusters nested within outer fold and stratum. Cluster ID is preserved as a string.
groups={}
for (fold,st,cl),idx in base.groupby(['outer','strata','survey_cluster_id'],sort=True).indices.items():
    groups.setdefault((fold,st),[]).append(np.asarray(idx,dtype=int))
assert sum(len(x) for sets in groups.values() for x in sets)==len(base)
rng=np.random.default_rng(SEED)
boot=np.empty((REPS,len(models)),dtype=float)
for r in range(REPS):
    sampled=[]
    for f in folds:
        sub=[]
        for (ff,st),clusters in groups.items():
            if ff!=f:continue
            draws=rng.integers(0,len(clusters),size=len(clusters))
            sub.extend(clusters[j] for j in draws)
        ix=np.concatenate(sub)
        if np.unique(y[ix]).size<2:raise ValueError('single-class replicate')
        sampled.append(ix)
    boot[r]=mean_fold_auc(sampled)
ref_idx=models.index('XGBoost')
results=[]
for j,m in enumerate(models):
    if m=='XGBoost':continue
    delta=point[ref_idx]-point[j]
    bd=boot[:,ref_idx]-boot[:,j]
    ci=np.quantile(bd,[.025,.975])
    # Two-sided centered-bootstrap test under null delta=0; approximate, exploratory.
    centered=bd-delta
    p=(1+np.sum(np.abs(centered)>=abs(delta)))/(REPS+1)
    results.append(dict(comparator=m,xgboost_mean_auc=point[ref_idx],comparator_mean_auc=point[j],difference=delta,ci_lower=ci[0],ci_upper=ci[1],bootstrap_p_two_sided=p))
out=pd.DataFrame(results)
out['holm_adjusted_p']=multipletests(out.bootstrap_p_two_sided,method='holm')[1]
out['ci_excludes_zero']=(out.ci_lower>0)|(out.ci_upper<0)
out.to_csv(ROOT/'Reviewer_Comment9_Paired_AUC_Comparison.csv',index=False)
pd.DataFrame(boot,columns=models).assign(replicate=np.arange(1,REPS+1)).to_csv(ROOT/'Reviewer_Comment9_Bootstrap500_AUC.csv',index=False)
print('Reproduced original nested-CV mean AUC to 1e-12 for all 8 models.')
print('N',len(base),'PSUs',base.survey_cluster_id.nunique(),'folds',len(folds),'strata',base.strata.nunique())
print(out.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
