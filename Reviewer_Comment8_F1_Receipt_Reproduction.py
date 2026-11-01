"""Reproduce survey-weighted VAS receipt F1 and 500 PSU-bootstrap CI from locked predictions."""
from pathlib import Path
import io,zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
root=Path(__file__).resolve().parent
with zipfile.ZipFile(root/'BDHS2022_VAS_PhaseE13_Locked_Holdout(4).zip') as z:
    h=pd.read_csv(io.BytesIO(z.read('holdout_predictions.csv')))
    saved=pd.read_csv(io.BytesIO(z.read('holdout_cluster_bootstrap.csv')))
s=pd.read_csv(root/'BDHS2022_VAS_FINAL_Source_Validated.csv',usecols=['source_KR_row','strata','survey_cluster_id'])
d=h.merge(s,on='source_KR_row',validate='one_to_one',suffixes=('','_source'))
assert len(d)==1458 and d.strata.notna().all() and (d.survey_cluster_id==d.survey_cluster_id_source).all()
y=d.observed.to_numpy(dtype=int); pred=d.predicted.to_numpy(dtype=int); w=d.sampling_weight.to_numpy(dtype=float); p=d.calibrated_probability.to_numpy(dtype=float)
def f1(ix,positive):
    yy=(y[ix]==positive); pp=(pred[ix]==positive); ww=w[ix]
    tp=ww[yy & pp].sum(); fp=ww[~yy & pp].sum(); fn=ww[yy & ~pp].sum()
    return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else np.nan
by_stratum={}
for (st,cl),ix in d.groupby(['strata','survey_cluster_id']).indices.items():
    by_stratum.setdefault(st,[]).append(np.asarray(ix))
rng=np.random.default_rng(991)
rows=[]
for i in range(500):
    ix=np.concatenate([np.concatenate([clusters[j] for j in rng.integers(0,len(clusters),len(clusters))]) for clusters in by_stratum.values()])
    rows.append({'replicate':i+1,'F1_VAS_receipt':f1(ix,1),'F1_VAS_nonreceipt':f1(ix,0),'ROC_AUC_check':roc_auc_score(y[ix],p[ix],sample_weight=w[ix])})
b=pd.DataFrame(rows)
assert np.allclose(b.ROC_AUC_check.to_numpy(),saved.weighted_auc.to_numpy(),rtol=0,atol=1e-12)
res=[]
for name,positive in [('VAS_receipt',1),('VAS_nonreceipt',0)]:
    ci=np.quantile(b['F1_'+name],[.025,.975]); val=f1(np.arange(len(d)),positive)
    res.append({'metric':'survey_weighted_F1_'+name,'estimate':val,'ci_lower':ci[0],'ci_upper':ci[1],'bootstrap_replicates':500})
    print(f'{name}: {val:.8f} (95% CI {ci[0]:.8f}–{ci[1]:.8f})')
pd.DataFrame(res).to_csv(root/'Reviewer_Comment8_F1_Results.csv',index=False)
b.drop(columns='ROC_AUC_check').to_csv(root/'Reviewer_Comment8_F1_Bootstrap500.csv',index=False)
print('PASS: 500 original stratified PSU bootstrap ROC-AUC values matched exactly.')
