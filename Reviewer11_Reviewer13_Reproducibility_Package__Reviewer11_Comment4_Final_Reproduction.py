"""Reviewer 11 Comment 4: fixed holdout predictions, original stratified PSU bootstrap.
Requires BDHS2022_VAS_PhaseE13_Locked_Holdout(4).zip and BDHS2022_VAS_FINAL_Source_Validated.csv.
Does not refit models or tune threshold. Uses original seed=991 and 500 draws.
"""
from pathlib import Path
import zipfile, io, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parent
ARCH=ROOT/'BDHS2022_VAS_PhaseE13_Locked_Holdout(4).zip'
SOURCE=ROOT/'BDHS2022_VAS_FINAL_Source_Validated.csv'
with zipfile.ZipFile(ARCH) as z:
    h=pd.read_csv(io.BytesIO(z.read('holdout_predictions.csv')))
    orig=json.loads(z.read('holdout_metrics.json'))
    oldboot=pd.read_csv(io.BytesIO(z.read('holdout_cluster_bootstrap.csv')))
source=pd.read_csv(SOURCE, usecols=['source_KR_row','strata','survey_cluster_id'])
assert source.source_KR_row.is_unique
h=h.merge(source[['source_KR_row','strata','survey_cluster_id']],on='source_KR_row',how='left',validate='one_to_one',suffixes=('','_source'))
assert h.strata.notna().all() and (h.survey_cluster_id==h.survey_cluster_id_source).all()
assert len(h)==1458 and h.survey_cluster_id.nunique()==135
Y=h.observed.to_numpy(dtype=int);P=h.calibrated_probability.to_numpy(float);C=h.predicted.to_numpy(dtype=int);W=h.sampling_weight.to_numpy(float)
assert np.all((P>=0)&(P<=1)) and set(np.unique(Y))=={0,1}
def safe(a,b): return a/b if b else float('nan')
def metrics(ix):
    y=Y[ix];p=P[ix];c=C[ix];w=W[ix]
    tn=w[(y==0)&(c==0)].sum();fp=w[(y==0)&(c==1)].sum();fn=w[(y==1)&(c==0)].sum();tp=w[(y==1)&(c==1)].sum()
    recall0=safe(tn,tn+fp);recall1=safe(tp,tp+fn)
    prec0=safe(tn,tn+fn);prec1=safe(tp,tp+fp)
    f10=safe(2*prec0*recall0,prec0+recall0);f11=safe(2*prec1*recall1,prec1+recall1)
    denom=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(TN=tn,FP=fp,FN=fn,TP=tp,
                TN_row_percent=recall0,FP_row_percent=1-recall0,
                FN_row_percent=1-recall1,TP_row_percent=recall1,
                NPV_VAS_positive=prec0,PPV_VAS_positive=prec1,
                precision_nonVAS=prec0,precision_VAS=prec1,
                recall_nonVAS=recall0,recall_VAS=recall1,
                F1_nonVAS=f10,F1_VAS=f11,macro_F1=(f10+f11)/2,
                balanced_accuracy=(recall0+recall1)/2,
                weighted_accuracy=safe(tn+tp,tn+fp+fn+tp),
                MCC=safe(tp*tn-fp*fn,denom),
                ROC_AUC=roc_auc_score(y,p,sample_weight=w),
                AP_VAS=average_precision_score(y,p,sample_weight=w),
                AP_nonVAS=average_precision_score(1-y,1-p,sample_weight=w))
point=metrics(np.arange(len(h)))
assert np.isclose(point['ROC_AUC'],orig['calibrated']['weighted_auc'])
assert np.isclose(point['balanced_accuracy'],orig['calibrated']['weighted_balanced_accuracy'])
assert np.allclose([point[x] for x in ['TN','FP','FN','TP']],np.array(orig['weighted_confusion_matrix']).ravel())
# Reconstruct original bootstrap index construction exactly (original phaseE13 script).
groups=h.groupby(['strata','survey_cluster_id']).indices
by_stratum={}
for (st,cl),idx in groups.items():by_stratum.setdefault(st,[]).append(np.asarray(idx))
rng=np.random.default_rng(991)
records=[]
for rep in range(500):
    ix=np.concatenate([np.concatenate([cls[j] for j in rng.integers(0,len(cls),len(cls))]) for cls in by_stratum.values()])
    if len(np.unique(Y[ix]))<2:continue
    records.append(metrics(ix))
boot=pd.DataFrame(records)
assert len(boot)==500
for new,old in [('ROC_AUC','weighted_auc'),('AP_VAS','weighted_ap'),('balanced_accuracy','weighted_balanced_accuracy'),('recall_VAS','weighted_sensitivity'),('recall_nonVAS','weighted_specificity')]:
    assert np.allclose(boot[new],oldboot[old],rtol=1e-9,atol=1e-9),(new,'original bootstrap mismatch')
rows=[]
for k,v in point.items():
    lo,hi=np.quantile(boot[k].dropna(),[.025,.975]);rows.append({'metric':k,'estimate':v,'CI_2.5%':lo,'CI_97.5%':hi})
result=pd.DataFrame(rows)
result.to_csv(ROOT/'Reviewer11_Comment4_Final_Metrics.csv',index=False)
boot.to_csv(ROOT/'Reviewer11_Comment4_Final_Bootstrap500.csv',index=False)
h[['source_KR_row','strata','survey_cluster_id']].to_csv(ROOT/'Reviewer11_Comment4_Holdout_Strata_Verification.csv',index=False)
print('VERIFIED: original bootstrap sequence reproduced for all five shared metrics')
print('n=',len(h),'PSUs=',h.survey_cluster_id.nunique(),'strata=',h.strata.nunique())
print(result.round(5).to_string(index=False))
