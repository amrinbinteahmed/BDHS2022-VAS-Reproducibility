"""Reproduce BDHS 2022 VAS locked-holdout metrics and PSU bootstrap CIs.
Inputs: PhaseE13 locked holdout ZIP and validated source CSV in same directory.
Requires pandas numpy scikit-learn. Does not retrain model.
"""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / 'BDHS2022_VAS_PhaseE13_Locked_Holdout(2).zip'
SOURCE = ROOT / 'BDHS2022_VAS_FINAL_Source_Validated.csv'
OUT = ROOT / 'VAS_Holdout_Verification'
OUT.mkdir(exist_ok=True)
with ZipFile(ZIP) as z:
    a = pd.read_csv(BytesIO(z.read('holdout_predictions.csv')))
src = pd.read_csv(SOURCE, usecols=['source_KR_row', 'strata'])
a = a.merge(src, on='source_KR_row', validate='one_to_one')
assert len(a) == 1458 and a.source_KR_row.is_unique
assert a.groupby('survey_cluster_id').strata.nunique().max() == 1

def metrics(t):
    y = t.observed.to_numpy(dtype=int)
    pred = t.predicted.to_numpy(dtype=int)
    prob = t.calibrated_probability.to_numpy(dtype=float)
    w = t.sampling_weight.to_numpy(dtype=float)
    assert set(np.unique(y)) <= {0,1} and set(np.unique(pred)) <= {0,1}
    tp = w[(y==1)&(pred==1)].sum(); tn = w[(y==0)&(pred==0)].sum()
    fp = w[(y==0)&(pred==1)].sum(); fn = w[(y==1)&(pred==0)].sum()
    safe = lambda x,d: x/d if d > 0 else np.nan
    sensitivity = safe(tp,tp+fn); specificity = safe(tn,tn+fp)
    precision_nonvas = safe(tn,tn+fn)
    f1_nonvas = safe(2*tn,2*tn+fn+fp)
    mcc_denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = safe(tp*tn-fp*fn,mcc_denom)
    prevalence = safe(w[y==1].sum(),w.sum())
    return dict(roc_auc=roc_auc_score(y,prob,sample_weight=w),
                average_precision_VAS=average_precision_score(y,prob,sample_weight=w),
                balanced_accuracy=(sensitivity+specificity)/2,
                sensitivity_VAS=sensitivity,specificity_nonVAS=specificity,
                brier_score=brier_score_loss(y,prob,sample_weight=w),
                log_loss=log_loss(y,prob,sample_weight=w,labels=[0,1]),
                accuracy=safe(tp+tn,tp+tn+fp+fn),
                precision_nonVAS=precision_nonvas,f1_nonVAS=f1_nonvas,
                average_precision_nonVAS=average_precision_score(1-y,1-prob,sample_weight=w),
                mcc=mcc, prevalence_VAS=prevalence,
                prevalence_nonVAS=1-prevalence,
                always_VAS_accuracy=prevalence,always_VAS_balanced_accuracy=0.5)

point = metrics(a)
# Same stratified PSU bootstrap and seed as Phase I2, with MCC added.
clusters = {s: [g for _,g in v.groupby('survey_cluster_id')] for s,v in a.groupby('strata')}
rng = np.random.default_rng(20261029)
reps = []
for _ in range(500):
    parts = []
    for s, groups in clusters.items():
        parts.extend(groups[j] for j in rng.integers(0,len(groups),size=len(groups)))
    reps.append(metrics(pd.concat(parts,ignore_index=True)))
boot = pd.DataFrame(reps)
rows = [{'metric':k,'estimate':v,'ci95_lower':boot[k].quantile(.025),
         'ci95_upper':boot[k].quantile(.975)} for k,v in point.items()]
result = pd.DataFrame(rows)
result.to_csv(OUT/'holdout_metrics_with_MCC_500_PSU_bootstrap.csv',index=False)
boot.to_csv(OUT/'bootstrap_500_replicates.csv',index=False)
print(result.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
print('\nNOTE: Fixed holdout predictions; CIs do not incorporate model training uncertainty.')
