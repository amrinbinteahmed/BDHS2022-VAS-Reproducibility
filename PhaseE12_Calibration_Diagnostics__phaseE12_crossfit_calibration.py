import zipfile,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
z=zipfile.ZipFile('/mnt/data/PhaseE89_complete.zip')
a=pd.concat([pd.read_csv(z.open(n)) for n in z.namelist() if n.endswith('_predictions.csv')],ignore_index=True)
out=Path('/mnt/data/PhaseE12_calibration');out.mkdir(exist_ok=True)
records=[];preds=[]
for model,sub in a.groupby('model'):
 for fold in (1,2,3):
  tr=sub[sub.outer!=fold];te=sub[sub.outer==fold]
  y=tr.observed.to_numpy();w=tr.weight.to_numpy();p=np.clip(tr.probability.to_numpy(),1e-5,1-1e-5)
  q=np.clip(te.probability.to_numpy(),1e-5,1-1e-5)
  logit=lambda v: np.log(v/(1-v))
  lr=LogisticRegression(C=1e4,max_iter=1000)
  lr.fit(logit(p).reshape(-1,1),y,sample_weight=w)
  iso=IsotonicRegression(out_of_bounds='clip',y_min=1e-5,y_max=1-1e-5)
  iso.fit(p,y,sample_weight=w)
  options={'raw':q,'sigmoid':lr.predict_proba(logit(q).reshape(-1,1))[:,1],'isotonic':iso.predict(q)}
  for method,v in options.items():
   v=np.clip(v,1e-5,1-1e-5)
   records.append({'model':model,'outer':fold,'calibration':method,'n':len(te),'weighted_brier':brier_score_loss(te.observed,v,sample_weight=te.weight),'weighted_logloss':log_loss(te.observed,v,sample_weight=te.weight),'weighted_auc':roc_auc_score(te.observed,v,sample_weight=te.weight),'mean_predicted':np.average(v,weights=te.weight),'observed_prevalence':np.average(te.observed,weights=te.weight)})
   preds.append(pd.DataFrame({'model':model,'outer':fold,'source_KR_row':te.source_KR_row.to_numpy(),'observed':te.observed.to_numpy(),'weight':te.weight.to_numpy(),'survey_cluster_id':te.survey_cluster_id.to_numpy(),'calibration':method,'probability':v}))
pd.DataFrame(records).to_csv(out/'calibration_fold_metrics.csv',index=False)
pd.concat(preds).to_csv(out/'calibration_crossfit_predictions.csv',index=False)
summary=pd.DataFrame(records).groupby(['model','calibration'],as_index=False).agg(mean_brier=('weighted_brier','mean'),mean_logloss=('weighted_logloss','mean'),mean_auc=('weighted_auc','mean'))
summary.to_csv(out/'calibration_summary.csv',index=False)
(out/'LIMITATIONS.md').write_text('Cross-fitted calibration DIAGNOSTIC only: for each outer fold, sigmoid/isotonic calibration was fitted on OOF predictions from the other two outer folds and assessed on the held-out outer fold. These calibration training predictions arise from differently trained models, so this is NOT a deployable final calibration pipeline and not a fully nested assessment. Selecting the calibration method from these results and reporting its same-fold performance incurs selection optimism. The frozen final test set remains untouched. Final calibration should use cross-fitted training predictions from the selected final model configuration and be evaluated once on holdout. Survey weights applied to calibrator and metrics; no survey-stratified inference here. Calibration cannot be inferred from ROC-AUC alone.\n')
print(summary.sort_values(['model','mean_brier']).to_string(index=False))
