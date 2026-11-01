from pathlib import Path
import numpy as np,pandas as pd,json,zipfile
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,balanced_accuracy_score,recall_score
P=Path('/mnt/data'); O=P/'PhaseG_Robustness';O.mkdir(exist_ok=True)
a=pd.read_csv(P/'PhaseE13_Locked_Holdout/holdout_predictions.csv'); d=pd.read_csv(P/'BDHS2022_VAS_FINAL_Source_Validated.csv');
a=a.merge(d[['source_KR_row','Vitamin_A','child_age_group','child_sex_code','wealth_quintile_code','residence_code','division_code','strata']],on='source_KR_row',validate='one_to_one')
assert len(a)==1458 and np.array_equal(a.observed,a.Vitamin_A)
cols=['child_age_group','child_sex_code','wealth_quintile_code','residence_code','division_code']
def met(t):
 y=t.observed.to_numpy().astype(int);w=t.sampling_weight.to_numpy();p=t.calibrated_probability.to_numpy();z=t.predicted.to_numpy().astype(int)
 return {'n':len(t),'clusters':t.survey_cluster_id.nunique(),'weighted_prevalence':np.average(y,weights=w),'roc_auc':roc_auc_score(y,p,sample_weight=w) if len(set(y))==2 else np.nan,'average_precision':average_precision_score(y,p,sample_weight=w) if len(set(y))==2 else np.nan,'brier':np.average((y-p)**2,weights=w),'sensitivity':recall_score(y,z,sample_weight=w,zero_division=0),'specificity':recall_score(1-y,1-z,sample_weight=w,zero_division=0),'balanced_accuracy':balanced_accuracy_score(y,z,sample_weight=w) if len(set(y))==2 else np.nan}
rows=[]
for c in ['overall']+cols:
 for v,t in ([('All',a)] if c=='overall' else a.groupby(c,dropna=False)):
  rows.append({'subgroup':c,'category':str(v),**met(t)})
res=pd.DataFrame(rows);res.to_csv(O/'subgroup_weighted_metrics.csv',index=False)
# Bootstrap survey-stratum cluster resampling; fixed predictions, no model refit. 
rng=np.random.default_rng(20261029); clusters={k:[g.copy() for _,g in t.groupby('survey_cluster_id')] for k,t in a.groupby('strata')}
boot=[]
for b in range(300):
 pieces=[]
 for st,groups in clusters.items():
  ix=rng.integers(0,len(groups),len(groups));pieces.extend(groups[j] for j in ix)
 t=pd.concat(pieces,ignore_index=True)
 for c in ['overall']+cols:
  for v,g in ([('All',t)] if c=='overall' else t.groupby(c,dropna=False)):
   m=met(g);boot.append({'rep':b,'subgroup':c,'category':str(v),'roc_auc':m['roc_auc'],'brier':m['brier'],'balanced_accuracy':m['balanced_accuracy']})
bt=pd.DataFrame(boot);bt.to_csv(O/'stratified_psu_bootstrap_replicates.csv',index=False)
ci=bt.groupby(['subgroup','category'])[['roc_auc','brier','balanced_accuracy']].quantile([.025,.975]).unstack().reset_index();ci.columns=['_'.join(map(str,x)).strip('_') for x in ci.columns];ci.to_csv(O/'subgroup_bootstrap_intervals.csv',index=False)
# Sensitivity to weights, and outcome coding (nonreceipt positive)
y=a.observed.to_numpy().astype(int);w=a.sampling_weight.to_numpy();p=a.calibrated_probability.to_numpy();z=a.predicted.to_numpy().astype(int)
sens={'weighted_auc':roc_auc_score(y,p,sample_weight=w),'unweighted_auc':roc_auc_score(y,p),'weighted_brier':np.average((y-p)**2,weights=w),'unweighted_brier':np.mean((y-p)**2),'nonreceipt_auc':roc_auc_score(1-y,1-p,sample_weight=w),'nonreceipt_average_precision':average_precision_score(1-y,1-p,sample_weight=w),'nonreceipt_sensitivity_at_locked_threshold':recall_score(1-y,1-z,sample_weight=w),'nonreceipt_precision_at_locked_threshold':np.sum(w[(y==0)&(z==0)])/np.sum(w[z==0])}
(O/'weighting_and_nonreceipt_sensitivity.json').write_text(json.dumps(sens,indent=2))
(O/'README.md').write_text('''# Phase G — Robustness and sensitivity analysis\n\nThis is a *post-hoc descriptive* analysis of the already-used, locked XGBoost holdout predictions (n=1458). No model retraining, new threshold, feature selection or holdout-driven model choice is performed.\n\nFiles: subgroup_weighted_metrics.csv, subgroup_bootstrap_intervals.csv, stratified_psu_bootstrap_replicates.csv, weighting_and_nonreceipt_sensitivity.json, phaseG_robustness.py.\n\nThe bootstrap resamples PSU clusters within survey strata and recomputes metrics using frozen predictions (300 replicates). It is NOT a bootstrap that refits the complete model, and should not be described as accounting for model-training uncertainty. For subgroups with limited clusters, intervals may be unstable. No multiple-comparison correction. Category codes should be mapped to original BDHS value labels before publication. Subgroup differences are exploratory and not causal.\n\nPotential additional robustness work: repeated group-aware nested CV; alternative survey weight specifications; refit bootstrap; temporal/external validation if data exist; subgroup fairness analysis with predeclared hypotheses.\n''')
with zipfile.ZipFile(P/'PhaseG_Robustness_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in O.iterdir():z.write(f,arcname=f.name)
 z.write(P/'phaseG_robustness.py',arcname='phaseG_robustness.py')
print(res[['subgroup','category','n','clusters','roc_auc','brier','balanced_accuracy']].to_string(index=False));print('SENS',sens)
