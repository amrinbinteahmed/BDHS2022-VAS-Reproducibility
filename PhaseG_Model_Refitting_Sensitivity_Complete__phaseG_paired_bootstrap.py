from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.metrics import roc_auc_score,brier_score_loss
root=Path('/mnt/data');out=root/'PhaseG_Refitting'; d=pd.read_csv(root/'BDHS2022_VAS_FINAL_Source_Validated.csv'); pred=pd.read_csv(out/'scenario_holdout_predictions.csv');piv=pred.pivot(index='source_KR_row',columns='scenario',values='calibrated_probability'); meta=pred.drop_duplicates('source_KR_row')[['source_KR_row','observed','sampling_weight','survey_cluster_id']].merge(d[['source_KR_row','strata']],on='source_KR_row',validate='one_to_one').set_index('source_KR_row').loc[piv.index]
cols=piv.columns.tolist();y=meta.observed.to_numpy();w=meta.sampling_weight.to_numpy();pp=piv.to_numpy();groups=meta.groupby(['strata','survey_cluster_id']).indices;by={}
for (s,c),ix in groups.items():by.setdefault(s,[]).append(np.array(ix))
rng=np.random.default_rng(1978);rows=[];primary=cols.index('primary_reproduction')
for i in range(400):
 idx=np.concatenate([np.concatenate([clusters[j] for j in rng.integers(len(clusters),size=len(clusters))]) for clusters in by.values()]);yy=y[idx]
 if len(np.unique(yy))!=2:continue
 scores=[roc_auc_score(yy,pp[idx,k],sample_weight=w[idx]) for k in range(len(cols))]
 for k,name in enumerate(cols):
  if k==primary:continue
  rows.append({'replicate':i,'comparison':name,'auc_difference_vs_primary':scores[k]-scores[primary]})
res=pd.DataFrame(rows);res.to_csv(out/'paired_stratified_PSU_bootstrap_auc_differences.csv',index=False)
summary=res.groupby('comparison').auc_difference_vs_primary.agg(['mean',lambda x: x.quantile(.025),lambda x:x.quantile(.975)]);summary.columns=['mean_bootstrap_difference','ci_2_5','ci_97_5'];summary.to_csv(out/'paired_auc_difference_intervals.csv')
print(summary)
