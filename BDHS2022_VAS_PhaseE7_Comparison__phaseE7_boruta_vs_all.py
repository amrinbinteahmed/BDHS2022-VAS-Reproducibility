"""Frozen Boruta fold-specific vs all-feature comparison; development only. No test outcome read."""
from pathlib import Path
import json, zipfile
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,recall_score,brier_score_loss
R=Path('/mnt/data');O=R/'PhaseE7_outputs';O.mkdir(exist_ok=True)
df=pd.read_csv(R/'BDHS2022_VAS_FINAL_Source_Validated.csv')
a=pd.read_csv(R/'PhaseD_outputs/holdout_assignments.csv',usecols=['source_KR_row','partition'])
f=pd.read_csv(R/'Boruta_verified/training_cv_fold_map.csv')
d=df.merge(a,on='source_KR_row',validate='one_to_one');d=d[d.partition=='train'].merge(f,on='source_KR_row',validate='one_to_one').reset_index(drop=True)
num=['child_age_months','maternal_age_years','living_children_count'];cat=['child_sex_code','maternal_education_code','paternal_education_code','paternal_occupation_code','maternal_work_code','wealth_quintile_code','residence_code','division_code','religion_code','media_weekly_any']
X=d[num+cat].copy();X['paternal_education_code']=X['paternal_education_code'].replace({8:np.nan});X['paternal_occupation_code']=X['paternal_occupation_code'].replace({98:np.nan})
y=d.Vitamin_A.to_numpy(int);w=d.sampling_weight.to_numpy(float);g=d.survey_cluster_id.astype(str).to_numpy()
rows=[]; preds=[];audit=[]
for fold in range(1,6):
 ti=np.where(d.validation_fold!=fold)[0];vi=np.where(d.validation_fold==fold)[0]
 assert set(g[ti]).isdisjoint(set(g[vi]))
 p=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
 xt=p.fit_transform(X.iloc[ti]);xv=p.transform(X.iloc[vi]);names=list(p.get_feature_names_out()); selected=pd.read_csv(R/f'Boruta_verified/fold_{fold}_features.csv'); confirmed=selected.loc[selected.confirmed,'feature'].tolist()
 # Boruta generated with num__ and cat__ prefix; verify exact names before selection
 idx=[names.index(s) for s in confirmed if s in names]
 assert len(idx)==len(confirmed) and len(idx)>0,(fold,confirmed,[n for n in names if 'age' in n]);
 totals={c:w[ti][y[ti]==c].sum() for c in (0,1)};wt=w[ti].copy()
 for c in (0,1):wt[y[ti]==c]*=w[ti].sum()/(2*totals[c])
 models={'Logistic Regression':lambda:LogisticRegression(C=.5,max_iter=500,random_state=42),'Random Forest':lambda:RandomForestClassifier(n_estimators=110,max_depth=8,min_samples_leaf=12,n_jobs=2,random_state=42),'XGBoost':lambda:XGBClassifier(n_estimators=110,max_depth=2,learning_rate=.05,n_jobs=2,eval_metric='logloss',random_state=42)}
 for feature_set,ix in [('All features',np.arange(xt.shape[1])),('Fold-specific Boruta',np.array(idx))]:
  for model_name,ctor in models.items():
   m=ctor();m.fit(xt[:,ix],y[ti],sample_weight=wt);prob=m.predict_proba(xv[:,ix])[:,1];pred=(prob>=.5).astype(int)
   rows.append(dict(fold=fold,features=feature_set,model=model_name,n_features=len(ix),n_validation=len(vi),roc_auc=roc_auc_score(y[vi],prob,sample_weight=w[vi]),average_precision=average_precision_score(y[vi],prob,sample_weight=w[vi]),balanced_accuracy=balanced_accuracy_score(y[vi],pred,sample_weight=w[vi]),sensitivity=recall_score(y[vi],pred,sample_weight=w[vi]),specificity=recall_score(1-y[vi],1-pred,sample_weight=w[vi]),brier=brier_score_loss(y[vi],prob,sample_weight=w[vi])))
   preds.extend(dict(fold=fold,source_KR_row=int(d.iloc[k].source_KR_row),features=feature_set,model=model_name,probability=float(pr),actual=int(y[k]),sampling_weight=float(w[k])) for k,pr in zip(vi,prob))
 audit.append(dict(fold=fold,n_train=len(ti),n_val=len(vi),confirmed=confirmed,n_features_all=xt.shape[1],psu_disjoint=True))
 print('Finished fold',fold,flush=True)
result=pd.DataFrame(rows);result.to_csv(O/'foldwise.csv',index=False);pd.DataFrame(preds).to_csv(O/'oof_predictions.csv',index=False);pd.DataFrame(audit).to_json(O/'fold_audit.json',orient='records',indent=2)
summary=result.groupby(['features','model'],as_index=False)[['roc_auc','average_precision','balanced_accuracy','sensitivity','specificity','brier','n_features']].mean();summary.to_csv(O/'summary.csv',index=False)
notes='Development-only paired comparison using frozen five group-exclusive folds and independently fitted preprocessing in each training fold. Boruta selection was obtained from each corresponding training fold (not from full training). Weights: survey weights times inverse weighted class prevalence. Fixed hyperparameters and fixed threshold 0.5; NOT nested tuning or unbiased final test. Boruta selection was not survey-weighted. No test outcome accessed. The five folds are for feature strategy comparison, not a claim of final tuned performance.'
(O/'README.txt').write_text(notes);print(summary.to_string(index=False))
with zipfile.ZipFile(R/'BDHS2022_VAS_PhaseE7_Comparison.zip','w',zipfile.ZIP_DEFLATED) as z:
 for path in O.iterdir():z.write(path,arcname=path.name)
 z.write(__file__,arcname=Path(__file__).name)
