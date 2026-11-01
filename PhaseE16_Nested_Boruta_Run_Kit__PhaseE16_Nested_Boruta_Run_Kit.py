"""Run on Colab: pip install boruta xgboost lightgbm scikit-learn pandas
Training-only 3x3 nested comparison of all features vs fold-local Boruta for XGBoost.
All 8 models extension possible; no holdout access by design.
"""
import numpy as np,pandas as pd
from pathlib import Path
from boruta import BorutaPy
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
R=Path('/mnt/data');d=pd.read_csv(R/'BDHS2022_VAS_FINAL_Source_Validated.csv');s=pd.read_csv(R/'PhaseD_outputs/holdout_assignments.csv');d=d.merge(s[['source_KR_row','partition']],on='source_KR_row',validate='one_to_one');a=d[d.partition=='train'].reset_index(drop=True)
cols=['child_age_months','maternal_age_years','living_children_count','child_sex_code','maternal_education_code','paternal_education_code','paternal_occupation_code','maternal_work_code','wealth_quintile_code','residence_code','division_code','religion_code','media_weekly_any']
y=a.Vitamin_A.to_numpy();g=a.survey_cluster_id.to_numpy();w=a.sampling_weight.to_numpy();outer=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026);records=[]
def transform(fit,eval):
 n=cols[:3];c=cols[3:];im=SimpleImputer(strategy='median');x1=im.fit_transform(fit[n]);z1=im.transform(eval[n]);ic=SimpleImputer(strategy='most_frequent');v=ic.fit_transform(fit[c]);z=ic.transform(eval[c]);enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1);return np.column_stack([x1,enc.fit_transform(v)]),np.column_stack([z1,enc.transform(z)])
def select_boruta(X,Y):
 selector=BorutaPy(RandomForestClassifier(n_estimators=200,max_depth=5,n_jobs=1,random_state=42,class_weight='balanced'),n_estimators='auto',max_iter=50,random_state=42,verbose=0);selector.fit(X,Y);mask=selector.support_.copy();return mask if mask.any() else (selector.support_ | selector.support_weak_) if selector.support_weak_.any() else np.ones(X.shape[1],dtype=bool)
def train_predict(x,z,Y,W,depth):
 prev=np.average(Y,weights=W);sw=W*np.where(Y==1,.5/prev,.5/(1-prev));sw/=sw.mean();m=XGBClassifier(n_estimators=90,learning_rate=.06,max_depth=depth,subsample=.85,colsample_bytree=.9,n_jobs=1,eval_metric='logloss',random_state=42);m.fit(x,Y,sample_weight=sw);return m.predict_proba(z)[:,1]
for oi,(ot,ov) in enumerate(outer.split(a[cols],y,g),1):
 inn=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=110+oi)
 for strategy in ['all','nested_boruta']:
  candidates=[]
  for depth in [2,4]:
   auc=[]
   for it,iv in inn.split(a.iloc[ot][cols],y[ot],g[ot]):
    ti=ot[it];vi=ot[iv];x,z=transform(a.iloc[ti],a.iloc[vi]);mask=select_boruta(x,y[ti]) if strategy=='nested_boruta' else np.ones(len(cols),bool);p=train_predict(x[:,mask],z[:,mask],y[ti],w[ti],depth);auc.append(roc_auc_score(y[vi],p,sample_weight=w[vi]))
   candidates.append((float(np.mean(auc)),depth))
  depth=max(candidates)[1];x,z=transform(a.iloc[ot],a.iloc[ov]);mask=select_boruta(x,y[ot]) if strategy=='nested_boruta' else np.ones(len(cols),bool);p=train_predict(x[:,mask],z[:,mask],y[ot],w[ot],depth)
  records.append({'outer':oi,'strategy':strategy,'depth':depth,'outer_weighted_auc':roc_auc_score(y[ov],p,sample_weight=w[ov]),'selected_features':';'.join(np.array(cols)[mask]),'n_features':int(mask.sum())});print(records[-1],flush=True)
pd.DataFrame(records).to_csv(R/'E16_nested_Boruta_results.csv',index=False)
