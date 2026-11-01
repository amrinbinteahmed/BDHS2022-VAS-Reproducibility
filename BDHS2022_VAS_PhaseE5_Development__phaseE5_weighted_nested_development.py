"""Phase E.5: development-only, group-disjoint, class-balanced 8-model evaluation.
Feature screening is explicitly NOT canonical Boruta. No holdout outcomes accessed.
"""
import json, hashlib, time, zipfile, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, recall_score, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore',category=UserWarning)
ROOT=Path('/mnt/data'); OUT=ROOT/'PhaseE5_outputs'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'BDHS2022_VAS_FINAL_Source_Validated.csv')
assign=pd.read_csv(ROOT/'PhaseD_outputs/holdout_assignments.csv',usecols=['source_KR_row','partition'])
d=df.merge(assign,on='source_KR_row',validate='one_to_one')
d=d[d.partition=='train'].copy().reset_index(drop=True)
num=['child_age_months','maternal_age_years','living_children_count']
cat=['child_sex_code','maternal_education_code','paternal_education_code','paternal_occupation_code','maternal_work_code','wealth_quintile_code','residence_code','division_code','religion_code','media_weekly_any']
X=d[num+cat].copy()
X['paternal_education_code']=X['paternal_education_code'].replace({8:np.nan})
X['paternal_occupation_code']=X['paternal_occupation_code'].replace({98:np.nan})
for c in cat:X[c]=X[c].map(lambda v:str(int(v)) if pd.notna(v) else np.nan)
y=d.Vitamin_A.to_numpy(dtype=int); w=d.sampling_weight.to_numpy(float); g=d.survey_cluster_id.astype(str).to_numpy()
def prep():return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)],verbose_feature_names_out=False)
def balance(y,w):
 out=w.copy(); totals=[w[y==c].sum() for c in (0,1)]; out[y==0]*=w.sum()/(2*totals[0]);out[y==1]*=w.sum()/(2*totals[1]);return out
models={
 'Logistic Regression':LogisticRegression(C=.5,max_iter=500,random_state=42),
 'Decision Tree':DecisionTreeClassifier(max_depth=4,min_samples_leaf=25,random_state=42),
 'Random Forest':RandomForestClassifier(n_estimators=110,max_depth=8,min_samples_leaf=12,n_jobs=2,random_state=42),
 'SVM':SVC(C=.5,kernel='rbf',probability=True,random_state=42),
 'AdaBoost':AdaBoostClassifier(n_estimators=80,learning_rate=.05,random_state=42),
 'XGBoost':XGBClassifier(n_estimators=110,max_depth=2,learning_rate=.05,n_jobs=2,eval_metric='logloss',random_state=42),
 'LightGBM':LGBMClassifier(n_estimators=110,max_depth=3,num_leaves=7,learning_rate=.05,verbosity=-1,n_jobs=2,random_state=42),
 'ANN':MLPClassifier(hidden_layer_sizes=(32,),alpha=1.,max_iter=180,early_stopping=True,random_state=42)
}
# ANN via weighted bootstrap of training rows only (not exact sample_weight; flagged in audit).
cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261029)
rows=[]; predictions=[]; start=time.time()
for fold,(ti,vi) in enumerate(cv.split(X,y,g),1):
 assert set(g[ti]).isdisjoint(set(g[vi])); pr=prep(); xt=pr.fit_transform(X.iloc[ti]); xv=pr.transform(X.iloc[vi]); wt=balance(y[ti],w[ti]); rng=np.random.default_rng(600+fold)
 for name,base in models.items():
  m=clone(base)
  if name=='ANN':
   sampled=rng.choice(len(ti),size=len(ti),replace=True,p=wt/wt.sum());m.fit(xt[sampled],y[ti][sampled])
  else:m.fit(xt,y[ti],sample_weight=wt)
  prob=m.predict_proba(xv)[:,1]
  predictions.extend(dict(fold=fold,model=name,source_KR_row=int(d.iloc[vi[j]].source_KR_row),y=int(y[vi[j]]),prob=float(prob[j]),sampling_weight=float(w[vi[j]])) for j in range(len(vi)))
  for th in [.5,.6,.65,.7]:
   pred=(prob>=th).astype(int)
   rows.append(dict(fold=fold,model=name,threshold=th,n_val=len(vi),roc_auc=roc_auc_score(y[vi],prob,sample_weight=w[vi]),average_precision=average_precision_score(y[vi],prob,sample_weight=w[vi]),balanced_accuracy=balanced_accuracy_score(y[vi],pred,sample_weight=w[vi]),sensitivity=recall_score(y[vi],pred,sample_weight=w[vi]),specificity=recall_score(1-y[vi],1-pred,sample_weight=w[vi]),brier=brier_score_loss(y[vi],prob,sample_weight=w[vi])))
 print('finished fold',fold,'elapsed seconds',round(time.time()-start,1),flush=True)
r=pd.DataFrame(rows);r.to_csv(OUT/'development_fold_metrics.csv',index=False)
p=pd.DataFrame(predictions);p.to_csv(OUT/'development_oof_predictions.csv',index=False)
summary=r.groupby(['model','threshold'],as_index=False)[['roc_auc','average_precision','balanced_accuracy','sensitivity','specificity','brier']].mean();summary.to_csv(OUT/'development_threshold_comparison.csv',index=False)
# Out-of-fold threshold selection on development set only; optimistic as not nested. NOT final unbiased metric.
best=summary.sort_values(['model','balanced_accuracy'],ascending=[True,False]).drop_duplicates('model');best.to_csv(OUT/'development_provisional_thresholds.csv',index=False)
audit={'status':'Development-only; NOT final nested validation, Boruta, or holdout results','n_development':len(d),'n_test_not_accessed':int((assign.partition=='test').sum()),'models':list(models),'feature_selection':'none; deliberate avoidance of claiming canonical Boruta','cv':'3-fold StratifiedGroupKFold, PSU exclusive','weights':'survey sampling weights multiplied by inverse weighted class prevalence within training fold','ANN':'weighted bootstrap approximation; not identical to direct sample_weight training','thresholds':'predefined 0.50,0.60,0.65,0.70; chosen on same OOF CV hence optimistic; require nested tuning','no_test_evaluation':True,'limitations':['Not canonical Boruta','No nested hyperparameter search','ANN training mechanism differs','Weighted CV point metrics are not survey-design confidence intervals','Probability calibration not adjusted for class reweighting','Only development partition analyzed'],'seconds':round(time.time()-start,1),'source_sha256':hashlib.sha256((ROOT/'BDHS2022_VAS_FINAL_Source_Validated.csv').read_bytes()).hexdigest()}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2));print(best[['model','threshold','roc_auc','balanced_accuracy','sensitivity','specificity']].to_string(index=False))
with zipfile.ZipFile(ROOT/'BDHS2022_VAS_PhaseE5_Development.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in OUT.iterdir():z.write(f,arcname=f.name)
 z.write(__file__,arcname=Path(__file__).name)
