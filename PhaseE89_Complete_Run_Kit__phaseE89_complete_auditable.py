"""Complete, resumable 3x3 nested group CV for all 8 prespecified models.
No holdout usage. Inner folds choose model config AND decision threshold.
Feature strategy: all 13 prespecified predictors. Boruta not nested here.
"""
import os, json, warnings, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, recall_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore',category=FutureWarning)
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--split',required=True);p.add_argument('--out',default='PhaseE89_complete');args=p.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(args.data);split=pd.read_csv(args.split)
d=d.merge(split[['source_KR_row','partition']],on='source_KR_row',validate='one_to_one')
train=d[d.partition=='train'].reset_index(drop=True);test=d[d.partition=='test']
assert len(train)==6092 and len(test)==1458
assert set(train.survey_cluster_id).isdisjoint(set(test.survey_cluster_id))
num=['child_age_months','maternal_age_years','living_children_count']
cat=['child_sex_code','maternal_education_code','paternal_education_code','paternal_occupation_code','maternal_work_code','wealth_quintile_code','residence_code','division_code','religion_code','media_weekly_any']
X=train[num+cat];y=train.Vitamin_A.to_numpy().astype(int);g=train.survey_cluster_id.to_numpy();w=train.sampling_weight.to_numpy()
def preprocess():
 return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
configs={'Logistic Regression':[{'C':.1},{'C':1.}], 'Decision Tree':[{'max_depth':3},{'max_depth':6}], 'Random Forest':[{'max_depth':4},{'max_depth':8}], 'SVM':[{'C':.1},{'C':1.}], 'AdaBoost':[{'n_estimators':50},{'n_estimators':100}], 'XGBoost':[{'max_depth':2},{'max_depth':4}], 'LightGBM':[{'max_depth':2},{'max_depth':4}], 'ANN':[{'hidden_layer_sizes':(20,)},{'hidden_layer_sizes':(40,)}]}
def estimator(name,params):
 if name=='Logistic Regression':return LogisticRegression(max_iter=500,random_state=42,**params)
 if name=='Decision Tree':return DecisionTreeClassifier(min_samples_leaf=30,random_state=42,**params)
 if name=='Random Forest':return RandomForestClassifier(n_estimators=80,min_samples_leaf=10,n_jobs=1,random_state=42,**params)
 if name=='SVM':return SVC(probability=True,random_state=42,**params)
 if name=='AdaBoost':return AdaBoostClassifier(random_state=42,**params)
 if name=='XGBoost':return XGBClassifier(n_estimators=90,learning_rate=.06,subsample=.85,colsample_bytree=.9,eval_metric='logloss',n_jobs=1,random_state=42,**params)
 if name=='LightGBM':return LGBMClassifier(n_estimators=90,learning_rate=.05,num_leaves=15,min_child_samples=30,verbosity=-1,n_jobs=1,random_state=42,**params)
 if name=='ANN':return MLPClassifier(max_iter=120,early_stopping=True,random_state=42,**params)
def predict(name,params,ti,vi):
 pp=preprocess();xt=pp.fit_transform(X.iloc[ti]);xv=pp.transform(X.iloc[vi]);m=estimator(name,params)
 sw=w[ti].copy();yt=y[ti]; prevalence=np.average(yt,weights=sw)
 sw=sw*np.where(yt==1,.5/max(prevalence,1e-8),.5/max(1-prevalence,1e-8));sw=sw/sw.mean()
 if name=='ANN':
  rng=np.random.default_rng(42+len(ti));idx=rng.choice(len(ti),len(ti),replace=True,p=sw/sw.sum());m.fit(xt[idx],yt[idx])
 else:m.fit(xt,yt,sample_weight=sw)
 return m.predict_proba(xv)[:,1]
outer=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026).split(X,y,g))
thresholds=np.arange(.35,.801,.05)
for oi,(ot,ov) in enumerate(outer,1):
 assert set(g[ot]).isdisjoint(set(g[ov]))
 inner=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=110+oi).split(X.iloc[ot],y[ot],g[ot]))
 for name,plist in configs.items():
  tag=f'outer{oi}_{name.replace(" ","_")}'
  if (out/f'{tag}_metrics.json').exists():print('SKIP',tag,flush=True);continue
  candidates=[]; innerrows=[]
  for ci,params in enumerate(plist):
   ppred=[];yy=[];ww=[]
   for ii,(it,iv) in enumerate(inner,1):
    ti=ot[it];vi=ot[iv];assert set(g[ti]).isdisjoint(set(g[vi]))
    pr=predict(name,params,ti,vi)
    auc=roc_auc_score(y[vi],pr,sample_weight=w[vi]);innerrows.append({'outer':oi,'model':name,'config':ci,'inner':ii,'auc':auc,'n_train':len(ti),'n_validation':len(vi)})
    ppred.extend(pr);yy.extend(y[vi]);ww.extend(w[vi])
   ppred=np.asarray(ppred);yy=np.asarray(yy);ww=np.asarray(ww)
   meanauc=np.mean([r['auc'] for r in innerrows if r['config']==ci])
   threshold_scores=[balanced_accuracy_score(yy,ppred>=t,sample_weight=ww) for t in thresholds]
   best_t=float(thresholds[np.argmax(threshold_scores)])
   candidates.append((meanauc,ci,params,best_t,float(max(threshold_scores))))
  chosen=max(candidates,key=lambda v:(v[0],-v[1]));_,ci,params,thr,innerba=chosen
  pr=predict(name,params,ot,ov); pred=(pr>=thr).astype(int)
  metrics={'outer':oi,'model':name,'config':ci,'params':params,'threshold':thr,'inner_mean_auc':chosen[0],'inner_balanced_accuracy_at_threshold':innerba,'outer_auc':roc_auc_score(y[ov],pr,sample_weight=w[ov]),'outer_ap':average_precision_score(y[ov],pr,sample_weight=w[ov]),'outer_balanced_accuracy':balanced_accuracy_score(y[ov],pred,sample_weight=w[ov]),'outer_sensitivity':recall_score(y[ov],pred,sample_weight=w[ov]),'outer_specificity':recall_score(1-y[ov],1-pred,sample_weight=w[ov]),'outer_brier':brier_score_loss(y[ov],pr,sample_weight=w[ov]),'outer_logloss':log_loss(y[ov],np.clip(pr,1e-8,1-1e-8),sample_weight=w[ov])}
  pd.DataFrame(innerrows).to_csv(out/f'{tag}_inner.csv',index=False)
  pd.DataFrame({'source_KR_row':train.source_KR_row.iloc[ov].to_numpy(),'outer':oi,'model':name,'observed':y[ov],'probability':pr,'prediction':pred,'weight':w[ov],'survey_cluster_id':g[ov]}).to_csv(out/f'{tag}_predictions.csv',index=False)
  (out/f'{tag}_metrics.json').write_text(json.dumps(metrics,indent=2,default=str));print(tag,round(metrics['outer_auc'],4),flush=True)
metrics=[json.loads(f.read_text()) for f in sorted(out.glob('outer*_metrics.json'))];summary=pd.DataFrame(metrics)
if len(summary):
 summary.to_csv(out/'outer_fold_metrics.csv',index=False)
 agg=summary.groupby('model').agg(n_folds=('outer','nunique'),mean_auc=('outer_auc','mean'),sd_auc=('outer_auc','std'),mean_balanced_accuracy=('outer_balanced_accuracy','mean'),mean_sensitivity=('outer_sensitivity','mean'),mean_specificity=('outer_specificity','mean'),mean_brier=('outer_brier','mean')).sort_values('mean_auc',ascending=False)
 agg.to_csv(out/'nested_summary.csv')
(out/'AUDIT.json').write_text(json.dumps({'completed_model_outer_pairs':len(metrics),'expected_pairs':24,'complete':len(metrics)==24,'outer_folds':3,'inner_folds':3,'test_used_for_tuning':False,'feature_selection':'all 13 prespecified predictors; NOT nested Boruta','weighting':'sampling weights x inverse class prevalence; ANN weighted bootstrap approximate','threshold':'optimized on pooled inner OOF only','calibration':'not performed; Brier scores diagnostic only','CI':'not yet estimated','final_holdout_evaluated':False},indent=2))
print('DONE',len(metrics),'of 24 model-fold pairs',flush=True)
