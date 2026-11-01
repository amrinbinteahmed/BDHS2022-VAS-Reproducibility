"""Predeclared XGBoost holdout evaluation. No retuning after holdout.
Model chosen from earlier 3-fold nested development: XGBoost; max_depth=2 by majority.
OOF group CV determines sigmoid calibration and threshold; holdout evaluated once.
"""
import json, warnings, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,recall_score,brier_score_loss,log_loss,confusion_matrix
from xgboost import XGBClassifier
warnings.filterwarnings('ignore',category=UserWarning)
parser = argparse.ArgumentParser(description="Locked XGBoost holdout evaluation; private data required")
parser.add_argument('--data', required=True, help='Validated analysis-ready CSV (private)')
parser.add_argument('--split', required=True, help='Original verified PSU-disjoint assignments CSV (private)')
parser.add_argument('--out', required=True, help='PRIVATE output directory; contains row-level predictions')
args = parser.parse_args()
out = Path(args.out).expanduser().resolve()
# Deliberately prevent writing individual-level holdout predictions into the repository.
repo_root = Path(__file__).resolve().parents[2]
if out == repo_root or repo_root in out.parents and 'PRIVATE' not in out.parts:
    parser.error('Output is within repository but outside PRIVATE/. Use --out PRIVATE/PhaseE13_Locked_Holdout')
if not Path(args.data).is_file() or not Path(args.split).is_file():
    parser.error('Both --data and --split must exist; see DATA_ACCESS.md')
out.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(args.data)
s = pd.read_csv(args.split)
required_data = {'source_KR_row','Vitamin_A','survey_cluster_id','sampling_weight', 'strata',
                 'child_age_months','maternal_age_years','living_children_count',
                 'child_sex_code','maternal_education_code','paternal_education_code',
                 'paternal_occupation_code','maternal_work_code','wealth_quintile_code',
                 'residence_code','division_code','religion_code','media_weekly_any'}
missing = required_data.difference(d.columns)
if missing:
    parser.error(f'Analysis data missing columns: {sorted(missing)}')
if not {'source_KR_row','partition'}.issubset(s.columns):
    parser.error('Split CSV requires source_KR_row and partition')
if d.source_KR_row.duplicated().any() or s.source_KR_row.duplicated().any():
    parser.error('Duplicate source_KR_row found in data or split')
if len(d) != len(s) or set(d.source_KR_row) != set(s.source_KR_row):
    parser.error('Data and split row identifiers differ; do not regenerate historical split blindly')
if set(s.partition) != {'train','test'}:
    parser.error('Split partition must contain train and test')
d=d.merge(s[['source_KR_row','partition']],on='source_KR_row',validate='one_to_one')
a=d[d.partition=='train'].reset_index(drop=True);b=d[d.partition=='test'].reset_index(drop=True)
assert len(a)==6092 and len(b)==1458
assert set(a.survey_cluster_id).isdisjoint(set(b.survey_cluster_id))
num=['child_age_months','maternal_age_years','living_children_count'];cat=['child_sex_code','maternal_education_code','paternal_education_code','paternal_occupation_code','maternal_work_code','wealth_quintile_code','residence_code','division_code','religion_code','media_weekly_any'];cols=num+cat
locked={'model':'XGBoost','reason':'highest mean outer ROC-AUC among 8 models in prior nested CV; differences not established significant','feature_strategy':'all 13 prespecified predictors','max_depth':2,'max_depth_reason':'selected in 2 of 3 prior outer folds','n_estimators':90,'learning_rate':0.06,'subsample':0.85,'colsample_bytree':0.9,'min_samples':'not applicable','calibration':'sigmoid Platt scaling on grouped OOF raw scores','threshold':'max weighted balanced accuracy on grouped OOF calibrated probabilities; ties smallest threshold','holdout_access':'only after locking model, preprocessing, calibration and threshold rule','limitations':['calibration fit on pooled OOF predictions from multiple fitted models','survey weights and class weights affect probability scale','holdout is cluster-disjoint, but inference bootstrap is an approximation','single selected model holdout evaluation; full 8-model holdout comparison not performed']}
(out/'LOCKED_PROTOCOL.json').write_text(json.dumps(locked,indent=2))
def prep():return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
def fitpredict(tra, val):
 p=prep();xt=p.fit_transform(tra[cols]);xv=p.transform(val[cols]);y=tra.Vitamin_A.to_numpy();w=tra.sampling_weight.to_numpy();prev=np.average(y,weights=w);sw=w*np.where(y==1,.5/prev,.5/(1-prev));sw/=sw.mean()
 m=XGBClassifier(n_estimators=90,learning_rate=.06,subsample=.85,colsample_bytree=.9,max_depth=2,eval_metric='logloss',n_jobs=1,random_state=42);m.fit(xt,y,sample_weight=sw);return m.predict_proba(xv)[:,1]
cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2468);oof=np.full(len(a),np.nan)
for fold,(ti,vi) in enumerate(cv.split(a[cols],a.Vitamin_A,a.survey_cluster_id),1):
 assert set(a.survey_cluster_id.iloc[ti]).isdisjoint(set(a.survey_cluster_id.iloc[vi]));oof[vi]=fitpredict(a.iloc[ti],a.iloc[vi]);print('OOF',fold,flush=True)
assert np.isfinite(oof).all()
cal=LogisticRegression(C=1e6,max_iter=2000,random_state=42);cal.fit(oof.reshape(-1,1),a.Vitamin_A,sample_weight=a.sampling_weight)
cal_oof=cal.predict_proba(oof.reshape(-1,1))[:,1]
thresholds=np.arange(.20,.801,.01);scores=[balanced_accuracy_score(a.Vitamin_A,cal_oof>=t,sample_weight=a.sampling_weight) for t in thresholds];threshold=float(thresholds[int(np.argmax(scores))]);locked['selected_threshold']=threshold;locked['oof_weighted_balanced_accuracy']=float(max(scores));(out/'LOCKED_PROTOCOL.json').write_text(json.dumps(locked,indent=2))
pd.DataFrame({'source_KR_row':a.source_KR_row,'outcome':a.Vitamin_A,'oof_raw_probability':oof,'oof_calibrated_probability':cal_oof,'sampling_weight':a.sampling_weight,'survey_cluster_id':a.survey_cluster_id}).to_csv(out/'training_oof_calibration.csv',index=False)
# Holdout touched only after protocol and threshold are locked.
raw=fitpredict(a,b);prob=cal.predict_proba(raw.reshape(-1,1))[:,1];y=b.Vitamin_A.to_numpy();w=b.sampling_weight.to_numpy();pred=(prob>=threshold).astype(int)
def metrics(y,p,pred,w):return {'weighted_auc':float(roc_auc_score(y,p,sample_weight=w)),'weighted_ap':float(average_precision_score(y,p,sample_weight=w)),'weighted_balanced_accuracy':float(balanced_accuracy_score(y,pred,sample_weight=w)),'weighted_sensitivity':float(recall_score(y,pred,sample_weight=w)),'weighted_specificity':float(recall_score(1-y,1-pred,sample_weight=w)),'weighted_brier':float(brier_score_loss(y,p,sample_weight=w)),'weighted_log_loss':float(log_loss(y,np.clip(p,1e-8,1-1e-8),sample_weight=w))}
res={'n_test':len(b),'n_clusters':int(b.survey_cluster_id.nunique()),'threshold':threshold,'calibrated':metrics(y,prob,pred,w),'raw':metrics(y,raw,(raw>=.5).astype(int),w),'weighted_confusion_matrix':confusion_matrix(y,pred,sample_weight=w).tolist(),'unweighted_confusion_matrix':confusion_matrix(y,pred).tolist()}
# Stratified PSU bootstrap conditional on fixed predictions; approximate survey uncertainty.
rng=np.random.default_rng(991);groups=b.groupby(['strata','survey_cluster_id']).indices if 'strata' in b.columns else b.groupby(['strata_alternative','survey_cluster_id']).indices
by_stratum={}
for (st,cl),idx in groups.items():by_stratum.setdefault(st,[]).append(np.asarray(idx))
boot=[]
for rep in range(500):
 ix=np.concatenate([np.concatenate([cls[j] for j in rng.integers(0,len(cls),len(cls))]) for cls in by_stratum.values()]); yy=y[ix]
 if len(np.unique(yy))<2:continue
 boot.append(metrics(yy,prob[ix],pred[ix],w[ix]))
ci={k:[float(np.quantile([r[k] for r in boot],.025)),float(np.quantile([r[k] for r in boot],.975))] for k in boot[0]}
res['approximate_stratified_PSU_bootstrap_CI']=ci;res['bootstrap_replicates']=len(boot)
(out/'holdout_metrics.json').write_text(json.dumps(res,indent=2));pd.DataFrame({'source_KR_row':b.source_KR_row,'observed':y,'raw_probability':raw,'calibrated_probability':prob,'predicted':pred,'sampling_weight':w,'survey_cluster_id':b.survey_cluster_id}).to_csv(out/'holdout_predictions.csv',index=False)
pd.DataFrame(boot).to_csv(out/'holdout_cluster_bootstrap.csv',index=False)
print(json.dumps(res,indent=2),flush=True)
