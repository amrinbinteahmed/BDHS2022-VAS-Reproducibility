"""Post-holdout exploratory sensitivity: fixed XGBoost hyperparameters, frozen partition.
No model reselection; survey-weighted OOF calibration per scenario, fixed primary threshold 0.74.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss,balanced_accuracy_score,recall_score
from xgboost import XGBClassifier
ROOT=Path('/mnt/data');OUT=ROOT/'PhaseG_Refitting';OUT.mkdir(exist_ok=True)
d=pd.read_csv(ROOT/'BDHS2022_VAS_FINAL_Source_Validated.csv');s=pd.read_csv(ROOT/'PhaseD_outputs/holdout_assignments.csv')
d=d.merge(s[['source_KR_row','partition']],on='source_KR_row',validate='one_to_one')
a=d.query('partition=="train"').reset_index(drop=True);b=d.query('partition=="test"').reset_index(drop=True)
assert (len(a),len(b))==(6092,1458)
assert set(a.survey_cluster_id).isdisjoint(set(b.survey_cluster_id))
num=['child_age_months','maternal_age_years','living_children_count'];cat=['child_sex_code','maternal_education_code','paternal_education_code','paternal_occupation_code','maternal_work_code','wealth_quintile_code','residence_code','division_code','religion_code','media_weekly_any']
scenarios={'primary_reproduction':('balanced_survey',num+cat),'survey_only_no_class_balance':('survey_only',num+cat),'unweighted_no_class_balance':('unweighted',num+cat),'omit_media_exposure':('balanced_survey',num+[x for x in cat if x!='media_weekly_any'])}
cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2468)
rows=[];allpred=[];oofrows=[]
for name,(weighting,cols) in scenarios.items():
    nums=[x for x in num if x in cols];cats=[x for x in cat if x in cols]
    def fitpredict(tr,va):
        prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)])
        xt=prep.fit_transform(tr[cols]);xv=prep.transform(va[cols]);y=tr.Vitamin_A.to_numpy();w=tr.sampling_weight.to_numpy()
        if weighting=='balanced_survey':
            prev=np.average(y,weights=w);sw=w*np.where(y==1,.5/prev,.5/(1-prev));sw=sw/sw.mean()
        elif weighting=='survey_only':sw=w/w.mean()
        else:sw=np.ones(len(y))
        m=XGBClassifier(n_estimators=90,learning_rate=.06,subsample=.85,colsample_bytree=.9,max_depth=2,eval_metric='logloss',n_jobs=1,random_state=42)
        m.fit(xt,y,sample_weight=sw)
        return m.predict_proba(xv)[:,1]
    oof=np.full(len(a),np.nan)
    for fold,(ti,vi) in enumerate(cv.split(a[cols],a.Vitamin_A,a.survey_cluster_id),1):
        oof[vi]=fitpredict(a.iloc[ti],a.iloc[vi]);print(name,'OOF',fold,flush=True)
    assert np.isfinite(oof).all()
    cal=LogisticRegression(C=1e6,max_iter=2000,random_state=42)
    cal.fit(oof.reshape(-1,1),a.Vitamin_A,sample_weight=a.sampling_weight)
    raw=fitpredict(a,b);p=cal.predict_proba(raw.reshape(-1,1))[:,1]
    y=b.Vitamin_A.to_numpy();w=b.sampling_weight.to_numpy();decision=p>=.74
    rows.append({'scenario':name,'n_train':len(a),'n_test':len(b),'predictor_count':len(cols),'weighted_auc':roc_auc_score(y,p,sample_weight=w),'weighted_brier':brier_score_loss(y,p,sample_weight=w),'weighted_balanced_accuracy_at_primary_threshold':balanced_accuracy_score(y,decision,sample_weight=w),'weighted_sensitivity_at_primary_threshold':recall_score(y,decision,sample_weight=w),'weighted_specificity_at_primary_threshold':recall_score(1-y,~decision,sample_weight=w)})
    allpred.append(pd.DataFrame({'source_KR_row':b.source_KR_row,'scenario':name,'observed':y,'sampling_weight':w,'survey_cluster_id':b.survey_cluster_id,'raw_probability':raw,'calibrated_probability':p}))
    oofrows.append(pd.DataFrame({'source_KR_row':a.source_KR_row,'scenario':name,'observed':a.Vitamin_A,'oof_probability':oof}))
    print(name,rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'model_refit_comparison.csv',index=False)
pd.concat(allpred).to_csv(OUT/'scenario_holdout_predictions.csv',index=False)
pd.concat(oofrows).to_csv(OUT/'scenario_training_oof.csv',index=False)
# Confirm primary exact reproduction against locked predictions.
ref=pd.read_csv(ROOT/'PhaseE13_Locked_Holdout/holdout_predictions.csv')
rep=allpred[0].merge(ref[['source_KR_row','raw_probability','calibrated_probability']],on='source_KR_row',suffixes=('_refit','_locked'),validate='one_to_one')
checks={'primary_raw_max_abs_difference':float(np.max(abs(rep.raw_probability_refit-rep.raw_probability_locked))),'primary_calibrated_max_abs_difference':float(np.max(abs(rep.calibrated_probability_refit-rep.calibrated_probability_locked))),'primary_reproduced':bool(np.allclose(rep.calibrated_probability_refit,rep.calibrated_probability_locked,rtol=1e-9,atol=1e-9))}
(OUT/'reproduction_check.json').write_text(json.dumps(checks,indent=2))
(OUT/'README.md').write_text('''# Phase G — Exploratory model-refitting sensitivity\n\nFour scenarios refitted with frozen XGBoost hyperparameters and the same frozen train/holdout partition. Three PSU-group-disjoint training folds produce OOF scores for sigmoid calibration; the calibrator uses survey sampling weights. Each full training model predicts the existing holdout. No holdout-driven selection or threshold tuning. Primary threshold 0.74 is fixed for all scenarios.\n\n**Critical**: Alternative scenarios were conceived AFTER primary holdout access, so all comparisons are post-hoc exploratory, NOT new independent validations. Primary XGBoost remains unchanged. These are not survey-design-based ML estimators. No full refitting bootstrap is provided; existing cluster bootstrap is conditional on predictions.\n''')
print('CHECKS',checks)
