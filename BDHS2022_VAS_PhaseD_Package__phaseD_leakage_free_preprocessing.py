"""Phase D: reproducible, cluster-exclusive holdout and training-only preprocessing.
Inputs: source-validated BDHS 2022 VAS CSV. Run: python phaseD_leakage_free_preprocessing.py
No model selection, Boruta, or imputation is fitted on full data.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'BDHS2022_VAS_FINAL_Source_Validated.csv'
OUT=ROOT/'PhaseD_outputs'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(SOURCE)
assert len(df)==7550 and df['Vitamin_A'].notna().all()
assert df['survey_cluster_id'].notna().all()
assert (df['sampling_weight']>0).all()

# Do not use survey identifiers, survey weight, source row IDs, or outcome as predictors.
NUMERIC=['child_age_months','maternal_age_years','living_children_count']
CATEGORICAL=['child_sex_code','maternal_education_code','paternal_education_code',
             'paternal_occupation_code','maternal_work_code','wealth_quintile_code',
             'residence_code','division_code','religion_code','media_weekly_any']
FEATURES=NUMERIC+CATEGORICAL
X=df[FEATURES].copy()
# DHS paternal education 8 = don't know; paternal occupation 98 = don't know.
X['paternal_education_code']=X['paternal_education_code'].replace({8:np.nan})
X['paternal_occupation_code']=X['paternal_occupation_code'].replace({98:np.nan})
# Cast nominal variables to string labels to prevent accidental numeric ordinal modeling.
for col in CATEGORICAL:
    X[col]=X[col].map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
y=df['Vitamin_A'].astype(int)
groups=df['survey_cluster_id'].astype(str)

# Group-exclusive holdout, with deterministic candidate selection minimizing outcome and weight balance differences.
# Choosing by label prevalence is an outcome-informed split choice; predeclared seed and single split avoids such optimization.
splitter=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=20261029)
train_idx,test_idx=next(splitter.split(X,y,groups))
assert not set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx]))

# CV splits only on development set. StratifiedGroupKFold balances class proportions approximately.
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261029)
folds=[]
for k,(a,b) in enumerate(cv.split(X.iloc[train_idx],y.iloc[train_idx],groups.iloc[train_idx]),1):
    ia=train_idx[a]; ib=train_idx[b]
    assert not set(groups.iloc[ia]).intersection(set(groups.iloc[ib]))
    folds.append({'fold':k,'train_rows':len(ia),'validation_rows':len(ib),
                  'train_clusters':groups.iloc[ia].nunique(),'validation_clusters':groups.iloc[ib].nunique(),
                  'train_positive_fraction':round(float(y.iloc[ia].mean()),5),
                  'validation_positive_fraction':round(float(y.iloc[ib].mean()),5)})
    pd.DataFrame({'source_KR_row':df.iloc[ib]['source_KR_row'].values,'fold':k}).to_csv(OUT/f'cv_fold_{k}_validation_rows.csv',index=False)

# Fit preprocessing only on development data for exploratory matrix export. In Phase E, re-fit pipeline inside EACH CV fold.
num_pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())])
cat_pipe=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                   ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
prep=ColumnTransformer([('numeric',num_pipe,NUMERIC),('categorical',cat_pipe,CATEGORICAL)],
                       remainder='drop',verbose_feature_names_out=True)
train_matrix=prep.fit_transform(X.iloc[train_idx])
test_matrix=prep.transform(X.iloc[test_idx])
assert train_matrix.shape[1]==test_matrix.shape[1]
assert np.isfinite(train_matrix).all() and np.isfinite(test_matrix).all()
columns=prep.get_feature_names_out().tolist()
pd.DataFrame(train_matrix,columns=columns).assign(source_KR_row=df.iloc[train_idx]['source_KR_row'].values,
                Vitamin_A=y.iloc[train_idx].values,sampling_weight=df.iloc[train_idx]['sampling_weight'].values).to_csv(OUT/'train_preprocessed.csv',index=False)
pd.DataFrame(test_matrix,columns=columns).assign(source_KR_row=df.iloc[test_idx]['source_KR_row'].values,
                Vitamin_A=y.iloc[test_idx].values,sampling_weight=df.iloc[test_idx]['sampling_weight'].values).to_csv(OUT/'test_preprocessed.csv',index=False)
assign=pd.DataFrame({'source_KR_row':df['source_KR_row'],'survey_cluster_id':groups,
                     'partition':'train','Vitamin_A':y,'sampling_weight':df['sampling_weight']})
assign.loc[test_idx,'partition']='test'
assign.to_csv(OUT/'holdout_assignments.csv',index=False)
dump(prep,OUT/'train_fitted_preprocessor.joblib')

report={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'status':'Phase D complete; Phase E cross-validation MUST fit preprocessing separately within folds',
        'n_total':len(df),'n_train':len(train_idx),'n_test':len(test_idx),
        'clusters_total':groups.nunique(),'clusters_train':groups.iloc[train_idx].nunique(),
        'clusters_test':groups.iloc[test_idx].nunique(),
        'train_positive_rate':round(float(y.iloc[train_idx].mean()),6),
        'test_positive_rate':round(float(y.iloc[test_idx].mean()),6),
        'train_weighted_positive_rate':round(float(np.average(y.iloc[train_idx],weights=df.iloc[train_idx]['sampling_weight'])),6),
        'test_weighted_positive_rate':round(float(np.average(y.iloc[test_idx],weights=df.iloc[test_idx]['sampling_weight'])),6),
        'feature_count_after_encoding':len(columns),'numeric_features':NUMERIC,'categorical_features':CATEGORICAL,
        'train_missing_after_special_code_handling':{c:int(X.iloc[train_idx][c].isna().sum()) for c in FEATURES},
        'cv_folds':folds,'important_caveats':[
            'The held-out test partition must not be used for model selection or threshold tuning.',
            'GroupShuffleSplit is cluster-exclusive but not exactly stratified by survey stratum; evaluate representativeness.',
            'Sampling weights are retained; actual weighted fitting/metrics and survey-aware uncertainty are Phase E.',
            'The exported training-fitted preprocessor is NOT valid for fold-level CV; clone and fit inside each fold.',
            'Maternal BMI and breastfeeding excluded from primary model pending sensitivity analyses.',
            'No Boruta or other supervised feature selection has yet been performed.' ]}
(OUT/'PhaseD_Audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['n_total','n_train','n_test','clusters_train','clusters_test','train_positive_rate','test_positive_rate','feature_count_after_encoding']},indent=2))
