"""Phase E: BorutaPy selection with frozen PSU-exclusive holdout.
Usage: python run_boruta_phaseE.py --data BDHS2022_VAS_FINAL_Source_Validated.csv --split holdout_assignments.csv --out Boruta_results
Requires: pip install pandas numpy scikit-learn boruta
This is canonical BorutaPy, UNWEIGHTED selection; survey weights are not used by BorutaPy.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from boruta import BorutaPy

NUM = ['child_age_months','maternal_age_years','living_children_count']
CAT = ['child_sex_code','maternal_education_code','paternal_education_code',
       'paternal_occupation_code','maternal_work_code','wealth_quintile_code',
       'residence_code','division_code','religion_code','media_weekly_any']
SEED=20261029

def make_preprocessor():
    return ColumnTransformer([
        ('num',SimpleImputer(strategy='median',add_indicator=False),NUM),
        ('cat',__import__('sklearn').pipeline.Pipeline([
            ('impute',SimpleImputer(strategy='most_frequent')),
            ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)
    ],sparse_threshold=0)

def select(train_df, label, outdir, max_iter, n_estimators):
    p=make_preprocessor()
    X=p.fit_transform(train_df[NUM+CAT])
    X=np.asarray(X,dtype=np.float64)
    names=p.get_feature_names_out()
    y=train_df['Vitamin_A'].astype(int).to_numpy()
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=6,
        class_weight='balanced',n_jobs=-1,random_state=SEED)
    boruta=BorutaPy(estimator=rf,n_estimators=n_estimators,perc=100,
        alpha=0.05,two_step=True,max_iter=max_iter,random_state=SEED,verbose=0)
    boruta.fit(X,y)
    res=pd.DataFrame({'feature':names,'confirmed':boruta.support_,
        'tentative':boruta.support_weak_,'rank':boruta.ranking_})
    res.to_csv(outdir/f'{label}_features.csv',index=False)
    return {'fold':label,'n_rows':len(train_df),'n_features':len(names),
            'confirmed':int(boruta.support_.sum()),'tentative':int(boruta.support_weak_.sum()),
            'rejected':int((~boruta.support_ & ~boruta.support_weak_).sum())}

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data',required=True);a.add_argument('--split',required=True)
    a.add_argument('--out',default='Boruta_results');a.add_argument('--max-iter',type=int,default=100)
    a.add_argument('--trees',type=int,default=300)
    args=a.parse_args(); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data); splits=pd.read_csv(args.split)
    assert df.source_KR_row.is_unique and splits.source_KR_row.is_unique
    df=df.merge(splits[['source_KR_row','partition']],on='source_KR_row',validate='one_to_one')
    assert len(df)==len(splits)
    train=df.loc[df.partition.eq('train')].copy()
    test=df.loc[df.partition.eq('test')].copy()
    assert not set(train.survey_cluster_id).intersection(set(test.survey_cluster_id))
    assert len(train)==6092 and len(test)==1458, 'Frozen Phase D counts changed; inspect before running.'
    assert train[NUM+CAT].notna().any().all()
    # No test predictors or labels are used for any fit, fold, or feature selection.
    sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    summaries=[]; fold_map=[]
    for i,(idx,_val_idx) in enumerate(sgkf.split(train,train.Vitamin_A,train.survey_cluster_id),1):
        fit=train.iloc[idx]; val=train.iloc[_val_idx]
        assert set(fit.survey_cluster_id).isdisjoint(set(val.survey_cluster_id))
        summaries.append(select(fit,f'fold_{i}',out,args.max_iter,args.trees))
        fold_map.extend({'source_KR_row':int(k),'validation_fold':i} for k in val.source_KR_row)
        print('Finished',summaries[-1],flush=True)
    pd.DataFrame(fold_map).to_csv(out/'training_cv_fold_map.csv',index=False)
    summaries.append(select(train,'full_training',out,args.max_iter,args.trees))
    pd.DataFrame(summaries).to_csv(out/'boruta_summary.csv',index=False)
    meta={'note':'BorutaPy on fold-specific training data only. Unweighted selection; RF uses class_weight=balanced. Never touched holdout for fitting. Tentative features are NOT confirmed.',
          'data_rows':len(df),'training_rows':len(train),'untouched_holdout_rows':len(test),
          'seed':SEED,'max_iter':args.max_iter,'trees':args.trees,
          'python':sys.version,'boruta':'BorutaPy'}
    (out/'audit.json').write_text(json.dumps(meta,indent=2))
    print('DONE: zip the Boruta_results folder and upload it.')
if __name__=='__main__':main()
