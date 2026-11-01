from pathlib import Path
import pandas as pd, json
p=Path('/mnt/data/Boruta_verified')
data=pd.read_csv('/mnt/data/BDHS2022_VAS_FINAL_Source_Validated.csv')
split=pd.read_csv('/mnt/data/PhaseD_outputs/holdout_assignments.csv')
fm=pd.read_csv(p/'training_cv_fold_map.csv')
assert len(data)==7550 and data.source_KR_row.is_unique
assert len(split)==7550 and split.source_KR_row.is_unique
train=split[split.partition=='train']; test=split[split.partition=='test']
assert len(train)==6092 and len(test)==1458
assert set(train.survey_cluster_id).isdisjoint(test.survey_cluster_id)
assert fm.source_KR_row.is_unique and set(fm.source_KR_row)==set(train.source_KR_row)
assert fm.validation_fold.value_counts().index.isin(range(1,6)).all()
features={}
for k in range(1,6):
 d=pd.read_csv(p/f'fold_{k}_features.csv');assert d.feature.is_unique
 features[k]=d.set_index('feature')
full=pd.read_csv(p/'full_training_features.csv').set_index('feature')
rows=[]
for feature in sorted(set(full.index).union(*[set(f.index) for f in features.values()])):
 row={'feature':feature,'full_confirmed':bool(full.loc[feature,'confirmed']) if feature in full.index else None,'full_tentative':bool(full.loc[feature,'tentative']) if feature in full.index else None}
 for k,f in features.items():
  row[f'fold_{k}_status']= ('confirmed' if f.loc[feature,'confirmed'] else 'tentative' if f.loc[feature,'tentative'] else 'rejected') if feature in f.index else 'not_encoded'
 row['confirmed_folds']=sum(row[f'fold_{k}_status']=='confirmed' for k in features)
 row['tentative_folds']=sum(row[f'fold_{k}_status']=='tentative' for k in features)
 rows.append(row)
out=pd.DataFrame(rows).sort_values(['confirmed_folds','tentative_folds'],ascending=False)
out.to_csv('/mnt/data/Boruta_Feature_Stability_Audit.csv',index=False)
report={'data_rows':len(data),'training_rows':len(train),'holdout_rows':len(test),'psu_overlap':0,'fold_map_complete':True,'full_training_confirmed':full.index[full.confirmed].tolist(),'fold_confirmed_counts':{str(k):int(f.confirmed.sum()) for k,f in features.items()},'limitation':'BorutaPy selection unweighted, RF class_weight balanced. Not nested hyperparameter selection.'}
Path('/mnt/data/Boruta_Independent_Audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
