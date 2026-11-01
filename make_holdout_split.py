"""Reconstruct the fixed, PSU-disjoint development/holdout assignment."""
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--expect-final-counts',action='store_true');a=p.parse_args()
d=pd.read_csv(a.data)
required={'source_KR_row','survey_cluster_id','Vitamin_A','sampling_weight'}
assert required.issubset(d.columns),f'Missing: {required-set(d.columns)}'
assert d.source_KR_row.is_unique and d.survey_cluster_id.notna().all()
assert set(d.Vitamin_A.unique()) <= {0,1}
tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=20261029).split(d,d.Vitamin_A,d.survey_cluster_id.astype(str)))
assert set(d.iloc[tr].survey_cluster_id).isdisjoint(set(d.iloc[te].survey_cluster_id))
assignment=pd.DataFrame({'source_KR_row':d.source_KR_row,'partition':'train'})
assignment.loc[te,'partition']='test'
if a.expect_final_counts:
 assert len(d)==7550 and len(tr)==6092 and len(te)==1458,(len(d),len(tr),len(te))
o=Path(a.out_dir);o.mkdir(parents=True,exist_ok=True)
assignment.to_csv(o/'holdout_assignments.csv',index=False)
summary={'total':len(d),'train':len(tr),'test':len(te),'train_psus':int(d.iloc[tr].survey_cluster_id.nunique()),'test_psus':int(d.iloc[te].survey_cluster_id.nunique()),'seed':20261029,'test_size_fraction_of_groups':.20,'psu_overlap':0}
(o/'split_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
