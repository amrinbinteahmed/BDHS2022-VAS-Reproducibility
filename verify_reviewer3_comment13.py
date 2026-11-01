"""Reproduce survey-weighted holdout class-specific sensitivity metrics.
Run: python verify_reviewer3_comment13.py
Input: PhaseG_Refitting/scenario_holdout_predictions.csv
Threshold 0.74 was selected on development data; never optimize on holdout.
"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
src=ROOT/'PhaseG_Refitting'/'scenario_holdout_predictions.csv'
df=pd.read_csv(src)
threshold=0.74
rows=[]
for name,g in df.groupby('scenario',sort=False):
    y=g['observed'].to_numpy(dtype=int)
    w=g['sampling_weight'].to_numpy(dtype=float)
    pred=g['calibrated_probability'].to_numpy(dtype=float)>=threshold
    assert len(g)==1458 and np.isin(y,[0,1]).all() and (w>0).all()
    tn=w[(y==0)&(~pred)].sum(); fp=w[(y==0)&pred].sum()
    fn=w[(y==1)&(~pred)].sum(); tp=w[(y==1)&pred].sum()
    nonvas_sens=tn/(tn+fp)
    nonvas_precision=tn/(tn+fn)
    nonvas_f1=2*nonvas_precision*nonvas_sens/(nonvas_precision+nonvas_sens)
    vas_sens=tp/(tp+fn)
    balanced=(vas_sens+nonvas_sens)/2
    rows.append({'Scenario':name,'n':len(g),'Threshold':threshold,'Balanced accuracy':balanced,
                 'VAS sensitivity':vas_sens,'Non-VAS sensitivity':nonvas_sens,
                 'Non-VAS precision':nonvas_precision,'Non-VAS F1':nonvas_f1})
out=pd.DataFrame(rows)
assert np.isclose(out.loc[out.Scenario=='primary_reproduction','Non-VAS F1'].iloc[0],0.451639,atol=1e-6)
path=ROOT/'Reviewer3_Comment13_Verified_ClassSpecific_Sensitivity.csv'
out.to_csv(path,index=False,float_format='%.8f')
print(out.round(4).to_string(index=False));print('Saved:',path)
