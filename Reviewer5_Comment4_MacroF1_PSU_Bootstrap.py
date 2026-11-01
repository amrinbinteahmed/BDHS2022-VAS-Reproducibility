"""Reviewer #5 Comment 4: survey-weighted macro-F1 and 95% stratified-PSU bootstrap CI.

Reproduces the 500 bootstrap resamples of verify_VAS_holdout_table4.py
(seed 20261029), using fixed, locked holdout predictions. No model refitting.
Run: python Reviewer5_Comment4_MacroF1_PSU_Bootstrap.py
Dependencies: numpy, pandas (no sklearn needed).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRED = ROOT / 'PhaseE13_Locked_Holdout' / 'holdout_predictions.csv'
SOURCE = ROOT / 'BDHS2022_VAS_FINAL_Source_Validated.csv'
OUT = ROOT / 'Reviewer5_Comment4_MacroF1_Results.csv'
REPS = ROOT / 'Reviewer5_Comment4_MacroF1_Bootstrap_500.csv'

holdout = pd.read_csv(PRED)
strata = pd.read_csv(SOURCE, usecols=['source_KR_row', 'strata'])
data = holdout.merge(strata, on='source_KR_row', validate='one_to_one')
assert len(data) == 1458 and data.source_KR_row.is_unique
assert data.groupby('survey_cluster_id').strata.nunique().max() == 1
assert set(data.observed.unique()) == {0., 1.}
assert set(data.predicted.unique()) == {0, 1}
assert (data.sampling_weight > 0).all()

def weighted_macro_f1(t, always_vas=False):
    y = t.observed.to_numpy(dtype=int)
    pred = np.ones(len(t), dtype=int) if always_vas else t.predicted.to_numpy(dtype=int)
    w = t.sampling_weight.to_numpy(dtype=float)
    tp = w[(y == 1) & (pred == 1)].sum()
    tn = w[(y == 0) & (pred == 0)].sum()
    fp = w[(y == 0) & (pred == 1)].sum()
    fn = w[(y == 1) & (pred == 0)].sum()
    f1_vas = 2*tp/(2*tp + fp + fn) if (2*tp + fp + fn) else 0.0
    f1_nonvas = 2*tn/(2*tn + fp + fn) if (2*tn + fp + fn) else 0.0
    return (f1_vas + f1_nonvas)/2

point = {'XGBoost_macro_F1': weighted_macro_f1(data),
         'Always_VAS_macro_F1': weighted_macro_f1(data, always_vas=True)}
clusters = {s: [g for _, g in v.groupby('survey_cluster_id')]
            for s, v in data.groupby('strata')}
rng = np.random.default_rng(20261029)
replicates = []
for _ in range(500):
    parts = []
    for groups in clusters.values():
        parts.extend(groups[j] for j in rng.integers(0, len(groups), size=len(groups)))
    sample = pd.concat(parts, ignore_index=True)
    replicates.append({'XGBoost_macro_F1': weighted_macro_f1(sample),
                       'Always_VAS_macro_F1': weighted_macro_f1(sample, always_vas=True)})
boot = pd.DataFrame(replicates)
boot.to_csv(REPS, index=False)
summary = pd.DataFrame([{'Metric': name, 'Estimate': value,
                         'CI_95_Lower': boot[name].quantile(.025),
                         'CI_95_Upper': boot[name].quantile(.975)}
                        for name, value in point.items()])
summary.to_csv(OUT, index=False)
print(summary.to_string(index=False, float_format=lambda z: f'{z:.6f}'))
print('Fixed-prediction bootstrap; excludes model training and selection uncertainty.')
