"""Reproduce survey-weighted Cohen's kappa and its original stratified-PSU bootstrap CI.

Place in same folder as:
  BDHS2022_VAS_PhaseE13_Locked_Holdout(4).zip
  BDHS2022_VAS_FINAL_Source_Validated.csv

No model refitting, recalibration, or threshold tuning.
"""
from pathlib import Path
import io, json, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ARCH = ROOT / 'BDHS2022_VAS_PhaseE13_Locked_Holdout(4).zip'
SOURCE = ROOT / 'BDHS2022_VAS_FINAL_Source_Validated.csv'
with zipfile.ZipFile(ARCH) as z:
    holdout = pd.read_csv(io.BytesIO(z.read('holdout_predictions.csv')))
    saved_boot = pd.read_csv(io.BytesIO(z.read('holdout_cluster_bootstrap.csv')))
source = pd.read_csv(SOURCE, usecols=['source_KR_row', 'strata', 'survey_cluster_id'])
assert source.source_KR_row.is_unique
h = holdout.merge(source, on='source_KR_row', how='left', validate='one_to_one', suffixes=('', '_source'))
assert h.strata.notna().all()
assert (h.survey_cluster_id == h.survey_cluster_id_source).all()
assert len(h) == 1458 and h.survey_cluster_id.nunique() == 135 and h.strata.nunique() == 16
Y = h.observed.to_numpy(dtype=int)
C = h.predicted.to_numpy(dtype=int)
W = h.sampling_weight.to_numpy(dtype=float)
P = h.calibrated_probability.to_numpy(dtype=float)

def weighted_kappa(indices):
    y, c, w = Y[indices], C[indices], W[indices]
    total = w.sum()
    observed_agreement = w[y == c].sum() / total
    actual_one = w[y == 1].sum() / total
    predicted_one = w[c == 1].sum() / total
    chance_agreement = (actual_one * predicted_one + (1 - actual_one) * (1 - predicted_one))
    return (observed_agreement - chance_agreement) / (1 - chance_agreement)

point = weighted_kappa(np.arange(len(h)))
# Exactly the same stratum/PSU grouping and RNG draw order as the original Phase E13 bootstrap.
groups = h.groupby(['strata', 'survey_cluster_id']).indices
by_stratum = {}
for (stratum, cluster), indices in groups.items():
    by_stratum.setdefault(stratum, []).append(np.asarray(indices))
rng = np.random.default_rng(991)
records = []
for rep in range(500):
    indices = np.concatenate([
        np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
        for clusters in by_stratum.values()
    ])
    records.append({
        'replicate': rep + 1,
        'weighted_cohen_kappa': weighted_kappa(indices),
        'weighted_auc_check': roc_auc_score(Y[indices], P[indices], sample_weight=W[indices]),
    })
boot = pd.DataFrame(records)
assert np.allclose(boot.weighted_auc_check.to_numpy(), saved_boot.weighted_auc.to_numpy(), atol=1e-9, rtol=1e-9), 'Original bootstrap sequence mismatch'
lo, hi = np.quantile(boot.weighted_cohen_kappa, [0.025, 0.975])
print(f'Survey-weighted Cohen kappa: {point:.4f} (95% percentile CI: {lo:.4f}–{hi:.4f})')
print('Verified original 500 stratified-PSU bootstrap replicates against saved ROC-AUC sequence.')
boot[['replicate', 'weighted_cohen_kappa']].to_csv(ROOT / 'Reviewer13_Comment12_Cohen_Kappa_Bootstrap500.csv', index=False)
pd.DataFrame([{'metric': 'survey_weighted_cohen_kappa', 'estimate': point, 'ci_lower': lo, 'ci_upper': hi, 'bootstrap_replicates': 500}]).to_csv(ROOT / 'Reviewer13_Comment12_Cohen_Kappa_Result.csv', index=False)
