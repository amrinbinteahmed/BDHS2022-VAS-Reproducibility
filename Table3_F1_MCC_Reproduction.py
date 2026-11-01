"""Reproduce survey-weighted F1 and MCC for all 8 models in Table 3.
Input: PhaseE89_complete.zip (original nested-CV out-of-fold predictions).
No model refitting. Output: Table3_F1_MCC_Foldwise.csv and Table3_F1_MCC_Results.csv.
"""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import pandas as pd

ZIP_PATH = Path('PhaseE89_complete.zip')
MODELS = ['XGBoost', 'LightGBM', 'Random Forest', 'AdaBoost',
          'SVM', 'Decision Tree', 'ANN', 'Logistic Regression']


def weighted_metrics(df):
    y = df['observed'].to_numpy(dtype=int)
    pred = df['prediction'].to_numpy(dtype=int)
    w = df['weight'].to_numpy(dtype=float)
    assert len(df) > 0 and np.isfinite(w).all() and (w >= 0).all()
    assert set(np.unique(y)).issubset({0, 1})
    assert set(np.unique(pred)).issubset({0, 1})
    tp = w[(y == 1) & (pred == 1)].sum()
    tn = w[(y == 0) & (pred == 0)].sum()
    fp = w[(y == 0) & (pred == 1)].sum()
    fn = w[(y == 1) & (pred == 0)].sum()

    def safe_div(num, den):
        return float(num / den) if den > 0 else np.nan

    f1_receipt = safe_div(2 * tp, 2 * tp + fp + fn)
    f1_nonreceipt = safe_div(2 * tn, 2 * tn + fp + fn)
    macro_f1 = (f1_receipt + f1_nonreceipt) / 2
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(tp * tn - fp * fn, mcc_denom)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    balacc = (sensitivity + specificity) / 2
    return {'f1_receipt': f1_receipt, 'f1_nonreceipt': f1_nonreceipt,
            'macro_f1': macro_f1, 'mcc': mcc, 'balacc': balacc}


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f'Place {ZIP_PATH.name} in the working directory.')
    fold_rows = []
    with ZipFile(ZIP_PATH) as archive:
        for model in MODELS:
            for fold in (1, 2, 3):
                member = f'outer{fold}_{model.replace(" ", "_")}_predictions.csv'
                df = pd.read_csv(BytesIO(archive.read(member)))
                needed = {'source_KR_row', 'observed', 'prediction', 'weight'}
                assert needed.issubset(df.columns), (member, needed - set(df.columns))
                assert not df['source_KR_row'].duplicated().any(), member
                fold_rows.append({'model': model, 'fold': fold, 'n': len(df),
                                  **weighted_metrics(df)})

    foldwise = pd.DataFrame(fold_rows)
    assert foldwise.groupby('model')['n'].sum().eq(6092).all()
    summary = (foldwise.groupby('model', sort=False)
               [['f1_receipt', 'f1_nonreceipt', 'macro_f1', 'mcc', 'balacc']]
               .mean().reset_index())
    foldwise.to_csv('Table3_F1_MCC_Foldwise.csv', index=False)
    summary.to_csv('Table3_F1_MCC_Results.csv', index=False)
    print('\nTABLE 3 — THREE-OUTER-FOLD MEAN SURVEY-WEIGHTED METRICS')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nSaved: Table3_F1_MCC_Foldwise.csv')
    print('Saved: Table3_F1_MCC_Results.csv')


if __name__ == '__main__':
    main()
