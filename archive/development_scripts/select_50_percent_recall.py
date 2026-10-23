"""SITR: select a >=50% NOAA-report recall threshold on 2021-2022 validation.

Run from the Flood project folder: python select_50_percent_recall.py
Requires the existing train_model_extended.py and its saved test_predictions.csv.
This evaluates NOAA report matching, not verified flood occurrence or warning safety.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from evaluate_model_extended import new_forest, prepare

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / 'data' / 'extended_model_results' / 'test_predictions.csv'
OUT = ROOT / 'data' / 'extended_model_results' / 'recall_target'
TARGET_RECALL = 0.50


def score(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': round(float(threshold), 6),
        'recall': float(tp / (tp + fn)) if tp + fn else 0.0,
        'precision': float(tp / (tp + fp)) if tp + fp else 0.0,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'alerts': int(tp + fp),
    }


def main():
    print('=== SITR 50% NOAA-REPORT RECALL TARGET ===')
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f'Missing {PREDICTIONS}; run train_model_extended.py first.')
    df, features = prepare()
    fit = df[df['target_date'] <= pd.Timestamp('2020-12-31')]
    validation = df[df['target_date'].between('2021-01-01', '2022-12-31')]
    if fit.empty or validation.empty or fit['next_day_noaa_report'].nunique() < 2:
        raise ValueError('Insufficient fitting or validation data.')
    model = new_forest()
    model.fit(fit[features], fit['next_day_noaa_report'])
    y_val = validation['next_day_noaa_report'].to_numpy(dtype=int)
    p_val = model.predict_proba(validation[features])[:, list(model.classes_).index(1)]
    if y_val.sum() == 0:
        raise ValueError('No NOAA-reported days in validation period.')

    # Candidate cutoffs include every unique predicted probability. Choosing the
    # highest cutoff reaching the target minimizes alerts on validation.
    candidates = np.unique(np.r_[0.0, p_val, 1.0])
    rows = [score(y_val, p_val, t) for t in candidates]
    eligible = [r for r in rows if r['recall'] >= TARGET_RECALL]
    if not eligible:
        raise RuntimeError('No threshold meets the validation recall target.')
    chosen = max(eligible, key=lambda r: r['threshold'])

    test = pd.read_csv(PREDICTIONS)
    required = {'target_date', 'next_day_noaa_report', 'forest_probability'}
    if missing := required - set(test.columns):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    test['target_date'] = pd.to_datetime(test['target_date'], errors='raise')
    if test['target_date'].min() < pd.Timestamp('2023-01-01'):
        raise ValueError('Test predictions contain pre-2023 dates.')
    y_test = test['next_day_noaa_report'].to_numpy(dtype=int)
    p_test = test['forest_probability'].to_numpy(dtype=float)
    if not np.isfinite(p_test).all() or ((p_test < 0) | (p_test > 1)).any():
        raise ValueError('Invalid test probabilities.')
    observed = score(y_test, p_test, chosen['threshold'])

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        'target_validation_recall': TARGET_RECALL,
        'selected_threshold': chosen['threshold'],
        'validation_period': '2021-2022',
        'validation_metrics': chosen,
        'retrospective_test_period': '2023-2025 (previously examined)',
        'retrospective_test_metrics': observed,
        'caution': '50% is a validation recall goal, not 50% accuracy or verified real-world flood detection. NOAA non-reports are not confirmed non-floods. Threshold calibration may differ after refitting.',
    }
    (OUT / 'recall_target_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame([chosen, observed], index=['validation', 'retrospective_test']).to_csv(OUT / 'recall_target_comparison.csv')
    print(f'Validation: {len(y_val)} county-days, {int(y_val.sum())} reported days')
    print(f'Selected threshold on validation: {chosen["threshold"]:.6f}')
    print(f'Validation: recall={chosen["recall"]:.1%}, precision={chosen["precision"]:.1%}, '
          f'detected={chosen["tp"]}/{chosen["tp"] + chosen["fn"]}, alerts={chosen["alerts"]}')
    print(f'Retrospective test: recall={observed["recall"]:.1%}, precision={observed["precision"]:.1%}, '
          f'detected={observed["tp"]}/{observed["tp"] + observed["fn"]}, alerts={observed["alerts"]}')
    print(f'Saved: {OUT}')
    print('CAUTION: This is NOAA report recall, NOT verified flood prediction accuracy.')
    print('CAUTION: The test period was previously examined; results are exploratory.')


if __name__ == '__main__':
    main()
