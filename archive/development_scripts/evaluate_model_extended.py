"""SITR: validation-selected thresholds and held-out test evaluation.

Run: python evaluate_model_extended.py
Requires: pandas, numpy, scikit-learn, matplotlib.
Requires train_model_extended.py and its saved test_predictions.csv.

NOAA missing reports are not verified non-flood days. This experiment measures
agreement with NOAA reports, NOT verified flood prediction or public safety.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from train_model_extended import prepare

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'data' / 'extended_model_results'
PREDICTIONS = RESULTS / 'test_predictions.csv'
OUT = RESULTS / 'evaluation'
OUT.mkdir(parents=True, exist_ok=True)

# The test set (2023-2025) must NOT be used to select a threshold.
# Select using 2021-2022, after fitting on 2015-2020.
VALIDATION_START = '2021-01-01'
VALIDATION_END = '2022-12-31'
TRAIN_END = '2020-12-31'
TEST_START = '2023-01-01'
THRESHOLDS = np.round(np.arange(0.01, 0.51, 0.01), 2)
DISPLAY_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def metrics_at(y, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average='binary', zero_division=0
    )
    return dict(threshold=float(threshold), precision=float(precision),
                recall=float(recall), f1=float(f1), tp=int(tp), fp=int(fp),
                fn=int(fn), tn=int(tn), alerts=int(tp + fp))


def new_forest():
    # Must match the original train_model_extended.py hyperparameters.
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('forest', RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )),
    ])


def main():
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f'Missing {PREDICTIONS}. Run train_model_extended.py first.')
    print('=== SITR EXTENDED MODEL EVALUATION ===')
    df, features = prepare()
    fit = df[df['target_date'] <= pd.Timestamp(TRAIN_END)]
    valid = df[df['target_date'].between(VALIDATION_START, VALIDATION_END)]
    if fit.empty or valid.empty:
        raise ValueError('Training or validation period is empty.')
    if fit['next_day_noaa_report'].nunique() < 2 or valid['next_day_noaa_report'].nunique() < 2:
        raise ValueError('Training/validation needs both report and no-report observations.')

    print(f'Validation fit: {len(fit)} county-days through {TRAIN_END}')
    print(f'Validation: {len(valid)} county-days, {int(valid["next_day_noaa_report"].sum())} reported days')
    model = new_forest()
    model.fit(fit[features], fit['next_day_noaa_report'])
    val_prob = model.predict_proba(valid[features])[:, list(model.classes_).index(1)]
    val_y = valid['next_day_noaa_report'].to_numpy(dtype=int)
    val_rows = [metrics_at(val_y, val_prob, t) for t in THRESHOLDS]
    val_table = pd.DataFrame(val_rows)
    val_table.to_csv(OUT / 'validation_thresholds.csv', index=False)

    # F1 on reported events is a diagnostic, not a guarantee of actual flood skill.
    # On ties, choose the higher threshold (fewer alerts).
    best = sorted(val_rows, key=lambda r: (-r['f1'], -r['threshold']))[0]
    chosen = best['threshold']
    print(f'Validation-selected threshold: {chosen:.2f}')
    print(f'Validation precision={best["precision"]:.3f}, recall={best["recall"]:.3f}, '
          f'F1={best["f1"]:.3f}, alerts={best["alerts"]}')

    test = pd.read_csv(PREDICTIONS)
    needed = {'target_date', 'next_day_noaa_report', 'forest_probability', 'baseline_probability'}
    if missing := needed - set(test.columns):
        raise ValueError(f'Test predictions missing columns: {sorted(missing)}')
    test['target_date'] = pd.to_datetime(test['target_date'], errors='raise')
    if (test['target_date'] < pd.Timestamp(TEST_START)).any():
        raise ValueError('Test file includes dates before 2023; rerun train_model_extended.py.')
    if not test['next_day_noaa_report'].isin([0, 1]).all():
        raise ValueError('Invalid test labels.')
    y = test['next_day_noaa_report'].to_numpy(dtype=int)
    p = test['forest_probability'].to_numpy(dtype=float)
    base = test['baseline_probability'].to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid forest probabilities.')
    if len(np.unique(y)) != 2:
        raise ValueError('Test period has only one report class.')
    print(f'\nHeld-out test: {len(y)} county-days, {int(y.sum())} reported days')
    print(f'Random Forest average precision: {average_precision_score(y, p):.4f}')
    print(f'Constant baseline average precision: {average_precision_score(y, base):.4f}')
    print(f'Random Forest ROC-AUC: {roc_auc_score(y, p):.4f}')

    display = sorted(set(DISPLAY_THRESHOLDS + [chosen]))
    test_rows = [metrics_at(y, p, t) for t in display]
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(OUT / 'test_threshold_comparison.csv', index=False)
    print('\nTest thresholds (diagnostic comparison; NOT selected on test):')
    print(test_table[['threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'alerts']].to_string(index=False))
    print('\nTest results at VALIDATION-selected threshold:')
    chosen_test = metrics_at(y, p, chosen)
    print(json.dumps(chosen_test, indent=2))

    # Curve is descriptive; avoid using the test plot to retune the threshold.
    pr_precision, pr_recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(pr_recall, pr_precision, label='Random Forest')
    ax.axhline(y.mean(), linestyle='--', label=f'Report prevalence ({y.mean():.3%})')
    ax.set(xlabel='Recall of NOAA-reported days', ylabel='Precision against NOAA reports',
           title='SITR held-out test precision–recall (2023–2025)')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / 'test_precision_recall.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(val_table['threshold'], val_table['precision'], label='Validation precision')
    ax.plot(val_table['threshold'], val_table['recall'], label='Validation recall')
    ax.plot(val_table['threshold'], val_table['f1'], label='Validation F1')
    ax.axvline(chosen, linestyle='--', label=f'Selected {chosen:.2f}')
    ax.set(xlabel='Decision threshold', ylabel='Score',
           title='SITR threshold selection on 2021–2022 validation')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / 'validation_threshold_metrics.png', dpi=160)
    plt.close(fig)

    summary = {
        'threshold_selection': 'Maximize validation F1; ties favor higher threshold',
        'validation_fit_through': TRAIN_END,
        'validation_period': [VALIDATION_START, VALIDATION_END],
        'test_period': '2023-2025, untouched for threshold selection',
        'selected_threshold': chosen,
        'validation_at_selected_threshold': best,
        'test_at_selected_threshold': chosen_test,
        'test_average_precision': float(average_precision_score(y, p)),
        'test_baseline_average_precision': float(average_precision_score(y, base)),
        'test_roc_auc': float(roc_auc_score(y, p)),
        'warning': 'NOAA no-report days are not verified flood-free days; exploratory only.',
    }
    (OUT / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nSaved evaluation tables, charts, and summary to: {OUT}')
    print('IMPORTANT: Threshold was selected on a separately trained validation model.')
    print('Test probabilities come from the original model trained through 2022.')
    print('Probability calibration can differ; results are exploratory.')
    print('NOAA no-report days are not verified flood-free days. Not a warning system.')


if __name__ == '__main__':
    main()
