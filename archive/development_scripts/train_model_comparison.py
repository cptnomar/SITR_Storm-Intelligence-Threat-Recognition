"""SITR: chronological comparison of next-day NOAA report prediction models.

Run: python train_model_comparison.py
Requires: pandas, numpy, scikit-learn, joblib
Input: data/extended_training_data.csv
Output: data/model_comparison/

IMPORTANT: No NOAA report does not prove no flooding. This is an exploratory
report-prediction study, NOT a public flood warning system.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'data' / 'extended_training_data.csv'
OUT = ROOT / 'data' / 'model_comparison'
FEATURES = [
    'daily_rainfall_in', 'max_hourly_rainfall_in', 'max_wind_gust_mph',
    'avg_soil_moisture', 'rain_24h_prior_in', 'rain_48h_prior_in',
    'rain_72h_prior_in', 'max_gust_prior_day_mph', 'soil_moisture_prior_day',
]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(f'Missing {INPUT}; run prepare_extended_data.py first.')
    df = pd.read_csv(INPUT, dtype={'county': 'string'})
    missing = set(FEATURES + ['date', 'county', 'noaa_event_reported']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    if df.duplicated(['county', 'date']).any():
        raise ValueError('Duplicate county-date rows.')
    if not df['noaa_event_reported'].isin([0, 1]).all():
        raise ValueError('NOAA report label must be 0 or 1.')
    df = df.sort_values(['county', 'date']).copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Weather observed on day t predicts reports on day t+1, in same county.
    df['target_date'] = df.groupby('county')['date'].shift(-1)
    df['target'] = df.groupby('county')['noaa_event_reported'].shift(-1)
    df = df.loc[df['target_date'].eq(df['date'] + pd.Timedelta(days=1))].copy()
    before = len(df)
    df = df.dropna(subset=FEATURES).copy()
    print('Excluded missing-history rows:', before - len(df))
    df['target'] = df['target'].astype(int)
    dummies = pd.get_dummies(df['county'], prefix='county', dtype=int)
    df = pd.concat([df, dummies], axis=1)
    return df, FEATURES + sorted(dummies.columns)


def models():
    return {
        'Logistic Regression': make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=25,
            l2_regularization=1.0, random_state=42
        ),
    }


def scores(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=float(threshold), precision=float(precision), recall=float(recall),
                f1=float(f1), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                alerts=int(pred.sum()), average_precision=float(average_precision_score(y, prob)),
                roc_auc=float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None)


def main():
    print('=== SITR CHRONOLOGICAL MODEL COMPARISON ===')
    df, features = load_data()
    fit = df[df['target_date'] <= '2020-12-31']
    val = df[(df['target_date'] >= '2021-01-01') & (df['target_date'] <= '2022-12-31')]
    full_train = df[df['target_date'] <= '2022-12-31']
    test = df[(df['target_date'] >= '2023-01-01') & (df['target_date'] <= '2025-12-31')]
    for name, part in [('Fit 2015-2020', fit), ('Validation 2021-2022', val),
                       ('Refit 2015-2022', full_train), ('Test 2023-2025', test)]:
        print(f'{name}: {len(part)} rows; {int(part["target"].sum())} NOAA-reported days')
        if part.empty or part['target'].nunique() != 2:
            raise ValueError(f'{name} needs rows from both report classes.')
    X_fit, y_fit = fit[features], fit['target'].to_numpy()
    X_val, y_val = val[features], val['target'].to_numpy()
    X_full, y_full = full_train[features], full_train['target'].to_numpy()
    X_test, y_test = test[features], test['target'].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)

    validation_rows = []
    candidates = {}
    for name, model in models().items():
        model.fit(X_fit, y_fit)
        p = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
        for threshold in THRESHOLDS:
            validation_rows.append({'model': name, **scores(y_val, p, threshold)})
        candidates[name] = model
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / 'validation_thresholds.csv', index=False)

    # Select on validation ONLY; prefer fewer alerts when F1 ties.
    chosen = validation.sort_values(
        ['f1', 'precision', 'alerts'], ascending=[False, False, True]
    ).iloc[0]
    winner = str(chosen['model'])
    threshold = float(chosen['threshold'])
    print(f'\nValidation-selected model: {winner}; threshold={threshold:.2f}')
    print(f'Validation precision={chosen["precision"]:.3f}, recall={chosen["recall"]:.3f}, '
          f'F1={chosen["f1"]:.3f}, alerts={int(chosen["alerts"])}')

    # Fit all models again on all pre-test years; do NOT choose using test metrics.
    prediction_table = test[['county', 'date', 'target_date', 'target']].copy()
    test_results = []
    for name, model in models().items():
        model.fit(X_full, y_full)
        p = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        prediction_table[name.lower().replace(' ', '_') + '_probability'] = p
        # Only the winner uses the chosen decision threshold.
        metric = scores(y_test, p, threshold)
        test_results.append({'model': name, **metric})
        if name == winner:
            joblib.dump({'model': model, 'features': features, 'threshold': threshold,
                         'target': 'next-day NOAA report', 'training_end': '2022-12-31'},
                        OUT / 'selected_model.joblib')
    baseline = DummyClassifier(strategy='prior').fit(X_full, y_full)
    base_p = baseline.predict_proba(X_test)[:, list(baseline.classes_).index(1)]
    prediction_table['baseline_probability'] = base_p
    prediction_table.to_csv(OUT / 'test_predictions.csv', index=False)
    results = pd.DataFrame(test_results)
    results.to_csv(OUT / 'test_model_comparison.csv', index=False)
    print('\nTest results (descriptive; selection was on validation):')
    print(results[['model', 'threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn',
                   'alerts', 'average_precision', 'roc_auc']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nTest baseline average precision: {average_precision_score(y_test, base_p):.4f}')
    with (OUT / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump({'selected_model': winner, 'selected_threshold': threshold,
                   'validation': {k: (v.item() if hasattr(v, 'item') else v)
                                  for k, v in chosen.to_dict().items()},
                   'test_results': test_results,
                   'test_baseline_average_precision': float(average_precision_score(y_test, base_p))},
                  file, indent=2)
    print(f'\nFiles saved to: {OUT}')
    print('CAUTION: No NOAA report is not verified flood-free. This is NOT a warning system.')
    print('Threshold calibration can change after refitting; treat comparisons as exploratory.')


if __name__ == '__main__':
    main()
