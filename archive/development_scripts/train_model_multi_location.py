"""SITR: controlled Random Forest comparison with multi-location weather.

Run: python train_model_multi_location.py
Requires: pandas, numpy, scikit-learn, joblib
Inputs: data/extended_training_data_multi_location.csv
Outputs: data/multi_location_model_results/

Predicts NEXT-DAY NOAA flood REPORTS from current/prior weather. No report
is not proof of no flood. Exploratory research, NOT a flood warning system.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'data' / 'extended_training_data_multi_location.csv'
OUT = ROOT / 'data' / 'multi_location_model_results'
BASE_FEATURES = [
    'daily_rainfall_in', 'max_hourly_rainfall_in', 'max_wind_gust_mph',
    'avg_soil_moisture', 'rain_24h_prior_in', 'rain_48h_prior_in',
    'rain_72h_prior_in', 'max_gust_prior_day_mph', 'soil_moisture_prior_day',
]
NEW_FEATURES = [
    'ml_avg_daily_rainfall_in', 'ml_max_daily_rainfall_in',
    'ml_min_daily_rainfall_in', 'ml_max_hourly_rainfall_in',
    'ml_max_wind_gust_mph', 'ml_avg_soil_moisture',
    'ml_rainfall_spread_in',
]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def make_model():
    # Same hyperparameters as the original Random Forest experiment.
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )


def metrics(y, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold), 'precision': float(precision),
        'recall': float(recall), 'f1': float(f1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'alerts': int(pred.sum()),
        'average_precision': float(average_precision_score(y, probabilities)),
        'roc_auc': float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
    }


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(f'Missing {INPUT}. Run aggregate_multi_location_weather.py first.')
    df = pd.read_csv(INPUT, dtype={'county': 'string'})
    required = set(BASE_FEATURES + NEW_FEATURES + [
        'date', 'county', 'noaa_event_reported', 'ml_location_count'
    ])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    if df.duplicated(['county', 'date']).any():
        raise ValueError('Duplicate county/date records found.')
    if not df['noaa_event_reported'].isin([0, 1]).all():
        raise ValueError('Expected NOAA labels to be 0 or 1.')
    if not df['ml_location_count'].eq(3).all():
        raise ValueError('Expected three weather locations for every county-day.')
    df = df.sort_values(['county', 'date']).copy()
    for feature in BASE_FEATURES + NEW_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    # Only the weather from day t (or earlier) is used to predict day t+1.
    df['target_date'] = df.groupby('county')['date'].shift(-1)
    df['target'] = df.groupby('county')['noaa_event_reported'].shift(-1)
    df = df.loc[df['target_date'].eq(df['date'] + pd.Timedelta(days=1))].copy()
    before = len(df)
    df = df.dropna(subset=BASE_FEATURES + NEW_FEATURES).copy()
    print('Rows excluded due to missing weather features:', before - len(df))
    df['target'] = df['target'].astype(int)
    county_columns = pd.get_dummies(df['county'], prefix='county', dtype=int)
    df = pd.concat([df, county_columns], axis=1)
    county_features = sorted(county_columns.columns)
    return df, {
        'Original (one location)': BASE_FEATURES + county_features,
        'Expanded (three locations)': BASE_FEATURES + NEW_FEATURES + county_features,
    }


def main():
    print('=== SITR MULTI-LOCATION MODEL COMPARISON ===')
    df, feature_sets = load_data()
    fit = df[df['target_date'] <= '2020-12-31']
    validation = df[(df['target_date'] >= '2021-01-01') & (df['target_date'] <= '2022-12-31')]
    train = df[df['target_date'] <= '2022-12-31']
    test = df[(df['target_date'] >= '2023-01-01') & (df['target_date'] <= '2025-12-31')]
    for label, frame in [('Fit', fit), ('Validation', validation), ('Refit', train), ('Test', test)]:
        print(f'{label}: {len(frame)} county-days; {int(frame["target"].sum())} NOAA-reported days')
        if frame.empty or frame['target'].nunique() != 2:
            raise ValueError(f'{label} must have examples of both classes.')
    OUT.mkdir(parents=True, exist_ok=True)
    validation_results = []
    test_results = []
    predictions = test[['county', 'date', 'target_date', 'target']].copy()
    feature_importances = []
    for name, features in feature_sets.items():
        print(f'\nTraining {name} ({len(features)} features)...')
        model = make_model()
        model.fit(fit[features], fit['target'])
        val_p = model.predict_proba(validation[features])[:, list(model.classes_).index(1)]
        candidates = [{'model': name, **metrics(validation['target'].to_numpy(), val_p, t)}
                      for t in THRESHOLDS]
        validation_results.extend(candidates)
        # Threshold chosen using validation only, never test performance.
        chosen = sorted(candidates, key=lambda r: (-r['f1'], -r['precision'], r['alerts']))[0]
        threshold = chosen['threshold']
        print(f'  Validation threshold: {threshold:.2f}; precision={chosen["precision"]:.4f}, '
              f'recall={chosen["recall"]:.4f}, F1={chosen["f1"]:.4f}, alerts={chosen["alerts"]}')
        model = make_model()
        model.fit(train[features], train['target'])
        test_p = model.predict_proba(test[features])[:, list(model.classes_).index(1)]
        result = {'model': name, **metrics(test['target'].to_numpy(), test_p, threshold)}
        test_results.append(result)
        slug = 'original' if name.startswith('Original') else 'expanded'
        predictions[f'{slug}_probability'] = test_p
        predictions[f'{slug}_alert'] = (test_p >= threshold).astype(int)
        joblib.dump({
            'model': model, 'features': features, 'threshold': threshold,
            'target': 'next-day NOAA report', 'training_end': '2022-12-31',
            'model_name': name,
        }, OUT / f'{slug}_random_forest.joblib')
        feature_importances.extend({'model': name, 'feature': feature, 'importance': float(importance)}
                                   for feature, importance in zip(features, model.feature_importances_))

    pd.DataFrame(validation_results).to_csv(OUT / 'validation_thresholds.csv', index=False)
    pd.DataFrame(test_results).to_csv(OUT / 'test_comparison.csv', index=False)
    predictions.to_csv(OUT / 'test_predictions.csv', index=False)
    pd.DataFrame(feature_importances).sort_values(
        ['model', 'importance'], ascending=[True, False]
    ).to_csv(OUT / 'feature_importances.csv', index=False)
    with (OUT / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump({'test_results': test_results,
                   'note': 'Exploratory comparison: 2023-2025 results have been examined previously.'}, f, indent=2)
    print('\n2023-2025 retrospective comparison (thresholds selected on validation):')
    columns = ['model', 'threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn',
               'alerts', 'average_precision', 'roc_auc']
    print(pd.DataFrame(test_results)[columns].to_string(
        index=False, float_format=lambda x: f'{x:.4f}'
    ))
    print(f'\nSaved results to: {OUT}')
    print('IMPORTANT: NOAA no-report days are not verified flood-free days.')
    print('This is exploratory research, NOT a public flood warning system.')
    print('The 2023-2025 test period was previously examined; improvements are not independently validated.')


if __name__ == '__main__':
    main()
