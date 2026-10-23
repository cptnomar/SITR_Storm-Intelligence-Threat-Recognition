"""SITR: diagnose missed NOAA-reported county-days on the held-out test period.

Run: python analyze_flood_predictions.py
Requires: pandas, numpy (no matplotlib needed)
Inputs:
    data/model_comparison/test_predictions.csv
    data/model_comparison/summary.json
    data/extended_training_data.csv
Outputs: data/model_comparison/analysis/

IMPORTANT: No NOAA report is NOT proof of no flooding. This is exploratory
analysis of reports, not a flood-warning system.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
PRED_PATH = DATA / 'model_comparison' / 'test_predictions.csv'
SUMMARY_PATH = DATA / 'model_comparison' / 'summary.json'
WEATHER_PATH = DATA / 'extended_training_data.csv'
OUT = DATA / 'model_comparison' / 'analysis'

PROBABILITY_COLUMNS = {
    'Logistic Regression': 'logistic_regression_probability',
    'Random Forest': 'random_forest_probability',
    'HistGradientBoosting': 'histgradientboosting_probability',
}


def load_inputs():
    for path in [PRED_PATH, SUMMARY_PATH, WEATHER_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}\nRun train_model_comparison.py and prepare_extended_data.py first.')

    pred = pd.read_csv(PRED_PATH)
    with SUMMARY_PATH.open(encoding='utf-8') as f:
        summary = json.load(f)
    weather = pd.read_csv(WEATHER_PATH)

    required_pred = {'county', 'date', 'target_date', 'target'} | set(PROBABILITY_COLUMNS.values())
    required_weather = {'county', 'date', 'daily_rainfall_in', 'max_hourly_rainfall_in',
                        'rain_24h_prior_in', 'rain_48h_prior_in', 'rain_72h_prior_in',
                        'max_wind_gust_mph', 'avg_soil_moisture', 'noaa_event_count'}
    for name, frame, required in [('predictions', pred, required_pred),
                                  ('weather', weather, required_weather)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{name} missing columns: {sorted(missing)}')

    for frame in (pred, weather):
        frame['county'] = frame['county'].astype('string').str.strip().str.upper()
        frame['date'] = pd.to_datetime(frame['date'], errors='raise').dt.normalize()
    pred['target_date'] = pd.to_datetime(pred['target_date'], errors='raise').dt.normalize()
    if pred.duplicated(['county', 'target_date']).any():
        raise ValueError('Duplicate predictions for county/target_date.')
    if weather.duplicated(['county', 'date']).any():
        raise ValueError('Duplicate weather county/date records.')
    if not pred['target'].isin([0, 1]).all():
        raise ValueError('Prediction target must contain only 0 or 1.')
    if not (pred['target_date'] == pred['date'] + pd.Timedelta(days=1)).all():
        raise ValueError('Some predictions do not target the following day.')
    if not pred['target_date'].between('2023-01-01', '2025-12-31').all():
        raise ValueError('Expected held-out test target dates from 2023 through 2025.')
    for col in PROBABILITY_COLUMNS.values():
        pred[col] = pd.to_numeric(pred[col], errors='raise')
        if pred[col].isna().any() or not pred[col].between(0, 1).all():
            raise ValueError(f'Invalid probabilities in {col}.')

    selected = summary.get('selected_model')
    if selected not in PROBABILITY_COLUMNS:
        raise ValueError(f'Unknown selected model in summary.json: {selected!r}')
    threshold = float(summary['selected_threshold'])
    if not 0 <= threshold <= 1:
        raise ValueError('Invalid selected threshold.')
    return pred, weather, selected, threshold


def main():
    print('=== SITR MISSED FLOOD-REPORT ANALYSIS ===')
    pred, weather, selected, threshold = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)

    # Weather on day t is known at prediction time; target day t+1 weather
    # is used only for retrospective diagnosis, never as model input.
    observed = weather[['county', 'date', 'daily_rainfall_in', 'max_hourly_rainfall_in',
                        'rain_24h_prior_in', 'rain_48h_prior_in', 'rain_72h_prior_in',
                        'max_wind_gust_mph', 'avg_soil_moisture']].copy()
    observed = observed.rename(columns={
        'daily_rainfall_in': 'observed_day_rainfall_in',
        'max_hourly_rainfall_in': 'observed_day_max_hourly_rainfall_in',
        'rain_24h_prior_in': 'observed_day_prior24h_rain_in',
        'rain_48h_prior_in': 'observed_day_prior48h_rain_in',
        'rain_72h_prior_in': 'observed_day_prior72h_rain_in',
        'max_wind_gust_mph': 'observed_day_max_wind_gust_mph',
        'avg_soil_moisture': 'observed_day_soil_moisture',
    })
    enriched = pred.merge(observed, on=['county', 'date'], how='left', validate='one_to_one')
    target_weather = weather[['county', 'date', 'daily_rainfall_in',
                              'max_hourly_rainfall_in', 'noaa_event_count']].rename(columns={
        'date': 'target_date',
        'daily_rainfall_in': 'target_day_rainfall_in_RETROSPECTIVE_ONLY',
        'max_hourly_rainfall_in': 'target_day_max_hourly_rainfall_in_RETROSPECTIVE_ONLY',
        'noaa_event_count': 'target_day_noaa_event_count',
    })
    enriched = enriched.merge(target_weather, on=['county', 'target_date'], how='left',
                              validate='one_to_one')
    if enriched['observed_day_rainfall_in'].isna().any() or enriched['target_day_noaa_event_count'].isna().any():
        raise ValueError('Could not match some predictions to weather; inspect county/date keys.')
    if not (enriched['target'].eq((enriched['target_day_noaa_event_count'] > 0).astype(int))).all():
        raise ValueError('Prediction labels disagree with the prepared NOAA event counts.')

    prob_col = PROBABILITY_COLUMNS[selected]
    enriched['selected_probability'] = enriched[prob_col]
    enriched['selected_alert'] = (enriched['selected_probability'] >= threshold).astype(int)
    enriched['outcome'] = np.select(
        [(enriched['target'] == 1) & (enriched['selected_alert'] == 1),
         (enriched['target'] == 1) & (enriched['selected_alert'] == 0),
         (enriched['target'] == 0) & (enriched['selected_alert'] == 1)],
        ['reported_detected', 'reported_missed', 'alert_without_noaa_report'],
        default='no_alert_no_noaa_report',
    )
    enriched['year'] = enriched['target_date'].dt.year
    enriched = enriched.sort_values(['target_date', 'county'])
    enriched.to_csv(OUT / 'all_test_predictions_diagnostic.csv', index=False)

    reported = enriched.loc[enriched['target'] == 1].copy()
    reported.to_csv(OUT / 'all_reported_county_days.csv', index=False)
    missed = reported.loc[reported['selected_alert'] == 0].copy()
    missed.to_csv(OUT / 'missed_reported_county_days.csv', index=False)
    alerts_without_report = enriched.loc[enriched['outcome'] == 'alert_without_noaa_report'].copy()
    alerts_without_report.to_csv(OUT / 'alerts_without_noaa_report.csv', index=False)

    county = enriched.groupby('county').agg(
        test_county_days=('target', 'size'),
        reported_days=('target', 'sum'),
        alerts=('selected_alert', 'sum'),
    )
    county['detected_reports'] = reported.groupby('county')['selected_alert'].sum().reindex(county.index, fill_value=0)
    county['missed_reports'] = county['reported_days'] - county['detected_reports']
    county['report_recall'] = county['detected_reports'] / county['reported_days'].replace(0, np.nan)
    county.to_csv(OUT / 'county_summary.csv')

    year = enriched.groupby('year').agg(
        test_county_days=('target', 'size'), reported_days=('target', 'sum'), alerts=('selected_alert', 'sum'))
    year['detected_reports'] = reported.groupby('year')['selected_alert'].sum().reindex(year.index, fill_value=0)
    year['missed_reports'] = year['reported_days'] - year['detected_reports']
    year['report_recall'] = year['detected_reports'] / year['reported_days'].replace(0, np.nan)
    year.to_csv(OUT / 'year_summary.csv')

    # Consecutive report-positive county-days are grouped into runs. These
    # runs are NOT guaranteed to correspond to individual physical storms.
    reported = reported.sort_values(['county', 'target_date']).copy()
    previous = reported.groupby('county')['target_date'].shift(1)
    starts_new = previous.isna() | ((reported['target_date'] - previous).dt.days != 1)
    reported['run_number'] = starts_new.groupby(reported['county']).cumsum()
    runs = reported.groupby(['county', 'run_number']).agg(
        start=('target_date', 'min'), end=('target_date', 'max'),
        reported_county_days=('target', 'size'),
        detected_county_days=('selected_alert', 'sum'),
        max_model_probability=('selected_probability', 'max'),
    ).reset_index()
    runs['any_day_detected'] = runs['detected_county_days'] > 0
    runs.to_csv(OUT / 'consecutive_report_runs.csv', index=False)

    summary = {
        'selected_model': selected, 'selected_threshold': threshold,
        'test_county_days': int(len(enriched)),
        'reported_county_days': int(len(reported)),
        'detected_reported_county_days': int(reported['selected_alert'].sum()),
        'missed_reported_county_days': int(len(missed)),
        'alerts_without_noaa_report': int(len(alerts_without_report)),
        'consecutive_report_runs': int(len(runs)),
        'runs_with_at_least_one_detection': int(runs['any_day_detected'].sum()),
    }
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'Selected model: {selected} | validation threshold: {threshold:.2f}')
    print(f'Test county-days: {len(enriched)}')
    print(f'NOAA-reported days: {len(reported)} | detected: {int(reported["selected_alert"].sum())} | missed: {len(missed)}')
    print(f'Alerts without matching NOAA report: {len(alerts_without_report)}')
    print('\nBy county:')
    print(county.to_string(float_format=lambda x: f'{x:.3f}'))
    print('\nBy year:')
    print(year.to_string(float_format=lambda x: f'{x:.3f}'))
    print(f'\nConsecutive NOAA-report runs: {len(runs)}; at least one day detected: {int(runs["any_day_detected"].sum())}')
    print('\n10 missed reported county-days with highest predicted probabilities:')
    display_cols = ['target_date', 'county', 'selected_probability',
                    'observed_day_rainfall_in', 'target_day_rainfall_in_RETROSPECTIVE_ONLY']
    print(missed.sort_values('selected_probability', ascending=False)[display_cols].head(10).to_string(index=False))
    print(f'\nSaved diagnostic files to: {OUT}')
    print('CAUTION: Target-day weather is retrospective context ONLY; never use it as a predictor.')
    print('CAUTION: Days without NOAA reports are NOT verified flood-free days.')


if __name__ == '__main__':
    main()
