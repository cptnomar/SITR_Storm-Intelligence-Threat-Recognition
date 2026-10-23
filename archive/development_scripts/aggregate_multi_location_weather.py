"""SITR: aggregate 10 additional ERA5 weather points into county-day features.

Run: python aggregate_multi_location_weather.py
Requires: pandas
Reads data/multi_location_weather/*.csv and data/extended_training_data.csv.
Writes NEW files only; does not overwrite original training data or models.
Weather features are calculated for observation day D, suitable for a next-day
(D+1) target, not for predicting floods earlier on day D.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FOLDER = DATA / 'multi_location_weather'
BASE = DATA / 'extended_training_data.csv'
DAILY_OUT = DATA / 'multi_location_county_day_features.csv'
MERGED_OUT = DATA / 'extended_training_data_multi_location.csv'
EXPECTED = {
    'HARRIS': {'katy', 'baytown'},
    'FORT BEND': {'sugar_land', 'rosenberg'},
    'MONTGOMERY': {'the_woodlands', 'montgomery'},
    'BRAZORIA': {'pearland', 'lake_jackson'},
    'GALVESTON': {'league_city', 'texas_city'},
}
MM_TO_IN = 1 / 25.4
KMH_TO_MPH = 0.6213711922


def main():
    if not BASE.is_file():
        raise FileNotFoundError(f'Missing {BASE}. Run prepare_extended_data.py first.')
    if not FOLDER.is_dir():
        raise FileNotFoundError(f'Missing {FOLDER}. Run download_multi_location_weather.py first.')
    base = pd.read_csv(BASE, low_memory=False)
    required_base = {'date', 'county', 'daily_rainfall_in', 'max_hourly_rainfall_in',
                     'max_wind_gust_mph', 'avg_soil_moisture', 'weather_hours'}
    if not required_base.issubset(base.columns):
        raise ValueError(f'Base training data missing columns: {sorted(required_base - set(base.columns))}')
    base['date'] = pd.to_datetime(base['date'], errors='raise').dt.normalize()
    base['county'] = base['county'].astype(str).str.upper().str.strip()
    if base.duplicated(['county', 'date']).any():
        raise ValueError('Base training data has duplicate county/date pairs.')
    dates = pd.date_range('2015-01-01', '2025-12-31', freq='D')
    expected_files = [(county, loc, year, FOLDER / f'{county.lower().replace(" ", "_")}__{loc}__{year}.csv')
                      for county, locations in EXPECTED.items() for loc in sorted(locations)
                      for year in range(2015, 2026)]
    missing_files = [str(path) for _, _, _, path in expected_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f'Missing {len(missing_files)} location-year files, e.g. {missing_files[:5]}')
    pieces = []
    print('=== SITR MULTI-LOCATION COUNTY-DAY AGGREGATION ===', flush=True)
    for index, (county, loc, year, path) in enumerate(expected_files, 1):
        frame = pd.read_csv(path, usecols=['time', 'precipitation', 'wind_gusts_10m',
                                            'soil_moisture_0_to_7cm', 'county', 'location'])
        if frame.empty:
            raise ValueError(f'Empty weather file: {path}')
        if set(frame['county'].dropna().astype(str).str.upper().str.strip()) != {county}:
            raise ValueError(f'County mismatch: {path}')
        # Keep local calendar dates: offset-bearing timestamps include DST transitions.
        frame['date'] = pd.to_datetime(frame['time'].astype(str).str[:10], errors='raise')
        if frame['date'].dt.year.ne(year).any():
            raise ValueError(f'Unexpected calendar year in {path}')
        if frame['time'].duplicated().any():
            raise ValueError(f'Duplicate hourly timestamps in {path}')
        for c in ['precipitation', 'wind_gusts_10m', 'soil_moisture_0_to_7cm']:
            frame[c] = pd.to_numeric(frame[c], errors='coerce')
        if frame['precipitation'].isna().any():
            raise ValueError(f'Missing hourly precipitation in {path}; refusing to treat as zero')
        if (frame['precipitation'] < 0).any():
            raise ValueError(f'Negative precipitation in {path}')
        daily = (frame.groupby('date', as_index=False)
                 .agg(rainfall_mm=('precipitation', 'sum'),
                      max_hourly_mm=('precipitation', 'max'),
                      max_gust_kmh=('wind_gusts_10m', 'max'),
                      avg_soil_moisture=('soil_moisture_0_to_7cm', 'mean'),
                      hours=('time', 'size')))
        year_days = dates[dates.year == year]
        if len(daily) != len(year_days) or not daily['date'].equals(pd.Series(year_days)):
            raise ValueError(f'Incomplete or unsorted daily coverage in {path}')
        # Expected hours for each local date (DST 23/25h). Calculate from successive localized midnights.
        start = pd.DatetimeIndex(year_days).tz_localize('America/Chicago')
        end = (pd.DatetimeIndex(year_days) + pd.Timedelta(days=1)).tz_localize('America/Chicago')
        expected_hours = (end - start).total_seconds() / 3600
        if not (daily['hours'].to_numpy() == expected_hours).all():
            raise ValueError(f'Hourly coverage mismatch in {path}')
        daily['county'] = county
        daily['location_key'] = loc
        daily['rainfall_in'] = daily['rainfall_mm'] * MM_TO_IN
        daily['max_hourly_in'] = daily['max_hourly_mm'] * MM_TO_IN
        daily['max_gust_mph'] = daily['max_gust_kmh'] * KMH_TO_MPH
        pieces.append(daily[['county', 'date', 'location_key', 'rainfall_in',
                             'max_hourly_in', 'max_gust_mph', 'avg_soil_moisture']])
        if index % 22 == 0:
            print(f'Processed {index}/{len(expected_files)} location-year files', flush=True)
    additional = pd.concat(pieces, ignore_index=True)
    # Use the existing representative point as the third weather location.
    original = base[['county', 'date', 'daily_rainfall_in', 'max_hourly_rainfall_in',
                     'max_wind_gust_mph', 'avg_soil_moisture']].copy()
    original = original.rename(columns={'daily_rainfall_in': 'rainfall_in',
                                         'max_hourly_rainfall_in': 'max_hourly_in',
                                         'max_wind_gust_mph': 'max_gust_mph'})
    original['location_key'] = 'original'
    all_points = pd.concat([additional, original], ignore_index=True)
    if all_points.duplicated(['county', 'date', 'location_key']).any():
        raise ValueError('Duplicate county/date/location observations')
    daily_features = (all_points.groupby(['county', 'date'], as_index=False)
                      .agg(ml_location_count=('location_key', 'nunique'),
                           ml_avg_daily_rainfall_in=('rainfall_in', 'mean'),
                           ml_max_daily_rainfall_in=('rainfall_in', 'max'),
                           ml_min_daily_rainfall_in=('rainfall_in', 'min'),
                           ml_max_hourly_rainfall_in=('max_hourly_in', 'max'),
                           ml_max_wind_gust_mph=('max_gust_mph', 'max'),
                           ml_avg_soil_moisture=('avg_soil_moisture', 'mean')))
    daily_features['ml_rainfall_spread_in'] = (daily_features['ml_max_daily_rainfall_in']
                                               - daily_features['ml_min_daily_rainfall_in'])
    merged = base.merge(daily_features, on=['county', 'date'], how='left', validate='one_to_one')
    if len(merged) != len(base) or merged['ml_location_count'].ne(3).any():
        raise ValueError('Expected exactly three weather locations for every base county-day.')
    if merged[list(daily_features.columns.difference(['county', 'date']))].isna().any().any():
        raise ValueError('Missing aggregated weather values; refusing to save incomplete dataset.')
    # Only lagged weather can be used for a next-day prediction at end of current day.
    # This file contains same-day observations D; the training script must shift the
    # flood-report label to D+1 (as in existing SITR training scripts).
    daily_features.to_csv(DAILY_OUT, index=False, date_format='%Y-%m-%d')
    merged.to_csv(MERGED_OUT, index=False, date_format='%Y-%m-%d')
    print(f'Base county-days: {len(base):,}')
    print(f'Additional location-day records: {len(additional):,}')
    print(f'Locations per county-day: {int(merged["ml_location_count"].min())}')
    print(f'NOAA-reported days unchanged: {int(merged["noaa_event_reported"].sum()):,}' if 'noaa_event_reported' in merged else '')
    print(f'Saved county-day features: {DAILY_OUT}')
    print(f'Saved enriched training data: {MERGED_OUT}')
    print('Existing extended_training_data.csv and trained models were NOT changed.')
    print('IMPORTANT: The new features have not been added to model training yet.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
