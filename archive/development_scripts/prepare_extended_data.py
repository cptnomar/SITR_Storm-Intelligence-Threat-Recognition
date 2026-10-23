"""Prepare county-day historical features and observed reports for SITR.

Run: python prepare_extended_data.py
Requires: pandas
Input files (in ./data): extended_weather.csv,
    houston_noaa_flood_events.csv, houston_fema_flood_claims_2017_present.csv
Output: ./data/extended_training_data.csv

This is a descriptive research table, NOT a validated flood-risk model.
Unreported NOAA days are unknown, not confirmed flood-free.
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent / 'data'
WEATHER = DATA / 'extended_weather.csv'
NOAA = DATA / 'houston_noaa_flood_events.csv'
FEMA = DATA / 'houston_fema_flood_claims_2017_present.csv'
OUTPUT = DATA / 'extended_training_data.csv'
START_DATE = pd.Timestamp('2015-01-01')
END_DATE = pd.Timestamp('2025-12-31')

COUNTY_FIPS = {
    'HARRIS': '48201', 'FORT BEND': '48157',
    'MONTGOMERY': '48339', 'BRAZORIA': '48039',
    'GALVESTON': '48167',
}
COUNTIES = set(COUNTY_FIPS)


def dates(series, label):
    """Parse date-like values, warning on failures instead of hiding them."""
    raw = series.astype('string').str.strip()
    # For NOAA, datetime values are local and already represent event-local dates.
    # Parsing the date component avoids timezone/day-boundary shifts.
    parsed = pd.to_datetime(raw.str.slice(0, 10), errors='coerce')
    bad = (parsed.isna() & raw.notna() & raw.ne('')).sum()
    if bad:
        print(f'WARNING: {label}: {bad} dates could not be parsed')
    return parsed.dt.normalize()


def weather_table():
    print('Reading hourly weather... [diagnostic version 2]')
    w = pd.read_csv(WEATHER, low_memory=False)
    needed = {'time', 'location', 'precipitation', 'wind_gusts_10m', 'soil_moisture_0_to_7cm'}
    missing = needed - set(w.columns)
    if missing:
        raise ValueError(f'Weather CSV missing columns: {sorted(missing)}')
    # Open-Meteo local ISO timestamps start YYYY-MM-DD. Using their date prefix
    # preserves the local day across daylight saving time transitions.
    print('  Raw weather rows:', len(w))
    print('  Location examples:', w['location'].dropna().astype(str).unique()[:8].tolist())
    print('  Timestamp examples:', w['time'].dropna().astype(str).head(3).tolist())
    w['date'] = dates(w['time'], 'weather time')
    # Extract county names without relying on regex capture alignment.
    w['county'] = (w['location'].astype('string').str.split('(', n=1).str[0]
                   .str.strip().str.upper())
    print('  Parsed county examples:', w['county'].dropna().unique()[:8].tolist())
    print('  Valid dates:', int(w['date'].notna().sum()))
    w = w[w['county'].isin(COUNTIES) & w['date'].notna()].copy()
    print('  Weather rows after filtering:', len(w))
    if w.empty:
        raise ValueError('No weather rows matched the five counties')
    for col in ['precipitation', 'wind_gusts_10m', 'soil_moisture_0_to_7cm']:
        w[col] = pd.to_numeric(w[col], errors='coerce')
    daily = (w.groupby(['county', 'date'], as_index=False)
             .agg(daily_rainfall_in=('precipitation', 'sum'),
                  max_hourly_rainfall_in=('precipitation', 'max'),
                  max_wind_gust_mph=('wind_gusts_10m', 'max'),
                  avg_soil_moisture=('soil_moisture_0_to_7cm', 'mean'),
                  weather_hours=('time', 'size')))
    daily = daily.sort_values(['county', 'date']).reset_index(drop=True)
    # An extra 2014-12-31 hour can appear due to the API time boundary.
    # Incomplete days cannot be treated as complete 24-hour rainfall totals.
    print('  Hourly coverage distribution:', daily['weather_hours'].value_counts().sort_index().to_dict())
    print('  Partial days (<23 hours):', int((daily['weather_hours'] < 23).sum()))
    # Reindex each county to a complete daily calendar so rolling windows don't
    # silently jump over missing days. A rolling sum requires full coverage.
    complete = []
    for county, group in daily.groupby('county'):
        group = group.set_index('date')
        group = group.reindex(pd.date_range(group.index.min(), group.index.max(), freq='D'))
        group.index.name = 'date'
        group['county'] = county
        # Allow 23/25-hour local days at DST changes, but reject partial days.
        # This prevents the single 2014-12-31 hour becoming a 24-hour feature.
        valid_rain = group['daily_rainfall_in'].where(group['weather_hours'].between(23, 25))
        group['rain_24h_prior_in'] = valid_rain.shift(1)
        group['rain_48h_prior_in'] = valid_rain.shift(1).rolling(2, min_periods=2).sum()
        group['rain_72h_prior_in'] = valid_rain.shift(1).rolling(3, min_periods=3).sum()
        group['max_gust_prior_day_mph'] = group['max_wind_gust_mph'].where(group['weather_hours'].between(23, 25)).shift(1)
        group['soil_moisture_prior_day'] = group['avg_soil_moisture'].where(group['weather_hours'].between(23, 25)).shift(1)
        complete.append(group.reset_index())
    result = pd.concat(complete, ignore_index=True)
    result = result[result['date'].between(START_DATE, END_DATE)].copy()
    print('  County-day rows (2015-2025):', len(result))
    print('  Days with nonstandard hourly coverage:', int((~result['weather_hours'].between(23, 25)).sum()))
    return result


def noaa_table():
    print('Reading NOAA flood events...')
    n = pd.read_csv(NOAA, low_memory=False)
    required = {'CZ_NAME', 'BEGIN_DATE_TIME', 'END_DATE_TIME', 'EVENT_ID'}
    missing = required - set(n.columns)
    if missing:
        raise ValueError(f'NOAA CSV missing columns: {sorted(missing)}')
    n['county'] = n['CZ_NAME'].astype('string').str.upper().str.strip()
    n = n[n['county'].isin(COUNTIES)].copy()
    # NOAA timestamps often look like "08/26/2017 19:00:00". Support that and
    # common alternate formats explicitly, then flag anything still unparsed.
    def parse_noaa(s):
        result = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
        for fmt in ('%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M',
                    '%d-%b-%y %H:%M:%S', '%d-%b-%y %H:%M',
                    '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
            candidate = pd.to_datetime(s, format=fmt, errors='coerce')
            result = result.fillna(candidate)
        return result
    n['start'] = parse_noaa(n['BEGIN_DATE_TIME'].astype('string'))
    n['end'] = parse_noaa(n['END_DATE_TIME'].astype('string'))
    bad = (n['start'].isna() | n['end'].isna() | (n['end'] < n['start']))
    if bad.any():
        print(f'WARNING: {bad.sum()} NOAA events have invalid dates; excluded from date matching')
    print('  NOAA input events:', len(n))
    print('  NOAA valid events:', int((~bad).sum()))
    n = n[~bad].drop_duplicates('EVENT_ID')
    rows = []
    for event in n.itertuples(index=False):
        # Events can cross midnight: count each day they overlap.
        days = pd.date_range(event.start.normalize(), event.end.normalize(), freq='D')
        for day in days:
            rows.append((event.county, day, event.EVENT_ID))
    if not rows:
        raise ValueError('No NOAA event dates parsed; inspect BEGIN_DATE_TIME examples')
    events = pd.DataFrame(rows, columns=['county', 'date', 'event_id'])
    events = (events.groupby(['county', 'date'], as_index=False)
              .agg(noaa_event_count=('event_id', 'nunique')))
    print('  NOAA distinct reported county-days:', len(events))
    return events


def fema_table():
    print('Reading FEMA insurance claims...')
    f = pd.read_csv(FEMA, dtype={'countyCode': 'string'}, low_memory=False)
    needed = {'dateOfLoss', 'countyCode'}
    missing = needed - set(f.columns)
    if missing:
        raise ValueError(f'FEMA CSV missing columns: {sorted(missing)}')
    f['date'] = dates(f['dateOfLoss'], 'FEMA dateOfLoss')
    f['fips'] = f['countyCode'].astype('string').str.replace(r'\.0$', '', regex=True).str.zfill(5)
    f['county'] = f['fips'].map({v: k for k, v in COUNTY_FIPS.items()})
    f = f[f['county'].notna() & f['date'].notna()]
    return (f.groupby(['county', 'date'], as_index=False)
            .size().rename(columns={'size': 'fema_claim_count'}))


def main():
    for path in (WEATHER, NOAA, FEMA):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input file: {path}')
    daily = weather_table()
    if daily.empty:
        raise ValueError('Weather table is empty. Stopping instead of writing an empty output CSV.')
    events = noaa_table()
    claims = fema_table()
    result = daily.merge(events, on=['county', 'date'], how='left')
    result = result.merge(claims, on=['county', 'date'], how='left')
    result['noaa_event_count'] = result['noaa_event_count'].fillna(0).astype(int)
    result['fema_claim_count'] = result['fema_claim_count'].fillna(0).astype(int)
    result['noaa_event_reported'] = (result['noaa_event_count'] > 0).astype(int)
    # 0 means "no NOAA event in our file", NOT verified no flooding.
    result['county_fips'] = result['county'].map(COUNTY_FIPS)
    if result.empty:
        raise ValueError('Merged result is empty. Output file was not overwritten.')
    expected = len(pd.date_range(START_DATE, END_DATE)) * len(COUNTIES)
    if len(result) != expected or result.duplicated(['county', 'date']).any():
        raise ValueError(f'Expected {expected} unique county-days, got {len(result)}. Output not overwritten.')
    print('  Dataset dates:', result['date'].min().date(), 'to', result['date'].max().date())
    print('  County-days with incomplete coverage:', int((~result['weather_hours'].between(23, 25)).sum()))
    print('  Missing prior-24h rainfall:', int(result['rain_24h_prior_in'].isna().sum()))
    print('  Missing prior-72h rainfall:', int(result['rain_72h_prior_in'].isna().sum()))
    result.to_csv(OUTPUT, index=False)
    print('\n=== SITR DATA PREPARATION COMPLETE ===')
    print('County-days:', len(result))
    print('NOAA-reported county-days:', int(result['noaa_event_reported'].sum()))
    print('FEMA claims matched:', int(result['fema_claim_count'].sum()))
    print('Missing prior-48h rainfall:', int(result['rain_48h_prior_in'].isna().sum()))
    print('Saved:', OUTPUT)
    print('NOTE: Do not use noaa_event_reported=0 as a verified negative label.')
    print('NOTE: For advance prediction, use *_prior_* features, not same-day weather.')


if __name__ == '__main__':
    main()
