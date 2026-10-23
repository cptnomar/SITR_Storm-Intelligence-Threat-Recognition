"""SITR: download additional historical hourly weather locations (2015-2025).

Run: python download_multi_location_weather.py
Requires: pip install requests pandas

This is a *separate* weather dataset; it does not overwrite extended_weather.csv.
Source: Open-Meteo Historical Weather API, ERA5 reanalysis (model=era5).
The weather is gridded/reanalysis data, not measurements at a rain gauge.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"
LOCAL_TZ = ZoneInfo("America/Chicago")
START_YEAR = 2015
END_YEAR = 2025
HOURLY = [
    "temperature_2m", "precipitation", "rain", "surface_pressure",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
    "wind_gusts_10m", "soil_moisture_0_to_7cm",
]
# Two *additional* locations per county; the existing extended_weather.csv
# already contains one representative location per county.
LOCATIONS = [
    ("HARRIS", "Katy", 29.7858, -95.8245),
    ("HARRIS", "Baytown", 29.7355, -94.9774),
    ("FORT BEND", "Sugar Land", 29.6197, -95.6349),
    ("FORT BEND", "Rosenberg", 29.5572, -95.8086),
    ("MONTGOMERY", "The Woodlands", 30.1658, -95.4613),
    ("MONTGOMERY", "Montgomery", 30.3883, -95.6963),
    ("BRAZORIA", "Pearland", 29.5636, -95.2860),
    ("BRAZORIA", "Lake Jackson", 29.0339, -95.4344),
    ("GALVESTON", "League City", 29.5075, -95.0949),
    ("GALVESTON", "Texas City", 29.3838, -94.9027),
]
COLUMNS = ["time", *HOURLY, "location", "county", "latitude", "longitude"]


def slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def download_one(session: requests.Session, county: str, name: str,
                 lat: float, lon: float, year: int, output: Path) -> int:
    # Query UTC and convert to America/Chicago so DST 23/25-hour days are correct.
    # Extra boundary dates cover all hours belonging to the local calendar year.
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{year - 1}-12-31",
        "end_date": f"{year + 1}-01-02",
        "hourly": ",".join(HOURLY),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "models": "era5",
    }
    last_error = None
    for attempt in range(1, 6):
        try:
            response = session.get(API_URL, params=params, timeout=120)
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:250]}")
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(payload.get("reason", "API error"))
            hourly = payload.get("hourly", {})
            missing = [key for key in ["time", *HOURLY] if key not in hourly]
            if missing:
                raise ValueError(f"Missing API fields: {missing}")
            frame = pd.DataFrame({key: hourly[key] for key in ["time", *HOURLY]})
            # UTC source timestamps are unambiguous, including DST transitions.
            utc = pd.to_datetime(frame["time"], utc=True, errors="raise")
            local = utc.dt.tz_convert("America/Chicago")
            frame = frame.loc[local.dt.year == year].copy()
            local = local.loc[frame.index]
            # Offset-bearing ISO strings match the project's historical format.
            frame["time"] = local.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            frame["location"] = name
            frame["county"] = county
            frame["latitude"] = lat
            frame["longitude"] = lon
            frame = frame[COLUMNS]
            if not len(frame) or frame["time"].duplicated().any():
                raise ValueError("Empty or duplicated local-time weather data")
            # Check every local hour in the year is present (DST handled).
            expected = (pd.Timestamp(f"{year + 1}-01-01", tz="America/Chicago") -
                        pd.Timestamp(f"{year}-01-01", tz="America/Chicago")) / pd.Timedelta(hours=1)
            if len(frame) != int(expected):
                raise ValueError(f"Expected {int(expected)} hours, got {len(frame)}")
            tmp = output.with_suffix(".tmp")
            frame.to_csv(tmp, index=False)
            tmp.replace(output)
            return len(frame)
        except (requests.RequestException, ValueError, RuntimeError, KeyError) as exc:
            last_error = exc
            if attempt < 5:
                delay = min(60, 3 * (2 ** (attempt - 1)))
                print(f"    Attempt {attempt} failed: {exc}; retrying in {delay}s", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"Failed {county}/{name}/{year}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--force", action="store_true", help="Redownload existing yearly files")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("start-year must be <= end-year")
    root = Path(__file__).resolve().parent
    outdir = root / "data" / "multi_location_weather"
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(county, name, lat, lon, year)
            for county, name, lat, lon in LOCATIONS
            for year in range(args.start_year, args.end_year + 1)]
    print("=== SITR MULTI-LOCATION HISTORICAL WEATHER DOWNLOAD ===")
    print(f"{len(LOCATIONS)} additional locations, {args.start_year}-{args.end_year}; {len(jobs)} location-years")
    print("Source: Open-Meteo ERA5; units: mm, km/h, Celsius; times: America/Chicago")
    print("Existing data/extended_weather.csv will NOT be modified.\n")
    failed = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "SITR historical flood research / educational project"})
        for i, (county, name, lat, lon, year) in enumerate(jobs, start=1):
            output = outdir / f"{slug(county)}__{slug(name)}__{year}.csv"
            if output.exists() and output.stat().st_size > 100 and not args.force:
                print(f"[{i}/{len(jobs)}] SKIP {county} / {name} / {year}", flush=True)
                continue
            print(f"[{i}/{len(jobs)}] GET  {county} / {name} / {year}", flush=True)
            try:
                n = download_one(session, county, name, lat, lon, year, output)
                print(f"    Saved {n:,} hourly rows -> {output.name}", flush=True)
            except Exception as exc:
                failed.append((county, name, year, str(exc)))
                print(f"    ERROR: {exc}", flush=True)
            time.sleep(0.8)
    # A manifest documents location coverage without combining large CSV files.
    manifest = outdir / "locations.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["county", "location", "latitude", "longitude"])
        writer.writerows(LOCATIONS)
    print(f"\nDone. Data directory: {outdir}")
    print(f"Location manifest: {manifest}")
    print(f"Failures: {len(failed)}")
    for county, name, year, error in failed:
        print(f"  {county} / {name} / {year}: {error}")
    print("Next step: aggregate these new hourly files into county-day features.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
