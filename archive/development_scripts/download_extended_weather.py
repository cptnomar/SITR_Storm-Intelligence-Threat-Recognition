
import pandas as pd
import requests
import time

from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

DATA_FOLDER = Path(__file__).resolve().parent / "data"
DATA_FOLDER.mkdir(exist_ok=True)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

START_YEAR = 2015
END_YEAR = 2025

# Reference locations for five Houston-area counties
LOCATIONS = {
    "Harris (Houston)": (29.7604, -95.3698),
    "Fort Bend (Richmond)": (29.5822, -95.7608),
    "Montgomery (Conroe)": (30.3119, -95.4561),
    "Brazoria (Angleton)": (29.1694, -95.4319),
    "Galveston (Galveston)": (29.3013, -94.7977)
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "soil_moisture_0_to_7cm"
]


# ==========================================
# DOWNLOAD FUNCTION
# ==========================================

def download_weather(location, latitude, longitude, year):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(HOURLY_VARIABLES),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Chicago",
        "timeformat": "unixtime",
        "models": "era5"
    }

    # Retry temporary API/network failures
    for attempt in range(3):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            if "hourly" not in data:
                raise ValueError(
                    f"Missing hourly data: {data}"
                )

            df = pd.DataFrame(data["hourly"])

            # Convert Unix timestamps to Houston local time
            df["time"] = (
                pd.to_datetime(df["time"], unit="s", utc=True)
                .dt.tz_convert("America/Chicago")
                .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            )

            df["location"] = location
            df["latitude"] = latitude
            df["longitude"] = longitude

            return df

        except (requests.RequestException, ValueError) as error:
            print(f"  Attempt {attempt + 1} failed: {error}")

            if attempt == 2:
                raise

            time.sleep(5 * (attempt + 1))


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("=== SITR HISTORICAL WEATHER DOWNLOAD ===")

    all_files = []

    for location, (latitude, longitude) in LOCATIONS.items():

        print(f"\nLocation: {location}")

        safe_name = location.split(" (")[0].replace(" ", "_")

        for year in range(START_YEAR, END_YEAR + 1):

            filename = (
                DATA_FOLDER
                / f"extended_{safe_name}_{year}.csv"
            )

            # Avoid downloading completed files again
            if filename.exists() and filename.stat().st_size > 0:
                print(f"  {year}: Already downloaded")
                all_files.append(filename)
                continue

            print(f"  Downloading {year}...")

            try:
                df = download_weather(
                    location,
                    latitude,
                    longitude,
                    year
                )

                df.to_csv(filename, index=False)

                all_files.append(filename)

                print(f"  Saved {len(df)} rows")

                # Be considerate of API usage
                time.sleep(2)

            except Exception as error:
                print(f"  ERROR for {location}, {year}: {error}")

    # ======================================
    # COMBINE ALL DOWNLOADED FILES
    # ======================================

    if not all_files:
        print("\nNo data was downloaded.")
        return

    print("\nCombining downloaded files...")

    combined = pd.concat(
        [pd.read_csv(file) for file in all_files],
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=["location", "time"]
    )

    combined = combined.sort_values(
        ["location", "time"]
    )

    output_file = DATA_FOLDER / "extended_weather.csv"

    combined.to_csv(output_file, index=False)

    print("\n=== DOWNLOAD COMPLETE ===")
    print("Total records:", len(combined))
    print("Locations:", combined["location"].nunique())
    print("Output file:", output_file)

    print("\nMissing values:")
    print(combined.isnull().sum())


if __name__ == "__main__":
    main()
