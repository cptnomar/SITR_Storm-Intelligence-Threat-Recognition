
import io
import re
import time
import requests
import pandas as pd

from pathlib import Path
from urllib.parse import urljoin

# ==========================================
# CONFIGURATION
# ==========================================

DATA_FOLDER = Path(__file__).resolve().parent / "data"
DATA_FOLDER.mkdir(exist_ok=True)

BASE_URL = (
    "https://www.ncei.noaa.gov/pub/data/"
    "swdi/stormevents/csvfiles/"
)

START_YEAR = 2015
END_YEAR = 2025

COUNTIES = {
    "HARRIS",
    "FORT BEND",
    "MONTGOMERY",
    "BRAZORIA",
    "GALVESTON"
}

EVENT_TYPES = {
    "Flood",
    "Flash Flood"
}

session = requests.Session()
session.headers.update({
    "User-Agent": "SITR-research-project/1.0"
})


# ==========================================
# FIND AVAILABLE NOAA FILES
# ==========================================

def find_files():

    print("Checking NOAA file directory...")

    response = session.get(BASE_URL, timeout=60)
    response.raise_for_status()

    filenames = re.findall(
        r'StormEvents_details-ftp_v1\.0_d\d{4}_c\d{8}\.csv\.gz',
        response.text
    )

    selected = {}

    for filename in set(filenames):

        match = re.search(
            r"_d(\d{4})_c(\d{8})",
            filename
        )

        if not match:
            continue

        year = int(match.group(1))
        created = match.group(2)

        if START_YEAR <= year <= END_YEAR:

            # Select latest available revision
            if (
                year not in selected
                or created > selected[year][0]
            ):
                selected[year] = (created, filename)

    return {
        year: filename
        for year, (_, filename) in selected.items()
    }


# ==========================================
# DOWNLOAD AND FILTER ONE YEAR
# ==========================================

def download_year(year, filename):

    url = urljoin(BASE_URL, filename)

    print(f"\nDownloading {year}...")

    response = session.get(url, timeout=180)
    response.raise_for_status()

    # Pandas can read gzip-compressed CSV data
    df = pd.read_csv(
        io.BytesIO(response.content),
        compression="gzip",
        low_memory=False
    )

    print(f"  Nationwide events: {len(df)}")

    # Normalize names before filtering
    df["STATE"] = (
        df["STATE"].fillna("").astype(str).str.upper().str.strip()
    )

    df["CZ_NAME"] = (
        df["CZ_NAME"].fillna("").astype(str).str.upper().str.strip()
    )

    df["EVENT_TYPE"] = (
        df["EVENT_TYPE"].fillna("").astype(str).str.strip()
    )

    # Restrict to our five Houston-area counties
    # CZ_TYPE C indicates a county-based event
    filtered = df[
        (df["STATE"] == "TEXAS")
        & (df["CZ_TYPE"] == "C")
        & (df["CZ_NAME"].isin(COUNTIES))
        & (df["EVENT_TYPE"].isin(EVENT_TYPES))
    ].copy()

    filtered["source_year"] = year

    print(f"  Houston-area flood events: {len(filtered)}")

    return filtered


# ==========================================
# MAIN
# ==========================================

def main():

    print("=== SITR FLOOD EVENT DOWNLOAD ===")

    available = find_files()
    all_data = []
    
    print("\n=== CHECKING 2024 NOAA FILE ===")

    if 2024 in available:
        print("2024 file found:", available[2024])
    else:
        print("WARNING: 2024 file was not found!")

    for year in range(START_YEAR, END_YEAR + 1):

        if year not in available:
            print(f"WARNING: No NOAA file found for {year}")
            continue

        cache_file = (
            DATA_FOLDER / f"noaa_flood_events_{year}.csv"
        )

        if cache_file.exists():
            print(f"\n{year}: Loading existing file")
            df = pd.read_csv(cache_file, low_memory=False)

        else:
            try:
                df = download_year(
                    year,
                    available[year]
                )

                df.to_csv(cache_file, index=False)

                time.sleep(2)

            except Exception as error:
                print(f"ERROR downloading {year}: {error}")
                continue

        all_data.append(df)

    if not all_data:
        print("No flood data collected.")
        return

    # ======================================
    # COMBINE DATA
    # ======================================

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=["EVENT_ID"]
    )

    # Parse dates without inventing missing times
    combined["BEGIN_DATE"] = pd.to_datetime(
        combined["BEGIN_YEARMONTH"].astype(str).str[:4]
        + "-"
        + combined["BEGIN_YEARMONTH"].astype(str).str[4:6]
        + "-01",
        errors="coerce"
    )

    # BEGIN_DATE_TIME contains the actual start
    # date and time supplied by NOAA ## delete from here
    combined["event_start"] = pd.to_datetime(
    combined["BEGIN_DATE_TIME"],
    format="%d-%b-%y %H:%M:%S",
    errors="coerce"
)

    combined["event_end"] = pd.to_datetime(
    combined["END_DATE_TIME"],
    format="%d-%b-%y %H:%M:%S",
    errors="coerce"
)

    combined["event_date"] = (
        combined["event_start"].dt.strftime("%Y-%m-%d")
    )

    output = DATA_FOLDER / "houston_noaa_flood_events.csv"

    combined.to_csv(output, index=False)

    print("\n=== DOWNLOAD COMPLETE ===")
    print("Total flood events:", len(combined))
    print("Counties:", combined["CZ_NAME"].nunique())
    print("Output file:", output)

    print("\nEvents by county:")
    print(combined["CZ_NAME"].value_counts())

    print("\nEvents by type:")
    print(combined["EVENT_TYPE"].value_counts())

    print("\nEvents by year:")
    print(combined["source_year"].value_counts().sort_index())


if __name__ == "__main__":
    main()
