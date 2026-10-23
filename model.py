
import pandas as pd
from pathlib import Path

DATA_FOLDER = Path(__file__).resolve().parent / "data"

# Load historical weather data
weather = pd.read_csv(
    DATA_FOLDER / "houston_major_storms_hourly_weather.csv"
)

# Convert timestamps into datetime objects
weather["time"] = pd.to_datetime(weather["time"])

# Extract the date
weather["date"] = weather["time"].dt.date

# Group hourly observations by location and date
daily_weather = weather.groupby(
    ["location", "date"]
).agg(
    daily_rainfall=("precipitation", "sum"),
    max_hourly_rainfall=("precipitation", "max"),
    max_wind_gust=("wind_gusts_10m", "max"),
    avg_soil_moisture=("soil_moisture_0_to_7cm", "mean")
).reset_index()

# Match county names with their FIPS codes
county_fips = {
    "Harris": "48201",
    "Fort Bend": "48157",
    "Montgomery": "48339",
    "Brazoria": "48039",
    "Galveston": "48167",
    "Liberty": "48291",
    "Waller": "48473",
    "Chambers": "48071",
    "Austin": "48015"
}

# Extract county name from the location column
# Example: "Harris (Houston)" becomes "Harris"
daily_weather["county"] = (
    daily_weather["location"]
    .str.split(" (", regex=False)
    .str[0]
    .str.strip()
)

# Assign the appropriate FIPS code
daily_weather["county_fips"] = (
    daily_weather["county"].map(county_fips)
)

# Check for unmatched counties
unmatched = daily_weather[
    daily_weather["county_fips"].isna()
]

if not unmatched.empty:
    print("WARNING: Unmatched county names:")
    print(unmatched["location"].unique())
else:
    print("All counties matched successfully!")

# Display the results
print(
    daily_weather[
        ["location", "county", "county_fips"]
    ].drop_duplicates().to_string(index=False)
)


print("=== DAILY WEATHER DATA ===")
print(daily_weather.head(10).to_string(index=False))

print("\nNumber of daily records:", len(daily_weather))

# Save processed dataset
daily_weather.to_csv(
    DATA_FOLDER / "daily_weather.csv",
    index=False
)

print("\nDaily weather CSV created successfully!")


# ==========================================
# COMBINE WEATHER DATA WITH FEMA FLOOD CLAIMS
# ==========================================

# Find the FEMA claims CSV
fema_files = list(
    DATA_FOLDER.glob("houston_fema_flood_claims*.csv")
)

if not fema_files:
    raise FileNotFoundError("FEMA claims CSV not found!")

# Load FEMA claims
fema = pd.read_csv(
    fema_files[0],
    dtype={"countyCode": str},
    low_memory=False
)

# Convert FEMA loss dates into date objects
fema["date"] = (
    pd.to_datetime(fema["dateOfLoss"], utc=True)
    .dt.date
)

# Standardize county FIPS codes
fema["county_fips"] = (
    fema["countyCode"].str.zfill(5)
)

# Count reported FEMA claims by county and date
daily_claims = (
    fema.groupby(["county_fips", "date"])
    .size()
    .reset_index(name="claim_count")
)

print("\n=== DAILY FEMA CLAIMS ===")
print(daily_claims.head(10).to_string(index=False))

# Make sure weather FIPS codes are strings
daily_weather["county_fips"] = (
    daily_weather["county_fips"].astype(str).str.zfill(5)
)

# Combine daily weather with FEMA claim counts
training_data = daily_weather.merge(
    daily_claims,
    on=["county_fips", "date"],
    how="left",
    validate="one_to_one"
)

# Missing matches mean no claims were found in
# the supplied FEMA dataset for that county/date.
training_data["claim_count"] = (
    training_data["claim_count"].fillna(0).astype(int)
)

# Create a binary target
# 1 = at least one FEMA claim recorded
# 0 = no FEMA claim found in the dataset
training_data["claim_reported"] = (
    training_data["claim_count"] > 0
).astype(int)

# Display results
print("\n=== COMBINED TRAINING DATA ===")
print(training_data.head(10).to_string(index=False))

print("\nTotal records:", len(training_data))

print("\nClaim reported distribution:")
print(training_data["claim_reported"].value_counts())

# Save the combined dataset
training_data.to_csv(
    DATA_FOLDER / "training_data.csv",
    index=False
)

print("\nTraining data CSV created successfully!")
