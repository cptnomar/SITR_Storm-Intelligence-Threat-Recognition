
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
file = DATA / "extended_training_data.csv"

df = pd.read_csv(file)

print("=== SITR DATA VALIDATION ===")

print("\nTotal rows:", len(df))
print("Columns:", df.columns.tolist())

print("\nFirst 5 records:")
print(df.head().to_string(index=False))

print("\n=== DATE CHECK ===")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("Earliest date:", df["date"].min())
print("Latest date:", df["date"].max())
print("Invalid dates:", df["date"].isna().sum())

print("\n=== COUNTY CHECK ===")

print(df["county"].value_counts())

print("\n=== DUPLICATE CHECK ===")

duplicates = df.duplicated(
    subset=["county", "date"]
).sum()

print("Duplicate county-days:", duplicates)

print("\n=== NOAA FLOOD LABEL CHECK ===")

print(df["noaa_event_reported"].value_counts(dropna=False))

print("\n=== MISSING VALUES ===")

print(df.isna().sum().to_string())

print("\n=== RAINFALL FEATURE CHECK ===")

rain_columns = [
    col for col in df.columns
    if "rain" in col.lower()
]

print(df[rain_columns].describe().to_string())

print("\n=== VALIDATION COMPLETE ===")
