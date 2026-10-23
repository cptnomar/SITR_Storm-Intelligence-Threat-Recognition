
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Locate our training dataset
DATA_FOLDER = Path(__file__).resolve().parent / "data"

df = pd.read_csv(DATA_FOLDER / "training_data.csv")

# Features the model will learn from
features = [
    "daily_rainfall",
    "max_hourly_rainfall",
    "max_wind_gust",
    "avg_soil_moisture"
]

# What we want to predict
target = "claim_reported"

# Verify required columns exist
required = features + [target, "date"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Separate inputs and outputs
X = df[features]
y = df[target]

# Convert dates into datetime objects
df["date"] = pd.to_datetime(df["date"])

# Define storm windows
# These are approximate dates for the four storms
storm_windows = {
    "Harvey_2017": ("2017-08-23", "2017-09-03"),
    "Imelda_2019": ("2019-09-16", "2019-09-22"),
    "Nicholas_2021": ("2021-09-12", "2021-09-17"),
    "Beryl_2024": ("2024-07-05", "2024-07-13")
}

# Identify which storm each observation belongs to
df["storm_group"] = "Unknown"

for storm, (start, end) in storm_windows.items():
    mask = df["date"].between(start, end)
    df.loc[mask, "storm_group"] = storm

print("=== STORM DISTRIBUTION ===")
print(df["storm_group"].value_counts())

# Hold out Beryl for testing
train_mask = df["storm_group"] != "Beryl_2024"
test_mask = df["storm_group"] == "Beryl_2024"

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

if X_train.empty or X_test.empty:
    raise ValueError("Training or testing dataset is empty!")

# Create Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)

# Predict on the unseen storm
predictions = model.predict(X_test)

print("\n=== MODEL RESULTS ===")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)

# Show which weather features the model used most
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance.to_string())

 #print!!!
print("\n=== UNKNOWN STORM RECORDS ===")

unknown = df[df["storm_group"] == "Unknown"]

print(
    unknown[
        ["date", "location", "daily_rainfall", "claim_reported"]
    ].to_string(index=False)
)

# ==========================================
# LEAVE-ONE-STORM-OUT CROSS-VALIDATION
# ==========================================

print("\n=== CROSS-VALIDATION RESULTS ===")

results = []

for storm in df["storm_group"].unique():

    # Separate training and testing records
    train = df[df["storm_group"] != storm]
    test = df[df["storm_group"] == storm]

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # Ensure the training data contains both classes
    if y_train.nunique() < 2:
        print(f"Skipping {storm}: only one training class")
        continue

    # Train a fresh model for each storm
    cv_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )

    cv_model.fit(X_train, y_train)

    # Predict on the excluded storm
    predictions = cv_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Baseline: always predict the majority
    # class found in the training data
    majority_class = y_train.mode()[0]
    baseline_predictions = [
        majority_class
    ] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )

    results.append({
        "Storm": storm,
        "Test Records": len(test),
        "Accuracy": round(accuracy, 3),
        "Baseline Accuracy": round(baseline_accuracy, 3)
    })

# Display results
results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

print("\nAverage model accuracy:",
      results_df["Accuracy"].mean())

print("Average baseline accuracy:",
      results_df["Baseline Accuracy"].mean())
