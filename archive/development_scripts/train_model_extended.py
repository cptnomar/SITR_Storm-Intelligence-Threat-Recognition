"""SITR: next-day NOAA flood-report prediction experiment.

Run: python train_model_extended.py
Requires: pandas, scikit-learn, joblib (installed with scikit-learn).
Input: data/extended_training_data.csv

IMPORTANT: A missing NOAA report is NOT proof of no flooding. Metrics below
measure agreement with recorded NOAA reports, not true flood detection skill.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "extended_training_data.csv"
OUTPUT = ROOT / "data" / "extended_model_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

# At the END of date t, today's weather is known. Predict a NOAA report on t+1.
# Never include same-day NOAA reports, FEMA claims, or future weather as inputs.
WEATHER_FEATURES = [
    "daily_rainfall_in",
    "max_hourly_rainfall_in",
    "max_wind_gust_mph",
    "avg_soil_moisture",
    "rain_24h_prior_in",
    "rain_48h_prior_in",
    "rain_72h_prior_in",
    "max_gust_prior_day_mph",
    "soil_moisture_prior_day",
]
TRAIN_LAST_DATE = "2022-12-31"
TEST_FIRST_DATE = "2023-01-01"
DECISION_THRESHOLD = 0.50


def prepare():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing {INPUT}. Run prepare_extended_data.py first.")
    df = pd.read_csv(INPUT, dtype={"county": "string"})
    required = {"date", "county", "noaa_event_reported", *WEATHER_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df.duplicated(["county", "date"]).any():
        raise ValueError("Duplicate county/date rows: fix preparation before training.")
    if df["noaa_event_reported"].isna().any() or not df["noaa_event_reported"].isin([0, 1]).all():
        raise ValueError("noaa_event_reported must contain only 0 or 1.")

    df = df.sort_values(["county", "date"]).copy()
    for feature in WEATHER_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    # Shift the label *within each county*: row t predicts reported event on t+1.
    df["target_date"] = df.groupby("county")["date"].shift(-1)
    df["next_day_noaa_report"] = df.groupby("county")["noaa_event_reported"].shift(-1)
    # Exclude county end-of-series and gaps; never call a distant date "tomorrow".
    next_day = df["target_date"].eq(df["date"] + pd.Timedelta(days=1))
    df = df.loc[next_day].copy()
    df["next_day_noaa_report"] = df["next_day_noaa_report"].astype(int)

    # Drop incomplete feature-history days rather than imputing unknown past as zero.
    before = len(df)
    df = df.dropna(subset=WEATHER_FEATURES).copy()
    print(f"Rows excluded due to missing weather history: {before - len(df)}")

    # County one-hot encoding is explicit so future inference can use same columns.
    county_dummies = pd.get_dummies(df["county"], prefix="county", dtype=int)
    df = pd.concat([df, county_dummies], axis=1)
    feature_cols = WEATHER_FEATURES + sorted(county_dummies.columns.tolist())
    return df, feature_cols


def report(name, y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    print(f"\n--- {name} (threshold={threshold:.2f}) ---")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print("Report prevalence:", f"{np.mean(y_true):.4f}")
    if len(np.unique(y_true)) == 2:
        print(f"Average precision (PR-AUC summary): {average_precision_score(y_true, probabilities):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_true, probabilities):.4f}")
    else:
        print("PR-AUC and ROC-AUC undefined: test labels have only one class.")
    print(classification_report(y_true, predicted, labels=[0, 1], zero_division=0))
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "confusion_matrix": matrix.tolist(),
        "average_precision": float(average_precision_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2 else None,
    }


def main():
    print("=== SITR EXTENDED MODEL TRAINING ===")
    df, features = prepare()
    train = df[df["target_date"] <= pd.Timestamp(TRAIN_LAST_DATE)].copy()
    test = df[df["target_date"] >= pd.Timestamp(TEST_FIRST_DATE)].copy()
    if train.empty or test.empty:
        raise ValueError("Empty training or test period. Check date coverage.")
    if train["next_day_noaa_report"].nunique() < 2:
        raise ValueError("Training data has only one target class.")

    X_train, y_train = train[features], train["next_day_noaa_report"]
    X_test, y_test = test[features], test["next_day_noaa_report"]
    print(f"Train: {len(train)} rows, {int(y_train.sum())} reported county-days")
    print(f"Test:  {len(test)} rows, {int(y_test.sum())} reported county-days")
    print(f"Train target dates: {train['target_date'].min().date()} to {train['target_date'].max().date()}")
    print(f"Test target dates:  {test['target_date'].min().date()} to {test['target_date'].max().date()}")
    print("Weather and county inputs:", ", ".join(features))

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)
    baseline_prob = baseline.predict_proba(X_test)[:, list(baseline.classes_).index(1)]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    model_prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

    baseline_metrics = report("Always-prior-probability baseline", y_test, baseline_prob, DECISION_THRESHOLD)
    forest_metrics = report("Random Forest", y_test, model_prob, DECISION_THRESHOLD)

    importance = pd.Series(model.named_steps["forest"].feature_importances_, index=features)
    print("\nTop feature importances (associations, not causation):")
    print(importance.sort_values(ascending=False).head(10).round(4).to_string())

    predictions = test[["county", "date", "target_date", "next_day_noaa_report"]].copy()
    predictions["baseline_probability"] = baseline_prob
    predictions["forest_probability"] = model_prob
    predictions["forest_predicted_report"] = (model_prob >= DECISION_THRESHOLD).astype(int)
    predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)
    joblib.dump({"model": model, "features": features, "threshold": DECISION_THRESHOLD,
                 "prediction_target": "NOAA report in county on next calendar day"},
                OUTPUT / "random_forest.joblib")
    with (OUTPUT / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"train_rows": len(train), "test_rows": len(test),
                   "train_reported_days": int(y_train.sum()), "test_reported_days": int(y_test.sum()),
                   "baseline": baseline_metrics, "random_forest": forest_metrics,
                   "threshold": DECISION_THRESHOLD}, f, indent=2)
    print(f"\nSaved model, predictions and metrics to: {OUTPUT}")
    print("CAUTION: Unreported NOAA days are unknown, NOT verified flood-free days.")
    print("This is an exploratory report-prediction experiment, not a safety warning system.")


if __name__ == "__main__":
    main()
