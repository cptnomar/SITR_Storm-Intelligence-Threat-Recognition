# SITR — Storm Intelligence & Threat Recognition

**Historical flood exploration and experimental AI flood-report analysis for Harris County, Texas.**

*Project slogan: “Singing in the rain” — Jinsoul from LOONA*

SITR is a Python/Streamlit dashboard combining historical weather observations, NOAA flood-event records, a geographic rainfall visualization, and a retrospective machine-learning analysis of recent weather. It was developed as an educational research and hackathon project.

> **Important:** SITR is **not** an emergency warning system. Its AI scores are **not calibrated probabilities of flooding**, and a low score does not mean an area is safe from flooding. Always consult the [National Weather Service](https://www.weather.gov/) and local emergency authorities for official alerts.

## Features

### Historical Flood Map

- Explore historical Harris County weather by date, with previous/next-day controls and automatic playback.
- View three **illustrative** geographic rainfall regions representing sampled weather near **Houston, Katy, and Baytown**.
- Compare daily rainfall intensity using color-coded regions.
- See regions turn **purple** when the dataset records a NOAA flood event in Harris County on that date.
- Inspect historical next-day NOAA flood-report model scores when saved predictions are available.

**Map limitation:** The three regions are illustrative partitions based on weather sampling locations, **not official floodplain boundaries**. A county-wide NOAA flood report does **not** establish that every region flooded.

### Live Houston Weather & Previous Seven Days

- Display today's Houston forecast rainfall, precipitation chance, and high/low temperatures.
- Retrieve recent weather from Open-Meteo and analyze the **previous seven days, excluding today**.
- Show experimental AI flood-report scores and the weather inputs behind them when sufficient data is available.
- Label results according to whether inputs were retrieved or partially estimated.
- Show active flood-related National Weather Service alerts for Harris County separately from the AI model.

**Note:** The seven-day analysis is *retrospective*, not a seven-day future flood forecast. The chance of precipitation is not the chance of flooding.

## Machine Learning

SITR uses a **Random Forest classifier** trained on historical weather and NOAA flood-report labels. The simplified model uses four weather measurements, plus county identification:

| Feature | Description |
| --- | --- |
| Daily rainfall | Total rainfall for the input day |
| Maximum hourly rainfall | Largest hourly rainfall amount |
| Previous 72-hour rainfall | Accumulated rainfall from preceding days |
| Average soil moisture | Soil wetness measurement |
| County identifier | Encoded county used in training |

The model is trained to score whether **a NOAA flood report may be recorded on the following day**. It does **not** directly predict whether a particular street or neighborhood will flood.

### Held-out evaluation (2023–2025)

| Metric | Result |
| --- | ---: |
| ROC-AUC | **0.763** |
| Precision at a 50% score threshold | **2.6%** |
| Recall at a 50% score threshold | **15.8%** |
| Reported flood days correctly flagged | **6 of 38** |

**ROC-AUC is not classification accuracy.** The model ranks reported flood days above unreported days better than chance, but its alert precision and recall remain low. It should be treated as an experiment rather than an operational flood-prediction tool. Scores generated from live/archived Open-Meteo inputs or estimated missing values have **not** been separately validated.

## Technology

- **Python** — data preparation, modeling, and dashboard logic
- **Streamlit** — interactive web interface
- **Pandas** — weather and prediction data processing
- **PyDeck** — interactive geographic visualization
- **Shapely** — illustrative map geometry
- **scikit-learn / joblib** — Random Forest training and model loading
- **Requests** — weather, map-boundary, and alert API calls
- **Open-Meteo** — forecast and recent historical weather
- **NOAA / National Weather Service** — historical flood reports and official active alerts

## Getting Started

### 1. Requirements

- Python 3.10 or newer recommended
- Internet access for live weather, official alerts, and first-time map-boundary retrieval
- The historical datasets and trained model files described below

Install dependencies from your project directory:

```bash
python -m pip install streamlit pandas pydeck requests shapely scikit-learn joblib numpy
```

### 2. Project files

The dashboard expects the following structure (the file names below are important):

```text
Flood/
├── sitr_map_dashboard_with_risk.py
├── train_sitr_simplified.py          # Optional: retrain the model
└── data/
    ├── extended_training_data.csv
    ├── multi_location_weather/
    │   ├── harris__katy__2015.csv
    │   ├── harris__baytown__2015.csv
    │   └── ...                       # Corresponding years used by the map
    ├── extended_model_results/
    │   └── test_predictions.csv      # Optional historical score display
    └── simplified_model_results/
        └── random_forest.joblib      # Required for recent AI scores
```

The dashboard may also create `data/harris_county_boundary.geojson` as a local cache.

If you downloaded the **SITR Simplified Model Package**, extract the model into `data/simplified_model_results/random_forest.joblib`. The training script itself writes its outputs to `sitr_simplified_model_results/` at the project root, so **copy the trained `.joblib` file into `data/simplified_model_results/`** if retraining locally.

**GitHub note:** Large weather datasets and serialized model files may exceed GitHub's recommended file sizes. Consider Git LFS, a release asset, or a documented download link. If those files are not included in your repository, visitors must obtain them before running the complete dashboard. Only load `.joblib` files from sources you trust.

### 3. Start the dashboard

In a terminal opened in the project directory:

```bash
python -m streamlit run sitr_map_dashboard_with_risk.py
```

If you have kept the latest dashboard's downloaded name instead:

```bash
python -m streamlit run sitr_dashboard_branded.py
```

Streamlit will display a local URL, usually **http://localhost:8501** (or another available port such as 8502). Press **Ctrl+C** in the terminal to stop it.

## Retraining the Simplified Model

With `data/extended_training_data.csv` in place:

```bash
python train_sitr_simplified.py
```

The training script uses data through **2022** for training and **2023–2025** as a held-out test period. It generates a serialized model, test predictions, and evaluation metrics in `sitr_simplified_model_results/`.

After training, place the resulting `random_forest.joblib` at `data/simplified_model_results/random_forest.joblib` so the dashboard can find it.

## Data Sources and Interpretation

- **Historical weather:** ERA5-derived weather data used in the project's training dataset and sample-location weather files.
- **Flood-event labels:** NOAA-reported flood events aggregated by county and date. An unreported day is **not** proof that no flooding occurred.
- **Recent weather:** Open-Meteo forecast and historical/archive endpoints. These products may differ from the ERA5 data used for training.
- **Official alerts:** National Weather Service alerts for Harris County, displayed independently from the model.

When recent measurements are incomplete, the dashboard may substitute historical monthly median values for selected inputs. Such rows should be labeled as **estimated**, and the resulting scores are exploratory. If too much rainfall data is missing, the dashboard withholds the score.

## Limitations

1. The model predicts the presence of a **next-day NOAA flood report**, not confirmed flooding at an exact location.
2. The reported score percentages are **not validated flood probabilities**.
3. NOAA event records are incomplete proxies for real-world flooding.
4. The geographic regions are illustrative, not official flood-risk maps.
5. Recent weather and estimated missing values introduce differences from the model's training distribution.
6. The dashboard depends on external data services and locally available datasets/model files.

## Project Name

**SITR** stands for **Storm Intelligence & Threat Recognition**.

*“Singing in the rain” — Jinsoul from LOONA* is a playful project slogan, **not** the expansion of the SITR acronym.

---

*Built as an educational flood-data visualization and machine-learning demonstration focused on Harris County, Texas.*
