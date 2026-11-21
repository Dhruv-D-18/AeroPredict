# AeroPredict: Flight Delay Risk Classification & Regression Dashboard

AeroPredict is an end-to-end, industrial-grade **Big Data & Artificial Intelligence** application designed to predict flight delays, assess delay risk probabilities, and estimate exact late arrival times (in minutes). Its primary engine is a **multi-task Deep Neural Network (PyTorch)** trained on half of the 6.7M-flight 2023 dataset, with optional **PySpark MLlib GBT** and **Scikit-Learn** fallbacks. The web dashboard is built with **Flask**, **MongoDB**, and **Plotly**.

This project was developed as a final course project for **Big Data Visualization and Data Analytics (Semester 6)**.

---

## 🏗️ System Architecture

The following diagram outlines the data flow from raw storage to the real-time user interface:

```
                  +----------------------------------------------+
                  |  RAW DATASETS (6.5 Million Records in HDFS)  |
                  |  - US_flights_2023.csv (1.1 GB)              |
                  |  - weather_meteo_by_airport.csv (Daily)      |
                  |  - Cancelled_Diverted_2023.csv (Exclusions)  |
                  |  - airports_geolocation.csv (GPS coords)     |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |     Apache Spark Distributed ETL Engine      |
                  |     - Removes cancelled/diverted flights     |
                  |     - Attaches airport geolocations          |
                  |     - Joins weather variables on date+airport|
                  |     - Natively imputes missing sensor data   |
                  +----------------------------------------------+
                                    /            \
                                   /              \
                                  v                v
       +--------------------------------------+  +-------------------------------------+
       |  DNN V2 - PyTorch (PRIMARY)          |  |  CLASSIC FALLBACKS                  |
       |  - Embeddings + 34 features, 100%    |  |  - GBT Classifier (Spark MLlib)     |
       |  - Head 1: delay probability (>15m)  |  |  - Linear SVM (scikit-learn)        |
       |  - Head 2: delay minutes             |  |  - GBT Regressor (scikit-learn)     |
       +--------------------------------------+  +-------------------------------------+
                                 \                /
                                  \              /
                                   v            v
                  +----------------------------------------------+
                  |            Flask Premium Frontend            |
                  |   - /predict-page: User enters flight data   |
                  |     (Autofills weather & flight metrics)     |
                  |   - /visualizations: Plotly analytics        |
                  |   - Settings: model info (DNN / fallback)    |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |            MongoDB Logging System            |
                  |   - Stores all runtime predictions in DB     |
                  |   - Collection: aeropredict.predictions      |
                  +----------------------------------------------+
```

---

## 📊 Core Project Statistics

Below are the key figures and performance metrics computed on the flight logs:

| Metric | Value | Meaning / Context |
| :--- | :--- | :--- |
| **Total Raw Records** | **6.5 Million** flights | Total US domestic flights processed through HDFS. |
| **Cleaned Dataset** | **1.96 Million** flights | Count after filtering cancellations, diversions, and duplicates. |
| **HDFS Target Directory** | `hdfs://localhost:9000/flight_project/` | Active Hadoop Distributed File System repository. |
| **Spark partitions** | **16 Partitions** | The scale of parallel execution chunks across the cluster. |
| **Spark ETL Run Time** | **~16 seconds** | Time Spark takes to join, clean, and impute 6.5M rows. |
| **DNN V2 Training Rows** | **6.41 Million** flights | Every valid row of the 2023 dataset (100%, no sampling). |
| **DNN V2 Accuracy** | **84.40%** | Delay classification accuracy on a 168,586-flight holdout. |
| **Risk Thresholds** | **Low ($\le$ 25%), Med (25% - 40%), High ($\ge$ 40%)** | Delay probability brackets. |
| **MongoDB Port** | **27017** | Database connection port for runtime auditing. |

---

## 🏁 ML vs DL Model Comparison

Three engines are kept in the repository so the classic **Machine Learning**
baseline and both **Deep Learning** generations can be compared side by side.

### Classification — predicting a delay > 15 minutes

| | **Classic ML (Spark MLlib GBT + SVM)** | **DNN V1 (50% data)** | **DNN V2 (100% data)** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 80.88% | 81.44% | **84.40%** |
| **AUC-ROC** | — (not recorded) | 0.7306 | **0.7996** |
| **Precision @0.5** | — (not recorded) | 64.69% | **78.80%** |
| **Recall @0.5** | — (not recorded) | 14.39% | **29.37%** |
| **F1 @0.5** | — (not recorded) | 23.55% | **42.79%** |
| **Training data** | 1.96M cleaned flights | 3.37M (50% sample) | **6.41M (100%)** |
| **Features** | 14 | 36 (14 num + 22 one-hot) | **34 num + 5 embeddings** |
| **Parameters** | — | 51,522 | 219,718 |
| **Runtime requirement** | Spark + Hadoop + Java | CPU-only PyTorch | CPU-only PyTorch |

### Regression — estimating delay length in minutes

| | **Classic ML (GBR)** | **DNN V1** | **DNN V2** |
| :--- | :--- | :--- | :--- |
| **MAE** | ~18.2 min* | 37.38 min | **35.80 min** |
| **RMSE** | — | 63.08 min | **61.54 min** |
| **R²** | — | −0.039 | **+0.017** |
| **MAE (all flights)** | — | 48.65 min | 49.23 min |
| **Scored on** | delayed + on-time mix* | delayed only | delayed only |

\* The baseline's ~18.2 min was reported on a *mixed* subset where ~80% of rows
are near-zero and easy to predict, so it is **not comparable** to the delayed-only
numbers. On the same "all flights" basis V2 scores 49.23 min. Delay duration is
inherently noisy — most of it is not explainable from data available at booking
time.

### DNN V2 risk-tier behaviour (the cut-offs the app actually uses)

| Tier | Threshold | % of test flights flagged | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Medium+High** | P ≥ 25% | 20.68% | 52.55% | 54.71% | 0.536 |
| **High** | P ≥ 40% | 10.73% | 69.33% | 37.45% | 0.486 |
| (reference) | P ≥ 50% | 7.40% | 78.80% | 29.37% | 0.428 |

Mean predicted probability on the test split was **18.18%** vs **19.87%** of
flights actually delayed — the classification head is well calibrated, so the
same tier cut-offs mean the same thing in the UI as they do in the metrics.

### What V2's preprocessing buys (controlled ablation)

Same 10% of the data, same 8 epochs, only the preprocessing/architecture changed:

| | DNN V1 pipeline | DNN V2 pipeline | Gain |
| :--- | :--- | :--- | :--- |
| AUC-ROC | 0.7106 | **0.7753** | **+0.065** |
| Accuracy | 80.83% | **84.02%** | **+3.2 pts** |
| Delay MAE | 38.03 min | **36.83 min** | −1.2 min |

### DNN V2 preprocessing pipeline

| Step | What it does |
| :--- | :--- |
| **Categorical embeddings** | Airline, departure airport, arrival airport, departure slot and haul type become learned vectors (8/16/16/4/3 dims) instead of 351 near-empty one-hot columns |
| **Arrival-side context** | Adds arrival-airport weather (`tavg`, `prcp`, `wspd`, `pres`) and coordinates — V1 only knew the departure airport |
| **Great-circle distance** | Haversine dep→arr in km (0–8,200), replacing the coarse 3-level haul bucket as a numeric signal |
| **Cyclical calendar** | Month, day-of-year and day-of-week as sin/cos pairs, so 31 Dec and 1 Jan are adjacent |
| **Smoothed target encodings** | Departure-airport, arrival-airport, airline, **route**, **airport × slot**, airline × slot and **airport × month** delay rates, Bayesian-smoothed `(sum + prior·k)/(count + k)`; fitted on training rows only so the test split never leaks |
| **Volume features** | `log1p(flights on that route)`, `log1p(flights from that airport that day)` for congestion pressure |
| **Robust scaling** | Clip each numeric feature to the train p0.1–p99.9 range, then standardise — one 571 mm rain day or a 57-year-old aircraft can no longer dominate |
| **Route facts for serving** | Observed median `Flight_Duration` + haul label per route (6,586 routes) are stored in the preprocessor, so the web app no longer guesses them with a crude distance formula |

Leakage audit: the strongest correlation between any engineered feature and the
target is `route_rate` at **|r| = 0.21**, which is healthy signal rather than leakage.

### DNN V2 training run (reproducible)

| Setting | Value |
| :--- | :--- |
| Data | `--sample-frac 1.0` → **6,743,404 rows** (19.87% delayed) |
| Split | 6,406,233 train / 168,585 val / 168,586 test (stratified) |
| Architecture | 34 numeric + embeddings `8/16/16/4/3` → 81 dims → 512 → 256 → 128 → 2 heads |
| Parameters | 219,718 |
| Epochs | 12 (best epoch 11) |
| Optimiser | AdamW lr 2.5e-3, `ReduceLROnPlateau`, batch 16,384, grad-clip 5.0 |
| Loss | BCE (classification) + 0.5 × SmoothL1 (regression, delayed rows only) |
| Duration | 6,070 s (~101 min) on 12 CPU threads |
| Command | `python train_dnn_v2.py` |

### DNN V1 training run (previous baseline)

| Setting | Value |
| :--- | :--- |
| Sample | `--sample-frac 0.5` → 3,371,702 rows (19.86% delayed) |
| Split | 2,697,361 train / 337,170 val / 337,171 test (stratified) |
| Architecture | 36 → 256 → 128 → 64 → 2 heads, 51,522 parameters |
| Epochs | 15 / 15 (best epoch = 15, still improving, no early stop) |
| Optimiser | AdamW, lr 1e-3, `ReduceLROnPlateau`, batch 8192 |
| Duration | 754 s (~12.6 min) on 12 CPU threads |
| Command | `python train_dnn.py --sample-frac 0.5 --epochs 15` |

---

## 🧠 Key Features

### 1. Multi-Task Deep Learning (primary) + Classic Fallbacks
- **DNN V2 (PyTorch, primary)**: trained on the **full 6.7M-flight 2023 dataset**. Categorical fields (airline, both airports, departure slot, haul type) enter as **learned embeddings**, alongside 34 numeric features — departure *and* arrival airport weather, great-circle distance, cyclical calendar encodings and smoothed per-airport/route/airline delay rates fitted on training rows only. Trunk `512 → 256 → 128` splits into:
  - *Head 1 — classification*: probability the flight arrives more than 15 minutes late.
  - *Head 2 — regression*: estimated arrival delay in minutes (trained on delayed flights only).
  Both outputs come from a single forward pass, so risk and delay minutes are always consistent.
- **DNN V1 (PyTorch, baseline)**: the earlier `36 → 256 → 128 → 64` MLP trained on 50% of the data. Kept in the repo for comparison; used automatically if the V2 artifacts are missing.
- **Tier 3 (fallback)**: If no DNN artifacts exist, the app falls back to the **Spark MLlib GBT classifier** (when `gbt_full_pipeline/` exists) or the local **Scikit-Learn SVM**, plus the **Gradient Boosting Regressor** for delay minutes. Delay minutes are shown when the risk is **Medium** or **High**.

### 2. Live Interactive Visualizations
A dedicated dashboard rendering live interactive Plotly charts, including:
- **Geospatial Delay Hotspots**: An interactive maps dashboard displaying North American airport flight nodes colored by delay frequency.
- **Delay Heatmap**: Cross-references departure hours and days of the week to pinpoint peak-delay intervals.
- **Weather Correlation Graph**: Scatter sample mapping temperature and rainfall levels against delay lengths, indicating a strong correlation with precipitation (**16.47% feature importance**).

### 3. Autofill Intelligence
When booking details are entered, Flask fills weather, airport delay history and typical
flight duration automatically. Weather is resolved in this order:

1. **Open-Meteo live forecast** for departures within the next 15 days;
2. **observed airport-day weather** from the training file, for 2023 dates;
3. **2023 monthly climatology** for anything further out.

The panel states which source answered (and the model receives that same value), so a
forecast-driven prediction is never confused with a climatology-driven one.

---

## 🌤 Live Weather (Open-Meteo)

For future departures the app no longer relies only on monthly averages. It asks
[Open-Meteo](https://open-meteo.com/) for a daily forecast at **both** airports and feeds
those values into the DNN, using the same five fields the model was trained on
(temperature, precipitation, wind, pressure, snow).

### Source priority

| # | Source | Used when | What the model receives |
| :--- | :--- | :--- | :--- |
| 1 | **Open-Meteo forecast** | date is within `today .. today+14` | calibrated forecast, injected through `wx_override` |
| 2 | **2023 observed** | date is in 2023 | exact airport-day value from the training file |
| 3 | **2023 monthly climatology** | anything further out, or offline | monthly airport average |

Any network error, timeout, unknown airport or out-of-range date silently falls through
to the next source, so a prediction never fails because of the weather API.

### Why a calibration layer?

The training file (`weather_meteo_by_airport.csv`) and Open-Meteo are **different weather
products**. They agree closely on temperature and pressure, but Open-Meteo's wind is
systematically lower (different averaging height) and its coarser precipitation grid
misses local convective storms. Feeding raw values would land the model outside its
training distribution.

`calibrate_open_meteo.py` therefore fits a per-airport correction
`training_value = a × open_meteo_value + b` against Open-Meteo's **Historical Forecast
Archive for 2023** — the same model family the live endpoint uses — and stores it in
`wx_calibration.json` (364 airports). Snow is fitted globally because the unit mismatch
(~85×) is identical everywhere; per-airport fits are noise.

### Measured value

40,000 held-out 2023 flights, scored through DNN V2 with **only the weather source
changed** (`validate_open_meteo.py`, results in `wx_validation.json`):

| Weather source | AUC-ROC | Accuracy | MAE (delayed) | Mean P |
| :--- | :--- | :--- | :--- | :--- |
| 2023 dataset weather *(upper bound)* | 0.8073 | 84.50% | 34.52 min | 0.185 |
| Monthly climatology *(previous behaviour)* | 0.7860 | 83.68% | 35.27 min | 0.185 |
| Raw Open-Meteo | 0.8053 | 84.47% | 34.78 min | 0.182 |
| **Calibrated Open-Meteo (served)** | **0.8055** | **84.39%** | **34.70 min** | **0.185** |

Calibration recovers almost the entire gap between climatology and reality:
**+0.0195 AUC, +0.71 accuracy points and −0.57 min MAE** versus monthly averages, ending
within **0.002 AUC / 0.18 min MAE** of the true-weather upper bound. The accuracy
difference between raw and calibrated (0.08 pt) is sampling noise; calibrated is better on
AUC and MAE, and it maps wind/rain to the scale the model actually learned.

### Limits and attribution

* Forecast horizon is **15 days** (`today` through `today+14`); Open-Meteo rejects later
  dates with HTTP 400, which the client treats as a normal miss.
* Forecasts are cached for **1 hour**; one batched request covers both airports.
* Free tier: **10,000 requests/day**, no API key, Python standard library only.
* Weather data by **Open-Meteo.com, licensed CC BY 4.0** — the required attribution is
  shown in the page footer and returned with every forecast-driven prediction.

---

## 📁 Repository Structure

```
├── flight_delay/               # Raw flight dataset files (Ignored in Git except metadata)
│   ├── airports_geolocation.csv # Airport coordinates metadata (lat/lon)
│   ├── US_flights_2023.csv      # Raw 6.7M-flight dataset (not committed)
│   ├── weather_meteo_by_airport.csv
│   ├── datapreprocessing.py     # Legacy pandas preprocessing
│   └── datatraining.py          # Legacy RandomForest training
├── static/                     # Web app static assets
│   ├── styles.css               # Styling rules (modern dashboard design)
│   ├── plane.png                # Navbar artwork
│   ├── take-off.jpg             # Landing-page background
│   └── viz_data/                # Pre-processed JSON outputs for Plotly charts
├── templates/                  # Flask Jinja2 HTML templates
│   ├── home.html                # Landing page
│   ├── predict.html             # Predictive UI form and results panel
│   ├── visualizations.html      # Plots dashboard
│   ├── history.html             # MongoDB prediction history
│   └── settings.html            # Model/UI settings & MongoDB status
├── dnn_common.py               # PyTorch networks (V1 MLP + V2 embeddings) + helpers
├── dnn_features_v2.py          # V2 preprocessing: embeddings, weather, target encodings
├── weather_api.py              # Open-Meteo client (15-day forecast, cached, fail-soft)
├── calibrate_open_meteo.py     # Fits wx_calibration.json from the 2023 forecast archive
├── validate_open_meteo.py      # 4-way weather-source comparison on real 2023 flights
├── wx_calibration.json         # Per-airport Open-Meteo -> training-scale correction
├── wx_validation.json          # Measured 4-way weather-source comparison (README table)
├── train_dnn.py                # Train the V1 multi-task DNN (50% sample)
├── train_dnn_v2.py             # Train the V2 DNN on the FULL dataset
├── dnn_v2_model.pt             # Saved V2 weights + architecture config
├── dnn_v2_preprocessor.pkl     # V2 vocabularies, weather arrays, rate tables, scaler
├── dnn_v2_metrics.json         # V2 test metrics (shown in the UI)
├── dnn_model.pt                # Saved V1 weights + config (baseline for comparison)
├── dnn_preprocessor.pkl        # V1 feature columns, scaler stats, target scaler
├── dnn_metrics.json            # V1 test metrics
├── gbt_full_pipeline/          # Saved Spark MLlib GBT Classifier (optional, ignored)
├── aeropredict_model.pkl       # SVM Classifier fallback pickle
├── delay_regressor.pkl         # Gradient Boosting Regressor model pickle
├── scaler.pkl                  # SVM Standard Scaler
├── regressor_scaler.pkl        # Regressor Standard Scaler
├── airport_coords.csv          # Lookups for airport locations
├── airport_delay_lookup.csv    # Lookups for airport historical delay ratios
├── weather_averages.csv        # Lookups for historical daily weather averages
├── flask_app.py                # Main Flask web application backend
├── train_model.py              # Script to train SVM fallback (+ Spark GBT)
├── train_regressor.py          # Script to train delay regressor (GBT Regressor)
├── make_gbt_flask_pipeline.py  # Script to train Spark MLlib GBT model
├── benchmark.py                # HDFS / Spark / MongoDB benchmark (optional)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Standard git exclusions for large data/models
└── README.md                   # This project guide
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.10+**
- **MongoDB** (optional — runs on port `27017` for prediction history)
- *Optional:* **Java JDK 8/11 + Apache Spark 3.5.x** only if you want to run the legacy GBT pipeline. The DNN and the Flask app run without Spark/Hadoop.

### 2. Environment Setup
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd AeroPredict
   ```

2. Create a virtual environment and install python dependencies:
   ```bash
   python -m venv venv
   source venv/Scripts/activate     # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. *(Optional — legacy Spark GBT only)* declare Hadoop/PySpark paths:
   ```powershell
   $env:SPARK_HOME="C:\path\to\spark-3.5.x-bin-hadoop3"
   $env:PATH="$env:SPARK_HOME\bin;$env:PATH"
   ```

### 3. Database Initialization (Optional)
Ensure a local instance of MongoDB is running:
```bash
# Test connection via CLI or MongoDB Compass
mongosh "mongodb://localhost:27017"
```
*Note: Flask connects automatically. If MongoDB is offline, Flask will display a warning in the console and continue predicting in offline mode, bypassing database logs.*

---

## Pictures

<p align="center">
  <img width="49%" alt="home_page" src="https://github.com/user-attachments/assets/fa1b7678-3abc-4216-95e4-8155ff3477a6" />
  <img width="49%" alt="Prediction_page" src="https://github.com/user-attachments/assets/71f224c5-64d0-41a8-b82f-3b9e9a2a8fad" />
</p>
<p align="center">
  <img width="49%" alt="Prediction_result" src="https://github.com/user-attachments/assets/b18972d8-3d2d-456f-96e1-bb2bb684b8b2" />
  <img width="49%" alt="Prediction_result2" src="https://github.com/user-attachments/assets/ef6cf9b4-38ca-4bce-ac50-5b39ca50535f" />
</p>


## 🚀 How to Run

### 1. Model Training
*Note: Pre-trained DNN, SVM and Regressor artifacts are included, so the app starts immediately. To re-train:*

- **Train the V2 multi-task DNN (primary, full dataset, no Spark needed)**:
  ```bash
  python train_dnn_v2.py                        # 100% of rows, 12 epochs
  python train_dnn_v2.py --sample-frac 0.02     # quick smoke test
  ```
  Writes `dnn_v2_model.pt`, `dnn_v2_preprocessor.pkl`, `dnn_v2_metrics.json`.
  The app picks V2 up automatically; V1 is used only if V2 artifacts are absent.

- **Train the V1 DNN baseline (50% sample, for comparison)**:
  ```bash
  python train_dnn.py --sample-frac 0.5 --epochs 15
  ```
  Writes `dnn_model.pt`, `dnn_preprocessor.pkl`, `dnn_metrics.json`.

- **Optional legacy fallbacks**:
  ```bash
  python make_gbt_flask_pipeline.py   # Spark GBT pipeline
  python train_model.py               # Spark GBT + SVM pickle
  python train_regressor.py           # sklearn GBT regressor
  ```

### 2. Start the Web Dashboard
The DNN runs on CPU, so plain Python is enough, and the app can be launched
from any directory:
```bash
python flask_app.py
```
Then open:
```
http://127.0.0.1:5000/
```

On startup the app prints exactly which model it is serving, e.g.:

```
------------------------------------------------------------------
  ACTIVE MODEL : DNN V2 (PyTorch)
  Test accuracy: 84.40%   AUC: 0.7996
  Delay MAE    : 35.8 min (delayed flights)
  Trained on   : 6,406,233 flights (100% of 6,743,404)
  Parameters   : 219,718   numeric features: 34
  Weather      : Open-Meteo forecast 2026-11-05 .. 2026-11-19 (calibrated for 364 airports)
------------------------------------------------------------------
```

Model selection is automatic at import time:
`dnn_v2_model.pt` + `dnn_v2_preprocessor.pkl` → V1 DNN artifacts →
Spark GBT pipeline (only if `gbt_full_pipeline/` exists) → SVM.
No Spark, Hadoop or Java is needed for the DNN path.

### 3. Weather Calibration (one-off, optional)
The app ships with `wx_calibration.json`, so live forecasts work out of the box.
Re-run this only if you change the training weather file or want to refresh the
correction against a newer Open-Meteo archive:

```bash
python calibrate_open_meteo.py            # ~18 archive requests, resumable
python validate_open_meteo.py --rows 40000  # optional: 4-way weather comparison
```

Both scripts cache responses in `.wx_cache/` (gitignored) and use only the
Python standard library for HTTP, so no API key or extra dependency is needed.

---

## 🏫 Credits & Course Details
- **Course**: Big Data Visualization and Data Analytics (Semester 6)
- **Project Team**: [Your Name/Teammate Names]
- **Institution**: [Your University/College Name]
