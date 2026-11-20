"""
AeroPredict - Flask HTML Frontend
===================================
Serves the final multi-task DNN V2 (PyTorch, trained on the full 6.7M-flight
2023 dataset).  One forward pass returns both the delay probability and the
estimated delay minutes.

Model priority:  DNN V2  ->  DNN V1  ->  Spark GBT  ->  SVM
Spark / Hadoop / Java are *not* required at runtime.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os, json, joblib
import numpy as np
from datetime import datetime, date
from functools import lru_cache
from pathlib import Path
from pymongo import MongoClient

# Resolve every artifact / lookup relative to this file so the app starts
# correctly no matter which directory `python flask_app.py` is launched from.
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

try:
    import weather_api
except Exception as _weather_import_error:      # pragma: no cover
    weather_api = None
    print(f"weather_api unavailable ({_weather_import_error}) - climatology only")

app = Flask(__name__)

# ==============================
# MONGODB CONNECTION
# ==============================
mongo_ok  = False
mongo_col = None
try:
    _mc = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    _mc.server_info()                          # test connection
    mongo_col = _mc["aeropredict"]["predictions"]
    mongo_ok  = True
    print("MongoDB connected — aeropredict.predictions")
except Exception as _me:
    print(f"MongoDB not available: {_me}")

# ==============================
# OPTIONAL SPARK + GBT MODEL
# ==============================
# Spark/Hadoop are NOT required anymore. Spark is only started when a saved
# GBT pipeline actually exists on disk; otherwise the app runs PyTorch/sklearn.
PIPELINE_PATH = "gbt_full_pipeline"
DELAY_LOOKUP  = "airport_delay_lookup.csv"
COORDS_LOOKUP = "airport_coords.csv"
WEATHER_LOOKUP= "weather_averages.csv"
VIZ_DATA_DIR  = Path(app.static_folder) / "viz_data"

gbt_loaded = False
pipeline_model = None
spark = None

if os.path.isdir(PIPELINE_PATH):
    try:
        print("Starting Spark session for GBT pipeline...")
        from pyspark.sql import SparkSession
        from pyspark.ml import PipelineModel

        spark = SparkSession.builder \
            .appName("AeroPredict_Flask") \
            .config("spark.driver.memory", "3g") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        pipeline_model = PipelineModel.load(PIPELINE_PATH)
        gbt_loaded = True
        print("GBT PipelineModel loaded.")
    except Exception as e:
        print(f"GBT not available: {e}")
else:
    print("GBT pipeline folder not found - Spark disabled.")

# ==============================
# SVM FALLBACK MODEL
# ==============================
svm_loaded  = False
svm_model   = None
svm_scaler  = None
svm_columns = None
try:
    svm_model   = joblib.load("aeropredict_model.pkl")
    svm_scaler  = joblib.load("scaler.pkl")
    svm_columns = joblib.load("model_columns.pkl")
    svm_loaded  = True
    print("SVM fallback model loaded.")
except Exception as e:
    print(f"SVM fallback not available: {e}")

model_loaded = gbt_loaded or svm_loaded

# ==============================
# DEEP NEURAL NETWORK (PyTorch) - primary model
#   V2 (embeddings + engineered features) preferred, V1 as fallback
# ==============================
dnn_loaded    = False
dnn_model     = None
dnn_prep      = None
dnn_metrics   = {}

dnn_v2_loaded  = False
dnn_v2_model   = None
dnn_v2_builder = None
dnn_v2_state   = None
dnn_v2_metrics = {}

try:
    from dnn_common import load_dnn, make_features, predict_dnn, load_dnn_v2, predict_dnn_v2
except Exception as e:                       # torch missing
    print(f"PyTorch helpers unavailable: {e}")
    load_dnn = make_features = predict_dnn = load_dnn_v2 = predict_dnn_v2 = None

# --- V2 first ---
if load_dnn_v2 is not None:
    try:
        from dnn_features_v2 import FeatureBuilder

        if not (os.path.exists("dnn_v2_model.pt") and os.path.exists("dnn_v2_preprocessor.pkl")):
            raise FileNotFoundError(
                "dnn_v2_model.pt / dnn_v2_preprocessor.pkl missing - run: python train_dnn_v2.py"
            )

        dnn_v2_model, dnn_v2_state = load_dnn_v2("dnn_v2_model.pt", "dnn_v2_preprocessor.pkl")
        dnn_v2_builder = FeatureBuilder.from_state(dnn_v2_state)
        if not getattr(dnn_v2_builder, "route_stats", None):
            print("WARNING: V2 preprocessor has no route_stats - retrain to fix train/serve skew.")
        dnn_v2_loaded = True
        print("DNN V2 (PyTorch embeddings) model loaded.")
        try:
            with open("dnn_v2_metrics.json", "r", encoding="utf-8") as fp:
                dnn_v2_metrics = json.load(fp)
        except Exception:
            dnn_v2_metrics = {}
    except Exception as e:
        print(f"DNN V2 not available (run train_dnn_v2.py): {e}")

# --- V1 fallback ---
if not dnn_v2_loaded and load_dnn is not None:
    try:
        dnn_model, dnn_prep = load_dnn("dnn_model.pt", "dnn_preprocessor.pkl")
        dnn_loaded = True
        print("DNN V1 (PyTorch) model loaded.")
        try:
            with open("dnn_metrics.json", "r", encoding="utf-8") as fp:
                dnn_metrics = json.load(fp)
        except Exception:
            dnn_metrics = {}
    except Exception as e:
        print(f"DNN V1 not available (run train_dnn.py): {e}")

model_loaded = dnn_v2_loaded or dnn_loaded or model_loaded
MODEL_LABEL = (
    "DNN V2 (PyTorch)" if dnn_v2_loaded
    else "DNN V1 (PyTorch)" if dnn_loaded
    else "GBT (Spark MLlib)" if gbt_loaded
    else "SVM (scikit-learn)"
)

# ==============================
# REGRESSION MODEL (delay mins)
# ==============================
reg_loaded    = False
reg_model     = None
reg_scaler    = None
reg_columns   = None
try:
    reg_model   = joblib.load("delay_regressor.pkl")
    reg_scaler  = joblib.load("regressor_scaler.pkl")
    reg_columns = joblib.load("regressor_columns.pkl")
    reg_loaded  = True
    print("Delay regressor loaded.")
except Exception as e:
    print(f"Regressor not available (run train_regressor.py): {e}")

# ==============================
# LOOKUP TABLES
# ==============================
delay_lookup   = {}
coords_lookup  = {}
weather_lookup = {}
airports_list  = []

try:
    df_delay = pd.read_csv(DELAY_LOOKUP)
    delay_lookup = dict(zip(df_delay["Dep_Airport"], df_delay["dep_airport_delay_avg"]))
    df_coords = pd.read_csv(COORDS_LOOKUP)
    for _, r in df_coords.iterrows():
        coords_lookup[r["IATA_CODE"]] = (float(r["LATITUDE"]), float(r["LONGITUDE"]))
    airports_list = sorted(delay_lookup.keys())
    print(f"Loaded {len(airports_list)} airports.")
except Exception as e:
    print(f"Lookup load error: {e}")
    airports_list = ["ATL","DFW","LAX","ORD","DEN","JFK","SFO","BOS","EWR","MIA"]

try:
    df_w = pd.read_csv(WEATHER_LOOKUP)
    for _, r in df_w.iterrows():
        key = (str(r.get("airport_id", r.get("Dep_Airport", ""))), int(r.get("month", 0)))
        weather_lookup[key] = (
            float(r.get("tavg", 15.0)), float(r.get("prcp", 2.0)),
            float(r.get("wspd", 15.0)), float(r.get("pres", 1013.0))
        )
    print("Weather lookup loaded.")
except Exception as e:
    print(f"Weather load error: {e}")

# ==============================
# AIRLINE DISPLAY MAP
# ==============================
AIRLINE_DISPLAY = {
    'Alaska Airlines Inc.'      : 'AS — Alaska Airlines',
    'Allegiant Air'             : 'G4 — Allegiant Air',
    'American Airlines Inc.'    : 'AA — American Airlines',
    'American Eagle Airlines Inc.': 'MQ — American Eagle',
    'Delta Air Lines Inc'       : 'DL — Delta Air Lines',
    'Endeavor Air'              : '9E — Endeavor Air',
    'Frontier Airlines Inc.'    : 'F9 — Frontier Airlines',
    'Hawaiian Airlines Inc.'    : 'HA — Hawaiian Airlines',
    'JetBlue Airways'           : 'B6 — JetBlue Airways',
    'PSA Airlines'              : 'OH — PSA Airlines',
    'Republic Airways'          : 'YX — Republic Airways',
    'Skywest Airlines Inc.'     : 'OO — SkyWest Airlines',
    'Southwest Airlines Co.'    : 'WN — Southwest Airlines',
    'Spirit Air Lines'          : 'NK — Spirit Air Lines',
    'United Air Lines Inc.'     : 'UA — United Airlines',
}

analytics_stats = {
    "total_flights" : 1966327,
    "avg_delay"     : 12.4,
    "model_accuracy": "80.88%",
    "model_type"    : MODEL_LABEL,
    "airlines"      : len(AIRLINE_DISPLAY),
    "airports"      : len(airports_list),
}

# Facts about the model that is actually serving requests (shown in the UI).
model_info = {
    "label"  : MODEL_LABEL,
    "version": "V2" if dnn_v2_loaded else "V1" if dnn_loaded else "GBT" if gbt_loaded else "SVM",
}

if dnn_v2_loaded and dnn_v2_metrics.get("test"):
    _t    = dnn_v2_metrics["test"]
    _rows = dnn_v2_metrics.get("rows", {})
    _arch = dnn_v2_metrics.get("architecture", {})
    analytics_stats.update({
        "total_flights" : int(_rows.get("total", 6743404)),
        "avg_delay"     : round(float(dnn_v2_state.get("y_mean", 12.4)), 1),
        "delayed_pct"   : round(float(dnn_v2_metrics.get("delayed_ratio", 0.1987)) * 100, 2),
        "model_accuracy": f"{_t['accuracy_at_0.5'] * 100:.2f}%",
        "regressor_mae" : round(_t["regression"]["mae_minutes"], 1),
        "test_auc"      : round(_t["auc"], 4),
        "train_rows"    : int(_rows.get("train", 0)),
    })
    model_info.update({
        "accuracy"   : analytics_stats["model_accuracy"],
        "auc"        : analytics_stats["test_auc"],
        "mae"        : analytics_stats["regressor_mae"],
        "train_rows" : analytics_stats["train_rows"],
        "total_rows" : analytics_stats["total_flights"],
        "delayed_pct": analytics_stats["delayed_pct"],
        "parameters" : int(_arch.get("parameters", 0)),
        "features"   : int(dnn_v2_metrics.get("features", {}).get("numeric", 34)),
        "epochs"     : int(dnn_v2_metrics.get("epochs_run", 0)),
    })
elif dnn_loaded and dnn_metrics.get("test"):
    _test = dnn_metrics["test"]
    analytics_stats["model_accuracy"] = f"{_test['accuracy'] * 100:.2f}%"
    analytics_stats["regressor_mae"]  = round(_test.get("mae_minutes", 0.0), 1)
    analytics_stats["test_auc"]       = round(_test.get("auc", 0.0), 4)
    model_info.update({
        "accuracy": analytics_stats["model_accuracy"],
        "auc"     : analytics_stats["test_auc"],
        "mae"     : analytics_stats["regressor_mae"],
    })

# Make it unmistakable at startup which model is actually serving predictions.
print("-" * 66)
print(f"  ACTIVE MODEL : {MODEL_LABEL}")
if model_info.get("accuracy"):
    print(f"  Test accuracy: {model_info['accuracy']}   AUC: {model_info.get('auc', '-')}")
    print(f"  Delay MAE    : {model_info.get('mae', '-')} min (delayed flights)")
if dnn_v2_loaded:
    print(f"  Trained on   : {model_info['train_rows']:,} flights "
          f"(100% of {model_info['total_rows']:,})")
    print(f"  Parameters   : {model_info['parameters']:,}   "
          f"numeric features: {model_info['features']}")
if weather_api is not None:
    _cal = weather_api.load_calibration()
    _n_ap = int(_cal.get("airports_fitted", 0))
    _start, _end = weather_api.forecast_horizon()
    _state = (f"calibrated for {_n_ap} airports" if _n_ap
              else "UNCALIBRATED - run calibrate_open_meteo.py")
    print(f"  Weather      : Open-Meteo forecast {_start} .. {_end} ({_state})")
else:
    print("  Weather      : 2023 monthly climatology only")
print("-" * 66)

# ==============================
# HELPERS
# ==============================
DAY_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
TIME_ORDER = ['Morning', 'Afternoon', 'Evening', 'Night']


def _records(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@lru_cache(maxsize=1)
def load_visualization_payload():
    temp_df = pd.DataFrame(_records(VIZ_DATA_DIR / "weather_vs_delay.json"))
    temp_df = temp_df.rename(columns={"tavg": "temperature_c", "Arr_Delay": "arrival_delay"})
    temp_df["arrival_delay"] = pd.to_numeric(temp_df["arrival_delay"], errors="coerce")
    temp_df["temperature_c"] = pd.to_numeric(temp_df["temperature_c"], errors="coerce")
    temp_df = temp_df.dropna(subset=["temperature_c", "arrival_delay"])
    temp_df = temp_df[temp_df["temperature_c"].between(-35, 45)]
    delay_floor = float(temp_df["arrival_delay"].quantile(0.02))
    delay_ceiling = float(temp_df["arrival_delay"].quantile(0.98))
    temp_df["arrival_delay_clip"] = temp_df["arrival_delay"].clip(delay_floor, delay_ceiling)
    temp_df["temp_band"] = (np.round(temp_df["temperature_c"] / 2) * 2).astype(int)
    temp_trend = (
        temp_df.groupby("temp_band")
        .agg(
            avg_delay=("arrival_delay_clip", "mean"),
            median_delay=("arrival_delay_clip", "median"),
            flight_count=("arrival_delay", "size"),
        )
        .reset_index()
        .sort_values("temp_band")
    )
    temp_trend = temp_trend[temp_trend["flight_count"] >= 40]
    temp_points = temp_df.sample(min(len(temp_df), 2000), random_state=42)

    prcp_df = pd.DataFrame(_records(VIZ_DATA_DIR / "scatterplot_weather.json"))
    prcp_df["prcp"] = pd.to_numeric(prcp_df["prcp"], errors="coerce")
    prcp_df["Average_Delay"] = pd.to_numeric(prcp_df["Average_Delay"], errors="coerce")
    prcp_df = prcp_df.dropna(subset=["prcp", "Average_Delay"])
    prcp_df = prcp_df[prcp_df["prcp"] <= prcp_df["prcp"].quantile(0.99)]
    prcp_df["delay_clip"] = prcp_df["Average_Delay"].clip(
        prcp_df["Average_Delay"].quantile(0.02),
        prcp_df["Average_Delay"].quantile(0.98),
    )
    prcp_df["rain_band"] = (np.floor(prcp_df["prcp"] / 2) * 2).astype(int)
    prcp_trend = (
        prcp_df.groupby("rain_band")
        .agg(avg_delay=("delay_clip", "mean"), point_count=("delay_clip", "size"))
        .reset_index()
        .sort_values("rain_band")
    )

    map_df = pd.DataFrame(_records(VIZ_DATA_DIR / "airport_delay_map.json"))
    map_df["Average_Delay"] = pd.to_numeric(map_df["Average_Delay"], errors="coerce").fillna(0.0)
    map_df["Average_Delay_Display"] = map_df["Average_Delay"].clip(-15, 45)
    map_df["Bubble_Size"] = 10 + np.clip(map_df["Average_Delay_Display"], 0, None) * 0.8

    heatmap_df = pd.DataFrame(_records(VIZ_DATA_DIR / "delay_heatmap.json"))
    heatmap_df["Day"] = pd.Categorical(heatmap_df["Day"], categories=DAY_ORDER, ordered=True)
    heatmap_df["DepTime_label"] = pd.Categorical(
        heatmap_df["DepTime_label"], categories=TIME_ORDER, ordered=True
    )
    heatmap_df = heatmap_df.sort_values(["Day", "DepTime_label"])
    heatmap_matrix = (
        heatmap_df.pivot(index="Day", columns="DepTime_label", values="Average_Delay")
        .reindex(index=DAY_ORDER, columns=TIME_ORDER)
    )

    monthly_avg_df = pd.DataFrame(_records(VIZ_DATA_DIR / "monthly_delay.json"))
    monthly_count_df = pd.DataFrame(_records(VIZ_DATA_DIR / "monthly_delay_count.json"))
    airline_df = pd.DataFrame(_records(VIZ_DATA_DIR / "top_delayed_airlines.json")).sort_values(
        "Average_Arrival_Delay", ascending=False
    )

    peak_month = monthly_avg_df.loc[
        monthly_avg_df["Average_Arrival_Delay_Minutes"].idxmax()
    ]
    top_airline = airline_df.iloc[0]
    busiest_delay_month = monthly_count_df.loc[monthly_count_df["Total_Delays"].idxmax()]

    return {
        "summary": {
            "peak_month": {
                "label": str(peak_month["Month"]),
                "value": round(float(peak_month["Average_Arrival_Delay_Minutes"]), 2),
            },
            "top_airline": {
                "label": str(top_airline["Airline"]),
                "value": round(float(top_airline["Average_Arrival_Delay"]), 2),
            },
            "delay_volume_peak": {
                "label": str(busiest_delay_month["Month"]),
                "value": int(busiest_delay_month["Total_Delays"]),
            },
        },
        "temperature_delay": {
            "trend": temp_trend.to_dict(orient="records"),
            "sample": temp_points[["temperature_c", "arrival_delay_clip"]].rename(
                columns={"arrival_delay_clip": "arrival_delay"}
            ).to_dict(orient="records"),
        },
        "precipitation_delay": prcp_trend.to_dict(orient="records"),
        "airport_map": map_df[
            ["Dep_Airport", "LATITUDE", "LONGITUDE", "Average_Delay", "Average_Delay_Display", "Bubble_Size"]
        ].to_dict(orient="records"),
        "delay_heatmap": {
            "days": DAY_ORDER,
            "times": TIME_ORDER,
            "values": heatmap_matrix.fillna(0).round(2).values.tolist(),
        },
        "monthly_delay": monthly_avg_df.to_dict(orient="records"),
        "monthly_delay_count": monthly_count_df.to_dict(orient="records"),
        "top_airlines": airline_df.to_dict(orient="records"),
    }


def get_weather(airport, month):
    for key in weather_lookup:
        if key[0] == airport and key[1] == month:
            return weather_lookup[key]
    return (15.0, 2.0, 15.0, 1013.0)

def get_coords(airport):
    if airport in coords_lookup:
        return coords_lookup[airport]
    return (39.0, -98.0)

def get_delay_avg(airport):
    return float(delay_lookup.get(airport, 0.20))

def log_prediction(doc):
    """Safely insert a prediction document into MongoDB."""
    if not mongo_ok:
        return
    try:
        mongo_col.insert_one(doc)
    except Exception as e:
        print(f"MongoDB insert error: {e}")

def get_route_stats(dep, arr):
    """Route facts used as model inputs.

    For DNN V2 this returns the *observed* median duration and haul label from
    the training data (6,586 routes).  The old deg->miles formula is only a
    fallback for the legacy models; it puts V2 off-distribution.
    """
    if dnn_v2_loaded:
        return dnn_v2_builder.route_features(dep, arr)

    dep_c = coords_lookup.get(dep, (39.0, -98.0))
    arr_c = coords_lookup.get(arr, (41.9, -87.9))
    lat_d = abs(float(dep_c[0]) - float(arr_c[0]))
    lon_d = abs(float(dep_c[1]) - float(arr_c[1]))
    est_dist = ((lat_d**2 + lon_d**2) ** 0.5) * 69
    est_dur  = max(30, int(est_dist / 8))
    if est_dist < 800:
        dist_type = 'Short Haul >1500Mi'
    elif est_dist < 1500:
        dist_type = 'Medium Haul <3000Mi'
    else:
        dist_type = 'Long Haul <6000Mi'
    return est_dur, dist_type

def _observed_weather(dep, arr, date_str, year):
    """Exact airport-day weather from the training file, when that date is covered."""
    if dnn_v2_builder is None:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    if d.year != year:
        return None
    dos = d.timetuple().tm_yday
    values = {}
    for code in (dep, arr):
        idx = dnn_v2_builder.ap_index.get(code)
        if idx is None:
            return None
        rec = {c: float(dnn_v2_builder.wx_daily[c][idx, dos])
               for c in ("tavg", "prcp", "wspd", "pres", "snow")}
        if not all(np.isfinite(v) for v in rec.values()):
            return None
        values[code] = rec
    d_val = values[dep]
    return (d_val["tavg"], d_val["prcp"], d_val["wspd"], d_val["pres"])


def resolve_weather(dep, arr, date_str, month):
    """Weather inputs for a flight, plus the model override for the forecast.

    Returns ``(tavg, prcp, wspd, pres, wx_override, info)``.

    * Inside Open-Meteo's 15-day horizon -> live forecast (calibrated to the
      training scale), injected into the model through ``wx_override``.
    * A date inside the training year -> the exact airport-day value from the
      training file (no override needed; the model reads it directly).
    * Anything else, or any network problem -> the 2023 monthly climatology
      already baked into the V2 preprocessor.
    """
    tavg, prcp, wspd, pres = get_weather(dep, month)
    year = dnn_v2_state.get("data_year", 2023) if dnn_v2_state else 2023
    info = {
        "source": "climatology",
        "label": f"{year} monthly climatology",
        "detail": "historical monthly average",
        "attribution": "",
    }
    override = None

    # 1) live forecast (only inside the horizon)
    pair = None
    if weather_api is not None and weather_api.forecast_available(date_str):
        try:
            pair = weather_api.get_pair(dep, arr, date_str, coords_lookup)
        except Exception as exc:                  # never let a forecast break a prediction
            print(f"Forecast lookup failed: {exc}")
            pair = None

    if pair:
        d, a = pair["dep"], pair["arr"]
        tavg, prcp, wspd, pres = d["tavg"], d["prcp"], d["wspd"], d["pres"]
        date_id = int(pair["date"].replace("-", ""))
        override = {(dep, date_id): d, (arr, date_id): a}
        return tavg, prcp, wspd, pres, override, {
            "source": "forecast",
            "label": pair["label"],
            "detail": "live forecast for both airports",
            "attribution": weather_api.ATTRIBUTION,
        }

    # 2) a date inside the training year is scored with the *actual* observation
    #    (the model does the same), so show that instead of climate averages.
    observed = _observed_weather(dep, arr, date_str, year)
    if observed is not None:
        tavg, prcp, wspd, pres = observed
        return tavg, prcp, wspd, pres, override, {
            "source": "observed",
            "label": f"{year} observed weather (training data)",
            "detail": "exact airport-day value used by the model",
            "attribution": "",
        }

    # 3) monthly climatology
    return tavg, prcp, wspd, pres, override, info


def parse_date(date_str):
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        return dt.month, dt.day, dt.weekday()+1, days[dt.weekday()], dt.weekday() >= 5
    except:
        return 1, 1, 1, 'Monday', False

def svm_predict(airline_key, dep_airport, arr_airport, dep_time, month, day,
                day_of_week, is_weekend, tavg, prcp, wspd, pres, lat, lon, duration, dist_type):
    row = {c: 0 for c in svm_columns}
    ak = f'Airline_{airline_key}'
    dt = f'DepTime_label_{dep_time}'
    di = f'Distance_type_{dist_type}'
    if ak in row: row[ak] = 1
    if dt in row: row[dt] = 1
    if di in row: row[di] = 1
    row.update({
        'month': month, 'day': day, 'Day_Of_Week': day_of_week,
        'is_weekend': int(is_weekend), 'Flight_Duration': duration,
        'Aicraft_age': 12, 'tavg': tavg, 'prcp': prcp,
        'wspd': wspd, 'pres': pres, 'LATITUDE': lat, 'LONGITUDE': lon,
    })
    df = pd.DataFrame([row])[svm_columns]
    X  = svm_scaler.transform(df.values)
    p  = svm_model.predict_proba(X)[0]
    return float(p[1]), float(p[0])

def regressor_predict(airline_key, dep_time, dist_type, month, day, day_of_week,
                      is_weekend, tavg, prcp, wspd, pres, lat, lon, duration):
    """Predict estimated delay minutes using the GBT Regressor."""
    if not reg_loaded:
        return None
    try:
        row = {c: 0 for c in reg_columns}
        ak = f'Airline_{airline_key}'
        dt = f'DepTime_label_{dep_time}'
        di = f'Distance_type_{dist_type}'
        if ak in row: row[ak] = 1
        if dt in row: row[dt] = 1
        if di in row: row[di] = 1
        row.update({
            'month': month, 'day': day, 'Day_Of_Week': day_of_week,
            'is_weekend': int(is_weekend), 'Flight_Duration': duration,
            'Aicraft_age': 12, 'tavg': tavg, 'prcp': prcp,
            'wspd': wspd, 'pres': pres, 'LATITUDE': lat, 'LONGITUDE': lon,
        })
        df  = pd.DataFrame([row])[[c for c in reg_columns if c in row]]
        # align to exact training columns
        df  = df.reindex(columns=reg_columns, fill_value=0)
        X   = reg_scaler.transform(df.values)
        est = float(reg_model.predict(X)[0])
        return max(0, round(est))
    except Exception as e:
        print(f"Regressor predict error: {e}")
        return None

# ==============================
# ROUTES
# ==============================
@app.route("/api/airlines")
def api_airlines():
    return jsonify(list(AIRLINE_DISPLAY.keys()))

@app.route("/api/airports")
def api_airports():
    return jsonify(airports_list)

@app.route("/api/autofill")
def api_autofill():
    dep      = request.args.get('dep', '').upper().strip()
    arr      = request.args.get('arr', '').upper().strip()
    date_str = request.args.get('date', '')
    dep_time = request.args.get('dep_time', 'Morning')

    month, day, dow, day_name, is_wknd = parse_date(date_str)
    tavg, prcp, wspd, pres, wx_override, wx_info = resolve_weather(dep, arr, date_str, month)
    est_dur, dist_type     = get_route_stats(dep, arr)

    return jsonify({
        'day_of_week'     : day_name,
        'is_weekend'      : 'Yes' if is_wknd else 'No',
        'flight_duration' : est_dur,
        'route_type'      : dist_type,
        'tavg'            : round(tavg, 2),
        'prcp'            : round(prcp, 2),
        'wspd'            : round(wspd, 2),
        'pres'            : round(pres, 2),
        'aircraft_age'    : 12,
        'dep_airport'     : dep,
        'weather_source'  : wx_info['label'],
        'weather_detail'  : wx_info['detail'],
        'weather_used'    : bool(wx_override),
        'attribution'     : wx_info['attribution'],
    })

@app.context_processor
def inject_globals():
    """Inject airlines, airports and active-model facts into every template."""
    return dict(airlines=AIRLINE_DISPLAY, airports=airports_list, model_info=model_info)

@app.route("/")
def home():
    return render_template("home.html", stats=analytics_stats)

@app.route("/visualizations")
def visualizations():
    return render_template("visualizations.html", stats=analytics_stats)

@app.route("/settings")
def settings():
    return render_template("settings.html", stats=analytics_stats)

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html",
        stats=analytics_stats,
        airlines=AIRLINE_DISPLAY,
        airports=airports_list,
        model_loaded=model_loaded,
        prediction=None,
    )

@app.route("/predict", methods=["POST"])
def predict():
    airline_key = request.form.get("airline_key", "").strip()
    dep_airport = request.form.get("dep_airport", "").upper().strip()
    arr_airport = request.form.get("arr_airport", "").upper().strip()
    date_str    = request.form.get("travel_date", "")
    dep_time    = request.form.get("dep_time", "Morning")

    if not airline_key or not dep_airport or not arr_airport:
        return render_template("predict.html",
            stats=analytics_stats, airlines=AIRLINE_DISPLAY,
            airports=airports_list, model_loaded=model_loaded,
            error="Please fill in Airline, Departure and Arrival airports.",
            prediction=None,
        )

    if not date_str:                      # form is `required`, but stay safe
        date_str = date.today().isoformat()
    if dep_airport == arr_airport:
        return render_template("predict.html",
            stats=analytics_stats, airlines=AIRLINE_DISPLAY,
            airports=airports_list, model_loaded=model_loaded,
            error="Departure and arrival airports must be different.",
            prediction=None,
        )

    month, day, dow, day_name, is_wknd = parse_date(date_str)
    tavg, prcp, wspd, pres, wx_override, wx_info = resolve_weather(
        dep_airport, arr_airport, date_str, month)
    lat, lon               = get_coords(dep_airport)
    est_dur, dist_type     = get_route_stats(dep_airport, arr_airport)

    autofill = {
        'day_of_week'    : day_name,
        'is_weekend'     : 'Yes' if is_wknd else 'No',
        'flight_duration': est_dur,
        'route_type'     : dist_type,
        'tavg'           : round(float(tavg), 2),
        'prcp'           : round(float(prcp), 2),
        'wspd'           : round(float(wspd), 2),
        'pres'           : round(float(pres), 2),
        'aircraft_age'   : 12,
        'weather_source' : wx_info['label'],
        'weather_detail' : wx_info['detail'],
    }

    try:
        # Priority: DNN V2 -> DNN V1 -> GBT (Spark, if present) -> SVM fallback
        dnn_est_mins = None
        if dnn_v2_loaded:
            row = {
                "FlightDate"    : date_str,
                "Day_Of_Week"   : dow,
                "Airline"       : airline_key,
                "Dep_Airport"   : dep_airport,
                "Arr_Airport"   : arr_airport,
                "DepTime_label" : dep_time,
                "Flight_Duration": est_dur,
                "Distance_type" : dist_type,
                "Aicraft_age"   : 12,
            }
            x_num, x_cat = dnn_v2_builder.build_one(row, wx_override=wx_override)
            delay_prob, dnn_est_mins = predict_dnn_v2(dnn_v2_model, dnn_v2_state, x_num, x_cat)
            on_time_prob = 1.0 - delay_prob
        elif dnn_loaded:
            numeric_values = {
                "Day_Of_Week"           : dow,
                "Flight_Duration"       : est_dur,
                "Aicraft_age"           : 12,
                "tavg"                  : float(tavg),
                "prcp"                  : float(prcp),
                "wspd"                  : float(wspd),
                "pres"                  : float(pres),
                "LATITUDE"              : float(lat),
                "LONGITUDE"             : float(lon),
                "month"                 : month,
                "day"                   : day,
                "is_weekend"            : int(is_wknd),
                "dep_airport_delay_avg" : get_delay_avg(dep_airport),
                "arr_airport_delay_avg" : get_delay_avg(arr_airport),
            }
            features = make_features(
                dnn_prep["columns"],
                numeric_values,
                {
                    "Airline"       : airline_key,
                    "DepTime_label" : dep_time,
                    "Distance_type" : dist_type,
                },
            )
            delay_prob, dnn_est_mins = predict_dnn(dnn_model, dnn_prep, features)
            on_time_prob = 1.0 - delay_prob
        elif gbt_loaded:
            dep_delay_avg = get_delay_avg(dep_airport)
            arr_delay_avg = get_delay_avg(arr_airport)
            schema_row = [{
                "Airline": airline_key, "DepTime_label": dep_time,
                "Distance_type": dist_type, "tavg": float(tavg),
                "prcp": float(prcp), "wspd": float(wspd), "pres": float(pres),
                "LATITUDE": float(lat), "LONGITUDE": float(lon),
                "month": int(month), "day_of_week": int(dow),
                "dep_airport_delay_avg": float(dep_delay_avg),
                "arr_airport_delay_avg": float(arr_delay_avg),
                "delay_label": 0
            }]
            pred_df    = spark.createDataFrame(schema_row)
            result_row = pipeline_model.transform(pred_df).select("prediction","probability").first()
            prob_arr   = result_row["probability"]
            delay_prob   = float(prob_arr[1])
            on_time_prob = float(prob_arr[0])
        elif svm_loaded:
            delay_prob, on_time_prob = svm_predict(
                airline_key, dep_airport, arr_airport, dep_time,
                month, day, dow, is_wknd, float(tavg), float(prcp),
                float(wspd), float(pres), float(lat), float(lon),
                est_dur, dist_type
            )
        else:
            raise Exception("No model available")

        # Risk classification
        if delay_prob >= 0.40:
            risk, icon, big_label, verdict, color = \
                "high","🔴","DELAYED","High Delay Risk","#ef4444"
        elif delay_prob >= 0.25:
            risk, icon, big_label, verdict, color = \
                "medium","⚠️","POSSIBLY DELAYED","Moderate Risk — Monitor Flight","#f59e0b"
        else:
            risk, icon, big_label, verdict, color = \
                "low","✅","ON TIME","Low Risk","#10b981"

        # ── Regression: estimate delay minutes if flight is delayed ──
        est_delay_mins = None
        if risk in ('high', 'medium'):
            if dnn_v2_loaded or dnn_loaded:
                est_delay_mins = int(round(dnn_est_mins))
            else:
                est_delay_mins = regressor_predict(
                    airline_key, dep_time, dist_type,
                    month, day, dow, is_wknd,
                    float(tavg), float(prcp), float(wspd), float(pres),
                    float(lat), float(lon), est_dur
                )

        prediction = {
            'delay_prob'    : round(delay_prob * 100, 1),
            'on_time_prob'  : round(on_time_prob * 100, 1),
            'risk'          : risk,
            'icon'          : icon,
            'big_label'     : big_label,
            'verdict'       : verdict,
            'color'         : color,
            'airline_display': AIRLINE_DISPLAY.get(airline_key, airline_key),
            'route'         : f"{dep_airport} → {arr_airport}",
            'date_display'  : date_str,
            'dep_time'      : dep_time,
            'est_delay_mins': est_delay_mins,
            'reg_model_label': ("DNN V2 regression head — delay given a delayed flight"
                                if dnn_v2_loaded
                                else "DNN regression head estimate" if dnn_loaded
                                else "GBT Regression model estimate"),
            'model_used'    : MODEL_LABEL,
            'weather_source': wx_info['label'],
            'weather_attribution': wx_info['attribution'],
            'weather_override': bool(wx_override),
        }

        # ── Log to MongoDB ──
        log_prediction({
            "timestamp"      : datetime.utcnow(),
            "source"         : "flask",
            "airline"        : AIRLINE_DISPLAY.get(airline_key, airline_key),
            "dep_airport"    : dep_airport,
            "arr_airport"    : arr_airport,
            "route"          : f"{dep_airport} → {arr_airport}",
            "travel_date"    : date_str,
            "dep_time"       : dep_time,
            "delay_prob"     : round(delay_prob * 100, 1),
            "on_time_prob"   : round(on_time_prob * 100, 1),
            "prediction"     : big_label,
            "risk"           : risk,
            "model_used"     : MODEL_LABEL,
            "est_delay_mins" : est_delay_mins,
            "flight_duration": est_dur,
            "route_type"     : dist_type,
            "weather"        : {"tavg": round(float(tavg),2), "prcp": round(float(prcp),2),
                                "wspd": round(float(wspd),2), "pres": round(float(pres),2),
                                "source": wx_info['source']},
        })

        return render_template("predict.html",
            stats=analytics_stats, airlines=AIRLINE_DISPLAY,
            airports=airports_list, model_loaded=model_loaded,
            prediction=prediction, autofill=autofill,
            form_values={
                'airline_key': airline_key, 'dep_airport': dep_airport,
                'arr_airport': arr_airport, 'travel_date': date_str,
                'dep_time': dep_time,
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("predict.html",
            stats=analytics_stats, airlines=AIRLINE_DISPLAY,
            airports=airports_list, model_loaded=model_loaded,
            error=f"Prediction error: {str(e)}", prediction=None,
        )

@app.route("/history")
def history():
    records = []
    if mongo_ok:
        try:
            records = list(mongo_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(25))
        except Exception as e:
            print(f"MongoDB fetch error: {e}")
    return render_template("history.html", records=records, stats=analytics_stats,
                           mongo_ok=mongo_ok)

@app.route("/api/analytics")
def get_analytics():
    return jsonify(analytics_stats)

@app.route("/api/visualization-data")
def visualization_data():
    return jsonify(_json_ready(load_visualization_payload()))

@app.errorhandler(404)
def not_found(error):
    return render_template("home.html", stats=analytics_stats), 404

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
