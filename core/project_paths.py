"""
AeroPredict — project file locations
=====================================
Every script in this repository finds its data, models and outputs through
this one file, so the folder layout can be understood (and changed) in one place.

Usage from any script, in any sub-folder:

    import sys
    from pathlib import Path
    _ROOT = next(p for p in Path(__file__).resolve().parents if (p / "core" / "project_paths.py").exists())
    sys.path.insert(0, str(_ROOT / "core"))
    import project_paths as P

    df = pd.read_csv(P.RAW_FLIGHTS)
"""
from pathlib import Path

# ── Repository root ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

CORE_DIR       = ROOT / "core"
WEBAPP_DIR     = ROOT / "webapp"
DATA_DIR       = ROOT / "data"
MODELS_DIR     = ROOT / "models"
TRAINING_DIR   = ROOT / "training"
TOOLS_DIR      = ROOT / "tools"
EVALUATION_DIR = ROOT / "evaluation"
DATABASE_DIR   = ROOT / "database"
DOCS_DIR       = ROOT / "docs"

# ── Data ────────────────────────────────────────────────────────────────
RAW_DIR        = DATA_DIR / "raw"          # original Kaggle / 2023 CSVs (large, not in git)
PROCESSED_DIR  = DATA_DIR / "processed"    # cleaned / merged datasets
LOOKUPS_DIR    = DATA_DIR / "lookups"      # small tables the web app needs at startup
WEATHER_DIR    = DATA_DIR / "weather"      # Open-Meteo calibration + validation
GRAPH_DIR      = DATA_DIR / "graph"        # Phase 2: airport graph + time-slot series

RAW_FLIGHTS        = RAW_DIR / "US_flights_2023.csv"
RAW_WEATHER        = RAW_DIR / "weather_meteo_by_airport.csv"
RAW_AIRPORTS       = RAW_DIR / "airports_geolocation.csv"
RAW_CANCELLED      = RAW_DIR / "Cancelled_Diverted_2023.csv"
RAW_FLIGHTS_2024   = RAW_DIR / "maj us flight - january 2024.csv"

PROCESSED_FINAL    = PROCESSED_DIR / "preprocessed_flight_data_final.csv"
PROCESSED_V2       = PROCESSED_DIR / "preprocessed_flight_datav2.csv"
ROUTE_STATS        = PROCESSED_DIR / "route_stats.csv"

AIRPORT_COORDS     = LOOKUPS_DIR / "airport_coords.csv"
AIRPORT_DELAY      = LOOKUPS_DIR / "airport_delay_lookup.csv"
WEATHER_AVERAGES   = LOOKUPS_DIR / "weather_averages.csv"

WX_CALIBRATION     = WEATHER_DIR / "wx_calibration.json"
WX_VALIDATION      = WEATHER_DIR / "wx_validation.json"
WX_CACHE_DIR       = ROOT / ".wx_cache"    # Open-Meteo response cache (git-ignored)

# ── Models ──────────────────────────────────────────────────────────────
DNN_V2_DIR         = MODELS_DIR / "dnn_v2"       # Phase 1 — primary model
DNN_V2_MODEL       = DNN_V2_DIR / "dnn_v2_model.pt"
DNN_V2_PREP        = DNN_V2_DIR / "dnn_v2_preprocessor.pkl"
DNN_V2_METRICS     = DNN_V2_DIR / "dnn_v2_metrics.json"

DNN_V1_DIR         = MODELS_DIR / "dnn_v1"       # Phase 1 — baseline (50% data)
DNN_V1_MODEL       = DNN_V1_DIR / "dnn_model.pt"
DNN_V1_PREP        = DNN_V1_DIR / "dnn_preprocessor.pkl"
DNN_V1_METRICS     = DNN_V1_DIR / "dnn_metrics.json"

CLASSIC_DIR        = MODELS_DIR / "classic_ml"   # Semester 6 fallbacks
SVM_MODEL          = CLASSIC_DIR / "aeropredict_model.pkl"
SVM_SCALER         = CLASSIC_DIR / "scaler.pkl"
SVM_COLUMNS        = CLASSIC_DIR / "model_columns.pkl"
REG_MODEL          = CLASSIC_DIR / "delay_regressor.pkl"
REG_SCALER         = CLASSIC_DIR / "regressor_scaler.pkl"
REG_COLUMNS        = CLASSIC_DIR / "regressor_columns.pkl"
GBT_PIPELINE       = CLASSIC_DIR / "gbt_full_pipeline"   # Spark MLlib (optional folder)

ST_GNN_DIR         = MODELS_DIR / "st_gnn"       # Phase 2 — AeroPredict 2.0
ST_GNN_MODEL       = ST_GNN_DIR / "st_gnn_model.pt"          # GCN + GRU (served by the app)
ST_GNN_GAT_MODEL   = ST_GNN_DIR / "st_gnn_gat_model.pt"      # GAT + GRU
TEMPORAL_DIR       = MODELS_DIR / "temporal"     # Phase 2 — GRU / LSTM baselines

# ── Web app ─────────────────────────────────────────────────────────────
VIZ_DATA_DIR       = WEBAPP_DIR / "static" / "viz_data"

# ── Evaluation (Phase 2 ablation study) ─────────────────────────────────
RESULTS_DIR        = EVALUATION_DIR / "results"
FIGURES_DIR        = EVALUATION_DIR / "figures"
