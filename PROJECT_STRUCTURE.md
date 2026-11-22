# AeroPredict — Project Structure

Where every file lives, and where it used to be before the restructure.
**Rule of thumb:** website → `webapp/` · shared code → `core/` · datasets → `data/` · trained models → `models/` ·
scripts that train models → `training/` · one-off utilities → `tools/` · Phase 2 benchmarks → `evaluation/`.

All file paths used by the code are defined once in **`core/project_paths.py`**.

## Quick start

```bash
python run_app.py                                     # website → http://127.0.0.1:5000/
python training/phase1_dnn/train_dnn_v2.py            # retrain the Phase 1 DNN → models/dnn_v2/
```

## Folder map

| Folder | What goes here | Phase |
| :--- | :--- | :--- |
| `webapp/` | Flask app, HTML templates, CSS/JS, chart data | Phase 1 (Phase 2 dashboard features added here) |
| `core/` | Code shared by the app and training scripts | Phase 1 |
| `data/raw/` | Original 2023 datasets | — |
| `data/processed/` | Cleaned / merged datasets | — |
| `data/lookups/` | Small tables loaded when the app starts | Phase 1 |
| `data/weather/` | Open-Meteo calibration & validation | Phase 1 |
| `data/graph/` | Airport graph + hourly series | **Phase 2** |
| `models/dnn_v2/` | Primary DNN checkpoint (served) | Phase 1 |
| `models/dnn_v1/` | Baseline DNN checkpoint | Phase 1 |
| `models/classic_ml/` | SVM / GBT / regressor pickles | Semester 6 |
| `models/st_gnn/` | ST-GNN checkpoints (GCN + GRU, GAT + GRU) | **Phase 2** |
| `models/temporal/` | GRU / LSTM baselines | **Phase 2** |
| `training/phase1_dnn/` | DNN training scripts | Phase 1 |
| `training/phase2_st_gnn/` | Graph building, GRU/LSTM/ST-GNN training, propagation analysis | **Phase 2** |
| `training/legacy_sem6/` | Spark / HDFS / scikit-learn scripts | Semester 6 |
| `tools/` | Weather calibration, lookup rebuilding | Phase 1 |
| `evaluation/` | Ablation study results & figures | **Phase 2** |
| `database/` | PostgreSQL schema / setup | **Phase 2** |
| `docs/` | Phase 2 blueprint, report, viva material | — |

## Phase 2 files (AeroPredict 2.0)

| File | What it does |
| :--- | :--- |
| `training/phase2_st_gnn/build_graph.py` | Airport graph + per-slot time series → `data/graph/graph.npz` |
| `training/phase2_st_gnn/train_spatiotemporal.py` | Trains `gru`, `lstm`, `stgnn_gcn`, `stgnn_gat` (GPU if available) |
| `training/phase2_st_gnn/propagation_analysis.py` | Delay spillover across connected airports → `data/graph/propagation.json` |
| `evaluation/run_ablation.py` | ML vs DNN vs GRU/LSTM vs ST-GNN → `evaluation/results/`, `evaluation/figures/` |
| `core/st_data.py` · `core/st_models.py` · `core/st_inference.py` | Data pipeline · model architectures · web-app inference |
| `webapp/templates/network.html` | Network page: early-warning map, propagation, what-if simulator, model comparison |

## Old location → new location

| Old (root unless noted) | New |
| :--- | :--- |
| `flask_app.py` | `webapp/flask_app.py` (start with `python run_app.py`) |
| `templates/` | `webapp/templates/` |
| `static/` | `webapp/static/` |
| `dnn_common.py` | `core/dnn_common.py` |
| `dnn_features_v2.py` | `core/dnn_features_v2.py` |
| `weather_api.py` | `core/weather_api.py` |
| `dnn_v2_model.pt`, `dnn_v2_preprocessor.pkl`, `dnn_v2_metrics.json` | `models/dnn_v2/` |
| `dnn_model.pt`, `dnn_preprocessor.pkl`, `dnn_metrics.json` | `models/dnn_v1/` |
| `aeropredict_model.pkl`, `scaler.pkl`, `model_columns.pkl` | `models/classic_ml/` |
| `delay_regressor.pkl`, `regressor_scaler.pkl`, `regressor_columns.pkl` | `models/classic_ml/` |
| `gbt_full_pipeline/` (if you have it) | `models/classic_ml/gbt_full_pipeline/` |
| `airport_coords.csv`, `airport_delay_lookup.csv`, `weather_averages.csv` | `data/lookups/` |
| `wx_calibration.json`, `wx_validation.json` | `data/weather/` |
| `flight_delay/airports_geolocation.csv` (and the other raw CSVs) | `data/raw/` |
| `preprocessed_flight_data_final.csv` | `data/processed/` |
| `train_dnn_v2.py`, `train_dnn.py` | `training/phase1_dnn/` |
| `train_model.py`, `train_regressor.py`, `make_gbt_flask_pipeline.py` | `training/legacy_sem6/` |
| `phase2_preprocessing.py` *(Semester 6 Spark step — not DL Phase 2)* | `training/legacy_sem6/` |
| `pyspark_pipeline.py`, `benchmark.py` | `training/legacy_sem6/` |
| `flight_delay/datapreprocessing.py`, `flight_delay/datatraining.py` | `training/legacy_sem6/` |
| `calibrate_open_meteo.py`, `validate_open_meteo.py` | `tools/weather_calibration/` |
| `compute_lookups.py` | `tools/lookups/` |
| — *(new)* | `run_app.py`, `core/project_paths.py`, `PROJECT_STRUCTURE.md`, folder `README.md`s |
| — *(new)* | `docs/phase2_blueprint/next_steps.pdf` |

## Adding a new script

Start any new script (e.g. `training/phase2_st_gnn/train_st_gnn.py`) with:

```python
import sys
from pathlib import Path
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "core" / "project_paths.py").exists())
sys.path.insert(0, str(_ROOT / "core"))
import project_paths as P

edges = P.GRAPH_DIR / "adjacency.npy"
torch.save(model.state_dict(), P.ST_GNN_MODEL)
```
