# core/ — shared Python code

| File | Used for |
| :--- | :--- |
| `project_paths.py` | **Every file location in the project.** Change a folder here, not in scripts. |
| `dnn_common.py` | PyTorch networks (DNN V1 / V2) + load & predict helpers |
| `dnn_features_v2.py` | DNN V2 feature engineering (embeddings, weather, target encodings) |
| `weather_api.py` | Open-Meteo 15-day forecast client (cached, fail-soft) |
| `st_data.py` | Phase 2 — turns `data/graph/graph.npz` into model inputs (features, adjacency, time windows) |
| `st_models.py` | Phase 2 — GRU/LSTM, ST-GNN (GCN/GAT + GRU), uncertainty head |
| `st_inference.py` | Phase 2 — network forecasts, what-if simulator, propagation footprint (used by the web app) |
