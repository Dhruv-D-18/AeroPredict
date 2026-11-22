# models/ — trained model files

| Folder | What it holds | Made by |
| :--- | :--- | :--- |
| `dnn_v2/` | **Phase 1 primary model** — multi-task DNN V2 (`dnn_v2_model.pt`, `dnn_v2_preprocessor.pkl`, `dnn_v2_metrics.json`) | `training/phase1_dnn/train_dnn_v2.py` |
| `dnn_v1/` | Phase 1 baseline DNN trained on 50% of the data | `training/phase1_dnn/train_dnn.py` |
| `classic_ml/` | Semester 6 fallbacks — SVM, GBT regressor, scalers (+ optional Spark `gbt_full_pipeline/`) | `training/legacy_sem6/` |
| `st_gnn/` | **Phase 2** — ST-GNN `st_gnn_model.pt` (GCN + GRU, served by the app) and `st_gnn_gat_model.pt` (GAT + GRU) | `training/phase2_st_gnn/train_spatiotemporal.py` |
| `temporal/` | **Phase 2** — GRU / LSTM baselines (`gru_model.pt`, `lstm_model.pt`) | `training/phase2_st_gnn/train_spatiotemporal.py` |

The web app loads these automatically (DNN V2 → DNN V1 → Spark GBT → SVM).
All paths are defined in `core/project_paths.py`.
