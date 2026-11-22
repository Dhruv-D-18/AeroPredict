# training/ — scripts that build the models

| Folder | Contents | Status |
| :--- | :--- | :--- |
| `phase1_dnn/` | `train_dnn_v2.py` (primary, 100% data) · `train_dnn.py` (V1 baseline) | Done (Phase 1) |
| `phase2_st_gnn/` | ST-GNN: airport graph + GCN/GAT + GRU, 1h/2h/3h forecasting | To build (Phase 2) |
| `legacy_sem6/` | Semester 6 Big Data stack: Spark/HDFS preprocessing, GBT, SVM, GBT regressor, benchmark | Kept for the ablation baseline |

Run every script from the project root, e.g.

```bash
python training/phase1_dnn/train_dnn_v2.py                    # full run → models/dnn_v2/
python training/phase1_dnn/train_dnn_v2.py --sample-frac 0.02 # quick smoke test
```

> `legacy_sem6/phase2_preprocessing.py` is the **Semester 6 Spark** pre-processing step,
> not part of the Deep Learning Phase 2.
