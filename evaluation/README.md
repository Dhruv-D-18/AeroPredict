# evaluation/ — Phase 2 ablation study (Module 5)

```bash
python evaluation/run_ablation.py
```

Every model gets the **same task on the same data**: forecast each airport's departure delay
rate for the next 1, 2 and 3 time slots on the held-out **December 2023** windows.

| Model | Architecture | Spatial graph? | Temporal? |
| :--- | :--- | :---: | :---: |
| Persistence / historical mean | naive baselines | ✗ | ✗ |
| Model 1 · Classic ML | scikit-learn gradient boosting on lag features | ✗ | lag features |
| Model 2 · Phase 1 DNN V2 | per-flight multi-task DNN, averaged per airport-slot | ✗ | ✗ |
| Model 3 · Temporal only | GRU / LSTM | ✗ | ✓ |
| Model 4 · AeroPredict 2.0 | ST-GNN (GCN or GAT + GRU) | ✓ | ✓ |

Metrics: MAE & RMSE per horizon; disruption alerts (slot with ≥ 5 departures and ≥ 40 % delayed):
PR-AUC, Precision, Recall, F1 (threshold chosen on the validation months).

> **DNN V2 caveat:** it was trained in Phase 1 on a random 95 % of *all* 2023 flights, so it has
> already seen most December flights — its score in this table is optimistic (in-sample).

Outputs
- `results/ablation_results.json`, `ablation_table.md`, `ablation_table.csv`
- `results/<model>_training.json` — loss curves + test metrics from training
- `figures/mae_by_horizon.png`, `pr_curves.png`, `loss_curves.png`, `propagation_spillover.png`
