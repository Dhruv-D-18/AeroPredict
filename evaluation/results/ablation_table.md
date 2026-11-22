| Model | Role | Graph | Temporal | MAE +1 | MAE +2 | MAE +3 | RMSE | PR-AUC | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | baseline | ✗ | last slot | 0.1354 | 0.1559 | 0.1489 | 0.2185 | 0.103 | 0.129 | 0.298 | 0.180 |
| Historical mean | baseline | ✗ | no | 0.1300 | 0.1307 | 0.1306 | 0.1649 | 0.132 | 0.141 | 0.329 | 0.197 |
| Gradient boosting (ML) | Model 1 | ✗ | lag features | 0.1036 | 0.1077 | 0.1084 | 0.1436 | 0.318 | 0.341 | 0.407 | 0.371 |
| DNN V2 (Phase 1)* | Model 2 | ✗ | no | 0.0826 | 0.0828 | 0.0828 | 0.1203 | 0.588 | 0.542 | 0.611 | 0.574 |
| GRU | Model 3 | ✗ | yes | 0.1033 | 0.1090 | 0.1104 | 0.1456 | 0.301 | 0.312 | 0.387 | 0.345 |
| LSTM | Model 3 | ✗ | yes | 0.1039 | 0.1082 | 0.1096 | 0.1457 | 0.283 | 0.293 | 0.383 | 0.332 |
| ST-GNN (GCN + GRU) | Model 4 | ✓ | yes | 0.0998 | 0.1045 | 0.1058 | 0.1428 | 0.308 | 0.325 | 0.388 | 0.354 |
| ST-GNN (GAT + GRU) | Model 4 | ✓ | yes | 0.0989 | 0.1022 | 0.1032 | 0.1443 | 0.284 | 0.311 | 0.382 | 0.343 |

* DNN V2 was trained in Phase 1 on a random 95 % of all 2023 flights, so it has seen most December flights — its score here is optimistic (in-sample).
