# models/st_gnn/ — Phase 2 (AeroPredict 2.0)

* `st_gnn_model.pt` — ST-GNN with GCN + GRU (the web app's Network page uses this one)
* `st_gnn_gat_model.pt` — ST-GNN with graph attention (GAT) + GRU

Train with `python training/phase2_st_gnn/train_spatiotemporal.py --model stgnn_gcn` (or `stgnn_gat`).
