# training/phase2_st_gnn/ — AeroPredict 2.0 (Phase 2)

Run from the project root, in this order:

| Step | Command | Output |
| :-- | :--- | :--- |
| 1 | `python training/phase2_st_gnn/build_graph.py` | `data/graph/graph.npz`, `airports.json`, `test_flights.csv.gz` |
| 2 | `python training/phase2_st_gnn/train_spatiotemporal.py --model stgnn_gcn` | `models/st_gnn/st_gnn_model.pt` (served by the app) |
| 2b | `... --model stgnn_gat` · `--model gru` · `--model lstm` | other ablation models |
| 3 | `python training/phase2_st_gnn/propagation_analysis.py` | `data/graph/propagation.json` |
| 4 | `python evaluation/run_ablation.py` | `evaluation/results/`, `evaluation/figures/` |

## What each piece is

* **Airport network graph** (`build_graph.py`) — 350 airports (nodes), 6,576 directed routes (edges)
  with flight count, mean flight time and great-circle distance. Built from
  `data/processed/preprocessed_flight_data_final.csv` (6.55 M flights, 2023).
* **Time series** — every airport, every time slot (Morning 06–12, Afternoon 12–18, Evening 18–22,
  Night 22–06) for all 365 days → 1,460 steps. The processed file has no departure hour,
  so one step ≈ 6 h; forecasts are for the next **1, 2 and 3 slots**.
* **Target** — airport departure delay rate (share of flights arriving 15+ min late).
* **Models** (`core/st_models.py`, pure PyTorch — no extra graph library needed)
  * `gru`, `lstm` — temporal only (per-airport RNN), ablation Model 3
  * `stgnn_gcn` — GCN (Kipf & Welling) at each time step → GRU → GCN, ablation Model 4
  * `stgnn_gat` — same with graph attention (GAT) instead of GCN
  * every model outputs a mean **and** a variance (Gaussian NLL) → 95 % uncertainty bands
* **Split by time** — train Jan–Sep 2023 · validation Oct–Nov · test December 2023.
* **Propagation analysis** — spillover between airports connected by a route vs unconnected
  pairs at similar distances (controls for regional weather).

## GPU

The training script uses CUDA automatically. On an RTX 2050 (4 GB):
`stgnn_gcn` ≈ 1–2 min, `stgnn_gat` ≈ 5–10 min (batch 4), `gru`/`lstm` < 1 min.
Check with `python -c "import torch; print(torch.cuda.is_available())"`.
