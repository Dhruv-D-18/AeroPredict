# data/graph/ — Phase 2 graph data

Created by `training/phase2_st_gnn/build_graph.py` (≈ 30 s):

| File | Contents |
| :--- | :--- |
| `graph.npz` | 350 airports × 1,460 time slots: departures, delayed departures, arrivals, daily weather; 6,576 route edges (count, duration, distance) |
| `airports.json` | node index → IATA code, coordinates, yearly flights |
| `test_flights.csv.gz` | decoded Oct–Dec 2023 flights, used to score DNN V2 in the ablation study |
| `propagation.json` | delay spillover per route + connected-vs-unconnected statistics (`propagation_analysis.py`) |
