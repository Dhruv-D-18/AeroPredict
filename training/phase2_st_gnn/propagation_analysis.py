"""
AeroPredict 2.0 — Step 3: delay propagation analysis across connected airports
=============================================================================
    python training/phase2_st_gnn/propagation_analysis.py

Question: when airport A has unusual delays in slot t, do airports that A flies to
show unusual delays in slot t+1 — more than airports that are NOT connected to A?

Method (per directed route A → B)
* residual  r_A(t) = delay rate at A in slot t − A's usual rate for that slot type
                     (only slots with ≥ 5 departures; usual rate from Jan–Sep)
* spillover = Pearson correlation of r_A(t) with r_B(t + 1)
* uplift    = mean r_B(t + 1) in slots right after A was disrupted (rate ≥ 40 %)
* control   = the same statistics for airport pairs with NO direct route,
              matched by distance band, so regional weather is not counted as propagation.

Output: data/graph/propagation.json (used by the web app's propagation panel and the report)
"""
# ── locate project folders (see core/project_paths.py) ──
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "project_paths.py").exists())
_sys.path.insert(0, str(_ROOT / "core"))
import project_paths as P

import json

import numpy as np
from scipy import stats

from st_data import GraphData, TRAIN_END_DAY, DISRUPTION_RATE

MIN_N = 5            # departures needed for a slot's rate to count
MIN_ROUTE = 300      # yearly flights for a route to be analysed
MIN_PAIRS = 150      # overlapping valid slots needed for a correlation
BANDS = [(0, 800), (800, 1600), (1600, 99999)]


def haversine(lat1, lon1, lat2, lon2):
    r = np.radians
    a = np.sin(r(lat2 - lat1) / 2) ** 2 + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(r(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def pair_stats(res, dis, i, j):
    a, b = res[:-1, i], res[1:, j]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_PAIRS:
        return None
    corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
    d = dis[:-1, i] & np.isfinite(b)
    uplift = float(np.nanmean(b[d])) if d.sum() >= 10 else None
    return corr, uplift, int(ok.sum()), int(d.sum())


def main():
    g = GraphData()
    N, T = g.N, g.T
    slot = np.arange(T) % 4
    tr = (np.arange(T) // 4) < TRAIN_END_DAY
    rate = np.where(g.dep_n >= MIN_N, g.dep_d / np.maximum(g.dep_n, 1), np.nan)
    usual = np.zeros((4, N))
    for s in range(4):
        sel = tr & (slot == s)
        usual[s] = np.nan_to_num(np.nanmean(rate[sel], axis=0), nan=0.19)
    res = rate - usual[slot]
    dis = (rate >= DISRUPTION_RATE)

    busy = g.dep_n.sum(0) >= 2000
    connected = np.zeros((N, N), bool)
    connected[g.edge_src, g.edge_dst] = True
    dist = haversine(g.lat[:, None], g.lon[:, None], g.lat[None, :], g.lon[None, :])

    routes = []
    for e in np.nonzero(g.edge_count >= MIN_ROUTE)[0]:
        i, j = int(g.edge_src[e]), int(g.edge_dst[e])
        if not (busy[i] and busy[j]):
            continue
        s = pair_stats(res, dis, i, j)
        if s is None:
            continue
        routes.append({"from": g.airports[i], "to": g.airports[j], "flights": int(g.edge_count[e]),
                       "distance_km": round(float(dist[i, j])), "spillover": round(s[0], 4),
                       "uplift_pp": None if s[1] is None else round(100 * s[1], 2),
                       "slots": s[2], "disrupted_slots": s[3]})

    # distance-matched control: unconnected busy pairs
    rng = np.random.default_rng(42)
    busy_idx = np.nonzero(busy)[0]
    control = {k: [] for k in range(len(BANDS))}
    route_band = [next(k for k, (lo, hi) in enumerate(BANDS) if lo <= r["distance_km"] < hi) for r in routes]
    need = {k: max(200, 2 * route_band.count(k)) for k in range(len(BANDS))}
    tries = 0
    while any(len(control[k]) < need[k] for k in control) and tries < 400_000:
        tries += 1
        i, j = rng.choice(busy_idx, 2, replace=False)
        if connected[i, j] or connected[j, i]:
            continue
        k = next(k for k, (lo, hi) in enumerate(BANDS) if lo <= dist[i, j] < hi)
        if len(control[k]) >= need[k]:
            continue
        s = pair_stats(res, dis, i, j)
        if s is not None:
            control[k].append(s)

    summary = []
    for k, (lo, hi) in enumerate(BANDS):
        rc = [r["spillover"] for r, b in zip(routes, route_band) if b == k]
        ru = [r["uplift_pp"] for r, b in zip(routes, route_band) if b == k and r["uplift_pp"] is not None]
        cc = [c[0] for c in control[k]]
        cu = [100 * c[1] for c in control[k] if c[1] is not None]
        if not rc or not cc:
            continue
        p = float(stats.mannwhitneyu(rc, cc, alternative="greater").pvalue)
        summary.append({"band_km": f"{lo}-{hi if hi < 99999 else '∞'}",
                        "connected_pairs": len(rc), "control_pairs": len(cc),
                        "connected_mean_spillover": round(float(np.mean(rc)), 4),
                        "control_mean_spillover": round(float(np.mean(cc)), 4),
                        "connected_mean_uplift_pp": round(float(np.mean(ru)), 2) if ru else None,
                        "control_mean_uplift_pp": round(float(np.mean(cu)), 2) if cu else None,
                        "p_value_connected_greater": p})

    all_c = [r["spillover"] for r in routes]
    all_k = [c[0] for k in control for c in control[k]]
    overall = {"routes_analysed": len(routes), "control_pairs": len(all_k),
               "connected_mean_spillover": round(float(np.mean(all_c)), 4),
               "control_mean_spillover": round(float(np.mean(all_k)), 4),
               "p_value": float(stats.mannwhitneyu(all_c, all_k, alternative="greater").pvalue)}

    per_airport = {}
    for r in routes:
        per_airport.setdefault(r["from"], []).append(r)
    for k in per_airport:
        per_airport[k] = sorted(per_airport[k], key=lambda r: -r["spillover"])[:12]

    out = {"method": __doc__.split("Method")[1].split("Output")[0].strip(),
           "overall": overall, "by_distance": summary,
           "top_routes": sorted(routes, key=lambda r: -r["spillover"])[:25],
           "per_airport": per_airport}
    with open(P.GRAPH_DIR / "propagation.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=1)
    print(json.dumps(overall, indent=2))
    for s in summary:
        print(s)
    print(f"saved {P.GRAPH_DIR / 'propagation.json'}")


if __name__ == "__main__":
    main()
