"""
AeroPredict 2.0 — Module 5: ablation study (ML vs DNN vs RNN vs ST-GNN)
=======================================================================
    python evaluation/run_ablation.py

Task (identical for every model): forecast each airport's departure delay rate
for the next 1, 2 and 3 time slots, on the held-out December 2023 test windows.

| Model                        | Spatial graph | Temporal | Notes |
| 0a Persistence               | no  | last slot | "tomorrow looks like today" |
| 0b Historical mean           | no  | no  | airport × slot average (Jan–Sep) |
| 1  Gradient boosting (ML)    | no  | lag features | scikit-learn HistGradientBoosting |
| 2  DNN V2 (Phase 1)          | no  | no  | per-flight model, averaged per airport-slot |
| 3  GRU / LSTM                | no  | yes | training/phase2_st_gnn --model gru|lstm |
| 4  ST-GNN (GCN/GAT + GRU)    | yes | yes | AeroPredict 2.0 |

Metrics: MAE & RMSE per horizon (slots with ≥ 3 departures); disruption alerts
(slot with ≥ 5 departures and ≥ 40 % delayed): PR-AUC, and Precision / Recall / F1
at the threshold that maximises F1 on the validation months (Oct–Nov).

Outputs: evaluation/results/ablation_results.json, ablation_table.md, ablation_table.csv
         evaluation/figures/*.png
"""
# ── locate project folders (see core/project_paths.py) ──
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "project_paths.py").exists())
_sys.path.insert(0, str(_ROOT / "core"))
import project_paths as P

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve

from st_data import (GraphData, HORIZONS, T_IN, DISRUPTION_RATE, DISRUPTION_MIN_FLIGHTS,
                     TRAIN_END_DAY)

FIG = P.FIGURES_DIR
RES = P.RESULTS_DIR


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── ground truth for a set of windows ──────────────────────────────────
def truth(g, ts):
    idx = ts[:, None] + 1 + np.arange(HORIZONS)[None, :]          # [W, H]
    n = g.dep_n[idx]                                               # [W, H, N]
    y = np.nan_to_num(g.rate[idx], nan=0.0)
    return y, n


# ── baselines ───────────────────────────────────────────────────────────
def usual_rates(g):
    tr = (np.arange(g.T) // 4) < TRAIN_END_DAY
    slot = np.arange(g.T) % 4
    u = np.zeros((4, g.N))
    for s in range(4):
        sel = tr & (slot == s)
        num, den = g.dep_d[sel].sum(0), g.dep_n[sel].sum(0)
        u[s] = (num + 0.19 * 20) / (den + 20)
    return u


def persistence(g, ts):
    last = np.where(g.dep_n[ts] > 0, g.dep_d[ts] / np.maximum(g.dep_n[ts], 1), g.ap_rate[None, :])
    return np.repeat(last[:, None, :], HORIZONS, axis=1)


def historical(g, ts, usual):
    idx = ts[:, None] + 1 + np.arange(HORIZONS)[None, :]
    return usual[idx % 4]                                          # [W, H, N]


def gbt_features(g, ts, usual):
    rows = []
    rate_f = np.where(g.dep_n > 0, g.dep_d / np.maximum(g.dep_n, 1), g.ap_rate[None, :])
    for h in range(HORIZONS):
        tgt = ts + 1 + h
        lags = np.stack([rate_f[ts - k] for k in range(T_IN)], -1)                 # [W, N, T_IN]
        vols = np.stack([np.log1p(g.dep_n[ts - k]) for k in range(T_IN)], -1)
        wx = g.weather_day[tgt // 4]                                                # [W, N, 4]
        W = len(ts)
        feat = np.concatenate([
            lags, vols, wx,
            np.broadcast_to(usual[tgt % 4][:, :, None], (W, g.N, 1)),
            np.broadcast_to((tgt % 4)[:, None, None], (W, g.N, 1)).astype(float),
            np.broadcast_to(np.full((W, 1, 1), h, float), (W, g.N, 1)),
            (g.typical_ln[tgt % 4, g._dow_of_t()[tgt]])[:, :, None],             # typical volume (known in advance)
        ], -1)
        rows.append(feat)
    return np.stack(rows, 1)                                                        # [W, H, N, F]


def gbt_model(g, tr, va, te, usual, seed=42):
    Xtr = gbt_features(g, tr, usual)
    ytr, ntr = truth(g, tr)
    m = ntr >= 3
    X, y, w = Xtr[m], ytr[m], np.minimum(ntr[m], 30) / 30
    rng = np.random.default_rng(seed)
    if len(y) > 400_000:
        k = rng.choice(len(y), 400_000, replace=False); X, y, w = X[k], y[k], w[k]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          random_state=seed)
    model.fit(X, y, sample_weight=w)
    out = {}
    for name, ts in (("val", va), ("test", te)):
        F = gbt_features(g, ts, usual)
        out[name] = np.clip(model.predict(F.reshape(-1, F.shape[-1])).reshape(F.shape[:-1]), 0, 1)
    return out


def dnn_v2(g, va, te):
    """Per-flight DNN V2 probabilities averaged per airport-slot."""
    path = P.GRAPH_DIR / "test_flights.csv.gz"
    if not (path.exists() and P.DNN_V2_MODEL.exists()):
        log("DNN V2 skipped (needs models/dnn_v2 and data/graph/test_flights.csv.gz)")
        return None
    import torch
    from dnn_common import load_dnn_v2
    from dnn_features_v2 import FeatureBuilder
    model, state = load_dnn_v2(P.DNN_V2_MODEL, P.DNN_V2_PREP)
    fb = FeatureBuilder.from_state(state)
    df = pd.read_csv(path)
    log(f"DNN V2: scoring {len(df):,} flights (Oct–Dec 2023)")
    probs = np.zeros(len(df), np.float32)
    with torch.no_grad():
        for i in range(0, len(df), 200_000):
            part = df.iloc[i:i + 200_000].copy()
            X, cat, _, _ = fb.build(part, with_target=False)
            fb.transform(X)
            logits, _ = model(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(cat.astype(np.int64)))
            probs[i:i + len(part)] = torch.sigmoid(logits).numpy()
    day = (pd.to_datetime(df["FlightDate"]) - pd.Timestamp(str(g.days[0]))).dt.days.to_numpy()
    slot = df["DepTime_label"].map({"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}).to_numpy()
    t = day * 4 + slot
    ap = df["Dep_Airport"].map(g.index).to_numpy()
    s = np.zeros((g.T, g.N)); c = np.zeros((g.T, g.N))
    np.add.at(s, (t, ap), probs); np.add.at(c, (t, ap), 1)
    cell = np.where(c > 0, s / np.maximum(c, 1), g.ap_rate[None, :])
    out = {}
    for name, ts in (("val", va), ("test", te)):
        out[name] = cell[ts[:, None] + 1 + np.arange(HORIZONS)[None, :]]
    return out


def neural(g, va, te, name, path):
    if not path.exists():
        log(f"{name} skipped ({path.name} not trained yet)")
        return None
    import torch
    from st_models import load_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(path, device)
    adj = torch.from_numpy(g.adj).to(device); mask = torch.from_numpy(g.adj_mask).to(device)
    out = {}
    with torch.no_grad():
        for split, ts in (("val", va), ("test", te)):
            ms, ss = [], []
            for i in range(0, len(ts), 8):
                x, f, _, _, _ = g.batch(ts[i:i + 8])
                mean, logvar = model(torch.from_numpy(x).to(device), torch.from_numpy(f).to(device), adj, mask)
                ms.append(mean.cpu().numpy()); ss.append(np.exp(0.5 * logvar.cpu().numpy()))
            out[split] = np.concatenate(ms); out[split + "_sd"] = np.concatenate(ss)
    out["params"] = cfg.get("parameters")
    return out


# ── metrics ─────────────────────────────────────────────────────────────
def score(pred, y, n, pred_val, y_val, n_val, sd=None):
    r = {}
    for h in range(HORIZONS):
        m = n[:, h] >= 3
        e = pred[:, h][m] - y[:, h][m]
        r[f"mae_h{h + 1}"] = float(np.abs(e).mean())
        r[f"rmse_h{h + 1}"] = float(np.sqrt((e ** 2).mean()))
        if sd is not None:
            r[f"cover95_h{h + 1}"] = float((np.abs(e) <= 1.96 * sd[:, h][m]).mean())
    m = n >= 3
    r["mae"] = float(np.abs(pred[m] - y[m]).mean())
    r["rmse"] = float(np.sqrt(((pred[m] - y[m]) ** 2).mean()))
    # disruption alerts
    d, dv = n >= DISRUPTION_MIN_FLIGHTS, n_val >= DISRUPTION_MIN_FLIGHTS
    lab, s = (y[d] >= DISRUPTION_RATE).astype(int), pred[d]
    lab_v, s_v = (y_val[dv] >= DISRUPTION_RATE).astype(int), pred_val[dv]
    r["pr_auc"] = float(average_precision_score(lab, s))
    pr, rc, th = precision_recall_curve(lab_v, s_v)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    thr = float(th[np.argmax(f1[:-1])]) if len(th) else 0.5
    pred_lab = s >= thr
    tp = int((pred_lab & (lab == 1)).sum()); fp = int((pred_lab & (lab == 0)).sum()); fn = int((~pred_lab & (lab == 1)).sum())
    P_ = tp / max(tp + fp, 1); R_ = tp / max(tp + fn, 1)
    r.update({"alert_threshold": thr, "precision": P_, "recall": R_, "f1": 2 * P_ * R_ / max(P_ + R_, 1e-9),
              "disruption_base_rate": float(lab.mean())})
    return r, (lab, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dnn", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    g = GraphData()
    tr, va, te = g.window_starts("train"), g.window_starts("val"), g.window_starts("test")
    y, n = truth(g, te); yv, nv = truth(g, va)
    usual = usual_rates(g)
    log(f"test windows {len(te)} (December 2023) · validation windows {len(va)}")

    entries = [
        ("Persistence", "baseline", False, "last slot", {"val": persistence(g, va), "test": persistence(g, te)}),
        ("Historical mean", "baseline", False, "no", {"val": historical(g, va, usual), "test": historical(g, te, usual)}),
    ]
    log("training gradient boosting (Model 1) …")
    entries.append(("Gradient boosting (ML)", "Model 1", False, "lag features", gbt_model(g, tr, va, te, usual)))
    if not args.skip_dnn:
        d = dnn_v2(g, va, te)
        if d is not None:
            entries.append(("DNN V2 (Phase 1)*", "Model 2", False, "no", d))
    for name, key, path, spatial in (("GRU", "gru", P.MODELS_DIR / "temporal" / "gru_model.pt", False),
                                     ("LSTM", "lstm", P.MODELS_DIR / "temporal" / "lstm_model.pt", False),
                                     ("ST-GNN (GCN + GRU)", "stgnn_gcn", P.ST_GNN_MODEL, True),
                                     ("ST-GNN (GAT + GRU)", "stgnn_gat", P.ST_GNN_DIR / "st_gnn_gat_model.pt", True)):
        out = neural(g, va, te, name, path)
        if out is not None:
            entries.append((name, "Model 4" if spatial else "Model 3", spatial, "yes", out))

    results, curves = [], {}
    for name, role, spatial, temporal, out in entries:
        r, cur = score(out["test"], y, n, out["val"], yv, nv, out.get("test_sd"))
        r.update({"model": name, "role": role, "spatial_graph": spatial, "temporal": temporal,
                  "parameters": out.get("params")})
        results.append(r); curves[name] = cur
        log(f"{name:24s} MAE h1 {r['mae_h1']:.4f} h2 {r['mae_h2']:.4f} h3 {r['mae_h3']:.4f} · "
            f"PR-AUC {r['pr_auc']:.3f} · F1 {r['f1']:.3f}")

    best = min(results, key=lambda r: r["mae"])
    meta = {"task": "airport departure delay-rate forecast, horizons +1/+2/+3 slots (~6 h each)",
            "test_period": "2023-12-01 … 2023-12-31", "validation_period": "2023-10-01 … 2023-11-30",
            "train_period": "2023-01-01 … 2023-09-30", "test_windows": int(len(te)),
            "disruption_definition": f"≥ {DISRUPTION_MIN_FLIGHTS} departures and ≥ {int(DISRUPTION_RATE * 100)} % delayed",
            "note_dnn": "* DNN V2 was trained in Phase 1 on a random 95 % of all 2023 flights, so it has seen "
                        "most December flights — its score here is optimistic (in-sample).",
            "best_model_by_mae": best["model"]}
    with open(RES / "ablation_results.json", "w", encoding="utf-8") as fp:
        json.dump({"meta": meta, "results": results}, fp, indent=2)

    cols = ["model", "role", "spatial_graph", "temporal", "mae_h1", "mae_h2", "mae_h3", "rmse", "pr_auc",
            "precision", "recall", "f1"]
    df = pd.DataFrame(results)[cols]
    df.to_csv(RES / "ablation_table.csv", index=False)
    lines = ["| Model | Role | Graph | Temporal | MAE +1 | MAE +2 | MAE +3 | RMSE | PR-AUC | Precision | Recall | F1 |",
             "| :--- | :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['model']} | {r['role']} | {'✓' if r['spatial_graph'] else '✗'} | {r['temporal']} | "
                     f"{r['mae_h1']:.4f} | {r['mae_h2']:.4f} | {r['mae_h3']:.4f} | {r['rmse']:.4f} | "
                     f"{r['pr_auc']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    (RES / "ablation_table.md").write_text("\n".join(lines) + "\n\n" + meta["note_dnn"] + "\n", encoding="utf-8")
    make_figures(results, curves)
    log(f"best by MAE: {best['model']}   → {RES / 'ablation_results.json'}")


def make_figures(results, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#B8C4D0", "#9AA8B8", "#E2A53B", "#C2493F", "#8CC4EC", "#5FA0D6", "#1769AA", "#12304A"]

    # MAE per horizon
    fig, ax = plt.subplots(figsize=(9, 4.2))
    k = len(results); w = 0.8 / k
    for i, r in enumerate(results):
        ax.bar(np.arange(3) + i * w - 0.4 + w / 2, [r["mae_h1"], r["mae_h2"], r["mae_h3"]], w,
               label=r["model"], color=colors[i % len(colors)])
    ax.set_xticks(range(3)); ax.set_xticklabels(["+1 slot", "+2 slots", "+3 slots"])
    ax.set_ylabel("MAE (delay rate)"); ax.set_title("Forecast error by horizon — December 2023 test")
    ax.legend(fontsize=8, ncol=2, frameon=False); fig.tight_layout()
    fig.savefig(FIG / "mae_by_horizon.png", dpi=160); plt.close(fig)

    # PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, r in enumerate(results):
        lab, s = curves[r["model"]]
        pr, rc, _ = precision_recall_curve(lab, s)
        ax.plot(rc, pr, label=f"{r['model']} (AP {r['pr_auc']:.3f})", color=colors[i % len(colors)], lw=1.8)
    ax.axhline(results[0]["disruption_base_rate"], color="#9AA8B8", ls=":", lw=1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Disruption alerts — precision/recall")
    ax.legend(fontsize=7.5, frameon=False); fig.tight_layout()
    fig.savefig(FIG / "pr_curves.png", dpi=160); plt.close(fig)

    # training curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, key in enumerate(["gru", "lstm", "stgnn_gcn", "stgnn_gat"]):
        p = RES / f"{key}_training.json"
        if p.exists():
            h = json.loads(p.read_text())["history"]
            ax.plot([e["epoch"] for e in h], [e["val_mae"] for e in h], label=key, color=colors[4 + i], lw=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MAE"); ax.set_title("Training curves")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "loss_curves.png", dpi=160); plt.close(fig)

    # propagation
    pp = P.GRAPH_DIR / "propagation.json"
    if pp.exists():
        bands = json.loads(pp.read_text())["by_distance"]
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(bands))
        ax.bar(x - 0.2, [b["connected_mean_spillover"] for b in bands], 0.4, label="Connected by a route", color="#1769AA")
        ax.bar(x + 0.2, [b["control_mean_spillover"] for b in bands], 0.4, label="Not connected (control)", color="#B8C4D0")
        ax.set_xticks(x); ax.set_xticklabels([b["band_km"] + " km" for b in bands])
        ax.set_ylabel("Delay spillover (corr. t → t+1)"); ax.set_title("Delay propagation across the network")
        ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "propagation_spillover.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
