"""
AeroPredict 2.0 — Step 2: train the temporal and spatio-temporal models
=======================================================================
    python training/phase2_st_gnn/train_spatiotemporal.py --model stgnn_gcn   # ST-GNN (GCN + GRU)  → main model
    python training/phase2_st_gnn/train_spatiotemporal.py --model stgnn_gat   # ST-GNN (GAT + GRU)
    python training/phase2_st_gnn/train_spatiotemporal.py --model gru         # temporal-only GRU baseline
    python training/phase2_st_gnn/train_spatiotemporal.py --model lstm        # temporal-only LSTM baseline

Uses the GPU automatically (CUDA) when available, e.g. an RTX 2050.
Needs data/graph/graph.npz  →  run build_graph.py first.

Outputs
  models/st_gnn/st_gnn_model.pt        (stgnn_gcn — served by the web app)
  models/st_gnn/st_gnn_gat_model.pt    (stgnn_gat)
  models/temporal/gru_model.pt, lstm_model.pt
  evaluation/results/<model>_training.json   loss curves + test metrics
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
from datetime import datetime, timezone

import numpy as np
import torch

from st_data import GraphData, T_IN, HORIZONS, F_HIST, F_FUT
from st_models import build_model, gaussian_nll

OUT = {
    "stgnn_gcn": P.ST_GNN_DIR / "st_gnn_model.pt",
    "stgnn_gat": P.ST_GNN_DIR / "st_gnn_gat_model.pt",
    "gru": P.MODELS_DIR / "temporal" / "gru_model.pt",
    "lstm": P.MODELS_DIR / "temporal" / "lstm_model.pt",
}
DEFAULT_BATCH = {"stgnn_gcn": 16, "stgnn_gat": 4, "gru": 32, "lstm": 32}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_t(arrs, device):
    return [torch.from_numpy(a).to(device) for a in arrs]


@torch.no_grad()
def evaluate(model, data, ts, adj, mask, device, bs):
    model.eval()
    preds, sds, ys, ms, ns = [], [], [], [], []
    for i in range(0, len(ts), bs):
        x, f, y, m, n = data.batch(ts[i:i + bs])
        xt, ft = to_t((x, f), device)
        mean, logvar = model(xt, ft, adj, mask)
        preds.append(mean.cpu().numpy()); sds.append(np.exp(0.5 * logvar.cpu().numpy()))
        ys.append(y); ms.append(m); ns.append(n)
    return (np.concatenate(preds), np.concatenate(sds), np.concatenate(ys),
            np.concatenate(ms), np.concatenate(ns))


def metrics(pred, sd, y, m, n):
    out = {}
    for h in range(HORIZONS):
        sel = m[:, h] > 0
        err = pred[:, h][sel] - y[:, h][sel]
        cover = np.abs(err) <= 1.96 * sd[:, h][sel]
        out[f"h{h + 1}"] = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
                            "coverage_95": float(cover.mean()), "cells": int(sel.sum())}
    sel = m > 0
    err = pred[sel] - y[sel]
    out["all"] = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stgnn_gcn", choices=list(OUT))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          (args.device if args.device != "auto" else "cpu"))
    bs = args.batch_size or DEFAULT_BATCH[args.model]
    log(f"model={args.model}  device={device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "") + f"  batch={bs}")

    data = GraphData()
    tr, va, te = data.window_starts("train"), data.window_starts("val"), data.window_starts("test")
    log(f"graph: {data.N} airports, {len(data.edge_src):,} routes, {data.T} slots  |  "
        f"windows train {len(tr)} · val {len(va)} · test {len(te)}")
    adj = torch.from_numpy(data.adj).to(device)
    mask = torch.from_numpy(data.adj_mask).to(device)

    model = build_model(args.model, data.N, F_HIST, F_FUT, HORIZONS, args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"parameters: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    best, best_state, wait, history = 1e9, None, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr)
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), bs):
            x, f, y, m, n = data.batch(perm[i:i + bs])
            w = m * np.minimum(n, 30.0) / 30.0                      # trust busy slots more
            xt, ft, yt, wt = to_t((x, f, y, w.astype(np.float32)), device)
            mean, logvar = model(xt, ft, adj, mask)
            loss = gaussian_nll(mean, logvar, yt, wt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(x); cnt += len(x)
        pv, sv, yv, mv, nv = evaluate(model, data, va, adj, mask, device, bs)
        val = metrics(pv, sv, yv, mv, nv)
        val_mae = val["all"]["mae"]
        sched.step(val_mae)
        history.append({"epoch": epoch, "train_nll": tot / cnt, "val_mae": val_mae, "val_rmse": val["all"]["rmse"],
                        "lr": opt.param_groups[0]["lr"]})
        log(f"epoch {epoch:3d}  train NLL {tot / cnt:+.4f}  val MAE {val_mae:.4f}  RMSE {val['all']['rmse']:.4f}")
        if val_mae < best - 1e-5:
            best, wait = val_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                log("early stopping"); break

    model.load_state_dict(best_state)
    pt, st, yt_, mt, nt = evaluate(model, data, te, adj, mask, device, bs)
    test = metrics(pt, st, yt_, mt, nt)
    log(f"TEST (December 2023)  MAE h1 {test['h1']['mae']:.4f} · h2 {test['h2']['mae']:.4f} · h3 {test['h3']['mae']:.4f}"
        f"   95% band coverage h1 {test['h1']['coverage_95']:.2f}")

    cfg = {"model": args.model, "n_nodes": data.N, "f_hist": F_HIST, "f_fut": F_FUT, "horizons": HORIZONS,
           "hidden": args.hidden, "t_in": T_IN, "parameters": n_params,
           "trained_at": datetime.now(timezone.utc).isoformat(), "device": str(device),
           "epochs_run": len(history), "best_val_mae": best, "train_seconds": round(time.time() - t0, 1)}
    out = OUT[args.model]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg}, out)
    P.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(P.RESULTS_DIR / f"{args.model}_training.json", "w", encoding="utf-8") as fp:
        json.dump({"config": cfg, "history": history, "test": test}, fp, indent=2)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
