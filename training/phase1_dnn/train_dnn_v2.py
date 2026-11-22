"""
AeroPredict V2 - Multi-task DNN trained on the FULL dataset
==========================================================
    python train_dnn_v2.py                      # 100% of rows, 20 epochs
    python train_dnn_v2.py --sample-frac 0.02   # quick smoke test

What changed vs ``train_dnn.py`` (V1)
-------------------------------------
* **100% of the dataset** instead of a 50% sample (~6.7M rows).
* **Upgraded preprocessing** (see ``dnn_features_v2.py``): categorical
  embeddings, arrival-airport weather/coordinates, haversine distance,
  cyclical calendar features, leakage-safe smoothed target encodings and
  volume features, plus quantile clipping before standardisation.
* **Wider network** with embeddings: numeric tower + 5 embeddings -> 512-256-128.
* **Tiered evaluation** at the app's real 25% / 40% cut-offs, not just 0.5.
* A small holdout (2.5% val / 2.5% test) is kept so the metrics stay honest.
  Training still sees ~6.4M flights -- 1.9x more than V1.

Outputs (default: project root)
-------------------------------
    dnn_v2_model.pt         weights + architecture config
    dnn_v2_preprocessor.pkl FeatureBuilder state (vocab, weather, rates, scaler)
    dnn_v2_metrics.json     test metrics + history + V1 comparison
"""

from __future__ import annotations

# ── locate project folders (see core/project_paths.py) ──
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "project_paths.py").exists())
_sys.path.insert(0, str(_ROOT / "core"))
import project_paths as P

import argparse
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from dnn_common import DelayNetV2
from dnn_features_v2 import (
    DELAY_CLIP,
    DELAY_THRESHOLD_MIN,
    NUMERIC_COLS,
    FeatureBuilder,
    load_flights,
    log,
)

RAW_PATH = P.RAW_FLIGHTS
WEATHER_PATH = P.RAW_WEATHER
AIRPORTS_PATH = P.RAW_AIRPORTS


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X_num, X_cat, y_cls, y_reg, idx, batch_size, bce, huber, reg_weight):
    model.eval()
    total = 0.0
    probs, regs = [], []
    for start in range(0, len(idx), batch_size):
        bi = idx[start:start + batch_size]
        logits, reg = model(X_num[bi], X_cat[bi])
        mask = y_cls[bi] > 0.5
        reg_loss = huber(reg[mask], y_reg[bi][mask]).mean() if mask.any() else torch.zeros(())
        loss = bce(logits, y_cls[bi]) + reg_weight * reg_loss
        total += loss.item() * len(bi)
        probs.append(torch.sigmoid(logits).numpy())
        regs.append(reg.numpy())
    return total / len(idx), np.concatenate(probs), np.concatenate(regs)


def cls_metrics(y_true, prob, thr):
    pred = (prob >= thr).astype(np.int8)
    return {
        "threshold": thr,
        "flagged_pct": float(pred.mean()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def reg_metrics(y_true, y_pred, mask):
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "mae_minutes": float(mean_absolute_error(yt, yp)),
        "rmse_minutes": float(mean_squared_error(yt, yp) ** 0.5),
        "r2": float(r2_score(yt, yp)),
        "rows": int(mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="Train the V2 multi-task DNN on the full dataset")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="fraction of rows to use (1.0 = whole dataset)")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--hidden", type=int, nargs=3, default=[512, 256, 128])
    ap.add_argument("--dropout", type=float, nargs=3, default=[0.25, 0.15, 0.10])
    ap.add_argument("--val-frac", type=float, default=0.025)
    ap.add_argument("--test-frac", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(P.DNN_V2_DIR))
    args = ap.parse_args()

    start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    log("=" * 70)
    log("  AeroPredict V2 - Multi-task DNN (embeddings + engineered features)")
    log("=" * 70)
    log(f"  Threads    : {os.cpu_count()}   Device: CPU")
    log(f"  Data       : {args.sample_frac:.0%} of the raw dataset")
    log(f"  Epochs     : {args.epochs}   Batch: {args.batch_size}   LR: {args.lr}")

    # ---------------------------------------------------------------- data
    log("\n[1/8] Loading raw flights...")
    df = load_flights(RAW_PATH, args.sample_frac, args.chunksize, args.seed)
    n = len(df)
    if n < 1000:
        raise SystemExit("Not enough rows to train.")

    log("[2/8] Fitting vocabularies + geo/weather context...")
    builder = FeatureBuilder().fit_vocab(df).fit_context(AIRPORTS_PATH, WEATHER_PATH)
    log(f"      airports in vocab : {builder.n_ap - 1}")
    log(f"      categorical sizes : {dict(zip(['Airline', 'Dep_Airport', 'Arr_Airport', 'DepTime_label', 'Distance_type'], builder.cardinalities))}")

    log("[3/8] Engineering features...")
    t0 = time.time()
    X, cat, y_cls, y_delay = builder.build(df)
    log(f"      features: {X.shape[1]} numeric + {cat.shape[1]} embedded "
        f"({cat.shape[1]} x ~{sum(builder.cardinalities) / cat.shape[1]:.0f} levels)")
    log(f"      matrix  : {X.shape[0]:,} x {X.shape[1]}  ({time.time() - t0:.1f}s)")
    log(f"      delayed > {DELAY_THRESHOLD_MIN:.0f} min: {y_cls.mean():.2%}")

    # --------------------------------------------------------------- split
    log("[4/8] Splitting train / val / test (stratified)...")
    idx_all = np.arange(n)
    holdout = args.val_frac + args.test_frac
    train_idx, rest_idx = train_test_split(
        idx_all, test_size=holdout, random_state=args.seed, stratify=y_cls
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=args.test_frac / holdout, random_state=args.seed,
        stratify=y_cls[rest_idx],
    )
    del rest_idx
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    log(f"      train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}")

    # --------------------------------------------- leakage-safe aggregation
    log("[5/8] Fitting smoothed target encodings on TRAIN rows only...")
    t0 = time.time()
    builder.fit_stats(df, y_cls, train_mask)
    del df
    log(f"      prior delay rate: {builder.prior:.4f}   ({time.time() - t0:.1f}s)")

    log("[6/8] Clipping outliers + standardising...")
    builder.fit_scaler(X, train_mask)
    builder.transform(X)
    y_mean = float(y_delay[train_idx].mean(dtype=np.float64))
    y_std = float(y_delay[train_idx].std(dtype=np.float64)) or 1.0
    y_reg = ((y_delay - y_mean) / y_std).astype(np.float32)
    log(f"      regression target: mean {y_mean:.2f} min, std {y_std:.2f} min")

    X_num = torch.from_numpy(X)
    X_cat = torch.from_numpy(cat.astype(np.int64))
    yc_t = torch.from_numpy(y_cls)
    yr_t = torch.from_numpy(y_reg)
    train_t = torch.from_numpy(train_idx.astype(np.int64))
    val_t = torch.from_numpy(val_idx.astype(np.int64))
    test_t = torch.from_numpy(test_idx.astype(np.int64))

    # --------------------------------------------------------------- model
    log("[7/8] Training...")
    model = DelayNetV2(
        n_numeric=X.shape[1],
        cardinalities=builder.cardinalities,
        hidden=tuple(args.hidden),
        dropout=tuple(args.dropout),
    )
    n_params = sum(p.numel() for p in model.parameters())
    emb = ", ".join(f"{c}->{d}" for c, d in zip(builder.cardinalities, model.emb_dims))
    log(f"      embeddings : {emb}")
    log(f"      trunk      : {X.shape[1] + sum(model.emb_dims)} -> "
        f"{' -> '.join(map(str, args.hidden))} -> 2 heads")
    log(f"      parameters : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-4
    )
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss(beta=0.5)

    best_val, best_state, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_t))
        running, t_epoch = 0.0, time.time()
        for b in range(0, len(train_t), args.batch_size):
            bi = train_t[perm[b:b + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            logits, reg = model(X_num[bi], X_cat[bi])
            mask = yc_t[bi] > 0.5
            reg_loss = (huber(reg[mask], yr_t[bi][mask]).mean() if mask.any()
                        else torch.zeros(()))
            loss = bce(logits, yc_t[bi]) + args.reg_weight * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.detach().item() * len(bi)
        train_loss = running / len(train_t)

        val_loss, val_prob, val_reg = evaluate(
            model, X_num, X_cat, yc_t, yr_t, val_t, args.batch_size, bce, huber, args.reg_weight
        )
        val_pred_min = np.clip(val_reg * y_std + y_mean, 0, 300)
        vmask = y_cls[val_idx] > 0.5
        val_acc = accuracy_score(y_cls[val_idx], (val_prob >= 0.5).astype(np.int8))
        val_auc = roc_auc_score(y_cls[val_idx], val_prob)
        val_mae = mean_absolute_error(y_delay[val_idx][vmask], val_pred_min[vmask])
        scheduler.step(val_loss)

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(float(val_acc), 5),
            "val_auc": round(float(val_auc), 5),
            "val_mae_minutes": round(float(val_mae), 3),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "seconds": round(time.time() - t_epoch, 1),
        })
        log(f"      epoch {epoch:02d} | {time.time() - t_epoch:5.1f}s | train {train_loss:.4f} "
            f"| val {val_loss:.4f} | acc {val_acc:.4f} | auc {val_auc:.4f} "
            f"| MAE {val_mae:.2f} min | lr {optimizer.param_groups[0]['lr']:.1e}"
            f"{'  *best' if improved else ''}")
        if epoch - best_epoch >= args.patience:
            log(f"      Early stopping (no improvement for {args.patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------------------------------------------- metrics
    log("[8/8] Evaluating on held-out test split...")
    _, test_prob, test_reg = evaluate(
        model, X_num, X_cat, yc_t, yr_t, test_t, args.batch_size, bce, huber, args.reg_weight
    )
    test_pred_min = np.clip(test_reg * y_std + y_mean, 0, 300)
    y_test = y_cls[test_idx]

    metrics = {
        "model": "Multi-task DNN V2 (embeddings + engineered features)",
        "sample_fraction": args.sample_frac,
        "rows": {"total": int(n), "train": int(len(train_idx)),
                 "val": int(len(val_idx)), "test": int(len(test_idx))},
        "delayed_ratio": float(y_cls.mean()),
        "features": {"numeric": int(X.shape[1]), "embedded": int(cat.shape[1]),
                     "cardinalities": builder.cardinalities,
                     "names": NUMERIC_COLS},
        "architecture": {"hidden": args.hidden, "dropout": args.dropout,
                         "emb_dims": model.emb_dims, "parameters": int(n_params)},
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_seconds": round(time.time() - start, 1),
        "test": {
            "auc": float(roc_auc_score(y_test, test_prob)),
            "accuracy_at_0.5": float(accuracy_score(y_test, (test_prob >= 0.5).astype(np.int8))),
            "tiers": {f"{t:.2f}": cls_metrics(y_test, test_prob, t) for t in (0.25, 0.40, 0.50)},
            "regression": reg_metrics(y_delay[test_idx], test_pred_min, y_test > 0.5),
            "regression_all_flights": {
                "mae_minutes": float(mean_absolute_error(y_delay[test_idx], test_pred_min))
            },
            "mean_predicted_prob": float(test_prob.mean()),
        },
        "history": history,
    }

    # keep V1 numbers alongside so the README table can be generated
    try:
        with open(P.DNN_V1_METRICS, "r", encoding="utf-8") as fp:
            v1 = json.load(fp)
        metrics["v1_baseline"] = {
            "accuracy": v1["test"]["accuracy"], "auc": v1["test"]["auc"],
            "precision": v1["test"]["precision"], "recall": v1["test"]["recall"],
            "f1": v1["test"]["f1"], "mae_minutes": v1["test"]["mae_minutes"],
            "rows": v1["rows"], "features": v1.get("feature_count"),
            "train_seconds": v1.get("train_seconds"),
        }
    except Exception:
        pass

    t = metrics["test"]
    log("")
    log("  " + "-" * 62)
    log("  [Multi-task DNN V2 - test set]")
    log(f"  AUC-ROC      : {t['auc']:.4f}   (V1 0.7306)")
    log(f"  Accuracy@0.5 : {t['accuracy_at_0.5'] * 100:.2f}%   (V1 81.44%)")
    for thr in ("0.25", "0.40", "0.50"):
        m = t["tiers"][thr]
        log(f"  Tier P>={thr}  : flagged {m['flagged_pct'] * 100:5.2f}% | "
            f"prec {m['precision'] * 100:5.2f}% | rec {m['recall'] * 100:5.2f}% | "
            f"F1 {m['f1'] * 100:5.2f}%")
    r = t["regression"]
    log(f"  Delay MAE    : {r['mae_minutes']:.2f} min | RMSE {r['rmse_minutes']:.2f} | "
        f"R2 {r['r2']:.4f}   (V1 37.38 min)")
    log(f"  MAE all rows : {t['regression_all_flights']['mae_minutes']:.2f} min")
    log("  " + "-" * 62)

    model_path = os.path.join(args.out_dir, "dnn_v2_model.pt")
    prep_path = os.path.join(args.out_dir, "dnn_v2_preprocessor.pkl")
    metrics_path = os.path.join(args.out_dir, "dnn_v2_metrics.json")

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "n_numeric": int(X.shape[1]),
            "cardinalities": builder.cardinalities,
            "emb_dims": model.emb_dims,
            "hidden": list(args.hidden),
            "dropout": list(args.dropout),
            "delay_threshold_min": DELAY_THRESHOLD_MIN,
            "sample_fraction": args.sample_frac,
            "feature_names": NUMERIC_COLS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    }, model_path)

    state = builder.to_state()
    state.update({
        "y_mean": y_mean,
        "y_std": y_std,
        "delay_clip": DELAY_CLIP,
        "feature_names": NUMERIC_COLS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    })
    joblib.dump(state, prep_path, compress=3)
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    log(f"  [SAVED] {model_path}")
    log(f"  [SAVED] {prep_path}")
    log(f"  [SAVED] {metrics_path}")
    log(f"\n  Done in {time.time() - start:.1f}s "
        f"({(time.time() - start) / 60:.1f} min). Run the app with: python flask_app.py")


if __name__ == "__main__":
    main()
