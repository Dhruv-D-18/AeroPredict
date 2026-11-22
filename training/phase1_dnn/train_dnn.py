"""
AeroPredict - PyTorch DNN training (no Spark / no Hadoop)
=========================================================
Trains a **multi-task deep neural network** on a configurable fraction
(default **50%**) of the raw 2023 flight dataset:

    shared MLP trunk
        |-- classification head -> P(arrival delay > 15 min)
        |-- regression head     -> arrival delay in minutes (delayed flights only)

The regression loss is masked to flights that were actually delayed, so the
head models "how late, given that it is late" — which is what the UI shows
for Medium/High risk predictions.

Feature engineering mirrors ``train_regressor.py`` (weather + airport
coordinates + calendar features + one-hot airline / departure slot /
distance type), but everything runs locally with pandas + PyTorch.

Run:
    python train_dnn.py --sample-frac 0.5 --epochs 12

Outputs (in --out-dir, default project root):
    dnn_model.pt            <- weights + config
    dnn_preprocessor.pkl    <- feature columns, scaler stats, target scaler
    dnn_metrics.json        <- train/test metrics (used by the Flask UI)
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

from dnn_common import DelayNet

RAW_PATH = P.RAW_FLIGHTS
WEATHER_PATH = P.RAW_WEATHER
AIRPORTS_PATH = P.RAW_AIRPORTS

USE_COLS = [
    "FlightDate", "Day_Of_Week", "Airline", "Dep_Airport", "Arr_Airport",
    "DepTime_label", "Flight_Duration", "Distance_type", "Arr_Delay",
    "Aicraft_age",
]
CATEGORICAL_COLS = ["Airline", "DepTime_label", "Distance_type"]
NUMERIC_COLS = [
    "Day_Of_Week", "Flight_Duration", "Aicraft_age",
    "tavg", "prcp", "wspd", "pres",
    "LATITUDE", "LONGITUDE",
    "month", "day", "is_weekend",
    "dep_airport_delay_avg", "arr_airport_delay_avg",
]
DELAY_THRESHOLD_MIN = 15.0
DELAY_CLIP = (-30.0, 300.0)


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Load a random sample
# ---------------------------------------------------------------------------
def load_sample(sample_frac: float, chunksize: int, seed: int) -> pd.DataFrame:
    """Read the raw CSV in chunks and keep ``sample_frac`` of the valid rows.

    Sampling happens per chunk (before concatenation) so peak memory stays
    well below a full 6.5M-row DataFrame.  Sampling only rows that have a
    non-null ``Arr_Delay`` also removes cancelled / diverted flights, which is
    the same effect as the old Spark ``left_anti`` join.
    """
    parts = []
    rows_seen = 0
    rows_kept = 0

    reader = pd.read_csv(
        RAW_PATH,
        usecols=USE_COLS,
        chunksize=chunksize,
        low_memory=False,
    )
    for i, chunk in enumerate(reader):
        rows_seen += len(chunk)
        chunk = chunk.dropna(subset=["Arr_Delay"])
        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=seed + i)
        parts.append(chunk)
        rows_kept += len(chunk)
        if (i + 1) % 4 == 0:
            log(f"      ... {rows_seen:,} rows read, {rows_kept:,} sampled")

    df = pd.concat(parts, ignore_index=True)
    del parts
    log(f"      Raw rows read : {rows_seen:,}")
    log(f"      Sampled rows  : {len(df):,}  ({sample_frac:.0%} of valid rows)")
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    log("[3/7] Merging weather + airport coordinates + delay-rate lookups...")
    weather = pd.read_csv(WEATHER_PATH)
    weather.columns = weather.columns.str.strip()
    airports = pd.read_csv(AIRPORTS_PATH)
    delay_lookup = pd.read_csv(P.AIRPORT_DELAY)
    delay_rate = dict(
        zip(delay_lookup["Dep_Airport"], delay_lookup["dep_airport_delay_avg"])
    )
    global_delay_rate = float(delay_lookup["dep_airport_delay_avg"].mean())

    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    weather["time"] = pd.to_datetime(weather["time"], errors="coerce")
    df = df.dropna(subset=["FlightDate"])

    df = df.merge(
        airports[["IATA_CODE", "LATITUDE", "LONGITUDE"]],
        left_on="Dep_Airport", right_on="IATA_CODE", how="left",
    ).drop(columns=["IATA_CODE"], errors="ignore")

    df = df.merge(
        weather[["airport_id", "time", "tavg", "prcp", "wspd", "pres"]],
        left_on=["Dep_Airport", "FlightDate"],
        right_on=["airport_id", "time"],
        how="left",
    ).drop(columns=["airport_id", "time"], errors="ignore")

    df["dep_airport_delay_avg"] = (
        df["Dep_Airport"].map(delay_rate).fillna(global_delay_rate).astype("float32")
    )
    df["arr_airport_delay_avg"] = (
        df["Arr_Airport"].map(delay_rate).fillna(global_delay_rate).astype("float32")
    )

    log("[4/7] Engineering features...")
    df["Arr_Delay"] = df["Arr_Delay"].clip(*DELAY_CLIP)
    df["month"] = df["FlightDate"].dt.month
    df["day"] = df["FlightDate"].dt.day
    df["is_weekend"] = df["Day_Of_Week"].isin([6, 7]).astype("int8")

    # Impute: medians for weather / aircraft age / duration, constants for coords.
    weather_cols = ["tavg", "prcp", "wspd", "pres"]
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
    df["Aicraft_age"] = df["Aicraft_age"].fillna(df["Aicraft_age"].median())
    df["Flight_Duration"] = df["Flight_Duration"].fillna(df["Flight_Duration"].median())
    df["Day_Of_Week"] = df["Day_Of_Week"].fillna(1)
    df["LATITUDE"] = df["LATITUDE"].fillna(39.0)
    df["LONGITUDE"] = df["LONGITUDE"].fillna(-98.0)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("Unknown").astype(str)

    df = df.drop(columns=["Dep_Airport", "Arr_Airport", "FlightDate"], errors="ignore")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype("float32")

    y_delay = df.pop("Arr_Delay").to_numpy(dtype=np.float32)
    y_cls = (y_delay > DELAY_THRESHOLD_MIN).astype(np.float32)

    df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS, dtype=np.float32)
    feature_columns = list(df_enc.columns)
    X = df_enc.to_numpy(dtype=np.float32)
    del df, df_enc

    log(f"      Feature matrix : {X.shape[0]:,} x {X.shape[1]}")
    log(f"      Delayed > 15min: {y_cls.mean():.2%} of rows")
    return X, y_cls, y_delay, feature_columns


# ---------------------------------------------------------------------------
# 3. Standardisation (chunked to avoid huge float64 copies)
# ---------------------------------------------------------------------------
def mean_std(X: np.ndarray, idx: np.ndarray, chunk: int = 500_000):
    n = len(idx)
    total = np.zeros(X.shape[1], dtype=np.float64)
    total_sq = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, n, chunk):
        part = X[idx[start:start + chunk]].astype(np.float64)
        total += part.sum(axis=0)
        total_sq += np.square(part).sum(axis=0)
    mean = total / n
    var = np.maximum(total_sq / n - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X, y_cls, y_reg, idx, batch_size, bce, huber, reg_weight):
    model.eval()
    total_loss = 0.0
    probs, regs = [], []
    for start in range(0, len(idx), batch_size):
        bi = idx[start:start + batch_size]
        logits, reg = model(X[bi])
        # The regression head only learns "delay given delay": mask to delayed flights.
        mask = y_cls[bi] > 0.5
        reg_loss = huber(reg[mask], y_reg[bi][mask]).mean() if mask.any() else torch.zeros(())
        loss = bce(logits, y_cls[bi]) + reg_weight * reg_loss
        total_loss += loss.item() * len(bi)
        probs.append(torch.sigmoid(logits).numpy())
        regs.append(reg.numpy())
    probs = np.concatenate(probs)
    regs = np.concatenate(regs)
    return total_loss / len(idx), probs, regs


def regression_metrics(y_true_min, y_pred_min):
    return {
        "mae_minutes": float(mean_absolute_error(y_true_min, y_pred_min)),
        "rmse_minutes": float(mean_squared_error(y_true_min, y_pred_min) ** 0.5),
        "r2": float(r2_score(y_true_min, y_pred_min)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train AeroPredict multi-task DNN")
    ap.add_argument("--sample-frac", type=float, default=0.50,
                    help="fraction of valid rows to use (default 0.50)")
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg-weight", type=float, default=0.5,
                    help="weight of the regression loss in the combined objective")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--hidden", type=int, nargs=3, default=[256, 128, 64])
    ap.add_argument("--dropout", type=float, nargs=2, default=[0.20, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(P.DNN_V1_DIR))
    args = ap.parse_args()

    start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    log("=" * 66)
    log("  AeroPredict - Multi-task DNN (classification + regression)")
    log("=" * 66)
    log(f"  Device     : CPU ({os.cpu_count()} threads)")
    log(f"  Sample     : {args.sample_frac:.0%} of flights")
    log(f"  Epochs     : {args.epochs}   Batch: {args.batch_size}   LR: {args.lr}")

    log("\n[1/7] Loading sampled flight data...")
    df = load_sample(args.sample_frac, args.chunksize, args.seed)

    log("[2/7] Preparing targets...")
    df = df.dropna(subset=["Arr_Delay"]).reset_index(drop=True)
    log(f"      Usable rows    : {len(df):,}")

    X, y_cls, y_delay, feature_columns = build_features(df)
    del df

    n = len(X)
    all_idx = np.arange(n)
    log("[5/7] Splitting 80 / 10 / 10 (stratified on delay label)...")
    train_idx, rest_idx = train_test_split(
        all_idx, test_size=0.20, random_state=args.seed, stratify=y_cls
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, random_state=args.seed, stratify=y_cls[rest_idx]
    )
    log(f"      Train {len(train_idx):,} | Val {len(val_idx):,} | Test {len(test_idx):,}")

    mean, scale = mean_std(X, train_idx)
    X -= mean
    X /= scale
    log("      Standardisation fitted on training split only.")

    y_mean = float(np.mean(y_delay[train_idx], dtype=np.float64))
    y_std = float(np.std(y_delay[train_idx], dtype=np.float64)) or 1.0
    y_reg = ((y_delay - y_mean) / y_std).astype(np.float32)
    log(f"      Regression target: mean {y_mean:.2f} min, std {y_std:.2f} min")

    X_t = torch.from_numpy(X)
    yc_t = torch.from_numpy(y_cls)
    yr_t = torch.from_numpy(y_reg)
    train_t = torch.from_numpy(train_idx.astype(np.int64))
    val_t = torch.from_numpy(val_idx.astype(np.int64))
    test_t = torch.from_numpy(test_idx.astype(np.int64))

    log("[6/7] Training network...")
    model = DelayNet(
        input_dim=X.shape[1],
        hidden=tuple(args.hidden),
        dropout=tuple(args.dropout),
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"      Architecture   : {X.shape[1]} -> {' -> '.join(map(str, args.hidden))} -> 2 heads")
    log(f"      Parameters     : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss(beta=0.5)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_t))
        running = 0.0
        t0 = time.time()

        for b in range(0, len(train_t), args.batch_size):
            batch_idx = train_t[perm[b:b + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            logits, reg = model(X_t[batch_idx])
            bce_loss = bce(logits, yc_t[batch_idx])
            mask = yc_t[batch_idx] > 0.5
            reg_loss = (
                huber(reg[mask], yr_t[batch_idx][mask]).mean()
                if mask.any() else torch.zeros(())
            )
            loss = bce_loss + args.reg_weight * reg_loss
            loss.backward()
            optimizer.step()
            running += loss.detach().item() * len(batch_idx)

        train_loss = running / len(train_t)
        val_loss, val_prob, val_reg = evaluate(
            model, X_t, yc_t, yr_t, val_t, args.batch_size, bce, huber, args.reg_weight
        )
        val_acc = accuracy_score(y_cls[val_idx], (val_prob >= 0.5).astype(np.int8))
        val_auc = roc_auc_score(y_cls[val_idx], val_prob)
        val_pred_min = np.clip(val_reg * y_std + y_mean, 0, 300)
        val_mask = y_cls[val_idx] > 0.5
        val_mae = mean_absolute_error(y_delay[val_idx][val_mask], val_pred_min[val_mask])
        scheduler.step(val_loss)

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5), "val_accuracy": round(float(val_acc), 5),
            "val_auc": round(float(val_auc), 5), "val_mae_minutes": round(float(val_mae), 3),
        })
        log(f"      epoch {epoch:02d} | {time.time()-t0:5.1f}s | train {train_loss:.4f} "
            f"| val {val_loss:.4f} | acc {val_acc:.4f} | auc {val_auc:.4f} "
            f"| MAE {val_mae:.2f} min{'  *best' if improved else ''}")

        if epoch - best_epoch >= args.patience:
            log(f"      Early stopping (no improvement for {args.patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    log("[7/7] Evaluating on held-out test split...")
    _, test_prob, test_reg = evaluate(
        model, X_t, yc_t, yr_t, test_t, args.batch_size, bce, huber, args.reg_weight
    )
    test_pred_cls = (test_prob >= 0.5).astype(np.int8)
    test_pred_min = np.clip(test_reg * y_std + y_mean, 0, 300)

    test_metrics = {
        "accuracy": float(accuracy_score(y_cls[test_idx], test_pred_cls)),
        "auc": float(roc_auc_score(y_cls[test_idx], test_prob)),
        "precision": float(precision_score(y_cls[test_idx], test_pred_cls, zero_division=0)),
        "recall": float(recall_score(y_cls[test_idx], test_pred_cls, zero_division=0)),
        "f1": float(f1_score(y_cls[test_idx], test_pred_cls, zero_division=0)),
    }
    delayed_test = y_cls[test_idx] > 0.5
    test_metrics.update(regression_metrics(y_delay[test_idx][delayed_test],
                                           test_pred_min[delayed_test]))
    test_metrics["mae_minutes_all_flights"] = float(
        mean_absolute_error(y_delay[test_idx], test_pred_min)
    )
    test_metrics["delayed_test_rows"] = int(delayed_test.sum())

    log("")
    log("  " + "-" * 54)
    log("  [Multi-task DNN - test set]")
    log(f"  Accuracy : {test_metrics['accuracy']*100:.2f}%")
    log(f"  AUC-ROC  : {test_metrics['auc']:.4f}")
    log(f"  Precision: {test_metrics['precision']*100:.2f}%   "
        f"Recall: {test_metrics['recall']*100:.2f}%   F1: {test_metrics['f1']*100:.2f}%")
    log(f"  Delay MAE: {test_metrics['mae_minutes']:.2f} min   "
        f"RMSE: {test_metrics['rmse_minutes']:.2f} min   (delayed flights only)")
    log(f"  Delay R2 : {test_metrics['r2']:.4f}   "
        f"MAE all flights: {test_metrics['mae_minutes_all_flights']:.2f} min")
    log("  " + "-" * 54)

    model_path = os.path.join(args.out_dir, "dnn_model.pt")
    prep_path = os.path.join(args.out_dir, "dnn_preprocessor.pkl")
    metrics_path = os.path.join(args.out_dir, "dnn_metrics.json")

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": int(X.shape[1]),
            "hidden": list(args.hidden),
            "dropout": list(args.dropout),
            "delay_threshold_min": DELAY_THRESHOLD_MIN,
            "sample_fraction": args.sample_frac,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    }, model_path)

    joblib.dump({
        "columns": feature_columns,
        "mean": mean,
        "scale": scale,
        "y_mean": y_mean,
        "y_std": y_std,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "delay_threshold_min": DELAY_THRESHOLD_MIN,
    }, prep_path)

    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump({
            "model": "Multi-task DNN (PyTorch MLP)",
            "sample_fraction": args.sample_frac,
            "rows": {
                "usable": int(n),
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "delayed_ratio": float(y_cls.mean()),
            "feature_count": int(X.shape[1]),
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "train_seconds": round(time.time() - start, 1),
            "test": {k: round(v, 6) for k, v in test_metrics.items()},
            "history": history,
        }, fp, indent=2)

    log(f"  [SAVED] {model_path}")
    log(f"  [SAVED] {prep_path}")
    log(f"  [SAVED] {metrics_path}")
    log(f"\n  Done in {time.time()-start:.1f}s. Run the app with: python flask_app.py")


if __name__ == "__main__":
    main()
