"""
validate_open_meteo.py
======================
Does the live-forecast weather path actually help?

Scores the *same* held-out 2023 flights through DNN V2 four times, changing only
the weather source:

1. ``dataset``     - the real 2023 weather the model was trained on (upper bound)
2. ``climatology`` - monthly means, i.e. what the app used for future dates before
3. ``open-meteo``  - raw Open-Meteo archive values (no correction)
4. ``calibrated``  - Open-Meteo archive corrected by ``wx_calibration.json``

If (4) lands close to (1) and clearly above (2), the integration is sound.
Uses the Historical Forecast API, i.e. the same model family as the live
forecast endpoint, so the result transfers to future dates.

    python validate_open_meteo.py --rows 40000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

import calibrate_open_meteo as cal_mod
from dnn_common import load_dnn_v2, predict_dnn_v2
from dnn_features_v2 import FeatureBuilder

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "wx_validation.json"

COLS = ["FlightDate", "Day_Of_Week", "Airline", "Dep_Airport", "Arr_Airport",
        "DepTime_label", "Flight_Duration", "Distance_type", "Arr_Delay",
        "Aicraft_age"]
WX_COLS = ["tavg", "prcp", "wspd", "pres", "snow"]


def calibration_airports(coords: dict) -> list[str]:
    """Exactly the airport list ``calibrate_open_meteo.py`` fetches, in order.

    Using the same list means the batch boundaries - and therefore the cache
    keys - match, so a previous calibration run is reused with zero requests.
    """
    ds = pd.read_csv(cal_mod.WEATHER_CSV, usecols=["airport_id"])
    counts = ds["airport_id"].astype(str).value_counts()
    return [c for c in sorted(counts[counts >= cal_mod.MIN_PAIRS].index) if c in coords]


def build_archive(coords: dict) -> dict:
    """{(airport, yyyymmdd): {col: value}} for every calibrated airport."""
    codes = calibration_airports(coords)
    print(f"  loading archive for {len(codes)} airports (cached after calibration)...")

    frames = []
    for i in range(0, len(codes), cal_mod.BATCH_SIZE):
        batch = codes[i : i + cal_mod.BATCH_SIZE]
        try:
            frames.append(cal_mod.fetch_batch(batch, coords, "2023-01-01", "2023-12-31"))
        except Exception as exc:                  # noqa: BLE001
            print(f"  batch failed ({batch[0]}..{batch[-1]}): {exc}")
        time.sleep(cal_mod.BATCH_PAUSE)
    om = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if om.empty:
        return {}
    om["date_id"] = (om["time"].dt.year * 10000 + om["time"].dt.month * 100
                     + om["time"].dt.day).astype(int)

    out = {}
    for row in om.itertuples(index=False):
        out[(row.airport_id, int(row.date_id))] = {
            "tavg": row.tavg, "prcp": row.prcp, "wspd": row.wspd,
            "pres": row.pres, "snow": row.snow,
        }
    return out


def calibrate_all(raw: dict, calibration: dict) -> dict:
    """Apply the per-airport correction to every raw archive record."""
    from weather_api import NON_NEGATIVE

    def params(ap, col):
        per = calibration.get("airports", {}).get(ap)
        if per and col in per:
            return per[col]
        return calibration.get("global", {}).get(col, {})

    out = {}
    for (ap, did), vals in raw.items():
        fixed = {}
        for col, v in vals.items():
            if v is None or not np.isfinite(v):
                fixed[col] = None
                continue
            p = params(ap, col)
            x = float(p.get("a", 1.0)) * float(v) + float(p.get("b", 0.0))
            fixed[col] = max(x, 0.0) if col in NON_NEGATIVE else x
        out[(ap, did)] = fixed
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("--model", default="dnn_v2_model.pt")
    ap.add_argument("--preprocessor", default="dnn_v2_preprocessor.pkl")
    args = ap.parse_args()

    model, state = load_dnn_v2(args.model, args.preprocessor)
    builder = FeatureBuilder.from_state(state)

    coords_df = pd.read_csv(BASE_DIR / "airport_coords.csv")
    coords = {str(r["IATA_CODE"]): (float(r["LATITUDE"]), float(r["LONGITUDE"]))
              for _, r in coords_df.iterrows()}

    df = pd.read_csv(BASE_DIR / "flight_delay" / "US_flights_2023.csv",
                     usecols=COLS, nrows=800_000, low_memory=False)
    df = df.sample(min(args.rows, len(df)), random_state=17).reset_index(drop=True)
    print(f"scoring {len(df):,} real 2023 flights")

    t0 = time.time()
    raw_archive = build_archive(coords)
    print(f"  archive records: {len(raw_archive):,}  ({time.time() - t0:.0f}s)")

    calibration = json.load(open(BASE_DIR / "wx_calibration.json", encoding="utf-8"))
    fixed_archive = calibrate_all(raw_archive, calibration)

    def score(tag: str, wx_override=None, force_climatology: bool = False):
        b = FeatureBuilder.from_state(state)
        if force_climatology:
            b.data_year = 1900                      # no exact day -> monthly mean
        X, cat, y_cls, y_delay = b.build(df.copy(), wx_override=wx_override)
        b.transform(X)
        prob, minutes = predict_dnn_v2(model, state, X, cat)
        mask = y_cls > 0.5

        coverage = 1.0
        if wx_override:
            dids = (pd.to_datetime(df["FlightDate"]).dt.strftime("%Y%m%d").astype(int))
            hits = [(ap, int(d)) in wx_override
                    for ap, d in zip(df["Dep_Airport"], dids)]
            coverage = round(float(np.mean(hits)), 3)

        res = {
            "weather_source": tag,
            "auc": round(float(roc_auc_score(y_cls, prob)), 4),
            "accuracy_at_0.5": round(float(accuracy_score(y_cls, prob >= 0.5)), 4),
            "mae_delayed": round(float(mean_absolute_error(y_delay[mask], minutes[mask])), 2),
            "mean_prob": round(float(prob.mean()), 4),
            "coverage": coverage,
        }
        print(f"  {tag:12s} AUC {res['auc']:.4f} | acc {res['accuracy_at_0.5']*100:.2f}% "
              f"| MAE {res['mae_delayed']:.2f} | mean P {res['mean_prob']:.3f}")
        return res

    print("\n4-way comparison (identical flights, only weather changes):")
    results = [score("dataset")]
    results.append(score("climatology", force_climatology=True))
    results.append(score("open-meteo", wx_override=raw_archive))
    results.append(score("calibrated", wx_override=fixed_archive))

    payload = {
        "rows": int(len(df)),
        "delayed_ratio": round(float((df["Arr_Delay"] > 15).mean()), 4),
        "results": results,
        "calibration": calibration.get("fitted_at"),
    }
    with open(OUTPUT, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=1)
    print(f"\nwrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
