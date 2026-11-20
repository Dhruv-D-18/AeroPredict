"""
calibrate_open_meteo.py
=======================
Fit a per-airport correction from **Open-Meteo** onto the **training weather
scale** used by DNN V2.

The model was trained on ``flight_delay/weather_meteo_by_airport.csv``.  A live
Open-Meteo forecast is a slightly different measurement of the same weather:

* temperature / pressure agree closely (a small constant offset);
* wind is systematically lower (different averaging height) -> needs a scale;
* rain is under-reported (coarser grid, misses convective cells) -> needs a
  scale.

We cannot compare a 2023 dataset against a 2026 forecast, so we compare against
Open-Meteo's *historical forecast archive* for 2023 - the same model family the
live endpoint uses.  The correction therefore transfers to future dates.

Writes ``wx_calibration.json`` (a few hundred KB).  Safe to re-run; already
fetched airport-days are cached in ``.wx_cache/``.

Usage
-----
    python calibrate_open_meteo.py            # all airports (~20 requests)
    python calibrate_open_meteo.py --limit 40 # quick check on 40 airports
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
WEATHER_CSV = BASE_DIR / "flight_delay" / "weather_meteo_by_airport.csv"
COORDS_CSV = BASE_DIR / "airport_coords.csv"
OUTPUT = BASE_DIR / "wx_calibration.json"
CACHE_DIR = BASE_DIR / ".wx_cache" / "archive"

ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
YEAR = 2023
BATCH_SIZE = 20
BATCH_PAUSE = 0.8          # be polite: 10k requests/day allowance
MIN_PAIRS = 30

OM_VARS = {
    "temperature_2m_mean": "tavg",
    "precipitation_sum": "prcp",
    "wind_speed_10m_mean": "wspd",
    "pressure_msl_mean": "pres",
    "snowfall_sum": "snow",
}
COLS = list(OM_VARS.values())
WINDOW = {"tavg": 6.0, "pres": 8.0}          # sane bounds for constant offsets


# ------------------------------------------------------------------ fetch
def _http_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "AeroPredict/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_batch_once(codes: list[str], coords: dict,
                      start: str, end: str) -> pd.DataFrame:
    """One archive request for a batch of airports -> long DataFrame."""
    cached = CACHE_DIR / f"archive_{start}_{end}_{codes[0]}_{codes[-1]}.json"
    if cached.is_file():
        with open(cached, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    else:
        params = {
            "latitude": ",".join(f"{coords[c][0]:.4f}" for c in codes),
            "longitude": ",".join(f"{coords[c][1]:.4f}" for c in codes),
            "daily": ",".join(OM_VARS),
            "timezone": "auto",
            "start_date": start,
            "end_date": end,
        }
        payload = _http_json(f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}", 60)
        if isinstance(payload, dict):
            payload = [payload]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cached, "w", encoding="utf-8") as fp:
            json.dump(payload, fp)

    frames = []
    for code, block in zip(codes, payload):
        if not isinstance(block, dict) or block.get("error"):
            continue
        daily = block.get("daily") or {}
        if not daily.get("time"):
            continue
        df = pd.DataFrame({"time": pd.to_datetime(daily["time"]), "airport_id": code})
        for om_name, col in OM_VARS.items():
            df[col] = pd.to_numeric(pd.Series(daily.get(om_name)), errors="coerce")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_batch(codes: list[str], coords: dict, start: str, end: str,
                depth: int = 0) -> pd.DataFrame:
    """Daily archive weather for a batch of airports, resilient to bad requests.

    A single problematic airport (or a server-side hiccup) makes Open-Meteo
    reject the whole multi-location request, so on failure the batch is split in
    half and retried - down to one airport per request.
    """
    try:
        return _fetch_batch_once(codes, coords, start, end)
    except Exception:                              # noqa: BLE001
        if len(codes) <= 1 or depth >= 3:
            raise
        mid = len(codes) // 2
        left = fetch_batch(codes[:mid], coords, start, end, depth + 1)
        right = fetch_batch(codes[mid:], coords, start, end, depth + 1)
        if left.empty:
            return right
        if right.empty:
            return left
        return pd.concat([left, right], ignore_index=True)


# -------------------------------------------------------------------- fit
def fit_variable(col: str, om: np.ndarray, ds: np.ndarray) -> dict:
    """Fit ``dataset ~ a * open_meteo + b`` for one variable."""
    mask = np.isfinite(om) & np.isfinite(ds)
    om, ds = om[mask], ds[mask]
    n = int(len(om))
    if n < MIN_PAIRS:
        return {}

    if col == "snow":
        # ``snow`` in the training file is on a completely different scale from
        # Open-Meteo's ``snowfall_sum`` (cm): it looks like a raw accumulation
        # in another unit (~120x larger).  Fit only on days where *either*
        # source reports snow, otherwise the ~95% zero days dominate and the
        # ratio collapses.
        event = (om > 0) | (ds > 0)
        if int(event.sum()) < MIN_PAIRS:
            return {}
        om_e, ds_e = om[event], ds[event]
        om_mean = float(np.mean(om_e))
        a = float(np.mean(ds_e) / om_mean) if om_mean > 1e-3 else 1.0
        a = float(np.clip(a, 1.0, 300.0))
        b = 0.0
    elif col in ("tavg", "pres"):
        a = 1.0
        b = float(np.median(ds - om))
        b = float(np.clip(b, -WINDOW[col], WINDOW[col]))
    else:
        # Scale-only correction: ratio of means is robust to the many dry /
        # calm hours and cannot produce negatives.
        om_mean = float(np.mean(om))
        floor = 0.05 if col == "prcp" else 1e-3
        a = float(np.mean(ds) / om_mean) if om_mean > floor else 1.0
        a = float(np.clip(a, 0.3 if col == "wspd" else 1.0, 8.0))
        b = 0.0

    corr = 0.0
    if n > 2 and om.std() > 0 and ds.std() > 0:
        corr = float(np.corrcoef(om, ds)[0, 1])
    return {"a": round(a, 4), "b": round(b, 4), "n": n, "r": round(corr, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N airports")
    args = ap.parse_args()

    if not WEATHER_CSV.is_file():
        raise SystemExit(f"missing {WEATHER_CSV}")

    ds = pd.read_csv(WEATHER_CSV)
    ds.columns = ds.columns.str.strip()
    ds["time"] = pd.to_datetime(ds["time"], errors="coerce").dt.normalize()
    ds = ds.dropna(subset=["time", "airport_id"])
    ds["airport_id"] = ds["airport_id"].astype(str)

    coords_df = pd.read_csv(COORDS_CSV)
    coords = {
        str(r["IATA_CODE"]): (float(r["LATITUDE"]), float(r["LONGITUDE"]))
        for _, r in coords_df.iterrows()
    }

    counts = ds.groupby("airport_id").size()
    codes = [c for c in sorted(counts[counts >= MIN_PAIRS].index) if c in coords]
    if args.limit:
        codes = codes[: args.limit]

    start, end = f"{YEAR}-01-01", f"{YEAR}-12-31"
    print(f"Fitting calibration for {len(codes)} airports against the {YEAR} archive...")

    frames = []
    for i in range(0, len(codes), BATCH_SIZE):
        batch = codes[i : i + BATCH_SIZE]
        try:
            frames.append(fetch_batch(batch, coords, start, end))
        except Exception as exc:                  # noqa: BLE001
            print(f"  batch {i // BATCH_SIZE + 1} failed ({batch[0]}..{batch[-1]}): {exc}")
        print(f"  {min(i + BATCH_SIZE, len(codes)):3d}/{len(codes)} airports", end="\r")
        time.sleep(BATCH_PAUSE)
    print()

    om = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if om.empty:
        raise SystemExit("no archive data downloaded - check the network")

    merged = om.merge(ds, on=["airport_id", "time"], suffixes=("_om", "_ds"))
    print(f"matched airport-days: {len(merged):,}")

    global_params = {
        col: fit_variable(col, merged[f"{col}_om"].to_numpy(float),
                          merged[f"{col}_ds"].to_numpy(float))
        for col in COLS
    }
    global_params = {k: v for k, v in global_params.items() if v}

    airports: dict[str, dict] = {}
    for code, g in merged.groupby("airport_id"):
        fit = {}
        # Snow is fitted globally only: the unit mismatch is the same everywhere
        # and per-airport fits on a handful of snowy days are pure noise.
        for col in ("tavg", "prcp", "wspd", "pres"):
            p = fit_variable(col, g[f"{col}_om"].to_numpy(float),
                             g[f"{col}_ds"].to_numpy(float))
            if p:
                fit[col] = p
        if fit:
            airports[str(code)] = fit

    payload = {
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": (f"Open-Meteo historical forecast archive {YEAR} vs "
                      f"weather_meteo_by_airport.csv"),
        "applied_as": "dataset_value = a * open_meteo_value + b",
        "airports_fitted": len(airports),
        "matched_rows": int(len(merged)),
        "global": global_params,
        "airports": airports,
    }
    with open(OUTPUT, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=1)

    print(f"\nwrote {OUTPUT.name}  ({OUTPUT.stat().st_size / 1024:.0f} KB, "
          f"{len(airports)} airports)")
    print("\nglobal correction (a*x + b):")
    for col in COLS:
        p = global_params.get(col)
        if p:
            print(f"  {col:5s} a={p['a']:.3f} b={p['b']:+.3f}  r={p['r']:+.2f}  n={p['n']:,}")

    # Cached forecasts were produced before the correction existed.
    stale = list((BASE_DIR / ".wx_cache").glob("fc_*.json"))
    for f in stale:
        f.unlink(missing_ok=True)
    if stale:
        print(f"cleared {len(stale)} stale forecast cache file(s)")


if __name__ == "__main__":
    main()
