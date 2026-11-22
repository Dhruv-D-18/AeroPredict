"""
AeroPredict 2.0 — Step 1: build the airport network graph + time series
=======================================================================
Reads   data/processed/preprocessed_flight_data_final.csv  (6.7M flights, 2023 + Jan 2024)
Writes  data/graph/graph.npz            airport graph + per-airport time series
        data/graph/airports.json        airport list (index → IATA code, lat, lon)
        data/graph/test_flights.csv.gz  decoded test-period flights (used by the ablation study)

Graph
-----
* Nodes  : US airports (N ≈ 350)
* Edges  : scheduled routes A → B, with flight count, mean flight time, distance
* Series : one step per *time slot* (Morning 06-12, Afternoon 12-18, Evening 18-22,
           Night 22-06) for every day in the file (2023; Jan 2024 rows if present).
           The processed file has no departure hour, so the finest time step
           available is the 4-slot day (≈ 6 h per step).

Decoding the processed file
---------------------------
The Semester-6 preprocessing label-encoded the airports and dropped the year:
* Dep_Airport_Encoded → IATA via the departure LATITUDE/LONGITUDE in each row.
* Arr_Airport_Encoded → same alphabetical LabelEncoder order (verified below).
* Year                → January rows are 2023 or 2024; the weekday decides which.

Run from the project root:
    python training/phase2_st_gnn/build_graph.py
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

SLOTS = ["Morning", "Afternoon", "Evening", "Night"]       # chronological order within a day
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-01-31")
N_DAYS = (END - START).days + 1
T = N_DAYS * len(SLOTS)

AIRLINE_BASE = "Alaska Airlines Inc."          # dropped by get_dummies(drop_first=True)
DIST_BASE = "Long Haul <6000Mi"
WX = ["tavg", "prcp", "wspd", "pres"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def decode_one_hot(chunk, prefix, base):
    cols = [c for c in chunk.columns if c.startswith(prefix)]
    m = chunk[cols].to_numpy(dtype=bool)
    names = np.array([c[len(prefix):] for c in cols] + [base])
    idx = np.where(m.any(axis=1), m.argmax(axis=1), len(cols))
    return names[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(P.PROCESSED_FINAL))
    ap.add_argument("--chunksize", type=int, default=400_000)
    ap.add_argument("--test-start", default="2023-10-01",
                    help="flights on/after this date (validation + test months) are exported for the ablation study")
    ap.add_argument("--no-test-flights", action="store_true")
    args = ap.parse_args()

    P.GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    geo = pd.read_csv(P.RAW_AIRPORTS)
    geo_codes = geo["IATA_CODE"].astype(str).to_numpy()
    geo_ll = geo[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float)

    # Accumulators are sized generously and trimmed once airports are known.
    MAXN = 512
    dep_n = np.zeros((T, MAXN), np.float32); dep_d = np.zeros((T, MAXN), np.float32)
    arr_n = np.zeros((T, MAXN), np.float32); arr_d = np.zeros((T, MAXN), np.float32)
    wx_sum = np.zeros((N_DAYS, MAXN, len(WX)), np.float64); wx_cnt = np.zeros((N_DAYS, MAXN), np.float64)
    edge_n = np.zeros((MAXN, MAXN), np.float64); edge_dur = np.zeros((MAXN, MAXN), np.float64)
    dep_ll = {}                       # encoded dep id → (lat, lon)
    arr_codes_seen = set()
    test_parts = []
    n_rows = n_2024 = 0
    test_start = pd.Timestamp(args.test_start)

    reader = pd.read_csv(args.csv, chunksize=args.chunksize)
    for ci, ch in enumerate(reader):
        d_enc = ch["Dep_Airport_Encoded"].to_numpy(np.int64)
        a_enc = ch["Arr_Airport_Encoded"].to_numpy(np.int64)
        if max(d_enc.max(), a_enc.max()) >= MAXN:
            raise SystemExit("more airports than expected")
        # remember coordinates of each departure code
        first = pd.DataFrame({"e": d_enc, "lat": ch["LATITUDE"].to_numpy(), "lon": ch["LONGITUDE"].to_numpy()}).drop_duplicates("e")
        for e, la, lo in first.itertuples(index=False):
            dep_ll.setdefault(int(e), (float(la), float(lo)))
        arr_codes_seen.update(np.unique(a_enc).tolist())

        # ── recover the date (year from weekday for January rows) ──
        month = ch["month"].to_numpy(np.int64); day = ch["day"].to_numpy(np.int64)
        dow = ch["Day_Of_Week"].to_numpy(np.int64)            # 1 = Monday … 7 = Sunday
        d23 = pd.to_datetime(dict(year=np.full_like(day, 2023), month=month, day=day), errors="coerce")
        d24 = pd.to_datetime(dict(year=np.full_like(day, 2024), month=np.ones_like(day),
                                  day=np.where(month == 1, day, 1)), errors="coerce")
        is24 = (month == 1) & ((d24.dt.weekday.to_numpy() + 1) == dow) & ((d23.dt.weekday.to_numpy() + 1) != dow)
        date = np.where(is24, d24.to_numpy(), d23.to_numpy())
        day_idx = ((pd.to_datetime(date) - START).days).to_numpy()
        ok = (day_idx >= 0) & (day_idx < N_DAYS)

        slot_name = decode_one_hot(ch, "DepTime_label_", "Afternoon")
        slot = pd.Series(slot_name).map({s: i for i, s in enumerate(SLOTS)}).to_numpy(np.int64)
        t = day_idx * len(SLOTS) + slot
        y = ch["is_delayed"].to_numpy(np.float32)

        t_ok, d_ok, a_ok, y_ok = t[ok], d_enc[ok], a_enc[ok], y[ok]
        np.add.at(dep_n, (t_ok, d_ok), 1.0); np.add.at(dep_d, (t_ok, d_ok), y_ok)
        np.add.at(arr_n, (t_ok, a_ok), 1.0); np.add.at(arr_d, (t_ok, a_ok), y_ok)
        wx_vals = ch[WX].to_numpy(np.float64)[ok]
        np.add.at(wx_sum, (day_idx[ok], d_ok), wx_vals); np.add.at(wx_cnt, (day_idx[ok], d_ok), 1.0)
        np.add.at(edge_n, (d_ok, a_ok), 1.0)
        np.add.at(edge_dur, (d_ok, a_ok), ch["Flight_Duration"].to_numpy(np.float64)[ok])

        n_rows += len(ch); n_2024 += int(is24.sum())
        is_test = ok & np.asarray(pd.to_datetime(date) >= test_start)
        if not args.no_test_flights and is_test.any():
            sub = ch.loc[is_test]
            test_parts.append(pd.DataFrame({
                "FlightDate": pd.to_datetime(date[is_test]).strftime("%Y-%m-%d"),
                "Day_Of_Week": sub["Day_Of_Week"].to_numpy(),
                "dep_enc": sub["Dep_Airport_Encoded"].to_numpy(),
                "arr_enc": sub["Arr_Airport_Encoded"].to_numpy(),
                "Airline": decode_one_hot(sub, "Airline_", AIRLINE_BASE),
                "DepTime_label": decode_one_hot(sub, "DepTime_label_", "Afternoon"),
                "Distance_type": decode_one_hot(sub, "Distance_type_", DIST_BASE),
                "Flight_Duration": sub["Flight_Duration"].to_numpy(),
                "Aicraft_age": sub["Aicraft_age"].to_numpy(),
                "is_delayed": sub["is_delayed"].to_numpy(),
            }))
        log(f"chunk {ci + 1}: {n_rows:,} rows read  ({n_2024:,} from Jan 2024)")

    # ── decode airport ids ──
    dep_ids = sorted(dep_ll)
    code_of = {}
    for e in dep_ids:
        la, lo = dep_ll[e]
        j = int(np.argmin(np.abs(geo_ll[:, 0] - la) + np.abs(geo_ll[:, 1] - lo)))
        code_of[e] = geo_codes[j]
    codes_in_order = [code_of[e] for e in dep_ids]
    if codes_in_order != sorted(codes_in_order) or dep_ids != list(range(len(dep_ids))):
        raise SystemExit("departure codes are not in LabelEncoder (alphabetical) order - cannot decode safely")
    if sorted(arr_codes_seen) != dep_ids:
        raise SystemExit("arrival encoder differs from departure encoder - cannot decode safely")
    N = len(dep_ids)
    codes = np.array(codes_in_order)
    log(f"decoded {N} airports (alphabetical LabelEncoder order verified via coordinates)")

    lat = np.array([dep_ll[e][0] for e in dep_ids]); lon = np.array([dep_ll[e][1] for e in dep_ids])
    dep_n, dep_d, arr_n, arr_d = dep_n[:, :N], dep_d[:, :N], arr_n[:, :N], arr_d[:, :N]
    wx_cnt = wx_cnt[:, :N]
    wx_day = np.where(wx_cnt[..., None] > 0, wx_sum[:, :N] / np.maximum(wx_cnt[..., None], 1), np.nan)
    # fill days without departures: airport monthly → airport overall → global mean
    for k in range(len(WX)):
        col = wx_day[..., k]
        ap_mean = np.nanmean(col, axis=0)
        ap_mean = np.where(np.isfinite(ap_mean), ap_mean, np.nanmean(col))
        wx_day[..., k] = np.where(np.isfinite(col), col, ap_mean[None, :])

    # ── edges ──
    en, ed = edge_n[:N, :N], edge_dur[:N, :N]
    src, dst = np.nonzero(en)
    e_cnt = en[src, dst]; e_dur = ed[src, dst] / e_cnt
    r = np.radians
    dlat = r(lat[dst] - lat[src]); dlon = r(lon[dst] - lon[src])
    a = np.sin(dlat / 2) ** 2 + np.cos(r(lat[src])) * np.cos(r(lat[dst])) * np.sin(dlon / 2) ** 2
    e_dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    log(f"{len(src):,} directed routes between {N} airports")

    # trim trailing days that have no flights at all (e.g. no Jan-2024 rows in the file)
    per_day = dep_n.reshape(N_DAYS, len(SLOTS), N).sum(axis=(1, 2))
    n_days = int(np.nonzero(per_day)[0].max()) + 1
    Tn = n_days * len(SLOTS)
    dep_n, dep_d, arr_n, arr_d, wx_day = dep_n[:Tn], dep_d[:Tn], arr_n[:Tn], arr_d[:Tn], wx_day[:n_days]
    stamps = [(START + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in range(n_days)]
    np.savez_compressed(
        P.GRAPH_DIR / "graph.npz",
        airports=codes, lat=lat, lon=lon, slots=np.array(SLOTS), days=np.array(stamps),
        dep_n=dep_n, dep_delayed=dep_d, arr_n=arr_n, arr_delayed=arr_d,
        weather=wx_day.astype(np.float32), weather_cols=np.array(WX),
        edge_src=src, edge_dst=dst, edge_count=e_cnt, edge_duration=e_dur, edge_distance_km=e_dist,
    )
    with open(P.GRAPH_DIR / "airports.json", "w", encoding="utf-8") as fp:
        json.dump([{"i": i, "code": c, "lat": float(lat[i]), "lon": float(lon[i]),
                    "flights": int(dep_n[:, i].sum())} for i, c in enumerate(codes)], fp)
    log(f"saved {P.GRAPH_DIR / 'graph.npz'}  (T={Tn} slots = {n_days} days × 4, N={N} airports, {stamps[0]} … {stamps[-1]})")

    if test_parts:
        tf = pd.concat(test_parts, ignore_index=True)
        tf["Dep_Airport"] = codes[tf.pop("dep_enc").to_numpy()]
        tf["Arr_Airport"] = codes[tf.pop("arr_enc").to_numpy()]
        tf.to_csv(P.GRAPH_DIR / "test_flights.csv.gz", index=False, compression="gzip")
        log(f"saved {len(tf):,} test-period flights (from {args.test_start}) → test_flights.csv.gz")
    log(f"done: {n_rows:,} flights, {n_2024:,} dated to 2024")


if __name__ == "__main__":
    main()
