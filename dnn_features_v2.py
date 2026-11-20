"""
AeroPredict V2 - Preprocessing & feature engineering for the deep network
=========================================================================
Shared by ``train_dnn_v2.py`` (training) and ``flask_app.py`` (serving), so the
exact same transformations are applied in both places.

Why V2 exists
-------------
V1 fed the network 14 numeric + 22 one-hot columns.  V2 adds the signal that
actually drives delays, and encodes it in a way a neural net can use:

1. **Categorical embeddings** instead of one-hot columns
   Airline, departure airport, arrival airport, departure slot and haul type
   each get a learned embedding.  One-hot encoding 263 airports would have
   created 263 near-empty columns; an 8-dim embedding learns that LGA/EWR/JFK
   behave alike and shares that knowledge with rare airports.

2. **Arrival-side information** (V1 only looked at the departure airport)
   - arrival airport weather: tavg / prcp / wspd / pres
   - arrival airport coordinates
   - great-circle distance dep -> arr (haversine), replacing the coarse
     3-level ``Distance_type`` bucket with a continuous km value.

3. **Cyclical calendar encoding**
   month / day-of-year / day-of-week as sin+cos pairs, so the net knows that
   31 Dec and 1 Jan are adjacent (raw integers or one-hots do not).

4. **Smoothed target encodings, fitted on the training split only**
   Departure-airport rate, arrival-airport rate, airline rate, route rate,
   airport x slot rate (peak congestion), airline x slot rate and
   airport x month rate (seasonality).
   Each is Bayesian-smoothed:  (sum + prior * k) / (count + k)
   so a route flown 3 times cannot inject a 100% delay rate.
   Fitting on train rows only is what keeps the measured lift honest - using
   all rows would leak the test labels into the features.

5. **Volume features** (label-free, so computed on all rows)
   log1p(flights on that route), log1p(flights from that airport that day).

Training/serving parity
-----------------------
``FeatureBuilder.build(df)`` takes a frame with the canonical raw columns and
returns ``(numeric_matrix, categorical_codes)``.  Bulk training and one-row
Flask inference call the same method, so the two paths cannot drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Raw columns
# --------------------------------------------------------------------------
RAW_COLS = [
    "FlightDate", "Day_Of_Week", "Airline", "Dep_Airport", "Arr_Airport",
    "DepTime_label", "Flight_Duration", "Distance_type", "Arr_Delay",
    "Aicraft_age",
]
CATEGORICAL_COLS = ["Airline", "Dep_Airport", "Arr_Airport", "DepTime_label", "Distance_type"]

NUMERIC_COLS = [
    # schedule
    "Day_Of_Week", "Flight_Duration", "Aicraft_age", "dep_time_num",
    # departure weather
    "dep_tavg", "dep_prcp", "dep_wspd", "dep_pres", "dep_snow",
    # arrival weather
    "arr_tavg", "arr_prcp", "arr_wspd", "arr_pres",
    # geography
    "dep_lat", "dep_lon", "arr_lat", "arr_lon", "dist_km",
    # calendar (cyclical)
    "month_sin", "month_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "is_weekend",
    # smoothed target encodings (train-only)
    "dep_ap_rate", "arr_ap_rate", "airline_rate", "route_rate",
    "ap_slot_rate", "airline_slot_rate", "ap_month_rate",
    # volume
    "route_volume_log", "ap_daily_flights_log",
]

DELAY_THRESHOLD_MIN = 15.0
DELAY_CLIP = (-30.0, 300.0)
SLOT_ORDER = ["Morning", "Afternoon", "Evening", "Night"]
UNK = "__UNK__"
WX_COLS = ["tavg", "prcp", "wspd", "pres", "snow"]
ARR_WX_COLS = ["tavg", "prcp", "wspd", "pres"]

DAY_KEY = 100_000_000   # airport * DAY_KEY + yyyymmdd


def log(msg=""):
    print(msg, flush=True)


def load_flights(path: str, frac: float = 1.0, chunksize: int = 500_000,
                 seed: int = 42) -> pd.DataFrame:
    """Read the raw CSV in chunks with categorical dtypes (memory friendly).

    ``frac=1.0`` keeps every row (full dataset); smaller values sample
    deterministically per chunk so runs are reproducible.
    """
    dtype = {
        "Airline": "category", "Dep_Airport": "category", "Arr_Airport": "category",
        "DepTime_label": "category", "Distance_type": "category",
    }
    parts, seen, kept = [], 0, 0
    reader = pd.read_csv(path, usecols=RAW_COLS, chunksize=chunksize,
                         dtype=dtype, low_memory=False)
    for i, chunk in enumerate(reader):
        seen += len(chunk)
        chunk = chunk.dropna(subset=["Arr_Delay"])
        if frac < 1.0:
            chunk = chunk.sample(frac=frac, random_state=seed + i)
        parts.append(chunk)
        kept += len(chunk)

    df = pd.concat(parts, ignore_index=True)
    del parts
    log(f"      raw rows {seen:,} -> kept {kept:,} ({kept / max(seen, 1):.1%})")
    return df


class FeatureBuilder:
    """Every fitted mapping needed to turn raw rows into model inputs."""

    def __init__(self):
        self.cat_levels: dict[str, list[str]] = {}
        self.cat_dtype: dict[str, pd.CategoricalDtype] = {}
        self.lat: dict[str, float] = {}
        self.lon: dict[str, float] = {}
        # weather stored as dense arrays [airport, day_of_year] / [airport, month]
        self.wx_daily: dict[str, np.ndarray] = {}
        self.wx_monthly: dict[str, np.ndarray] = {}
        self.global_wx: dict[str, float] = {}
        self.data_year: int = 2023
        self.ap_index: dict[str, int] = {}
        self.n_ap: int = 1
        self.stride: int = 1
        self.rates: dict[str, dict[int, float]] = {}
        self.volumes: dict[str, dict[int, float]] = {}
        self.ap_daily_mean: dict[int, float] = {}
        self.prior: float = 0.0
        self.num_mean: np.ndarray | None = None
        self.num_scale: np.ndarray | None = None
        self.num_lo: np.ndarray | None = None
        self.num_hi: np.ndarray | None = None
        self.feature_names: list[str] = list(NUMERIC_COLS)

    # ---------------------------------------------------------------- vocab
    def fit_vocab(self, df: pd.DataFrame) -> "FeatureBuilder":
        for col in CATEGORICAL_COLS:
            levels = sorted(df[col].astype(str).unique().tolist())
            if col == "DepTime_label":
                levels = [s for s in SLOT_ORDER if s in levels] + [
                    s for s in levels if s not in SLOT_ORDER
                ]
            self.cat_levels[col] = [UNK] + levels
            self.cat_dtype[col] = pd.CategoricalDtype(categories=self.cat_levels[col])
        airports = sorted(set(df["Dep_Airport"].astype(str)) | set(df["Arr_Airport"].astype(str)))
        self.ap_index = {a: i for i, a in enumerate(airports)}
        self.n_ap = len(airports) + 1              # index 0 = unknown airport
        self.stride = self.n_ap
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.cat_levels[c]) for c in CATEGORICAL_COLS]

    def codes(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(df), len(CATEGORICAL_COLS)), dtype=np.int32)
        for j, col in enumerate(CATEGORICAL_COLS):
            series = df[col].astype(str)
            codes = series.astype(self.cat_dtype[col]).cat.codes.to_numpy(np.int32)
            out[:, j] = np.where(codes < 0, 0, codes)   # unseen level -> UNK
        return out

    # ----------------------------------------------------------- geo/weather
    def fit_context(self, airports_csv: str, weather_csv: str) -> "FeatureBuilder":
        geo = pd.read_csv(airports_csv)
        self.lat = dict(zip(geo["IATA_CODE"], geo["LATITUDE"]))
        self.lon = dict(zip(geo["IATA_CODE"], geo["LONGITUDE"]))

        wx = pd.read_csv(weather_csv)
        wx.columns = wx.columns.str.strip()
        wx["time"] = pd.to_datetime(wx["time"], errors="coerce")
        wx = wx.dropna(subset=["time", "airport_id"])
        self.data_year = int(wx["time"].dt.year.mode().iloc[0])

        ap_id = wx["airport_id"].astype(str).map(self.ap_index)
        valid = ap_id.notna().to_numpy()
        wx = wx.loc[valid].copy()
        ap = ap_id[valid].to_numpy(np.int64)          # 1-based airport index
        dos = wx["time"].dt.dayofyear.to_numpy(np.int64)
        month = wx["time"].dt.month.to_numpy(np.int64)

        self.wx_daily = {}
        self.wx_monthly = {}
        self.global_wx = {}
        for col in WX_COLS:
            vals = wx[col].to_numpy(np.float64)
            self.global_wx[col] = float(np.nanmean(vals))
            daily = np.full((self.n_ap, 367), np.nan, dtype=np.float32)
            daily[ap, dos] = vals.astype(np.float32)
            monthly = np.full((self.n_ap, 13), np.nan, dtype=np.float32)
            g = pd.DataFrame({"ap": ap, "m": month, "v": vals}).groupby(["ap", "m"])["v"].mean()
            for (a, m), v in g.items():
                monthly[int(a), int(m)] = float(v)
            monthly = np.where(np.isfinite(monthly), monthly, self.global_wx[col]).astype(np.float32)
            monthly[:, 0] = self.global_wx[col]
            self.wx_daily[col] = daily
            self.wx_monthly[col] = monthly
        return self

    def _wx(self, ap: np.ndarray, years: np.ndarray, dos: np.ndarray,
            months: np.ndarray, col: str,
            override: np.ndarray | None = None) -> np.ndarray:
        """Weather for (airport, day).

        Priority: ``override`` (a live Open-Meteo forecast) -> exact daily value
        from the training file (same year) -> monthly climatology.
        """
        exact = self.wx_daily[col][ap, dos]
        out = np.where(np.isfinite(exact) & (years == self.data_year), exact, np.nan)
        if override is not None:
            out = np.where(np.isfinite(override), override, out)
        miss = ~np.isfinite(out)
        if miss.any():
            out[miss] = self.wx_monthly[col][ap[miss], months[miss]]
        return out.astype(np.float32)

    @staticmethod
    def _override_arrays(wx_override: dict | None,
                         airport_codes: np.ndarray,
                         date_ids: np.ndarray,
                         cols: list[str]) -> dict[str, np.ndarray] | None:
        """Turn ``{(airport, yyyymmdd): {col: value}}`` into per-column arrays.

        Used only on the serving path (one flight at a time), never for training.
        """
        if not wx_override:
            return None
        out = {c: np.full(len(airport_codes), np.nan, dtype=np.float32) for c in cols}
        for i, (ap, did) in enumerate(zip(airport_codes, date_ids)):
            rec = wx_override.get((str(ap), int(did)))
            if not rec:
                continue
            for c in cols:
                v = rec.get(c)
                if v is not None and np.isfinite(v):
                    out[c][i] = float(v)
        return out

    # -------------------------------------------------------------- features
    def build(self, df: pd.DataFrame, with_target: bool = True,
              wx_override: dict | None = None):
        """Return (numeric, categorical_codes, y_cls, y_delay).

        ``wx_override`` lets the serving layer inject a live weather forecast:
        a mapping ``{(airport_code, yyyymmdd): {"tavg": .., "prcp": .., ...}}``.
        Training calls leave it as ``None``, so features are unchanged.
        """
        n = len(df)
        date = pd.to_datetime(df["FlightDate"], errors="coerce")
        if date.isna().any():                 # incomplete form / future booking
            date = date.fillna(pd.Timestamp(f"{self.data_year}-01-01"))
        months = date.dt.month.to_numpy(np.int64)
        dos = date.dt.dayofyear.to_numpy(np.int64)
        years = date.dt.year.to_numpy(np.int64)
        date_ids = (years * 10000 + months * 100 + date.dt.day.to_numpy(np.int64))
        dow = (pd.to_numeric(df["Day_Of_Week"], errors="coerce")
               .fillna(1.0).to_numpy(np.float32))

        ap_dep_s = df["Dep_Airport"].astype(str)
        ap_arr_s = df["Arr_Airport"].astype(str)
        ap_dep = ap_dep_s.map(self.ap_index).fillna(0).to_numpy(np.int64)
        ap_arr = ap_arr_s.map(self.ap_index).fillna(0).to_numpy(np.int64)

        dep_lat = ap_dep_s.map(self.lat).fillna(39.0).to_numpy(np.float32)
        dep_lon = ap_dep_s.map(self.lon).fillna(-98.0).to_numpy(np.float32)
        arr_lat = ap_arr_s.map(self.lat).fillna(39.0).to_numpy(np.float32)
        arr_lon = ap_arr_s.map(self.lon).fillna(-98.0).to_numpy(np.float32)

        cat = self.codes(df)
        slot_id = cat[:, 3].astype(np.int64)
        airline_id = cat[:, 0].astype(np.int64)

        feats: dict[str, np.ndarray] = {
            "Day_Of_Week": dow,
            "Flight_Duration": pd.to_numeric(df["Flight_Duration"], errors="coerce")
                                 .fillna(120.0).to_numpy(np.float32),
            "Aicraft_age": pd.to_numeric(df["Aicraft_age"], errors="coerce")
                             .fillna(12.0).to_numpy(np.float32),
            "dep_time_num": slot_id.astype(np.float32),
            "dep_lat": dep_lat, "dep_lon": dep_lon,
            "arr_lat": arr_lat, "arr_lon": arr_lon,
            "dist_km": haversine(dep_lat, dep_lon, arr_lat, arr_lon),
            "month_sin": np.sin(2 * np.pi * months / 12.0).astype(np.float32),
            "month_cos": np.cos(2 * np.pi * months / 12.0).astype(np.float32),
            "doy_sin": np.sin(2 * np.pi * dos / 365.25).astype(np.float32),
            "doy_cos": np.cos(2 * np.pi * dos / 365.25).astype(np.float32),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0).astype(np.float32),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0).astype(np.float32),
            "is_weekend": np.isin(dow, [6, 7]).astype(np.float32),
        }
        dep_ovr = self._override_arrays(wx_override, ap_dep_s.to_numpy(), date_ids, WX_COLS)
        arr_ovr = self._override_arrays(wx_override, ap_arr_s.to_numpy(), date_ids, ARR_WX_COLS)
        for col in WX_COLS:
            feats[f"dep_{col}"] = self._wx(
                ap_dep, years, dos, months, col, dep_ovr[col] if dep_ovr else None)
        for col in ARR_WX_COLS:
            feats[f"arr_{col}"] = self._wx(
                ap_arr, years, dos, months, col, arr_ovr[col] if arr_ovr else None)

        rate_keys = {
            "dep_ap_rate": ap_dep,
            "arr_ap_rate": ap_arr,
            "airline_rate": airline_id,
            "route_rate": ap_dep * self.stride + ap_arr,
            "ap_slot_rate": ap_dep * 8 + slot_id,
            "airline_slot_rate": airline_id * 8 + slot_id,
            "ap_month_rate": ap_dep * 16 + months,
        }
        for name, key in rate_keys.items():
            feats[name] = self._map_lookup(self.rates.get(name), key, self.prior)

        feats["route_volume_log"] = np.log1p(
            self._map_lookup(self.volumes.get("route_volume_log"),
                             ap_dep * self.stride + ap_arr, 0.0)
        )
        daily = self._map_lookup(self.volumes.get("ap_daily_flights_log"),
                                 ap_dep * DAY_KEY + date_ids, 0.0)
        if self.ap_daily_mean:      # unseen date (future booking) -> airport average
            miss = daily <= 0
            if miss.any():
                uniq, inv = np.unique(ap_dep[miss], return_inverse=True)
                fill = np.fromiter((self.ap_daily_mean.get(int(a), 0.0) for a in uniq),
                                   dtype=np.float32, count=len(uniq))
                daily[miss] = fill[inv]
        feats["ap_daily_flights_log"] = np.log1p(daily)

        X = np.empty((n, len(NUMERIC_COLS)), dtype=np.float32)
        for j, name in enumerate(NUMERIC_COLS):
            X[:, j] = feats[name]
        del feats, daily

        y_cls = y_delay = None
        if with_target:
            y_delay = np.clip(df["Arr_Delay"].to_numpy(np.float32), *DELAY_CLIP)
            y_cls = (y_delay > DELAY_THRESHOLD_MIN).astype(np.float32)
        return X, cat, y_cls, y_delay

    @staticmethod
    def _map_lookup(table, key: np.ndarray, default: float) -> np.ndarray:
        if table is None:
            return np.full(len(key), default, dtype=np.float32)
        uniq, inv = np.unique(key, return_inverse=True)
        vals = np.fromiter((table.get(int(k), default) for k in uniq),
                           dtype=np.float32, count=len(uniq))
        return vals[inv]

    # ---------------------------------------------------------------- stats
    def fit_stats(self, df: pd.DataFrame, y_cls: np.ndarray, train_mask: np.ndarray):
        """Fit smoothed target encodings on TRAIN rows; volumes on all rows."""
        sub = df.loc[train_mask]
        y = y_cls[train_mask]
        self.prior = float(y.mean())

        date = pd.to_datetime(sub["FlightDate"], errors="coerce")
        months = date.dt.month.to_numpy(np.int64)
        ap_dep = sub["Dep_Airport"].astype(str).map(self.ap_index).fillna(0).to_numpy(np.int64)
        ap_arr = sub["Arr_Airport"].astype(str).map(self.ap_index).fillna(0).to_numpy(np.int64)
        cat = self.codes(sub)
        slot_id = cat[:, 3].astype(np.int64)
        airline_id = cat[:, 0].astype(np.int64)

        specs = {
            "dep_ap_rate": (ap_dep, 20),
            "arr_ap_rate": (ap_arr, 20),
            "airline_rate": (airline_id, 20),
            "route_rate": (ap_dep * self.stride + ap_arr, 10),
            "ap_slot_rate": (ap_dep * 8 + slot_id, 20),
            "airline_slot_rate": (airline_id * 8 + slot_id, 20),
            "ap_month_rate": (ap_dep * 16 + months, 20),
        }
        self.rates = {}
        for name, (key, k) in specs.items():
            g = pd.DataFrame({"k": key, "y": y}).groupby("k")["y"].agg(["sum", "count"])
            smoothed = (g["sum"] + self.prior * k) / (g["count"] + k)
            self.rates[name] = {int(i): float(v) for i, v in smoothed.items()}

        # label-free volume features -> use every row
        full_dep = df["Dep_Airport"].astype(str).map(self.ap_index).fillna(0).to_numpy(np.int64)
        full_arr = df["Arr_Airport"].astype(str).map(self.ap_index).fillna(0).to_numpy(np.int64)
        fdate = pd.to_datetime(df["FlightDate"], errors="coerce")
        full_dates = (fdate.dt.year * 10000 + fdate.dt.month * 100 + fdate.dt.day).to_numpy(np.int64)

        route_counts = (pd.DataFrame({"k": full_dep * self.stride + full_arr})
                        .groupby("k").size())
        daily_counts = (pd.DataFrame({"k": full_dep * DAY_KEY + full_dates})
                        .groupby("k").size())
        ap_daily = (pd.DataFrame({"ap": full_dep, "d": full_dates})
                    .groupby(["ap", "d"]).size().groupby("ap").mean())

        self.volumes = {
            "route_volume_log": {int(i): float(v) for i, v in route_counts.items()},
            "ap_daily_flights_log": {int(i): float(v) for i, v in daily_counts.items()},
        }
        self.ap_daily_mean = {int(i): float(v) for i, v in ap_daily.items()}

        # ---- route-level schedule facts used at SERVING time -------------
        # The app must not invent a flight duration / haul bucket with a crude
        # formula: that puts the network off-distribution.  Store what the
        # dataset actually says for each route.
        self.fit_route_stats(df)
        return self

    def fit_route_stats(self, df: pd.DataFrame):
        """Learn observed duration + haul label per route (for serving parity)."""
        full_dep = df["Dep_Airport"].astype(str).map(self.ap_index).fillna(0).to_numpy(np.int64)
        full_arr = df["Arr_Airport"].astype(str).map(self.ap_index).fillna(0).to_numpy(np.int64)
        cat_full = self.codes(df)
        dur = df["Flight_Duration"].to_numpy(np.float32)
        dep_s = df["Dep_Airport"].astype(str)
        arr_s = df["Arr_Airport"].astype(str)
        rt = pd.DataFrame({
            "k": full_dep * self.stride + full_arr,
            "dur": dur,
            "dist": cat_full[:, 4],
            "dkm": haversine(dep_s.map(self.lat).fillna(39.0).to_numpy(np.float32),
                             dep_s.map(self.lon).fillna(-98.0).to_numpy(np.float32),
                             arr_s.map(self.lat).fillna(39.0).to_numpy(np.float32),
                             arr_s.map(self.lon).fillna(-98.0).to_numpy(np.float32)),
        })
        g = rt.groupby("k").agg(dur=("dur", "median"),
                                dist=("dist", lambda s: int(s.mode().iat[0])))
        self.route_stats = {int(k): (float(r["dur"]), int(r["dist"])) for k, r in g.iterrows()}

        per_code = rt.groupby("dist")["dkm"].mean()
        self.dist_centroid = {int(k): float(v) for k, v in per_code.items()}
        speed = (rt["dkm"] / rt["dur"].clip(lower=1)).replace([np.inf, -np.inf], np.nan)
        self.km_per_min = float(speed.median())
        return self

    def route_features(self, dep: str, arr: str) -> tuple[float, str]:
        """Return ``(flight_duration, distance_type_label)`` for a route.

        Uses the observed median duration / haul label from the training data so
        the served features match what the network saw.  Unknown routes fall
        back to haversine distance and the learned average block speed.
        """
        a, b = self.ap_index.get(dep), self.ap_index.get(arr)
        if a is not None and b is not None:
            rec = getattr(self, "route_stats", {}).get(a * self.stride + b)
            if rec is not None:
                dur, code = rec
                levels = self.cat_levels.get("Distance_type", [])
                label = levels[code] if 0 <= code < len(levels) else "Short Haul >1500Mi"
                return float(dur), label

        dkm = float(haversine(self.lat.get(dep, 39.0), self.lon.get(dep, -98.0),
                              self.lat.get(arr, 39.0), self.lon.get(arr, -98.0)))
        centroids = getattr(self, "dist_centroid", {})
        if centroids:
            code = min(centroids, key=lambda c: abs(centroids[c] - dkm))
        else:
            code = 1
        levels = self.cat_levels.get("Distance_type", [])
        label = levels[code] if 0 <= code < len(levels) else "Short Haul >1500Mi"
        speed = getattr(self, "km_per_min", 10.0) or 10.0
        return float(np.clip(dkm / speed, 30.0, 400.0)), label

    def fit_scaler(self, X: np.ndarray, train_mask: np.ndarray):
        """Clip to the train p0.1/p99.9 range, then standardise."""
        sub = X[train_mask]
        self.num_lo = np.quantile(sub, 0.001, axis=0).astype(np.float32)
        self.num_hi = np.quantile(sub, 0.999, axis=0).astype(np.float32)
        np.clip(X, self.num_lo, self.num_hi, out=X)
        self.num_mean = sub.mean(axis=0).astype(np.float32)
        self.num_scale = np.maximum(sub.std(axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        np.clip(X, self.num_lo, self.num_hi, out=X)
        X -= self.num_mean
        X /= self.num_scale
        return X

    # ---------------------------------------------------------- persistence
    def to_state(self) -> dict:
        return {
            "cat_levels": self.cat_levels,
            "lat": self.lat, "lon": self.lon,
            "wx_daily": self.wx_daily, "wx_monthly": self.wx_monthly,
            "global_wx": self.global_wx, "data_year": self.data_year,
            "ap_index": self.ap_index, "n_ap": self.n_ap, "stride": self.stride,
            "rates": self.rates, "volumes": self.volumes,
            "ap_daily_mean": self.ap_daily_mean, "prior": self.prior,
            "route_stats": getattr(self, "route_stats", {}),
            "dist_centroid": getattr(self, "dist_centroid", {}),
            "km_per_min": getattr(self, "km_per_min", 10.0),
            "num_mean": self.num_mean, "num_scale": self.num_scale,
            "num_lo": self.num_lo, "num_hi": self.num_hi,
            "feature_names": self.feature_names,
            "categorical_cols": CATEGORICAL_COLS,
            "numeric_cols": NUMERIC_COLS,
            "cardinalities": self.cardinalities,
            "delay_threshold_min": DELAY_THRESHOLD_MIN,
        }

    @classmethod
    def from_state(cls, state: dict) -> "FeatureBuilder":
        self = cls()
        self.cat_levels = state["cat_levels"]
        self.cat_dtype = {c: pd.CategoricalDtype(categories=self.cat_levels[c])
                          for c in CATEGORICAL_COLS}
        self.lat, self.lon = state["lat"], state["lon"]
        self.wx_daily, self.wx_monthly = state["wx_daily"], state["wx_monthly"]
        self.global_wx = state["global_wx"]
        self.data_year = state["data_year"]
        self.ap_index, self.n_ap, self.stride = state["ap_index"], state["n_ap"], state["stride"]
        self.rates, self.volumes = state["rates"], state["volumes"]
        self.ap_daily_mean = state["ap_daily_mean"]
        self.prior = state["prior"]
        self.route_stats = state.get("route_stats", {})
        self.dist_centroid = state.get("dist_centroid", {})
        self.km_per_min = state.get("km_per_min", 10.0)
        self.num_mean, self.num_scale = state["num_mean"], state["num_scale"]
        self.num_lo, self.num_hi = state["num_lo"], state["num_hi"]
        self.feature_names = state["feature_names"]
        return self

    # ---------------------------------------------------- single-row helper
    def build_one(self, row: dict, wx_override: dict | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Build (1, n_numeric) and (1, n_cat) for one flight (Flask path).

        Missing / blank fields are filled with sane defaults so a partially
        completed form can never crash inference.
        """
        df = pd.DataFrame([row])
        for col in CATEGORICAL_COLS:
            if col not in df:
                df[col] = UNK
            else:
                df[col] = df[col].where(df[col].notna(), UNK)
            df[col] = df[col].astype(str).replace({"": UNK, "nan": UNK, "None": UNK})
        if "FlightDate" not in df or df["FlightDate"].isna().any() or \
                not str(df["FlightDate"].iloc[0]).strip():
            df["FlightDate"] = f"{self.data_year}-01-01"
        if "Day_Of_Week" not in df:
            df["Day_Of_Week"] = (pd.to_datetime(df["FlightDate"], errors="coerce")
                                 .dt.weekday + 1).fillna(1)
        if "Aicraft_age" not in df:
            df["Aicraft_age"] = 12.0
        if "Flight_Duration" not in df:
            df["Flight_Duration"] = 120.0
        X, cat, _, _ = self.build(df, with_target=False, wx_override=wx_override)
        self.transform(X)
        return X, cat


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (vectorised) coordinate pairs."""
    r = 6371.0088
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return (2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float32)
