"""
AeroPredict 2.0 — spatio-temporal data pipeline
===============================================
Turns data/graph/graph.npz into model-ready tensors, shared by training,
evaluation and the web app so every part sees identical features.

Time step   : one slot (Morning, Afternoon, Evening, Night) ≈ 6 h
Input       : the last T_IN = 8 slots (2 days) for every airport
Target      : departure delay rate (share of flights arriving 15+ min late)
              for the next 1, 2 and 3 slots  →  horizons h = 1, 2, 3
Split (time): train Jan–Sep 2023 · validation Oct–Nov · test December 2023

Node features per slot (F_HIST = 16)
  0  departure delay rate (airport mean when the slot had no flights)
  1  log(1 + departures)            2  arrival delay rate     3  log(1 + arrivals)
  4–7 weather: tavg, prcp, wspd, pres (z-scored)
  8–11 slot one-hot                 12–13 day-of-week sin/cos
  14 has-flights flag               15 airport long-run delay rate (train only)
Future covariates per horizon step (F_FUT = 12): weather (4), slot one-hot (4), dow sin/cos (2),
airport's usual delay rate for that slot type (train only), typical departures for that
airport × slot × weekday (train only). All are known in advance — the weather is what the
what-if simulator changes. (The *actual* number of flights is not used: cancellations would leak.)
"""
from __future__ import annotations

import numpy as np

import project_paths as P

SLOTS = ["Morning", "Afternoon", "Evening", "Night"]
T_IN = 8
HORIZONS = 3
MIN_FLIGHTS = 3                   # target cells with fewer departures are ignored
DISRUPTION_RATE = 0.40            # ≥ 40% delayed = disruption (≈ top 10% of busy slots)
DISRUPTION_MIN_FLIGHTS = 5
TRAIN_END_DAY = 273               # 2023-10-01
VAL_END_DAY = 334                 # 2023-12-01
F_HIST = 16
F_FUT = 12


class GraphData:
    """Loads graph.npz and exposes normalised arrays + adjacency."""

    def __init__(self, path=None):
        g = np.load(path or (P.GRAPH_DIR / "graph.npz"), allow_pickle=False)
        self.airports = g["airports"].astype(str)
        self.lat, self.lon = g["lat"], g["lon"]
        self.days = g["days"].astype(str)
        self.dep_n = g["dep_n"].astype(np.float32)
        self.dep_d = g["dep_delayed"].astype(np.float32)
        self.arr_n = g["arr_n"].astype(np.float32)
        self.arr_d = g["arr_delayed"].astype(np.float32)
        self.weather_day = g["weather"].astype(np.float32)          # [D, N, 4]
        self.edge_src, self.edge_dst = g["edge_src"], g["edge_dst"]
        self.edge_count = g["edge_count"]
        self.edge_duration = g["edge_duration"]
        self.edge_distance = g["edge_distance_km"]
        self.T, self.N = self.dep_n.shape
        self.index = {c: i for i, c in enumerate(self.airports)}

        tr = slice(0, TRAIN_END_DAY * 4)
        # airport long-run delay rate (train period only — no leakage)
        self.ap_rate = (self.dep_d[tr].sum(0) + 0.19 * 20) / (self.dep_n[tr].sum(0) + 20)
        self.ap_arr_rate = (self.arr_d[tr].sum(0) + 0.19 * 20) / (self.arr_n[tr].sum(0) + 20)
        wtr = self.weather_day[:TRAIN_END_DAY]
        self.wx_mean = wtr.reshape(-1, 4).mean(0)
        self.wx_std = wtr.reshape(-1, 4).std(0) + 1e-6
        ln = np.log1p(self.dep_n[tr])
        self.vol_scale = float(ln.max()) or 1.0
        self.ln_arr_scale = float(np.log1p(self.arr_n[tr]).max()) or 1.0

        self.rate = np.where(self.dep_n > 0, self.dep_d / np.maximum(self.dep_n, 1), np.nan)
        self.usual, self.typical_ln = self._usual_tables()
        self.hist = self._build_hist()                               # [T, N, F_HIST]
        self.fut = self._build_future()                              # [T, N, F_FUT]
        self.adj = self._adjacency()

    # ── features ────────────────────────────────────────────────────────
    def weather_slots(self, weather_day=None):
        w = self.weather_day if weather_day is None else weather_day
        return np.repeat(w, 4, axis=0)                               # [T, N, 4]

    def _calendar(self):
        t = np.arange(self.T)
        slot = t % 4
        day = t // 4
        dow = (np.array([np.datetime64(d, "D") for d in self.days]).astype("datetime64[D]").view("int64") + 3) % 7
        dow_t = dow[day]
        onehot = np.eye(4, dtype=np.float32)[slot]
        ang = 2 * np.pi * dow_t / 7.0
        return onehot, np.stack([np.sin(ang), np.cos(ang)], 1).astype(np.float32)

    def _build_hist(self, dep_n=None, dep_d=None, weather_day=None):
        dep_n = self.dep_n if dep_n is None else dep_n
        dep_d = self.dep_d if dep_d is None else dep_d
        T, N = dep_n.shape
        has = (dep_n > 0).astype(np.float32)
        rate = np.where(dep_n > 0, dep_d / np.maximum(dep_n, 1), self.ap_rate[None, :])
        arr_rate = np.where(self.arr_n > 0, self.arr_d / np.maximum(self.arr_n, 1), self.ap_arr_rate[None, :])
        wx = (self.weather_slots(weather_day) - self.wx_mean) / self.wx_std
        onehot, dow = self._calendar()
        x = np.zeros((T, N, F_HIST), np.float32)
        x[..., 0] = rate
        x[..., 1] = np.log1p(dep_n) / self.vol_scale
        x[..., 2] = arr_rate
        x[..., 3] = np.log1p(self.arr_n) / self.ln_arr_scale
        x[..., 4:8] = wx
        x[..., 8:12] = onehot[:, None, :]
        x[..., 12:14] = dow[:, None, :]
        x[..., 14] = has
        x[..., 15] = self.ap_rate[None, :]
        return x

    def _usual_tables(self):
        """Train-period airport × slot delay rate and airport × slot × weekday typical volume."""
        tr = (np.arange(self.T) // 4) < TRAIN_END_DAY
        slot = np.arange(self.T) % 4
        dow = self._dow_of_t()
        usual = np.zeros((4, self.N), np.float32)
        typical = np.zeros((4, 7, self.N), np.float32)
        for s in range(4):
            sel = tr & (slot == s)
            usual[s] = (self.dep_d[sel].sum(0) + 0.19 * 20) / (self.dep_n[sel].sum(0) + 20)
            for d in range(7):
                sd = sel & (dow == d)
                typical[s, d] = np.log1p(self.dep_n[sd].mean(0)) if sd.any() else 0.0
        return usual, typical

    def _dow_of_t(self):
        day = np.arange(self.T) // 4
        d0 = np.array([np.datetime64(d, "D") for d in self.days]).astype("int64")
        return ((d0 + 3) % 7)[day]                                 # Monday = 0

    def _build_future(self, weather_day=None):
        wx = (self.weather_slots(weather_day) - self.wx_mean) / self.wx_std
        onehot, dow = self._calendar()
        slot = np.arange(self.T) % 4
        f = np.zeros((self.T, self.N, F_FUT), np.float32)
        f[..., 0:4] = wx
        f[..., 4:8] = onehot[:, None, :]
        f[..., 8:10] = dow[:, None, :]
        f[..., 10] = self.usual[slot]
        f[..., 11] = self.typical_ln[slot, self._dow_of_t()] / self.vol_scale
        return f

    def _adjacency(self):
        """Symmetric, degree-normalised adjacency  Â = D^-1/2 (A + I) D^-1/2."""
        A = np.zeros((self.N, self.N), np.float32)
        w = np.log1p(self.edge_count).astype(np.float32)
        A[self.edge_src, self.edge_dst] = w
        A = np.maximum(A, A.T)
        A = A / (A.max() or 1.0)
        A_hat = A + np.eye(self.N, dtype=np.float32)
        d = A_hat.sum(1)
        dinv = 1.0 / np.sqrt(d)
        self.adj_mask = (A_hat > 0)
        return (dinv[:, None] * A_hat * dinv[None, :]).astype(np.float32)

    # ── windows ─────────────────────────────────────────────────────────
    def window_starts(self, split: str):
        """Index t of the *last observed* slot for each sample in a split."""
        ts = np.arange(T_IN - 1, self.T - HORIZONS)
        target_day = (ts + 1) // 4
        if split == "train":
            keep = target_day < TRAIN_END_DAY
        elif split == "val":
            keep = (target_day >= TRAIN_END_DAY) & (target_day < VAL_END_DAY)
        elif split == "test":
            keep = (target_day >= VAL_END_DAY)
        else:
            keep = np.ones_like(ts, bool)
        return ts[keep]

    def sample(self, t, hist=None, fut=None):
        """Inputs and targets for a forecast issued after slot t."""
        hist = self.hist if hist is None else hist
        fut = self.fut if fut is None else fut
        x = hist[t - T_IN + 1: t + 1]                                    # [T_IN, N, F]
        f = fut[t + 1: t + 1 + HORIZONS]                                 # [H, N, F_FUT]
        y = np.nan_to_num(self.rate[t + 1: t + 1 + HORIZONS], nan=0.0)   # [H, N]
        n = self.dep_n[t + 1: t + 1 + HORIZONS]
        m = (n >= MIN_FLIGHTS).astype(np.float32)
        return x, f, y.astype(np.float32), m, n

    def batch(self, ts, hist=None, fut=None):
        xs, fs, ys, ms, ns = zip(*(self.sample(int(t), hist, fut) for t in ts))
        return (np.stack(xs), np.stack(fs), np.stack(ys), np.stack(ms), np.stack(ns))

    def slot_label(self, t):
        return f"{self.days[t // 4]} {SLOTS[t % 4]}"

    def t_of(self, day: str, slot: str) -> int:
        d = int(np.nonzero(self.days == day)[0][0])
        return d * 4 + SLOTS.index(slot)
