"""
AeroPredict 2.0 — network inference for the web app
===================================================
NetworkForecaster wraps the trained ST-GNN (or a fallback GRU/LSTM) and the airport graph:

* forecast(t)          airport disruption forecast for the next 1/2/3 slots after slot t,
                       with 95 % bands and (for past dates) the observed outcome
* simulate(...)        what-if: inject extra delays and/or bad weather at one airport and
                       re-run the network model → change at every other airport
* footprint(airport)   model-based propagation: which airports the model expects to be hit
                       when this airport is disrupted
"""
from __future__ import annotations

import json

import numpy as np

import project_paths as P
from st_data import GraphData, HORIZONS, T_IN, SLOTS

RISK_MED, RISK_HIGH = 0.25, 0.40
BUSY_LN = np.log1p(8.0)            # airports with ≥ ~8 typical departures in the slot are ranked


class NetworkForecaster:
    def __init__(self, device: str = "cpu"):
        import torch
        from st_models import load_checkpoint
        self.torch = torch
        self.g = GraphData()
        self.device = device
        candidates = [("ST-GNN (GCN + GRU)", P.ST_GNN_MODEL),
                      ("ST-GNN (GAT + GRU)", P.ST_GNN_DIR / "st_gnn_gat_model.pt"),
                      ("GRU (temporal only)", P.MODELS_DIR / "temporal" / "gru_model.pt"),
                      ("LSTM (temporal only)", P.MODELS_DIR / "temporal" / "lstm_model.pt")]
        for label, path in candidates:
            if path.exists():
                self.model, self.cfg = load_checkpoint(path, device)
                self.label = label
                break
        else:
            raise FileNotFoundError("no spatio-temporal model trained yet")
        self.adj = torch.from_numpy(self.g.adj).to(device)
        self.mask = torch.from_numpy(self.g.adj_mask).to(device)
        prop = P.GRAPH_DIR / "propagation.json"
        self.propagation = json.loads(prop.read_text(encoding="utf-8")) if prop.exists() else None
        abl = P.RESULTS_DIR / "ablation_results.json"
        self.ablation = json.loads(abl.read_text(encoding="utf-8")) if abl.exists() else None

    # ── helpers ────────────────────────────────────────────────────────
    @property
    def t_min(self):
        return T_IN - 1

    @property
    def t_max(self):
        return self.g.T - 1                     # forecasts after the last slot have no outcome yet

    def default_t(self):
        """Most disrupted slot of the December test month (network-wide delay rate of the next slot)."""
        if getattr(self, "_default_t", None) is None:
            g = self.g
            ts = g.window_starts("test")
            nxt = ts + 1
            rate = g.dep_d[nxt].sum(1) / np.maximum(g.dep_n[nxt].sum(1), 1)
            rate = np.where(g.dep_n[nxt].sum(1) > 2000, rate, 0)   # ignore thin night slots
            self._default_t = int(ts[int(np.argmax(rate))])
        return self._default_t

    def _run(self, x, f):
        torch = self.torch
        with torch.no_grad():
            xt = torch.from_numpy(x[None]).to(self.device)
            ft = torch.from_numpy(f[None]).to(self.device)
            mean, logvar = self.model(xt, ft, self.adj, self.mask)
        return mean[0].cpu().numpy(), np.exp(0.5 * logvar[0].cpu().numpy())      # [H, N]

    def _inputs(self, t):
        g = self.g
        x = g.hist[t - T_IN + 1: t + 1].copy()
        if t + 1 + HORIZONS <= g.T:
            f = g.fut[t + 1: t + 1 + HORIZONS].copy()
        else:                                    # beyond the data: calendar + usual rates, climatology weather
            f = np.repeat(g.fut[t: t + 1], HORIZONS, axis=0).copy()
            for h in range(HORIZONS):
                s = (t + 1 + h) % 4
                f[h, :, 4:8] = np.eye(4)[s]
                f[h, :, 10] = g.usual[s]
        return x, f

    def _target_labels(self, t):
        out = []
        for h in range(1, HORIZONS + 1):
            tt = t + h
            day = self.g.days[min(tt // 4, len(self.g.days) - 1)] if tt // 4 < len(self.g.days) else "next day"
            out.append(f"{day} · {SLOTS[tt % 4]}")
        return out

    def _risk(self, r):
        return "high" if r >= RISK_HIGH else "medium" if r >= RISK_MED else "low"

    # ── public API ─────────────────────────────────────────────────────
    def meta(self, max_routes=260):
        g = self.g
        order = np.argsort(-g.edge_count)
        seen, routes = set(), []
        for e in order:
            a, b = int(g.edge_src[e]), int(g.edge_dst[e])
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            routes.append([a, b, int(g.edge_count[e])])
            if len(routes) >= max_routes:
                break
        busy = g.dep_n.sum(0)
        return {
            "model": self.label, "parameters": self.cfg.get("parameters"),
            "airports": [{"i": i, "code": c, "lat": float(g.lat[i]), "lon": float(g.lon[i]),
                          "flights": int(busy[i])} for i, c in enumerate(g.airports)],
            "routes": routes, "n_routes": int(len(g.edge_src)),
            "days": [str(g.days[0]), str(g.days[-1])], "slots": SLOTS,
            "default": {"day": str(g.days[self.default_t() // 4]), "slot": SLOTS[self.default_t() % 4]},
            "propagation": None if not self.propagation else
            {"overall": self.propagation["overall"], "by_distance": self.propagation["by_distance"]},
        }

    def forecast(self, t: int):
        g = self.g
        t = int(np.clip(t, self.t_min, self.t_max))
        x, f = self._inputs(t)
        mean, sd = self._run(x, f)
        tgt = [t + 1 + h for h in range(HORIZONS)]
        airports = []
        for i, code in enumerate(g.airports):
            row = {"i": i, "code": code, "mean": [], "lo": [], "hi": [], "actual": [], "flights": [], "usual": []}
            for h, tt in enumerate(tgt):
                m, s = float(mean[h, i]), float(sd[h, i])
                row["mean"].append(round(m, 4))
                row["lo"].append(round(max(0.0, m - 1.96 * s), 4))
                row["hi"].append(round(min(1.0, m + 1.96 * s), 4))
                if tt < g.T and g.dep_n[tt, i] >= 3:
                    row["actual"].append(round(float(g.rate[tt, i]), 4)); row["flights"].append(int(g.dep_n[tt, i]))
                else:
                    row["actual"].append(None); row["flights"].append(int(g.dep_n[tt, i]) if tt < g.T else None)
                row["usual"].append(round(float(g.usual[tt % 4, i]), 4))
            airports.append(row)
        warnings = []
        for h, tt in enumerate(tgt):
            typical = g.typical_ln[tt % 4, g._dow_of_t()[min(tt, g.T - 1)]]
            busy = typical >= BUSY_LN
            idx = np.argsort(-np.where(busy, mean[h], -1))[:10]
            warnings.append([{"code": g.airports[i], "mean": round(float(mean[h, i]), 4),
                              "lo": round(float(max(0, mean[h, i] - 1.96 * sd[h, i])), 4),
                              "hi": round(float(min(1, mean[h, i] + 1.96 * sd[h, i])), 4),
                              "usual": round(float(g.usual[tt % 4, i]), 4), "risk": self._risk(mean[h, i])}
                             for i in idx if busy[i]])
        return {"issued_after": g.slot_label(t), "t": t, "targets": self._target_labels(t),
                "airports": airports, "warnings": warnings, "model": self.label}

    def simulate(self, t: int, airport: str, delay_pp: float = 0.0, rain_mm: float = 0.0,
                 wind_kmh: float = 0.0, slots: int = 2):
        g = self.g
        t = int(np.clip(t, self.t_min, self.t_max))
        i = g.index[airport]
        x, f = self._inputs(t)
        base_mean, base_sd = self._run(x, f)
        xs, fs = x.copy(), f.copy()
        k = int(np.clip(slots, 1, T_IN))
        boost = delay_pp / 100.0
        xs[-k:, i, 0] = np.clip(xs[-k:, i, 0] + boost, 0, 1)        # departure delay rate
        xs[-k:, i, 2] = np.clip(xs[-k:, i, 2] + boost, 0, 1)        # arrival delay rate
        xs[-k:, i, 14] = 1.0
        dp, dw = rain_mm / g.wx_std[1], wind_kmh / g.wx_std[2]
        xs[-k:, i, 5] += dp; xs[-k:, i, 6] += dw                     # recent weather at the airport
        fs[:, i, 1] += dp; fs[:, i, 2] += dw                         # forecast weather for the next slots
        sim_mean, sim_sd = self._run(xs, fs)
        delta = sim_mean - base_mean                                 # [H, N]
        tgt = [t + 1 + h for h in range(HORIZONS)]
        vol = np.stack([np.expm1(g.typical_ln[tt % 4, g._dow_of_t()[min(tt, g.T - 1)]]) for tt in tgt])
        affected = []
        others = np.argsort(-delta.max(0))
        for j in others:
            if j == i or delta[:, j].max() < 0.002:
                continue
            affected.append({"code": g.airports[j], "delta_pp": [round(100 * float(d), 2) for d in delta[:, j]],
                             "base": [round(float(b), 4) for b in base_mean[:, j]],
                             "scenario": [round(float(s), 4) for s in sim_mean[:, j]],
                             "connected": bool(g.adj_mask[i, j])})
            if len(affected) >= 12:
                break
        return {
            "issued_after": g.slot_label(t), "targets": self._target_labels(t), "airport": airport,
            "scenario": {"delay_pp": delay_pp, "rain_mm": rain_mm, "wind_kmh": wind_kmh, "slots": k},
            "self": {"base": [round(float(v), 4) for v in base_mean[:, i]],
                     "scenario": [round(float(v), 4) for v in sim_mean[:, i]],
                     "base_hi": [round(float(min(1, b + 1.96 * s)), 4) for b, s in zip(base_mean[:, i], base_sd[:, i])],
                     "scenario_hi": [round(float(min(1, b + 1.96 * s)), 4) for b, s in zip(sim_mean[:, i], sim_sd[:, i])]},
            "network_extra_delayed_flights": [round(float((np.delete(delta[h], i) * np.delete(vol[h], i)).sum()), 1)
                                              for h in range(HORIZONS)],
            "airport_extra_delayed_flights": [round(float(delta[h, i] * vol[h, i]), 1) for h in range(HORIZONS)],
            "affected": affected,
            "delta": [[round(100 * float(v), 2) for v in delta[h]] for h in range(HORIZONS)],
        }

    def footprint(self, airport: str, t: int | None = None):
        t = self.default_t() if t is None else t
        sim = self.simulate(t, airport, delay_pp=30, slots=2)
        measured = []
        if self.propagation:
            measured = self.propagation["per_airport"].get(airport, [])
        return {"airport": airport, "measured": measured, "model": sim["affected"][:10],
                "issued_after": sim["issued_after"],
                "overall": self.propagation["overall"] if self.propagation else None}
