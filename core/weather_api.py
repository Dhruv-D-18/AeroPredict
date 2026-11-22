"""
weather_api.py
==============
Free, key-less daily weather for AeroPredict, powered by Open-Meteo.

Why Open-Meteo
--------------
* 15-day forecast (today .. today+14), no API key, 10k requests/day on the
  non-commercial tier, data licensed CC BY 4.0 (attribution required - see
  :data:`ATTRIBUTION`).
* A Historical Forecast API (archive back to 2021) exposes exactly the same
  fields, which is what makes calibration against the 2023 training data - and
  a future V3 retrain on forecast weather - possible.

Important: the V2 model was trained on ``flight_delay/weather_meteo_by_airport.csv``,
a *different* weather product.  Open-Meteo's wind is measured/averaged
differently (the fitted correction is ~1.1x globally and up to ~1.6x at
individual airports) and its ~3-11 km precipitation grid misses local
convective storms.  Raw Open-Meteo values therefore land off the model's
training distribution, so :func:`get_pair` applies the per-airport correction
stored in ``wx_calibration.json`` (built by ``calibrate_open_meteo.py``).

Anything that goes wrong - no network, timeout, API change, unknown airport,
date outside the 15-day horizon - returns ``None`` so the caller can fall back
to the 2023 monthly climatology without an error reaching the user.
"""

from __future__ import annotations

# ── locate project folders (see core/project_paths.py) ──
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "project_paths.py").exists())
_sys.path.insert(0, str(_ROOT / "core"))
import project_paths as P

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CALIBRATION_PATH = P.WX_CALIBRATION
CACHE_DIR = P.WX_CACHE_DIR

# Open-Meteo's forecast endpoint currently accepts today .. today+14 (15 dates,
# even though the marketing page says "up to 16 days").  Anything beyond that is
# rejected with HTTP 400, so keep the gate conservative.
HORIZON_DAYS = 14

TIMEOUT_SECONDS = 6
CACHE_TTL_SECONDS = 60 * 60          # forecasts refresh hourly at the source

ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"

# Open-Meteo daily variable -> column name used by the training data.
VAR_MAP = {
    "temperature_2m_mean": "tavg",
    "precipitation_sum": "prcp",
    "wind_speed_10m_mean": "wspd",
    "pressure_msl_mean": "pres",
    "snowfall_sum": "snow",
}
DAILY_VARS = list(VAR_MAP)

NON_NEGATIVE = {"prcp", "wspd", "snow"}

# Only reported once, so a dead network does not flood the console.
_warned = False


# ---------------------------------------------------------------- horizon
def forecast_horizon() -> tuple[date, date]:
    """First and last date that can be answered by the live forecast."""
    today = date.today()
    return today, today + timedelta(days=HORIZON_DAYS)


def forecast_available(target: str | date | None) -> bool:
    """True when ``target`` falls inside the free 15-day forecast window."""
    if target is None:
        return False
    if isinstance(target, str):
        try:
            target = datetime.strptime(target.strip(), "%Y-%m-%d").date()
        except (ValueError, AttributeError):
            return False
    if not isinstance(target, date):
        return False
    start, end = forecast_horizon()
    return start <= target <= end


# ----------------------------------------------------------------- cache
def _cache_get(key: str, ttl: int):
    path = CACHE_DIR / f"{key}.json"
    try:
        if time.time() - path.stat().st_mtime <= ttl:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
    except Exception:
        pass
    return None


def _cache_put(key: str, payload) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / f"{key}.json", "w", encoding="utf-8") as fp:
            json.dump(payload, fp, separators=(",", ":"))
    except Exception:
        pass


def clear_cache() -> int:
    """Delete cached forecasts; returns the number of files removed."""
    removed = 0
    if CACHE_DIR.is_dir():
        for f in CACHE_DIR.glob("fc_*.json"):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
    return removed


# ----------------------------------------------------------- calibration
_calibration_cache = None


def load_calibration() -> dict:
    """Per-airport Open-Meteo -> training-scale correction (empty if absent)."""
    global _calibration_cache
    if _calibration_cache is None:
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as fp:
                _calibration_cache = json.load(fp)
        except Exception:
            _calibration_cache = {}
    return _calibration_cache


def _params_for(airport: str, col: str) -> dict:
    cal = load_calibration()
    per_airport = cal.get("airports", {}).get(airport)
    if per_airport and col in per_airport:
        return per_airport[col]
    return cal.get("global", {}).get(col, {})


def calibrate(airport: str, values: dict) -> dict:
    """Map raw Open-Meteo values onto the scale the model was trained on."""
    out = {}
    for col, val in values.items():
        if val is None:
            out[col] = None
            continue
        p = _params_for(airport, col)
        a = float(p.get("a", 1.0))
        b = float(p.get("b", 0.0))
        v = a * float(val) + b
        out[col] = round(max(v, 0.0) if col in NON_NEGATIVE else v, 3)
    return out


# ------------------------------------------------------------------ HTTP
def _http_json(url: str, timeout: int = TIMEOUT_SECONDS):
    req = urllib.request.Request(url, headers={"User-Agent": "AeroPredict/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_daily(points: list[tuple[str, tuple[float, float]]], target: date,
                 url: str = FORECAST_URL) -> dict[str, dict]:
    """One batched Open-Meteo request for several airports on one date."""
    params = {
        "latitude": ",".join(f"{lat:.4f}" for _, (lat, _lon) in points),
        "longitude": ",".join(f"{lon:.4f}" for _, (_lat, lon) in points),
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
        "start_date": target.isoformat(),
        "end_date": target.isoformat(),
    }
    payload = _http_json(f"{url}?{urllib.parse.urlencode(params)}")
    if isinstance(payload, dict):
        payload = [payload]

    out: dict[str, dict] = {}
    for (code, _coords), block in zip(points, payload):
        if not isinstance(block, dict) or block.get("error"):
            continue
        daily = block.get("daily") or {}
        times = daily.get("time") or []
        if target.isoformat() not in times:
            continue
        i = times.index(target.isoformat())
        values = {}
        for om_name, col in VAR_MAP.items():
            series = daily.get(om_name)
            values[col] = None if not series else series[i]
        out[code] = values
    return out


# -------------------------------------------------------------- public API
def get_pair(dep: str, arr: str, date_str: str, coords: dict) -> dict | None:
    """Weather for both ends of a flight, or ``None`` to use climatology.

    ``coords`` maps IATA code -> ``(latitude, longitude)``.

    Returns::

        {"source": "forecast",
         "label": "Open-Meteo forecast (12 Nov 2026)",
         "dep": {"tavg":..., "prcp":..., "wspd":..., "pres":..., "snow":...},
         "arr": {"tavg":..., "prcp":..., "wspd":..., "pres":..., "snow":...}}
    """
    global _warned
    if not forecast_available(date_str):
        return None

    target = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    key = f"fc_{dep}_{arr}_{target:%Y%m%d}"
    cached = _cache_get(key, CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    missing = [a for a in (dep, arr) if a not in coords]
    if missing:
        return None

    points = [(dep, coords[dep]), (arr, coords[arr])]
    try:
        raw = _fetch_daily(points, target)
    except urllib.error.HTTPError as exc:
        # Out-of-range dates are a normal miss, not an incident - stay quiet.
        if exc.code != 400 and not _warned:
            _warned = True
            print(f"[weather_api] forecast unavailable, using climatology: {exc}")
        return None
    except Exception as exc:                      # noqa: BLE001 - fail silent
        if not _warned:
            _warned = True
            print(f"[weather_api] forecast unavailable, using climatology: {exc}")
        return None

    if dep not in raw or arr not in raw:
        return None

    needed = ("tavg", "prcp", "wspd", "pres")
    if any(raw[dep].get(c) is None for c in needed) or \
            any(raw[arr].get(c) is None for c in needed):
        return None

    payload = {
        "source": "forecast",
        "label": f"Open-Meteo forecast ({target:%d %b %Y})",
        "date": target.isoformat(),
        "dep": calibrate(dep, raw[dep]),
        "arr": calibrate(arr, raw[arr]),
    }
    _cache_put(key, payload)
    return payload


def self_test() -> bool:
    """Quick connectivity check - used by ``verify_install.py``."""
    today, _ = forecast_horizon()
    try:
        raw = _fetch_daily([("JFK", (40.6413, -73.7781))], today)
    except Exception as exc:                      # noqa: BLE001
        print(f"  Open-Meteo unreachable: {exc}")
        return False
    if "JFK" not in raw:
        print("  Open-Meteo returned no data for JFK")
        return False
    print(f"  Open-Meteo OK - JFK {today}: {raw['JFK']}")
    return True


if __name__ == "__main__":
    print(ATTRIBUTION)
    print(f"Horizon: {forecast_horizon()[0]} .. {forecast_horizon()[1]}")
    self_test()
