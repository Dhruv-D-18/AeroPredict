"""
AeroPredict PostgreSQL Database Manager
========================================
Matches the exact schema in your pgAdmin 4 (PostgreSQL 18):
  table: public.flight_predictions
  columns: id, created_at, airline, dep_airport, arr_airport, route,
           travel_date, dep_time, delay_prob, on_time_prob, est_delay_mins,
           risk_level, verdict, model_type, flight_duration, weather_json
"""
from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# Paths
DB_DIR = Path(__file__).resolve().parent
SQLITE_PATH = DB_DIR / "aeropredict_local.db"
ENV_PATH = DB_DIR.parent / ".env"

# Load .env if present
if ENV_PATH.exists():
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass

# PostgreSQL Configuration
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", 5432))
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "postgres")
PG_DBNAME = os.environ.get("PG_DBNAME", "postgres")

# Global status
db_ok = False
db_engine = "None"
db_error = None
_psycopg2 = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    _psycopg2 = psycopg2
except ImportError:
    _psycopg2 = None


def _get_pg_conn():
    if _psycopg2 is None:
        raise ImportError("psycopg2 is not installed")
    return _psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DBNAME,
        connect_timeout=2
    )


def _init_sqlite():
    conn = sqlite3.connect(str(SQLITE_PATH))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS flight_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            airline TEXT,
            dep_airport TEXT,
            arr_airport TEXT,
            route TEXT,
            travel_date TEXT,
            dep_time TEXT,
            delay_prob REAL,
            on_time_prob REAL,
            est_delay_mins INTEGER,
            risk_level TEXT,
            verdict TEXT,
            model_type TEXT,
            flight_duration REAL,
            weather_json TEXT
        );
    """)
    conn.commit()
    conn.close()


def init_database():
    """Initializes connection to the exact pgAdmin 4 table."""
    global db_ok, db_engine, db_error

    # 1. Try PostgreSQL first (pgAdmin 4 table)
    if _psycopg2 is not None:
        try:
            conn = _get_pg_conn()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS flight_predictions (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    airline VARCHAR(100),
                    dep_airport VARCHAR(10),
                    arr_airport VARCHAR(10),
                    route VARCHAR(30),
                    travel_date VARCHAR(20),
                    dep_time VARCHAR(20),
                    delay_prob REAL,
                    on_time_prob REAL,
                    est_delay_mins INTEGER,
                    risk_level VARCHAR(20),
                    verdict VARCHAR(100),
                    model_type VARCHAR(100),
                    flight_duration REAL,
                    weather_json TEXT
                );
            """)
            conn.commit()
            cur.close()
            conn.close()
            db_ok = True
            db_engine = "PostgreSQL (pgAdmin 4)"
            db_error = None
            print(f"Database connected — PostgreSQL ({PG_HOST}:{PG_PORT}/{PG_DBNAME})")
            return True, "PostgreSQL"
        except Exception as e:
            db_error = str(e)

    # 2. Safety Fallback to SQLite so web app never crashes
    try:
        _init_sqlite()
        db_ok = True
        db_engine = "SQLite (Local Fallback)"
        return True, "SQLite"
    except Exception as e:
        db_ok = False
        db_engine = "None"
        db_error = str(e)
        return False, str(e)


def save_prediction(doc: dict) -> bool:
    """Inserts a prediction record matching pgAdmin 4 table schema."""
    if not db_ok:
        return False

    ts = doc.get("timestamp") or datetime.now()
    if isinstance(ts, datetime):
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts_str = str(ts)

    weather_raw = doc.get("weather", {})
    weather_str = json.dumps(weather_raw) if isinstance(weather_raw, dict) else str(weather_raw)

    airline = doc.get("airline") or doc.get("airline_display") or ""
    dep = doc.get("dep_airport", "")
    arr = doc.get("arr_airport", "")
    route = doc.get("route") or f"{dep} → {arr}"
    travel_date = doc.get("travel_date") or doc.get("flight_date") or ""
    dep_time = doc.get("dep_time", "")
    delay_prob = float(doc.get("delay_prob", 0.0))
    on_time_prob = float(doc.get("on_time_prob", 0.0))
    est_delay = doc.get("est_delay_mins")
    risk = doc.get("risk") or doc.get("risk_level") or "low"
    verdict = doc.get("verdict") or doc.get("prediction") or ""
    model_type = doc.get("model_used") or doc.get("model_type") or "DNN V2"
    duration = float(doc.get("flight_duration") or 0.0)

    # 1. Try PostgreSQL if active
    if db_engine.startswith("PostgreSQL"):
        try:
            conn = _get_pg_conn()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO flight_predictions (
                    created_at, airline, dep_airport, arr_airport, route,
                    travel_date, dep_time, delay_prob, on_time_prob,
                    est_delay_mins, risk_level, verdict, model_type,
                    flight_duration, weather_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                ts, airline, dep, arr, route, travel_date, dep_time,
                delay_prob, on_time_prob, est_delay, risk, verdict,
                model_type, duration, weather_str
            ))
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"PostgreSQL insert error: {e}")

    # 2. Safety Fallback: SQLite
    try:
        conn = sqlite3.connect(str(SQLITE_PATH))
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO flight_predictions (
                created_at, airline, dep_airport, arr_airport, route,
                travel_date, dep_time, delay_prob, on_time_prob,
                est_delay_mins, risk_level, verdict, model_type,
                flight_duration, weather_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            ts_str, airline, dep, arr, route, travel_date, dep_time,
            delay_prob, on_time_prob, est_delay, risk, verdict,
            model_type, duration, weather_str
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"SQLite insert error: {e}")
        return False


def get_history(limit: int = 25) -> list[dict]:
    """Fetches records matching pgAdmin 4 schema for /history view."""
    if not db_ok:
        return []

    if db_engine.startswith("PostgreSQL"):
        try:
            conn = _get_pg_conn()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT 
                    id,
                    TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') as created_at,
                    TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') as timestamp,
                    airline, dep_airport, arr_airport, route,
                    travel_date, dep_time, delay_prob, on_time_prob,
                    est_delay_mins, risk_level as risk, verdict, model_type as model_used,
                    flight_duration, weather_json
                FROM flight_predictions
                ORDER BY id DESC
                LIMIT %s;
            """, (limit,))
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
            conn.close()
            return rows
        except Exception as e:
            print(f"PostgreSQL fetch error: {e}")

    try:
        conn = sqlite3.connect(str(SQLITE_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                id, created_at, created_at as timestamp,
                airline, dep_airport, arr_airport, route,
                travel_date, dep_time, delay_prob, on_time_prob,
                est_delay_mins, risk_level as risk, verdict, model_type as model_used,
                flight_duration, weather_json
            FROM flight_predictions
            ORDER BY id DESC
            LIMIT ?;
        """, (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"SQLite fetch error: {e}")
        return []


init_database()
