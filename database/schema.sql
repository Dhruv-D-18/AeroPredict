-- ============================================================
-- AeroPredict 2.0: PostgreSQL Relational Database Schema
-- Table: public.flight_predictions (Matches your active pgAdmin 4)
-- Server: PostgreSQL 18 | Database: postgres | User: postgres
-- ============================================================

CREATE TABLE IF NOT EXISTS public.flight_predictions (
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

-- Indexes for Fast Querying in pgAdmin 4
CREATE INDEX IF NOT EXISTS idx_flight_predictions_created_at ON public.flight_predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_flight_predictions_route ON public.flight_predictions(route);
CREATE INDEX IF NOT EXISTS idx_flight_predictions_airline ON public.flight_predictions(airline);
CREATE INDEX IF NOT EXISTS idx_flight_predictions_risk ON public.flight_predictions(risk_level);

-- ============================================================
-- Helpful Analytical Queries to Run in pgAdmin 4 for Your Viva:
-- ============================================================

-- 1. View all recorded predictions (Newest First):
-- SELECT * FROM public.flight_predictions ORDER BY id DESC;

-- 2. Breakdown of Predictions by Risk Level:
-- SELECT risk_level, COUNT(*) as count, ROUND(AVG(delay_prob)::numeric * 100, 1) as avg_delay_pct 
-- FROM public.flight_predictions GROUP BY risk_level ORDER BY count DESC;

-- 3. Top Most Frequent Routes Tested:
-- SELECT route, COUNT(*) as query_count, ROUND(AVG(est_delay_mins)::numeric, 1) as avg_delay_mins
-- FROM public.flight_predictions WHERE est_delay_mins IS NOT NULL 
-- GROUP BY route ORDER BY query_count DESC LIMIT 5;
