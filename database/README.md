# database/ — PostgreSQL Prediction Logging (Phase 2)

The web app logs runtime flight predictions and serves the `/history` dashboard using **PostgreSQL** (port 5432).

* `schema.sql`: Full SQL schema for `flight_predictions` table with performance indexes and sample viva analytical queries.
* `db_postgres.py`: PostgreSQL connection manager (via `psycopg2`) with zero-fail local fallback.
* Configuration: Set `PG_USER`, `PG_PASSWORD`, `PG_HOST`, `PG_PORT` in `.env` (defaults to `postgres/postgres` on `localhost:5432`).
