# database/ — prediction logging (Phase 2)

Today the web app logs predictions to **MongoDB** (`aeropredict.predictions`, port 27017).

Phase 2 plan (owner **Dhruv Dharkar**, target **Nov 23**): replace MongoDB in
`webapp/flask_app.py` with **PostgreSQL** (`psycopg2`) so predictions can be queried in pgAdmin 4.
Keep the SQL schema / setup scripts for that here (e.g. `schema.sql`).
