# data/ — every dataset in one place

| Folder | Put here | In git? |
| :--- | :--- | :--- |
| `raw/` | Original 2023 files: `US_flights_2023.csv`, `weather_meteo_by_airport.csv`, `Cancelled_Diverted_2023.csv`, `airports_geolocation.csv` | Only `airports_geolocation.csv` (others are too large) |
| `processed/` | Cleaned / merged outputs, e.g. `preprocessed_flight_data_final.csv` | No (large) |
| `lookups/` | Small tables the web app needs at startup: airport coordinates, airport delay rates, monthly weather averages | Yes |
| `weather/` | Open-Meteo calibration (`wx_calibration.json`) and validation results | Yes |
| `graph/` | **Phase 2** — airport adjacency matrix + hourly airport delay series for the ST-GNN | Decide when built |

Training scripts read the raw files from `data/raw/`, so download the full dataset there before retraining.
