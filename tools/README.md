# tools/ — one-off utilities

| Script | Purpose |
| :--- | :--- |
| `weather_calibration/calibrate_open_meteo.py` | Fits `data/weather/wx_calibration.json` |
| `weather_calibration/validate_open_meteo.py` | 4-way weather-source comparison → `data/weather/wx_validation.json` |
| `lookups/compute_lookups.py` | Rebuilds `data/lookups/weather_averages.csv` and route stats |
