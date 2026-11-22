# ── locate project folders (see core/project_paths.py) ──
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "project_paths.py").exists())
_sys.path.insert(0, str(_ROOT / "core"))
import project_paths as P

import pandas as pd

# 1. Monthly weather averages per airport
print('Computing weather averages...')
weather = pd.read_csv(P.RAW_WEATHER)
weather['time'] = pd.to_datetime(weather['time'])
weather['month'] = weather['time'].dt.month
weather_avg = weather.groupby(['airport_id', 'month'])[['tavg','prcp','wspd','pres']].mean().round(2)
weather_avg.to_csv(P.WEATHER_AVERAGES)
print('Done. Sample:')
print(weather_avg.head(5).to_string())

# 2. Typical flight duration and distance type per route
print('\nComputing route stats...')
df = pd.read_csv(P.PROCESSED_V2,
                 usecols=['Dep_Airport','Arr_Airport','Flight_Duration','Distance_type'])
route_stats = df.groupby(['Dep_Airport','Arr_Airport']).agg(
    avg_duration=('Flight_Duration','median'),
    distance_type=('Distance_type', lambda x: x.mode()[0])
).reset_index()
route_stats.to_csv(P.ROUTE_STATS, index=False)
print(f'Route stats saved. {len(route_stats)} unique routes found.')

# 3. Aircraft age median
df2 = pd.read_csv(P.PROCESSED_V2, usecols=['Aicraft_age'])
median_age = df2['Aicraft_age'].median()
print(f'Median aircraft age: {median_age}')
