# webapp/ — Flask website

```bash
python run_app.py              # from the project root  (or: python webapp/flask_app.py)
```

| Path | What it is |
| :--- | :--- |
| `flask_app.py` | Routes, prediction logic, MongoDB logging |
| `templates/` | Pages: `home`, `predict`, `visualizations`, `history`, `settings` (+ shared `base.html`) |
| `static/styles.css` | Design system for every page |
| `static/js/hero3d.js` | 3D aircraft on the home page (Three.js) |
| `static/vendor/` | Bundled Three.js + Inter font (works offline) |
| `static/viz_data/` | Pre-computed JSON used by the Analytics charts |

`templates/network.html` (route `/network`) is the Phase 2 dashboard: disruption early-warning map with
+1/+2/+3 slot forecasts and 95 % bands, bottleneck ranking, delay propagation, what-if simulator and
the model comparison. API: `/api/network/meta`, `/api/network/forecast`, `/api/network/simulate`,
`/api/network/propagation`, `/api/model-comparison`, `/api/whatif/flight` (departure-time what-if on the predict page).
