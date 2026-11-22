"""
AeroPredict — start the web app
================================
    python run_app.py

Opens on http://127.0.0.1:5000/  (same as running webapp/flask_app.py directly).
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "webapp" / "flask_app.py"), run_name="__main__")
