"""Compatibility wrapper.

The maintained Streamlit app is now ``app/streamlit_app.py``. Run:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
exec(APP_PATH.read_text(encoding="utf-8"), globals())
