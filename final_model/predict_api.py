"""Backward-compatible inference wrapper.

The maintained implementation lives under ``src/ccpredicting``. This module is
kept so older commands such as ``from final_model.predict_api import
predict_one`` continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ccpredicting.inference.predictor import CervixRiskPredictor, explain_one_ig_png, predict_one

__all__ = ["CervixRiskPredictor", "predict_one", "explain_one_ig_png"]
