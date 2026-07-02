"""CCPredicting: tabular deep-learning inference utilities for cervical lesion risk prediction."""

from .inference.predictor import CervixRiskPredictor, predict_one, explain_one_ig_png

__all__ = ["CervixRiskPredictor", "predict_one", "explain_one_ig_png"]
__version__ = "0.2.0"
