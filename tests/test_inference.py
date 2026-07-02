import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccpredicting.inference.predictor import CervixRiskPredictor
from ccpredicting.schema import DEFAULT_RECORD, normalize_record


def test_alias_maps_to_internal_feature():
    record = normalize_record(DEFAULT_RECORD)
    assert "pathology_fig" in record
    assert record["pathology_fig"] == DEFAULT_RECORD["clinician_image_assessment"]


def test_predictor_outputs_probability_range():
    predictor = CervixRiskPredictor(model_dir=ROOT / "final_model", device="cpu")
    pred = predictor.predict_one(DEFAULT_RECORD, mode="balanced")
    assert 0.0 <= pred["prob"] <= 1.0
    assert 0.0 <= pred["prob_raw"] <= 1.0
    assert pred["decision_mode"] == "balanced"
    assert pred["label"] in {"positive", "negative"}


def test_legacy_mode_still_works():
    predictor = CervixRiskPredictor(model_dir=ROOT / "final_model", device="cpu")
    pred = predictor.predict_one(DEFAULT_RECORD, mode="triage")
    assert pred["legacy_decision_mode"] == "triage"
    assert pred["decision_mode"] == "balanced"
