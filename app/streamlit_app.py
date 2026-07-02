"""Streamlit UI for the CCPredicting model.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ccpredicting.inference.predictor import CervixRiskPredictor
from ccpredicting.schema import DEFAULT_RECORD, PUBLIC_CLINICIAN_IMAGE_FIELD, VARIABLE_DICTIONARY

st.set_page_config(page_title="CCPredicting", layout="wide")


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_predictor() -> CervixRiskPredictor:
    return CervixRiskPredictor(model_dir=PROJECT_ROOT / "final_model")


def risk_band(probability: float) -> str:
    if probability < 0.10:
        return "Low"
    if probability < 0.30:
        return "Intermediate"
    return "High"


MODE_HELP = (
    "**high_sensitivity** uses the lower legacy screening threshold and is intended to reduce false negatives.\n\n"
    "**balanced** uses the legacy triage threshold and provides a more balanced sensitivity/specificity trade-off.\n\n"
    "**youden** uses the Youden-index threshold stored with the exported artifact."
)

IG_HELP = (
    "Integrated Gradients explains one prediction by approximating how each input feature contributes relative to a baseline. "
    "More steps are slower but usually more stable."
)

VARIABLE_LOOKUP = {item["variable"]: item for item in VARIABLE_DICTIONARY}


def help_text(variable: str) -> str:
    item = VARIABLE_LOOKUP.get(variable, {})
    return f"Type: {item.get('type', '')}\n\nAllowed: {item.get('allowed', '')}\n\n{item.get('description', '')}"


st.title("CCPredicting: Cervical High-Grade Lesion Risk Prediction")
st.caption(
    "Research prototype for structured clinical risk prediction. This app is not a medical device and must not be used as a standalone clinical decision system."
)

with st.sidebar:
    st.header("Input features")
    age = st.number_input("age", min_value=10, max_value=100, value=DEFAULT_RECORD["age"], step=1, help=help_text("age"))
    menopausal_status = st.selectbox("menopausal_status", [0, 1], index=DEFAULT_RECORD["menopausal_status"], help=help_text("menopausal_status"))
    gravidity = st.number_input("gravidity", min_value=0, max_value=30, value=DEFAULT_RECORD["gravidity"], step=1, help=help_text("gravidity"))
    parity = st.number_input("parity", min_value=0, max_value=20, value=DEFAULT_RECORD["parity"], step=1, help=help_text("parity"))
    child_alive = st.selectbox("child_alive", [0, 1], index=DEFAULT_RECORD["child_alive"], help=help_text("child_alive"))

    st.divider()
    st.subheader("HPV")
    hpv_overall = st.selectbox("HPV_overall", [0, 1], index=DEFAULT_RECORD["HPV_overall"], help=help_text("HPV_overall"))
    hpv16 = st.selectbox("HPV16", [0, 1], index=DEFAULT_RECORD["HPV16"], help=help_text("HPV16"))
    hpv18 = st.selectbox("HPV18", [0, 1], index=DEFAULT_RECORD["HPV18"], help=help_text("HPV18"))
    hpv_other_hr = st.selectbox("HPV_other_hr", [0, 1], index=DEFAULT_RECORD["HPV_other_hr"], help=help_text("HPV_other_hr"))

    st.divider()
    st.subheader("Cytology / colposcopy")
    cytology_grade = st.number_input("cytology_grade", min_value=0, max_value=5, value=DEFAULT_RECORD["cytology_grade"], step=1, help=help_text("cytology_grade"))
    colpo_impression = st.number_input("colpo_impression", min_value=0, max_value=4, value=DEFAULT_RECORD["colpo_impression"], step=1, help=help_text("colpo_impression"))
    tz_type = st.number_input("TZ_type", min_value=1, max_value=3, value=DEFAULT_RECORD["TZ_type"], step=1, help=help_text("TZ_type"))
    iodine_negative = st.selectbox("iodine_negative", [0, 1], index=DEFAULT_RECORD["iodine_negative"], help=help_text("iodine_negative"))
    atypical_vessels = st.selectbox("atypical_vessels", [0, 1], index=DEFAULT_RECORD["atypical_vessels"], help=help_text("atypical_vessels"))
    clinician_image_assessment = st.selectbox(
        PUBLIC_CLINICIAN_IMAGE_FIELD,
        [0, 1],
        index=DEFAULT_RECORD[PUBLIC_CLINICIAN_IMAGE_FIELD],
        help=help_text(PUBLIC_CLINICIAN_IMAGE_FIELD),
    )

    st.divider()
    mode = st.selectbox("Decision mode", ["high_sensitivity", "balanced", "youden"], index=1, help=MODE_HELP)
    ig_steps = st.slider("Integrated Gradients steps", min_value=16, max_value=96, value=48, step=8, help=IG_HELP)
    run = st.button("Predict", type="primary")

record = {
    "age": int(age),
    "menopausal_status": int(menopausal_status),
    "gravidity": int(gravidity),
    "parity": int(parity),
    "child_alive": int(child_alive),
    "HPV_overall": int(hpv_overall),
    "HPV16": int(hpv16),
    "HPV18": int(hpv18),
    "HPV_other_hr": int(hpv_other_hr),
    "cytology_grade": int(cytology_grade),
    "colpo_impression": int(colpo_impression),
    "TZ_type": int(tz_type),
    "iodine_negative": int(iodine_negative),
    "atypical_vessels": int(atypical_vessels),
    PUBLIC_CLINICIAN_IMAGE_FIELD: int(clinician_image_assessment),
}

left, right = st.columns([1.1, 1.0])
with left:
    st.subheader("Input record")
    st.dataframe(pd.DataFrame([record]), use_container_width=True)
with right:
    st.subheader("Variable dictionary")
    st.dataframe(pd.DataFrame(VARIABLE_DICTIONARY), use_container_width=True, hide_index=True)

if run:
    predictor = load_predictor()
    pred = predictor.predict_one(record, mode=mode)
    png_bytes, ig_table, _ = predictor.explain_one_ig_png(record, steps=int(ig_steps), top_k=10)

    st.divider()
    st.subheader("Prediction result")
    probability = float(pred["prob"])
    band = risk_band(probability)
    label = str(pred["label"]).upper()

    cols = st.columns(4)
    cols[0].metric("Calibrated risk", f"{probability:.4f}")
    cols[1].metric("Risk band", band)
    cols[2].metric("Decision", label)
    cols[3].metric("Threshold", f"{float(pred['threshold']):.2f}")

    st.info(
        f"Mode: {pred['decision_mode']} | Raw probability: {float(pred['prob_raw']):.4f} | "
        f"Calibration: {pred['meta'].get('calibration', 'unknown')}"
    )

    st.subheader("Integrated Gradients explanation")
    st.image(png_bytes, caption="Top features by absolute Integrated Gradients magnitude; bars retain signed attribution.")
    st.dataframe(pd.DataFrame(ig_table), use_container_width=True, hide_index=True)
