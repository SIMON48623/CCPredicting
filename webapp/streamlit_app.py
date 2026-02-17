import os
import sys
import streamlit as st
import pandas as pd

# -------------------------
# Path setup so Streamlit Cloud can import project modules
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from final_model.predict_api import predict_one, explain_one_ig_png

st.set_page_config(page_title="Cervix Risk Predictor", layout="wide")


# -------------------------
# Variable dictionary (from your uploaded PDF)
# -------------------------
VAR_DICT = [
    {"variable": "age", "type": "int (years)", "meaning": "Age at visit/exam; higher age generally increases risk."},
    {"variable": "menopausal_status", "type": "0/1", "meaning": "0=pre-menopause, 1=post-menopause; post-menopause may affect TZ visibility and diagnosis."},
    {"variable": "gravidity", "type": "int", "meaning": "Number of pregnancies (including miscarriage/ectopic)."},
    {"variable": "parity", "type": "int", "meaning": "Number of deliveries (≥28 weeks, including stillbirth)."},
    {"variable": "child_alive", "type": "0/1", "meaning": "1=at least one living child; 0=none / no delivery."},
    {"variable": "HPV_overall", "type": "0/1", "meaning": "High-risk HPV positive (core predictor variable)."},
    {"variable": "HPV16", "type": "0/1", "meaning": "HPV16 status; HPV16 is associated with highest CIN3+ risk."},
    {"variable": "HPV18", "type": "0/1", "meaning": "HPV18 status; stronger association with adenocarcinoma."},
    {"variable": "HPV_other_hr", "type": "0/1", "meaning": "Other high-risk HPV types (non-16/18)."},
    {"variable": "cytology_grade", "type": "int (0–5)", "meaning": "0=NILM, 1=ASC-US, 2=ASC-H, 3=LSIL, 4=HSIL, 5=AGC."},
    {"variable": "colpo_impression", "type": "int (0–4)", "meaning": "0=normal, 1=mild, 2=moderate, 3=severe, 4=highly suspicious for CIN3+."},
    {"variable": "TZ_type", "type": "int (1/2/3)", "meaning": "Transformation zone type: 1=visible, 2=partly visible, 3=not visible."},
    {"variable": "iodine_negative", "type": "0/1", "meaning": "Iodine test negative (abnormal uptake)."},
    {"variable": "atypical_vessels", "type": "0/1", "meaning": "Presence of atypical vessels on colposcopy."},
    {"variable": "pathology_fig", "type": "int (0–10)", "meaning": "Clinician imaging score used in the dataset (project-specific)."},
]

VAR_LOOKUP = {d["variable"]: d for d in VAR_DICT}


def _help(v: str) -> str:
    d = VAR_LOOKUP.get(v)
    if not d:
        return ""
    return f"Type: {d['type']}\n\n{d['meaning']}"


def risk_band(p: float) -> str:
    if p < 0.10:
        return "Low"
    if p < 0.30:
        return "Intermediate"
    return "High"


# -------------------------
# Sidebar: inputs (each with explanation)
# -------------------------
with st.sidebar:
    st.header("Inputs")

    age = st.number_input("age", min_value=10, max_value=100, value=45, step=1, help=_help("age"))
    menopausal_status = st.selectbox(
        "menopausal_status",
        options=[0, 1],
        index=0,
        format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)",
        help=_help("menopausal_status"),
    )
    gravidity = st.number_input("gravidity", min_value=0, max_value=30, value=2, step=1, help=_help("gravidity"))
    parity = st.number_input("parity", min_value=0, max_value=20, value=1, step=1, help=_help("parity"))
    child_alive = st.selectbox(
        "child_alive",
        options=[0, 1],
        index=1,
        format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)",
        help=_help("child_alive"),
    )

    st.divider()
    st.subheader("HPV")
    HPV_overall = st.selectbox(
        "HPV_overall",
        options=[0, 1],
        index=1,
        format_func=lambda x: "0 (Negative)" if x == 0 else "1 (Positive)",
        help=_help("HPV_overall"),
    )
    HPV16 = st.selectbox(
        "HPV16",
        options=[0, 1],
        index=0,
        format_func=lambda x: "0 (Negative)" if x == 0 else "1 (Positive)",
        help=_help("HPV16"),
    )
    HPV18 = st.selectbox(
        "HPV18",
        options=[0, 1],
        index=0,
        format_func=lambda x: "0 (Negative)" if x == 0 else "1 (Positive)",
        help=_help("HPV18"),
    )
    HPV_other_hr = st.selectbox(
        "HPV_other_hr",
        options=[0, 1],
        index=0,
        format_func=lambda x: "0 (Negative)" if x == 0 else "1 (Positive)",
        help=_help("HPV_other_hr"),
    )

    st.divider()
    st.subheader("Cytology / Colposcopy")
    cytology_grade = st.number_input("cytology_grade", min_value=0, max_value=5, value=3, step=1, help=_help("cytology_grade"))
    colpo_impression = st.number_input("colpo_impression", min_value=0, max_value=4, value=2, step=1, help=_help("colpo_impression"))
    TZ_type = st.number_input("TZ_type", min_value=1, max_value=3, value=2, step=1, help=_help("TZ_type"))
    iodine_negative = st.selectbox(
        "iodine_negative",
        options=[0, 1],
        index=0,
        format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)",
        help=_help("iodine_negative"),
    )
    atypical_vessels = st.selectbox(
        "atypical_vessels",
        options=[0, 1],
        index=0,
        format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)",
        help=_help("atypical_vessels"),
    )

    st.divider()
    pathology_fig = st.number_input("pathology_fig", min_value=0, max_value=10, value=2, step=1, help=_help("pathology_fig"))

    st.divider()
    mode = st.selectbox("Decision mode", options=["triage", "screen", "youden"], index=0)
    ig_steps = st.slider("IG steps", min_value=16, max_value=96, value=48, step=8)
    run = st.button("Predict", type="primary")


record = {
    "age": int(age),
    "menopausal_status": int(menopausal_status),
    "gravidity": int(gravidity),
    "parity": int(parity),
    "child_alive": int(child_alive),
    "HPV_overall": int(HPV_overall),
    "HPV16": int(HPV16),
    "HPV18": int(HPV18),
    "HPV_other_hr": int(HPV_other_hr),
    "cytology_grade": int(cytology_grade),
    "colpo_impression": int(colpo_impression),
    "TZ_type": int(TZ_type),
    "iodine_negative": int(iodine_negative),
    "atypical_vessels": int(atypical_vessels),
    "pathology_fig": int(pathology_fig),
}


# -------------------------
# Main: ONLY the 4 blocks you requested
# -------------------------
st.title("Cervical Lesion Risk Prediction")

if run:
    pred = predict_one(record, mode=mode)
    png_bytes, ig_table, ig_meta = explain_one_ig_png(record, steps=int(ig_steps), top_k=10)

    # 1) Prediction result
    st.subheader("Prediction result")
    p = float(pred["prob"])
    band = risk_band(p)
    st.metric(label="Calibrated risk (probability)", value=f"{p:.4f}", delta=f"{band} band")
    st.write(
        {
            "label": pred["label"],
            "decision_mode": pred["decision_mode"],
            "threshold": pred["threshold"],
            "prob_raw": pred["prob_raw"],
        }
    )

    # 2) Single-case IG
    st.subheader("Integrated Gradients (single case)")
    st.image(png_bytes, caption="Top features by |IG| (signed IG shown)")
    st.dataframe(pd.DataFrame(ig_table), use_container_width=True)

    # 3) Input data table
    st.subheader("Input data")
    st.dataframe(pd.DataFrame([record]), use_container_width=True)

    # 4) Variable dictionary
    st.subheader("Variable dictionary")
    st.dataframe(pd.DataFrame(VAR_DICT), use_container_width=True)

else:
    # show input + dictionary even before predicting (helpful for Streamlit app)
    st.subheader("Input data")
    st.dataframe(pd.DataFrame([record]), use_container_width=True)
    st.subheader("Variable dictionary")
    st.dataframe(pd.DataFrame(VAR_DICT), use_container_width=True)
