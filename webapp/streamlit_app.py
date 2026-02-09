import os
import sys
import json
import io
import streamlit as st
import pandas as pd

# -------------------------
# Paths & imports
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from final_model.predict_api import predict_one  # final package API

st.set_page_config(page_title="Cervix Risk Predictor", layout="wide")

# -------------------------
# Helpers
# -------------------------
def risk_band(p: float):
    # simple bands for UI presentation
    if p < 0.10:
        return "Low"
    elif p < 0.30:
        return "Intermediate"
    else:
        return "High"

def guidance_text(band: str):
    if band == "High":
        return "High estimated risk. Recommend specialist evaluation / colposcopy follow-up as appropriate."
    if band == "Intermediate":
        return "Intermediate estimated risk. Consider closer follow-up and guideline-based triage."
    return "Lower estimated risk. Continue routine screening/follow-up per guidelines."

@st.cache_resource(show_spinner=False)
def load_meta():
    meta_path = os.path.join(PROJ_ROOT, "final_model", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def run_predict(record: dict, mode: str):
    # predict_api already loads artifacts internally; caching is done there or via OS cache.
    return predict_one(record, mode=mode)

def read_ig_table():
    p = os.path.join(PROJ_ROOT, "tables", "ig_global_top15.csv")
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

def read_png_if_exists(rel_path: str):
    p = os.path.join(PROJ_ROOT, rel_path)
    return p if os.path.exists(p) else None

# -------------------------
# Header
# -------------------------
st.title("Cervical Lesion Risk Prediction (Tab-MFM Transformer)")
st.caption("Research prototype. Not for direct clinical diagnosis. External validation required.")

meta = load_meta()

tab_single, tab_batch, tab_docs = st.tabs(["Single patient", "Batch CSV", "Model & docs"])

# =========================
# Tab 1: Single patient
# =========================
with tab_single:
    with st.sidebar:
        st.header("Inputs")

        age = st.number_input("Age", min_value=10, max_value=100, value=45, step=1)
        menopausal_status = st.selectbox("Menopausal status", options=[0, 1],
                                         format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        gravidity = st.number_input("Gravidity", min_value=0, max_value=30, value=2, step=1)
        parity = st.number_input("Parity", min_value=0, max_value=20, value=1, step=1)

        st.divider()
        st.subheader("HPV")
        HPV_overall = st.selectbox("HPV_overall", options=[0, 1],
                                   format_func=lambda x: "Negative (0)" if x == 0 else "Positive (1)")
        HPV16 = st.selectbox("HPV16", options=[0, 1],
                             format_func=lambda x: "Negative (0)" if x == 0 else "Positive (1)")
        HPV18 = st.selectbox("HPV18", options=[0, 1],
                             format_func=lambda x: "Negative (0)" if x == 0 else "Positive (1)")
        HPV_other_hr = st.selectbox("HPV_other_hr", options=[0, 1],
                                    format_func=lambda x: "Negative (0)" if x == 0 else "Positive (1)")

        st.divider()
        st.subheader("Cytology / Colposcopy")
        cytology_grade = st.number_input("cytology_grade (ordinal)", min_value=0, max_value=10, value=3, step=1)
        colpo_impression = st.number_input("colpo_impression (ordinal)", min_value=0, max_value=10, value=2, step=1)
        TZ_type = st.number_input("TZ_type (ordinal)", min_value=0, max_value=10, value=2, step=1)
        iodine_negative = st.selectbox("iodine_negative", options=[0, 1],
                                       format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        atypical_vessels = st.selectbox("atypical_vessels", options=[0, 1],
                                        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")

        st.divider()
        st.subheader("Other")
        child_alive = st.selectbox("child_alive", options=[0, 1],
                                   format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")

        pathology_fig = st.number_input("pathology_fig (clinician imaging score)",
                                        min_value=0, max_value=10, value=2, step=1)

        st.divider()
        st.subheader("Decision")
        mode = st.selectbox("Decision mode", options=["triage", "screen", "youden"], index=0)

        run = st.button("Predict", type="primary")

    record = {
        "age": int(age),
        "menopausal_status": int(menopausal_status),
        "gravidity": int(gravidity),
        "parity": int(parity),
        "HPV_overall": int(HPV_overall),
        "HPV16": int(HPV16),
        "HPV18": int(HPV18),
        "HPV_other_hr": int(HPV_other_hr),
        "cytology_grade": int(cytology_grade),
        "colpo_impression": int(colpo_impression),
        "TZ_type": int(TZ_type),
        "iodine_negative": int(iodine_negative),
        "atypical_vessels": int(atypical_vessels),
        "child_alive": int(child_alive),
        "pathology_fig": int(pathology_fig),
    }

    col1, col2 = st.columns([1.2, 1.0], gap="large")

    with col1:
        st.subheader("Prediction result")

        if run:
            pred = run_predict(record, mode=mode)

            prob = float(pred["prob"])
            prob_raw = float(pred.get("prob_raw", prob))
            thr = float(pred["threshold"])
            label = str(pred["label"])

            band = risk_band(prob)

            st.metric("Calibrated risk probability", f"{prob:.3f}")
            st.caption(f"Raw (uncalibrated) probability: {prob_raw:.3f}")
            st.write(f"**Decision mode**: `{pred['decision_mode']}`")
            st.write(f"**Threshold**: `{thr:.3f}`")
            st.write(f"**Risk band**: **{band}**")
            st.write(f"**Predicted label**: **{label.upper()}**")

            if label.lower() == "positive":
                st.error(guidance_text(band))
            else:
                st.success(guidance_text(band))

            with st.expander("Show raw output JSON"):
                st.code(json.dumps(pred, ensure_ascii=False, indent=2), language="json")
        else:
            st.info("Fill inputs in the sidebar, then click **Predict**.")

    with col2:
        st.subheader("Inputs preview")
        st.dataframe(pd.DataFrame([record]))

        st.subheader("Global IG (optional)")
        ig = read_ig_table()
        if ig is not None:
            st.dataframe(ig)
        else:
            st.warning("Not found: tables/ig_global_top15.csv")

    st.divider()
    st.subheader("Interpretability figures (if available)")
    ig_one_png = read_png_if_exists(os.path.join("figures", "ig_one_demo_top10.png"))
    ig_glb_png = read_png_if_exists(os.path.join("figures", "ig_global_top15.png"))

    cA, cB = st.columns(2)
    with cA:
        st.caption("Single-case IG (Top10)")
        if ig_one_png:
            st.image(ig_one_png, use_column_width=True)
        else:
            st.info("Not found: figures/ig_one_demo_top10.png")
    with cB:
        st.caption("Global IG (Top15)")
        if ig_glb_png:
            st.image(ig_glb_png, use_column_width=True)
        else:
            st.info("Not found: figures/ig_global_top15.png")

# =========================
# Tab 2: Batch CSV
# =========================
with tab_batch:
    st.subheader("Batch prediction by CSV upload")
    st.caption("Upload a CSV containing the same columns as the input schema. Extra columns are allowed and will be kept.")

    mode_b = st.selectbox("Decision mode (batch)", options=["triage", "screen", "youden"], index=0, key="mode_batch")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.write("Preview:")
        st.dataframe(df.head(10))

        if st.button("Run batch prediction", type="primary"):
            outputs = []
            for _, row in df.iterrows():
                rec = row.to_dict()
                pred = run_predict(rec, mode=mode_b)
                outputs.append({
                    "prob_raw": float(pred.get("prob_raw", pred["prob"])),
                    "prob": float(pred["prob"]),
                    "threshold": float(pred["threshold"]),
                    "label": pred["label"],
                })
            out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
            st.success(f"Done. Rows: {len(out_df)}")
            st.dataframe(out_df.head(20))

            # Download
            buf = io.StringIO()
            out_df.to_csv(buf, index=False)
            st.download_button(
                label="Download predictions CSV",
                data=buf.getvalue().encode("utf-8"),
                file_name="patients_with_predictions.csv",
                mime="text/csv"
            )

# =========================
# Tab 3: Model & docs
# =========================
with tab_docs:
    st.subheader("Model metadata & thresholds")
    if meta:
        st.json(meta)
    else:
        st.info("metadata.json not found or empty.")

    st.subheader("Notes")
    st.write(
        "- `prob` is calibrated probability (Platt sigmoid).\n"
        "- `prob_raw` is uncalibrated.\n"
        "- Threshold modes are for research/triage demonstration.\n"
        "- Not for direct clinical diagnosis."
    )
