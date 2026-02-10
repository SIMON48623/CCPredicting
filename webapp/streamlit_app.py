import os
import sys
import time
import json
import io
import streamlit as st
import pandas as pd

# =========================
# Project import (backend is NOT displayed in UI)
# =========================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from final_model.predict_api import predict_one  # backend call; NOT shown on the webpage


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Cervix Risk Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Global CSS (UI only)
# =========================
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1200px; }
div[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.06); }
h1, h2, h3 { letter-spacing: -0.02em; }

.badge {
  display:inline-block; padding: .25rem .55rem; border-radius: 999px;
  border: 1px solid rgba(0,0,0,.08); font-size: .82rem;
  background: rgba(255,255,255,.7);
}
.hero {
  padding: 1.0rem 1.2rem;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(16,185,129,.10));
  border: 1px solid rgba(0,0,0,.06);
}
.hero-title { font-size: 1.55rem; font-weight: 760; margin: 0; }
.hero-sub { margin-top:.35rem; color: rgba(0,0,0,.65); }

.card {
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 18px;
  padding: 1rem 1rem;
  background: rgba(255,255,255,.85);
}
.card-title { font-weight: 700; margin-bottom: .35rem; }
.hr-soft { height: 1px; background: rgba(0,0,0,.06); margin: .8rem 0; }

.risk-banner {
  border-radius: 16px;
  padding: .9rem 1rem;
  border: 1px solid rgba(0,0,0,.08);
}
.risk-low { background: rgba(16,185,129,.10); }
.risk-mid { background: rgba(245,158,11,.12); }
.risk-high { background: rgba(239,68,68,.12); }
.risk-label { font-weight: 800; font-size: 1.05rem; }

.small-muted { color: rgba(0,0,0,0.6); font-size: 0.92rem; }
.stButton>button { border-radius: 14px !important; padding: .62rem 1.05rem !important; font-weight: 650 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Variable dictionary (from your PDF dictionary; EN phrasing)
# =========================
VAR_DICT = {
    "age": {
        "label": "Age (years)",
        "type": "Integer",
        "how": "Age at visit/examination.",
        "meaning": "Older age is associated with higher risk of cervical lesions.",
        "values": "Integer (years)",
    },
    "menopausal_status": {
        "label": "Menopausal status",
        "type": "Binary (0/1)",
        "how": "0 = Premenopausal; 1 = Postmenopausal.",
        "meaning": "Postmenopause may shift the transformation zone and affect evaluation.",
        "values": "0 or 1",
    },
    "gravidity": {
        "label": "Gravidity",
        "type": "Integer",
        "how": "Number of pregnancies (including miscarriage/ectopic).",
        "meaning": "Reproductive history descriptor.",
        "values": "0+",
    },
    "parity": {
        "label": "Parity",
        "type": "Integer",
        "how": "Number of deliveries (≥28 weeks, including stillbirth).",
        "meaning": "Reproductive history descriptor.",
        "values": "0+",
    },
    "child_alive": {
        "label": "Any living child",
        "type": "Binary (0/1)",
        "how": "1 = has ≥1 living child; 0 = none / no delivery.",
        "meaning": "Delivery outcome information used for research on pregnancy–lesion associations.",
        "values": "0 or 1",
    },
    "HPV_overall": {
        "label": "High-risk HPV positive (overall)",
        "type": "Binary (0/1)",
        "how": "Overall high-risk HPV status.",
        "meaning": "Core predictor in the model.",
        "values": "0 or 1",
    },
    "HPV16": {
        "label": "HPV16 positive",
        "type": "Binary (0/1)",
        "how": "HPV16 infection status.",
        "meaning": "HPV16 is associated with the highest CIN3+ risk.",
        "values": "0 or 1",
    },
    "HPV18": {
        "label": "HPV18 positive",
        "type": "Binary (0/1)",
        "how": "HPV18 infection status.",
        "meaning": "HPV18 is more strongly associated with adenocarcinoma-related lesions.",
        "values": "0 or 1",
    },
    "HPV_other_hr": {
        "label": "Other high-risk HPV positive",
        "type": "Binary (0/1)",
        "how": "Other high-risk HPV types.",
        "meaning": "High-risk HPV positivity beyond HPV16/18.",
        "values": "0 or 1",
    },
    "cytology_grade": {
        "label": "Cytology grade",
        "type": "Ordinal integer (0–5)",
        "how": "0=NILM, 1=ASC-US, 2=ASC-H, 3=LSIL, 4=HSIL, 5=AGC.",
        "meaning": "Cytology severity grade used as a key clinical predictor.",
        "values": "0–5",
    },
    "colpo_impression": {
        "label": "Colposcopy impression",
        "type": "Ordinal integer (0–4)",
        "how": "0=Normal; 1=Mild suspicion; 2=Moderate; 3=Severe; 4=Highly suspicious for CIN3+.",
        "meaning": "Clinician’s impression during colposcopy.",
        "values": "0–4",
    },
    "TZ_type": {
        "label": "Transformation zone (TZ) type",
        "type": "Ordinal (1–3)",
        "how": "1=Visible; 2=Partly visible; 3=Not visible.",
        "meaning": "TZ visibility can affect evaluation and sampling.",
        "values": "1–3",
    },
    "iodine_negative": {
        "label": "Iodine test negative",
        "type": "Binary (0/1)",
        "how": "Schiller test iodine staining negative.",
        "meaning": "Negative staining may suggest abnormal epithelium.",
        "values": "0 or 1",
    },
    "atypical_vessels": {
        "label": "Atypical vessels",
        "type": "Binary (0/1)",
        "how": "Presence of atypical vessels on colposcopy.",
        "meaning": "May be associated with higher-grade disease.",
        "values": "0 or 1",
    },
    # Project rule: pathology_fig is a normal predictor; NOT used to construct y (label source is pathology_group).
    "pathology_fig": {
        "label": "Pathology fig (ordinary clinical variable)",
        "type": "Integer (project-defined)",
        "how": "Enter as recorded in the dataset.",
        "meaning": "Treated as a routine clinical variable and NOT used to construct the label.",
        "values": "Project-defined (UI allows 0–10)",
    },
}

FEATURE_ORDER = [
    "age", "menopausal_status", "gravidity", "parity",
    "HPV_overall", "HPV16", "HPV18", "HPV_other_hr",
    "cytology_grade", "colpo_impression", "TZ_type",
    "iodine_negative", "atypical_vessels", "child_alive",
    "pathology_fig",
]

def help_text(k: str) -> str:
    v = VAR_DICT[k]
    return (
        f"Type: {v['type']}\n\n"
        f"How to fill: {v['how']}\n\n"
        f"Meaning: {v['meaning']}\n\n"
        f"Valid values: {v['values']}"
    )

def risk_band(prob: float) -> str:
    if prob < 0.10:
        return "Low"
    if prob < 0.30:
        return "Intermediate"
    return "High"

def band_class(band: str) -> str:
    return {"Low": "risk-low", "Intermediate": "risk-mid", "High": "risk-high"}.get(band, "risk-mid")

def band_guidance(band: str) -> str:
    if band == "High":
        return "Higher risk: consider guideline-based referral for specialist evaluation/colposcopy (research demo)."
    if band == "Intermediate":
        return "Intermediate risk: consider closer follow-up and guideline-based triage (research demo)."
    return "Lower risk: continue routine screening/follow-up per guidelines (research demo)."

def mode_explain(mode: str) -> str:
    if mode == "triage":
        return "High-sensitivity triage: prioritizes fewer missed cases (more positive alerts)."
    if mode == "youden":
        return "Youden operating point: balances sensitivity and specificity."
    if mode == "screen":
        return "Screening-oriented mode (project-defined thresholding)."
    return ""

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def make_template_csv() -> bytes:
    demo = {k: 0 for k in FEATURE_ORDER}
    demo.update({
        "age": 45,
        "menopausal_status": 0,
        "gravidity": 2,
        "parity": 1,
        "child_alive": 1,
        "HPV_overall": 1,
        "HPV16": 0,
        "HPV18": 0,
        "HPV_other_hr": 1,
        "cytology_grade": 3,
        "colpo_impression": 2,
        "TZ_type": 2,
        "iodine_negative": 0,
        "atypical_vessels": 0,
        "pathology_fig": 2,
    })
    df = pd.DataFrame([demo])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# =========================
# Header
# =========================
st.markdown(
    """
<div class="hero">
  <div class="badge">🧬 Calibrated Risk • Research Prototype</div>
  <p class="hero-title">Cervical Lesion Risk Prediction</p>
  <p class="hero-sub">Structured inputs → calibrated risk probability + decision mode output (not for clinical diagnosis).</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("⚠️ Disclaimer (click to expand)", expanded=False):
    st.markdown(
        """
- **Research prototype. Not for clinical diagnosis.**
- Developed on a single-center retrospective dataset; external validation is required before any clinical use.
- Do not enter personally identifiable information.
"""
    )

tab_single, tab_batch, tab_dict = st.tabs(["👤 Single-case", "📦 Batch", "📚 Data Dictionary"])


# =========================
# Sidebar: Inputs
# =========================
with st.sidebar:
    st.header("Inputs")
    st.caption("Each field includes a tooltip explanation (ⓘ).")

    # Demographics / reproductive
    age = st.number_input(VAR_DICT["age"]["label"], min_value=10, max_value=100, value=45, step=1, help=help_text("age"))
    menopausal_status = st.selectbox(
        VAR_DICT["menopausal_status"]["label"],
        options=[0, 1],
        format_func=lambda x: "0 = Premenopausal" if x == 0 else "1 = Postmenopausal",
        help=help_text("menopausal_status"),
    )
    gravidity = st.number_input(VAR_DICT["gravidity"]["label"], min_value=0, max_value=30, value=2, step=1, help=help_text("gravidity"))
    parity = st.number_input(VAR_DICT["parity"]["label"], min_value=0, max_value=20, value=1, step=1, help=help_text("parity"))
    child_alive = st.selectbox(
        VAR_DICT["child_alive"]["label"],
        options=[0, 1],
        format_func=lambda x: "0 = No" if x == 0 else "1 = Yes",
        help=help_text("child_alive"),
    )

    st.divider()
    st.subheader("HPV")
    HPV_overall = st.selectbox(VAR_DICT["HPV_overall"]["label"], [0, 1], format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive", help=help_text("HPV_overall"))
    HPV16 = st.selectbox(VAR_DICT["HPV16"]["label"], [0, 1], format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive", help=help_text("HPV16"))
    HPV18 = st.selectbox(VAR_DICT["HPV18"]["label"], [0, 1], format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive", help=help_text("HPV18"))
    HPV_other_hr = st.selectbox(VAR_DICT["HPV_other_hr"]["label"], [0, 1], format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive", help=help_text("HPV_other_hr"))

    st.divider()
    st.subheader("Cytology / Colposcopy")
    cytology_grade = st.number_input(VAR_DICT["cytology_grade"]["label"], min_value=0, max_value=5, value=3, step=1, help=help_text("cytology_grade"))
    colpo_impression = st.number_input(VAR_DICT["colpo_impression"]["label"], min_value=0, max_value=4, value=2, step=1, help=help_text("colpo_impression"))
    TZ_type = st.number_input(VAR_DICT["TZ_type"]["label"], min_value=1, max_value=3, value=2, step=1, help=help_text("TZ_type"))
    iodine_negative = st.selectbox(VAR_DICT["iodine_negative"]["label"], [0, 1], format_func=lambda x: "0 = No" if x == 0 else "1 = Yes", help=help_text("iodine_negative"))
    atypical_vessels = st.selectbox(VAR_DICT["atypical_vessels"]["label"], [0, 1], format_func=lambda x: "0 = No" if x == 0 else "1 = Yes", help=help_text("atypical_vessels"))

    st.divider()
    st.subheader("Other")
    pathology_fig = st.number_input(VAR_DICT["pathology_fig"]["label"], min_value=0, max_value=10, value=2, step=1, help=help_text("pathology_fig"))

    st.divider()
    st.subheader("Decision mode")
    mode = st.selectbox("Mode", options=["triage", "youden", "screen"], index=0)
    st.caption(mode_explain(mode))

    run_btn = st.button("🚀 Predict", type="primary", use_container_width=True)


# =========================
# Single-case tab
# =========================
with tab_single:
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

    c1, c2 = st.columns([1.2, 0.8], gap="large")

    with c1:
        st.markdown('<div class="card"><div class="card-title">Input summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([record]), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">Notes</div>', unsafe_allow_html=True)
        st.markdown(
            "- `pathology_group` is the label source and is **not** an input feature.\n"
            "- This UI does **not** display any backend code or raw JSON outputs.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">Prediction</div>', unsafe_allow_html=True)

        if run_btn:
            with st.spinner("Running inference..."):
                try:
                    t0 = time.time()
                    out = predict_one(record, mode=mode)
                    dt_ms = (time.time() - t0) * 1000

                    # Robust parsing (different backends may name keys differently)
                    prob = safe_float(out.get("prob"), None)
                    if prob is None:
                        prob = safe_float(out.get("risk"), 0.0)

                    decision = out.get("decision", None)
                    thr = safe_float(out.get("threshold"), None)
                    mode_out = out.get("mode", mode)
                    reasons = out.get("reasons", None)

                    band = risk_band(prob)
                    st.markdown(
                        f"<div class='risk-banner {band_class(band)}'>"
                        f"<div class='risk-label'>Risk band: {band}</div>"
                        f"<div class='small-muted'>Calibrated probability: <b>{prob:.3f}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.write(band_guidance(band))
                    st.markdown("<div class='hr-soft'></div>", unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Probability", f"{prob:.3f}")
                    m2.metric("Mode", str(mode_out))
                    m3.metric("Threshold", f"{thr:.3f}" if thr is not None else "—")

                    if decision is not None:
                        st.markdown(f"**Decision:** {decision}")

                    if reasons:
                        st.markdown("**Explanation (summary):**")
                        if isinstance(reasons, list):
                            for r in reasons[:8]:
                                st.write(f"• {r if isinstance(r, str) else json.dumps(r)}")
                        else:
                            st.write(str(reasons))

                    st.caption(f"Response time: {dt_ms:.0f} ms")

                except Exception:
                    st.error("Inference failed. Please check that the model bundle is available and inputs are valid.")
        else:
            st.info("Fill the inputs on the left and click **Predict**.")

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Batch tab
# =========================
with tab_batch:
    st.markdown('<div class="card"><div class="card-title">Batch prediction</div>', unsafe_allow_html=True)
    st.write("Upload a CSV with the required columns. The app will run inference row-by-row.")

    st.download_button(
        "⬇️ Download CSV template",
        data=make_template_csv(),
        file_name="cervix_modelA_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            st.success(f"Loaded {len(df)} rows.")
            run_batch = st.button("▶️ Run batch prediction", type="primary")
            if run_batch:
                out_rows = []
                prog = st.progress(0)
                n = len(df)

                for i, row in df.iterrows():
                    rec = {}
                    for k in FEATURE_ORDER:
                        # best-effort numeric casting
                        val = row[k]
                        if pd.isna(val):
                            rec[k] = 0
                        else:
                            try:
                                rec[k] = int(val)
                            except Exception:
                                rec[k] = val

                    try:
                        pred = predict_one(rec, mode=mode)
                        p = safe_float(pred.get("prob"), None)
                        if p is None:
                            p = safe_float(pred.get("risk"), None)
                        out_rows.append({"row": i, "prob": p, "band": risk_band(float(p)) if p is not None else "ERROR"})
                    except Exception:
                        out_rows.append({"row": i, "prob": None, "band": "ERROR"})

                    prog.progress(int((i + 1) / max(n, 1) * 100))

                out_df = pd.DataFrame(out_rows)
                st.dataframe(out_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇️ Download results CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Dictionary tab
# =========================
with tab_dict:
    st.markdown('<div class="card"><div class="card-title">Variable Dictionary</div>', unsafe_allow_html=True)
    st.write("User-facing descriptions for each input variable.")

    rows = []
    for k in FEATURE_ORDER:
        v = VAR_DICT[k]
        rows.append({
            "Variable": k,
            "Label": v["label"],
            "Type": v["type"],
            "How to fill": v["how"],
            "Meaning": v["meaning"],
            "Valid values": v["values"],
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Note: pathology_group is the label source and is intentionally NOT an input feature.")
    st.markdown("</div>", unsafe_allow_html=True)
