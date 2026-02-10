import os
import sys
import time
import json
import io
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# -------------------------
# Paths & imports
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from final_model.predict_api import predict_one

# -------------------------
# Page
# -------------------------
st.set_page_config(
    page_title="Clinical Risk Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Hospital-dashboard CSS
# -------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2.0rem; max-width: 1320px; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.06); }
div[data-testid="stSidebarContent"] { padding-top: 0.6rem; }

.header {
  border-radius: 16px;
  border: 1px solid rgba(0,0,0,.08);
  background: linear-gradient(180deg, rgba(248,250,252,1), rgba(241,245,249,1));
  padding: 0.95rem 1.15rem;
}
.header-title { font-size: 1.42rem; font-weight: 820; margin: 0; }
.header-sub { margin: .25rem 0 0; color: rgba(0,0,0,.62); }

.card {
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 16px;
  padding: 1rem 1rem;
  background: #ffffff;
}
.card-title { font-weight: 780; margin-bottom: .35rem; }

.hr-soft { height: 1px; background: rgba(0,0,0,.06); margin: .85rem 0; }

.risk-banner {
  border-radius: 14px;
  padding: .85rem .95rem;
  border: 1px solid rgba(0,0,0,.08);
}
.risk-low  { background: rgba(16,185,129,.10); }
.risk-mid  { background: rgba(245,158,11,.12); }
.risk-high { background: rgba(239,68,68,.12); }
.risk-label { font-weight: 840; font-size: 1.05rem; }

.stButton>button {
  border-radius: 12px !important;
  padding: .60rem 1.00rem !important;
  font-weight: 700 !important;
}

/* ---- table-like input summary ---- */
.summary-grid {
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 14px;
  overflow: hidden;
  background: #fff;
}
.summary-row {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
}
.summary-row.row2 {
  grid-template-columns: repeat(7, 1fr);
}
.summary-cell {
  padding: 10px 10px 8px 10px;
  border-right: 1px solid rgba(0,0,0,.06);
  border-top: 1px solid rgba(0,0,0,.06);
  min-height: 56px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.summary-row:first-child .summary-cell { border-top: none; }
.summary-cell:last-child { border-right: none; }

.summary-k {
  font-size: 0.82rem;
  color: rgba(0,0,0,.55);
  line-height: 1.1;
  margin-bottom: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.summary-v {
  font-size: 1.05rem;
  font-weight: 760;
  color: rgba(0,0,0,.90);
  line-height: 1.15;
}

@media (max-width: 1100px) {
  .summary-row { grid-template-columns: repeat(4, 1fr); }
  .summary-row.row2 { grid-template-columns: repeat(4, 1fr); }
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def risk_band(prob: float) -> str:
    if prob < 0.10:
        return "Low"
    if prob < 0.30:
        return "Intermediate"
    return "High"

def band_class(band: str) -> str:
    return {"Low": "risk-low", "Intermediate": "risk-mid", "High": "risk-high"}.get(band, "risk-mid")

def clinical_message(band: str) -> str:
    if band == "High":
        return "Higher risk: consider guideline-based specialist evaluation (research prototype)."
    if band == "Intermediate":
        return "Intermediate risk: consider closer follow-up and triage (research prototype)."
    return "Lower risk: continue routine screening/follow-up (research prototype)."

def render_input_summary_table(rec: dict):
    row1 = ["age", "menopausal_status", "gravidity", "parity", "child_alive", "HPV_overall", "HPV16", "HPV18"]
    row2 = ["HPV_other_hr", "cytology_grade", "colpo_impression", "TZ_type", "iodine_negative", "atypical_vessels", "pathology_fig"]

    def cell_html(k):
        v = rec.get(k, "")
        return f"""
        <div class="summary-cell">
          <div class="summary-k">{k}</div>
          <div class="summary-v">{v}</div>
        </div>
        """

    html = f"""
    <div class="summary-grid">
      <div class="summary-row">
        {''.join(cell_html(k) for k in row1)}
      </div>
      <div class="summary-row row2">
        {''.join(cell_html(k) for k in row2)}
      </div>
    </div>
    """
    # ✅ This prevents HTML being shown as raw text
    components.html(html, height=150, scrolling=False)

def make_template_csv() -> bytes:
    demo = {
        "age": 45, "menopausal_status": 0, "gravidity": 2, "parity": 1, "child_alive": 0,
        "HPV_overall": 0, "HPV16": 0, "HPV18": 0, "HPV_other_hr": 0,
        "cytology_grade": 3, "colpo_impression": 2, "TZ_type": 2,
        "iodine_negative": 0, "atypical_vessels": 0, "pathology_fig": 2
    }
    df = pd.DataFrame([demo])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def render_ig_payload(ig):
    if ig is None:
        st.info("IG explanation is not available.")
        return
    if isinstance(ig, dict) and ("positive" in ig or "negative" in ig):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Factors increasing risk**")
            for r in (ig.get("positive") or [])[:12]:
                st.write(f"- {r}")
        with c2:
            st.markdown("**Factors decreasing risk**")
            for r in (ig.get("negative") or [])[:12]:
                st.write(f"- {r}")
        return
    if isinstance(ig, list):
        for r in ig[:12]:
            st.write(f"- {r}")
        return
    st.write(str(ig))

# -------------------------
# Header
# -------------------------
st.markdown(
    """
<div class="header">
  <p class="header-title">Cervical Lesion Risk Prediction</p>
  <p class="header-sub">Clinical-style triage dashboard for calibrated risk estimation (research prototype).</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_single, tab_batch = st.tabs(["Single-case", "Batch"])

# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.subheader("Patient inputs")

    age = st.number_input("Age (years)", min_value=10, max_value=100, value=45, step=1)
    menopausal_status = st.selectbox("Menopausal status", options=[0, 1],
                                     format_func=lambda x: "0 = Premenopausal" if x == 0 else "1 = Postmenopausal")
    gravidity = st.number_input("Gravidity", min_value=0, max_value=30, value=2, step=1)
    parity = st.number_input("Parity", min_value=0, max_value=20, value=1, step=1)
    child_alive = st.selectbox("Any living child", options=[0, 1],
                               format_func=lambda x: "0 = No" if x == 0 else "1 = Yes")

    st.divider()
    st.subheader("HPV status")
    HPV_overall = st.selectbox("High-risk HPV (overall)", [0, 1],
                               format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive")
    HPV16 = st.selectbox("HPV16 positive", [0, 1],
                         format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive")
    HPV18 = st.selectbox("HPV18 positive", [0, 1],
                         format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive")
    HPV_other_hr = st.selectbox("Other high-risk HPV", [0, 1],
                                format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive")

    st.divider()
    st.subheader("Cytology and colposcopy")
    cytology_grade = st.number_input("Cytology grade (0–5)", min_value=0, max_value=5, value=3, step=1)
    colpo_impression = st.number_input("Colposcopy impression (0–4)", min_value=0, max_value=4, value=2, step=1)
    TZ_type = st.number_input("TZ type (1–3)", min_value=1, max_value=3, value=2, step=1)
    iodine_negative = st.selectbox("Iodine test negative", [0, 1],
                                   format_func=lambda x: "0 = No" if x == 0 else "1 = Yes")
    atypical_vessels = st.selectbox("Atypical vessels", [0, 1],
                                    format_func=lambda x: "0 = No" if x == 0 else "1 = Yes")

    st.divider()
    st.subheader("Other")
    pathology_fig = st.number_input("Pathology fig (routine variable)", min_value=0, max_value=10, value=2, step=1)

    st.divider()
    st.subheader("Decision mode")
    mode = st.selectbox("Mode", options=["triage", "youden", "screen"], index=0)

    run_btn = st.button("Run prediction", type="primary", use_container_width=True)

# -------------------------
# Single-case
# -------------------------
with tab_single:
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

    if "last_pred" not in st.session_state:
        st.session_state["last_pred"] = None
        st.session_state["last_record"] = None

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">Input summary</div>', unsafe_allow_html=True)
        render_input_summary_table(record)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><div class="card-title">Risk assessment</div>', unsafe_allow_html=True)

        if run_btn:
            with st.spinner("Computing risk..."):
                try:
                    t0 = time.time()
                    out = predict_one(record, mode=mode)
                    dt_ms = (time.time() - t0) * 1000

                    st.session_state["last_pred"] = out
                    st.session_state["last_record"] = dict(record)

                    prob = safe_float(out.get("prob"), 0.0)
                    band = risk_band(prob)

                    st.markdown(
                        f"<div class='risk-banner {band_class(band)}'>"
                        f"<div class='risk-label'>Risk band: {band}</div>"
                        f"<div style='color:rgba(0,0,0,.62)'>Calibrated probability: <b>{prob:.3f}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.write(clinical_message(band))
                    st.markdown("<div class='hr-soft'></div>", unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Probability", f"{prob:.3f}")
                    m2.metric("Mode", str(out.get("decision_mode", mode)))
                    thr = out.get("threshold", None)
                    m3.metric("Threshold", f"{float(thr):.3f}" if thr is not None else "—")

                    lab = out.get("label", None)
                    if lab is not None:
                        st.markdown(f"**Decision:** {str(lab).upper()}")

                    st.caption(f"Latency: {dt_ms:.0f} ms")

                except Exception:
                    st.error("Unable to compute risk. Please verify model availability and inputs.")
        else:
            if st.session_state["last_pred"] is None:
                st.info("Enter inputs in the sidebar and click Run prediction.")
            else:
                out = st.session_state["last_pred"]
                prob = safe_float(out.get("prob"), 0.0)
                band = risk_band(prob)
                st.markdown(
                    f"<div class='risk-banner {band_class(band)}'>"
                    f"<div class='risk-label'>Risk band: {band}</div>"
                    f"<div style='color:rgba(0,0,0,.62)'>Calibrated probability: <b>{prob:.3f}</b></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.write(clinical_message(band))

        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Explainability: IG only, follows Run prediction ----
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Explainability</div>', unsafe_allow_html=True)
    st.subheader("Integrated gradients (single-case)")

    if st.session_state["last_pred"] is None:
        st.info("Run prediction to view IG explanation for the current input case.")
    else:
        ig = st.session_state["last_pred"].get("ig", None)
        render_ig_payload(ig)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Batch
# -------------------------
with tab_batch:
    st.markdown('<div class="card"><div class="card-title">Batch prediction</div>', unsafe_allow_html=True)
    st.write("Upload a CSV containing the required columns. Predictions are computed row-by-row.")
    st.download_button(
        "Download CSV template",
        data=make_template_csv(),
        file_name="cervix_modelA_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        required = [
            "age","menopausal_status","gravidity","parity","child_alive",
            "HPV_overall","HPV16","HPV18","HPV_other_hr",
            "cytology_grade","colpo_impression","TZ_type",
            "iodine_negative","atypical_vessels","pathology_fig"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            st.success(f"Loaded {len(df)} rows.")
            run_batch = st.button("Run batch prediction", type="primary")
            if run_batch:
                out_rows = []
                prog = st.progress(0)
                n = len(df)
                for i, row in df.iterrows():
                    rec = {k: (0 if pd.isna(row[k]) else int(row[k])) for k in required}
                    try:
                        pred = predict_one(rec, mode=mode)
                        p = safe_float(pred.get("prob"), None)
                        out_rows.append({"row": i, "probability": p})
                    except Exception:
                        out_rows.append({"row": i, "probability": None})
                    prog.progress(int((i + 1) / max(n, 1) * 100))

                out_df = pd.DataFrame(out_rows)
                st.dataframe(out_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)
