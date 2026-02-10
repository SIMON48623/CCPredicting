import os
import sys
import time
import json
import io
from pathlib import Path

import streamlit as st
import pandas as pd

# =========================
# Backend import (never displayed)
# =========================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from final_model.predict_api import predict_one  # backend call only


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Clinical Risk Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Hospital-dashboard CSS
# =========================
st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2.0rem; max-width: 1320px; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.06); }
div[data-testid="stSidebarContent"] { padding-top: 0.6rem; }

h1, h2, h3 { letter-spacing: -0.02em; }
.small-muted { color: rgba(0,0,0,0.60); font-size: 0.92rem; }

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

.kv-label { font-size: .82rem; color: rgba(0,0,0,.55); line-height: 1.1; }
.kv-value { font-size: 1.02rem; font-weight: 720; margin-top: .1rem; }

.stButton>button {
  border-radius: 12px !important;
  padding: .60rem 1.00rem !important;
  font-weight: 700 !important;
}
button[data-baseweb="tab"] { font-weight: 680; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Variable dictionary (English tooltips)
# =========================
VAR_DICT = {
    "age": {
        "label": "Age (years)",
        "type": "Integer",
        "how": "Age at the time of examination.",
        "meaning": "Older age is generally associated with higher risk.",
        "values": "10–100 (UI constraint)",
    },
    "menopausal_status": {
        "label": "Menopausal status",
        "type": "Binary (0/1)",
        "how": "0 = Premenopausal; 1 = Postmenopausal.",
        "meaning": "May influence TZ visibility and clinical interpretation.",
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
        "how": "1 = has at least one living child; 0 = none.",
        "meaning": "Reproductive outcome descriptor.",
        "values": "0 or 1",
    },
    "HPV_overall": {
        "label": "High-risk HPV (overall)",
        "type": "Binary (0/1)",
        "how": "Overall high-risk HPV status.",
        "meaning": "Core clinical predictor.",
        "values": "0 or 1",
    },
    "HPV16": {
        "label": "HPV16 positive",
        "type": "Binary (0/1)",
        "how": "HPV16 infection status.",
        "meaning": "Associated with higher CIN3+ risk.",
        "values": "0 or 1",
    },
    "HPV18": {
        "label": "HPV18 positive",
        "type": "Binary (0/1)",
        "how": "HPV18 infection status.",
        "meaning": "Associated with glandular lesion risk.",
        "values": "0 or 1",
    },
    "HPV_other_hr": {
        "label": "Other high-risk HPV",
        "type": "Binary (0/1)",
        "how": "Other high-risk HPV types.",
        "meaning": "Captures high-risk HPV beyond 16/18.",
        "values": "0 or 1",
    },
    "cytology_grade": {
        "label": "Cytology grade",
        "type": "Ordinal (0–5)",
        "how": "0=NILM, 1=ASC-US, 2=ASC-H, 3=LSIL, 4=HSIL, 5=AGC.",
        "meaning": "Cytology severity grade.",
        "values": "0–5",
    },
    "colpo_impression": {
        "label": "Colposcopy impression",
        "type": "Ordinal (0–4)",
        "how": "0=Normal; 1=Mild; 2=Moderate; 3=Severe; 4=Highly suspicious.",
        "meaning": "Clinician impression during colposcopy.",
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
        "meaning": "May suggest abnormal epithelium.",
        "values": "0 or 1",
    },
    "atypical_vessels": {
        "label": "Atypical vessels",
        "type": "Binary (0/1)",
        "how": "Presence of atypical vessels.",
        "meaning": "Can be associated with higher-grade disease.",
        "values": "0 or 1",
    },
    "pathology_fig": {
        "label": "Pathology fig (routine variable)",
        "type": "Integer (project-defined)",
        "how": "Enter as recorded in the dataset.",
        "meaning": "Treated as a routine input variable.",
        "values": "0–10 (UI constraint)",
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
        f"Clinical meaning: {v['meaning']}\n\n"
        f"Valid values: {v['values']}"
    )


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


def render_two_row_summary(rec: dict):
    # Two-row layout: no horizontal scrolling
    row1 = ["age", "menopausal_status", "gravidity", "parity", "child_alive", "HPV_overall", "HPV16", "HPV18"]
    row2 = ["HPV_other_hr", "cytology_grade", "colpo_impression", "TZ_type", "iodine_negative", "atypical_vessels", "pathology_fig"]

    def kv_row(keys):
        cols = st.columns(len(keys))
        for col, k in zip(cols, keys):
            with col:
                st.markdown(f"<div class='kv-label'>{k}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kv-value'>{rec.get(k, '')}</div>", unsafe_allow_html=True)

    kv_row(row1)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    kv_row(row2)


# =========================
# Performance asset discovery
# =========================
def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def candidate_figure_paths(filename: str):
    # common folders in your project (safe: only reads files if exist)
    base = Path(PROJ_ROOT)
    candidates = [
        base / "paper_exports" / "figures" / filename,
        base / "paper_exports" / "figs" / filename,
        base / "figures" / filename,
        base / "plots" / filename,
        base / "outputs" / "figures" / filename,
        base / "results" / "figures" / filename,
        base / "webapp" / "assets" / filename,
    ]
    return [str(c) for c in candidates]


def try_show_figure(title: str, filenames):
    st.subheader(title)
    for fn in filenames:
        p = find_first_existing(candidate_figure_paths(fn))
        if p:
            st.image(p, use_container_width=True)
            return
    st.info("Figure not found in the project export folders.")


def try_show_table(title: str, rel_paths):
    st.subheader(title)
    p = find_first_existing([str(Path(PROJ_ROOT) / rp) for rp in rel_paths])
    if p:
        try:
            df = pd.read_csv(p)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            st.info("Table found, but cannot be displayed as CSV.")
        return
    st.info("Table not found in the project export folders.")


# =========================
# Header (with performance toggle)
# =========================
if "show_performance" not in st.session_state:
    st.session_state["show_performance"] = False

hdr_left, hdr_right = st.columns([0.78, 0.22], gap="large")
with hdr_left:
    st.markdown(
        """
<div class="header">
  <p class="header-title">Cervical Lesion Risk Prediction</p>
  <p class="header-sub">Clinical-style triage dashboard for calibrated risk estimation (research prototype).</p>
</div>
""",
        unsafe_allow_html=True,
    )

with hdr_right:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("View performance", use_container_width=True):
        st.session_state["show_performance"] = not st.session_state["show_performance"]


# =========================
# Main tabs (clinical triage)
# =========================
tab_single, tab_batch, tab_dictionary = st.tabs(["Single-case", "Batch", "Data Dictionary"])


# =========================
# Sidebar: Inputs
# =========================
with st.sidebar:
    st.subheader("Patient inputs")

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
    st.subheader("HPV status")
    HPV_overall = st.selectbox(VAR_DICT["HPV_overall"]["label"], [0, 1], format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive", help=help_text("HPV_overall"))
    HPV16 = st.selectbox(VAR_DICT["HPV16"]["label"], [0, 1], format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive", help=help_text("HPV16"))
    HPV18 = st.selectbox(VAR_DICT["HPV18"]["label"], [0, 1], format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive", help=help_text("HPV18"))
    HPV_other_hr = st.selectbox(VAR_DICT["HPV_other_hr"]["label"], [0, 1], format_func=lambda x: "0 = Negative" if x == 0 else "1 = Positive", help=help_text("HPV_other_hr"))

    st.divider()
    st.subheader("Cytology and colposcopy")
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

    run_btn = st.button("Run prediction", type="primary", use_container_width=True)


# =========================
# Single-case tab (Version 1)
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

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">Input summary</div>', unsafe_allow_html=True)
        render_two_row_summary(record)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><div class="card-title">Risk assessment</div>', unsafe_allow_html=True)

        if run_btn:
            with st.spinner("Computing risk..."):
                try:
                    t0 = time.time()
                    out = predict_one(record, mode=mode)
                    dt_ms = (time.time() - t0) * 1000

                    prob = safe_float(out.get("prob"), None)
                    if prob is None:
                        prob = safe_float(out.get("risk"), 0.0)

                    decision = out.get("decision", None)
                    thr = safe_float(out.get("threshold"), None)
                    mode_out = out.get("mode", mode)

                    band = risk_band(prob)

                    st.markdown(
                        f"<div class='risk-banner {band_class(band)}'>"
                        f"<div class='risk-label'>Risk band: {band}</div>"
                        f"<div class='small-muted'>Calibrated probability: <b>{prob:.3f}</b></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.write(clinical_message(band))
                    st.markdown("<div class='hr-soft'></div>", unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Probability", f"{prob:.3f}")
                    m2.metric("Mode", str(mode_out))
                    m3.metric("Threshold", f"{thr:.3f}" if thr is not None else "—")

                    if decision is not None:
                        st.markdown(f"**Decision:** {decision}")

                    reasons = out.get("reasons", None)
                    if reasons:
                        st.markdown("**Key contributing factors:**")
                        if isinstance(reasons, list):
                            for r in reasons[:6]:
                                st.write(f"- {r if isinstance(r, str) else json.dumps(r)}")
                        else:
                            st.write(str(reasons))

                    st.caption(f"Latency: {dt_ms:.0f} ms")

                except Exception:
                    st.error("Unable to compute risk. Please verify model availability and inputs.")
        else:
            st.info("Enter inputs in the sidebar and click Run prediction.")

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Batch tab
# =========================
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
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
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
                    rec = {}
                    for k in FEATURE_ORDER:
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
                        out_rows.append({
                            "row": i,
                            "probability": p,
                            "risk_band": risk_band(float(p)) if p is not None else "ERROR"
                        })
                    except Exception:
                        out_rows.append({"row": i, "probability": None, "risk_band": "ERROR"})

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


# =========================
# Dictionary tab
# =========================
with tab_dictionary:
    st.markdown('<div class="card"><div class="card-title">Data dictionary</div>', unsafe_allow_html=True)
    rows = []
    for k in FEATURE_ORDER:
        v = VAR_DICT[k]
        rows.append({
            "Variable": k,
            "Label": v["label"],
            "Type": v["type"],
            "How to fill": v["how"],
            "Clinical meaning": v["meaning"],
            "Valid values": v["values"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Hidden Performance section (Version 2 evidence)
# Appears only when user clicks "View performance"
# =========================
if st.session_state["show_performance"]:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Performance</div>', unsafe_allow_html=True)

    perf_tab1, perf_tab2, perf_tab3 = st.tabs(["Discrimination", "Calibration & Utility", "Explainability"])

    with perf_tab1:
        # ROC / PR
        try_show_figure("ROC curve", ["roc.png", "roc_curve.png", "ROC.png"])
        try_show_figure("Precision–Recall curve", ["pr.png", "pr_curve.png", "PR.png"])

    with perf_tab2:
        # Calibration / DCA / thresholds
        try_show_figure("Calibration curve", ["calibration.png", "calibration_curve.png", "reliability.png"])
        try_show_figure("Decision curve analysis", ["dca.png", "dca_curve.png", "decision_curve.png"])
        try_show_table(
            "Operating points (threshold summary)",
            [
                "paper_exports/thresholds_summary.csv",
                "paper_exports/thresholds.csv",
                "tables/thresholds_summary.csv",
                "tables/thresholds.csv",
            ],
        )
        try_show_table(
            "Overall metrics summary",
            [
                "paper_exports/metrics_overall.csv",
                "paper_exports/metrics_summary.csv",
                "tables/metrics_overall.csv",
                "tables/metrics_summary.csv",
            ],
        )

    with perf_tab3:
        # IG global / SHAP summary if exported as images
        try_show_figure("Integrated gradients (global)", ["ig_global.png", "ig_global_top.png", "ig_bar.png"])
        try_show_figure("SHAP summary (XGBoost)", ["shap_summary.png", "xgb_shap_summary.png"])

    st.markdown("</div>", unsafe_allow_html=True)
