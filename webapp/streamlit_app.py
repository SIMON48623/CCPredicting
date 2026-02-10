# =========================
# Performance asset discovery + plotting (FULL REPLACEMENT BLOCK)
# Replace everything from "# Performance asset discovery" to EOF with this block.
# =========================
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def candidate_figure_paths(filename: str):
    """
    IMPORTANT:
    Many exported figures in your project are placed in PROJ_ROOT directly (e.g., calibration_oof.png).
    So we must include base/filename first.
    """
    base = Path(PROJ_ROOT)
    candidates = [
        str(base / filename),  # ✅ project root
        str(base / "figures" / filename),
        str(base / "paper_exports" / filename),
        str(base / "paper_exports" / "figures" / filename),
        str(base / "paper_exports" / "figs" / filename),
        str(base / "plots" / filename),
        str(base / "outputs" / "figures" / filename),
        str(base / "results" / "figures" / filename),
        str(base / "webapp" / "assets" / filename),
    ]
    return candidates


def try_show_figure(title: str, filenames):
    st.subheader(title)
    # Allow passing "path/filename.png" too
    expanded = []
    for fn in filenames:
        if "/" in fn or "\\" in fn:
            expanded.append(str(Path(PROJ_ROOT) / fn))
        expanded.extend(candidate_figure_paths(fn))

    p = find_first_existing(expanded)
    if p:
        st.image(p, use_container_width=True)
    else:
        st.info("Figure not available.")


def try_show_table(title: str, rel_paths):
    st.subheader(title)
    expanded = []
    for rp in rel_paths:
        expanded.append(str(Path(PROJ_ROOT) / rp))
        # also allow direct filename in root
        expanded.append(str(Path(PROJ_ROOT) / Path(rp).name))

    p = find_first_existing(expanded)
    if p:
        try:
            df = pd.read_csv(p)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            st.info("Table found, but cannot be displayed as CSV.")
    else:
        st.info("Table not available.")


def plot_roc_pr_from_oof():
    """
    Computes ROC/PR from OOF predictions rather than relying on exported PNGs.
    Uses your project's standard OOF file and main probability column.
    """
    oof_path = Path(PROJ_ROOT) / "oof_calibrated_all_with_mfm.csv"
    if not oof_path.exists():
        st.info("OOF file not found: oof_calibrated_all_with_mfm.csv")
        return

    df = pd.read_csv(oof_path)

    # Accept a few possible label/prob column names
    y_candidates = ["y", "label", "Y", "target"]
    p_candidates = [
        "p_tab_mfm_sigmoid",  # your paper main
        "p_tab_sigmoid",
        "p_tab_mfm_raw",
        "p_tab",
        "prob",
        "risk",
    ]

    y_col = next((c for c in y_candidates if c in df.columns), None)
    p_col = next((c for c in p_candidates if c in df.columns), None)

    if y_col is None or p_col is None:
        st.info("Required columns not found in OOF CSV (need label + probability).")
        return

    y = df[y_col].astype(int).values
    p = df[p_col].astype(float).values

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)

    fig1 = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (OOF) • {p_col}")
    plt.legend(loc="lower right")
    st.pyplot(fig1, clear_figure=True)

    # PR
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    fig2 = plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall curve (OOF) • {p_col}")
    plt.legend(loc="lower left")
    st.pyplot(fig2, clear_figure=True)


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
# =========================
if st.session_state["show_performance"]:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Performance</div>', unsafe_allow_html=True)

    perf_tab1, perf_tab2, perf_tab3 = st.tabs(["Discrimination", "Calibration & Utility", "Explainability"])

    with perf_tab1:
        plot_roc_pr_from_oof()

    with perf_tab2:
        try_show_figure(
            "Calibration",
            [
                "calibration_oof.png",
                "calibration_compare.png",
                "calibration_compare_all.png",
                "reliability.png",
                "reliability_diagram.png",
            ],
        )
        try_show_figure(
            "Decision curve analysis",
            [
                "dca_oof.png",
                "dca_all.png",
                "dca_oof_all.png",
                "decision_curve.png",
            ],
        )

        try_show_table(
            "Operating points",
            [
                "paper_exports/thresholds_summary.csv",
                "paper_exports/threshold_report_full.csv",
                "paper_exports/threshold_report_mfm_sigmoid.csv",
                "threshold_report_mfm_sigmoid.csv",
                "threshold_report.csv",
            ],
        )
        try_show_table(
            "Overall metrics",
            [
                "paper_exports/metrics_overall.csv",
                "paper_exports/metrics_summary.csv",
                "metrics_summary.csv",
                "stacking_metrics.csv",
            ],
        )

    with perf_tab3:
        try_show_figure(
            "Integrated gradients",
            [
                "ig_global_top15.png",
                "figures/ig_global_top15.png",
                "ig_one_demo_top10.png",
                "figures/ig_one_demo_top10.png",
            ],
        )
        try_show_figure(
            "SHAP (XGBoost)",
            [
                "shap_xgb_summary.png",
                "shap_xgb_bar.png",
            ],
        )

    st.markdown("</div>", unsafe_allow_html=True)
