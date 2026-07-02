# 06b_oof_calibrate_tab_and_merge.py
# Fold-wise calibration for OOF probabilities WITHOUT CalibratedClassifierCV
# - sigmoid: LogisticRegression on p_raw (Platt scaling)
# - isotonic: IsotonicRegression on p_raw
# Compatible across sklearn versions, avoids "regressor/classifier" detection issues.

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def safe_float_array(x):
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)


def clip_probs(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps)


def ensure_raw_col(df, base_name):
    """
    Accept either:
      - p_xxx_raw (preferred)
      - p_xxx (legacy)
    If only legacy exists, create *_raw copy.
    Return raw column name or None.
    """
    raw = f"{base_name}_raw"
    legacy = base_name
    if raw in df.columns:
        return raw
    if legacy in df.columns:
        df[raw] = df[legacy].copy()
        return raw
    return None


def foldwise_calibrate_probs_platt(y, fold, p_raw):
    """sigmoid calibration (Platt scaling) using LogisticRegression on p_raw as feature."""
    y = np.asarray(y, dtype=int)
    fold = np.asarray(fold, dtype=int)
    p_raw = clip_probs(safe_float_array(p_raw))

    n = len(y)
    idx_all = np.arange(n, dtype=int)
    p_cal = np.full(n, np.nan, dtype=float)

    X_all = p_raw.reshape(-1, 1)

    for f in np.unique(fold):
        val_idx = idx_all[fold == f]
        tr_idx = idx_all[fold != f]

        # Platt scaling
        lr = LogisticRegression(solver="lbfgs", max_iter=2000)
        lr.fit(X_all[tr_idx], y[tr_idx])
        p_cal[val_idx] = lr.predict_proba(X_all[val_idx])[:, 1]

    return clip_probs(p_cal)


def foldwise_calibrate_probs_isotonic(y, fold, p_raw):
    """isotonic calibration using IsotonicRegression on p_raw."""
    y = np.asarray(y, dtype=int)
    fold = np.asarray(fold, dtype=int)
    p_raw = clip_probs(safe_float_array(p_raw))

    n = len(y)
    idx_all = np.arange(n, dtype=int)
    p_cal = np.full(n, np.nan, dtype=float)

    for f in np.unique(fold):
        val_idx = idx_all[fold == f]
        tr_idx = idx_all[fold != f]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw[tr_idx], y[tr_idx])
        p_cal[val_idx] = iso.predict(p_raw[val_idx])

    return clip_probs(p_cal)


def metrics_line(name, y, p):
    p = clip_probs(safe_float_array(p))
    return dict(
        model=name,
        AUC=float(roc_auc_score(y, p)),
        AP=float(average_precision_score(y, p)),
        Brier=float(brier_score_loss(y, p)),
    )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # prefer with_mfm if exists
    cand1 = os.path.join(base_dir, "oof_predictions_all_with_mfm.csv")
    cand2 = os.path.join(base_dir, "oof_predictions_all.csv")

    if os.path.exists(cand1):
        in_path = cand1
    elif os.path.exists(cand2):
        in_path = cand2
    else:
        raise FileNotFoundError(
            "Cannot find input OOF file. Expected one of:\n"
            f" - {cand1}\n - {cand2}"
        )

    df = pd.read_csv(in_path)

    for c in ["y", "fold"]:
        if c not in df.columns:
            raise ValueError(f"Input file missing column: {c}")

    y = pd.to_numeric(df["y"], errors="coerce").astype(int).to_numpy()
    fold = pd.to_numeric(df["fold"], errors="coerce").astype(int).to_numpy()

    print(f"Loaded: {os.path.basename(in_path)}")
    print(f"N={len(df)}, prevalence={y.mean():.4f}")
    print("Columns:", list(df.columns))

    # accept legacy columns
    lr_raw = ensure_raw_col(df, "p_lr")
    xgb_raw = ensure_raw_col(df, "p_xgb")
    tab_raw = ensure_raw_col(df, "p_tab")
    mfm_raw = ensure_raw_col(df, "p_tab_mfm")  # optional

    raw_list = []
    if lr_raw: raw_list.append(("LR", lr_raw))
    if xgb_raw: raw_list.append(("XGB", xgb_raw))
    if tab_raw: raw_list.append(("TAB", tab_raw))
    if mfm_raw: raw_list.append(("TAB-MFM", mfm_raw))

    if len(raw_list) == 0:
        raise ValueError(
            "No probability columns found. Expected at least one of:\n"
            "p_lr_raw/p_lr, p_xgb_raw/p_xgb, p_tab_raw/p_tab, p_tab_mfm_raw/p_tab_mfm"
        )

    out = df.copy()
    all_metrics = []

    for model_name, raw_col in raw_list:
        p_raw = out[raw_col].values
        all_metrics.append(metrics_line(f"{model_name} raw ({raw_col})", y, p_raw))

        sig_col = raw_col.replace("_raw", "_sigmoid")
        iso_col = raw_col.replace("_raw", "_isotonic")

        p_sig = foldwise_calibrate_probs_platt(y, fold, p_raw)
        p_iso = foldwise_calibrate_probs_isotonic(y, fold, p_raw)

        out[sig_col] = p_sig
        out[iso_col] = p_iso

        m_raw = all_metrics[-1]
        m_sig = metrics_line(f"{model_name} sigmoid ({sig_col})", y, p_sig)
        m_iso = metrics_line(f"{model_name} isotonic ({iso_col})", y, p_iso)
        all_metrics.extend([m_sig, m_iso])

        print(f"\n=== {model_name} calibration check (OOF, fold-wise) ===")
        print(f"{model_name:7s} raw     | Brier={m_raw['Brier']:.4f} AUC={m_raw['AUC']:.3f} AP={m_raw['AP']:.3f}")
        print(f"{model_name:7s} sigmoid | Brier={m_sig['Brier']:.4f} AUC={m_sig['AUC']:.3f} AP={m_sig['AP']:.3f}")
        print(f"{model_name:7s} isotonic| Brier={m_iso['Brier']:.4f} AUC={m_iso['AUC']:.3f} AP={m_iso['AP']:.3f}")

    out_path = os.path.join(base_dir, "oof_calibrated_all.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    metrics_df = pd.DataFrame(all_metrics).sort_values(["AP", "AUC"], ascending=False)
    metrics_path = os.path.join(base_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    print("\nTop metrics (sorted by AP then AUC):")
    with pd.option_context("display.max_rows", 80, "display.width", 140):
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
