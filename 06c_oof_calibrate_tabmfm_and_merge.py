# 06c_oof_calibrate_tabmfm_and_merge.py
# Fold-wise OOF calibration for TabTransformer-MFM outputs and merge into master table.
#
# Inputs:
#   - oof_calibrated_all.csv          (existing master with y, fold, LR/XGB/TAB columns)
#   - tab_mfm_oof_predictions.csv     (y, fold, p_tab_mfm_raw)
#
# Outputs:
#   - oof_calibrated_all_with_mfm.csv (adds p_tab_mfm_raw / p_tab_mfm_sigmoid / p_tab_mfm_isotonic)
#
# Prints:
#   - calibration check (OOF, fold-wise): Brier / AUC / AP for MFM raw/sigmoid/isotonic

import os
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA"

MASTER_IN = os.path.join(BASE_DIR, "oof_calibrated_all.csv")
MFM_IN    = os.path.join(BASE_DIR, "tab_mfm_oof_predictions.csv")

OUT_CSV   = os.path.join(BASE_DIR, "oof_calibrated_all_with_mfm.csv")

Y_COL = "y"
FOLD_COL = "fold"
P_RAW_COL = "p_tab_mfm_raw"

# =========================
# METRICS
# =========================
def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")

def safe_ap(y, p):
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")

def safe_brier(y, p):
    try:
        return float(brier_score_loss(y, p))
    except Exception:
        return float("nan")

def check_alignment(df_master, df_mfm):
    if len(df_master) != len(df_mfm):
        raise ValueError(f"Row mismatch: master={len(df_master)} vs mfm={len(df_mfm)}")

    y_ok = np.array_equal(df_master[Y_COL].values.astype(int), df_mfm[Y_COL].values.astype(int))
    f_ok = np.array_equal(df_master[FOLD_COL].values.astype(int), df_mfm[FOLD_COL].values.astype(int))

    if not y_ok or not f_ok:
        # show a quick diff head
        print("Alignment failed. Showing head comparisons:")
        print("master head:\n", df_master[[Y_COL, FOLD_COL]].head())
        print("mfm head:\n", df_mfm[[Y_COL, FOLD_COL]].head())
        raise ValueError(f"Alignment check failed: y_ok={y_ok}, fold_ok={f_ok}")

    print(f"rows all/mfm: {len(df_master)} {len(df_mfm)}")
    print("y aligned:", y_ok)
    print("fold aligned:", f_ok)

def foldwise_platt_sigmoid(y, p, fold_ids):
    """
    For each fold k:
      fit logistic regression on other folds' (p -> y)
      predict calibrated proba for fold k
    """
    y = y.astype(int)
    p = p.astype(float)
    fold_ids = fold_ids.astype(int)

    out = np.zeros_like(p, dtype=float)
    for k in np.unique(fold_ids):
        tr = fold_ids != k
        va = fold_ids == k

        # LR on 1-D score
        lr = LogisticRegression(solver="lbfgs", max_iter=5000)
        lr.fit(p[tr].reshape(-1, 1), y[tr])
        out[va] = lr.predict_proba(p[va].reshape(-1, 1))[:, 1]
        print(f"Fold {k} done. val N={va.sum()} pos={y[va].mean():.3f}")
    return out

def foldwise_isotonic(y, p, fold_ids):
    """
    For each fold k:
      fit isotonic on other folds' (p -> y)
      predict for fold k
    """
    y = y.astype(int)
    p = p.astype(float)
    fold_ids = fold_ids.astype(int)

    out = np.zeros_like(p, dtype=float)
    for k in np.unique(fold_ids):
        tr = fold_ids != k
        va = fold_ids == k

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p[tr], y[tr])
        out[va] = iso.transform(p[va])
    return out


def main():
    df_master = pd.read_csv(MASTER_IN)
    df_mfm = pd.read_csv(MFM_IN)

    # basic checks
    for c in [Y_COL, FOLD_COL]:
        if c not in df_master.columns:
            raise ValueError(f"Missing {c} in {MASTER_IN}")
        if c not in df_mfm.columns:
            raise ValueError(f"Missing {c} in {MFM_IN}")
    if P_RAW_COL not in df_mfm.columns:
        raise ValueError(f"Missing {P_RAW_COL} in {MFM_IN}")

    # sort by original order (assume already aligned); do strict alignment check
    check_alignment(df_master, df_mfm)

    y = df_master[Y_COL].values.astype(int)
    fold_ids = df_master[FOLD_COL].values.astype(int)

    p_raw = df_mfm[P_RAW_COL].values.astype(float)

    # fold-wise calibrations
    p_sig = foldwise_platt_sigmoid(y, p_raw, fold_ids)
    p_iso = foldwise_isotonic(y, p_raw, fold_ids)

    # metrics
    print("\n=== TAB-MFM calibration check (OOF, fold-wise) ===")
    print(f"TAB-MFM raw      | Brier={safe_brier(y,p_raw):.4f} AUC={safe_auc(y,p_raw):.3f} AP={safe_ap(y,p_raw):.3f}")
    print(f"TAB-MFM sigmoid   | Brier={safe_brier(y,p_sig):.4f} AUC={safe_auc(y,p_sig):.3f} AP={safe_ap(y,p_sig):.3f}")
    print(f"TAB-MFM isotonic  | Brier={safe_brier(y,p_iso):.4f} AUC={safe_auc(y,p_iso):.3f} AP={safe_ap(y,p_iso):.3f}")

    # merge
    df_out = df_master.copy()
    df_out["p_tab_mfm_raw"] = p_raw
    df_out["p_tab_mfm_sigmoid"] = p_sig
    df_out["p_tab_mfm_isotonic"] = p_iso

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\nSaved:", OUT_CSV)

    # show head quick
    print(df_out[[Y_COL, FOLD_COL, "p_tab_mfm_raw", "p_tab_mfm_sigmoid", "p_tab_mfm_isotonic"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
