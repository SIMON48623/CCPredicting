# 09_metrics_summary.py
# Summarize AUC/AP/Brier for multiple OOF probability columns.
# Input: oof_calibrated_all_with_mfm.csv
# Output: metrics_summary.csv

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE_DIR = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA"
IN_CSV   = os.path.join(BASE_DIR, "oof_calibrated_all_with_mfm.csv")
OUT_CSV  = os.path.join(BASE_DIR, "metrics_summary.csv")

Y_COL = "y"

# 你可以在这里控制要汇总哪些列
CANDIDATE_COLS = [
    # LR
    "p_lr_raw", "p_lr_sigmoid", "p_lr_isotonic",
    # XGB
    "p_xgb_raw", "p_xgb_sigmoid", "p_xgb_isotonic",
    # TAB (old)
    "p_tab_raw", "p_tab_sigmoid", "p_tab_isotonic",
    # TAB-MFM (new)
    "p_tab_mfm_raw", "p_tab_mfm_sigmoid", "p_tab_mfm_isotonic",
    # ENS (if exists)
    "p_ens",
]

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

def main():
    df = pd.read_csv(IN_CSV)
    if Y_COL not in df.columns:
        raise ValueError(f"Missing {Y_COL} in {IN_CSV}")

    y = df[Y_COL].astype(int).values

    rows = []
    for col in CANDIDATE_COLS:
        if col not in df.columns:
            continue
        p = df[col].astype(float).values
        rows.append({
            "model": col,
            "AUC": safe_auc(y, p),
            "AP": safe_ap(y, p),
            "Brier": safe_brier(y, p),
        })

    out = pd.DataFrame(rows).sort_values(["AP", "AUC"], ascending=[False, False]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("Saved:", OUT_CSV)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
