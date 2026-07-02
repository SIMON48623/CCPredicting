# 07b_calibration_compare_plot_all.py
# Calibration plot for LR/XGB/TAB + TAB-MFM (raw & calibrated)
# Input: oof_calibrated_all_with_mfm.csv
# Output: calibration_compare_all.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

BASE_DIR = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA"
IN_CSV   = os.path.join(BASE_DIR, "oof_calibrated_all_with_mfm.csv")
OUT_PNG  = os.path.join(BASE_DIR, "calibration_compare_all.png")

Y_COL = "y"

# 你要保留“两条都在”，这里就把 MFM raw + MFM sigmoid 都画上
PLOT_COLS = [
    ("LR sigmoid",      "p_lr_sigmoid"),
    ("XGB isotonic",    "p_xgb_isotonic"),
    ("TAB sigmoid",     "p_tab_sigmoid"),
    ("TAB-MFM raw",     "p_tab_mfm_raw"),
    ("TAB-MFM sigmoid", "p_tab_mfm_sigmoid"),
]

# 分箱（越多越抖；10-15 通常够）
N_BINS = 12

def calibration_curve_bins(y, p, n_bins=12):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = bin_ids == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
        ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)

def main():
    df = pd.read_csv(IN_CSV)
    y = df[Y_COL].astype(int).values

    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], marker="o", label="Ideal")

    for name, col in PLOT_COLS:
        if col not in df.columns:
            continue
        p = df[col].astype(float).values
        brier = brier_score_loss(y, p)
        xs, ys, ns = calibration_curve_bins(y, p, n_bins=N_BINS)
        plt.plot(xs, ys, marker="o", label=f"{name} (Brier={brier:.3f})")

    plt.title("Calibration: LR/XGB/TAB + TAB-MFM (OOF)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction (CIN2+)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=200)
    print("Saved:", OUT_PNG)
    plt.show()

if __name__ == "__main__":
    main()
