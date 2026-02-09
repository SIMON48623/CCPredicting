# 08b_dca_plot_all.py
# Decision Curve Analysis (DCA) for multiple models, using OOF probabilities
# Input: oof_calibrated_all_with_mfm.csv
# Output: dca_all.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA"
IN_CSV   = os.path.join(BASE_DIR, "oof_calibrated_all_with_mfm.csv")
OUT_PNG  = os.path.join(BASE_DIR, "dca_all.png")

Y_COL = "y"

# 这里我们两条都保留：MFM sigmoid（临床概率主推） + MFM raw（性能对照）
DCA_COLS = [
    ("LR sigmoid",      "p_lr_sigmoid"),
    ("XGB isotonic",    "p_xgb_isotonic"),
    ("TAB sigmoid",     "p_tab_sigmoid"),
    ("TAB-MFM sigmoid", "p_tab_mfm_sigmoid"),
    ("TAB-MFM raw",     "p_tab_mfm_raw"),
]

PT_MIN = 0.05
PT_MAX = 0.40
PT_STEP = 0.005

def net_benefit(y, p, pt):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    pred = (p >= pt).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    n = len(y)
    w = pt / (1.0 - pt)
    nb = (tp / n) - (fp / n) * w
    return nb

def net_benefit_treat_all(y, pt):
    y = np.asarray(y).astype(int)
    n = len(y)
    prev = y.mean()
    w = pt / (1.0 - pt)
    # treat-all: TP=prev*n, FP=(1-prev)*n
    nb = prev - (1.0 - prev) * w
    return nb

def main():
    df = pd.read_csv(IN_CSV)
    y = df[Y_COL].astype(int).values
    prev = y.mean()

    pts = np.arange(PT_MIN, PT_MAX + 1e-12, PT_STEP)

    plt.figure(figsize=(10, 7))

    # treat none
    nb_none = np.zeros_like(pts)
    plt.plot(pts, nb_none, marker="o", markersize=3, label="Treat none")

    # treat all
    nb_all = np.array([net_benefit_treat_all(y, pt) for pt in pts])
    plt.plot(pts, nb_all, marker="o", markersize=3, label="Treat all")

    # model curves
    curves = {}
    for name, col in DCA_COLS:
        if col not in df.columns:
            continue
        p = df[col].astype(float).values
        nb = np.array([net_benefit(y, p, pt) for pt in pts])
        curves[name] = nb
        plt.plot(pts, nb, marker="o", markersize=3, label=name)

    plt.title("Decision Curve Analysis (OOF): LR/XGB/TAB + TAB-MFM")
    plt.xlabel("Threshold probability (pt)")
    plt.ylabel("Net benefit")
    plt.grid(True, alpha=0.3)

    # annotate best gain of MFM sigmoid vs LR sigmoid (if both exist)
    if "TAB-MFM sigmoid" in curves and "LR sigmoid" in curves:
        diff = curves["TAB-MFM sigmoid"] - curves["LR sigmoid"]
        j = int(np.argmax(diff))
        pt_star = float(pts[j])
        dnb = float(diff[j])
        plt.annotate(f"Max(MFM-LR) at pt={pt_star:.3f}\nΔNB={dnb:.3f}",
                     xy=(pt_star, curves["TAB-MFM sigmoid"][j]),
                     xytext=(pt_star+0.03, curves["TAB-MFM sigmoid"][j]+0.02),
                     arrowprops=dict(arrowstyle="->"))

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Saved: {OUT_PNG}")
    print(f"prevalence={prev:.4f}")
    plt.show()

if __name__ == "__main__":
    main()
