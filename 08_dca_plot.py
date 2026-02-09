# SAVE: YES (Decision Curve Analysis on OOF calibrated probabilities)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\oof_calibrated.csv"
df = pd.read_csv(path)

y = df["y"].values.astype(int)
p_lr = df["p_lr_sigmoid"].values.astype(float)
p_xgb = df["p_xgb_isotonic"].values.astype(float)

prev = y.mean()
N = len(y)

def net_benefit(y_true, p, pt):
    """
    Net Benefit = TP/N - FP/N * (pt/(1-pt))
    """
    y_pred = (p >= pt).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    nb = (tp / N) - (fp / N) * (pt / (1 - pt))
    return nb

def nb_treat_all(y_true, pt):
    # treat all => TP = positives, FP = negatives
    tp = np.sum(y_true == 1)
    fp = np.sum(y_true == 0)
    return (tp / N) - (fp / N) * (pt / (1 - pt))

def nb_treat_none(pt):
    return 0.0

# 常用阈值范围：0.05~0.60（更贴近临床“可行动”区间）
pts = np.arange(0.05, 0.61, 0.01)

nb_lr = [net_benefit(y, p_lr, pt) for pt in pts]
nb_xgb = [net_benefit(y, p_xgb, pt) for pt in pts]
nb_all = [nb_treat_all(y, pt) for pt in pts]
nb_none = [nb_treat_none(pt) for pt in pts]

# 打印一个小摘要：在几个典型阈值点谁更好
check_pts = [0.10, 0.20, 0.30, 0.40, 0.50]
print("Prevalence:", prev)
for pt in check_pts:
    print(f"pt={pt:.2f} | NB(LR)={net_benefit(y,p_lr,pt):.4f}  NB(XGB)={net_benefit(y,p_xgb,pt):.4f}  NB(All)={nb_treat_all(y,pt):.4f}")

plt.figure()
plt.plot(pts, nb_lr, marker="o")
plt.plot(pts, nb_xgb, marker="o")
plt.plot(pts, nb_all, marker="o")
plt.plot(pts, nb_none, marker="o")

plt.xlabel("Threshold probability (pt)")
plt.ylabel("Net benefit")
plt.title("Decision Curve Analysis (OOF calibrated)")
plt.legend(["LR sigmoid", "XGB isotonic", "Treat all", "Treat none"], loc="best")
plt.grid(True)

save_png = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\dca_oof.png"
plt.savefig(save_png, dpi=300, bbox_inches="tight")
print("Saved:", save_png)

plt.show()
