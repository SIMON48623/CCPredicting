# SAVE: YES (calibration plot: before vs after)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

path = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\oof_calibrated.csv"
df = pd.read_csv(path)

y = df["y"].values.astype(int)

p_lr_raw = df["p_lr_raw"].values
p_lr_sig = df["p_lr_sigmoid"].values
p_xgb_iso = df["p_xgb_isotonic"].values

b_lr_raw = brier_score_loss(y, p_lr_raw)
b_lr_sig = brier_score_loss(y, p_lr_sig)
b_xgb_iso = brier_score_loss(y, p_xgb_iso)

# quantile bins keeps similar sample size per bin
frac_raw, mean_raw = calibration_curve(y, p_lr_raw, n_bins=10, strategy="quantile")
frac_sig, mean_sig = calibration_curve(y, p_lr_sig, n_bins=10, strategy="quantile")
frac_xi,  mean_xi  = calibration_curve(y, p_xgb_iso, n_bins=10, strategy="quantile")

plt.figure()
plt.plot([0, 1], [0, 1], marker="o")
plt.plot(mean_raw, frac_raw, marker="o")
plt.plot(mean_sig, frac_sig, marker="o")
plt.plot(mean_xi,  frac_xi,  marker="o")

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed fraction (CIN2+)")
plt.title("Calibration: before vs after (OOF)")
plt.legend([
    "Ideal",
    f"LR raw (Brier={b_lr_raw:.3f})",
    f"LR sigmoid (Brier={b_lr_sig:.3f})",
    f"XGB isotonic (Brier={b_xgb_iso:.3f})"
], loc="best")
plt.grid(True)

save_png = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\calibration_compare.png"
plt.savefig(save_png, dpi=300, bbox_inches="tight")
print("Saved:", save_png)

plt.show()
