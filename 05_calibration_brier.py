# SAVE: YES (Brier + calibration curve from OOF)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

oof_path = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\oof_predictions.csv"
df = pd.read_csv(oof_path)

y = df["y"].values.astype(int)
p_lr = df["p_lr"].values.astype(float)
p_xgb = df["p_xgb"].values.astype(float)

# 1) Brier scores
brier_lr = brier_score_loss(y, p_lr)
brier_xgb = brier_score_loss(y, p_xgb)

print("Brier (LR) :", brier_lr)
print("Brier (XGB):", brier_xgb)

# 2) Calibration curves (10 bins)
frac_lr, mean_lr = calibration_curve(y, p_lr, n_bins=10, strategy="quantile")
frac_x, mean_x = calibration_curve(y, p_xgb, n_bins=10, strategy="quantile")

plt.figure()
plt.plot([0, 1], [0, 1], marker="o")          # ideal
plt.plot(mean_lr, frac_lr, marker="o")       # LR
plt.plot(mean_x,  frac_x,  marker="o")       # XGB
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed fraction (CIN2+)")
plt.title("Calibration curve (OOF)")
plt.legend([f"Ideal", f"LR (Brier={brier_lr:.3f})", f"XGB (Brier={brier_xgb:.3f})"], loc="best")
plt.grid(True)

save_png = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\calibration_oof.png"
plt.savefig(save_png, dpi=200, bbox_inches="tight")
print("Saved plot:", save_png)

plt.show()
