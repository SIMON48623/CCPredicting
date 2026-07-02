import pandas as pd
import numpy as np
import os

BASE = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA"
f_all = os.path.join(BASE, "oof_predictions_all.csv")
f_tab = os.path.join(BASE, "tab_oof_predictions.csv")

a = pd.read_csv(f_all)
t = pd.read_csv(f_tab)

print("rows all/tab:", len(a), len(t))
print("y aligned:", np.array_equal(a["y"].values, t["y"].values))
print("fold aligned:", np.array_equal(a["fold"].values, t["fold"].values))

# 关键：p_tab 是否一致（必须一致才说明链路是最新的）
diff = np.max(np.abs(a["p_tab"].values.astype(float) - t["p_tab"].values.astype(float)))
corr = np.corrcoef(a["p_tab"].values.astype(float), t["p_tab"].values.astype(float))[0,1]
print("p_tab max_abs_diff:", diff)
print("p_tab corr:", corr)

print("\nHEAD all p_tab:", a["p_tab"].head().tolist())
print("HEAD tab p_tab:", t["p_tab"].head().tolist())
