# 12_threshold_report.py
# Threshold report for TAB-MFM raw (performance) and TAB-MFM sigmoid (calibrated probability)
# Input: oof_calibrated_all_with_mfm.csv
# Outputs:
#   - threshold_report_mfm_raw.csv / threshold_recommendation_mfm_raw.txt
#   - threshold_report_mfm_sigmoid.csv / threshold_recommendation_mfm_sigmoid.txt

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

BASE_DIR = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA"
IN_CSV   = os.path.join(BASE_DIR, "oof_calibrated_all_with_mfm.csv")

Y_COL = "y"

TARGETS = [
    ("mfm_raw",     "p_tab_mfm_raw"),
    ("mfm_sigmoid", "p_tab_mfm_sigmoid"),
]

# 你之前常用 DCA 区间 0.05~0.40，这里保持一致更利于临床讨论
THS = np.round(np.arange(0.05, 0.401, 0.01), 2)

def compute_metrics(y, p, th):
    y = y.astype(int)
    pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()

    se = tp / (tp + fn + 1e-12)  # sensitivity
    sp = tn / (tn + fp + 1e-12)  # specificity
    ppv = tp / (tp + fp + 1e-12)
    npv = tn / (tn + fn + 1e-12)
    f1 = f1_score(y, pred, zero_division=0)
    youden = se + sp - 1.0
    return {
        "th": th,
        "Se": se, "Sp": sp, "PPV": ppv, "NPV": npv,
        "F1": f1, "Youden": youden,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }

def format_recommendation(tag, n, prev, df_rep):
    # 1) screen-first: high sensitivity
    row1 = df_rep.iloc[(df_rep["Se"] - 0.95).abs().argsort()[:1]].iloc[0]
    # 2) balanced: choose 0.20~0.30 if available, else median
    mid = df_rep[(df_rep["th"] >= 0.20) & (df_rep["th"] <= 0.30)]
    row2a = mid.iloc[(mid["F1"].values).argmax()] if len(mid) else df_rep.iloc[(df_rep["F1"].values).argmax()]
    # 3) max youden
    row3 = df_rep.iloc[(df_rep["Youden"].values).argmax()]

    def per1000(row):
        tp = row["TP"] / n * 1000
        fp = row["FP"] / n * 1000
        fn = row["FN"] / n * 1000
        return tp, fp, fn

    tp1, fp1, fn1 = per1000(row1)
    tp2, fp2, fn2 = per1000(row2a)
    tp3, fp3, fn3 = per1000(row3)

    txt = []
    txt.append(f"阈值选择建议（基于 OOF 概率，方案：{tag}）")
    txt.append(f"- 数据规模 N={n}, 阳性率(CIN2+)≈{prev:.3f}")
    txt.append("")
    txt.append("1) 筛查优先（更高敏感度）：")
    txt.append(f"   建议阈值 pt≈{row1['th']:.2f}")
    txt.append(f"   Se={row1['Se']:.3f}, Sp={row1['Sp']:.3f}, PPV={row1['PPV']:.3f}, NPV={row1['NPV']:.3f}, F1={row1['F1']:.3f}")
    txt.append(f"   每1000人预计：TP={tp1:.1f}, FP={fp1:.1f}, FN={fn1:.1f}")
    txt.append("")
    txt.append("2) 分流/资源平衡（常用区间 0.20~0.30，取F1较优）：")
    txt.append(f"   建议阈值 pt≈{row2a['th']:.2f}")
    txt.append(f"   Se={row2a['Se']:.3f}, Sp={row2a['Sp']:.3f}, PPV={row2a['PPV']:.3f}, NPV={row2a['NPV']:.3f}, F1={row2a['F1']:.3f}")
    txt.append(f"   每1000人预计：TP={tp2:.1f}, FP={fp2:.1f}, FN={fn2:.1f}")
    txt.append("")
    txt.append("3) 综合权衡（最大 Youden 指数）：")
    txt.append(f"   推荐阈值 pt≈{row3['th']:.2f}")
    txt.append(f"   Se={row3['Se']:.3f}, Sp={row3['Sp']:.3f}, PPV={row3['PPV']:.3f}, NPV={row3['NPV']:.3f}, F1={row3['F1']:.3f}")
    txt.append(f"   每1000人预计：TP={tp3:.1f}, FP={fp3:.1f}, FN={fn3:.1f}")
    txt.append("")
    txt.append("备注：")
    txt.append("- 若强调“概率解释/临床沟通”，优先使用 sigmoid 校准版；")
    txt.append("- 若强调“检出/排序性能”，可用 raw 作为性能主推，并同时给出校准版概率供解释。")
    return "\n".join(txt)

def run_one(tag, p_col):
    df = pd.read_csv(IN_CSV)
    if Y_COL not in df.columns:
        raise ValueError(f"Missing {Y_COL} in {IN_CSV}")
    if p_col not in df.columns:
        raise ValueError(f"Missing {p_col} in {IN_CSV}")

    y = df[Y_COL].astype(int).values
    p = df[p_col].astype(float).values
    n = len(y)
    prev = y.mean()

    rows = [compute_metrics(y, p, float(th)) for th in THS]
    rep = pd.DataFrame(rows)

    out_csv = os.path.join(BASE_DIR, f"threshold_report_{tag}.csv")
    out_txt = os.path.join(BASE_DIR, f"threshold_recommendation_{tag}.txt")

    rep.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(format_recommendation(tag, n, prev, rep))

    print("Saved:", out_csv)
    print("Saved:", out_txt)

def main():
    for tag, col in TARGETS:
        run_one(tag, col)

if __name__ == "__main__":
    main()
