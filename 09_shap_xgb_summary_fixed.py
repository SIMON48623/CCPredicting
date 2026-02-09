import os
import numpy as np
import pandas as pd

import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# =========================
# 1) 配置：路径 + 列
# =========================
XLSX_PATH = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
SHEET_NAME = "data_modelA(1)"  # 你之前打印出来的 sheet 名

# 你当前项目的严格 drop（防泄露 + 无意义 ID/姓名）
DROP_COLS = ["patient_id", "patient_name", "pathology_group"]

# 标签：你现在用的是 CIN2+（y_cin2plus），对应 pathology_group=1 为阳性
LABEL_COL = "y_cin2plus"

# 你保留的特征（注意：你现在把 pathology_fig 当作特征了——它是医生影像判断，严格说会引入“与金标准强相关的临床判断”，
# 不一定算泄露，但论文里要写清楚：这是“临床评估变量”。如果你想更纯粹的“客观检查变量”，后面我会让你做一版不含它的消融）
FEATURE_COLS = [
    "age", "menopausal_status", "gravidity", "parity",
    "HPV_overall", "HPV16", "HPV18", "HPV_other_hr",
    "cytology_grade", "colpo_impression", "TZ_type",
    "iodine_negative", "atypical_vessels", "child_alive",
    "pathology_fig",
]

# 采样多少行来画 SHAP（全量 879 行也能画，但慢；先 400 行够用）
N_SAMPLE = 400
RANDOM_STATE = 42

# 输出图路径
OUT_PNG = os.path.join(os.path.dirname(__file__), "shap_xgb_summary.png")


# =========================
# 2) 读数据 + 造标签 + 清洗
# =========================
def load_data():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)

    # ---- 你说：非数值都当未知，洗掉 ----
    # 但为了可重复，我们只对“应该是数值”的列做强制转数值：
    for c in ["age", "gravidity", "parity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 造标签：pathology_group == 1 -> CIN2+ 阳性 ----
    # pathology_group 里你截图显示有 0/1 以及少量 '/' 和 nan
    pg = pd.to_numeric(df["pathology_group"], errors="coerce")
    y = (pg == 1).astype(float)  # float 方便后续 dropna
    df[LABEL_COL] = y

    # drop unlabeled
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    return df


# =========================
# 3) 训练一个 XGB（用于解释）
# =========================
def train_xgb(X, y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos_weight = (neg / max(pos, 1))

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    clf.fit(X, y)
    return clf


# =========================
# 4) 用 XGBoost 原生 pred_contribs 算 SHAP
#    ✅ 关键：不再使用 shap.TreeExplainer（避开 base_score bug）
# =========================
def compute_shap_by_pred_contribs(clf, X, feature_names):
    booster = clf.get_booster()
    dmat = xgb.DMatrix(X, feature_names=feature_names)

    # (n, n_features + 1)，最后一列是 bias（base value）
    contrib = booster.predict(dmat, pred_contribs=True)

    # 拆开
    shap_values = contrib[:, :-1]
    base_values = contrib[:, -1]

    return shap_values, base_values


# =========================
# 5) 主程序
# =========================
def main():
    df = load_data()

    # 严格 drop
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 准备 X/y
    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].values.astype(int)

    print("Usable N:", len(df))
    print("Positive rate:", y.mean())
    print("X shape:", X.shape)

    # 缺失简单处理：XGBoost 能处理 NaN，这里不做 impute（解释性阶段足够）
    # 如果你想和 baseline 完全一致，也可以在这里加你 Step4 的折内预处理流程。

    # 训练
    clf = train_xgb(X, y)
    print("XGB fitted OK.")

    # 取样（避免图太慢）
    rs = np.random.RandomState(RANDOM_STATE)
    if len(X) > N_SAMPLE:
        idx = rs.choice(len(X), size=N_SAMPLE, replace=False)
        X_plot = X.iloc[idx].copy()
    else:
        X_plot = X.copy()

    # 计算 SHAP
    shap_vals, base_vals = compute_shap_by_pred_contribs(
        clf, X_plot, feature_names=list(X_plot.columns)
    )

    # 画 summary
    plt.figure()
    shap.summary_plot(shap_vals, X_plot, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    print("Saved:", OUT_PNG)

    # 如果你还想要 bar 版（平均|SHAP|）
    OUT_BAR = os.path.join(os.path.dirname(__file__), "shap_xgb_bar.png")
    plt.figure()
    shap.summary_plot(shap_vals, X_plot, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT_BAR, dpi=300)
    print("Saved:", OUT_BAR)


if __name__ == "__main__":
    main()
