import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap


# =======================
# CONFIG (你只需要改这里)
# =======================
XLSX_PATH = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
SHEET_NAME = None  # None=默认第一张；如果要指定，写 "data_modelA(1)" 之类
OUT_DIR = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA"

RANDOM_STATE = 42
SHAP_SAMPLE = 600   # 做 SHAP 的样本数（越大越慢，先 400~800 都可以）


def read_data(xlsx_path: str, sheet_name=None) -> pd.DataFrame:
    if sheet_name is None:
        df = pd.read_excel(xlsx_path)
    else:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    return df


def make_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    y_cin2plus: from pathology_group
    - pathology_group == 1 => positive
    - pathology_group == 0 => negative
    - anything else => treated as missing and dropped
    """
    if "pathology_group" not in df.columns:
        raise ValueError("Missing column: pathology_group")

    pg = pd.to_numeric(df["pathology_group"], errors="coerce")
    df = df.copy()
    df["y_cin2plus"] = pg  # temporarily numeric 0/1/NaN

    before = len(df)
    df = df.dropna(subset=["y_cin2plus"]).copy()
    df["y_cin2plus"] = (df["y_cin2plus"].astype(int) == 1).astype(int)

    pos = int(df["y_cin2plus"].sum())
    neg = int((df["y_cin2plus"] == 0).sum())
    print(f"Label usable N = {len(df)} (dropped {before-len(df)} unlabeled)")
    print(f"Positive = {pos} ({pos/len(df):.3f}), Negative = {neg} ({neg/len(df):.3f})")
    return df


def coerce_numeric_only(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) load
    df = read_data(XLSX_PATH, sheet_name=SHEET_NAME)
    print("Raw shape:", df.shape)
    print("Columns:", list(df.columns))

    # 2) label
    df = make_label(df)

    # 3) strict drop (ID + name + pathology_group)；保留 pathology_fig 作为医生判断特征（不是金标准）
    drop_cols = []
    for c in ["patient_id", "patient_name", "pathology_group"]:
        if c in df.columns:
            drop_cols.append(c)
    df = df.drop(columns=drop_cols, errors="ignore")

    # 4) feature set（与你现在一致：包含 pathology_fig）
    feat_cols = [
        "age", "menopausal_status", "gravidity", "parity",
        "HPV_overall", "HPV16", "HPV18", "HPV_other_hr",
        "cytology_grade", "colpo_impression", "TZ_type",
        "iodine_negative", "atypical_vessels", "child_alive",
        "pathology_fig"
    ]

    df = coerce_numeric_only(df, feat_cols)

    X = df[feat_cols].copy()
    y = df["y_cin2plus"].astype(int).values

    print("X shape:", X.shape, "y mean:", y.mean())

    # 5) fill missing (tree model不需要标准化；只填补即可)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    feature_names = feat_cols

    # 6) train a clean XGBClassifier (no weird base_score strings)
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = float(neg / max(pos, 1))
    print("scale_pos_weight:", scale_pos_weight)

    clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    clf.fit(X_imp, y)
    print("XGB fitted OK.")

    # 7) SHAP on a sample (speed)
    n = X_imp.shape[0]
    k = min(SHAP_SAMPLE, n)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(n, size=k, replace=False)
    X_sample = X_imp[idx]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # SHAP may return list for binary in some versions
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # 8) save plots
    # beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    out_swarm = os.path.join(OUT_DIR, "shap_xgb_beeswarm.png")
    plt.tight_layout()
    plt.savefig(out_swarm, dpi=300)
    plt.close()
    print("Saved:", out_swarm)

    # bar
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    out_bar = os.path.join(OUT_DIR, "shap_xgb_bar.png")
    plt.tight_layout()
    plt.savefig(out_bar, dpi=300)
    plt.close()
    print("Saved:", out_bar)

    # 9) save importance table
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    out_csv = os.path.join(OUT_DIR, "shap_xgb_importance.csv")
    imp.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)
    print("\nTop 10 importance:\n", imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
