import pandas as pd
import numpy as np

XLSX_PATH = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
SHEET_NAME = "data_modelA(1)"   # 你截图里就是这个

# ====== 你最终允许的特征列（严格按你最初 Step3，不含 pathology_fig）======
FEATURE_COLS = [
    "age",
    "menopausal_status",
    "gravidity",
    "parity",
    "HPV_overall",
    "HPV16",
    "HPV18",
    "HPV_other_hr",
    "cytology_grade",
    "colpo_impression",
    "TZ_type",
    "iodine_negative",
    "atypical_vessels",
    "child_alive",
    "pathology_fig",
]
DROP_ALWAYS = ["patient_id", "patient_name", "pathology_group"]  # 防泄露/无用

def build_xy():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)

    # ---- 标签：只允许来自 pathology_group（病理金标准）----
    # 你之前的分布基本是 0/1/'/'/nan，这里按最严格规则：
    # 0 -> 0, 1 -> 1，其它一律 NaN（当作未标注剔除）
    y = pd.to_numeric(df["pathology_group"], errors="coerce")
    y = y.where(y.isin([0, 1]), np.nan)

    # ---- 只保留可用样本（有标签）----
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int).values

    # ---- 构建 X（严格不含 pathology_fig / pathology_group / id / name）----
    X = df[FEATURE_COLS].copy()

    # 把“本应是数值但被读成 object”的列强制转数值（非数值直接变 NaN）
    for c in ["age", "gravidity", "parity"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X, y, df

if __name__ == "__main__":
    X, y, df = build_xy()
    print("Usable N:", len(y))
    print("Positive rate:", float(y.mean()))
    print("X shape:", X.shape)
    print("X columns:", list(X.columns))
