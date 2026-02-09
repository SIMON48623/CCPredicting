import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# === 1) 读入 + 与之前一致的清洗/构造 X,y ===
xlsx_path = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
df = pd.read_excel(xlsx_path, sheet_name="data_modelA(1)")

for c in ["age", "gravidity", "parity"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

pg = df["pathology_group"].astype(str).str.strip()
pg = pg.where(~pg.str.lower().isin({"", "nan", "none", "null", "unknown", "未知", "不详", "/"}), other=np.nan)
y = pd.to_numeric(pg, errors="coerce")
y = y.where(y.isin([0, 1]), other=np.nan)

keep = y.notna()
df = df.loc[keep].copy()
y = y.loc[keep].astype(int)

drop_cols = ["patient_id", "patient_name", "pathology_group"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# === 2) 定义列类型（先按常识划分，下一步可以微调） ===
num_cols = ["age", "gravidity", "parity"]

# 这些虽然是 0/1，但我们先当作类别/二值（OneHot 会产生 2 列或 1 列，二者都可）
bin_cols = ["HPV_overall", "HPV16", "HPV18", "HPV_other_hr",
            "iodine_negative", "atypical_vessels", "child_alive",
            "menopausal_status", "pathology_fig"]

cat_cols = ["cytology_grade", "colpo_impression", "TZ_type"]

# === 3) 预处理器（所有 fit 都在 CV 训练折内发生） ===
numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", cat_tf, cat_cols + bin_cols),
    ],
    remainder="drop"
)

# === 4) 跑 Stratified 5-fold，只验证不会报错，并打印每折阳性率 ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Total N:", len(y), "pos rate:", y.mean())

for i, (tr, va) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    # 关键：只在训练折 fit
    Xt_tr = preprocess.fit_transform(X_tr, y_tr)
    Xt_va = preprocess.transform(X_va)

    print(f"Fold {i}: train N={len(tr)} pos={y_tr.mean():.3f} | val N={len(va)} pos={y_va.mean():.3f} | Xt_tr shape={Xt_tr.shape} Xt_va shape={Xt_va.shape}")
