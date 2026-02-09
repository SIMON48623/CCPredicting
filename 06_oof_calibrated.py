# SAVE: YES (OOF calibrated probabilities with fold-wise fitting)

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

from xgboost import XGBClassifier


# ---------------------------
# 0) Load + build X,y
# ---------------------------
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

# ---------------------------
# 1) Column types + preprocess
# ---------------------------
num_cols = ["age", "gravidity", "parity"]
bin_cols = [
    "HPV_overall", "HPV16", "HPV18", "HPV_other_hr",
    "iodine_negative", "atypical_vessels", "child_alive",
    "menopausal_status", "pathology_fig"
]
cat_cols = ["cytology_grade", "colpo_impression", "TZ_type"]

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

# ---------------------------
# 2) Models
# ---------------------------
base_lr = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
])

def make_xgb(scale_pos_weight: float):
    return Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        ))
    ])

# ---------------------------
# 3) OOF calibrated predictions (outer 5-fold)
# ---------------------------
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = pd.DataFrame({"y": y.values})
oof["fold"] = 0

# raw prob
oof["p_lr_raw"] = 0.0
oof["p_xgb_raw"] = 0.0

# calibrated prob
oof["p_lr_sigmoid"] = 0.0
oof["p_lr_isotonic"] = 0.0
oof["p_xgb_sigmoid"] = 0.0
oof["p_xgb_isotonic"] = 0.0

for fold, (tr, va) in enumerate(outer.split(X, y), 1):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    # ---- LR raw ----
    base_lr.fit(X_tr, y_tr)
    oof.loc[va, "p_lr_raw"] = base_lr.predict_proba(X_va)[:, 1]

    # ---- LR calibrated (fit calibrator on TRAIN fold only, using CV inside train) ----
    # cv=3 means calibrator uses only train-fold data split internally
    lr_sig = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
    lr_iso = CalibratedClassifierCV(base_lr, method="isotonic", cv=3)
    lr_sig.fit(X_tr, y_tr)
    lr_iso.fit(X_tr, y_tr)
    oof.loc[va, "p_lr_sigmoid"] = lr_sig.predict_proba(X_va)[:, 1]
    oof.loc[va, "p_lr_isotonic"] = lr_iso.predict_proba(X_va)[:, 1]

    # ---- XGB (fold-specific pos weight) ----
    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    spw = neg / max(pos, 1)

    base_xgb = make_xgb(spw)
    base_xgb.fit(X_tr, y_tr)
    oof.loc[va, "p_xgb_raw"] = base_xgb.predict_proba(X_va)[:, 1]

    xgb_sig = CalibratedClassifierCV(base_xgb, method="sigmoid", cv=3)
    xgb_iso = CalibratedClassifierCV(base_xgb, method="isotonic", cv=3)
    xgb_sig.fit(X_tr, y_tr)
    xgb_iso.fit(X_tr, y_tr)
    oof.loc[va, "p_xgb_sigmoid"] = xgb_sig.predict_proba(X_va)[:, 1]
    oof.loc[va, "p_xgb_isotonic"] = xgb_iso.predict_proba(X_va)[:, 1]

    oof.loc[va, "fold"] = fold
    print(f"Fold {fold} done. val N={len(va)} pos={y_va.mean():.3f}")

# ---------------------------
# 4) Compare Brier (and AUC/AP just for sanity)
# ---------------------------
def report(name, p):
    b = brier_score_loss(y, p)
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    print(f"{name:14s} | Brier={b:.4f} AUC={auc:.3f} AP={ap:.3f}")

print("\n=== OOF calibration comparison ===")
report("LR raw", oof["p_lr_raw"].values)
report("LR sigmoid", oof["p_lr_sigmoid"].values)
report("LR isotonic", oof["p_lr_isotonic"].values)

report("XGB raw", oof["p_xgb_raw"].values)
report("XGB sigmoid", oof["p_xgb_sigmoid"].values)
report("XGB isotonic", oof["p_xgb_isotonic"].values)

# ---------------------------
# 5) Save
# ---------------------------
save_path = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\oof_calibrated.csv"
oof.to_csv(save_path, index=False, encoding="utf-8-sig")
print("\nSaved:", save_path)
print(oof.head())
