# 04_oof_predictions.py
# Build OOF predictions for LR & XGB, and merge latest TAB OOF into a single file (no version mixing).
#
# Outputs:
#   - oof_predictions.csv        (y, fold, p_lr, p_xgb)
#   - oof_predictions_all.csv    (y, fold, p_lr, p_xgb, p_tab)  [if tab_oof_predictions.csv exists]
#
# Key fixes:
#   - Replace placeholders like '_' with NaN and coerce ALL 15 features to numeric
#     to avoid SimpleImputer(strategy="median") failing on strings.
#   - Strict alignment check when merging TAB: y and fold must match exactly.

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


# =======================
# 0) CONFIG
# =======================
BASE_DIR = os.path.dirname(__file__)

XLSX_PATH  = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
SHEET_NAME = "data_modelA(1)"
RANDOM_SEED = 42
N_SPLITS = 5

LABEL_COL = "pathology_group"

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

DROP_ALWAYS = ["patient_id", "patient_name", LABEL_COL]

# Output files
OUT_LR_XGB = os.path.join(BASE_DIR, "oof_predictions.csv")
OUT_ALL    = os.path.join(BASE_DIR, "oof_predictions_all.csv")

# TAB OOF produced by 11_tab_transformer_cv.py
TAB_OOF = os.path.join(BASE_DIR, "tab_oof_predictions.csv")

# Placeholders that should be treated as missing
MISSING_TOKENS = ["_", " ", "", "NA", "N/A", "na", "None", "none", "NULL", "null", "unknown", "UNKNOWN"]


# =======================
# 1) DATA LOADING (robust)
# =======================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def load_xy():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)

    # label: 0/1 only
    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    y = y.where(y.isin([0, 1]), np.nan)

    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int).values

    # drop always
    for c in DROP_ALWAYS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # ensure features exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS].copy()

    # ---- FIX: clean placeholders and coerce to numeric ----
    X = X.replace(MISSING_TOKENS, np.nan)
    for c in FEATURE_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # ------------------------------------------------------

    return X, y


# =======================
# 2) PREPROCESSOR
# =======================
def build_preprocessor():
    """
    Keep it stable with your earlier LR/XGB pipeline:
    - numeric: median impute + standardize
    - categorical: most_frequent + onehot
    Here we treat these as categorical (binary/ordinal-coded), and the rest as numeric.
    """
    cat_cols = [
        "menopausal_status",
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
    ]
    num_cols = [c for c in FEATURE_COLS if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop"
    )
    return pre


# =======================
# 3) OOF TRAINING
# =======================
def run_lr_xgb_oof(X: pd.DataFrame, y: np.ndarray):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_id = np.zeros(len(y), dtype=int)
    p_lr = np.zeros(len(y), dtype=float)
    p_xgb = np.zeros(len(y), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        fold_id[va] = fold

        pre = build_preprocessor()

        # LR
        lr = LogisticRegression(max_iter=500, solver="lbfgs")
        lr_pipe = Pipeline([("pre", pre), ("clf", lr)])
        lr_pipe.fit(X_tr, y_tr)
        p_lr[va] = lr_pipe.predict_proba(X_va)[:, 1]

        # XGB
        xgb = XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
        )
        xgb_pipe = Pipeline([("pre", pre), ("clf", xgb)])
        xgb_pipe.fit(X_tr, y_tr)
        p_xgb[va] = xgb_pipe.predict_proba(X_va)[:, 1]

        print(f"Fold {fold} done. val N={len(va)} pos={y_va.mean():.3f}", flush=True)

    df_out = pd.DataFrame({
        "y": y.astype(int),
        "fold": fold_id.astype(int),
        "p_lr": p_lr,
        "p_xgb": p_xgb,
    })

    df_out.to_csv(OUT_LR_XGB, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_LR_XGB, flush=True)
    print(df_out.head(), flush=True)

    return df_out


# =======================
# 4) MERGE TAB (optional)
# =======================
def merge_tab_if_exists(df_lr_xgb: pd.DataFrame):
    if not os.path.exists(TAB_OOF):
        print(f"[warn] TAB OOF not found: {TAB_OOF}", flush=True)
        print("[warn] Skipping oof_predictions_all.csv (no p_tab).", flush=True)
        return

    tab = pd.read_csv(TAB_OOF)

    # required columns
    for c in ["y", "fold", "p_tab"]:
        if c not in tab.columns:
            raise ValueError(f"TAB OOF missing column: {c}")

    # strict alignment check (prevents silent mixing)
    if len(tab) != len(df_lr_xgb):
        raise RuntimeError(f"Row count mismatch: LR/XGB={len(df_lr_xgb)} vs TAB={len(tab)}")

    if not np.array_equal(df_lr_xgb["y"].values, tab["y"].values):
        raise RuntimeError("Alignment mismatch on 'y' between LR/XGB and TAB OOF.")

    if not np.array_equal(df_lr_xgb["fold"].values, tab["fold"].values):
        raise RuntimeError("Alignment mismatch on 'fold' between LR/XGB and TAB OOF.")

    out_all = df_lr_xgb.copy()
    out_all["p_tab"] = tab["p_tab"].astype(float).values

    out_all.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_ALL, flush=True)
    print(out_all.head(), flush=True)


def main():
    set_seed(RANDOM_SEED)

    X, y = load_xy()
    print(f"Usable N: {len(y)} | Positive rate: {y.mean():.4f}", flush=True)
    print(f"X shape: {X.shape}", flush=True)
    print(f"FEATURE_COLS: {FEATURE_COLS}", flush=True)

    if X.shape[1] != 15:
        raise RuntimeError(f"Expected 15 features, got {X.shape[1]}")

    df_lr_xgb = run_lr_xgb_oof(X, y)
    merge_tab_if_exists(df_lr_xgb)


if __name__ == "__main__":
    main()
