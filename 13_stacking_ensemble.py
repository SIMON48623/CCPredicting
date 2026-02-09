# 13_stacking_ensemble.py
# OOF stacking ensemble (no leakage):
#   - Reads oof_calibrated_all.csv
#   - Trains meta-model with Stratified 5-fold CV to produce meta-OOF p_ens
#   - Fits final meta-model on all data and saves it for deployment
# Outputs:
#   - oof_calibrated_all_with_ens.csv (adds p_ens)
#   - stacking_meta_model.joblib
#   - stacking_metrics.csv

import os
import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib


# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA"
IN_CSV   = os.path.join(BASE_DIR, "oof_calibrated_all.csv")

OUT_CSV_WITH_ENS = os.path.join(BASE_DIR, "oof_calibrated_all_with_ens.csv")
OUT_MODEL        = os.path.join(BASE_DIR, "stacking_meta_model.joblib")
OUT_METRICS      = os.path.join(BASE_DIR, "stacking_metrics.csv")

RANDOM_SEED = 42
N_SPLITS = 5

Y_COL = "y"

# 默认用这三个做融合（你也可删减）
META_FEATURE_COLS = [
    "p_tab_sigmoid",
    "p_lr_sigmoid",
    "p_xgb_isotonic",
]

META_MODEL = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        C=1.0,
        random_state=RANDOM_SEED
    ))
])


# =========================
# UTILS
# =========================
def check_cols(df, cols, in_csv):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {in_csv}: {missing}")

def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")

def safe_ap(y, p):
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")

def safe_brier(y, p):
    try:
        return float(brier_score_loss(y, p))
    except Exception:
        return float("nan")

def fill_nan_with_median(X):
    # X: numpy array [N, d]
    X2 = X.copy()
    med = np.nanmedian(X2, axis=0)
    # if a whole column is NaN -> nanmedian becomes NaN; replace with 0
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(X2))
    X2[inds] = np.take(med, inds[1])
    return X2


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(IN_CSV)
    check_cols(df, [Y_COL] + META_FEATURE_COLS, IN_CSV)

    y = df[Y_COL].values.astype(int)
    X = df[META_FEATURE_COLS].values.astype(float)

    # fill NaN if any
    X = fill_nan_with_median(X)

    n = len(y)
    prev = float(y.mean())
    print(f"N={n}, prevalence={prev:.4f}")
    print("Meta features:", META_FEATURE_COLS)

    # -------------------------
    # 1) Meta OOF (2nd-level CV)
    # -------------------------
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    p_ens_oof = np.zeros(n, dtype=float)

    fold_rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        model = copy.deepcopy(META_MODEL)
        model.fit(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:, 1]
        p_ens_oof[va] = p_va

        auc = safe_auc(y_va, p_va)
        ap  = safe_ap(y_va, p_va)
        br  = safe_brier(y_va, p_va)

        fold_rows.append((fold, len(va), float(y_va.mean()), auc, ap, br))
        print(f"[fold {fold}] valN={len(va)} pos={y_va.mean():.3f} | AUC={auc:.3f} AP={ap:.3f} Brier={br:.3f}")

    ens_auc = safe_auc(y, p_ens_oof)
    ens_ap  = safe_ap(y, p_ens_oof)
    ens_br  = safe_brier(y, p_ens_oof)

    print("\n=== Ensemble (meta-OOF) overall ===")
    print(f"ENS AUC={ens_auc:.4f} | AP={ens_ap:.4f} | Brier={ens_br:.4f}")

    # -------------------------
    # 2) Baselines for reference
    # -------------------------
    metric_rows = []

    def add_metric(name, col):
        if col not in df.columns:
            return
        p = df[col].values.astype(float)
        metric_rows.append({
            "model": name,
            "col": col,
            "AUC": safe_auc(y, p),
            "AP": safe_ap(y, p),
            "Brier": safe_brier(y, p),
        })

    add_metric("TAB sigmoid", "p_tab_sigmoid")
    add_metric("LR sigmoid", "p_lr_sigmoid")
    add_metric("XGB isotonic", "p_xgb_isotonic")
    add_metric("TAB raw", "p_tab_raw")
    add_metric("LR raw", "p_lr_raw")
    add_metric("XGB raw", "p_xgb_raw")

    metric_rows.append({
        "model": "ENS (stacking meta-OOF)",
        "col": "p_ens",
        "AUC": ens_auc,
        "AP": ens_ap,
        "Brier": ens_br,
    })

    metrics_df = pd.DataFrame(metric_rows).sort_values(["AP", "AUC"], ascending=False).reset_index(drop=True)
    metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_METRICS)
    print("\nTop metrics (sorted by AP then AUC):")
    print(metrics_df.head(20).to_string(index=False))

    # -------------------------
    # 3) Save df with ensemble OOF
    # -------------------------
    df_out = df.copy()
    df_out["p_ens"] = p_ens_oof.astype(float)
    df_out.to_csv(OUT_CSV_WITH_ENS, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_CSV_WITH_ENS)

    # -------------------------
    # 4) Fit final meta-model on ALL data for deployment
    # -------------------------
    final_model = copy.deepcopy(META_MODEL)
    final_model.fit(X, y)

    payload = {
        "meta_feature_cols": META_FEATURE_COLS,
        "median_impute": np.nanmedian(df[META_FEATURE_COLS].values.astype(float), axis=0).tolist(),
        "model": final_model
    }
    joblib.dump(payload, OUT_MODEL)
    print("Saved:", OUT_MODEL)

    fold_df = pd.DataFrame(fold_rows, columns=["fold", "valN", "val_pos_rate", "AUC", "AP", "Brier"])
    print("\nFold summary (meta-OOF):")
    print(fold_df.to_string(index=False))


if __name__ == "__main__":
    main()
