# 17_subgroup_eval_tabmfm.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- inputs ---
OOF_PATH = os.path.join(BASE_DIR, "oof_calibrated_all_with_mfm.csv")

DEFAULT_XLSX = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
LOCAL_XLSX = os.path.join(BASE_DIR, "data_modelA终.xlsx")
XLSX_PATH = LOCAL_XLSX if os.path.exists(LOCAL_XLSX) else DEFAULT_XLSX
SHEET_NAME = "data_modelA(1)"

# --- subgroup columns we will use (must exist in xlsx) ---
SUBGROUP_COLS = ["age", "HPV_overall", "HPV16", "HPV18", "cytology_grade"]

def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))

def safe_ap(y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))

def metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    return {
        "n": int(len(y)),
        "pos_rate": float(y.mean()) if len(y) else np.nan,
        "auc": safe_auc(y, p),
        "ap": safe_ap(y, p),
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }

def load_xlsx_features_and_y():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)

    y = pd.to_numeric(df["pathology_group"], errors="coerce")
    y = y.where(y.isin([0, 1]), np.nan)
    keep = y.notna()

    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int).values

    # Keep subgroup cols only
    missing = [c for c in SUBGROUP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing subgroup columns in XLSX: {missing}")

    sub = df[SUBGROUP_COLS].copy()
    # numeric coercion (safe)
    for c in ["age", "cytology_grade"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    for c in ["HPV_overall", "HPV16", "HPV18"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    return sub.reset_index(drop=True), y

def main():
    oof = pd.read_csv(OOF_PATH)
    if "p_tab_mfm_sigmoid" not in oof.columns:
        raise ValueError("oof file missing column: p_tab_mfm_sigmoid")
    y_oof = oof["y"].astype(int).values
    p = oof["p_tab_mfm_sigmoid"].astype(float).values

    sub, y_x = load_xlsx_features_and_y()

    # alignment guard (critical)
    if len(y_x) != len(y_oof):
        raise ValueError(f"Row mismatch: xlsx={len(y_x)} vs oof={len(y_oof)}")
    if not np.array_equal(y_x, y_oof):
        raise ValueError("y mismatch between XLSX and OOF (order/filters not aligned)")

    df = pd.concat([sub, oof[["y"]]], axis=1)

    rows = []
    # overall
    rows.append({"group": "ALL", **metrics(y_oof, p)})

    # Age bins
    if df["age"].notna().any():
        rows.append({"group": "age<35", **metrics(df.loc[df["age"] < 35, "y"], p[df["age"] < 35])})
        rows.append({"group": "age35-49", **metrics(df.loc[(df["age"] >= 35) & (df["age"] < 50), "y"], p[(df["age"] >= 35) & (df["age"] < 50)])})
        rows.append({"group": "age>=50", **metrics(df.loc[df["age"] >= 50, "y"], p[df["age"] >= 50])})

    # HPV overall / 16 / 18
    for col in ["HPV_overall", "HPV16", "HPV18"]:
        if df[col].notna().any():
            rows.append({"group": f"{col}=0", **metrics(df.loc[df[col] == 0, "y"], p[df[col] == 0])})
            rows.append({"group": f"{col}=1", **metrics(df.loc[df[col] == 1, "y"], p[df[col] == 1])})

    # Cytology bins (0-2 vs 3-5)
    if df["cytology_grade"].notna().any():
        low_mask = df["cytology_grade"].between(0, 2)
        high_mask = df["cytology_grade"].between(3, 5)
        rows.append({"group": "cytology_low(0-2)", **metrics(df.loc[low_mask, "y"], p[low_mask])})
        rows.append({"group": "cytology_high(3-5)", **metrics(df.loc[high_mask, "y"], p[high_mask])})

    out = pd.DataFrame(rows)
    out_path = os.path.join(BASE_DIR, "subgroup_metrics_tabmfm_sigmoid.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved:", out_path)
    print(out)

if __name__ == "__main__":
    main()
