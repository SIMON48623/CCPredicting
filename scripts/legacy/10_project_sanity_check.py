import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = [
    "dataset_build.py",
    "02_baseline_lr_cv.py",
    "03_baseline_xgb_cv.py",
    "04_oof_predictions.py",
    "oof_predictions.csv",
    "oof_calibrated.csv",
    "calibration_oof.png",
    "calibration_compare.png",
    "dca_oof.png",
    "shap_xgb_summary.png",
    "shap_xgb_bar.png",
]

def check_exists():
    print("=== File existence check ===")
    missing = []
    for f in REQUIRED_FILES:
        p = os.path.join(ROOT, f)
        ok = os.path.exists(p)
        print(f"[{'OK' if ok else 'MISS'}] {f}")
        if not ok:
            missing.append(f)
    return missing

def check_oof_predictions():
    print("\n=== oof_predictions.csv check ===")
    p = os.path.join(ROOT, "oof_predictions.csv")
    if not os.path.exists(p):
        print("[MISS] oof_predictions.csv not found, skip.")
        return

    df = pd.read_csv(p)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    required_cols = {"y", "fold"}
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    if not required_cols.issubset(df.columns):
        print("[WARN] missing columns:", required_cols - set(df.columns))
    if len(prob_cols) == 0:
        print("[WARN] no probability columns starting with 'p_' found.")
    else:
        print("[OK] prob columns:", prob_cols)

def check_oof_calibrated():
    print("\n=== oof_calibrated.csv check ===")
    p = os.path.join(ROOT, "oof_calibrated.csv")
    if not os.path.exists(p):
        print("[MISS] oof_calibrated.csv not found, skip.")
        return

    df = pd.read_csv(p)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    required_cols = {"y", "fold"}
    if not required_cols.issubset(df.columns):
        print("[WARN] missing columns:", required_cols - set(df.columns))
    else:
        print("[OK] has y and fold")

def main():
    missing = check_exists()
    check_oof_predictions()
    check_oof_calibrated()

    print("\n=== Summary ===")
    if missing:
        print("[FAIL] Missing files:", missing)
    else:
        print("[PASS] All required files exist.")

if __name__ == "__main__":
    main()
