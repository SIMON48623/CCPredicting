# 15_run_tabmfm_ablations.py
# (FIXED) One-click runner for Tab-MFM ablations using ENV vars + subprocess (most robust on Windows).

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SCRIPT_14 = os.path.join(BASE_DIR, "14_tab_mfm_pretrain_and_finetune_cv.py")

def run_one(tag, do_pretrain, mask_ratio, use_col_id_emb):
    env = os.environ.copy()
    env["TABMFM_TAG"] = str(tag)
    env["TABMFM_DO_PRETRAIN"] = "1" if do_pretrain else "0"
    env["TABMFM_MASK_RATIO"] = str(mask_ratio)
    env["TABMFM_USE_COL_ID_EMB"] = "1" if use_col_id_emb else "0"

    print("\n" + "="*90)
    print(f"[RUN] tag={tag} | do_pretrain={do_pretrain} | mask_ratio={mask_ratio} | use_col_id_emb={use_col_id_emb}")
    print("="*90, flush=True)

    subprocess.check_call([PY, SCRIPT_14], env=env)

def main():
    configs = [
        ("mfm_pretrain_mask030_colid1", True,  0.30, True),
        ("mfm_nopretrain_mask030_colid1", False, 0.30, True),
        ("mfm_pretrain_mask030_colid0", True,  0.30, False),
        ("mfm_pretrain_mask015_colid1", True,  0.15, True),
    ]

    for (tag, do_pretrain, mask_ratio, use_col_id_emb) in configs:
        run_one(tag, do_pretrain, mask_ratio, use_col_id_emb)

    print("\nAll ablations finished.")
    print("Next (recommended):")
    print("  python 06c_oof_calibrate_tabmfm_and_merge.py")
    print("  python 09_metrics_summary.py")
    print("  python 12_threshold_report.py")

if __name__ == "__main__":
    main()
