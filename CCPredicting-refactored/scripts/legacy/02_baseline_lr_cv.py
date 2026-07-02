import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

from dataset_build import build_xy



def best_threshold_by_f1(y_true, p):
    # grid search threshold on validation probabilities
    ths = np.linspace(0.05, 0.95, 19)
    best = (0.5, -1)
    for th in ths:
        y_hat = (p >= th).astype(int)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best[1]:
            best = (th, f1)
    return best


def main():
    X, y, _ = build_xy()
    print(f"Total N: {len(y)}  pos rate: {y.mean():.6f}")

    num_cols = list(X.columns)  # all numeric/coded columns
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols)
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced"
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        pipe.fit(Xtr, ytr)
        pva = pipe.predict_proba(Xva)[:, 1]

        # choose threshold by inner F1 (simple on same val; keep consistent with your earlier scripts)
        th, inner_best_f1 = best_threshold_by_f1(yva, pva)
        yhat = (pva >= th).astype(int)

        auc = roc_auc_score(yva, pva)
        ap = average_precision_score(yva, pva)
        f1 = f1_score(yva, yhat, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(yva, yhat).ravel()
        se = tp / (tp + fn + 1e-12)
        sp = tn / (tn + fp + 1e-12)

        rows.append({
            "fold": fold, "th": th,
            "auc": auc, "ap": ap, "f1": f1,
            "sensitivity": se, "specificity": sp,
            "inner_best_f1": inner_best_f1
        })

        print(f"Fold {fold}: th={th:.2f} | AUC={auc:.3f} AP={ap:.3f} F1={f1:.3f} "
              f"Se={se:.3f} Sp={sp:.3f} (inner best F1={inner_best_f1:.3f})")

    res = pd.DataFrame(rows)
    print("\n=== CV summary (mean + std) ===")
    for c in ["auc", "ap", "f1", "sensitivity", "specificity"]:
        print(f"{c:12s}: {res[c].mean():.3f} ± {res[c].std(ddof=1):.3f}")

    print("\nThresholds per fold:", [round(x, 2) for x in res["th"].tolist()])


if __name__ == "__main__":
    main()
