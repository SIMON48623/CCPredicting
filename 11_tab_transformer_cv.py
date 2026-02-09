# 11_tab_transformer_cv.py
# FT-Transformer style (NUM + CAT embeddings) + Stratified 5-fold OOF (no leakage)
# Label: pathology_group (0/1 only)
#
# Output (backward compatible + safer):
#   - tab_oof_predictions.csv         (y, fold, p_tab_raw, p_tab)
#   - tab_oof_predictions_<TAG>.csv   (same content)
#
# Notes:
# - Numeric: median impute + standardize (fit on train fold only)
# - Categorical: per-fold vocab (fit on train fold only), unseen -> 0 (UNK)
# - CPU friendly
#
# You do NOT need to modify anything. Defaults keep your current pipeline working.

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =======================
# 1) CONFIG
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# keep your original absolute path, but also support "data_modelA终.xlsx" placed in BASE_DIR
DEFAULT_XLSX = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
LOCAL_XLSX = os.path.join(BASE_DIR, "data_modelA终.xlsx")
XLSX_PATH = LOCAL_XLSX if os.path.exists(LOCAL_XLSX) else DEFAULT_XLSX

SHEET_NAME  = "data_modelA(1)"
RANDOM_SEED = 42

TAG = "tab"  # used only for an additional tagged output file

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

NUM_COLS = [
    "age",
    "gravidity",
    "parity",
    "cytology_grade",
    "colpo_impression",
    "TZ_type",
    "pathology_fig",
]
CAT_COLS = [
    "menopausal_status",
    "HPV_overall",
    "HPV16",
    "HPV18",
    "HPV_other_hr",
    "iodine_negative",
    "atypical_vessels",
    "child_alive",
]

DROP_ALWAYS = ["patient_id", "patient_name", "pathology_group"]

# CV
N_SPLITS = 5

# Training (CPU)
DEVICE = "cpu"
BATCH_SIZE = 128
EPOCHS = 80
LR = 8e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
EVAL_EVERY = 1

# Model size (CPU)
D_TOKEN = 64
N_HEAD = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.15


# =======================
# 2) UTILS
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

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

def fit_num_stats(X_num):
    med = np.nanmedian(X_num, axis=0)
    X_f = np.where(np.isnan(X_num), med, X_num)
    mean = X_f.mean(axis=0)
    std = X_f.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return med, mean, std

def transform_num(X_num, med, mean, std):
    X_f = np.where(np.isnan(X_num), med, X_num)
    return (X_f - mean) / std

def build_cat_vocab(train_series: pd.Series):
    s = train_series.copy()
    s = s.where(s.notna(), "__NaN__").astype(str)
    uniq = pd.unique(s)
    vocab = {v: (i + 1) for i, v in enumerate(uniq)}  # 0 reserved for UNK
    return vocab

def map_cat(series: pd.Series, vocab: dict):
    s = series.copy()
    s = s.where(s.notna(), "__NaN__").astype(str)
    return s.map(lambda x: vocab.get(x, 0)).astype(np.int64).values


# =======================
# 3) DATA
# =======================
def load_xy_from_xlsx(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    y = pd.to_numeric(df["pathology_group"], errors="coerce")
    y = y.where(y.isin([0, 1]), np.nan)

    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int).values

    for c in DROP_ALWAYS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in XLSX: {missing}")

    X = df[FEATURE_COLS].copy()

    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    for c in CAT_COLS:
        if c not in X.columns:
            raise ValueError(f"Missing CAT col: {c}")

    return X, y


class TabMixDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = torch.from_numpy(X_num).float()
        self.X_cat = torch.from_numpy(X_cat).long()
        self.y = None if y is None else torch.from_numpy(y).float()

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X_num[idx], self.X_cat[idx]
        return (self.X_num[idx], self.X_cat[idx]), self.y[idx]


# =======================
# 4) MODEL
# =======================
class FTTransformer(nn.Module):
    def __init__(self, n_num, cat_cardinalities, d_token=64, n_head=4, n_layers=2, d_ff=128, dropout=0.15):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token

        self.num_linears = nn.ModuleList([nn.Linear(1, d_token) for _ in range(n_num)])
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(int(card) + 1, d_token) for card in cat_cardinalities
        ])

        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))

        seq_len = 1 + n_num + self.n_cat
        self.pos_embed = nn.Embedding(seq_len, d_token)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)
        )

        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, x_num, x_cat):
        B = x_num.size(0)
        cls_tok = self.cls.expand(B, -1, -1)

        num_toks = []
        for i in range(self.n_num):
            v = x_num[:, i:i+1]
            t = self.num_linears[i](v).unsqueeze(1)
            num_toks.append(t)
        num_toks = torch.cat(num_toks, dim=1) if self.n_num > 0 else torch.empty(B, 0, self.d_token, device=x_num.device)

        cat_toks = []
        for j in range(self.n_cat):
            emb = self.cat_embeds[j](x_cat[:, j])
            cat_toks.append(emb.unsqueeze(1))
        cat_toks = torch.cat(cat_toks, dim=1) if self.n_cat > 0 else torch.empty(B, 0, self.d_token, device=x_num.device)

        seq = torch.cat([cls_tok, num_toks, cat_toks], dim=1)

        L = seq.size(1)
        pos_ids = torch.arange(L, device=seq.device).unsqueeze(0).repeat(B, 1)
        seq = seq + self.pos_embed(pos_ids)

        h = self.encoder(seq)
        cls_out = h[:, 0, :]
        logit = self.head(cls_out).squeeze(-1)
        return logit


# =======================
# 5) TRAIN / EVAL
# =======================
@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    ps = []
    for (xb_num, xb_cat), _yb in loader:
        xb_num = xb_num.to(device)
        xb_cat = xb_cat.to(device)
        logit = model(xb_num, xb_cat).cpu().numpy()
        ps.append(sigmoid_np(logit))
    return np.concatenate(ps, axis=0)

def train_one_epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    total = 0.0
    n = 0
    for (xb_num, xb_cat), yb in loader:
        xb_num = xb_num.to(device)
        xb_cat = xb_cat.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logit = model(xb_num, xb_cat)
        loss = loss_fn(logit, yb)
        loss.backward()
        optimizer.step()

        total += loss.item() * xb_num.size(0)
        n += xb_num.size(0)
    return total / max(n, 1)

def build_fold_arrays(X_df, tr_idx, va_idx):
    X_tr = X_df.iloc[tr_idx].copy()
    X_va = X_df.iloc[va_idx].copy()

    X_tr_num = X_tr[NUM_COLS].values.astype(np.float32)
    X_va_num = X_va[NUM_COLS].values.astype(np.float32)

    med, mean, std = fit_num_stats(X_tr_num)
    X_tr_num = transform_num(X_tr_num, med, mean, std).astype(np.float32)
    X_va_num = transform_num(X_va_num, med, mean, std).astype(np.float32)

    vocabs = []
    X_tr_cat_list = []
    X_va_cat_list = []
    for c in CAT_COLS:
        vocab = build_cat_vocab(X_tr[c])
        vocabs.append(vocab)
        X_tr_cat_list.append(map_cat(X_tr[c], vocab))
        X_va_cat_list.append(map_cat(X_va[c], vocab))

    X_tr_cat = np.stack(X_tr_cat_list, axis=1).astype(np.int64)
    X_va_cat = np.stack(X_va_cat_list, axis=1).astype(np.int64)

    cat_cards = [len(v) for v in vocabs]
    return X_tr_num, X_tr_cat, X_va_num, X_va_cat, cat_cards

def run_cv(X_df, y):
    device = torch.device(DEVICE)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    oof_p = np.zeros(len(y), dtype=np.float32)
    oof_fold = np.zeros(len(y), dtype=np.int32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_df, y), start=1):
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        print(f"\n===== Fold {fold}/{N_SPLITS} =====", flush=True)
        print(f"train N={len(tr_idx)} pos={y_tr.mean():.3f} | val N={len(va_idx)} pos={y_va.mean():.3f}", flush=True)

        X_tr_num, X_tr_cat, X_va_num, X_va_cat, cat_cards = build_fold_arrays(X_df, tr_idx, va_idx)

        dl_tr = DataLoader(TabMixDataset(X_tr_num, X_tr_cat, y_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        dl_va = DataLoader(TabMixDataset(X_va_num, X_va_cat, y_va), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        pos = float(y_tr.sum())
        neg = float(len(y_tr) - y_tr.sum())
        pos_weight = torch.tensor([neg / (pos + 1e-12)], dtype=torch.float32, device=device)

        model = FTTransformer(
            n_num=len(NUM_COLS),
            cat_cardinalities=cat_cards,
            d_token=D_TOKEN,
            n_head=N_HEAD,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            dropout=DROPOUT
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc = -1.0
        best_state = None
        bad = 0

        for ep in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(model, dl_tr, device, optimizer, loss_fn)

            if ep % EVAL_EVERY == 0:
                p_va = predict_proba(model, dl_va, device)
                auc = safe_auc(y_va, p_va)
                ap = safe_ap(y_va, p_va)

                print(f"[epoch] {ep:03d}/{EPOCHS} tr_loss={tr_loss:.4f} val_auc={auc:.3f} val_ap={ap:.3f}", flush=True)

                if not np.isnan(auc) and auc > best_auc + 1e-6:
                    best_auc = auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        print(f"[earlystop] no improve for {PATIENCE} evals, stop.", flush=True)
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        p_va = predict_proba(model, dl_va, device)
        auc = safe_auc(y_va, p_va)
        ap = safe_ap(y_va, p_va)
        print(f"[fold] AUC={auc:.3f} AP={ap:.3f} (best_auc_during_train={best_auc:.3f})", flush=True)

        oof_p[va_idx] = p_va.astype(np.float32)
        oof_fold[va_idx] = fold

    # save (backward compatible)
    out_base = os.path.join(BASE_DIR, "tab_oof_predictions.csv")
    out_tag  = os.path.join(BASE_DIR, f"tab_oof_predictions_{TAG}.csv")

    out_df = pd.DataFrame({
        "y": y.astype(int),
        "fold": oof_fold.astype(int),
        "p_tab_raw": oof_p.astype(float),
        "p_tab": oof_p.astype(float),  # legacy name used by 04/06b merging logic
    })
    out_df.to_csv(out_base, index=False, encoding="utf-8-sig")
    out_df.to_csv(out_tag, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_base}", flush=True)
    print(f"Saved: {out_tag}", flush=True)


def main():
    set_seed(RANDOM_SEED)

    X_df, y = load_xy_from_xlsx(XLSX_PATH, SHEET_NAME)

    print(f"Usable N: {len(y)} | Positive rate: {y.mean():.4f}", flush=True)
    print(f"XLSX_PATH: {XLSX_PATH}", flush=True)
    print(f"FEATURE_COLS: {FEATURE_COLS}", flush=True)
    print(f"NUM_COLS: {NUM_COLS}", flush=True)
    print(f"CAT_COLS: {CAT_COLS}", flush=True)

    assert len(NUM_COLS) + len(CAT_COLS) == len(FEATURE_COLS)
    run_cv(X_df, y)


if __name__ == "__main__":
    main()
