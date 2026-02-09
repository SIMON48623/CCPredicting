# 14_tab_mfm_pretrain_and_finetune_cv.py
# (FIXED) TabTransformer + MFM pretrain/finetune with CV
# Key fix: ablation knobs are read from ENV (so external runner can control without editing this file).

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

DEFAULT_XLSX = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
LOCAL_XLSX = os.path.join(BASE_DIR, "data_modelA终.xlsx")
XLSX_PATH = LOCAL_XLSX if os.path.exists(LOCAL_XLSX) else DEFAULT_XLSX

SHEET_NAME = "data_modelA(1)"
RANDOM_SEED = 42

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, None)
    if v is None:
        return bool(default)
    v = str(v).strip().lower()
    return v in ("1", "true", "yes", "y", "t")

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key, None)
    if v is None:
        return float(default)
    return float(v)

def _env_str(key: str, default: str) -> str:
    v = os.getenv(key, None)
    return default if v is None else str(v)

# -------- Ablation knobs (controlled by ENV, no manual edits needed) --------
# Runner will set:
#   TABMFM_TAG, TABMFM_DO_PRETRAIN, TABMFM_MASK_RATIO, TABMFM_USE_COL_ID_EMB
TAG = _env_str("TABMFM_TAG", "mfm_default")
DO_PRETRAIN = _env_bool("TABMFM_DO_PRETRAIN", True)
MASK_RATIO = _env_float("TABMFM_MASK_RATIO", 0.30)
USE_COL_ID_EMB = _env_bool("TABMFM_USE_COL_ID_EMB", True)
# ---------------------------------------------------------------------------

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

DROP_ALWAYS = ["patient_id", "patient_name", "pathology_group"]

NUM_COLS = ["age", "gravidity", "parity", "cytology_grade", "colpo_impression", "TZ_type", "pathology_fig"]
CAT_COLS = ["menopausal_status", "HPV_overall", "HPV16", "HPV18", "HPV_other_hr",
            "iodine_negative", "atypical_vessels", "child_alive"]
assert set(NUM_COLS + CAT_COLS) == set(FEATURE_COLS)

# CV / device
N_SPLITS = 5
BATCH_SIZE = 128
DEVICE = "cpu"

# Transformer (CPU-friendly)
D_MODEL = 64
N_HEAD  = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.15

# Pretrain (MFM)
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-4
PRETRAIN_PRINT_EVERY = 120

# Finetune (supervised)
FINETUNE_EPOCHS = 60
FINETUNE_LR = 8e-4
FINETUNE_WEIGHT_DECAY = 1e-4
EARLYSTOP_PATIENCE = 8
FINETUNE_PRINT_EVERY = 120

# Outputs
OUT_BASE = os.path.join(BASE_DIR, "tab_mfm_oof_predictions.csv")
OUT_TAG  = os.path.join(BASE_DIR, f"tab_mfm_oof_predictions_{TAG}.csv")


# =======================
# 2) Utils
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def to_numeric_df(df, cols):
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def fit_num_imputer_scaler(X_num):
    med = np.nanmedian(X_num, axis=0)
    X_filled = np.where(np.isnan(X_num), med, X_num)
    mean = X_filled.mean(axis=0)
    std = X_filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return med, mean, std

def transform_num(X_num, med, mean, std):
    X_filled = np.where(np.isnan(X_num), med, X_num)
    return (X_filled - mean) / std

def fit_cat_mappings(df_cat_train):
    maps = []
    cards = []
    for c in df_cat_train.columns:
        s = df_cat_train[c]
        uniq = pd.Series(s.dropna().unique())
        uniq_keys = uniq.astype(str).tolist()
        m = {k: i + 1 for i, k in enumerate(uniq_keys)}  # 0=UNK
        card = len(m) + 2  # +UNK(0) +MASK(last)
        maps.append(m)
        cards.append(card)
    return maps, cards

def transform_cat(df_cat, maps):
    X = np.zeros((len(df_cat), df_cat.shape[1]), dtype=np.int64)
    for j, c in enumerate(df_cat.columns):
        s = df_cat[c]
        m = maps[j]
        keys = s.astype(str).values
        idx = np.array([m.get(k, 0) for k in keys], dtype=np.int64)
        X[:, j] = idx
    return X


# =======================
# 3) Data builder
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
        raise ValueError(f"Missing feature columns: {missing}")

    return df[FEATURE_COLS].copy(), y


class TabPackDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = torch.from_numpy(X_num).float()
        self.X_cat = torch.from_numpy(X_cat).long()
        self.y = None if y is None else torch.from_numpy(y).float()

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X_num[idx], self.X_cat[idx]
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


# =======================
# 4) Model
# =======================
class TabTokTransformer(nn.Module):
    def __init__(self, n_num, cat_cards, d_model=64, n_head=4, n_layers=2, d_ff=128, dropout=0.15, use_col_id_emb=True):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cards)
        self.n_features = n_num + self.n_cat
        self.d_model = d_model
        self.use_col_id_emb = bool(use_col_id_emb)

        self.num_value_proj = nn.Linear(1, d_model)
        self.cat_embs = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cards])

        self.col_id_emb = nn.Embedding(self.n_features, d_model) if self.use_col_id_emb else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self.num_recon = nn.Linear(d_model, 1)
        self.cat_recon = nn.ModuleList([nn.Linear(d_model, card) for card in cat_cards])

    def forward_tokens(self, x_num, x_cat):
        B = x_num.size(0)
        num_tok = self.num_value_proj(x_num.unsqueeze(-1))

        cat_toks = [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embs)]
        cat_tok = torch.stack(cat_toks, dim=1) if self.n_cat > 0 else torch.empty(B, 0, self.d_model, device=x_num.device)

        tok = torch.cat([num_tok, cat_tok], dim=1) if self.n_cat > 0 else num_tok

        if self.col_id_emb is not None:
            fids = torch.arange(self.n_features, device=tok.device).unsqueeze(0).repeat(B, 1)
            tok = tok + self.col_id_emb(fids)
        return tok

    def encode(self, x_num, x_cat):
        tok = self.forward_tokens(x_num, x_cat)
        return self.encoder(tok)

    def classify(self, x_num, x_cat):
        h = self.encode(x_num, x_cat)
        pooled = h.mean(dim=1)
        return self.cls_head(pooled).squeeze(-1)

    def recon_num(self, h_num):
        return self.num_recon(h_num).squeeze(-1)

    def recon_cat(self, h_cat):
        return [head(h_cat[:, j, :]) for j, head in enumerate(self.cat_recon)]


# =======================
# 5) MFM masking
# =======================
def apply_mfm_mask(x_num, x_cat, cat_cards, mask_ratio=0.30):
    B, n_num = x_num.shape
    _, n_cat = x_cat.shape
    F = n_num + n_cat
    mask = (torch.rand(B, F, device=x_num.device) < float(mask_ratio))

    x_num_m = x_num.clone()
    x_cat_m = x_cat.clone()

    if n_num > 0:
        m_num = mask[:, :n_num]
        x_num_m[m_num] = 0.0

    if n_cat > 0:
        m_cat = mask[:, n_num:]
        for j in range(n_cat):
            mask_idx = cat_cards[j] - 1
            x_cat_m[m_cat[:, j], j] = mask_idx

    return x_num_m, x_cat_m, mask


# =======================
# 6) Train loops
# =======================
@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    ps = []
    for xb_num, xb_cat, *_ in loader:
        xb_num = xb_num.to(device)
        xb_cat = xb_cat.to(device)
        logit = model.classify(xb_num, xb_cat).cpu().numpy()
        ps.append(sigmoid_np(logit))
    return np.concatenate(ps, axis=0)

def pretrain_mfm(model, loader, device, cat_cards, epochs=20, lr=1e-3, weight_decay=1e-4, mask_ratio=0.30):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss(reduction="none")
    ce  = nn.CrossEntropyLoss(reduction="none")

    for ep in range(epochs):
        running = 0.0
        denom = 0.0
        for i, (xb_num, xb_cat, *_ ) in enumerate(loader):
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)

            xb_num_m, xb_cat_m, mask = apply_mfm_mask(xb_num, xb_cat, cat_cards, mask_ratio=mask_ratio)
            h = model.encode(xb_num_m, xb_cat_m)
            n_num = model.n_num
            n_cat = model.n_cat

            loss = 0.0

            if n_num > 0:
                h_num = h[:, :n_num, :]
                pred_num = model.recon_num(h_num)
                m_num = mask[:, :n_num]
                if m_num.any():
                    l_num = mse(pred_num, xb_num)
                    loss = loss + l_num[m_num].mean()

            if n_cat > 0:
                h_cat = h[:, n_num:, :]
                logits_list = model.recon_cat(h_cat)
                m_cat = mask[:, n_num:]
                terms = []
                for j in range(n_cat):
                    mj = m_cat[:, j]
                    if mj.any():
                        logits = logits_list[j]
                        target = xb_cat[:, j]
                        l = ce(logits, target)
                        terms.append(l[mj].mean())
                if terms:
                    loss = loss + torch.stack(terms).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            denom += 1.0

            if (i % PRETRAIN_PRINT_EVERY) == 0:
                print(f"[pretrain] tag={TAG} ep={ep:02d} batch={i:04d} loss={loss.item():.4f}", flush=True)

        print(f"[pretrain] tag={TAG} ep={ep:02d}/{epochs} mean_loss={(running/max(denom,1.0)):.4f}", flush=True)

def finetune_supervised(model, dl_tr, dl_va, device, epochs=60, lr=8e-4, weight_decay=1e-4, patience=8):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_tr_all = []
    for _, _, yb in dl_tr:
        y_tr_all.append(yb.numpy())
    y_tr_all = np.concatenate(y_tr_all, axis=0).astype(int)
    pos = y_tr_all.sum()
    neg = len(y_tr_all) - pos
    pos_weight = torch.tensor([neg / (pos + 1e-12)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        running = 0.0
        denom = 0.0

        for i, (xb_num, xb_cat, yb) in enumerate(dl_tr):
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)

            logit = model.classify(xb_num, xb_cat)
            loss = loss_fn(logit, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            denom += 1.0

            if (i % FINETUNE_PRINT_EVERY) == 0:
                print(f"[finetune] tag={TAG} ep={ep:02d} batch={i:04d} loss={loss.item():.4f}", flush=True)

        p_va = predict_proba(model, dl_va, device)
        y_va = []
        for _, _, yb in dl_va:
            y_va.append(yb.numpy())
        y_va = np.concatenate(y_va, axis=0).astype(int)

        try:
            auc = roc_auc_score(y_va, p_va)
        except Exception:
            auc = float("nan")

        mean_loss = running / max(denom, 1.0)
        print(f"[epoch] tag={TAG} {ep:02d}/{epochs} tr_loss={mean_loss:.4f} val_auc={auc:.3f}", flush=True)

        if not np.isnan(auc) and auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[earlystop] tag={TAG} no improve for {patience} evals, stop.", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc


# =======================
# 7) CV runner
# =======================
def run_cv(df_all, y_all):
    device = torch.device(DEVICE)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    oof_p = np.zeros(len(y_all), dtype=np.float32)
    oof_fold = np.zeros(len(y_all), dtype=np.int32)

    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df_all, y_all), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} | tag={TAG} =====", flush=True)

        df_tr = df_all.iloc[tr_idx].copy()
        df_va = df_all.iloc[va_idx].copy()
        y_tr = y_all[tr_idx].astype(int)
        y_va = y_all[va_idx].astype(int)

        df_tr_num = to_numeric_df(df_tr, NUM_COLS)
        df_va_num = to_numeric_df(df_va, NUM_COLS)

        X_tr_num_raw = df_tr_num.values.astype(np.float32)
        X_va_num_raw = df_va_num.values.astype(np.float32)

        med, mean, std = fit_num_imputer_scaler(X_tr_num_raw)
        X_tr_num = transform_num(X_tr_num_raw, med, mean, std).astype(np.float32)
        X_va_num = transform_num(X_va_num_raw, med, mean, std).astype(np.float32)

        df_tr_cat = to_numeric_df(df_tr, CAT_COLS)
        df_va_cat = to_numeric_df(df_va, CAT_COLS)

        cat_maps, cat_cards = fit_cat_mappings(df_tr_cat)
        X_tr_cat = transform_cat(df_tr_cat, cat_maps)
        X_va_cat = transform_cat(df_va_cat, cat_maps)

        dl_tr = DataLoader(TabPackDataset(X_tr_num, X_tr_cat, y_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        dl_va = DataLoader(TabPackDataset(X_va_num, X_va_cat, y_va), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model = TabTokTransformer(
            n_num=len(NUM_COLS),
            cat_cards=cat_cards,
            d_model=D_MODEL,
            n_head=N_HEAD,
            n_layers=N_LAYERS,
            d_ff=D_FF,
            dropout=DROPOUT,
            use_col_id_emb=USE_COL_ID_EMB
        ).to(device)

        if DO_PRETRAIN:
            print(f"[stage] MFM pretraining ... (mask_ratio={MASK_RATIO}, use_col_id_emb={USE_COL_ID_EMB})", flush=True)
            dl_pre = DataLoader(TabPackDataset(X_tr_num, X_tr_cat, y_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
            pretrain_mfm(
                model=model,
                loader=dl_pre,
                device=device,
                cat_cards=cat_cards,
                epochs=PRETRAIN_EPOCHS,
                lr=PRETRAIN_LR,
                weight_decay=PRETRAIN_WEIGHT_DECAY,
                mask_ratio=MASK_RATIO
            )
        else:
            print("[stage] NO pretraining (ablation).", flush=True)

        print("[stage] Finetuning for classification ...", flush=True)
        best_auc = finetune_supervised(
            model=model,
            dl_tr=dl_tr,
            dl_va=dl_va,
            device=device,
            epochs=FINETUNE_EPOCHS,
            lr=FINETUNE_LR,
            weight_decay=FINETUNE_WEIGHT_DECAY,
            patience=EARLYSTOP_PATIENCE
        )

        p_va = predict_proba(model, dl_va, device)
        auc = roc_auc_score(y_va, p_va)
        ap  = average_precision_score(y_va, p_va)

        print(f"[fold] tag={TAG} AUC={auc:.3f} AP={ap:.3f} (best_auc_during_train={best_auc:.3f})", flush=True)

        oof_p[va_idx] = p_va.astype(np.float32)
        oof_fold[va_idx] = fold
        fold_metrics.append((fold, auc, ap))

    fold_arr = np.array(fold_metrics, dtype=float)
    print("\n===== CV summary (mean ± std) =====", flush=True)
    print(f"AUC : {fold_arr[:, 1].mean():.3f} ± {fold_arr[:, 1].std():.3f}", flush=True)
    print(f"AP  : {fold_arr[:, 2].mean():.3f} ± {fold_arr[:, 2].std():.3f}", flush=True)

    out_df = pd.DataFrame({
        "y": y_all.astype(int),
        "fold": oof_fold.astype(int),
        "p_tab_mfm_raw": oof_p.astype(float),
        "tag": TAG,
        "do_pretrain": int(DO_PRETRAIN),
        "mask_ratio": float(MASK_RATIO),
        "use_col_id_emb": int(USE_COL_ID_EMB),
    })

    out_df.to_csv(OUT_BASE, index=False, encoding="utf-8-sig")
    out_df.to_csv(OUT_TAG, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_BASE}", flush=True)
    print(f"Saved: {OUT_TAG}", flush=True)


def main():
    set_seed(RANDOM_SEED)
    df_feat, y = load_xy_from_xlsx(XLSX_PATH, SHEET_NAME)

    print(f"Usable N: {len(y)} | Positive rate: {y.mean():.4f}", flush=True)
    print(f"XLSX_PATH: {XLSX_PATH}", flush=True)
    print(f"TAG={TAG} | DO_PRETRAIN={DO_PRETRAIN} | MASK_RATIO={MASK_RATIO} | USE_COL_ID_EMB={USE_COL_ID_EMB}", flush=True)

    run_cv(df_feat, y)


if __name__ == "__main__":
    main()
