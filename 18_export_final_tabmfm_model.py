# 18_export_final_tabmfm_model.py
# (FIXED) Export final tab-mfm model as state_dict (portable) + preprocess + calibrator + metadata

import os, json, random
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.join(BASE_DIR, "final_model")
os.makedirs(FINAL_DIR, exist_ok=True)

DEFAULT_XLSX = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
LOCAL_XLSX = os.path.join(BASE_DIR, "data_modelA终.xlsx")
XLSX_PATH = LOCAL_XLSX if os.path.exists(LOCAL_XLSX) else DEFAULT_XLSX
SHEET_NAME = "data_modelA(1)"

FEATURE_COLS = [
    "age","menopausal_status","gravidity","parity","HPV_overall","HPV16","HPV18","HPV_other_hr",
    "cytology_grade","colpo_impression","TZ_type","iodine_negative","atypical_vessels","child_alive","pathology_fig",
]
NUM_COLS = ["age","gravidity","parity","cytology_grade","colpo_impression","TZ_type","pathology_fig"]
CAT_COLS = ["menopausal_status","HPV_overall","HPV16","HPV18","HPV_other_hr","iodine_negative","atypical_vessels","child_alive"]

DEVICE = "cpu"
BATCH_SIZE = 128

# architecture must match training scripts
D_MODEL = 64
N_HEAD  = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.15

DO_PRETRAIN = True
MASK_RATIO = 0.15
USE_COL_ID_EMB = True

PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 1e-3
PRETRAIN_WD = 1e-4

FINETUNE_EPOCHS = 80
FINETUNE_LR = 8e-4
FINETUNE_WD = 1e-4

# thresholds (from your txt)
THR_SCREEN = 0.12
THR_TRIAGE = 0.28
THR_YOUDEN = 0.33

SEED = 42

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def sigmoid_np(x):
    return 1/(1+np.exp(-x))

def load_xy():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
    y = pd.to_numeric(df["pathology_group"], errors="coerce")
    y = y.where(y.isin([0,1]), np.nan)
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int).values
    X = df[FEATURE_COLS].copy()
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.reset_index(drop=True), y

def fit_num_stats(X_num):
    med = np.nanmedian(X_num, axis=0)
    Xf = np.where(np.isnan(X_num), med, X_num)
    mean = Xf.mean(axis=0)
    std = Xf.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return med, mean, std

def transform_num(X_num, med, mean, std):
    Xf = np.where(np.isnan(X_num), med, X_num)
    return (Xf - mean) / std

def fit_cat_maps(df_cat):
    maps = []
    cards = []
    for c in df_cat.columns:
        s = df_cat[c]
        uniq = pd.Series(s.dropna().astype(int).unique())
        keys = uniq.astype(str).tolist()
        m = {k: i+1 for i,k in enumerate(keys)}  # 0=UNK
        card = len(m) + 2  # +UNK +MASK
        maps.append(m); cards.append(card)
    return maps, cards

def transform_cat(df_cat, maps):
    X = np.zeros((len(df_cat), df_cat.shape[1]), dtype=np.int64)
    for j,c in enumerate(df_cat.columns):
        m = maps[j]
        keys = df_cat[c].astype("Int64").astype(str).values
        X[:, j] = np.array([m.get(k, 0) for k in keys], dtype=np.int64)
    return X

class TabPackDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = torch.from_numpy(X_num).float()
        self.X_cat = torch.from_numpy(X_cat).long()
        self.y = None if y is None else torch.from_numpy(y).float()
    def __len__(self): return self.X_num.shape[0]
    def __getitem__(self, i):
        if self.y is None: return self.X_num[i], self.X_cat[i]
        return self.X_num[i], self.X_cat[i], self.y[i]

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
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls_head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

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
        return self.encoder(self.forward_tokens(x_num, x_cat))

    def classify_logits(self, x_num, x_cat):
        h = self.encode(x_num, x_cat)
        pooled = h.mean(dim=1)
        return self.cls_head(pooled).squeeze(-1)

    def recon_num(self, h_num):
        return self.num_recon(h_num).squeeze(-1)

    def recon_cat(self, h_cat):
        return [head(h_cat[:, j, :]) for j, head in enumerate(self.cat_recon)]

def apply_mfm_mask(x_num, x_cat, cat_cards, mask_ratio):
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

def pretrain_mfm(model, loader, device, cat_cards, epochs, lr, wd, mask_ratio):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss(reduction="none")
    ce  = nn.CrossEntropyLoss(reduction="none")
    for ep in range(epochs):
        losses = []
        for xb_num, xb_cat, *_ in loader:
            xb_num = xb_num.to(device); xb_cat = xb_cat.to(device)
            xb_num_m, xb_cat_m, mask = apply_mfm_mask(xb_num, xb_cat, cat_cards, mask_ratio)
            h = model.encode(xb_num_m, xb_cat_m)
            n_num = model.n_num; n_cat = model.n_cat
            loss = 0.0
            if n_num > 0:
                pred_num = model.recon_num(h[:, :n_num, :])
                m_num = mask[:, :n_num]
                if m_num.any():
                    l = mse(pred_num, xb_num)
                    loss = loss + l[m_num].mean()
            if n_cat > 0:
                logits_list = model.recon_cat(h[:, n_num:, :])
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
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        print(f"[pretrain] ep={ep+1:02d}/{epochs} loss={np.mean(losses):.4f}", flush=True)

def finetune(model, loader, device, epochs, lr, wd):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    y_all = []
    for _, _, yb in loader:
        y_all.append(yb.numpy())
    y_all = np.concatenate(y_all).astype(int)
    pos = y_all.sum(); neg = len(y_all) - pos
    pos_weight = torch.tensor([neg/(pos+1e-12)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(epochs):
        losses = []
        for xb_num, xb_cat, yb in loader:
            xb_num = xb_num.to(device); xb_cat = xb_cat.to(device); yb = yb.to(device)
            logit = model.classify_logits(xb_num, xb_cat)
            loss = loss_fn(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"[finetune] ep={ep+1:03d}/{epochs} loss={np.mean(losses):.4f}", flush=True)

@torch.no_grad()
def predict_raw_prob(model, loader, device):
    model.eval()
    ps = []
    for xb_num, xb_cat in loader:
        xb_num = xb_num.to(device); xb_cat = xb_cat.to(device)
        logit = model.classify_logits(xb_num, xb_cat).cpu().numpy()
        ps.append(sigmoid_np(logit))
    return np.concatenate(ps, axis=0)

def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

    X_df, y = load_xy()

    X_num_raw = X_df[NUM_COLS].values.astype(np.float32)
    med, mean, std = fit_num_stats(X_num_raw)
    X_num = transform_num(X_num_raw, med, mean, std).astype(np.float32)

    cat_maps, cat_cards = fit_cat_maps(X_df[CAT_COLS])
    X_cat = transform_cat(X_df[CAT_COLS], cat_maps)

    dl_train = DataLoader(TabPackDataset(X_num, X_cat, y), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model = TabTokTransformer(
        n_num=len(NUM_COLS), cat_cards=cat_cards,
        d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT,
        use_col_id_emb=USE_COL_ID_EMB
    ).to(device)

    if DO_PRETRAIN:
        print(f"[stage] pretrain: mask_ratio={MASK_RATIO}, use_col_id_emb={USE_COL_ID_EMB}", flush=True)
        pretrain_mfm(model, dl_train, device, cat_cards, PRETRAIN_EPOCHS, PRETRAIN_LR, PRETRAIN_WD, MASK_RATIO)
    else:
        print("[stage] no pretrain", flush=True)

    print("[stage] finetune on full data", flush=True)
    finetune(model, dl_train, device, FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_WD)

    # Fit Platt calibrator on FULL-DATA raw probs (prototype); stable enough for deployment prototype
    dl_inf = DataLoader(TabPackDataset(X_num, X_cat, None), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    p_raw = predict_raw_prob(model, dl_inf, device)
    cal = LogisticRegression(solver="lbfgs", max_iter=2000)
    cal.fit(p_raw.reshape(-1,1), y)

    # Save portable artifacts
    torch.save(model.state_dict(), os.path.join(FINAL_DIR, "transformer_state.pt"))

    arch = {
        "d_model": D_MODEL,
        "n_head": N_HEAD,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "use_col_id_emb": USE_COL_ID_EMB,
        "cat_cards": cat_cards,
        "n_num": len(NUM_COLS),
    }
    json.dump(arch, open(os.path.join(FINAL_DIR, "arch.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    preprocess = {
        "feature_cols": FEATURE_COLS,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "num_med": med, "num_mean": mean, "num_std": std,
        "cat_maps": cat_maps,
        "cat_cards": cat_cards
    }
    joblib.dump(preprocess, os.path.join(FINAL_DIR, "preprocess.pkl"))
    joblib.dump(cal, os.path.join(FINAL_DIR, "calibrator.pkl"))

    meta = {
        "model_name": "tab_mfm_transformer",
        "calibration": "platt_sigmoid",
        "thresholds": {"screen": THR_SCREEN, "triage": THR_TRIAGE, "youden": THR_YOUDEN},
        "train": {"do_pretrain": DO_PRETRAIN, "mask_ratio": MASK_RATIO, "use_col_id_emb": USE_COL_ID_EMB}
    }
    json.dump(meta, open(os.path.join(FINAL_DIR, "metadata.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("Saved final_model package to:", FINAL_DIR)
    print(" - transformer_state.pt")
    print(" - arch.json")
    print(" - preprocess.pkl")
    print(" - calibrator.pkl")
    print(" - metadata.json")

if __name__ == "__main__":
    main()
