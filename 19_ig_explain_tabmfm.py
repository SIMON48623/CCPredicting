# 19_ig_explain_tabmfm.py
import os, json, random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.join(BASE_DIR, "final_model")

DEFAULT_XLSX = r"C:\Users\COMPUTER\Desktop\CCPredicting\cervix_modelA\data_modelA终.xlsx"
LOCAL_XLSX = os.path.join(BASE_DIR, "data_modelA终.xlsx")
XLSX_PATH = LOCAL_XLSX if os.path.exists(LOCAL_XLSX) else DEFAULT_XLSX
SHEET_NAME = "data_modelA(1)"

# --------- IG config ----------
N_STEPS = 64          # IG steps
GLOBAL_N = 200        # sample size for global importance (increase if you want)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Model definition (must match export) ----------
class TabTokTransformer(nn.Module):
    def __init__(self, n_num, cat_cards, d_model=64, n_head=4, n_layers=2, d_ff=128, dropout=0.15, use_col_id_emb=True):
        super().__init__()
        self.n_num = int(n_num)
        self.n_cat = int(len(cat_cards))
        self.n_features = self.n_num + self.n_cat
        self.d_model = int(d_model)
        self.use_col_id_emb = bool(use_col_id_emb)

        self.num_value_proj = nn.Linear(1, self.d_model)
        self.cat_embs = nn.ModuleList([nn.Embedding(int(card), self.d_model) for card in cat_cards])
        self.col_id_emb = nn.Embedding(self.n_features, self.d_model) if self.use_col_id_emb else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_head),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1)
        )

        # recon heads exist in weights; not used in IG
        self.num_recon = nn.Linear(self.d_model, 1)
        self.cat_recon = nn.ModuleList([nn.Linear(self.d_model, int(card)) for card in cat_cards])

    def forward_tokens(self, x_num, x_cat):
        B = x_num.size(0)
        num_tok = self.num_value_proj(x_num.unsqueeze(-1))  # (B, n_num, d)
        if self.n_cat > 0:
            cat_toks = [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embs)]
            cat_tok = torch.stack(cat_toks, dim=1)          # (B, n_cat, d)
            tok = torch.cat([num_tok, cat_tok], dim=1)      # (B, F, d)
        else:
            tok = num_tok
        if self.col_id_emb is not None:
            fids = torch.arange(self.n_features, device=tok.device).unsqueeze(0).repeat(B, 1)
            tok = tok + self.col_id_emb(fids)
        return tok

    def classify_logits(self, x_num, x_cat):
        h = self.encoder(self.forward_tokens(x_num, x_cat))
        pooled = h.mean(dim=1)
        return self.cls_head(pooled).squeeze(-1)

# --------- Load artifacts ----------
def load_predict_stack():
    preprocess = joblib.load(os.path.join(FINAL_DIR, "preprocess.pkl"))
    arch = json.load(open(os.path.join(FINAL_DIR, "arch.json"), "r", encoding="utf-8"))
    model = TabTokTransformer(
        n_num=int(arch["n_num"]),
        cat_cards=arch["cat_cards"],
        d_model=int(arch["d_model"]),
        n_head=int(arch["n_head"]),
        n_layers=int(arch["n_layers"]),
        d_ff=int(arch["d_ff"]),
        dropout=float(arch["dropout"]),
        use_col_id_emb=bool(arch["use_col_id_emb"]),
    ).to(DEVICE)
    state = torch.load(os.path.join(FINAL_DIR, "transformer_state.pt"), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return preprocess, model

def vectorize(preprocess, df_feat: pd.DataFrame):
    feat = preprocess["feature_cols"]
    num_cols = preprocess["num_cols"]
    cat_cols = preprocess["cat_cols"]

    df = df_feat[feat].copy()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X_num = df[num_cols].values.astype(np.float32)
    med = np.asarray(preprocess["num_med"], dtype=np.float32)
    mean = np.asarray(preprocess["num_mean"], dtype=np.float32)
    std = np.asarray(preprocess["num_std"], dtype=np.float32)
    X_num = np.where(np.isnan(X_num), med, X_num)
    X_num = (X_num - mean) / std

    # cat mapping
    cat_maps = preprocess["cat_maps"]
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, c in enumerate(cat_cols):
        keys = df[c].astype("Int64").astype(str).values
        X_cat[:, j] = np.array([cat_maps[j].get(k, 0) for k in keys], dtype=np.int64)

    return X_num, X_cat

def load_full_xlsx():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
    y = pd.to_numeric(df["pathology_group"], errors="coerce")
    y = y.where(y.isin([0,1]), np.nan)
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int).values
    return df.reset_index(drop=True), y

# --------- Integrated Gradients on token embeddings ----------
@torch.no_grad()
def tokens_from_input(model, x_num, x_cat):
    return model.forward_tokens(x_num, x_cat)

def ig_on_tokens(model, tok_in, tok_base, steps=64):
    """
    IG for a single sample tokens: tok_in/tok_base shape (1, F, d)
    Returns attributions per token (1, F)
    """
    assert tok_in.shape == tok_base.shape
    grads_acc = torch.zeros_like(tok_in)

    for i in range(1, steps+1):
        alpha = i / steps
        tok = tok_base + alpha * (tok_in - tok_base)
        tok.requires_grad_(True)

        # forward from tokens through encoder+head (replicate classify_logits but with tokens input)
        h = model.encoder(tok)
        pooled = h.mean(dim=1)
        logit = model.cls_head(pooled).squeeze(-1)  # (1,)

        model.zero_grad(set_to_none=True)
        logit.backward()
        grads_acc += tok.grad.detach()

    avg_grads = grads_acc / steps
    ig = (tok_in - tok_base) * avg_grads  # (1, F, d)
    # reduce embedding dim to get token score
    token_attr = ig.abs().sum(dim=-1)     # (1, F)
    return token_attr

def explain_one(preprocess, model, record: dict):
    # build 1-row df
    df = pd.DataFrame([record])
    # ensure all cols exist
    for c in preprocess["feature_cols"]:
        if c not in df.columns:
            df[c] = np.nan
    X_num, X_cat = vectorize(preprocess, df)

    x_num_t = torch.tensor(X_num, dtype=torch.float32, device=DEVICE)
    x_cat_t = torch.tensor(X_cat, dtype=torch.long, device=DEVICE)

    # baseline: numeric zeros, categorical UNK (0)
    base_num = torch.zeros_like(x_num_t)
    base_cat = torch.zeros_like(x_cat_t)

    tok_in = tokens_from_input(model, x_num_t, x_cat_t)
    tok_base = tokens_from_input(model, base_num, base_cat)

    token_attr = ig_on_tokens(model, tok_in, tok_base, steps=N_STEPS).detach().cpu().numpy().reshape(-1)

    feature_names = preprocess["num_cols"] + preprocess["cat_cols"]
    out = pd.DataFrame({"feature": feature_names, "ig_abs": token_attr})
    out["ig_norm"] = out["ig_abs"] / (out["ig_abs"].sum() + 1e-12)
    out = out.sort_values("ig_norm", ascending=False).reset_index(drop=True)
    return out

def global_importance(preprocess, model, df_all: pd.DataFrame):
    random.seed(SEED)
    idx = list(range(len(df_all)))
    random.shuffle(idx)
    idx = idx[:min(GLOBAL_N, len(idx))]

    feat = preprocess["feature_cols"]
    df_s = df_all.iloc[idx].copy()
    # fill missing columns if any
    for c in feat:
        if c not in df_s.columns:
            df_s[c] = np.nan

    X_num, X_cat = vectorize(preprocess, df_s)
    x_num_t = torch.tensor(X_num, dtype=torch.float32, device=DEVICE)
    x_cat_t = torch.tensor(X_cat, dtype=torch.long, device=DEVICE)

    # baseline batch
    base_num = torch.zeros_like(x_num_t)
    base_cat = torch.zeros_like(x_cat_t)

    # compute IG per sample (loop for stability)
    feature_names = preprocess["num_cols"] + preprocess["cat_cols"]
    agg = np.zeros(len(feature_names), dtype=np.float64)

    for i in range(len(df_s)):
        tok_in = tokens_from_input(model, x_num_t[i:i+1], x_cat_t[i:i+1])
        tok_base = tokens_from_input(model, base_num[i:i+1], base_cat[i:i+1])
        token_attr = ig_on_tokens(model, tok_in, tok_base, steps=N_STEPS).detach().cpu().numpy().reshape(-1)
        agg += token_attr

    agg = agg / max(len(df_s), 1)
    out = pd.DataFrame({"feature": feature_names, "mean_ig_abs": agg})
    out["mean_ig_norm"] = out["mean_ig_abs"] / (out["mean_ig_abs"].sum() + 1e-12)
    out = out.sort_values("mean_ig_norm", ascending=False).reset_index(drop=True)
    return out

def main():
    preprocess, model = load_predict_stack()
    df_all, y = load_full_xlsx()

    # --- demo record (same as your test) ---
    demo = {
        "age":45,"menopausal_status":0,"gravidity":2,"parity":1,"HPV_overall":1,"HPV16":0,"HPV18":0,"HPV_other_hr":1,
        "cytology_grade":3,"colpo_impression":2,"TZ_type":2,"iodine_negative":1,"atypical_vessels":0,"child_alive":1,"pathology_fig":2
    }

    one = explain_one(preprocess, model, demo)
    one_path = os.path.join(BASE_DIR, "ig_one_demo.csv")
    one.to_csv(one_path, index=False, encoding="utf-8-sig")
    print("Saved:", one_path)
    print(one.head(10))

    glob = global_importance(preprocess, model, df_all)
    glob_path = os.path.join(BASE_DIR, "ig_global_top.csv")
    glob.to_csv(glob_path, index=False, encoding="utf-8-sig")
    print("Saved:", glob_path)
    print(glob.head(15))

if __name__ == "__main__":
    main()
