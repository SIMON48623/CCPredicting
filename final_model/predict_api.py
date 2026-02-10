# final_model/predict_api.py
import os, json
import numpy as np
import joblib
import torch
import torch.nn as nn


class TabTokTransformer(nn.Module):
    """
    MUST match the exported architecture in 18_export_final_tabmfm_model.py,
    including recon heads (num_recon/cat_recon), otherwise state_dict load will fail.
    """
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

        # ---- recon heads (required for loading state_dict) ----
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


class CervixRiskPredictor:
    # IG steps kept modest for web latency
    IG_STEPS = 24
    IG_TOPK = 10

    def __init__(self, model_dir=None, device=None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir or base
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.preprocess = joblib.load(os.path.join(self.model_dir, "preprocess.pkl"))
        self.calibrator = joblib.load(os.path.join(self.model_dir, "calibrator.pkl"))
        self.meta = json.load(open(os.path.join(self.model_dir, "metadata.json"), "r", encoding="utf-8"))
        arch = json.load(open(os.path.join(self.model_dir, "arch.json"), "r", encoding="utf-8"))

        self.model = TabTokTransformer(
            n_num=int(arch["n_num"]),
            cat_cards=arch["cat_cards"],
            d_model=int(arch["d_model"]),
            n_head=int(arch["n_head"]),
            n_layers=int(arch["n_layers"]),
            d_ff=int(arch["d_ff"]),
            dropout=float(arch["dropout"]),
            use_col_id_emb=bool(arch["use_col_id_emb"]),
        ).to(self.device)

        state_path = os.path.join(self.model_dir, "transformer_state.pt")
        state = torch.load(state_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.num_cols = list(self.preprocess["num_cols"])
        self.cat_cols = list(self.preprocess["cat_cols"])
        self.feature_names = self.num_cols + self.cat_cols

    def _vectorize_one(self, record: dict):
        feat = self.preprocess["feature_cols"]
        num_cols = self.preprocess["num_cols"]
        cat_cols = self.preprocess["cat_cols"]

        row = {c: record.get(c, None) for c in feat}

        # numeric
        x_num = np.array([row.get(c, np.nan) for c in num_cols], dtype=np.float32)
        med = np.asarray(self.preprocess["num_med"], dtype=np.float32)
        mean = np.asarray(self.preprocess["num_mean"], dtype=np.float32)
        std = np.asarray(self.preprocess["num_std"], dtype=np.float32)
        x_num = np.where(np.isnan(x_num), med, x_num)
        x_num = (x_num - mean) / std

        # categorical
        cat_maps = self.preprocess["cat_maps"]
        x_cat = []
        for j, c in enumerate(cat_cols):
            v = row.get(c, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                key = "nan"
            else:
                try:
                    key = str(int(v))
                except Exception:
                    key = str(v)
            idx = cat_maps[j].get(key, 0)
            x_cat.append(idx)
        x_cat = np.array(x_cat, dtype=np.int64)

        return x_num[None, :], x_cat[None, :]

    def _logit_from_tokens(self, tok):
        h = self.model.encoder(tok)
        pooled = h.mean(dim=1)
        return self.model.cls_head(pooled).squeeze(-1)  # (B,)

    def _ig_single(self, x_num_t, x_cat_t, steps=24):
        """
        Integrated Gradients on token embeddings, single case.
        Returns token attribution per feature (signed + abs).
        """
        # baseline: zeros (num) + UNK=0 (cat)
        base_num = torch.zeros_like(x_num_t)
        base_cat = torch.zeros_like(x_cat_t)

        with torch.no_grad():
            tok_in = self.model.forward_tokens(x_num_t, x_cat_t)      # (1, F, d)
            tok_base = self.model.forward_tokens(base_num, base_cat)  # (1, F, d)

        grads_acc = torch.zeros_like(tok_in)

        for i in range(1, steps + 1):
            alpha = i / steps
            tok = tok_base + alpha * (tok_in - tok_base)
            tok.requires_grad_(True)

            logit = self._logit_from_tokens(tok)  # (1,)
            self.model.zero_grad(set_to_none=True)
            logit.backward()
            grads_acc += tok.grad.detach()

        avg_grads = grads_acc / steps
        ig = (tok_in - tok_base) * avg_grads  # (1, F, d)

        # signed + abs scores per token
        tok_signed = ig.sum(dim=-1).detach().cpu().numpy().reshape(-1)
        tok_abs = ig.abs().sum(dim=-1).detach().cpu().numpy().reshape(-1)

        # normalize by abs sum
        denom = float(tok_abs.sum() + 1e-12)
        tok_norm = tok_abs / denom

        return tok_signed, tok_abs, tok_norm

    @torch.no_grad()
    def predict_one(self, record: dict, mode="triage", explain=True):
        x_num, x_cat = self._vectorize_one(record)
        x_num_t = torch.tensor(x_num, dtype=torch.float32, device=self.device)
        x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=self.device)

        logit = float(self.model.classify_logits(x_num_t, x_cat_t).item())
        p_raw = float(1.0 / (1.0 + np.exp(-logit)))

        p_cal = float(self.calibrator.predict_proba(np.array([[p_raw]], dtype=np.float32))[:, 1][0])

        thr = self.meta["thresholds"]
        if mode not in thr:
            mode = "triage"
        pt = float(thr[mode])
        label = "positive" if p_cal >= pt else "negative"

        out = {
            "prob_raw": p_raw,
            "prob": p_cal,
            "decision_mode": mode,
            "threshold": pt,
            "label": label,
            "meta": {
                "model_name": self.meta.get("model_name"),
                "calibration": self.meta.get("calibration"),
                "train": self.meta.get("train", {})
            }
        }

        # ---- Single-case IG (Transformer-only) ----
        if explain:
            # Need grads: temporarily enable grad
            torch.set_grad_enabled(True)
            try:
                tok_signed, tok_abs, tok_norm = self._ig_single(x_num_t, x_cat_t, steps=int(self.IG_STEPS))

                # top-k by normalized abs attribution
                idx = np.argsort(-tok_norm)[: int(self.IG_TOPK)]

                pos_list = []
                neg_list = []
                for j in idx:
                    name = self.feature_names[j] if j < len(self.feature_names) else f"f{j}"
                    pct = tok_norm[j] * 100.0
                    sgn = tok_signed[j]

                    item = f"{name}: {pct:.1f}%"
                    if sgn >= 0:
                        pos_list.append(item)
                    else:
                        neg_list.append(item)

                out["ig"] = {
                    "positive": pos_list,
                    "negative": neg_list,
                }
            finally:
                torch.set_grad_enabled(False)

        return out


_predictor = None

def predict_one(record: dict, mode="triage"):
    global _predictor
    if _predictor is None:
        _predictor = CervixRiskPredictor()
    return _predictor.predict_one(record, mode=mode, explain=True)
