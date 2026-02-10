# final_model/predict_api.py
import os, json
import numpy as np
import joblib
import torch
import torch.nn as nn


class TabTokTransformer(nn.Module):
    """
    MUST match the exported architecture in your training/export script.
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
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        self.cls_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )

        # recon heads (kept to satisfy strict state_dict loading)
        self.num_recon = nn.Linear(self.d_model, 1)
        self.cat_recon = nn.ModuleList([nn.Linear(self.d_model, int(card)) for card in cat_cards])

    def forward_tokens(self, x_num, x_cat):
        B = x_num.size(0)
        num_tok = self.num_value_proj(x_num.unsqueeze(-1))  # [B, n_num, d]

        if self.n_cat > 0:
            cat_toks = [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embs)]
            cat_tok = torch.stack(cat_toks, dim=1)  # [B, n_cat, d]
            tok = torch.cat([num_tok, cat_tok], dim=1)  # [B, n_features, d]
        else:
            tok = num_tok

        if self.col_id_emb is not None:
            fids = torch.arange(self.n_features, device=tok.device).unsqueeze(0).repeat(B, 1)
            tok = tok + self.col_id_emb(fids)

        return tok

    def classify_logits(self, x_num, x_cat):
        tok = self.forward_tokens(x_num, x_cat)
        return self.classify_from_tokens(tok)

    def classify_from_tokens(self, tok):
        h = self.encoder(tok)
        pooled = h.mean(dim=1)
        return self.cls_head(pooled).squeeze(-1)


class CervixRiskPredictor:
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
        # weights_only available on torch>=2.x; if older, fall back safely
        try:
            state = torch.load(state_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.feature_cols = list(self.preprocess["feature_cols"])
        self.num_cols = list(self.preprocess["num_cols"])
        self.cat_cols = list(self.preprocess["cat_cols"])

    def _vectorize_one(self, record: dict):
        feat = self.feature_cols
        num_cols = self.num_cols
        cat_cols = self.cat_cols

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

    def _integrated_gradients_tokens(self, x_num_t, x_cat_t, steps=32):
        """
        IG on token embeddings (works for categorical as well by interpolating embeddings).
        Returns token attribution scores (signed) for each feature.
        """
        self.model.eval()

        # actual tokens
        tok_actual = self.model.forward_tokens(x_num_t, x_cat_t)  # [1, F, D]

        # baseline: numeric zeros (already standardized space), categorical known baseline idx=0
        x_num_base = torch.zeros_like(x_num_t)
        if x_cat_t.numel() > 0:
            x_cat_base = torch.zeros_like(x_cat_t)
        else:
            x_cat_base = x_cat_t
        tok_base = self.model.forward_tokens(x_num_base, x_cat_base)

        delta = tok_actual - tok_base  # [1, F, D]

        # Riemann sum of gradients along the straight-line path
        total_grad = torch.zeros_like(tok_actual)
        for i in range(1, steps + 1):
            alpha = float(i) / float(steps)
            tok_i = (tok_base + alpha * delta).detach().requires_grad_(True)

            logit = self.model.classify_from_tokens(tok_i)  # [1]
            # scalar
            s = logit.sum()
            s.backward()

            grad = tok_i.grad.detach()
            total_grad += grad

        avg_grad = total_grad / float(steps)
        ig = (delta * avg_grad).sum(dim=-1).squeeze(0)  # [F] signed

        return ig.detach().cpu().numpy()

    @torch.no_grad()
    def _confirm_forward(self, x_num_t, x_cat_t):
        logit = float(self.model.classify_logits(x_num_t, x_cat_t).item())
        p_raw = float(1.0 / (1.0 + np.exp(-logit)))
        p_cal = float(self.calibrator.predict_proba(np.array([[p_raw]], dtype=np.float32))[:, 1][0])
        return logit, p_raw, p_cal

    def predict_one(self, record: dict, mode="triage", return_ig=True):
        x_num, x_cat = self._vectorize_one(record)
        x_num_t = torch.tensor(x_num, dtype=torch.float32, device=self.device)
        x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=self.device)

        # forward
        logit, p_raw, p_cal = self._confirm_forward(x_num_t, x_cat_t)

        thr = self.meta["thresholds"]
        if mode not in thr:
            mode = "triage"
        pt = float(thr[mode])
        label = "positive" if p_cal >= pt else "negative"

        out = {
            "prob_raw": p_raw,
            "prob": float(p_cal),
            "decision_mode": mode,
            "threshold": pt,
            "label": label,
            "meta": {
                "model_name": self.meta.get("model_name"),
                "calibration": self.meta.get("calibration"),
                "train": self.meta.get("train", {}),
            },
        }

        if return_ig:
            # IG requires autograd -> temporarily enable grad
            try:
                with torch.enable_grad():
                    ig_scores = self._integrated_gradients_tokens(x_num_t, x_cat_t, steps=32)
                feat_names = self.num_cols + self.cat_cols

                # rank by magnitude, keep sign
                idx = np.argsort(-np.abs(ig_scores))
                pos, neg = [], []
                for j in idx[:12]:
                    name = feat_names[j] if j < len(feat_names) else f"f{j}"
                    val = float(ig_scores[j])
                    if val >= 0:
                        pos.append(f"{name} (contribution +{abs(val):.3f})")
                    else:
                        neg.append(f"{name} (contribution -{abs(val):.3f})")

                out["ig"] = {"positive": pos[:6], "negative": neg[:6]}
            except Exception:
                # don't break prediction if IG fails
                out["ig"] = None

        return out


_predictor = None


def predict_one(record: dict, mode="triage"):
    global _predictor
    if _predictor is None:
        _predictor = CervixRiskPredictor()
    return _predictor.predict_one(record, mode=mode, return_ig=True)
