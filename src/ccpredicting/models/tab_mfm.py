"""Tab-MFM / Tabular Transformer model definition used by the exported artifact.

This class must remain architecture-compatible with the exported
``final_model/transformer_state.pt`` state dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TabTokTransformer(nn.Module):
    """Feature-token Transformer for structured tabular prediction.

    The reconstruction heads are retained because the exported state dict was
    generated from the MFM pretraining + finetuning model. Removing them would
    break strict state-dict loading even though they are not used in inference.
    """

    def __init__(
        self,
        n_num: int,
        cat_cards: list[int],
        d_model: int = 64,
        n_head: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.15,
        use_col_id_emb: bool = True,
    ) -> None:
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

        self.num_recon = nn.Linear(self.d_model, 1)
        self.cat_recon = nn.ModuleList([nn.Linear(self.d_model, int(card)) for card in cat_cards])

    def forward_tokens(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Convert numeric and categorical features into feature tokens."""
        batch_size = x_num.size(0)
        num_tok = self.num_value_proj(x_num.unsqueeze(-1))
        if self.n_cat > 0:
            cat_toks = [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embs)]
            cat_tok = torch.stack(cat_toks, dim=1)
            tok = torch.cat([num_tok, cat_tok], dim=1)
        else:
            tok = num_tok

        if self.col_id_emb is not None:
            fids = torch.arange(self.n_features, device=tok.device).unsqueeze(0).repeat(batch_size, 1)
            tok = tok + self.col_id_emb(fids)
        return tok

    def classify_logits(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.forward_tokens(x_num, x_cat))
        pooled = h.mean(dim=1)
        return self.cls_head(pooled).squeeze(-1)

    def classify_logits_from_tokens(self, tok: torch.Tensor) -> torch.Tensor:
        """Classify from precomputed tokens, used by Integrated Gradients."""
        h = self.encoder(tok)
        pooled = h.mean(dim=1)
        return self.cls_head(pooled).squeeze(-1)
