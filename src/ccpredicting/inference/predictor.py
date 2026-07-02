"""Inference API for the exported CCPredicting Tab-MFM model."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ccpredicting.models.tab_mfm import TabTokTransformer
from ccpredicting.schema import display_mode, normalize_mode, normalize_record, validate_probability


def default_model_dir() -> Path:
    """Resolve the default model artifact directory.

    Priority:
    1. CCPREDICTING_MODEL_DIR environment variable.
    2. ``final_model`` under the repository root.
    3. ``final_model`` under the current working directory.
    """
    env_dir = os.getenv("CCPREDICTING_MODEL_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "final_model"
    if candidate.exists():
        return candidate
    return Path.cwd() / "final_model"


class CervixRiskPredictor:
    """Load exported preprocessing, calibration and Tab-MFM artifacts."""

    def __init__(self, model_dir: str | os.PathLike[str] | None = None, device: str | None = None) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else default_model_dir()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.preprocess = joblib.load(self.model_dir / "preprocess.pkl")
        self.calibrator = joblib.load(self.model_dir / "calibrator.pkl")
        with open(self.model_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(self.model_dir / "arch.json", "r", encoding="utf-8") as f:
            arch = json.load(f)

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

        state_path = self.model_dir / "transformer_state.pt"
        try:
            state = torch.load(state_path, map_location=self.device, weights_only=True)
        except TypeError:  # compatibility with older torch versions
            state = torch.load(state_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @property
    def feature_names(self) -> list[str]:
        return list(self.preprocess["num_cols"]) + list(self.preprocess["cat_cols"])

    def _vectorize_one(self, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        record = normalize_record(record)
        feature_cols = self.preprocess["feature_cols"]
        num_cols = self.preprocess["num_cols"]
        cat_cols = self.preprocess["cat_cols"]
        row = {c: record.get(c, None) for c in feature_cols}

        x_num = np.array([row.get(c, np.nan) for c in num_cols], dtype=np.float32)
        med = np.asarray(self.preprocess["num_med"], dtype=np.float32)
        mean = np.asarray(self.preprocess["num_mean"], dtype=np.float32)
        std = np.asarray(self.preprocess["num_std"], dtype=np.float32)
        x_num = np.where(np.isnan(x_num), med, x_num)
        x_num = (x_num - mean) / std

        cat_maps = self.preprocess["cat_maps"]
        x_cat = []
        for j, c in enumerate(cat_cols):
            value = row.get(c, None)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                key = "nan"
            else:
                try:
                    key = str(int(value))
                except Exception:
                    key = str(value)
            x_cat.append(cat_maps[j].get(key, 0))
        return x_num[None, :], np.array(x_cat, dtype=np.int64)[None, :]

    @torch.no_grad()
    def predict_one(self, record: dict[str, Any], mode: str = "balanced") -> dict[str, Any]:
        """Predict calibrated CIN2+ risk for one structured clinical record.

        ``mode`` accepts both public names (``high_sensitivity``, ``balanced``,
        ``youden``) and legacy names (``screen``, ``triage``, ``youden``).
        """
        x_num, x_cat = self._vectorize_one(record)
        x_num_t = torch.tensor(x_num, dtype=torch.float32, device=self.device)
        x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=self.device)

        logit = float(self.model.classify_logits(x_num_t, x_cat_t).item())
        p_raw = validate_probability(1.0 / (1.0 + np.exp(-logit)))
        p_cal = validate_probability(self.calibrator.predict_proba(np.array([[p_raw]], dtype=np.float32))[:, 1][0])

        legacy_mode = normalize_mode(mode)
        thresholds = self.meta.get("thresholds", {})
        threshold = float(thresholds.get(legacy_mode, thresholds.get("triage", 0.5)))
        label = "positive" if p_cal >= threshold else "negative"

        return {
            "prob_raw": p_raw,
            "prob": p_cal,
            "decision_mode": display_mode(legacy_mode),
            "legacy_decision_mode": legacy_mode,
            "threshold": threshold,
            "label": label,
            "meta": {
                "model_name": self.meta.get("model_name"),
                "calibration": self.meta.get("calibration"),
                "train": self.meta.get("train", {}),
            },
        }

    def explain_one_ig(self, record: dict[str, Any], steps: int = 48) -> dict[str, Any]:
        """Single-case Integrated Gradients on feature-token embeddings."""
        self.model.eval()
        x_num, x_cat = self._vectorize_one(record)
        x_num_t = torch.tensor(x_num, dtype=torch.float32, device=self.device)
        x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=self.device)

        x_num0 = torch.zeros_like(x_num_t)
        x_cat0 = torch.zeros_like(x_cat_t)
        with torch.no_grad():
            tok_in = self.model.forward_tokens(x_num_t, x_cat_t)
            tok_0 = self.model.forward_tokens(x_num0, x_cat0)
        delta = tok_in - tok_0

        steps = int(max(8, steps))
        alphas = torch.linspace(0.0, 1.0, steps, device=self.device)
        grad_sum = torch.zeros_like(tok_in)
        for alpha in alphas:
            tok = tok_0 + alpha * delta
            tok_leaf = tok.detach().clone().requires_grad_(True)
            logit = self.model.classify_logits_from_tokens(tok_leaf)
            self.model.zero_grad(set_to_none=True)
            logit.sum().backward()
            if tok_leaf.grad is None:
                raise RuntimeError("Integrated Gradients failed because token gradients are None.")
            grad_sum += tok_leaf.grad.detach()

        avg_grad = grad_sum / float(steps)
        attrs = (delta * avg_grad).squeeze(0).sum(dim=1).detach().cpu().numpy().astype(np.float32)
        pred = self.predict_one(record, mode="balanced")
        return {
            "feature_names": self.feature_names,
            "attributions": attrs.tolist(),
            "prob_raw": pred["prob_raw"],
            "prob": pred["prob"],
        }

    def explain_one_ig_png(self, record: dict[str, Any], steps: int = 48, top_k: int = 10):
        """Return ``(png_bytes, attribution_table, metadata)`` for one record."""
        out = self.explain_one_ig(record, steps=steps)
        names = out["feature_names"]
        attrs = np.asarray(out["attributions"], dtype=np.float32)
        k = int(max(1, min(len(attrs), top_k)))
        idx = np.argsort(-np.abs(attrs))[:k]
        sel_names = [names[i] for i in idx][::-1]
        sel_attrs = attrs[idx][::-1]

        fig = plt.figure(figsize=(7.2, 4.2))
        ax = fig.add_subplot(111)
        y = np.arange(len(sel_names))
        ax.barh(y, sel_attrs)
        ax.set_yticks(y)
        ax.set_yticklabels(sel_names)
        ax.set_xlabel("Integrated Gradients (signed)")
        ax.set_title("Single-case IG (top features)")
        ax.axvline(0.0, linewidth=1)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        buf.seek(0)

        table = [{"feature": names[i], "ig": float(attrs[i])} for i in idx]
        return buf.getvalue(), table, out


_predictor: CervixRiskPredictor | None = None


def get_default_predictor() -> CervixRiskPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CervixRiskPredictor()
    return _predictor


def predict_one(record: dict[str, Any], mode: str = "balanced") -> dict[str, Any]:
    return get_default_predictor().predict_one(record, mode=mode)


def explain_one_ig_png(record: dict[str, Any], steps: int = 48, top_k: int = 10):
    return get_default_predictor().explain_one_ig_png(record, steps=steps, top_k=top_k)
