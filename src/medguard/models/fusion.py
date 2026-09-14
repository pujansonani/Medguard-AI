"""
Multimodal Fusion Architectures for MEDGUARD AI:
1. Late Fusion (Learned / Calibrated Logit Ensembling)
2. Intermediate Concatenation Fusion
3. Gated Multimodal Fusion (Learned Sigmoid Modality Gating)
4. Cross-Modal Attention Fusion (Experimental)
Includes Monte Carlo Dropout for epistemic uncertainty estimation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from medguard.models.base import MultimodalModel
from medguard.models.temporal_nn import TemporalGRUModel, TemporalEncoderGRU
from medguard.models.clinical_nlp import ClinicalNLPModel, ClinicalTextEncoderNN, SimpleClinicalTokenizer
from medguard.utils.logging import get_logger

logger = get_logger("medguard.models.fusion")


class GatedFusionModule(nn.Module):
    """
    Sigmoid-gated multimodal fusion layer:
    g = sigmoid(W_s * e_s + W_t * e_t + b)
    e_f = g * Proj_s(e_s) + (1 - g) * Proj_t(e_t)
    """

    def __init__(self, structured_dim: int = 64, text_dim: int = 128, fusion_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.proj_struct = nn.Sequential(
            nn.Linear(structured_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        self.proj_text = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
        # Modality Gate: outputs scalar or elementwise gating weight in [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(structured_dim + text_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, e_s: torch.Tensor, e_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_s = self.proj_struct(e_s) # (B, fusion_dim)
        h_t = self.proj_text(e_t)   # (B, fusion_dim)

        combined = torch.cat([e_s, e_t], dim=-1)
        gate = self.gate_net(combined) # (B, fusion_dim)

        # Gated fusion
        e_f = gate * h_s + (1.0 - gate) * h_t
        e_f = self.dropout(e_f)
        return e_f, gate


class IntermediateConcatModule(nn.Module):
    """Intermediate fusion via concatenation and feedforward projection."""

    def __init__(self, structured_dim: int = 64, text_dim: int = 128, fusion_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(structured_dim + text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, e_s: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([e_s, e_t], dim=-1)
        return self.mlp(combined)


class CrossModalAttentionModule(nn.Module):
    """
    Experimental Cross-Modal Attention:
    Clinical text queries structured physiological sequence states (B, T, D).
    """

    def __init__(self, structured_seq_dim: int = 64, text_dim: int = 128, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.proj_seq = nn.Linear(structured_seq_dim, embed_dim)
        self.proj_text = nn.Linear(text_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, seq_hidden: torch.Tensor, e_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # seq_hidden: (B, T, struct_dim), e_t: (B, text_dim)
        k_v = self.proj_seq(seq_hidden) # (B, T, embed_dim)
        q = self.proj_text(e_t).unsqueeze(1) # (B, 1, embed_dim)

        attn_out, attn_weights = self.mha(q, k_v, k_v) # attn_out: (B, 1, embed_dim), weights: (B, 1, T)
        attn_out = self.norm(attn_out.squeeze(1)) # (B, embed_dim)

        fused = self.proj_out(torch.cat([attn_out, q.squeeze(1)], dim=-1))
        return fused, attn_weights.squeeze(1) # (B, embed_dim), (B, T)


class MultimodalEndToEndNet(nn.Module):
    """Complete End-to-End Multimodal PyTorch Network combining Structured Encoder, Text Encoder, and Fusion Head."""

    def __init__(
        self,
        struct_input_dim: int = 18,
        struct_hidden_dim: int = 64,
        text_hidden_dim: int = 128,
        fusion_type: str = "gated", # "gated", "intermediate", "cross_attention", "late"
        fusion_dim: int = 128,
        num_horizons: int = 4, # 6h, 12h, 24h, 48h
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_horizons = num_horizons

        # 1. Structured Temporal Encoder
        self.struct_encoder = TemporalEncoderGRU(
            input_dim=struct_input_dim,
            hidden_dim=struct_hidden_dim,
            embedding_dim=struct_hidden_dim,
            num_horizons=num_horizons,
            dropout=dropout,
        )

        # 2. Clinical Text Encoder
        self.text_encoder = ClinicalTextEncoderNN(
            vocab_size=32000,
            word_embed_dim=128,
            hidden_dim=text_hidden_dim,
            embedding_dim=text_hidden_dim,
            num_horizons=num_horizons,
            dropout=dropout,
        )

        # 3. Fusion Layer
        if fusion_type == "gated":
            self.fusion_layer = GatedFusionModule(struct_hidden_dim, text_hidden_dim, fusion_dim, dropout)
            final_dim = fusion_dim
        elif fusion_type == "intermediate":
            self.fusion_layer = IntermediateConcatModule(struct_hidden_dim, text_hidden_dim, fusion_dim, dropout)
            final_dim = fusion_dim
        elif fusion_type == "cross_attention":
            self.cross_attn = CrossModalAttentionModule(struct_hidden_dim, text_hidden_dim, embed_dim=64, num_heads=4, dropout=dropout)
            final_dim = 64
        elif fusion_type == "late":
            self.late_alpha = nn.Parameter(torch.tensor([0.5]))
            final_dim = None

        # 4. Multi-Horizon Prediction Head
        if fusion_type != "late":
            self.head = nn.Sequential(
                nn.Linear(final_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_horizons),
            )

    def forward(
        self,
        x_struct: torch.Tensor,
        mask_struct: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits_s, e_s, seq_hidden = self.struct_encoder(x_struct, mask_struct)
        logits_t, e_t, token_weights = self.text_encoder(input_ids, attention_mask)

        gate_weights = None
        cross_attn_weights = None

        if self.fusion_type == "gated":
            e_f, gate_weights = self.fusion_layer(e_s, e_t)
            logits_fused = self.head(e_f)
        elif self.fusion_type == "intermediate":
            e_f = self.fusion_layer(e_s, e_t)
            logits_fused = self.head(e_f)
        elif self.fusion_type == "cross_attention":
            e_f, cross_attn_weights = self.cross_attn(seq_hidden, e_t)
            logits_fused = self.head(e_f)
        elif self.fusion_type == "late":
            alpha = torch.sigmoid(self.late_alpha)
            logits_fused = alpha * logits_s + (1.0 - alpha) * logits_t
            gate_weights = alpha.expand(x_struct.size(0), 1)

        return {
            "logits": logits_fused,
            "logits_struct": logits_s,
            "logits_text": logits_t,
            "e_s": e_s,
            "e_t": e_t,
            "gate_weights": gate_weights,
            "cross_attn_weights": cross_attn_weights,
            "token_weights": token_weights,
        }


class MedguardMultimodalModel(MultimodalModel):
    """Production Wrapper for the End-to-End Multimodal Model."""

    def __init__(
        self,
        fusion_type: str = "gated",
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        super().__init__(name=f"MedguardFusion_{fusion_type}", version="0.1.0")
        self.fusion_type = fusion_type
        self.device = torch.device(device)
        self.tokenizer = SimpleClinicalTokenizer(max_length=256)
        self.net = MultimodalEndToEndNet(
            struct_input_dim=18,
            struct_hidden_dim=64,
            text_hidden_dim=128,
            fusion_type=fusion_type,
            fusion_dim=128,
            num_horizons=4,
        ).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        all_ids, all_masks = [], []
        for t in texts:
            ids, mask, _ = self.tokenizer.encode(t)
            all_ids.append(ids)
            all_masks.append(mask)
        return (
            torch.tensor(all_ids, dtype=torch.long).to(self.device),
            torch.tensor(all_masks, dtype=torch.long).to(self.device),
        )

    def fit(
        self,
        X_struct: np.ndarray,
        mask_struct: np.ndarray,
        texts: List[str],
        y_train: np.ndarray, # (N, 4)
        X_val: Optional[np.ndarray] = None,
        mask_val: Optional[np.ndarray] = None,
        texts_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MedguardMultimodalModel":
        self.net.train()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.5]).to(self.device))

        if y_train.ndim == 1:
            y_train = np.column_stack([y_train * 0.4, y_train * 0.6, y_train * 0.8, y_train])

        N = len(X_struct)
        num_batches = int(np.ceil(N / self.batch_size))

        for epoch in range(self.epochs):
            self.net.train()
            indices = np.random.permutation(N)

            for b in range(num_batches):
                batch_idx = indices[b * self.batch_size : (b + 1) * self.batch_size]
                bx_s = torch.tensor(X_struct[batch_idx], dtype=torch.float32).to(self.device)
                bmask_s = torch.tensor(mask_struct[batch_idx], dtype=torch.float32).to(self.device)
                b_texts = [texts[i] for i in batch_idx]
                b_ids, b_mask_t = self._tokenize_batch(b_texts)
                by = torch.tensor(y_train[batch_idx], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()
                out = self.net(bx_s, bmask_s, b_ids, b_mask_t)
                
                # Joint loss: Main fused loss + auxiliary single-modality supervision
                loss_f = criterion(out["logits"], by)
                loss_s = criterion(out["logits_struct"], by)
                loss_t = criterion(out["logits_text"], by)
                total_loss = loss_f + 0.3 * loss_s + 0.3 * loss_t

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X_struct: np.ndarray,
        mask_struct: np.ndarray,
        texts: List[str],
    ) -> np.ndarray:
        """Returns fused probabilities of shape (N, 4) for [6h, 12h, 24h, 48h]."""
        self.net.eval()
        with torch.no_grad():
            bx_s = torch.tensor(X_struct, dtype=torch.float32).to(self.device)
            bmask_s = torch.tensor(mask_struct, dtype=torch.float32).to(self.device)
            b_ids, b_mask_t = self._tokenize_batch(texts)
            out = self.net(bx_s, bmask_s, b_ids, b_mask_t)
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
        return probs

    def predict_with_uncertainty(
        self,
        X_struct: np.ndarray,
        mask_struct: np.ndarray,
        texts: List[str],
        num_mc_samples: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout for epistemic uncertainty estimation.
        Keeps dropout active during inference across K forward passes.

        Returns:
            mean_probs: (N, 4) expected mortality probabilities
            std_probs: (N, 4) epistemic uncertainty standard deviation
        """
        self.net.train() # Enable dropout
        mc_predictions = []

        with torch.no_grad():
            bx_s = torch.tensor(X_struct, dtype=torch.float32).to(self.device)
            bmask_s = torch.tensor(mask_struct, dtype=torch.float32).to(self.device)
            b_ids, b_mask_t = self._tokenize_batch(texts)

            for _ in range(num_mc_samples):
                out = self.net(bx_s, bmask_s, b_ids, b_mask_t)
                p = torch.sigmoid(out["logits"]).cpu().numpy()
                mc_predictions.append(p)

        self.net.eval()
        stacked = np.stack(mc_predictions, axis=0) # (K, N, 4)
        mean_probs = np.mean(stacked, axis=0)
        std_probs = np.std(stacked, axis=0)
        return mean_probs, std_probs

    def decompose_modality_contributions(
        self,
        X_struct: np.ndarray,
        mask_struct: np.ndarray,
        texts: List[str],
        horizon_idx: int = 3, # Default: 48h horizon
    ) -> List[Dict[str, float]]:
        """
        Decomposes risk for each patient into:
        - Structured-only risk: P(mortality | vitals, labs)
        - Text-only risk: P(mortality | clinical notes)
        - Multimodal risk: P(mortality | vitals, labs, notes)
        - Delta gains: (P_multi - P_struct) and (P_multi - P_text)
        - Gating weight: g in [0, 1]
        """
        self.net.eval()
        with torch.no_grad():
            bx_s = torch.tensor(X_struct, dtype=torch.float32).to(self.device)
            bmask_s = torch.tensor(mask_struct, dtype=torch.float32).to(self.device)
            b_ids, b_mask_t = self._tokenize_batch(texts)
            out = self.net(bx_s, bmask_s, b_ids, b_mask_t)

            p_fused = torch.sigmoid(out["logits"][:, horizon_idx]).cpu().numpy()
            p_struct = torch.sigmoid(out["logits_struct"][:, horizon_idx]).cpu().numpy()
            p_text = torch.sigmoid(out["logits_text"][:, horizon_idx]).cpu().numpy()
            
            gate = out["gate_weights"]
            g_vals = gate.mean(dim=-1).cpu().numpy() if gate is not None else np.full(len(p_fused), 0.5)

        results = []
        for i in range(len(p_fused)):
            pf = float(p_fused[i])
            ps = float(p_struct[i])
            pt = float(p_text[i])
            results.append({
                "multimodal_prob": round(pf, 4),
                "structured_prob": round(ps, 4),
                "text_prob": round(pt, 4),
                "gate_weight": round(float(g_vals[i]), 4),
                "delta_vs_structured": round(pf - ps, 4),
                "delta_vs_text": round(pf - pt, 4),
            })
        return results

    def get_modality_weights(
        self,
        X_struct: np.ndarray,
        mask_struct: np.ndarray,
        texts: List[str],
    ) -> Optional[np.ndarray]:
        """Returns gating coefficients g (mean value per patient) in [0, 1]."""
        self.net.eval()
        with torch.no_grad():
            bx_s = torch.tensor(X_struct, dtype=torch.float32).to(self.device)
            bmask_s = torch.tensor(mask_struct, dtype=torch.float32).to(self.device)
            b_ids, b_mask_t = self._tokenize_batch(texts)
            out = self.net(bx_s, bmask_s, b_ids, b_mask_t)
            if out["gate_weights"] is not None:
                return out["gate_weights"].mean(dim=-1).cpu().numpy()
        return None

    def save(self, filepath: Union[str, Path]) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "fusion_type": self.fusion_type,
            "vocab": self.tokenizer.vocab,
        }, filepath)

    def load(self, filepath: Union[str, Path]) -> "MedguardMultimodalModel":
        data = torch.load(filepath, map_location=self.device)
        self.fusion_type = data["fusion_type"]
        self.tokenizer.vocab = data["vocab"]
        self.net.load_state_dict(data["net"])
        self.is_fitted = True
        return self
