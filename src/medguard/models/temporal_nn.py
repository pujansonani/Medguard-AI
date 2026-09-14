"""
Temporal Deep Learning Models for Structured ICU Time-Series (GRU, LSTM, TCN).
Processes (Batch, Seq_len, Features) + (Batch, Seq_len, Mask) to generate temporal embeddings and multi-horizon logits.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from medguard.models.base import StructuredModel
from medguard.utils.logging import get_logger

logger = get_logger("medguard.models.temporal_nn")


class TemporalEncoderGRU(nn.Module):
    """
    Bidirectional/Unidirectional GRU with missingness mask concatenation and multi-horizon output head.
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        embedding_dim: int = 64,
        num_horizons: int = 4, # 6h, 12h, 24h, 48h
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.num_horizons = num_horizons

        # We concatenate raw values + missingness mask: Total input dimension = input_dim * 2
        total_input_dim = input_dim * 2

        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Temporal self-attention pooling
        self.attn_weights = nn.Linear(gru_out_dim, 1)

        # Output projection for structured embedding
        self.proj_embedding = nn.Sequential(
            nn.Linear(gru_out_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Multi-horizon prediction head (6h, 12h, 24h, 48h)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_horizons),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) scaled physiological measurements
            mask: (B, T, D) binary observation mask

        Returns:
            logits: (B, num_horizons)
            embedding: (B, embedding_dim)
            seq_hidden: (B, T, gru_out_dim) sequence representation
        """
        # Concatenate measurements and missingness indicators
        inputs = torch.cat([x, mask], dim=-1) # (B, T, 2D)
        h = self.input_proj(inputs) # (B, T, hidden_dim)

        out, _ = self.gru(h) # (B, T, gru_out_dim)

        # Attention pooling across sequence length T
        scores = self.attn_weights(out) # (B, T, 1)
        weights = F.softmax(scores, dim=1) # (B, T, 1)
        pooled = torch.sum(weights * out, dim=1) # (B, gru_out_dim)

        embedding = self.proj_embedding(pooled) # (B, embedding_dim)
        logits = self.head(embedding) # (B, num_horizons)

        return logits, embedding, out


class TemporalEncoderLSTM(nn.Module):
    """LSTM Temporal Architecture with Attention Pooling."""

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        embedding_dim: int = 64,
        num_horizons: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_horizons = num_horizons

        total_input_dim = input_dim * 2

        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attn_weights = nn.Linear(lstm_out_dim, 1)

        self.proj_embedding = nn.Sequential(
            nn.Linear(lstm_out_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_horizons),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.cat([x, mask], dim=-1)
        h = self.input_proj(inputs)
        out, _ = self.lstm(h)

        scores = self.attn_weights(out)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(weights * out, dim=1)

        embedding = self.proj_embedding(pooled)
        logits = self.head(embedding)
        return logits, embedding, out


class TemporalGRUModel(StructuredModel):
    """Wrapper class providing scikit-learn / MEDGUARD interface for PyTorch Temporal GRU."""

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        super().__init__(name="TemporalGRU", version="0.1.0")
        self.device = torch.device(device)
        self.net = TemporalEncoderGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            embedding_dim=hidden_dim,
            num_horizons=4,
        ).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(
        self,
        X_train: np.ndarray,
        mask_train: np.ndarray,
        y_train: np.ndarray, # (N, 4) multi-horizon or (N,)
        X_val: Optional[np.ndarray] = None,
        mask_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TemporalGRUModel":
        self.net.train()
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.5]).to(self.device))

        if y_train.ndim == 1:
            # Expand to 4 horizons if single target provided
            y_train = np.column_stack([y_train * 0.4, y_train * 0.6, y_train * 0.8, y_train])
        if y_val is not None and y_val.ndim == 1:
            y_val = np.column_stack([y_val * 0.4, y_val * 0.6, y_val * 0.8, y_val])

        N = len(X_train)
        num_batches = int(np.ceil(N / self.batch_size))

        for epoch in range(self.epochs):
            self.net.train()
            indices = np.random.permutation(N)
            epoch_loss = 0.0

            for b in range(num_batches):
                batch_idx = indices[b * self.batch_size : (b + 1) * self.batch_size]
                bx = torch.tensor(X_train[batch_idx], dtype=torch.float32).to(self.device)
                bmask = torch.tensor(mask_train[batch_idx], dtype=torch.float32).to(self.device)
                by = torch.tensor(y_train[batch_idx], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()
                logits, _, _ = self.net(bx, bmask)
                loss = criterion(logits, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Returns array of shape (N, 4) for horizons [6h, 12h, 24h, 48h]."""
        self.net.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(self.device)
            tmask = torch.tensor(mask, dtype=torch.float32).to(self.device)
            logits, _, _ = self.net(tx, tmask)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def get_embeddings(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract structured representation vectors e_s of shape (N, embedding_dim)."""
        self.net.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(self.device)
            tmask = torch.tensor(mask, dtype=torch.float32).to(self.device)
            _, embs, _ = self.net(tx, tmask)
            return embs.cpu().numpy()

    def save(self, filepath: Union[str, Path]) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), filepath)

    def load(self, filepath: Union[str, Path]) -> "TemporalGRUModel":
        state_dict = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.is_fitted = True
        return self
