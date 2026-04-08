"""Attention MIL model for donor-level prediction."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class CellEncoder(nn.Module):
    """MLP encoder that maps per-cell embeddings to a latent space.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input cell embedding (e.g. scVI latent dim).
    encoder_dims:
        Hidden layer widths.  The last value is the output dimensionality.
    dropout:
        Dropout probability applied after each hidden layer.
    """

    def __init__(self, input_dim: int, encoder_dims: List[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in encoder_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_dim = dim
        self.net = nn.Sequential(*layers)
        self.output_dim = encoder_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of cell embeddings.

        Parameters
        ----------
        x:
            Tensor of shape ``(n_cells, input_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape ``(n_cells, encoder_dims[-1])``.
        """
        return self.net(x)


class AttentionPooling(nn.Module):
    """Single- or multi-head attention pooling over a bag of cell embeddings.

    Parameters
    ----------
    input_dim:
        Dimensionality of each cell's encoded embedding.
    attention_dim:
        Internal projection dimensionality for the attention MLP.
    n_heads:
        Number of attention heads.
    """

    def __init__(self, input_dim: int, attention_dim: int, n_heads: int = 1) -> None:
        super().__init__()
        self.V = nn.Linear(input_dim, attention_dim, bias=False)
        self.w = nn.Linear(attention_dim, n_heads, bias=False)
        self.n_heads = n_heads

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool a bag of cell embeddings via learned attention.

        Parameters
        ----------
        h:
            Tensor of shape ``(n_cells, input_dim)``.

        Returns
        -------
        pooled : torch.Tensor
            Bag-level representation of shape ``(input_dim,)``.
        attn_weights : torch.Tensor
            Per-cell attention weights of shape ``(n_cells,)``.
        """
        scores = self.w(torch.tanh(self.V(h)))  # (n_cells, n_heads)
        weights = torch.softmax(scores, dim=0)  # (n_cells, n_heads)
        pooled = weights.T @ h  # (n_heads, input_dim)
        pooled = pooled.mean(dim=0)  # (input_dim,)
        return pooled, weights.mean(dim=1)  # mean attention per cell


class MILModel(nn.Module):
    """Full attention MIL model: CellEncoder → AttentionPooling → prediction head.

    Parameters
    ----------
    input_dim:
        Dimensionality of the per-cell input features.
    encoder_dims:
        Hidden layer widths for :class:`CellEncoder`.
    attention_dim:
        Internal dimensionality for :class:`AttentionPooling`.
    n_classes:
        Number of output classes.  Use ``1`` for binary classification or
        regression; use ``N`` for N-class classification.
    task:
        ``"classification"`` or ``"regression"``.
    dropout:
        Dropout probability.
    n_heads:
        Number of attention heads.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int],
        attention_dim: int,
        n_classes: int,
        task: str = "classification",
        dropout: float = 0.2,
        n_heads: int = 1,
    ) -> None:
        super().__init__()
        self.task = task
        self.n_classes = n_classes

        self.encoder = CellEncoder(input_dim, encoder_dims, dropout=dropout)
        self.attention = AttentionPooling(encoder_dims[-1], attention_dim, n_heads=n_heads)
        self.head = nn.Linear(encoder_dims[-1], n_classes)

    def forward_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single donor bag.

        Parameters
        ----------
        x:
            Tensor of shape ``(n_cells, input_dim)``.

        Returns
        -------
        logits : torch.Tensor
            Shape ``(n_classes,)`` or ``(1,)``.
        attn_weights : torch.Tensor
            Per-cell attention weights of shape ``(n_cells,)``.
        """
        h = self.encoder(x)           # (n_cells, enc_dim)
        pooled, attn = self.attention(h)  # (enc_dim,), (n_cells,)
        logits = self.head(pooled)    # (n_classes,)
        return logits, attn

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass, handling both single-bag and batched inputs.

        Parameters
        ----------
        x:
            Shape ``(n_cells, input_dim)`` for a single bag **or**
            ``(batch_size, n_cells, input_dim)`` for a batch.

        Returns
        -------
        logits : torch.Tensor
            Shape ``(batch_size, n_classes)`` or ``(n_classes,)``.
        attn_weights : torch.Tensor
            Shape ``(batch_size, n_cells)`` or ``(n_cells,)``.
        """
        if x.dim() == 2:
            # Single bag
            return self.forward_single(x)

        # Batched: (batch, n_cells, input_dim)
        batch_size = x.shape[0]
        all_logits = []
        all_attn = []
        for i in range(batch_size):
            logits_i, attn_i = self.forward_single(x[i])
            all_logits.append(logits_i)
            all_attn.append(attn_i)
        logits = torch.stack(all_logits, dim=0)   # (batch, n_classes)
        attn = torch.stack(all_attn, dim=0)        # (batch, n_cells)
        return logits, attn
