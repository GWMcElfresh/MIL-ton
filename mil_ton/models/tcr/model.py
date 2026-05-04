"""BertTCR classification model: 1-D CNN feature extractor + MIL ensemble head.

Architecture (per Zhang et al. 2024):

1. Input: per-sample bag of TCR BERT embeddings,
   shape ``(batch, n_tcrs, bert_hidden_size, max_tcr_len)`` or
   ``(n_tcrs, bert_hidden_size, max_tcr_len)`` for a single bag.
2. Parallel 1-D convolutions with different kernel sizes extract a
   fixed-length feature vector for each TCR in the bag.
3. A shared linear layer compresses each TCR's CNN features to a scalar
   *TCR score* – this is the **pre-MIL embedding** exposed by the API.
4. An ensemble of ``n_ensemble`` independent linear heads maps the bag of
   per-TCR scores to class logits; the averaged, sigmoid-activated result
   is the final probability output.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BertTCRModel(nn.Module):
    """CNN + MIL ensemble model for TCR repertoire classification.

    The model expects pre-computed BERT embeddings for each TCR in a
    repertoire (a *bag* in MIL terminology).  These can be produced by
    :class:`~mil_ton.models.tcr.encoder.TCRBertEncoder`.

    Parameters
    ----------
    n_tcrs:
        Fixed bag size – the number of TCRs per repertoire sample.
    n_classes:
        Number of output classes.
    filter_num:
        Number of 1-D conv filters for each kernel size.  Must have the
        same length as ``kernel_size``.
    kernel_size:
        Kernel widths for the parallel 1-D conv branches.
    max_tcr_len:
        Expected length (position dimension) of each BERT embedding after
        padding / truncation.  Sequences shorter than this are zero-padded;
        longer ones are truncated.
    dropout:
        Dropout probability applied to the FC and ensemble heads.
    n_ensemble:
        Number of independent linear heads whose predictions are averaged
        (ensemble / bagging strategy from the original paper).
    bert_hidden_size:
        Hidden dimension of the upstream BERT model (channel dimension of
        the ``(bert_hidden_size, max_tcr_len)`` tensor per TCR).  Set this
        to match the backbone you use (768 for tape ``bert-base``, 1024 for
        ``Rostlab/prot_bert_bfd``).
    """

    def __init__(
        self,
        n_tcrs: int,
        n_classes: int = 2,
        filter_num: List[int] = None,
        kernel_size: List[int] = None,
        max_tcr_len: int = 24,
        dropout: float = 0.4,
        n_ensemble: int = 5,
        bert_hidden_size: int = 768,
    ) -> None:
        super().__init__()

        if filter_num is None:
            filter_num = [3, 2, 1]
        if kernel_size is None:
            kernel_size = [2, 3, 4]

        if len(filter_num) != len(kernel_size):
            raise ValueError(
                f"filter_num (len {len(filter_num)}) and kernel_size "
                f"(len {len(kernel_size)}) must have equal length."
            )

        self.n_tcrs = n_tcrs
        self.n_classes = n_classes
        self.filter_num = list(filter_num)
        self.kernel_size = list(kernel_size)
        self.max_tcr_len = max_tcr_len
        self.bert_hidden_size = bert_hidden_size
        self._sum_filters = sum(self.filter_num)

        # Parallel 1-D conv branches: each branch processes one kernel width.
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=bert_hidden_size,
                        out_channels=filter_num[i],
                        kernel_size=k,
                        stride=1,
                    ),
                    nn.Sigmoid(),
                    nn.AdaptiveMaxPool1d(1),
                )
                for i, k in enumerate(kernel_size)
            ]
        )

        # Compress concatenated conv features to a scalar score per TCR.
        self.fc = nn.Linear(self._sum_filters, 1)

        # MIL ensemble heads: each maps bag of TCR scores → class logits.
        self.ensemble_heads = nn.ModuleList(
            [nn.Linear(n_tcrs, n_classes) for _ in range(n_ensemble)]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_or_truncate(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure the position dimension equals ``max_tcr_len``.

        Parameters
        ----------
        x:
            Shape ``(..., bert_hidden_size, L)`` for any leading dims.

        Returns
        -------
        torch.Tensor
            Shape ``(..., bert_hidden_size, max_tcr_len)``.
        """
        L = x.shape[-1]
        if L > self.max_tcr_len:
            return x[..., : self.max_tcr_len]
        if L < self.max_tcr_len:
            pad = (0, self.max_tcr_len - L)
            return F.pad(x, pad, mode="constant", value=0.0)
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a bag (or batch of bags) of TCR embeddings.

        Parameters
        ----------
        x:
            Pre-computed BERT embeddings.  Accepted shapes:

            * ``(n_tcrs, bert_hidden_size, max_tcr_len)`` – single bag.
            * ``(batch, n_tcrs, bert_hidden_size, max_tcr_len)`` – batched.

            If the position dimension differs from ``max_tcr_len`` the
            tensor is automatically padded / truncated (same strategy as the
            original BertTCR pre-processing step).

        Returns
        -------
        probs : torch.Tensor
            Sigmoid-activated class probabilities.
            Shape ``(batch, n_classes)`` or ``(n_classes,)``.
        pre_mil : torch.Tensor
            Per-TCR scalar score **before** the MIL ensemble step.
            This is the bag-level representation that feeds into the
            linear ensemble heads – useful for downstream analysis /
            transfer learning.
            Shape ``(batch, n_tcrs)`` or ``(n_tcrs,)``.
        """
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)  # (1, n_tcrs, H, L)

        batch_size, n_tcrs, H, L = x.shape
        x = self._pad_or_truncate(x)  # (..., H, max_tcr_len)

        # Flatten batch × TCR for the shared CNN.
        x_flat = x.reshape(batch_size * n_tcrs, H, self.max_tcr_len)

        # CNN branches → concatenate along filter dimension.
        conv_outs = [conv(x_flat) for conv in self.convs]  # each: (B*T, f_i, 1)
        cnn_feat = torch.cat(conv_outs, dim=1)              # (B*T, sum_f, 1)

        # Compress to scalar per TCR.
        cnn_feat = cnn_feat.view(batch_size * n_tcrs, 1, self._sum_filters)
        tcr_scores = self.dropout(self.fc(cnn_feat))       # (B*T, 1, 1)
        tcr_scores = tcr_scores.view(batch_size, n_tcrs)   # (B, T)
        # tcr_scores is the pre-MIL embedding: a scalar representation for
        # each TCR in the bag before the ensemble classification step.
        pre_mil = tcr_scores

        # MIL ensemble: average logits from n_ensemble independent heads.
        logit_sum = sum(
            self.dropout(head(tcr_scores)) for head in self.ensemble_heads
        )  # (B, n_classes)
        probs = self.sigmoid(logit_sum / len(self.ensemble_heads))

        if single:
            return probs.squeeze(0), pre_mil.squeeze(0)
        return probs, pre_mil
