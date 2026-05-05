"""BERT-based encoder for TCR amino-acid sequences.

Converts raw TCR amino-acid sequences into fixed-size embedding tensors
that are consumed by :class:`~mil_ton.models.tcr.model.BertTCRModel`.

TCR length handling
-------------------
TCR beta-chain CDR3 sequences vary in length (typically 10–25 amino
acids).  This module handles variable lengths in two stages:

1. **Sequence truncation** – sequences longer than ``max_tcr_len`` are
   truncated to the first ``max_tcr_len`` residues *before* BERT encoding.
2. **Embedding padding / truncation** – after BERT encoding the position
   dimension of the returned tensor is forced to exactly ``max_tcr_len``
   by zero-padding shorter sequences or truncating longer ones.  This
   mirrors the pre-processing step in the original BertTCR pipeline
   (``BERT_embedding.py``).

This two-step strategy ensures that ``BertTCRModel`` always receives
tensors of shape ``(n_tcrs, bert_hidden_size, max_tcr_len)`` regardless
of the raw sequence lengths in the input bag.

Backbone
--------
By default the encoder loads ``Rostlab/prot_bert_bfd`` from the
Hugging Face Hub – a ProtBERT model (1024 hidden dims) trained on UniRef
and BFD protein databases.  Any ``transformers``-compatible encoder-only
protein language model can be substituted via the ``model_name`` argument;
set ``bert_hidden_size`` in :class:`~mil_ton.models.tcr.model.BertTCRModel`
to match.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default protein BERT backbone used for TCR embedding.
_DEFAULT_BACKBONE = "Rostlab/prot_bert_bfd"


def _format_sequences_for_prot_bert(sequences: List[str]) -> List[str]:
    """Insert spaces between residues as required by ProtBERT tokenizers."""
    return [" ".join(list(seq.upper())) for seq in sequences]


def encode_tcrs(
    sequences: List[str],
    tokenizer,
    bert_model: torch.nn.Module,
    max_tcr_len: int = 24,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Encode a list of TCR amino-acid sequences into BERT embedding tensors.

    Parameters
    ----------
    sequences:
        Amino-acid strings, e.g. ``["CASSLDRGTEAFF", "CASGDRGTEAFF"]``.
        Sequences longer than ``max_tcr_len`` are truncated before
        tokenisation; resulting embeddings shorter than ``max_tcr_len``
        are zero-padded.
    tokenizer:
        A Hugging Face tokenizer compatible with ``bert_model``.
    bert_model:
        A Hugging Face encoder (e.g. ``BertModel``) that returns a
        ``BaseModelOutput`` with ``last_hidden_state`` of shape
        ``(batch, seq_len, hidden_size)``.
    max_tcr_len:
        Target position dimension after padding / truncation.
    device:
        Device for tensor operations.  Defaults to the device of the
        first parameter in ``bert_model``.

    Returns
    -------
    torch.Tensor
        Shape ``(n_tcrs, hidden_size, max_tcr_len)``.
    """
    if device is None:
        try:
            device = next(bert_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Truncate sequences before tokenisation to avoid unnecessary computation.
    truncated = [seq[:max_tcr_len] for seq in sequences]
    formatted = _format_sequences_for_prot_bert(truncated)

    encoding = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tcr_len + 2,  # +2 for [CLS] / [SEP]
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    # last_hidden_state: (n_tcrs, token_seq_len, hidden_size)
    hidden = outputs.last_hidden_state

    # Strip [CLS] and [SEP] tokens (first and last positions).
    hidden = hidden[:, 1:-1, :]  # (n_tcrs, seq_len, hidden_size)

    # Transpose to (n_tcrs, hidden_size, seq_len).
    hidden = hidden.transpose(1, 2)  # (n_tcrs, hidden_size, seq_len)

    # Pad or truncate the position dimension to exactly max_tcr_len.
    seq_len = hidden.shape[-1]
    if seq_len > max_tcr_len:
        hidden = hidden[:, :, :max_tcr_len]
    elif seq_len < max_tcr_len:
        pad = (0, max_tcr_len - seq_len)
        hidden = F.pad(hidden, pad, mode="constant", value=0.0)

    return hidden  # (n_tcrs, hidden_size, max_tcr_len)


class TCRBertEncoder:
    """Convenience wrapper that loads a protein BERT model from the Hub and
    exposes a single :meth:`encode` method.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.  Defaults to
        ``"Rostlab/prot_bert_bfd"`` (1024-dim, ProtBERT trained on BFD).
        Use ``"Rostlab/prot_bert"`` for the smaller UniRef-only version,
        or any other compatible protein language model.
    max_tcr_len:
        Target length for the position dimension of each TCR embedding.
        Sequences are truncated before encoding; embeddings are padded
        after encoding to reach this length.
    device:
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.  Defaults to
        CUDA if available.

    Examples
    --------
    >>> encoder = TCRBertEncoder()          # downloads weights once
    >>> embeddings = encoder.encode(["CASSLDRGTEAFF", "CASGDSSGANVLTF"])
    >>> embeddings.shape
    torch.Size([2, 1024, 24])
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_BACKBONE,
        max_tcr_len: int = 24,
        device: Optional[str] = None,
    ) -> None:
        try:
            from transformers import AutoTokenizer, BertModel
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for TCRBertEncoder. "
                "Install it with: pip install transformers"
            ) from exc

        self.max_tcr_len = max_tcr_len
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("Loading tokenizer from %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading BERT model from %s", model_name)
        self.bert = BertModel.from_pretrained(model_name).to(self.device)
        self.bert.eval()

        self.hidden_size: int = self.bert.config.hidden_size

    def encode(self, sequences: List[str]) -> torch.Tensor:
        """Encode a list of TCR sequences into BERT embeddings.

        Parameters
        ----------
        sequences:
            Amino-acid strings of any length.  Longer sequences are
            truncated to ``max_tcr_len``; shorter ones are zero-padded
            after encoding.

        Returns
        -------
        torch.Tensor
            Shape ``(n_tcrs, hidden_size, max_tcr_len)``.
        """
        return encode_tcrs(
            sequences,
            self.tokenizer,
            self.bert,
            max_tcr_len=self.max_tcr_len,
            device=self.device,
        )
