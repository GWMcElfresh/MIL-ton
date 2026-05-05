"""Dataset classes for TCR repertoire MIL training.

Two dataset implementations are provided:

``TCRBagDataset``
    Loads pre-computed BERT embedding tensors from ``.pth`` files (the
    format produced by ``BERT_embedding.py`` in BertTCR).  Each ``.pth``
    file stores a single repertoire bag of shape
    ``(n_tcrs, bert_hidden_size, max_tcr_len)``.

``TCRSequenceDataset``
    Works directly from a ``pandas.DataFrame`` with a ``'TCR'`` column
    and a ``'label'`` column.  Sequences are grouped into bags of fixed
    size ``n_tcrs``; an :class:`~mil_ton.models.tcr.encoder.TCRBertEncoder`
    is called lazily to embed each batch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TCRBagDataset(Dataset):
    """Dataset over pre-computed TCR BERT embedding ``.pth`` files.

    Each file must contain a single ``torch.Tensor`` of shape
    ``(n_tcrs, bert_hidden_size, max_tcr_len)`` – exactly the format
    saved by ``BERT_embedding.py`` from the BertTCR repository.

    Labels are inferred from the filename: files containing
    ``flag_positive`` are labelled ``1``; those containing
    ``flag_negative`` are labelled ``0``.

    Parameters
    ----------
    sample_dir:
        Directory containing the ``.pth`` embedding files.
    flag_positive:
        Substring that identifies *positive* (e.g. cancer patient) files.
    flag_negative:
        Substring that identifies *negative* (e.g. healthy) files.
    transform:
        Optional callable applied to each loaded tensor.

    Raises
    ------
    ValueError
        If a file name matches neither ``flag_positive`` nor
        ``flag_negative``.
    """

    def __init__(
        self,
        sample_dir: Union[str, Path],
        flag_positive: str = "Patient",
        flag_negative: str = "Health",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        sample_dir = Path(sample_dir)
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"sample_dir does not exist: {sample_dir}")

        self.paths: List[Path] = sorted(sample_dir.glob("*.pth"))
        self.labels: List[int] = []
        self.transform = transform

        for p in self.paths:
            name = p.name
            if flag_positive in name:
                self.labels.append(1)
            elif flag_negative in name:
                self.labels.append(0)
            else:
                raise ValueError(
                    f"File '{name}' matches neither flag_positive="
                    f"'{flag_positive}' nor flag_negative='{flag_negative}'."
                )

        if not self.paths:
            logger.warning("No .pth files found in %s", sample_dir)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return ``(embedding_tensor, label)``.

        Returns
        -------
        embedding : torch.Tensor
            Shape ``(n_tcrs, bert_hidden_size, max_tcr_len)``.
        label : int
            Binary label (0 or 1).
        """
        tensor = torch.load(self.paths[idx], weights_only=True)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, self.labels[idx]


class TCRSequenceDataset(Dataset):
    """Dataset that groups raw TCR sequences into fixed-size repertoire bags.

    Each item is a *bag* of ``n_tcrs`` sequences drawn from the pool of
    sequences that share the same ``label_col`` value.  Variable-length
    sequences are handled by :func:`~mil_ton.models.tcr.encoder.encode_tcrs`
    which pads / truncates to ``max_tcr_len`` after BERT encoding.

    Parameters
    ----------
    df:
        DataFrame with at least ``seq_col`` and ``label_col`` columns.
    n_tcrs:
        Number of TCRs to sample per bag.
    seq_col:
        Column containing amino-acid sequences.
    label_col:
        Column containing integer class labels.
    max_tcr_len:
        Maximum TCR length; longer sequences are truncated before
        BERT encoding.
    seed:
        Random seed for reproducible bag sampling.
    """

    def __init__(
        self,
        df,
        n_tcrs: int,
        seq_col: str = "TCR",
        label_col: str = "label",
        max_tcr_len: int = 24,
        seed: int = 42,
    ) -> None:
        import pandas as pd  # local import to avoid hard dep at module level

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        self.n_tcrs = n_tcrs
        self.max_tcr_len = max_tcr_len
        self.rng = np.random.default_rng(seed)

        # Build per-label pools.
        self._pools: dict[int, list[str]] = {}
        for label, group in df.groupby(label_col):
            seqs = group[seq_col].str[:max_tcr_len].tolist()
            self._pools[int(label)] = seqs

        # Each sample is represented by its label; bags are sampled on demand.
        self._labels: List[int] = []
        for label, seqs in self._pools.items():
            n_bags = max(1, len(seqs) // n_tcrs)
            self._labels.extend([label] * n_bags)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Tuple[List[str], int]:
        """Return ``(sequence_list, label)`` for bag ``idx``.

        Returns
        -------
        sequences : list[str]
            A bag of ``n_tcrs`` amino-acid strings (already truncated to
            ``max_tcr_len``).
        label : int
            Class label for this bag.
        """
        label = self._labels[idx]
        pool = self._pools[label]
        chosen = self.rng.choice(len(pool), size=self.n_tcrs, replace=len(pool) < self.n_tcrs)
        return [pool[i] for i in chosen], label
