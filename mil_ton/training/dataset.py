"""PyTorch dataset for donor-level MIL."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import anndata as ad

logger = logging.getLogger(__name__)


class DonorDataset(Dataset):
    """Dataset that yields per-donor bags of cell embeddings and labels.

    Parameters
    ----------
    adata:
        AnnData object with ``adata.obsm["X_scVI"]`` populated.
    donor_col:
        Column in ``adata.obs`` identifying donors.
    label_cols:
        Label column(s) in ``adata.obs``.  Labels are assumed constant per donor.
    cells_per_donor:
        Number of cells to sample per donor per call to ``__getitem__``.
    task:
        ``"classification"`` or ``"regression"``.
    cluster_col:
        Unused; reserved for future stratified sampling.
    seed:
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        donor_col: str,
        label_cols: List[str],
        cells_per_donor: int,
        task: str,
        cluster_col: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.donors: List[str] = []
        self.labels: List[np.ndarray] = []
        self.donor_indices: Dict[str, List[int]] = {}

        # O(1) barcode → integer-position lookup
        self.obs_index_map: Dict[str, int] = {
            bc: i for i, bc in enumerate(adata.obs_names)
        }

        for donor_id, group in adata.obs.groupby(donor_col):
            indices = [self.obs_index_map[bc] for bc in group.index]
            self.donor_indices[str(donor_id)] = indices
            label_vals = group[label_cols].iloc[0].values.astype(float)
            self.donors.append(str(donor_id))
            self.labels.append(label_vals)

        self.X = adata.obsm["X_scVI"]  # (n_cells, latent_dim) numpy array
        self.cells_per_donor = cells_per_donor
        self.task = task
        self.label_cols = label_cols
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.donors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        donor = self.donors[idx]
        indices = self.donor_indices[donor]

        # Sample cells
        if len(indices) >= self.cells_per_donor:
            chosen = self.rng.choice(indices, size=self.cells_per_donor, replace=False)
        else:
            chosen = self.rng.choice(indices, size=self.cells_per_donor, replace=True)

        X = torch.tensor(self.X[chosen], dtype=torch.float32)

        label = self.labels[idx]
        if self.task == "regression":
            y = torch.tensor(label, dtype=torch.float32)
        elif len(label) == 1 and self.task == "classification":
            y = torch.tensor(int(label[0]), dtype=torch.long)
        else:
            y = torch.tensor(label.astype(int), dtype=torch.long)

        return X, y


def split_donors(
    donors: List[str],
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split a donor list into train / validation / test subsets.

    Parameters
    ----------
    donors:
        List of donor identifiers.
    train_frac:
        Fraction of donors for training.
    val_frac:
        Fraction of donors for validation.  The remainder goes to test.
    seed:
        Random seed.

    Returns
    -------
    tuple[list, list, list]
        ``(train_donors, val_donors, test_donors)``.
    """
    rng = np.random.default_rng(seed)
    donors = list(donors)
    rng.shuffle(donors)

    n = len(donors)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    # Ensure at least 1 donor in test if possible
    n_test = n - n_train - n_val
    if n_test < 1 and n >= 3:
        n_val = max(1, n_val - 1)
        n_test = n - n_train - n_val

    train = donors[:n_train]
    val = donors[n_train : n_train + n_val]
    test = donors[n_train + n_val :]
    return train, val, test
