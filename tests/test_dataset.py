import numpy as np
import pytest
import torch

from mil_ton.training.dataset import DonorDataset, split_donors


def test_donor_dataset_basic(synthetic_adata):
    """DonorDataset should have 4 donors and return correct shapes."""
    ds = DonorDataset(
        synthetic_adata,
        donor_col="donor_id",
        label_cols=["disease_status"],
        cells_per_donor=20,
        task="classification",
        seed=0,
    )
    assert len(ds) == 4, f"Expected 4 donors, got {len(ds)}"

    X, y = ds[0]
    assert X.shape == (20, 30), f"Expected (20, 30), got {X.shape}"
    assert X.dtype == torch.float32
    assert y.dtype == torch.int64


def test_donor_dataset_sampling(synthetic_adata):
    """Sampled cell count should equal cells_per_donor."""
    cells_per_donor = 15
    ds = DonorDataset(
        synthetic_adata,
        donor_col="donor_id",
        label_cols=["disease_status"],
        cells_per_donor=cells_per_donor,
        task="classification",
        seed=0,
    )
    X, _ = ds[0]
    assert X.shape[0] == cells_per_donor


def test_split_donors():
    """split_donors should produce non-overlapping subsets that cover all donors."""
    donors = [f"d{i}" for i in range(20)]
    train, val, test = split_donors(donors, train_frac=0.7, val_frac=0.15, seed=42)

    assert set(train) | set(val) | set(test) == set(donors)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(train) >= 1
    assert len(val) >= 1
    assert len(test) >= 1
