import pytest
import torch
from torch.utils.data import DataLoader

from mil_ton.models.mil_model import MILModel
from mil_ton.training.dataset import DonorDataset
from mil_ton.training.trainer import Trainer
from mil_ton.config import TrainingConfig


def _tiny_model() -> MILModel:
    return MILModel(
        input_dim=30,
        encoder_dims=[16, 8],
        attention_dim=8,
        n_classes=1,
        task="classification",
        dropout=0.0,
        n_heads=1,
    )


def _make_loader(synthetic_adata, seed: int = 0) -> DataLoader:
    ds = DonorDataset(
        synthetic_adata,
        donor_col="donor_id",
        label_cols=["disease_status"],
        cells_per_donor=20,
        task="classification",
        seed=seed,
    )
    return DataLoader(ds, batch_size=4)


def test_trainer_classification_runs(synthetic_adata, tmp_path):
    """Trainer should complete one epoch and return a history dict."""
    loader = _make_loader(synthetic_adata)
    config = TrainingConfig(epochs=1, lr=1e-3)
    trainer = Trainer(
        model=_tiny_model(),
        config=config,
        task="classification",
        n_classes=1,
        output_dir=tmp_path,
        device="cpu",
    )
    history = trainer.train(train_loader=loader, val_loader=loader)

    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 1
    assert isinstance(history["train_loss"][0], float)


def test_trainer_saves_checkpoint(synthetic_adata, tmp_path):
    """Trainer should write model.pt to output_dir after training."""
    loader = _make_loader(synthetic_adata, seed=1)
    config = TrainingConfig(epochs=1, lr=1e-3)
    trainer = Trainer(
        model=_tiny_model(),
        config=config,
        task="classification",
        n_classes=1,
        output_dir=tmp_path,
        device="cpu",
    )
    trainer.train(train_loader=loader, val_loader=loader)

    checkpoint = tmp_path / "model.pt"
    assert checkpoint.exists(), "model.pt was not written after training"


def test_trainer_regression_runs(synthetic_adata, tmp_path):
    """Trainer should handle regression task for one epoch."""
    ds = DonorDataset(
        synthetic_adata,
        donor_col="donor_id",
        label_cols=["disease_status"],
        cells_per_donor=20,
        task="regression",
        seed=0,
    )
    loader = DataLoader(ds, batch_size=4)

    model = MILModel(
        input_dim=30,
        encoder_dims=[16, 8],
        attention_dim=8,
        n_classes=1,
        task="regression",
        dropout=0.0,
    )
    config = TrainingConfig(epochs=1, lr=1e-3)
    trainer = Trainer(
        model=model,
        config=config,
        task="regression",
        n_classes=1,
        output_dir=tmp_path,
        device="cpu",
    )
    history = trainer.train(train_loader=loader, val_loader=loader)
    assert len(history["train_loss"]) == 1
