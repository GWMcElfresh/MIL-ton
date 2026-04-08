"""Configuration dataclasses and YAML I/O for MIL-ton."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ScVIConfig:
    """Configuration for scVI model training."""

    n_latent: int = 30
    max_epochs: int = 400
    batch_key: Optional[str] = None
    early_stopping: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    donor_col: str = "donor_id"
    label_cols: list = field(default_factory=lambda: ["label"])
    task: str = "classification"  # "classification" or "regression"
    cells_per_donor: int = 5000
    min_cells: int = 50
    varfeats_path: Optional[str] = None


@dataclass
class MILConfig:
    """Configuration for the MIL model architecture."""

    encoder_dims: list = field(default_factory=lambda: [256, 128])
    attention_dim: int = 64
    dropout: float = 0.2
    n_heads: int = 1


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-4
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42


@dataclass
class Config:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    scvi: ScVIConfig = field(default_factory=ScVIConfig)
    mil: MILConfig = field(default_factory=MILConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _nested_update(dataclass_instance, data: dict):
    """Recursively update a dataclass from a dict."""
    for key, value in data.items():
        if hasattr(dataclass_instance, key):
            attr = getattr(dataclass_instance, key)
            if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
                _nested_update(attr, value)
            else:
                setattr(dataclass_instance, key, value)


def load_config(path: "str | Path") -> Config:
    """Load a Config from a YAML file, filling in defaults for missing fields.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Populated configuration object.
    """
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    config = Config()
    _nested_update(config, raw)
    return config


def save_config(config: Config, path: "str | Path") -> None:
    """Save a Config to a YAML file.

    Parameters
    ----------
    config:
        Configuration object to serialise.
    path:
        Destination path for the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(asdict(config), fh, default_flow_style=False)
