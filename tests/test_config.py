import pytest

from mil_ton.config import (
    Config,
    DataConfig,
    MILConfig,
    TrainingConfig,
    load_config,
    save_config,
)


def test_config_roundtrip(tmp_path):
    """save_config followed by load_config should reproduce all field values."""
    original = Config(
        data=DataConfig(donor_col="subject", label_cols=["status"], task="regression"),
        mil=MILConfig(encoder_dims=[64], attention_dim=16, dropout=0.1, n_heads=2),
        training=TrainingConfig(epochs=5, lr=0.01, seed=7),
    )
    yaml_path = tmp_path / "config.yaml"
    save_config(original, yaml_path)

    loaded = load_config(yaml_path)

    assert loaded.data.donor_col == original.data.donor_col
    assert loaded.data.label_cols == original.data.label_cols
    assert loaded.data.task == original.data.task
    assert loaded.mil.encoder_dims == original.mil.encoder_dims
    assert loaded.mil.n_heads == original.mil.n_heads
    assert loaded.mil.dropout == pytest.approx(original.mil.dropout)
    assert loaded.training.epochs == original.training.epochs
    assert loaded.training.seed == original.training.seed
    assert loaded.training.lr == pytest.approx(original.training.lr)


def test_load_config_defaults(tmp_path):
    """Minimal YAML should fill in defaults for unspecified fields."""
    yaml_path = tmp_path / "minimal.yaml"
    yaml_path.write_text("data:\n  donor_col: patient\n")

    cfg = load_config(yaml_path)

    assert cfg.data.donor_col == "patient"
    assert cfg.data.task == "classification"     # default
    assert cfg.mil.attention_dim == 64            # default
    assert cfg.training.epochs == 100             # default


def test_load_config_empty_file(tmp_path):
    """An empty YAML file should produce a Config with all defaults."""
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("")

    cfg = load_config(yaml_path)

    assert cfg.data.donor_col == "donor_id"
    assert cfg.training.seed == 42
