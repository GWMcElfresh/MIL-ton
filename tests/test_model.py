import pytest
import torch

from mil_ton.models.mil_model import AttentionPooling, CellEncoder, MILModel


def test_cell_encoder_forward():
    """CellEncoder should map (n_cells, input_dim) -> (n_cells, encoder_dims[-1])."""
    encoder = CellEncoder(input_dim=30, encoder_dims=[64, 32], dropout=0.0)
    encoder.eval()
    x = torch.randn(10, 30)
    out = encoder(x)
    assert out.shape == (10, 32), f"Expected (10, 32), got {out.shape}"


def test_attention_pooling_forward():
    """AttentionPooling should pool (n_cells, dim) -> (dim,) with cell weights."""
    pooling = AttentionPooling(input_dim=128, attention_dim=32, n_heads=1)
    h = torch.randn(10, 128)
    pooled, weights = pooling(h)
    assert pooled.shape == (128,), f"Expected (128,), got {pooled.shape}"
    assert weights.shape == (10,), f"Expected (10,), got {weights.shape}"
    assert abs(weights.sum().item() - 1.0) < 1e-5, "Attention weights should sum to 1"


def test_mil_model_classification():
    """MILModel binary classification: single bag forward."""
    model = MILModel(
        input_dim=30,
        encoder_dims=[64, 32],
        attention_dim=16,
        n_classes=1,
        task="classification",
        dropout=0.0,
        n_heads=1,
    )
    model.eval()
    x = torch.randn(20, 30)
    logits, attn = model(x)
    assert logits.shape == (1,), f"Expected (1,), got {logits.shape}"
    assert attn.shape == (20,), f"Expected (20,), got {attn.shape}"


def test_mil_model_regression():
    """MILModel regression: single bag forward."""
    model = MILModel(
        input_dim=30,
        encoder_dims=[64, 32],
        attention_dim=16,
        n_classes=1,
        task="regression",
        dropout=0.0,
    )
    model.eval()
    x = torch.randn(20, 30)
    logits, attn = model(x)
    assert logits.shape == (1,), f"Expected (1,), got {logits.shape}"
    assert attn.shape == (20,), f"Expected (20,), got {attn.shape}"


def test_mil_model_multiclass():
    """MILModel multiclass: single bag forward with 3 classes."""
    model = MILModel(
        input_dim=30,
        encoder_dims=[64, 32],
        attention_dim=16,
        n_classes=3,
        task="classification",
        dropout=0.0,
    )
    model.eval()
    x = torch.randn(20, 30)
    logits, attn = model(x)
    assert logits.shape == (3,), f"Expected (3,), got {logits.shape}"
    assert attn.shape == (20,), f"Expected (20,), got {attn.shape}"
