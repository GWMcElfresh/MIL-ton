import types

import pytest

import mil_ton.models.scvi_model as scvi_module
from mil_ton.models.scvi_model import _require_scvi


def test_require_scvi_raises_when_unavailable(monkeypatch):
    """_require_scvi() should raise ImportError when _scvi_lib is None."""
    monkeypatch.setattr(scvi_module, "_scvi_lib", None)
    with pytest.raises(ImportError, match="scvi-tools is required"):
        _require_scvi()


def test_require_scvi_returns_module_when_available(monkeypatch):
    """_require_scvi() should return the scvi module when it is present."""
    fake_scvi = types.ModuleType("scvi")
    monkeypatch.setattr(scvi_module, "_scvi_lib", fake_scvi)
    result = _require_scvi()
    assert result is fake_scvi
