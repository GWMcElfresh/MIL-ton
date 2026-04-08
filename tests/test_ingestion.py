import pytest
from mil_ton.data.ingestion import validate_columns


def test_validate_donor_col(synthetic_adata):
    """Missing donor_col should raise ValueError."""
    with pytest.raises(ValueError, match="Missing columns"):
        validate_columns(synthetic_adata, "nonexistent_donor", ["disease_status"])


def test_validate_label_cols(synthetic_adata):
    """Missing label column should raise ValueError."""
    with pytest.raises(ValueError, match="Missing columns"):
        validate_columns(synthetic_adata, "donor_id", ["nonexistent_label"])


def test_validate_passes(synthetic_adata):
    """Valid columns should not raise."""
    validate_columns(synthetic_adata, "donor_id", ["disease_status"])
