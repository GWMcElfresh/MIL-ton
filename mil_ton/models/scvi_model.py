"""scVI integration for MIL-ton."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import anndata as ad

from mil_ton.config import ScVIConfig

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

try:
    import scvi as _scvi_lib
except (ImportError, AttributeError):
    _scvi_lib = None  # type: ignore[assignment]


def _require_scvi() -> Any:
    """Return the scvi module or raise an informative ImportError."""
    if _scvi_lib is None:
        raise ImportError(
            "scvi-tools is required for scVI training. "
            "Install it with: pip install scvi-tools"
        )
    return _scvi_lib


def train_scvi(
    adata: ad.AnnData,
    config: ScVIConfig,
) -> tuple[ad.AnnData, Any]:
    """Set up and train a scVI model, storing latent representation in adata.

    Parameters
    ----------
    adata:
        Raw count AnnData (cells × genes).
    config:
        scVI training configuration.

    Returns
    -------
    tuple[ad.AnnData, Any]
        Updated AnnData with ``adata.obsm["X_scVI"]`` and the trained model.
    """
    scvi = _require_scvi()

    setup_kwargs: dict[str, Any] = {}
    if config.batch_key is not None:
        setup_kwargs["batch_key"] = config.batch_key

    logger.info(
        "Setting up scVI model (n_latent=%d, batch_key=%s)",
        config.n_latent,
        config.batch_key,
    )
    scvi.model.SCVI.setup_anndata(adata, **setup_kwargs)
    model = scvi.model.SCVI(adata, n_latent=config.n_latent)

    train_kwargs: dict[str, Any] = {"max_epochs": config.max_epochs}
    if config.early_stopping:
        train_kwargs["early_stopping"] = True

    logger.info("Training scVI model …")
    model.train(**train_kwargs)

    adata.obsm["X_scVI"] = model.get_latent_representation()
    logger.info("Stored latent representation in adata.obsm['X_scVI']")
    return adata, model


def save_scvi(model: Any, path: "str | Path") -> None:
    """Save a trained scVI model to disk.

    Parameters
    ----------
    model:
        Trained scVI model instance.
    path:
        Directory path to save the model.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path), overwrite=True)
    logger.info("Saved scVI model to %s", path)


def load_scvi(adata: ad.AnnData, path: "str | Path") -> Any:
    """Load a saved scVI model and populate adata with its latent representation.

    Parameters
    ----------
    adata:
        AnnData used to condition the loaded model.
    path:
        Directory path where the model was saved.

    Returns
    -------
    Any
        Loaded scVI model instance.
    """
    scvi = _require_scvi()
    path = Path(path)
    model = scvi.model.SCVI.load(str(path), adata=adata)
    adata.obsm["X_scVI"] = model.get_latent_representation()
    logger.info("Loaded scVI model from %s", path)
    return model
