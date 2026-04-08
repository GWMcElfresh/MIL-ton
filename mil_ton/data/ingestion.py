"""Data ingestion utilities for MIL-ton."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List

import anndata as ad
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def load_data(
    input_dir: "str | Path",
    donor_col: str,
    label_cols: List[str],
    varfeats_path: Optional[str] = None,
) -> ad.AnnData:
    """Load single-cell RNA-seq data from a directory.

    Attempts to load ``GEX.h5`` as a 10x HDF5 file first; if not found, falls
    back to any ``.h5ad`` file in ``input_dir``.  If a ``metadata.tsv`` file
    exists in ``input_dir``, its columns are merged into ``adata.obs`` on the
    cell-barcode index.  Optionally, a CSV of variable features can be provided
    to subset genes.

    Parameters
    ----------
    input_dir:
        Directory containing the data files.
    donor_col:
        Column name in ``adata.obs`` that identifies donors.
    label_cols:
        Column names in ``adata.obs`` that contain prediction labels.
    varfeats_path:
        Optional path to a single-column CSV of gene names to subset to.

    Returns
    -------
    ad.AnnData
        Loaded and validated AnnData object.

    Raises
    ------
    FileNotFoundError
        If no recognised data file can be found in ``input_dir``.
    ValueError
        If required columns are missing from ``adata.obs``.
    """
    input_dir = Path(input_dir)

    # --- Load expression matrix ---
    gex_path = input_dir / "GEX.h5"
    if gex_path.exists():
        logger.info("Loading 10x HDF5 file: %s", gex_path)
        try:
            adata = sc.read_10x_h5(str(gex_path))
        except (OSError, ValueError, KeyError) as exc:  # pragma: no cover
            logger.warning("sc.read_10x_h5 failed (%s); trying read_h5ad", exc)
            adata = ad.read_h5ad(str(gex_path))
    else:
        h5ad_files = sorted(input_dir.glob("*.h5ad"))
        if not h5ad_files:
            raise FileNotFoundError(
                f"No GEX.h5 or .h5ad file found in {input_dir}"
            )
        logger.info("Loading h5ad file: %s", h5ad_files[0])
        adata = ad.read_h5ad(str(h5ad_files[0]))

    logger.info("Loaded AnnData: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # --- Merge metadata ---
    metadata_path = input_dir / "metadata.tsv"
    if metadata_path.exists():
        logger.info("Merging metadata from %s", metadata_path)
        meta = pd.read_csv(metadata_path, sep="\t", index_col=0)
        # Only add columns not already present
        new_cols = [c for c in meta.columns if c not in adata.obs.columns]
        if new_cols:
            adata.obs = adata.obs.join(meta[new_cols], how="left")

    # --- Subset to variable features ---
    if varfeats_path is not None:
        varfeats_path = Path(varfeats_path)
        logger.info("Subsetting genes using %s", varfeats_path)
        genes = pd.read_csv(varfeats_path, header=None)[0].tolist()
        keep = [g for g in genes if g in adata.var_names]
        logger.info("Keeping %d / %d requested genes", len(keep), len(genes))
        adata = adata[:, keep].copy()

    # --- Validate columns ---
    validate_columns(adata, donor_col, label_cols)

    logger.info(
        "Final AnnData: %d cells × %d genes, %d donors",
        adata.n_obs,
        adata.n_vars,
        adata.obs[donor_col].nunique(),
    )
    return adata


def validate_columns(
    adata: ad.AnnData,
    donor_col: str,
    label_cols: List[str],
) -> None:
    """Validate that required columns exist in ``adata.obs``.

    Parameters
    ----------
    adata:
        AnnData object whose ``.obs`` DataFrame will be checked.
    donor_col:
        Expected donor-ID column name.
    label_cols:
        Expected label column name(s).

    Raises
    ------
    ValueError
        If any required column is absent from ``adata.obs``.
    """
    missing: List[str] = []
    if donor_col not in adata.obs.columns:
        missing.append(donor_col)
    for col in label_cols:
        if col not in adata.obs.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Missing columns in adata.obs: {missing}")
