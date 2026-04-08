"""Attention-weight interpretation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_attention_weights(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Collect per-cell attention weights for every donor in the DataLoader.

    Parameters
    ----------
    model:
        Trained MIL model.
    loader:
        DataLoader wrapping a :class:`~mil_ton.training.dataset.DonorDataset`.
    device:
        Inference device.

    Returns
    -------
    dict
        Mapping of ``donor_id -> attention_weights`` (1-D float array, length
        equals ``cells_per_donor``).
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    dataset = loader.dataset  # type: ignore[attr-defined]
    donor_iter = iter(dataset.donors)
    attention_dict: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            batch_size = X_batch.shape[0]
            for i in range(batch_size):
                _, attn = model(X_batch[i])
                donor_id = next(donor_iter)
                attention_dict[donor_id] = attn.cpu().numpy()

    return attention_dict


def map_attention_to_adata(
    adata,
    attention_dict: Dict[str, np.ndarray],
    donor_col: str,
    cell_sample_indices: Optional[Dict[str, np.ndarray]] = None,
    key: str = "attention_weight",
) -> None:
    """Write mean per-cell attention weights into ``adata.obs[key]``.

    For cells that appear in multiple samplings, the mean attention is stored.
    Cells not present in ``attention_dict`` receive ``NaN``.

    Parameters
    ----------
    adata:
        AnnData object to annotate in-place.
    attention_dict:
        Mapping returned by :func:`get_attention_weights`.
    donor_col:
        Column identifying donor membership in ``adata.obs``.
    cell_sample_indices:
        Optional mapping of donor_id -> integer positions of the sampled cells.
        When provided, attention weights are aligned to exact cell positions.
    key:
        Column name to add to ``adata.obs``.
    """
    weights = np.full(adata.n_obs, np.nan, dtype=float)

    obs_index_map = {bc: i for i, bc in enumerate(adata.obs_names)}

    for donor_id, attn in attention_dict.items():
        mask = adata.obs[donor_col] == donor_id
        donor_indices = [obs_index_map[bc] for bc in adata.obs_names[mask]]

        if cell_sample_indices is not None and donor_id in cell_sample_indices:
            sampled = cell_sample_indices[donor_id]
            # Assign to sampled positions; accumulate and average if overlapping
            donor_idx_pos = {idx: j for j, idx in enumerate(donor_indices)}
            counts = np.zeros(len(donor_indices), dtype=float)
            acc = np.zeros(len(donor_indices), dtype=float)
            for k, s_idx in enumerate(sampled):
                local = donor_idx_pos.get(s_idx)
                if local is not None:
                    acc[local] += attn[k]
                    counts[local] += 1
            nonzero = counts > 0
            acc[nonzero] /= counts[nonzero]
            for j, global_idx in enumerate(donor_indices):
                weights[global_idx] = acc[j]
        else:
            # Assign mean attention uniformly across donor cells
            mean_attn = float(np.nanmean(attn))
            for global_idx in donor_indices:
                weights[global_idx] = mean_attn

    adata.obs[key] = weights


def export_top_attended_cells(
    adata,
    attention_dict: Dict[str, np.ndarray],
    donor_col: str,
    n_top: int = 100,
    output_path: Optional["str | Path"] = None,
) -> pd.DataFrame:
    """Return (and optionally save) the top-attended cells per donor.

    Parameters
    ----------
    adata:
        AnnData object.
    attention_dict:
        Mapping of donor_id -> attention weights.
    donor_col:
        Donor column in ``adata.obs``.
    n_top:
        Maximum number of top cells to return per donor.
    output_path:
        If provided, the resulting DataFrame is saved as a CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``donor_id``, ``cell_barcode``,
        ``attention_weight``, and all ``adata.obs`` columns.
    """
    rows = []
    obs_index_map = {bc: i for i, bc in enumerate(adata.obs_names)}

    for donor_id, attn in attention_dict.items():
        mask = adata.obs[donor_col] == donor_id
        donor_barcodes = list(adata.obs_names[mask])

        if len(donor_barcodes) == 0:
            continue

        # Attention weights are computed over *randomly sampled* cells whose
        # exact identities are not stored here.  We therefore report the mean
        # attention score for the donor and list all donor cells ranked by that
        # scalar.  For cell-level interpretation, pass ``cell_sample_indices``
        # to :func:`map_attention_to_adata` and annotate ``adata.obs`` before
        # calling this function.
        if len(attn) == len(donor_barcodes):
            # Exact 1-to-1 mapping is possible (e.g. when cells_per_donor ≥
            # total donor cell count so all cells were included).
            top_k = min(n_top, len(donor_barcodes))
            top_local = np.argsort(attn)[::-1][:top_k]
            for local_idx in top_local:
                bc = donor_barcodes[local_idx]
                global_idx = obs_index_map[bc]
                row = {
                    "donor_id": donor_id,
                    "cell_barcode": bc,
                    "attention_weight": float(attn[local_idx]),
                }
                row.update(adata.obs.iloc[global_idx].to_dict())
                rows.append(row)
        else:
            # Sampled bag size differs from donor cell count; use mean attention
            # as a donor-level score and list top-N cells (ordered by original
            # index as a stable tie-breaker).
            mean_attn = float(np.mean(attn))
            logger.debug(
                "Donor %s: attention vector length (%d) != donor cell count (%d); "
                "reporting mean attention score for all cells.",
                donor_id,
                len(attn),
                len(donor_barcodes),
            )
            top_k = min(n_top, len(donor_barcodes))
            for bc in donor_barcodes[:top_k]:
                global_idx = obs_index_map[bc]
                row = {
                    "donor_id": donor_id,
                    "cell_barcode": bc,
                    "attention_weight": mean_attn,
                }
                row.update(adata.obs.iloc[global_idx].to_dict())
                rows.append(row)

    df = pd.DataFrame(rows)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved top-attended cells to %s", output_path)

    return df
