"""Donor-level prediction and export utilities."""

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


def predict_donors(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Run inference over a DataLoader and collect per-donor results.

    Parameters
    ----------
    model:
        Trained MIL model.
    loader:
        DataLoader whose dataset is a :class:`~mil_ton.training.dataset.DonorDataset`.
    task:
        ``"classification"`` or ``"regression"``.
    device:
        Device to run inference on.  Defaults to the model's current device.

    Returns
    -------
    dict
        Dictionary with keys ``"donors"``, ``"predictions"``,
        ``"probabilities"`` (classification only), and ``"labels"``.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    dataset = loader.dataset  # type: ignore[attr-defined]

    donors_out = []
    preds_out = []
    probs_out = []
    labels_out = []

    donor_iter = iter(dataset.donors)

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            batch_size = X_batch.shape[0]
            for i in range(batch_size):
                logits, _ = model(X_batch[i])
                donor_id = next(donor_iter)
                donors_out.append(donor_id)
                labels_out.append(y_batch[i].cpu().numpy())

                if task == "regression":
                    pred = logits.squeeze(-1).cpu().numpy()
                    preds_out.append(float(pred))
                    probs_out.append(float(pred))
                elif logits.shape[-1] == 1 or logits.dim() == 0:
                    prob = torch.sigmoid(logits).squeeze(-1).cpu().item()
                    preds_out.append(int(prob >= 0.5))
                    probs_out.append(prob)
                else:
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    pred_class = int(np.argmax(probs))
                    preds_out.append(pred_class)
                    probs_out.append(probs.tolist())

    return {
        "donors": donors_out,
        "predictions": preds_out,
        "probabilities": probs_out,
        "labels": [x.tolist() if hasattr(x, "tolist") else x for x in labels_out],
    }


def export_predictions(predictions: Dict[str, object], output_path: "str | Path") -> None:
    """Save predictions dictionary to a CSV file.

    Parameters
    ----------
    predictions:
        Output of :func:`predict_donors`.
    output_path:
        Destination CSV path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "donor_id": predictions["donors"],
            "prediction": predictions["predictions"],
            "probability": predictions["probabilities"],
            "label": predictions["labels"],
        }
    )
    df.to_csv(output_path, index=False)
    logger.info("Saved predictions to %s", output_path)
