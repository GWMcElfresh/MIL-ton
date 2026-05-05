"""Training and evaluation loop for :class:`~mil_ton.models.tcr.BertTCRModel`.

The API is intentionally similar to :class:`~mil_ton.training.trainer.Trainer`
so the two submodules feel consistent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TCRTrainer:
    """Training loop for :class:`~mil_ton.models.tcr.BertTCRModel`.

    Parameters
    ----------
    model:
        Instantiated :class:`~mil_ton.models.tcr.model.BertTCRModel`.
    epochs:
        Number of training epochs.
    lr:
        Learning rate for the Adam optimiser.
    weight_decay:
        L2 regularisation coefficient.
    output_dir:
        Directory for checkpoints and training history.  Created if needed.
    device:
        Torch device string.  Defaults to CUDA when available.
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        output_dir: "str | Path" = ".",
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # BCELoss is correct here because the model outputs per-class sigmoid
        # probabilities (not raw logits).  CrossEntropyLoss internally applies
        # log_softmax and therefore expects unnormalised logits; feeding it
        # post-sigmoid values produces incorrect gradients.
        self.loss_fn = nn.BCELoss()
        self._best_val_auc = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass; returns ``(loss, probs, pre_mil)``."""
        probs, pre_mil = self.model(x)
        # probs: (batch, n_classes) or (n_classes,) for single bag
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)

        # BCELoss requires targets with the same shape as probs.  Convert
        # integer class indices to one-hot float vectors so that each output
        # neuron (sigmoid probability) is compared against a 0/1 target.
        n_classes = probs.shape[-1]
        y_onehot = torch.zeros(
            (y.shape[0], n_classes), device=probs.device, dtype=probs.dtype
        )
        y_onehot.scatter_(1, y.long().unsqueeze(1), 1.0)

        loss = self.loss_fn(probs, y_onehot)
        return loss, probs, pre_mil

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Each batch from ``train_loader`` must yield ``(X, y)`` where:

        * ``X`` is a ``torch.Tensor`` of shape
          ``(batch, n_tcrs, bert_hidden_size, max_tcr_len)`` or
          ``(n_tcrs, bert_hidden_size, max_tcr_len)`` for batch_size=1.
        * ``y`` is a ``torch.LongTensor`` of class indices.

        Parameters
        ----------
        train_loader:
            DataLoader over training bags.
        val_loader:
            DataLoader over validation bags.

        Returns
        -------
        dict
            History with keys ``train_loss``, ``val_loss``,
            ``val_accuracy``, and ``val_auc``.
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auc": [],
        }

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics.get("accuracy", float("nan")))
            history["val_auc"].append(val_metrics.get("auc", float("nan")))

            if val_metrics.get("auc", 0.0) > self._best_val_auc:
                self._best_val_auc = val_metrics.get("auc", 0.0)
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "tcr_model.pt",
                )

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                    "val_acc=%.3f  val_auc=%.3f",
                    epoch,
                    self.epochs,
                    train_loss,
                    val_metrics["loss"],
                    val_metrics.get("accuracy", float("nan")),
                    val_metrics.get("auc", float("nan")),
                )

        with (self.output_dir / "tcr_history.json").open("w") as fh:
            json.dump(history, fh, indent=2)

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            loss, _, _ = self._step(X_batch, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a DataLoader.

        Parameters
        ----------
        loader:
            DataLoader to evaluate.  Must yield ``(X, y)`` pairs with the
            same shapes as ``train_loader`` in :meth:`train`.

        Returns
        -------
        dict
            Dictionary with keys:

            * ``loss`` – mean cross-entropy loss.
            * ``accuracy`` – fraction of correctly classified bags.
            * ``auc`` – ROC AUC (binary) or macro-OVR AUC (multiclass).
            * ``pre_mil_embeddings`` – list of per-bag pre-MIL tensors
              (one ``torch.Tensor`` per batch element).
        """
        self.model.eval()

        all_probs: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_pre_mil: List[torch.Tensor] = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss, probs, pre_mil = self._step(X_batch, y_batch)
                total_loss += loss.item()
                n_batches += 1

                all_probs.append(probs.cpu())
                all_labels.append(y_batch.cpu())
                all_pre_mil.append(pre_mil.cpu())

        if not all_probs:
            return {"loss": float("nan")}

        probs_all = torch.cat(all_probs, dim=0)   # (N, n_classes)
        labels_all = torch.cat(all_labels, dim=0)  # (N,)

        preds = probs_all.argmax(dim=-1)
        accuracy = (preds == labels_all).float().mean().item()

        auc = _compute_auc(probs_all, labels_all)

        return {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": accuracy,
            "auc": auc,
            "pre_mil_embeddings": all_pre_mil,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def train_berttcr(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    output_dir: "str | Path" = ".",
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Convenience wrapper: create a :class:`TCRTrainer` and run training.

    Parameters
    ----------
    model:
        :class:`~mil_ton.models.tcr.model.BertTCRModel` instance.
    train_loader / val_loader:
        DataLoaders yielding ``(X, y)`` bags.
    epochs, lr, weight_decay:
        Optimisation hyper-parameters.
    output_dir:
        Where to save the best checkpoint and history JSON.
    device:
        Torch device string.

    Returns
    -------
    dict
        Training history.
    """
    trainer = TCRTrainer(
        model=model,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        output_dir=output_dir,
        device=device,
    )
    return trainer.train(train_loader, val_loader)


def evaluate_berttcr(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate a trained :class:`~mil_ton.models.tcr.model.BertTCRModel`.

    Parameters
    ----------
    model:
        Trained model (can be in eval or train mode; will be set to eval).
    loader:
        DataLoader yielding ``(X, y)`` bags.
    device:
        Torch device string.

    Returns
    -------
    dict
        Dictionary with ``loss``, ``accuracy``, ``auc``, and
        ``pre_mil_embeddings``.
    """
    trainer = TCRTrainer(model=model, device=device)
    return trainer.evaluate(loader)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute ROC AUC; returns ``nan`` if sklearn is unavailable or if
    only one class is present in ``labels``."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")

    n_classes = probs.shape[-1]
    y = labels.numpy()
    p = probs.numpy()

    if len(set(y.tolist())) < 2:
        return float("nan")

    try:
        if n_classes == 2:
            return float(roc_auc_score(y, p[:, 1]))
        return float(
            roc_auc_score(y, p, multi_class="ovr", average="macro")
        )
    except Exception:
        return float("nan")
