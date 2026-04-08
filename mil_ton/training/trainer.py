"""MIL training loop with metric tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, R2Score
from torchmetrics.classification import AUROC, Accuracy

from mil_ton.config import TrainingConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for :class:`~mil_ton.models.MILModel`.

    Parameters
    ----------
    model:
        Instantiated MIL model.
    config:
        Training hyper-parameters.
    task:
        ``"classification"`` or ``"regression"``.
    n_classes:
        Number of target classes (1 for binary / regression).
    output_dir:
        Directory for checkpoints and training history.
    device:
        Torch device string (defaults to CUDA if available, else CPU).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        task: str,
        n_classes: int,
        output_dir: "str | Path",
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.task = task
        self.n_classes = n_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.loss_fn = self._build_loss()
        self._best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_loss(self) -> nn.Module:
        if self.task == "regression":
            return nn.MSELoss()
        if self.n_classes == 1:
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def _build_metrics(self):
        """Build torchmetrics metric objects for the current task."""
        if self.task == "regression":
            return {
                "rmse": MeanSquaredError(squared=False).to(self.device),
                "r2": R2Score().to(self.device),
            }
        if self.n_classes == 1:
            return {
                "auroc": AUROC(task="binary").to(self.device),
                "accuracy": Accuracy(task="binary").to(self.device),
            }
        return {
            "auroc": AUROC(task="multiclass", num_classes=self.n_classes).to(self.device),
            "accuracy": Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device),
        }

    def _compute_loss(
        self, logits: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if self.task == "regression":
            return self.loss_fn(logits.squeeze(-1), y.squeeze(-1))
        if self.n_classes == 1:
            return self.loss_fn(logits.squeeze(-1), y.float())
        return self.loss_fn(logits, y)

    def _update_metrics(self, metrics: dict, logits: torch.Tensor, y: torch.Tensor) -> None:
        with torch.no_grad():
            if self.task == "regression":
                preds = logits.squeeze(-1)
                target = y.squeeze(-1)
                metrics["rmse"].update(preds, target)
                metrics["r2"].update(preds, target)
            elif self.n_classes == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
                metrics["auroc"].update(probs, y)
                preds = (probs >= 0.5).long()
                metrics["accuracy"].update(preds, y)
            else:
                probs = torch.softmax(logits, dim=-1)
                metrics["auroc"].update(probs, y)
                metrics["accuracy"].update(logits.argmax(dim=-1), y)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader:
            DataLoader over training donors.
        val_loader:
            DataLoader over validation donors.

        Returns
        -------
        dict
            Training history with keys ``train_loss``, ``val_loss``, and
            per-epoch metric values.
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            for k, v in val_metrics.items():
                history.setdefault(f"val_{k}", []).append(v)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "model.pt",
                )

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch,
                    self.config.epochs,
                    train_loss,
                    val_loss,
                )

        # Persist full history
        with (self.output_dir / "history.json").open("w") as fh:
            json.dump(history, fh, indent=2)

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_bags = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            # Process bag by bag (MIL is per-donor)
            batch_loss = torch.tensor(0.0, device=self.device)
            batch_size = X_batch.shape[0]
            for i in range(batch_size):
                logits, _ = self.model(X_batch[i])
                loss = self._compute_loss(logits.unsqueeze(0), y_batch[i].unsqueeze(0))
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()
            n_bags += 1

        return total_loss / max(n_bags, 1)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a DataLoader.

        Parameters
        ----------
        loader:
            DataLoader to evaluate on.

        Returns
        -------
        dict
            Dictionary with ``loss`` and task-specific metrics.
        """
        self.model.eval()
        metrics = self._build_metrics()
        total_loss = 0.0
        n_bags = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                batch_size = X_batch.shape[0]
                for i in range(batch_size):
                    logits, _ = self.model(X_batch[i])
                    loss = self._compute_loss(
                        logits.unsqueeze(0), y_batch[i].unsqueeze(0)
                    )
                    total_loss += loss.item()
                    self._update_metrics(metrics, logits.unsqueeze(0), y_batch[i].unsqueeze(0))
                    n_bags += 1

        result: Dict[str, float] = {"loss": total_loss / max(n_bags, 1)}
        for name, metric in metrics.items():
            try:
                result[name] = metric.compute().item()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not compute metric '%s': %s", name, exc)
                result[name] = float("nan")
        return result
