"""Command-line interface for MIL-ton."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.logging import RichHandler

app = typer.Typer(
    name="mil-predict",
    help="Donor-level prediction pipeline for single-cell RNA-seq using MIL.",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@app.command()
def run(
    input_dir: Path = typer.Argument(..., help="Input data directory"),
    donor_col: str = typer.Option(..., help="Column name for donor IDs"),
    label_cols: List[str] = typer.Option(..., help="Label column name(s)"),
    task: str = typer.Option("classification", help="Task type: classification or regression"),
    output_dir: Path = typer.Option(Path("./results"), help="Output directory"),
    config: Optional[Path] = typer.Option(None, help="YAML config path"),
    n_latent: int = typer.Option(30, help="scVI latent dimensions"),
    cells_per_donor: int = typer.Option(5000, help="Cells per donor to sample"),
    epochs: int = typer.Option(100, help="Training epochs"),
    seed: int = typer.Option(42, help="Random seed"),
    skip_scvi: bool = typer.Option(False, "--skip-scvi", help="Skip scVI training"),
    batch_key: Optional[str] = typer.Option(None, help="Batch key for scVI"),
) -> None:
    """Run the full MIL-ton donor-level prediction pipeline."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from mil_ton.config import Config, load_config, save_config
    from mil_ton.data.ingestion import load_data
    from mil_ton.models.mil_model import MILModel
    from mil_ton.models.scvi_model import save_scvi, train_scvi
    from mil_ton.training.dataset import DonorDataset, split_donors
    from mil_ton.training.trainer import Trainer
    from mil_ton.inference.predict import export_predictions, predict_donors
    from mil_ton.inference.interpret import (
        export_top_attended_cells,
        get_attention_weights,
    )

    # ------------------------------------------------------------------ #
    # 1. Config                                                            #
    # ------------------------------------------------------------------ #
    if config is not None:
        cfg = load_config(config)
        logger.info("Loaded config from %s", config)
    else:
        cfg = Config()

    # CLI overrides
    cfg.data.donor_col = donor_col
    cfg.data.label_cols = list(label_cols)
    cfg.data.task = task
    cfg.data.cells_per_donor = cells_per_donor
    cfg.scvi.n_latent = n_latent
    cfg.scvi.batch_key = batch_key
    cfg.training.epochs = epochs
    cfg.training.seed = seed

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    # ------------------------------------------------------------------ #
    # 2. Load data                                                         #
    # ------------------------------------------------------------------ #
    logger.info("Loading data from %s", input_dir)
    adata = load_data(
        input_dir,
        donor_col=cfg.data.donor_col,
        label_cols=cfg.data.label_cols,
        varfeats_path=cfg.data.varfeats_path,
    )

    # ------------------------------------------------------------------ #
    # 3. scVI                                                              #
    # ------------------------------------------------------------------ #
    if skip_scvi:
        if "X_scVI" not in adata.obsm:
            logger.error(
                "--skip-scvi specified but adata.obsm['X_scVI'] is missing. "
                "Please run scVI first or provide pre-computed embeddings."
            )
            sys.exit(1)
        logger.info("Skipping scVI training; using pre-computed latent representation.")
    else:
        logger.info("Training scVI model …")
        adata, scvi_model = train_scvi(adata, cfg.scvi)
        save_scvi(scvi_model, output_dir / "scvi_model")

    latent_dim = adata.obsm["X_scVI"].shape[1]

    # ------------------------------------------------------------------ #
    # 4. Dataset / splits                                                  #
    # ------------------------------------------------------------------ #
    all_donors = adata.obs[cfg.data.donor_col].unique().tolist()
    train_donors, val_donors, test_donors = split_donors(
        all_donors,
        cfg.training.train_frac,
        cfg.training.val_frac,
        seed=cfg.training.seed,
    )
    logger.info(
        "Donors → train=%d  val=%d  test=%d",
        len(train_donors),
        len(val_donors),
        len(test_donors),
    )

    def make_subset(donor_list):
        ds = DonorDataset(
            adata,
            donor_col=cfg.data.donor_col,
            label_cols=cfg.data.label_cols,
            cells_per_donor=cfg.data.cells_per_donor,
            task=cfg.data.task,
            seed=cfg.training.seed,
        )
        indices = [ds.donors.index(d) for d in donor_list if d in ds.donors]
        return Subset(ds, indices)

    train_ds = make_subset(train_donors)
    val_ds = make_subset(val_donors)
    test_ds = make_subset(test_donors)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size)

    # ------------------------------------------------------------------ #
    # 5. Determine number of classes                                       #
    # ------------------------------------------------------------------ #
    if cfg.data.task == "regression":
        n_classes = len(cfg.data.label_cols)
    else:
        unique_labels = adata.obs[cfg.data.label_cols[0]].nunique()
        n_classes = 1 if unique_labels <= 2 else unique_labels

    # ------------------------------------------------------------------ #
    # 6. Train MIL                                                         #
    # ------------------------------------------------------------------ #
    torch.manual_seed(cfg.training.seed)
    model = MILModel(
        input_dim=latent_dim,
        encoder_dims=cfg.mil.encoder_dims,
        attention_dim=cfg.mil.attention_dim,
        n_classes=n_classes,
        task=cfg.data.task,
        dropout=cfg.mil.dropout,
        n_heads=cfg.mil.n_heads,
    )

    trainer = Trainer(
        model=model,
        config=cfg.training,
        task=cfg.data.task,
        n_classes=n_classes,
        output_dir=output_dir,
    )

    logger.info("Training MIL model …")
    history = trainer.train(train_loader, val_loader)

    # ------------------------------------------------------------------ #
    # 7. Test evaluation                                                   #
    # ------------------------------------------------------------------ #
    # Load best model
    model.load_state_dict(torch.load(output_dir / "model.pt", map_location="cpu"))
    test_metrics = trainer.evaluate(test_loader)
    logger.info("Test metrics: %s", test_metrics)

    with (output_dir / "metrics.json").open("w") as fh:
        json.dump({"test": test_metrics, "history": history}, fh, indent=2)

    # ------------------------------------------------------------------ #
    # 8. Predictions                                                       #
    # ------------------------------------------------------------------ #
    full_ds = DonorDataset(
        adata,
        donor_col=cfg.data.donor_col,
        label_cols=cfg.data.label_cols,
        cells_per_donor=cfg.data.cells_per_donor,
        task=cfg.data.task,
        seed=cfg.training.seed,
    )
    full_loader = DataLoader(full_ds, batch_size=cfg.training.batch_size)
    predictions = predict_donors(model, full_loader, task=cfg.data.task)
    export_predictions(predictions, output_dir / "predictions.csv")

    # ------------------------------------------------------------------ #
    # 9. Attention weights                                                 #
    # ------------------------------------------------------------------ #
    attention_dict = get_attention_weights(model, full_loader)
    export_top_attended_cells(
        adata,
        attention_dict,
        donor_col=cfg.data.donor_col,
        output_path=output_dir / "attention_weights.csv",
    )

    logger.info("Pipeline complete. Outputs saved to %s", output_dir)
