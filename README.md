# MIL-ton

**Donor-level prediction from single-cell RNA-seq data using Multiple Instance Learning (MIL)**

MIL-ton is a production-quality Python pipeline that takes single-cell RNA-seq data, computes per-cell latent representations with [scVI](https://scvi-tools.org/), and trains an attention-based MIL model to make donor-level predictions (disease classification, treatment response, etc.).

---

## Features

- **scVI integration** – learns a batch-corrected latent space from raw counts
- **Attention MIL** – aggregates per-cell embeddings into a donor-level representation with interpretable attention weights
- **Multi-task** – supports binary/multi-class classification and regression
- **CLI** – single `mil-predict run` command drives the entire pipeline
- **Interpretability** – exports per-cell attention weights for downstream analysis

---

## Installation

### With `uv` (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package and its dependencies
uv sync

# Or install with dev dependencies
uv sync --extra dev
```

### With pip

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10.

---

## Quick start

```bash
mil-predict run \
  --input-dir ./data \
  --donor-col donor_id \
  --label-cols disease_status \
  --task classification \
  --output-dir ./results
```

### Using a config file

```bash
mil-predict run \
  --input-dir ./data \
  --donor-col donor_id \
  --label-cols disease_status \
  --config configs/example.yaml \
  --output-dir ./results
```

---

## Input format

| File | Required | Description |
|---|---|---|
| `GEX.h5` | one of | 10x HDF5 expression matrix |
| `*.h5ad` | one of | AnnData HDF5 file |
| `metadata.tsv` | optional | Tab-separated metadata with cell barcodes as index |

The AnnData / metadata must contain:
- a **donor column** (e.g. `donor_id`) identifying each donor
- one or more **label columns** (e.g. `disease_status`) constant per donor

---

## Outputs

| File | Description |
|---|---|
| `config.yaml` | Full resolved configuration |
| `scvi_model/` | Saved scVI model |
| `model.pt` | Best MIL model checkpoint |
| `history.json` | Per-epoch train/val losses and metrics |
| `metrics.json` | Final test-set metrics |
| `predictions.csv` | Per-donor predictions and probabilities |
| `attention_weights.csv` | Top-attended cells per donor |

---

## Python API

```python
from mil_ton.config import load_config
from mil_ton.data.ingestion import load_data
from mil_ton.models.scvi_model import train_scvi
from mil_ton.models.mil_model import MILModel
from mil_ton.training.dataset import DonorDataset, split_donors
from mil_ton.training.trainer import Trainer

cfg = load_config("configs/example.yaml")
adata = load_data("data/", donor_col="donor_id", label_cols=["disease_status"])
adata, scvi_model = train_scvi(adata, cfg.scvi)

ds = DonorDataset(adata, donor_col="donor_id", label_cols=["disease_status"],
                  cells_per_donor=5000, task="classification")
```

---

## Docker

```bash
docker compose up
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

See [LICENSE](LICENSE).
