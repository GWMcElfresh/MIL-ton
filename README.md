# MIL-ton

**Donor-level prediction from single-cell RNA-seq data using Multiple Instance Learning (MIL)**

MIL-ton is a production-quality Python pipeline that takes single-cell RNA-seq data, computes per-cell latent representations with [scVI](https://scvi-tools.org/), and trains an attention-based MIL model to make donor-level predictions (disease classification, treatment response, etc.).

---

## Features

- **scVI integration** – learns a batch-corrected latent space from raw counts
- **Attention MIL** – aggregates per-cell embeddings into a donor-level representation with interpretable attention weights
- **Multi-task** – supports binary/multi-class classification and regression
- **CLI** – single `mil-predict` command drives the entire pipeline
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
mil-predict ./data \
  --donor-col donor_id \
  --label-cols disease_status \
  --task classification \
  --output-dir ./results
```

### Using a config file

```bash
mil-predict ./data \
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

## TCR submodule (BertTCR)

MIL-ton includes an implementation of the **BertTCR** model
([Zhang et al. 2024, *Brief Bioinform*](https://doi.org/10.1093/bib/bbae420),
[github.com/zhangbeibei-min/BertTCR](https://github.com/zhangbeibei-min/BertTCR))
for T-cell receptor repertoire classification.

### Architecture

```
TCR sequences (amino acids)
        │
        ▼  TCRBertEncoder  (transformers ProtBERT or equivalent)
(n_tcrs, hidden_size, max_tcr_len)   ← variable-length sequences
        │                              are padded/truncated here
        ▼  Parallel 1-D CNNs
(n_tcrs, sum_filters, 1)
        │
        ▼  Shared FC layer
(n_tcrs,)   ← per-TCR scalar score  ← pre-MIL embedding returned by API
        │
        ▼  MIL ensemble (5 × linear heads, averaged)
(n_classes,)  class probabilities
```

**Variable-length TCR handling** – BertTCR normalises variable CDR3 length
in two stages:

1. Sequences longer than `max_tcr_len` are **truncated** before BERT
   tokenisation.
2. After encoding, the position dimension is **zero-padded** (or
   truncated) to exactly `max_tcr_len`.  This mirrors the `BERT_embedding.py`
   pre-processing in the original repo.

### Usage

#### From pre-computed BERT embeddings (`.pth` files)

```python
from torch.utils.data import DataLoader
from mil_ton.models.tcr import BertTCRModel, TCRBagDataset
from mil_ton.training.tcr_trainer import train_berttcr, evaluate_berttcr

# Dataset: each .pth file = one repertoire bag
# Filenames must contain flag_positive or flag_negative
train_ds = TCRBagDataset("TrainingData/", flag_positive="Patient", flag_negative="Health")
val_ds   = TCRBagDataset("ValidationData/")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=8)

# Instantiate model – set bert_hidden_size to match your BERT backbone
model = BertTCRModel(
    n_tcrs=100,            # TCRs per repertoire bag
    n_classes=2,
    bert_hidden_size=768,  # 768 for tape bert-base, 1024 for Rostlab/prot_bert_bfd
)

# Train
history = train_berttcr(model, train_loader, val_loader, epochs=50)

# Evaluate – returns loss, accuracy, auc AND pre-MIL embeddings
metrics = evaluate_berttcr(model, val_loader)
print(metrics["auc"])
print(metrics["pre_mil_embeddings"])  # list of (batch, n_tcrs) tensors
```

#### Forward pass – getting both outputs

```python
import torch

# x: (batch, n_tcrs, bert_hidden_size, max_tcr_len)
x = torch.randn(4, 100, 768, 24)
probs, pre_mil = model(x)
# probs:   (4, 2)   – class probabilities after sigmoid
# pre_mil: (4, 100) – per-TCR scalar score before MIL ensemble
```

#### Encoding raw TCR sequences

```python
from mil_ton.models.tcr import TCRBertEncoder

encoder = TCRBertEncoder(
    model_name="Rostlab/prot_bert_bfd",  # any HuggingFace protein LM
    max_tcr_len=24,
)
embeddings = encoder.encode(["CASSLDRGTEAFF", "CASGDSSGANVLTF", "CASSQETQYF"])
# shape: (3, 1024, 24)  – ready to feed into BertTCRModel(bert_hidden_size=1024)
```

### Pre-MIL embeddings

Both `model.forward()` and `evaluate_berttcr()` expose the per-TCR scalar
scores **before** the MIL ensemble classification step.  These
`pre_mil` tensors of shape `(batch, n_tcrs)` capture the CNN-derived
"relevance" of each TCR in the bag and are useful for:

* downstream repertoire analysis and clustering,
* transfer learning / fine-tuning on new cancer types,
* interpretability (which TCRs drive the bag-level prediction?).

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
