"""Tests for the BertTCR TCR submodule.

All tests use synthetic tensors and avoid downloading model weights,
so they run fully offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


# ── helpers ─────────────────────────────────────────────────────────────────

BERT_HIDDEN = 32   # tiny hidden size for fast tests (models accept any size)
MAX_LEN = 8        # max TCR length used throughout tests
N_TCRS = 10        # bag size
N_CLASSES = 2


def _random_bag(batch: int = 1) -> torch.Tensor:
    """Return a batch of random BERT-like embeddings, shape (batch, N_TCRS, BERT_HIDDEN, MAX_LEN)."""
    return torch.randn(batch, N_TCRS, BERT_HIDDEN, MAX_LEN)


def _make_model(**kwargs):
    from mil_ton.models.tcr.model import BertTCRModel
    defaults = dict(
        n_tcrs=N_TCRS,
        n_classes=N_CLASSES,
        filter_num=[2, 2],
        kernel_size=[2, 3],
        max_tcr_len=MAX_LEN,
        dropout=0.0,
        n_ensemble=2,
        bert_hidden_size=BERT_HIDDEN,
    )
    defaults.update(kwargs)
    return BertTCRModel(**defaults).eval()


# ── BertTCRModel tests ───────────────────────────────────────────────────────

class TestBertTCRModel:
    def test_single_bag_output_shapes(self):
        """Single bag (no batch dim) → (n_classes,) and (n_tcrs,)."""
        model = _make_model()
        x = torch.randn(N_TCRS, BERT_HIDDEN, MAX_LEN)
        probs, pre_mil = model(x)
        assert probs.shape == (N_CLASSES,), f"probs shape {probs.shape}"
        assert pre_mil.shape == (N_TCRS,), f"pre_mil shape {pre_mil.shape}"

    def test_batched_output_shapes(self):
        """Batched input → (batch, n_classes) and (batch, n_tcrs)."""
        model = _make_model()
        batch = 4
        x = _random_bag(batch)
        probs, pre_mil = model(x)
        assert probs.shape == (batch, N_CLASSES)
        assert pre_mil.shape == (batch, N_TCRS)

    def test_probabilities_sum_to_one(self):
        """Sigmoid on ensemble mean does NOT sum to 1 for BertTCR (unlike softmax)."""
        # BertTCR uses sigmoid, so each output is in (0, 1) independently.
        model = _make_model()
        x = torch.randn(N_TCRS, BERT_HIDDEN, MAX_LEN)
        probs, _ = model(x)
        assert (probs >= 0).all() and (probs <= 1).all(), "probs must be in [0, 1]"

    def test_auto_padding_short_sequence(self):
        """Embedding shorter than max_tcr_len is zero-padded transparently."""
        model = _make_model(max_tcr_len=MAX_LEN)
        x_short = torch.randn(N_TCRS, BERT_HIDDEN, MAX_LEN - 3)  # shorter
        probs, pre_mil = model(x_short)
        assert probs.shape == (N_CLASSES,)
        assert pre_mil.shape == (N_TCRS,)

    def test_auto_truncation_long_sequence(self):
        """Embedding longer than max_tcr_len is truncated transparently."""
        model = _make_model(max_tcr_len=MAX_LEN)
        x_long = torch.randn(N_TCRS, BERT_HIDDEN, MAX_LEN + 5)  # longer
        probs, pre_mil = model(x_long)
        assert probs.shape == (N_CLASSES,)
        assert pre_mil.shape == (N_TCRS,)

    def test_invalid_filter_kernel_lengths_raise(self):
        """Mismatched filter_num and kernel_size lengths must raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            _make_model(filter_num=[2, 2, 2], kernel_size=[2, 3])

    def test_binary_classification(self):
        """n_classes=1 binary head returns (1,) logit."""
        model = _make_model(n_classes=1)
        x = torch.randn(N_TCRS, BERT_HIDDEN, MAX_LEN)
        probs, pre_mil = model(x)
        assert probs.shape == (1,)
        assert pre_mil.shape == (N_TCRS,)

    def test_configurable_bert_hidden_size(self):
        """Model works with any bert_hidden_size (e.g. 1024 for ProtBERT)."""
        model = _make_model(bert_hidden_size=64)
        x = torch.randn(N_TCRS, 64, MAX_LEN)
        probs, pre_mil = model(x)
        assert probs.shape == (N_CLASSES,)

    def test_pre_mil_embedding_is_detachable(self):
        """pre_mil tensor should be accessible without graph issues."""
        model = _make_model()
        x = _random_bag(1).squeeze(0)
        probs, pre_mil = model(x)
        _ = pre_mil.detach().numpy()

    def test_gradient_flows_to_pre_mil(self):
        """Gradients should flow through the pre-MIL embedding."""
        model = _make_model()
        x = torch.randn(N_TCRS, BERT_HIDDEN, MAX_LEN, requires_grad=True)
        model.train()
        probs, pre_mil = model(x)
        loss = probs.sum()
        loss.backward()
        assert x.grad is not None, "Expected gradient w.r.t. input"


# ── Variable TCR length handling ─────────────────────────────────────────────

class TestVariableLengthHandling:
    """Verify that TCRs of different sequence lengths are handled correctly.

    This is a key correctness concern called out in the problem statement.
    BertTCR handles variable lengths by:
      1. Truncating sequences to max_tcr_len before BERT encoding.
      2. Padding the position dimension of the embedding to max_tcr_len
         after BERT encoding.
    Both behaviours are exercised here at the model level.
    """

    @pytest.mark.parametrize("seq_len", [1, MAX_LEN // 2, MAX_LEN - 1, MAX_LEN, MAX_LEN + 1, MAX_LEN * 2])
    def test_variable_position_dim(self, seq_len: int):
        """Model accepts embeddings of any position length (pads/truncates)."""
        model = _make_model(max_tcr_len=MAX_LEN)
        x = torch.randn(N_TCRS, BERT_HIDDEN, seq_len)
        probs, pre_mil = model(x)
        assert probs.shape == (N_CLASSES,)
        assert pre_mil.shape == (N_TCRS,)

    def test_pad_truncate_determinism(self):
        """Embeddings with the same first max_tcr_len positions give the same
        output regardless of trailing padding."""
        model = _make_model(max_tcr_len=MAX_LEN)
        model.eval()
        base = torch.randn(N_TCRS, BERT_HIDDEN, MAX_LEN)
        padded = torch.cat([base, torch.zeros(N_TCRS, BERT_HIDDEN, 5)], dim=-1)
        with torch.no_grad():
            probs_base, _ = model(base)
            probs_padded, _ = model(padded)
        assert torch.allclose(probs_base, probs_padded, atol=1e-6)

    def test_encoder_format_sequences_helper(self):
        """_format_sequences_for_prot_bert inserts spaces between residues."""
        from mil_ton.models.tcr.encoder import _format_sequences_for_prot_bert
        seqs = ["ACG", "LVK"]
        result = _format_sequences_for_prot_bert(seqs)
        assert result == ["A C G", "L V K"]

    def test_encode_tcrs_uses_padding(self):
        """encode_tcrs pads short sequences to max_tcr_len (mocked BERT)."""
        import torch.nn as nn
        from mil_ton.models.tcr.encoder import encode_tcrs

        class _MockTokenizer:
            def __call__(self, texts, **kw):
                n = len(texts)
                # pretend every sequence tokenises to length 6 (incl. CLS/SEP)
                return {
                    "input_ids": torch.zeros(n, 6, dtype=torch.long),
                    "attention_mask": torch.ones(n, 6, dtype=torch.long),
                }

        class _MockBert(nn.Module):
            def __init__(self):
                super().__init__()
                # dummy param so device detection works
                self._p = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, attention_mask):
                batch, seq_len = input_ids.shape
                hidden = torch.zeros(batch, seq_len, BERT_HIDDEN)
                # Return object with last_hidden_state attribute
                class _Out:
                    pass
                out = _Out()
                out.last_hidden_state = hidden
                return out

        bert = _MockBert().eval()
        tokenizer = _MockTokenizer()
        sequences = ["AC", "LVKG"]  # shorter than MAX_LEN
        embeddings = encode_tcrs(sequences, tokenizer, bert, max_tcr_len=MAX_LEN)
        # Should be (n_tcrs, hidden_size, max_tcr_len)
        assert embeddings.shape == (2, BERT_HIDDEN, MAX_LEN)

    def test_encode_tcrs_truncates_long_sequences(self):
        """encode_tcrs truncates embeddings longer than max_tcr_len."""
        import torch.nn as nn
        from mil_ton.models.tcr.encoder import encode_tcrs

        target_len = 4  # deliberately small

        class _MockTokenizer:
            def __call__(self, texts, **kw):
                n = len(texts)
                long_len = 20  # simulate long embedding
                return {
                    "input_ids": torch.zeros(n, long_len, dtype=torch.long),
                    "attention_mask": torch.ones(n, long_len, dtype=torch.long),
                }

        class _MockBert(nn.Module):
            def __init__(self):
                super().__init__()
                self._p = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, attention_mask):
                batch, seq_len = input_ids.shape
                hidden = torch.randn(batch, seq_len, BERT_HIDDEN)
                class _Out:
                    pass
                out = _Out()
                out.last_hidden_state = hidden
                return out

        bert = _MockBert().eval()
        tokenizer = _MockTokenizer()
        sequences = ["A" * 30]  # very long
        embeddings = encode_tcrs(sequences, tokenizer, bert, max_tcr_len=target_len)
        assert embeddings.shape == (1, BERT_HIDDEN, target_len)


# ── TCRBagDataset ─────────────────────────────────────────────────────────────

class TestTCRBagDataset:
    def test_loads_pth_files(self):
        """Dataset loads .pth tensors and infers labels from filenames."""
        from mil_ton.models.tcr.dataset import TCRBagDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            for i in range(3):
                t = torch.randn(N_TCRS, BERT_HIDDEN, MAX_LEN)
                torch.save(t, p / f"Patient_{i}.pth")
            for i in range(2):
                t = torch.randn(N_TCRS, BERT_HIDDEN, MAX_LEN)
                torch.save(t, p / f"Health_{i}.pth")

            ds = TCRBagDataset(p, flag_positive="Patient", flag_negative="Health")
            assert len(ds) == 5

            tensor, label = ds[0]
            assert tensor.shape == (N_TCRS, BERT_HIDDEN, MAX_LEN)
            assert label in (0, 1)

    def test_all_labels_correct(self):
        """Positive files → label 1, negative files → label 0."""
        from mil_ton.models.tcr.dataset import TCRBagDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            torch.save(torch.zeros(N_TCRS, BERT_HIDDEN, MAX_LEN), p / "Patient_0.pth")
            torch.save(torch.zeros(N_TCRS, BERT_HIDDEN, MAX_LEN), p / "Health_0.pth")

            ds = TCRBagDataset(p)
            labels = {ds[i][1] for i in range(len(ds))}
            assert labels == {0, 1}

    def test_unknown_filename_raises(self):
        """A filename that matches neither flag should raise ValueError."""
        from mil_ton.models.tcr.dataset import TCRBagDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            torch.save(torch.zeros(2, 4, 4), p / "unknown_sample.pth")
            with pytest.raises(ValueError, match="neither"):
                TCRBagDataset(p)

    def test_missing_directory_raises(self):
        from mil_ton.models.tcr.dataset import TCRBagDataset
        with pytest.raises(FileNotFoundError):
            TCRBagDataset("/nonexistent/path")


# ── TCRTrainer / evaluate_berttcr ────────────────────────────────────────────

class TestTCRTrainer:
    def _make_loader(self, n_bags: int = 4, batch: int = 2) -> DataLoader:
        X = torch.randn(n_bags, N_TCRS, BERT_HIDDEN, MAX_LEN)
        y = torch.randint(0, N_CLASSES, (n_bags,))
        return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=False)

    def test_evaluate_returns_keys(self):
        """evaluate returns loss, accuracy, auc, pre_mil_embeddings."""
        from mil_ton.training.tcr_trainer import evaluate_berttcr
        model = _make_model()
        loader = self._make_loader()
        metrics = evaluate_berttcr(model, loader)
        for key in ("loss", "accuracy", "auc", "pre_mil_embeddings"):
            assert key in metrics, f"Missing key: {key}"

    def test_evaluate_pre_mil_shape(self):
        """pre_mil_embeddings are per-bag tensors of shape (batch, n_tcrs)."""
        from mil_ton.training.tcr_trainer import evaluate_berttcr
        model = _make_model()
        loader = self._make_loader(n_bags=4, batch=2)
        metrics = evaluate_berttcr(model, loader)
        for batch_pm in metrics["pre_mil_embeddings"]:
            # each element in the list corresponds to one batch
            assert batch_pm.dim() == 2
            assert batch_pm.shape[1] == N_TCRS

    def test_train_one_epoch(self):
        """train() runs without error for one epoch and returns history."""
        from mil_ton.training.tcr_trainer import TCRTrainer
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            model.train()
            trainer = TCRTrainer(model, epochs=1, output_dir=tmpdir)
            loader = self._make_loader()
            history = trainer.train(loader, loader)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1

    def test_train_saves_checkpoint(self):
        """train() saves tcr_model.pt when val_auc improves."""
        from mil_ton.training.tcr_trainer import TCRTrainer
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            trainer = TCRTrainer(model, epochs=2, output_dir=tmpdir)
            loader = self._make_loader()
            trainer.train(loader, loader)
            # File should exist (auc ≥ 0 in any run)
            assert Path(tmpdir, "tcr_model.pt").exists()

    def test_train_berttcr_convenience(self):
        """train_berttcr wrapper should return a non-empty history dict."""
        from mil_ton.training.tcr_trainer import train_berttcr
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model()
            loader = self._make_loader()
            history = train_berttcr(model, loader, loader, epochs=1, output_dir=tmpdir)
        assert history["train_loss"]
