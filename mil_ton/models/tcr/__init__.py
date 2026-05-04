"""BertTCR submodule: TCR repertoire classification via BERT + CNN + MIL.

Based on:
  Zhang et al. (2024) BertTCR: a Bert-based deep learning framework for
  predicting cancer-related immune status based on T cell receptor repertoire.
  Brief Bioinform 25(5):bbae420. doi:10.1093/bib/bbae420.
  https://github.com/zhangbeibei-min/BertTCR
"""

from mil_ton.models.tcr.model import BertTCRModel
from mil_ton.models.tcr.encoder import TCRBertEncoder, encode_tcrs
from mil_ton.models.tcr.dataset import TCRBagDataset

__all__ = [
    "BertTCRModel",
    "TCRBertEncoder",
    "encode_tcrs",
    "TCRBagDataset",
]
