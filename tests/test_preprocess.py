"""Tests for encoding sequences from a FASTA file."""
from pathlib import Path

import numpy as np

from rnatools.processing import Preprocess

FASTA_PATH = Path(__file__).parent / "data" / "sequences.fasta"


class FakeExtractor:
    """Stand-in for RiNALMo so tests do not download model weights."""

    def __init__(self, model_name="multimolecule/rinalmo-giga", batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size

    def transform(self, sequences):
        sequences = list(sequences)
        embeddings = np.arange(len(sequences) * 3, dtype=float).reshape(len(sequences), 3)
        return embeddings


def _preprocess(monkeypatch):
    monkeypatch.setattr("rnatools.processing.FeatureExtractor", FakeExtractor)
    return Preprocess(FASTA_PATH, batch_size=2, model_name="fake-model")


def test_transform_encodes_unique_fasta_sequences(monkeypatch):
    preprocess = _preprocess(monkeypatch)

    X = preprocess.transform()

    assert list(X[:, 0]) == ["ACGUACGU", "GGGGCCCC", "ACGTACGT"]
    assert list(preprocess.df["id"]) == ["mir-1", "mir-2", "mir-3"]
    assert X.shape == (3, 4)
    assert np.array_equal(X[:, 1:], np.arange(9, dtype=float).reshape(3, 3))
