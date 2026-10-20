"""Tests for k-mer CountVectorizer encoding."""
import numpy as np
import pytest

from rnatools.cli import kmers
from rnatools.kmer import KmerEncoder
from test_preprocess import FASTA_PATH


def test_k_is_the_ngram_size():
    encoder = KmerEncoder(k=2)

    assert encoder.vectorizer.ngram_range == (2, 2)
    assert list(encoder.vectorizer.get_feature_names_out())[1] == "AC"
    assert len(encoder.vectorizer.get_feature_names_out()) == 16


def test_transform_counts_overlapping_kmers_and_replaces_t():
    counts = KmerEncoder(k=2).transform(["ACGT", "AAAA"])
    names = list(KmerEncoder(k=2).vectorizer.get_feature_names_out())

    assert counts.shape == (2, 16)
    assert counts[0, names.index("AC")] == 1
    assert counts[0, names.index("CG")] == 1
    assert counts[0, names.index("GU")] == 1
    assert counts[1, names.index("AA")] == 3
    assert counts.sum(axis=1).tolist() == [3, 3]


def test_k_must_be_a_positive_integer():
    with pytest.raises(ValueError):
        KmerEncoder(k=0)
    with pytest.raises(ValueError):
        KmerEncoder(k=1.5)


def test_kmers_cli_writes_counts(tmp_path):
    output = tmp_path / "counts.npz"

    kmers([str(FASTA_PATH), "-k", "2", "-o", str(output)])

    data = np.load(output)
    names = list(data["kmers"])
    assert list(data["ids"]) == ["mir-1", "mir-2", "mir-3"]
    assert list(data["sequences"]) == ["ACGUACGU", "GGGGCCCC", "ACGTACGT"]
    assert data["counts"].shape == (3, 16)
    assert data["counts"][0, names.index("AC")] == 2
    assert np.array_equal(data["counts"][0], data["counts"][2])


def test_kmers_cli_uses_default_output(tmp_path):
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(FASTA_PATH.read_text())

    kmers([str(fasta), "-k", "1"])

    data = np.load(tmp_path / "sample.k1.npz")
    assert data["counts"].shape == (3, 4)
