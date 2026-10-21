"""Tests for the rnatools-rinalmo command."""
import h5py

from rnatools.cli import main
from test_preprocess import FASTA_PATH, FakeExtractor


def test_cli_writes_embeddings_from_fasta(tmp_path, monkeypatch):
    monkeypatch.setattr("rnatools.processing.FeatureExtractor", FakeExtractor)
    output = tmp_path / "embeddings.h5"

    main([str(FASTA_PATH), "-o", str(output), "--batch-size", "2", "--model-name", "fake-model"])

    with h5py.File(output) as data:
        assert list(data["ids"].asstr()[:]) == ["mir-1", "mir-2", "mir-3"]
        assert list(data["sequences"].asstr()[:]) == ["ACGUACGU", "GGGGCCCC", "ACGTACGT"]
        assert data["embeddings"].shape == (3, 3)


def test_cli_uses_default_output(tmp_path, monkeypatch):
    monkeypatch.setattr("rnatools.processing.FeatureExtractor", FakeExtractor)
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(FASTA_PATH.read_text())

    main([str(fasta)])

    with h5py.File(tmp_path / "sample.rinalmo.h5") as data:
        assert list(data["ids"].asstr()[:]) == ["mir-1", "mir-2", "mir-3"]
        assert list(data["sequences"].asstr()[:]) == ["ACGUACGU", "GGGGCCCC", "ACGTACGT"]
