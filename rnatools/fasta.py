"""FASTA loading."""
import pandas as pd
from Bio.SeqIO import parse


def load_fasta(fasta_path):
    """Load unique sequences from a FASTA file."""
    data = [{"id": rec.id, "sequence": str(rec.seq)} for rec in parse(fasta_path, "fasta")]
    return pd.DataFrame(data).drop_duplicates(subset=["sequence"])


def encode_fasta(fasta_path, encoder):
    """Load unique FASTA sequences and encode them with encoder.transform."""
    sequences = load_fasta(fasta_path)
    features = encoder.transform(sequences["sequence"])
    return sequences, features
