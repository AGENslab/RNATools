"""FASTA loading."""
import pandas as pd
from Bio.SeqIO import parse


def load_fasta(fasta_path):
    """Load unique sequences from a FASTA file."""
    data = [{"id": rec.id, "sequence": str(rec.seq)} for rec in parse(fasta_path, "fasta")]
    return pd.DataFrame(data).drop_duplicates(subset=["sequence"])
