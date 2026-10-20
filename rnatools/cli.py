"""Command-line interfaces for RNA encoding."""
import argparse
from pathlib import Path

import numpy as np

from rnatools.fasta import load_fasta
from rnatools.kmer import KmerEncoder


def main(argv=None):
    """Encode sequences from a FASTA file and save the embeddings."""
    parser = argparse.ArgumentParser(
        prog="rnatools-rinalmo",
        description="Encode RNA sequences from a FASTA file with RiNALMo.",
    )
    parser.add_argument("fasta", type=Path, help="Input FASTA file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .npz file (default: <fasta>.rinalmo.npz)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument(
        "--model-name",
        default="multimolecule/rinalmo-giga",
        help="RiNALMo checkpoint",
    )
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not args.fasta.is_file():
        parser.error(f"FASTA file not found: {args.fasta}")

    from rnatools.processing import Preprocess

    output = args.output or args.fasta.with_suffix(".rinalmo.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    preprocess = Preprocess(
        args.fasta,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
    matrix = preprocess.transform()
    np.savez_compressed(
        output,
        ids=np.asarray(preprocess.df["id"], dtype=str),
        sequences=np.asarray(matrix[:, 0], dtype=str),
        embeddings=np.asarray(matrix[:, 1:], dtype=np.float32),
    )
    print(f"Encoded {len(preprocess.df)} sequences to {output}")
    return 0


def kmers(argv=None):
    """Count k-mers in a FASTA file and save the count matrix."""
    parser = argparse.ArgumentParser(
        prog="rnatools-kmers",
        description="Count k-mers in RNA sequences from a FASTA file.",
    )
    parser.add_argument("fasta", type=Path, help="Input FASTA file")
    parser.add_argument("-k", type=int, required=True, help="k-mer size")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .npz file (default: <fasta>.k<k>.npz)",
    )
    args = parser.parse_args(argv)

    if not args.fasta.is_file():
        parser.error(f"FASTA file not found: {args.fasta}")
    try:
        encoder = KmerEncoder(args.k)
    except ValueError as exc:
        parser.error(str(exc))

    output = args.output or args.fasta.with_suffix(f".k{args.k}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    sequences = load_fasta(args.fasta)
    counts = encoder.transform(sequences["sequence"])
    np.savez_compressed(
        output,
        ids=np.asarray(sequences["id"], dtype=str),
        sequences=np.asarray(sequences["sequence"], dtype=str),
        kmers=np.asarray(encoder.vectorizer.get_feature_names_out(), dtype=str),
        counts=counts,
    )
    print(f"Encoded {len(sequences)} sequences to {output}")
    return 0
