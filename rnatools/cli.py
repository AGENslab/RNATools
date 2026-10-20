"""Command-line interface for RiNALMo encoding."""
import argparse
from pathlib import Path

import numpy as np

from rnatools.processing import Preprocess


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
