"""Command-line interfaces for RNA encoding."""
import argparse
from pathlib import Path

import h5py
import numpy as np

from rnatools.fasta import encode_fasta
from rnatools.kmer import KmerEncoder


def _require_fasta(parser, fasta):
    if not fasta.is_file():
        parser.error(f"FASTA file not found: {fasta}")


def _output_path(fasta, output, suffix):
    path = output or fasta.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(path, sequences, **arrays):
    np.savez_compressed(
        path,
        ids=np.asarray(sequences["id"], dtype=str),
        sequences=np.asarray(sequences["sequence"], dtype=str),
        **{name: np.asarray(values) for name, values in arrays.items()},
    )


def _save_embeddings(path, sequences, embeddings):
    text = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ids", data=np.asarray(sequences["id"], dtype=object), dtype=text)
        handle.create_dataset("sequences", data=np.asarray(sequences["sequence"], dtype=object), dtype=text)
        handle.create_dataset("embeddings", data=np.asarray(embeddings, dtype=np.float32), compression="gzip")


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
        help="Output .h5 file (default: <fasta>.rinalmo.h5)",
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
    _require_fasta(parser, args.fasta)

    from rnatools.processing import FeatureExtractor

    output = _output_path(args.fasta, args.output, ".rinalmo.h5")
    encoder = FeatureExtractor(model_name=args.model_name, batch_size=args.batch_size)
    sequences, embeddings = encode_fasta(args.fasta, encoder)
    _save_embeddings(output, sequences, embeddings)
    print(f"Encoded {len(sequences)} sequences to {output}")
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

    _require_fasta(parser, args.fasta)
    try:
        encoder = KmerEncoder(args.k)
    except ValueError as exc:
        parser.error(str(exc))

    output = _output_path(args.fasta, args.output, f".k{args.k}.npz")
    sequences, counts = encode_fasta(args.fasta, encoder)
    _save(
        output,
        sequences,
        kmers=np.asarray(encoder.vectorizer.get_feature_names_out(), dtype=str),
        counts=counts,
    )
    print(f"Encoded {len(sequences)} sequences to {output}")
    return 0
