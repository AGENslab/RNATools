# RNATools

Library for RNA encoding and processing. Sequences are embedded with [RiNALMo](https://huggingface.co/multimolecule/rinalmo-giga).

## Install

```bash
pip install -e .
```

For tests:

```bash
pip install -e ".[dev]"
pytest
```

## Usage

### From the command line

`rnatools-rinalmo` reads a FASTA file, drops duplicate sequences, and writes a compressed `.npz` with `ids`, `sequences`, and `embeddings`.

```bash
rnatools-rinalmo sequences.fasta -o embeddings.npz
```

By default the output is `<fasta>.rinalmo.npz`. `--batch-size` controls the encoding batches.

```python
import numpy as np

data = np.load("embeddings.npz")
data["ids"], data["sequences"], data["embeddings"]
```

`rnatools-kmers` counts k-mers of size `-k` and writes `ids`, `sequences`, `kmers`, and `counts`.

```bash
rnatools-kmers sequences.fasta -k 3 -o counts.npz
```

By default the output is `<fasta>.k<k>.npz`.

### Encode a FASTA file

`Preprocess` reads a FASTA file, drops duplicate sequences, and returns a feature matrix. The first column is the sequence. The remaining columns are the RiNALMo embedding.

```python
from rnatools import Preprocess

preprocess = Preprocess("sequences.fasta", batch_size=32)
X = preprocess.transform()

sequences = X[:, 0]
embeddings = X[:, 1:]
```

The loaded records stay available in `preprocess.df` (`id` and `sequence`).

### Encode a list of sequences

`FeatureExtractor` embeds sequences directly. `T` is replaced with `U` before encoding.

```python
from rnatools import FeatureExtractor

extractor = FeatureExtractor(batch_size=32)
embeddings = extractor.transform(["ACGUACGU", "GGGGCCCC"])
```

Both classes use `multimolecule/rinalmo-giga` by default and run on CUDA when it is available. Pass `model_name` to use another RiNALMo checkpoint.

### Count k-mers

`KmerEncoder` counts character n-grams of size `k` with `CountVectorizer`. Columns are every k-mer of A, C, G and U. `T` is replaced with `U`.

```python
from rnatools import KmerEncoder

counts = KmerEncoder(k=3).transform(["ACGUACGU", "GGGGCCCC"])
```

The same encoding is available as `rnatools-kmers`.
