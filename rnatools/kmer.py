"""K-mer encoding with scikit-learn CountVectorizer."""
from itertools import product

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

ALPHABET = "ACGU"


class KmerEncoder:
    """Count k-mer occurrences in RNA sequences.

    k is the n-gram size. Each column is one k-mer from A, C, G and U.
    """

    def __init__(self, k):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be an integer greater than or equal to 1")
        self.k = k
        vocabulary = ["".join(kmer) for kmer in product(ALPHABET, repeat=k)]
        self.vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            lowercase=False,
            vocabulary=vocabulary,
        )
        self.vectorizer.fit(vocabulary)

    def transform(self, X):
        """Return a matrix of k-mer counts, one row per sequence."""
        sequences = [str(s).upper().replace("T", "U") for s in np.asarray(X, dtype=object).ravel()]
        return self.vectorizer.transform(sequences).toarray()
