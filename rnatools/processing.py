"""Module for preprocessing RNA sequences for machine learning models."""
import numpy as np
import torch
from multimolecule import RnaTokenizer, RiNALMoModel
from tqdm import tqdm

from rnatools.fasta import load_fasta

class FeatureExtractor:
    """Extract embeddings from RNA sequences using the RiNALMo model."""
    def __init__(self, model_name="multimolecule/rinalmo-giga", batch_size=32):
        self.batch_size = batch_size
        self.tokenizer = RnaTokenizer.from_pretrained(model_name)
        self.model = RiNALMoModel.from_pretrained(model_name).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def transform(self, X):
        """Transform a list of RNA sequences into a feature matrix."""
        sequences = [str(s).upper().replace("T", "U") for s in np.asarray(X, dtype=object).ravel()]
        embs = []
        with torch.no_grad():
            for start in tqdm(range(0, len(sequences), self.batch_size), desc="Extracting embeddings"):
                batch = sequences[start : start + self.batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                embs.append(emb.cpu().numpy())
        return np.vstack(embs)

class Preprocess:
    """Encode RNA sequences from a FASTA file."""
    def __init__(self, fasta_path, batch_size=32, model_name="multimolecule/rinalmo-giga"):
        self.fasta_path = fasta_path
        self.feature_extractor = FeatureExtractor(model_name=model_name, batch_size=batch_size)

    def transform(self):
        """Encode FASTA sequences into a feature matrix."""
        self.df = load_fasta(self.fasta_path)
        embeddings = self.feature_extractor.transform(self.df["sequence"])
        X = np.empty((len(self.df), 1 + embeddings.shape[1]), dtype=object)
        X[:, 0] = self.df["sequence"].to_numpy()
        X[:, 1:] = embeddings
        return X