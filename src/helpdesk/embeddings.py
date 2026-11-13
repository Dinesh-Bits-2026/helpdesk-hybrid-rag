"""Stage 3: turn text into vectors ("embeddings").

In : a list of texts
Out: a matrix, one row per text, each row a unit-length vector (length 1).

Intuition: the model places every sentence as a point in a 384-dimensional space so that sentences
with similar MEANING land close together, even with no shared words. "can't remember my login"
and "reset your password" end up neighbours. Because every vector has length 1, the dot product of
two vectors IS their cosine similarity (1 = same direction, 0 = unrelated, negative = opposite).
"""
import hashlib
import logging
import re

import numpy as np

log = logging.getLogger("helpdesk.embeddings")


class SentenceTransformerEncoder:
    """The real semantic encoder. First run downloads the model (~90 MB) and caches it locally."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer     # imported late: heavy library
        log.info("Loading embedding model %s (first run downloads it)", model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode(self, texts) -> np.ndarray:
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)


class HashingEncoder:
    """Offline stand-in for tests and dry runs. It is NOT semantic: it just hashes words into buckets,
    so it behaves like a crude keyword matcher. It exists so the pipeline and tests run without
    downloading a model. Never report its results as 'vector search' results."""

    model_name = "hashing-test-encoder"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _bucket(self, token: str):
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        return h % self.dim, 1.0 if (h >> 40) & 1 else -1.0       # stable across runs (unlike hash())

    def encode(self, texts) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                idx, sign = self._bucket(tok)
                out[row, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


def build_encoder(cfg: dict, backend_override: str = None):
    emb = cfg["retrieval"]["embedding"]
    backend = backend_override or emb.get("backend", "sentence-transformers")
    if backend == "hashing":
        log.warning("Using the HASHING test encoder: results are not semantic")
        return HashingEncoder()
    if backend == "sentence-transformers":
        return SentenceTransformerEncoder(emb["model"], emb.get("batch_size", 32))
    raise ValueError(f"Unknown embedding backend: {backend}")
