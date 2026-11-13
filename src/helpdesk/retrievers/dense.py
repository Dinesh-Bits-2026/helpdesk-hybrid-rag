"""Stage 3: vector (semantic) retriever.

Index time : embed every article once (cached on disk, keyed by model + article text).
Query time : embed the query, score = dot product with every article vector (= cosine), return the top k.
Strength   : matches meaning ("can't remember my login" -> "Reset your password").
Weakness   : can blur exact tokens (error codes, IDs) and always returns *something*, even for junk queries.
"""
import hashlib
import logging
from pathlib import Path

import numpy as np

from .base import Retriever, top_hits

log = logging.getLogger("helpdesk.dense")


def _fingerprint(model_name: str, texts) -> str:
    h = hashlib.sha256(model_name.encode())
    for t in texts:
        h.update(b"\x00" + t.encode())
    return h.hexdigest()[:16]


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, docs, encoder, cache_dir=None):
        self.doc_ids = [d.id for d in docs]
        self.encoder = encoder
        texts = [d.full_text for d in docs]
        self.matrix = self._load_or_build(texts, cache_dir)
        log.info("Dense index: %d docs, %d dimensions", *self.matrix.shape)

    def _load_or_build(self, texts, cache_dir) -> np.ndarray:
        path = None
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            path = Path(cache_dir) / f"emb_{_fingerprint(self.encoder.model_name, texts)}.npy"
            if path.exists():
                log.info("Loaded cached embeddings from %s", path)
                return np.load(path)
        matrix = self.encoder.encode(texts)
        if path:
            np.save(path, matrix)
            log.info("Saved embeddings to %s", path)
        return matrix

    def scores(self, query: str) -> np.ndarray:
        q = self.encoder.encode([query])[0]
        return self.matrix @ q

    def search(self, query: str, k: int = 5) -> list:
        return top_hits(self.doc_ids, self.scores(query), k, positive_only=False)
