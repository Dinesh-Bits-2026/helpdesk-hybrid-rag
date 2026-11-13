"""Stage 2: keyword retrievers.

TF-IDF  : score = how much of the query's rare vocabulary the article contains (cosine of weighted vectors)
BM25    : like TF-IDF but (a) repeated words saturate and (b) long articles are not unfairly favoured
Both match WORDS, not MEANING. "can't remember my login" will not find "reset your password".
"""
import logging
import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..text import make_analyzer
from .base import Retriever, top_hits

log = logging.getLogger("helpdesk.lexical")


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, docs, lexical_cfg: dict):
        self.doc_ids = [d.id for d in docs]
        self.analyzer = make_analyzer(lexical_cfg)
        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            sublinear_tf=lexical_cfg.get("tfidf", {}).get("sublinear_tf", False),
        )
        self.matrix = self.vectorizer.fit_transform([d.full_text for d in docs])  # rows are L2-normalised
        log.info("TF-IDF index: %d docs, %d terms", *self.matrix.shape)

    def search(self, query: str, k: int = 5) -> list:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()    # dot of unit vectors = cosine similarity
        return top_hits(self.doc_ids, scores, k)


class BM25Retriever(Retriever):
    """Okapi BM25 written out by hand so every moving part is visible (and no extra dependency).

    score(q, d) = sum over query terms t of   idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len_d / avg_len))
      idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))    n_t = number of articles containing t
      k1 up -> repeated words keep adding score for longer   (k1 = 0: presence only)
      b  up -> long articles are penalised harder            (b = 0: ignore length)
    """
    name = "bm25"

    def __init__(self, docs, lexical_cfg: dict, k1: float = 1.5, b: float = 0.75):
        self.doc_ids = [d.id for d in docs]
        self.analyzer = make_analyzer(lexical_cfg)
        self.k1, self.b = k1, b
        tokenised = [self.analyzer(d.full_text) for d in docs]
        self.doc_len = np.array([len(t) for t in tokenised], dtype=float)
        self.avg_len = float(self.doc_len.mean())
        self.tf = [Counter(t) for t in tokenised]                 # term frequency per article
        df = Counter(term for c in self.tf for term in c)         # document frequency per term
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        log.info("BM25 index: %d docs, %d terms, avg length %.1f tokens (k1=%s, b=%s)",
                 n, len(self.idf), self.avg_len, k1, b)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.doc_ids))
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_len)
        for term in set(self.analyzer(query)):
            if term not in self.idf:
                continue
            tf = np.array([c.get(term, 0) for c in self.tf], dtype=float)
            has = tf > 0                                   # articles without the term score 0 for it
            out[has] += self.idf[term] * tf[has] * (self.k1 + 1) / (tf[has] + norm[has])
        return out

    def search(self, query: str, k: int = 5) -> list:
        return top_hits(self.doc_ids, self.scores(query), k)
