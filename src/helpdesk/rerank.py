"""Stage 5: re-ranking with a cross-encoder.

The first-stage retrievers (BM25, dense, RRF) are FAST but judge the query and each article SEPARATELY.
A cross-encoder reads the query and one article TOGETHER and scores how well that article answers the query.
It is slower (one model pass per pair) but much more precise, so we only use it on the ~20 candidates
the first stage hands over. Pattern: cheap retrieval for recall, expensive re-ranking for precision.

In : a query and a list of candidate articles
Out: the same articles re-ordered by the cross-encoder score (higher = more relevant).
Note: the score scale depends on the model and library version; only the ORDER is guaranteed to be
meaningful, so any threshold on it (Stage 6 guardrail) is calibrated on observed values.
"""
import logging
import re

from .retrievers.base import Hit

log = logging.getLogger("helpdesk.rerank")


class CrossEncoderScorer:
    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import CrossEncoder          # imported late: heavy library
        log.info("Loading cross-encoder %s (first run downloads it)", model_name)
        self.model_name, self.batch_size = model_name, batch_size
        self.model = CrossEncoder(model_name, device="cpu")

    def score(self, query: str, texts) -> list:
        pairs = [(query, t) for t in texts]
        return [float(x) for x in self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)]


class TokenOverlapScorer:
    """Offline stand-in for tests and dry runs: share of query words found in the text. NOT a real re-ranker."""
    model_name = "overlap-test-scorer"

    def score(self, query: str, texts) -> list:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [len(q & set(re.findall(r"[a-z0-9]+", t.lower()))) / max(len(q), 1) for t in texts]


class Reranker:
    def __init__(self, scorer, docs):
        self.scorer = scorer
        self.text = {d.id: d.full_text for d in docs}

    def rerank(self, query: str, hits, top_n: int) -> list:
        """Re-order candidate Hits; the new Hit.score is the cross-encoder score."""
        if not hits:
            return []
        scores = self.scorer.score(query, [self.text[h.doc_id] for h in hits])
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], hits[i].rank))   # ties keep first-stage order
        return [Hit(hits[i].doc_id, scores[i], pos) for pos, i in enumerate(order[:top_n], start=1)]


def build_scorer(cfg: dict, backend_override: str = None):
    rc = cfg["reranker"]
    backend = backend_override or rc.get("backend", "cross-encoder")
    if backend == "overlap":
        log.warning("Using the OVERLAP test scorer: results are not a real re-ranking")
        return TokenOverlapScorer()
    if backend == "cross-encoder":
        return CrossEncoderScorer(rc["model"], rc.get("batch_size", 16))
    raise ValueError(f"Unknown reranker backend: {backend}")
