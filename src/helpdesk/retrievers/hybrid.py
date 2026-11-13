"""Stage 4: Reciprocal Rank Fusion (RRF).

In : the ranked lists from several retrievers (here BM25 and dense)
Out: ONE ranked list.

score(article) = sum over retrievers of  weight / (rrf_k + rank_in_that_retriever)

Why ranks and not scores: BM25 scores run to ~10, cosine scores sit around 0.3-0.8. They are on different
scales and cannot be added fairly. Ranks (1st, 2nd, 3rd) are comparable, so RRF only uses ranks.
Effect: an article both retrievers like floats to the top; an article only one retriever likes still survives.
  rrf_k up   -> the gap between rank 1 and rank 5 shrinks (flatter)
  rrf_k down -> the top ranks dominate
"""
import logging
from collections import defaultdict

from .base import Hit, Retriever

log = logging.getLogger("helpdesk.hybrid")


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, retrievers, rrf_k: int = 60, pool: int = 20, weights: dict = None):
        self.retrievers = list(retrievers)
        self.rrf_k, self.pool = rrf_k, pool
        self.weights = weights or {}
        log.info("Hybrid RRF over %s (rrf_k=%s, pool=%s, weights=%s)",
                 [r.name for r in self.retrievers], rrf_k, pool, self.weights or "equal")

    def search_detailed(self, query: str, k: int = 5) -> list:
        """Fused results with provenance: which retriever ranked each article where (used by the UI and the report)."""
        score, ranks = defaultdict(float), defaultdict(dict)
        for r in self.retrievers:
            w = self.weights.get(r.name, 1.0)
            for h in r.search(query, self.pool):
                score[h.doc_id] += w / (self.rrf_k + h.rank)
                ranks[h.doc_id][r.name] = h.rank
        order = sorted(score, key=lambda d: (-score[d], min(ranks[d].values()), d))   # deterministic ties
        return [{"doc_id": d, "rrf_score": score[d], "ranks": dict(ranks[d]), "rank": i}
                for i, d in enumerate(order[:k], start=1)]

    def search(self, query: str, k: int = 5) -> list:
        return [Hit(x["doc_id"], x["rrf_score"], x["rank"]) for x in self.search_detailed(query, k)]
