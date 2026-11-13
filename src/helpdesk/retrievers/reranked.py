"""Wrap any first-stage retriever with a re-ranker: candidates from `base`, final order from the cross-encoder."""
from .base import Retriever


class RerankedRetriever(Retriever):
    def __init__(self, base: Retriever, reranker, pool: int = 20):
        self.base, self.reranker, self.pool = base, reranker, pool
        self.name = f"{base.name}+rerank"

    def search_detailed(self, query: str, k: int = 5) -> list:
        """Final results plus the first-stage rank each article had, to show what the re-ranker changed."""
        cands = self.base.search(query, self.pool)
        before = {h.doc_id: h.rank for h in cands}
        return [{"doc_id": h.doc_id, "score": h.score, "rank": h.rank, "first_stage_rank": before[h.doc_id]}
                for h in self.reranker.rerank(query, cands, k)]

    def search(self, query: str, k: int = 5) -> list:
        return self.reranker.rerank(query, self.base.search(query, self.pool), k)
