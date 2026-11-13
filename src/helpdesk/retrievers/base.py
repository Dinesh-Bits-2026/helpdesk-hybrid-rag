"""Common shape for every retriever, so fusion (Stage 4) can treat them all the same."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    rank: int          # 1 = best


class Retriever:
    name = "base"

    def search(self, query: str, k: int = 5) -> list:
        """Return up to k Hit objects, best first. Only documents with a positive score are returned."""
        raise NotImplementedError


def top_hits(doc_ids, scores, k, positive_only=True) -> list:
    """Sort scores high->low (ties broken by original order, so results are deterministic).

    Keyword retrievers drop zero scores (no shared word = no match). Vector retrievers keep
    everything (positive_only=False): every article has SOME similarity to any query."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    hits = []
    for i in order:
        if (positive_only and scores[i] <= 0) or len(hits) == k:
            break
        hits.append(Hit(doc_ids[i], float(scores[i]), len(hits) + 1))
    return hits
