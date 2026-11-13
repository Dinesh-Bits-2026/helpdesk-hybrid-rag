"""Retrieval metrics. All take the ranked list of article ids a retriever returned, and the set of correct ids.

hit_at_k    : 1 if ANY correct article is in the top k, else 0        ("did we find it at all?")
recall_at_k : share of the correct articles found in the top k          ("how many of them?")
mrr         : 1 / rank of the first correct article, 0 if none          ("how high up was it?")
"""


def hit_at_k(ranked_ids, relevant_ids, k) -> float:
    return 1.0 if set(ranked_ids[:k]) & set(relevant_ids) else 0.0


def recall_at_k(ranked_ids, relevant_ids, k) -> float:
    rel = set(relevant_ids)
    return len(set(ranked_ids[:k]) & rel) / len(rel) if rel else 0.0


def mrr(ranked_ids, relevant_ids) -> float:
    rel = set(relevant_ids)
    for pos, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in rel:
            return 1.0 / pos
    return 0.0


def evaluate(retriever, queries, k=5) -> list:
    """Run every answerable query; return one result dict per query."""
    rows = []
    for q in queries:
        if q.expect_refusal:
            continue
        ranked = [h.doc_id for h in retriever.search(q.query, k)]
        rows.append({
            "qid": q.id, "set": q.set, "type": q.type, "query": q.query, "method": retriever.name,
            "relevant": ",".join(q.relevant_ids), "top3": ",".join(ranked[:3]),
            "hit@1": hit_at_k(ranked, q.relevant_ids, 1),
            "recall@3": recall_at_k(ranked, q.relevant_ids, 3),
            "mrr": mrr(ranked, q.relevant_ids),
        })
    return rows
