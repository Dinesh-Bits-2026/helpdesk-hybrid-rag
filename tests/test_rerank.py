import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpdesk.ingest import Document
from helpdesk.rerank import Reranker, TokenOverlapScorer
from helpdesk.retrievers import Hit, Retriever, RerankedRetriever


def docs():
    return [Document("A", "x", "Printer", "clear the print queue"),
            Document("B", "x", "VPN", "install the vpn client and connect"),
            Document("C", "x", "Leave", "apply for annual leave")]


class Fixed(Retriever):
    name = "fixed"
    def __init__(self, order): self.order = order
    def search(self, query, k=5):
        return [Hit(d, 1.0, i) for i, d in enumerate(self.order[:k], start=1)]


def test_reranker_reorders_by_scorer():
    hits = [Hit("A", 1, 1), Hit("B", 1, 2), Hit("C", 1, 3)]
    out = Reranker(TokenOverlapScorer(), docs()).rerank("connect the vpn client", hits, 3)
    assert [h.doc_id for h in out] == ["B", "A", "C"]
    assert [h.rank for h in out] == [1, 2, 3]


def test_top_n_and_ties_keep_first_stage_order():
    hits = [Hit("A", 1, 1), Hit("C", 1, 2)]                       # neither shares a word: tie
    out = Reranker(TokenOverlapScorer(), docs()).rerank("zebra", hits, 2)
    assert [h.doc_id for h in out] == ["A", "C"]
    assert len(Reranker(TokenOverlapScorer(), docs()).rerank("vpn", [Hit("B", 1, 1), Hit("A", 1, 2)], 1)) == 1


def test_wrapper_only_rescores_the_pool_and_reports_first_stage_rank():
    r = RerankedRetriever(Fixed(["A", "C", "B"]), Reranker(TokenOverlapScorer(), docs()), pool=3)
    detail = r.search_detailed("install vpn client", 2)
    assert detail[0]["doc_id"] == "B" and detail[0]["first_stage_rank"] == 3
    r2 = RerankedRetriever(Fixed(["A", "C", "B"]), Reranker(TokenOverlapScorer(), docs()), pool=2)
    assert "B" not in [h.doc_id for h in r2.search("install vpn client", 3)]    # B was outside the pool
    assert r.name == "fixed+rerank"
