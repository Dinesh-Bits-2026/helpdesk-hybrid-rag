import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpdesk.retrievers import Hit, HybridRetriever, Retriever


class Fixed(Retriever):
    """A retriever that returns a pre-set ranking, so fusion can be tested by hand."""
    def __init__(self, name, order):
        self.name, self.order = name, order
    def search(self, query, k=5):
        return [Hit(d, 1.0, i) for i, d in enumerate(self.order[:k], start=1)]


def test_rrf_matches_hand_calculation():
    a, b = Fixed("a", ["X", "Y", "Z"]), Fixed("b", ["Z", "X", "Q"])
    res = {r["doc_id"]: r for r in HybridRetriever([a, b], rrf_k=60).search_detailed("q", 10)}
    assert abs(res["X"]["rrf_score"] - (1 / 61 + 1 / 62)) < 1e-12     # rank 1 in a, rank 2 in b
    assert abs(res["Z"]["rrf_score"] - (1 / 63 + 1 / 61)) < 1e-12     # rank 3 in a, rank 1 in b
    assert res["X"]["ranks"] == {"a": 1, "b": 2}


def test_agreement_beats_single_vote_but_single_vote_survives():
    a, b = Fixed("a", ["X", "Y"]), Fixed("b", ["X", "Q"])
    ids = [h.doc_id for h in HybridRetriever([a, b]).search("q", 5)]
    assert ids[0] == "X"                      # both retrievers agree
    assert set(ids) == {"X", "Y", "Q"}        # articles liked by only one retriever are kept


def test_pool_limits_candidates_and_k_limits_output():
    a, b = Fixed("a", list("ABCDE")), Fixed("b", list("EDCBA"))
    h = HybridRetriever([a, b], pool=2)
    assert {x.doc_id for x in h.search("q", 10)} == {"A", "B", "E", "D"}
    assert len(h.search("q", 3)) == 3


def test_weights_change_the_winner():
    a, b = Fixed("a", ["X", "Y"]), Fixed("b", ["Y", "X"])
    assert HybridRetriever([a, b], weights={"a": 2.0, "b": 1.0}).search("q", 1)[0].doc_id == "X"
    assert HybridRetriever([a, b], weights={"a": 1.0, "b": 2.0}).search("q", 1)[0].doc_id == "Y"


def test_ties_are_deterministic():
    a, b = Fixed("a", ["X", "Y"]), Fixed("b", ["Y", "X"])
    first = [h.doc_id for h in HybridRetriever([a, b]).search("q", 2)]
    assert first == [h.doc_id for h in HybridRetriever([a, b]).search("q", 2)]
