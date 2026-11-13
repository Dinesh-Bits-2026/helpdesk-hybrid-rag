import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpdesk.config import load_config, resolve_path
from helpdesk.ingest import Document, load_corpus
from helpdesk.retrievers import BM25Retriever, TfidfRetriever

PLAIN = {"stemming": "none", "remove_stopwords": False}


def tiny_docs():
    # titles empty so only the text counts
    return [Document("A", "x", "", "vpn vpn"), Document("B", "x", "", "vpn"), Document("C", "x", "", "printer")]


def test_bm25_matches_hand_calculation():
    r = BM25Retriever(tiny_docs(), PLAIN, k1=1.5, b=0.75)
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))          # 'vpn' is in 2 of 3 docs
    avg = (2 + 1 + 1) / 3
    expected_A = idf * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 2 / avg))
    assert abs(r.scores("vpn")[0] - expected_A) < 1e-9


def test_bm25_saturation_and_length():
    r = BM25Retriever(tiny_docs(), PLAIN, k1=1.5, b=0.75)
    s = r.scores("vpn")
    assert s[0] > s[1] > 0 and s[2] == 0          # more mentions help...
    assert s[0] < 2 * s[1]                         # ...but less than proportionally (saturation)


def test_bm25_k1_zero_is_presence_only():
    r = BM25Retriever(tiny_docs(), PLAIN, k1=0.0, b=0.75)
    s = r.scores("vpn")
    assert abs(s[0] - s[1]) < 1e-9 and s[2] == 0


def test_no_shared_word_means_no_hits():
    for cls in (BM25Retriever, TfidfRetriever):
        r = cls(tiny_docs(), PLAIN)
        assert r.search("zebra", 3) == []


def test_real_corpus_error_code_query_finds_right_article():
    cfg = load_config()
    docs = load_corpus(resolve_path(cfg, "corpus"))
    for r in (TfidfRetriever(docs, cfg["lexical"]), BM25Retriever(docs, cfg["lexical"])):
        assert r.search("VPN error 809", 3)[0].doc_id == "KB-009"
        assert r.search("0x80070005", 1)[0].doc_id == "KB-020"


def test_results_are_ranked_and_deterministic():
    cfg = load_config()
    docs = load_corpus(resolve_path(cfg, "corpus"))
    r = BM25Retriever(docs, cfg["lexical"])
    a, b = r.search("password reset", 5), r.search("password reset", 5)
    assert a == b
    assert [h.rank for h in a] == list(range(1, len(a) + 1))
    assert all(a[i].score >= a[i + 1].score for i in range(len(a) - 1))
