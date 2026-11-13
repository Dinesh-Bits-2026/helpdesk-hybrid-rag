import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from helpdesk.embeddings import HashingEncoder
from helpdesk.ingest import Document
from helpdesk.retrievers import DenseRetriever


def docs():
    return [Document("A", "x", "VPN setup", "install the vpn client and sign in"),
            Document("B", "x", "Printer help", "clear the print queue and restart the spooler"),
            Document("C", "x", "Leave", "apply for annual leave in the portal")]


class CountingEncoder(HashingEncoder):
    calls = 0
    def encode(self, texts):
        CountingEncoder.calls += 1
        return super().encode(texts)


def test_vectors_are_unit_length_and_deterministic():
    e = HashingEncoder()
    a, b = e.encode(["vpn error", "printer"]), e.encode(["vpn error", "printer"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, b)


def test_cosine_is_dot_product_and_ranks_matching_doc_first():
    r = DenseRetriever(docs(), HashingEncoder())
    hits = r.search("install vpn client", 3)
    assert hits[0].doc_id == "A"
    assert all(-1.0001 <= h.score <= 1.0001 for h in hits)
    assert len(hits) == 3                       # vector search always returns k results


def test_cache_is_reused_and_invalidated_when_text_changes():
    with tempfile.TemporaryDirectory() as d:
        CountingEncoder.calls = 0
        DenseRetriever(docs(), CountingEncoder(), d)
        assert CountingEncoder.calls == 1       # built once
        DenseRetriever(docs(), CountingEncoder(), d)
        assert CountingEncoder.calls == 1       # loaded from disk, no new encode of the articles
        changed = docs()
        changed[0] = Document("A", "x", "VPN setup", "a different text")
        DenseRetriever(changed, CountingEncoder(), d)
        assert CountingEncoder.calls == 2       # text changed -> rebuilt
