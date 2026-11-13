import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from types import SimpleNamespace
from helpdesk.ui_helpers import (answer_html, chips_html, classify_found, no_sources_html, source_cards_html,
                                 source_colors, status_pill_html)

SRC = [{"id": "KB-009", "title": "VPN <b>error</b> 809", "score": 9.5}, {"id": "KB-021", "title": "Outlook", "score": -0.3}]
DET = [{"doc_id": "KB-009", "ranks": {"bm25": 1, "dense": 1}}, {"doc_id": "KB-021", "ranks": {"bm25": 2}}]


def test_classify_found():
    assert classify_found({"bm25": 1, "dense": 3}) == "both"
    assert classify_found({"bm25": 2}) == "keyword"
    assert classify_found({"dense": 4}) == "semantic"
    assert classify_found({"dense": 15}) == "rerank" and classify_found({}) == "rerank"


def test_cards_are_escaped_coloured_and_mark_cited():
    h = source_cards_html(SRC, DET, ["KB-009"])
    assert "<b>error</b>" not in h and "&lt;b&gt;error" in h                 # no HTML injection via titles
    assert h.count("Cited in answer") == 1 and h.count("Not cited") == 1
    assert "#1c9a6b" in h and "#e08a1e" in h                                 # both = green, keyword only = amber


def test_answer_pills_match_source_colours_and_escape_text():
    colors = source_colors(SRC, DET)
    h = answer_html("Restart <script>x</script> **now** [KB-009, KB-021].\nDone", colors)
    assert "<script>" not in h and "<b>now</b>" in h and "<br>" in h
    assert 'background:#1c9a6b">KB-009' in h and 'background:#e08a1e">KB-021' in h


def test_chips_and_status():
    r = SimpleNamespace(pii_types=["EMAIL"], status="answered", top_dense_cosine=0.8, sources=SRC, model="m",
                        cited_ids=["KB-009"], latency_ms={"total": 1210})
    h = chips_html(r, 0.23)
    assert "Personal data masked: EMAIL" in h and "above gate 0.23" in h and "Citations verified (1)" in h and "1.2 s" in h
    low = SimpleNamespace(pii_types=[], status="refused_out_of_scope", top_dense_cosine=0.1, sources=[], model="",
                          cited_ids=[], latency_ms={"total": 9})
    assert "below gate" in chips_html(low, 0.23)
    assert "Out of scope" in status_pill_html("refused_out_of_scope")
    assert "0.10" in no_sources_html("refused_out_of_scope", 0.1, 0.23)
