import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpdesk.metrics import hit_at_k, recall_at_k, mrr


def test_hit_and_recall():
    ranked = ["a", "b", "c", "d"]
    assert hit_at_k(ranked, ["c"], 2) == 0.0
    assert hit_at_k(ranked, ["c"], 3) == 1.0
    assert recall_at_k(ranked, ["a", "c", "z"], 3) == 2 / 3


def test_mrr():
    assert mrr(["a", "b", "c"], ["b"]) == 0.5
    assert mrr(["a", "b", "c"], ["z"]) == 0.0
    assert mrr(["a", "b", "c"], ["a", "b"]) == 1.0
