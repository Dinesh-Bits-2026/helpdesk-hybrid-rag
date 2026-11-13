import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from helpdesk.config import load_config, resolve_path
from helpdesk.ingest import load_corpus, load_test_queries


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def docs(cfg):
    return load_corpus(resolve_path(cfg, "corpus"))


def test_corpus_size_and_unique_ids(docs):
    assert len(docs) == 40
    assert len({d.id for d in docs}) == len(docs)


def test_articles_are_reasonable_length(docs):
    for d in docs:
        assert 25 <= len(d.text.split()) <= 120, d.id


def test_test_queries_reference_real_articles(cfg, docs):
    qs = load_test_queries(resolve_path(cfg, "test_queries"), [d.id for d in docs])
    assert len(qs) == 29
    types = {q.type for q in qs}
    assert types == {"keyword", "paraphrase", "mixed", "out_of_scope"}
    assert sum(q.expect_refusal for q in qs) == 5
    assert {q.set for q in qs} == {"core", "extended"}
    assert sum(q.set == "core" for q in qs) == 17


def test_config_has_required_sections(cfg):
    for key in ("retrieval", "reranker", "guardrails", "generation", "paths"):
        assert key in cfg
