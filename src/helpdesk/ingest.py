"""Stage 1: load and validate the knowledge base and the test queries.

In : corpus.json (list of articles), test_queries.json (queries + the correct article ids)
Out: validated Python objects the retrievers will use in Stages 2-4.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("helpdesk.ingest")

VALID_QUERY_TYPES = {"keyword", "paraphrase", "mixed", "out_of_scope"}


@dataclass(frozen=True)
class Document:
    id: str
    category: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        """Title + body. Titles carry strong keywords, so retrievers index both."""
        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class TestQuery:
    id: str
    type: str
    query: str
    relevant_ids: tuple
    expect_refusal: bool
    set: str = "core"          # core = original queries; extended = added after seeing Stage 3 results


def load_corpus(path) -> list:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    docs, seen = [], set()
    for item in raw:
        missing = {"id", "category", "title", "text"} - item.keys()
        if missing:
            raise ValueError(f"Article {item.get('id')} is missing fields: {missing}")
        if not item["text"].strip():
            raise ValueError(f"Article {item['id']} has empty text")
        if item["id"] in seen:
            raise ValueError(f"Duplicate article id: {item['id']}")
        seen.add(item["id"])
        docs.append(Document(item["id"], item["category"], item["title"], item["text"]))
    log.info("Loaded %d articles from %s", len(docs), path)
    return docs


def load_test_queries(path, corpus_ids) -> list:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    corpus_ids, queries = set(corpus_ids), []
    for item in raw:
        if item["type"] not in VALID_QUERY_TYPES:
            raise ValueError(f"{item['id']}: unknown type {item['type']}")
        unknown = set(item["relevant_ids"]) - corpus_ids
        if unknown:
            raise ValueError(f"{item['id']}: relevant ids not in corpus: {unknown}")
        if item["expect_refusal"] != (item["type"] == "out_of_scope"):
            raise ValueError(f"{item['id']}: expect_refusal must match out_of_scope type")
        if not item["expect_refusal"] and not item["relevant_ids"]:
            raise ValueError(f"{item['id']}: answerable query needs relevant_ids")
        queries.append(TestQuery(item["id"], item["type"], item["query"],
                                 tuple(item["relevant_ids"]), item["expect_refusal"], item.get("set", "core")))
    log.info("Loaded %d test queries from %s", len(queries), path)
    return queries
