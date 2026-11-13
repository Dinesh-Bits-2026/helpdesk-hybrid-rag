"""Stage 6: the end-to-end assistant.

question -> input guardrails -> retrieval confidence gate -> retrieve + re-rank -> LLM -> output checks -> logged response
"""
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from .generation import ExtractiveGenerator
from .guardrails import check_input, validate_output

log = logging.getLogger("helpdesk.pipeline")

NO_ANSWER_MESSAGE = "I couldn't find this in the knowledge base. Please raise a service desk ticket."
INJECTION_MESSAGE = "I can't help with that request."


@dataclass
class Response:
    status: str            # answered | uncited | degraded | refused_out_of_scope | refused_injection | no_answer | empty
    answer: str
    query: str             # already PII-masked
    sources: list = field(default_factory=list)
    model: str = ""
    top_dense_cosine: float = None
    pii_types: list = field(default_factory=list)
    cited_ids: list = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)
    raw_answer: str = ""   # model text before validation (logged truncated, to diagnose uncited answers)


class HelpdeskAssistant:
    def __init__(self, cfg, docs, retriever, confidence_fn, generator, fallback=None, request_log=None):
        self.cfg, self.g = cfg, cfg["guardrails"]
        self.docs = {d.id: d for d in docs}
        self.retriever, self.confidence_fn = retriever, confidence_fn
        self.generator, self.fallback = generator, fallback or ExtractiveGenerator()
        self.top_n = cfg["reranker"]["top_n"]
        self.request_log = Path(request_log) if request_log else None

    def _log(self, r: Response):
        if not self.request_log:
            return
        self.request_log.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(r)
        row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row["answer"] = r.answer[:200]
        row["raw_answer"] = r.raw_answer[:300]
        with open(self.request_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _finish(self, r: Response, t0: float) -> Response:
        r.latency_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)
        log.info("status=%s cosine=%s sources=%s total_ms=%s", r.status, r.top_dense_cosine,
                 [s["id"] for s in r.sources], r.latency_ms["total"])
        self._log(r)
        return r

    def answer(self, question: str) -> Response:
        t0 = time.perf_counter()
        chk = check_input(question, self.g)
        r = Response(status="", answer="", query=chk.clean_text, pii_types=chk.pii_types)
        if not chk.clean_text:
            r.status, r.answer = "empty", "Please type a question."
            return self._finish(r, t0)
        if chk.injection:
            r.status, r.answer = "refused_injection", INJECTION_MESSAGE
            return self._finish(r, t0)

        t = time.perf_counter()
        r.top_dense_cosine = round(float(self.confidence_fn(chk.clean_text)), 4)
        if r.top_dense_cosine < self.g["min_dense_cosine"]:
            r.status, r.answer = "refused_out_of_scope", self.cfg["domain"]["out_of_scope_message"]
            r.latency_ms["gate"] = round((time.perf_counter() - t) * 1000, 1)
            return self._finish(r, t0)

        hits = self.retriever.search(chk.clean_text, self.top_n)
        r.latency_ms["retrieval"] = round((time.perf_counter() - t) * 1000, 1)
        if not hits:
            r.status, r.answer = "no_answer", NO_ANSWER_MESSAGE
            return self._finish(r, t0)
        srcs = [self.docs[h.doc_id] for h in hits]
        r.sources = [{"id": d.id, "title": d.title, "score": round(h.score, 4)} for d, h in zip(srcs, hits)]

        t = time.perf_counter()
        degraded = False
        try:
            gen = self.generator.generate(chk.clean_text, srcs)
        except Exception as exc:                                  # LLM down or rate-limited: degrade, don't crash
            log.error("Generation failed, using extractive fallback: %s", exc)
            gen, degraded = self.fallback.generate(chk.clean_text, srcs), True
        r.latency_ms["generation"] = round((time.perf_counter() - t) * 1000, 1)
        r.model, r.raw_answer = gen.model, gen.text

        out = validate_output(gen.text, [d.id for d in srcs])
        if out.not_in_kb:
            r.status, r.answer = "no_answer", NO_ANSWER_MESSAGE
        else:
            r.answer, r.cited_ids = out.text, out.cited_ids
            if self.g.get("require_citation", True) and not out.cited_ids:
                r.status = "uncited"
            else:
                r.status = "degraded" if degraded else "answered"
        return self._finish(r, t0)


def build_assistant(cfg, docs, backend=None, rerank_backend=None, use_llm=True, pipeline=None):
    """Assemble retrievers, re-ranker and generator from config. Returns (assistant, parts)."""
    from .config import PROJECT_ROOT
    from .embeddings import build_encoder
    from .generation import GroqGenerator
    from .rerank import Reranker, build_scorer
    from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankedRetriever

    R, fz = cfg["retrieval"], cfg["retrieval"]["fusion"]
    bm25 = BM25Retriever(docs, cfg["lexical"], R["bm25"]["k1"], R["bm25"]["b"])
    dense = DenseRetriever(docs, build_encoder(cfg, backend), PROJECT_ROOT / R["embedding"]["cache_dir"])
    hybrid = HybridRetriever([bm25, dense], fz["rrf_k"], R["candidate_pool"], fz.get("weights"))
    rr = Reranker(build_scorer(cfg, rerank_backend), docs)
    pool = cfg["reranker"]["candidate_pool"]
    parts = {"bm25": bm25, "dense": dense, "hybrid": hybrid,
             "dense_rerank": RerankedRetriever(dense, rr, pool), "hybrid_rerank": RerankedRetriever(hybrid, rr, pool)}
    name = pipeline or cfg["pipeline"]["default"]
    generator = GroqGenerator(cfg) if use_llm else ExtractiveGenerator()
    log_path = PROJECT_ROOT / cfg["logging"]["request_log"]
    assistant = HelpdeskAssistant(cfg, docs, parts[name], lambda q: dense.scores(q).max(), generator,
                                  request_log=log_path)
    return assistant, parts
