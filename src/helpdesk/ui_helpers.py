"""Pure helpers for the Streamlit app: HTML snippets (escaped), colours and small data shaping.
Kept out of app.py so they can be unit-tested without Streamlit."""
import html
import re

# status -> (label, hint, colour)
STATUS_STYLE = {
    "answered": ("Answered", "Grounded in the cited articles", "#1c9a6b"),
    "uncited": ("Unverified citation", "The model did not cite a retrieved article, so treat with care", "#e08a1e"),
    "degraded": ("Fallback answer", "The LLM was unavailable; showing the best article text", "#b8960b"),
    "refused_out_of_scope": ("Out of scope", "The best article was not similar enough, so the LLM was not called", "#7d8794"),
    "refused_injection": ("Blocked", "The request looked like a prompt-injection attempt", "#d64545"),
    "no_answer": ("Not in knowledge base", "The model found no answer in the retrieved articles", "#5b6fd6"),
    "empty": ("Empty question", "", "#7d8794"),
}
STATUS_LABEL = {k: (v[0], v[1]) for k, v in STATUS_STYLE.items()}   # kept for backwards compatibility

# how a source was found -> (label, colour)
FOUND_STYLE = {
    "both": ("Keyword + semantic", "#1c9a6b"),
    "keyword": ("Keyword only", "#e08a1e"),
    "semantic": ("Semantic only", "#2a6fdb"),
    "rerank": ("Surfaced by re-ranker", "#8a94a6"),
}

PIPELINE_LABELS = {
    "hybrid_rerank": "Hybrid + re-rank (recommended)",
    "dense_rerank": "Vectors + re-rank",
    "hybrid": "Hybrid (keywords + vectors)",
    "dense": "Vectors only",
    "bm25": "Keywords only (BM25)",
}

CSS = """
<style>
.pill{display:inline-block;padding:2px 11px;border-radius:999px;font-size:.78rem;font-weight:700;color:#fff;margin:0 8px 4px 0}
.chip{display:inline-block;padding:2px 9px;border-radius:6px;font-size:.76rem;margin:3px 6px 3px 0;border:1px solid rgba(128,128,128,.4)}
.chip.ok{background:rgba(28,154,107,.15);border-color:rgba(28,154,107,.55)}
.chip.warn{background:rgba(224,138,30,.16);border-color:rgba(224,138,30,.6)}
.chip.bad{background:rgba(214,69,69,.16);border-color:rgba(214,69,69,.6)}
.chip.info{background:rgba(42,111,219,.14);border-color:rgba(42,111,219,.55)}
.cite{display:inline-block;padding:0 7px;border-radius:5px;font-size:.78rem;font-weight:700;color:#fff;margin:0 2px}
.src{border:1px solid rgba(128,128,128,.3);border-left:7px solid #888;border-radius:8px;padding:10px 12px;margin:9px 0;background:rgba(128,128,128,.07)}
.src.dim{opacity:.7}
.src .kb{font-family:monospace;font-weight:700;margin-right:6px}
.src .meta{font-size:.78rem;opacity:.85;margin-top:3px}
.badge{float:right;font-size:.72rem;font-weight:700;padding:1px 8px;border-radius:999px}
.badge.yes{background:#1c9a6b;color:#fff}
.badge.no{background:rgba(128,128,128,.3)}
.bar{height:6px;border-radius:3px;background:rgba(128,128,128,.25);margin-top:7px}
.bar span{display:block;height:6px;border-radius:3px}
.legend{font-size:.76rem;margin:2px 0 6px 0}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin:0 4px -1px 8px}
.empty{border:1px dashed rgba(128,128,128,.5);border-radius:8px;padding:14px;font-size:.9rem;opacity:.85}
</style>
"""


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def trace_rows(sources, hybrid_detail) -> list:
    """Rows for the technical table: final order and score, plus each first-stage retriever's rank."""
    ranks = {x["doc_id"]: x["ranks"] for x in hybrid_detail}
    return [{"Article": s["id"], "Title": s["title"], "Re-rank score": s["score"],
             "BM25 rank": ranks.get(s["id"], {}).get("bm25", "-"),
             "Dense rank": ranks.get(s["id"], {}).get("dense", "-")} for s in sources]


def classify_found(ranks: dict, top: int = 5) -> str:
    """Was the article in the top `top` of the keyword retriever, the vector retriever, both, or neither?"""
    kw = ranks.get("bm25") is not None and ranks["bm25"] <= top
    sem = ranks.get("dense") is not None and ranks["dense"] <= top
    return "both" if kw and sem else "keyword" if kw else "semantic" if sem else "rerank"


def source_colors(sources, hybrid_detail) -> dict:
    by_id = {x["doc_id"]: x["ranks"] for x in hybrid_detail}
    return {s["id"]: FOUND_STYLE[classify_found(by_id.get(s["id"], {}))][1] for s in sources}


def status_pill_html(status: str) -> str:
    label, _, color = STATUS_STYLE.get(status, (status, "", "#7d8794"))
    return f'<span class="pill" style="background:{color}">{_e(label)}</span>'


def answer_html(text: str, colors: dict = None) -> str:
    """Escape the answer, show [KB-009] style citations as coloured pills (same colour as the source card), keep line breaks."""
    colors = colors or {}
    out = _e(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)

    def pills(m):
        return "".join(f'<span class="cite" style="background:{colors.get(i, "#6b7280")}">{i}</span>'
                       for i in re.findall(r"KB-\d{3}", m.group(1)))

    out = re.sub(r"\[((?:KB-\d{3})(?:,\s*KB-\d{3})*)\]", pills, out)
    return out.replace("\n", "<br>")


def chips_html(resp, gate: float) -> str:
    """Compact strip showing which guardrails and stages touched this answer."""
    chips = []
    if resp.pii_types:
        chips.append(("warn", "Personal data masked: " + ", ".join(resp.pii_types)))
    if resp.status == "refused_injection":
        chips.append(("bad", "Prompt-injection pattern detected"))
    elif resp.top_dense_cosine is not None:
        ok = resp.top_dense_cosine >= gate
        chips.append(("ok" if ok else "bad", f"Relevance {resp.top_dense_cosine:.2f} {'above' if ok else 'below'} gate {gate:.2f}"))
    if resp.sources:
        chips.append(("info", f"{len(resp.sources)} sources"))
    if resp.model:
        chips.append(("", resp.model))
    if resp.status in ("answered", "degraded"):
        chips.append(("ok", f"Citations verified ({len(resp.cited_ids)})"))
    elif resp.status == "uncited":
        chips.append(("warn", "No verified citation"))
    total = resp.latency_ms.get("total")
    if total:
        chips.append(("", f"{total / 1000:.1f} s" if total >= 1000 else f"{total:.0f} ms"))
    return "".join(f'<span class="chip {k}">{_e(t)}</span>' for k, t in chips)


def legend_html() -> str:
    items = "".join(f'<i style="background:{c}"></i>{_e(l)}' for l, c in FOUND_STYLE.values())
    return f'<div class="legend"><b>Found by (top 5 of each search):</b>{items}</div>'


def source_cards_html(sources, hybrid_detail, cited_ids) -> str:
    """One card per retrieved article. Border colour = how it was found; green badge = cited in the answer; bar = relative re-rank score."""
    by_id = {x["doc_id"]: x["ranks"] for x in hybrid_detail}
    scores = [s["score"] for s in sources]
    lo, hi = (min(scores), max(scores)) if scores else (0, 1)
    cards = []
    for s in sources:
        ranks = by_id.get(s["id"], {})
        found_label, color = FOUND_STYLE[classify_found(ranks)]
        cited = s["id"] in cited_ids
        width = 100 if hi == lo else 12 + 88 * (s["score"] - lo) / (hi - lo)
        meta = " &middot; ".join([f"Keyword rank {ranks['bm25']}" if "bm25" in ranks else "Keyword: not in top 20",
                                  f"Semantic rank {ranks['dense']}" if "dense" in ranks else "Semantic: not in top 20",
                                  f"Re-rank score {s['score']:.2f}"])
        badge = '<span class="badge yes">Cited in answer</span>' if cited else '<span class="badge no">Not cited</span>'
        cards.append(
            f'<div class="src{"" if cited else " dim"}" style="border-left-color:{color}">{badge}'
            f'<span class="kb" style="color:{color}">{_e(s["id"])}</span><b>{_e(s["title"])}</b>'
            f'<div class="meta">{_e(found_label)} &middot; {meta}</div>'
            f'<div class="bar"><span style="width:{width:.0f}%;background:{color}"></span></div></div>')
    return legend_html() + "".join(cards)


def no_sources_html(status: str, cosine, gate: float) -> str:
    label, hint, color = STATUS_STYLE.get(status, (status, "", "#7d8794"))
    extra = f" Best article similarity {cosine:.2f}, gate {gate:.2f}." if cosine is not None and status == "refused_out_of_scope" else ""
    return f'<div class="empty" style="border-color:{color}"><b>{_e(label)}</b><br>{_e(hint)}{_e(extra)}</div>'
