"""Build the assignment report (PDF) automatically from this project's own results.

  python build_report.py                 -> report/Conversational_AI_Assignment1_Report.pdf
  python build_report.py --strict        -> stop with an error if a required screenshot is missing
  python build_report.py --include-decision-log   -> also append the decision log as an appendix (off by default)

What it does
  * reads results/*.csv, data/*.json, config/config.yaml and docs/decision_log.md (so every number is from YOUR runs)
  * draws the architecture diagram and the charts itself
  * inserts your screenshots from report/screenshots/ (file names listed in SCREENSHOT_GUIDE.md);
    a missing screenshot becomes a red placeholder box and is listed at the end
  * your name / student id come from report/report_meta.yaml

Run the experiments first (run_stage5.py, run_stage6.py --calibrate, run_stage6.py), then this script.
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyBboxPatch
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Flowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default="results")
ap.add_argument("--screenshots-dir", default="report/screenshots")
ap.add_argument("--meta", default="report/report_meta.yaml")
ap.add_argument("--out", default="report/Conversational_AI_Assignment1_Report.pdf")
ap.add_argument("--strict", action="store_true")
ap.add_argument("--include-decision-log", action="store_true", help="append the decision log (off by default)")
args = ap.parse_args()

RES = ROOT / args.results_dir
SHOTS = ROOT / args.screenshots_dir
ASSETS = ROOT / "report" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / args.out
OUT.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ text safety (built-in PDF fonts are cp1252 only)
_MAP = {"\u2011": "-", "\u2010": "-", "\u2212": "-", "\u3010": "[", "\u3011": "]", "\u2192": "->", "\u2265": ">=",
        "\u2264": "<=", "\u00d7": "x", "\u2713": "ok", "\u00a0": " "}


def clean(s) -> str:
    s = str(s)
    for k, v in _MAP.items():
        s = s.replace(k, v)
    return s.encode("cp1252", "replace").decode("cp1252")


def esc(s) -> str:
    return clean(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------ inputs
cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
meta = {"title": "Hybrid Search Helpdesk Assistant", "subtitle": "Experiential Learning Assignment 1",
        "author": "YOUR NAME", "student_id": "YOUR BITS ID", "program": "M.Tech (AI/ML), BITS Pilani WILP",
        "course": "AIMLCZG521 Conversational AI", "submission_date": "", "group_note": "", "environment_note": ""}
if (ROOT / args.meta).exists():
    meta.update({k: v for k, v in (yaml.safe_load((ROOT / args.meta).read_text(encoding="utf-8")) or {}).items() if v is not None})

missing_inputs, missing_shots = [], []


def read_csv(name):
    p = RES / name
    if not p.exists():
        missing_inputs.append(name)
        return None
    return pd.read_csv(p)


ev = read_csv("stage5_evaluation.csv")
gs = read_csv("stage5_evaluation_guardrail_scores.csv")
e2e = read_csv("stage6_end_to_end.csv")
cal = read_csv("stage6_calibration.csv")
val = read_csv("stage6_validation_scores.csv")
corpus = json.loads((ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))
queries = json.loads((ROOT / "data" / "test_queries.json").read_text(encoding="utf-8"))
val_queries = json.loads((ROOT / "data" / "guardrail_validation.json").read_text(encoding="utf-8"))
log_path = ROOT / "docs" / "decision_log.md"

ORDER = ["bm25", "dense", "hybrid", "dense+rerank", "hybrid+rerank"]
NAMES = {"bm25": "BM25 (keywords)", "dense": "Dense (vectors)", "hybrid": "Hybrid (RRF)",
         "dense+rerank": "Dense + re-rank", "hybrid+rerank": "Hybrid + re-rank (default)"}
COL = {"bm25": "#9aa5b1", "dense": "#2a6fdb", "hybrid": "#e08a1e", "dense+rerank": "#1c9a6b", "hybrid+rerank": "#7a3db8"}
THR = cfg["guardrails"]["min_dense_cosine"]
RRFK = cfg["retrieval"]["fusion"]["rrf_k"]

# ------------------------------------------------------------------ figures
def draw_architecture(path):
    fig, ax = plt.subplots(figsize=(11, 6.3))
    ax.set_xlim(0, 110); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, text, fc, ec=None, fs=8.6, tc="#1b2430"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec or fc, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#4a5568", lw=1.3))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7.4, color="#4a5568", ha="center")

    ax.text(1, 59.5, "Request path (one question)", fontsize=11, fontweight="bold", color="#1f3b63")
    box(2, 46, 16, 8, "User\nquestion", "#e8eef7", "#1f3b63")
    box(24, 46, 24, 8, "1. Input guardrails\nmask personal data,\nblock injection", "#fde9cf", "#e08a1e")
    box(54, 46, 24, 8, f"2. Relevance gate\nbest vector cosine\n< {THR}: refuse", "#fde9cf", "#e08a1e")
    box(84, 46, 24, 8, "Polite refusal\n(LLM never called)", "#f9d9d9", "#d64545")
    arrow(18, 50, 24, 50); arrow(48, 50, 54, 50); arrow(78, 50, 84, 50, "below gate", 0, 1.2)
    box(26, 28, 24, 8, "BM25 keyword search\n(own implementation)\ntop 20", "#dfeafc", "#2a6fdb")
    box(58, 28, 24, 8, "Vector search\nMiniLM embeddings, cosine\ntop 20", "#dfeafc", "#2a6fdb")
    arrow(66, 46, 40, 36, "pass", -3.5, 1.2); arrow(66, 46, 70, 36)
    box(44, 14.5, 26, 8, f"3. Rank fusion (RRF, k={RRFK})\nagreeing articles rise", "#dcf2e8", "#1c9a6b")
    arrow(38, 28, 50, 22.5); arrow(70, 28, 64, 22.5)
    box(76, 14.5, 32, 8, "4. Cross-encoder re-rank\ntop 20 candidates -> top 3", "#e9defa", "#7a3db8")
    arrow(70, 18.5, 76, 18.5)
    box(76, 2, 32, 8, f"5. LLM answer (Groq)\n{cfg['generation']['model']}\nonly from the 3 articles, cites ids", "#e8eef7", "#1f3b63", fs=8)
    arrow(92, 14.5, 92, 10)
    box(40, 2, 30, 8, "6. Output checks\ncitations must be retrieved ids,\nNOT_IN_KB allowed", "#fde9cf", "#e08a1e")
    arrow(76, 6, 70, 6)
    box(2, 2, 30, 8, "Answer + sources shown in UI\nmasked request log (JSONL)", "#e8eef7", "#1f3b63")
    arrow(40, 6, 32, 6)
    ax.text(2, 28, "Retrieval\n(two views of\nthe same text)", fontsize=8, color="#4a5568", va="center")
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


def draw_charts():
    out = {}
    if ev is not None:
        methods = [m for m in ORDER if m in set(ev.method)]
        overall = ev.groupby("method")["mrr"].mean().reindex(methods)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.bar([NAMES[m].replace(" (", "\n(") for m in methods], overall.values, color=[COL[m] for m in methods])
        for i, v in enumerate(overall.values):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.08); ax.set_ylabel("MRR (higher is better)"); ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title("Retrieval quality by pipeline (all answerable queries)", fontsize=10)
        plt.tight_layout(); fig.savefig(ASSETS / "chart_overall.png", dpi=170); plt.close(fig)
        out["overall"] = ASSETS / "chart_overall.png"
        bt = ev.groupby(["type", "method"])["mrr"].mean().unstack().reindex(columns=methods)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        w = 0.8 / len(methods)
        for j, m in enumerate(methods):
            ax.bar(np.arange(len(bt)) + (j - (len(methods) - 1) / 2) * w, bt[m].values, w, label=NAMES[m], color=COL[m])
        ax.set_xticks(np.arange(len(bt))); ax.set_xticklabels(bt.index)
        ax.set_ylim(0, 1.15); ax.set_ylabel("MRR"); ax.legend(fontsize=6.5, ncol=3, loc="upper center")
        ax.set_title("Retrieval quality by query type", fontsize=10)
        plt.tight_layout(); fig.savefig(ASSETS / "chart_by_type.png", dpi=170); plt.close(fig)
        out["by_type"] = ASSETS / "chart_by_type.png"
    if gs is not None or val is not None:
        groups = []
        if gs is not None:
            groups += [("In-scope\n(test set)", gs[~gs.refuse].dense_cos.values, "#1c9a6b"),
                       ("Out-of-scope\n(test set)", gs[gs.refuse].dense_cos.values, "#7d8794")]
        if val is not None:
            groups += [("In-scope\n(validation)", val[val.in_scope].cosine.values, "#1c9a6b"),
                       ("Out-of-scope\n(validation)", val[~val.in_scope].cosine.values, "#7d8794")]
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for i, (lab, ys, c) in enumerate(groups):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(ys)), ys, color=c, s=26, alpha=0.85)
        ax.axhline(THR, color="#d64545", ls="--", lw=1.3); ax.text(len(groups) - 0.55, THR + 0.01, f"gate = {THR}", color="#d64545", fontsize=8, ha="right")
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.set_ylabel("best vector cosine"); ax.set_title("What the relevance gate sees", fontsize=10)
        plt.tight_layout(); fig.savefig(ASSETS / "chart_gate.png", dpi=170); plt.close(fig)
        out["gate"] = ASSETS / "chart_gate.png"
    draw_architecture(ASSETS / "architecture.png")
    out["arch"] = ASSETS / "architecture.png"
    return out


FIG = draw_charts()

# ------------------------------------------------------------------ PDF building blocks
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.8, leading=14, spaceAfter=6)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.2, leading=10.5, spaceAfter=0)
CAP = ParagraphStyle("cap", parent=BODY, fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=colors.HexColor("#4a5568"), alignment=TA_CENTER, spaceAfter=10)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#1f3b63"), spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor("#2a4d80"), spaceBefore=8, spaceAfter=4)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=colors.HexColor("#1f3b63"), spaceAfter=4)
W = A4[0] - 4 * cm


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="-") for t in items]


def table(rows, widths, header=True, font=8.2, align_right_from=None):
    st = ParagraphStyle("c", parent=SMALL, fontSize=font, leading=font + 2.2)
    sth = ParagraphStyle("ch", parent=st, textColor=colors.white, fontName="Helvetica-Bold")
    data = [[Paragraph(str(c), sth if (header and i == 0) else st) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c5cdd8")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b63")))
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f6fa")))
    t.setStyle(TableStyle(style))
    return t


counter = {"fig": 0, "tab": 0}


def caption(kind, text):
    counter[kind] += 1
    return P(f"{'Figure' if kind == 'fig' else 'Table'} {counter[kind]}: {esc(text)}", CAP)


def fit_image(path, max_w=W, max_h=11.5 * cm):
    w, h = PILImage.open(path).size
    s = min(max_w / w, max_h / h)
    return Image(str(path), width=w * s, height=h * s)


def figure(path, text):
    return KeepTogether([fit_image(path), caption("fig", text)])


class Placeholder(Flowable):
    def __init__(self, name, what, w=W, h=5 * cm):
        super().__init__()
        self.name, self.what, self.w, self.h = name, what, w, h

    def wrap(self, aw, ah):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setStrokeColor(colors.HexColor("#d64545")); c.setDash(4, 3); c.setFillColor(colors.HexColor("#fdf1f1"))
        c.rect(0, 0, self.w, self.h, fill=1)
        c.setDash(); c.setFillColor(colors.HexColor("#d64545")); c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(self.w / 2, self.h / 2 + 8, f"SCREENSHOT MISSING: {clean(self.name)}")
        c.setFont("Helvetica", 8.2); c.setFillColor(colors.HexColor("#5a1f1f"))
        c.drawCentredString(self.w / 2, self.h / 2 - 8, clean(self.what)[:110])


def shot(name, text, what, optional=False):
    p = SHOTS / name
    if p.exists():
        return [figure(p, text)]
    if optional:
        return []
    missing_shots.append((name, what))
    return [KeepTogether([Placeholder(name, what), caption("fig", text + " [placeholder]")])]


def note_missing(name):
    return [P(f"<font color='#d64545'><b>Results file {esc(name)} not found.</b> Run the experiments first.</font>")]


# ------------------------------------------------------------------ numbers used in the text
by_type = overall = None
n_ans = n_oos = 0
if ev is not None:
    methods = [m for m in ORDER if m in set(ev.method)]
    overall = ev.groupby("method")[["hit@1", "recall@3", "mrr"]].mean().reindex(methods)
    by_type = ev.groupby(["type", "method"])["mrr"].mean().unstack().reindex(columns=methods)
    n_ans = ev.qid.nunique()
n_oos = sum(q["expect_refusal"] for q in queries)
n_core = sum(q.get("set", "core") == "core" for q in queries)
n_ext = len(queries) - n_core

# ------------------------------------------------------------------ story
S = []
auth = esc(meta["author"]); sid = esc(meta["student_id"])
S += [Spacer(1, 1.2 * cm), P(esc(meta["title"]), TITLE), P(esc(meta["subtitle"]), H2), Spacer(1, 0.3 * cm),
      table([["Student", f"{auth} ({sid})"], ["Programme", esc(meta["program"])], ["Course", esc(meta["course"])],
             ["Date", esc(meta["submission_date"] or date.today().strftime("%d %B %Y"))]], [3.2 * cm, W - 3.2 * cm], header=False, font=9)]
if meta.get("group_note"):
    S.append(P(esc(meta["group_note"]), SMALL))
if meta.get("environment_note"):
    S.append(Spacer(1, 4)); S.append(P(esc(meta["environment_note"]), SMALL))
S.append(Spacer(1, 0.4 * cm))

# ---- 1. summary
S.append(P("1. Summary", H1))
S.append(P(
    "Enterprise helpdesks answer the same questions again and again, but employees rarely use the words the knowledge base uses. "
    "This project builds a small, configurable helpdesk assistant that combines <b>keyword search (BM25)</b> and "
    "<b>semantic vector search</b>, merges them with <b>Reciprocal Rank Fusion (RRF)</b>, re-orders the best candidates with a "
    "<b>cross-encoder re-ranker</b>, and lets an <b>LLM</b> answer only from the retrieved articles with citations. "
    "Three layers of <b>guardrails</b> (input checks, a relevance gate, output checks) keep it from answering what it should not. "
    "Everything was measured on labelled test queries, and the report states plainly where the design helped and where it did not."))
if overall is not None:
    best = overall.mrr.max()
    winners = [NAMES[m] for m in overall.index if abs(overall.mrr[m] - best) < 1e-9]
    S.append(P(f"<b>Headline result.</b> On {n_ans} answerable test queries the best pipeline by MRR was "
               f"<b>{esc(' and '.join(winners))}</b> (MRR {best:.3f}); keyword search alone scored {overall.mrr.get('bm25', float('nan')):.3f}."))
S.append(P("<b>Contribution list.</b>"))
S += bullets([
    "A complete pipeline: ingest, BM25, embeddings, RRF fusion, cross-encoder re-ranking, Groq LLM generation, guardrails and a Streamlit UI.",
    "An evaluation harness with labelled queries, metrics (hit@1, recall@3, MRR) and an honest comparison of five pipelines.",
    "Enterprise-style engineering: configuration file, reproducible runs, masked request logs, graceful degradation, automated tests.",
])

# ---- 2. problem
S.append(P("2. Problem and use case", H1))
S.append(P(
    "A typical IT and employee helpdesk receives questions such as <i>\"VPN error 809\"</i> and <i>\"I cannot remember my login credentials\"</i>. "
    "The first is an exact-term lookup, where keyword search is strong. The second shares no words with the article titled "
    "<i>\"Reset your password\"</i>, where keyword search fails and semantic search is needed. A useful assistant must handle both, "
    "must cite its sources, and must refuse questions it cannot answer (for example a personal Gmail problem, which sounds like helpdesk "
    "content but is outside the knowledge base)."))
S.append(P("<b>Scope.</b> The knowledge base is synthetic (40 short articles in 7 categories) so that no company data is involved. "
           "The same code is domain-agnostic: the corpus path, domain name and messages live in the configuration file."))

# ---- 3. concepts
S.append(P("3. Concepts in plain language", H1))
S.append(P("3.1 Retrieval-augmented generation (RAG)", H2))
S.append(P("An LLM on its own may invent policies. In RAG the system first <i>retrieves</i> the relevant articles and then asks the LLM to answer "
           "only from them and to cite them. Quality therefore depends first on retrieval, which is why most of this project measures retrieval."))
S.append(P("3.2 Keyword search: TF-IDF and BM25", H2))
S.append(P("<b>Input:</b> query words and article words. <b>Output:</b> a score per article. A word counts more when it is rare across articles "
           "(\"809\" matters more than \"error\"). BM25 adds two corrections: repeated words give diminishing returns (parameter <b>k1</b>), and long "
           f"articles are not favoured just for being long (parameter <b>b</b>). Settings used: k1 = {cfg['retrieval']['bm25']['k1']}, b = {cfg['retrieval']['bm25']['b']}. "
           "Keyword search matches <i>words</i>, not <i>meaning</i>."))
S.append(P("3.3 Semantic search: embeddings and cosine similarity", H2))
S.append(P(f"A small language model ({esc(cfg['retrieval']['embedding']['model'].split('/')[-1])}) turns every text into a vector of numbers so that texts with similar "
           "meaning lie close together. The score is the cosine of the angle between the query vector and the article vector (1 = same direction, "
           "0 = unrelated). Vectors are length-normalised, so cosine equals a dot product. Strength: matches synonyms and paraphrases. "
           "Weakness: it always returns something, even for a nonsense question, so a score threshold is needed."))
S.append(P("3.4 Rank fusion (RRF)", H2))
ex = 1 / (RRFK + 1) + 1 / (RRFK + 2)
S.append(P(f"BM25 scores (about 10) and cosine scores (about 0.4) are on different scales, so they cannot be added. RRF uses only ranks: "
           f"<b>score = sum over retrievers of 1 / (k + rank)</b>, with k = {RRFK}. Example: an article ranked 1st by BM25 and 2nd by the vector search scores "
           f"1/{RRFK + 1} + 1/{RRFK + 2} = {ex:.4f}. Articles that both retrievers like rise; articles only one retriever finds still survive."))
S.append(P("3.5 Re-ranking with a cross-encoder", H2))
S.append(P(f"The retrievers judge the query and each article separately. A cross-encoder ({esc(cfg['reranker']['model'].split('/')[-1])}) reads the query and one "
           f"article <i>together</i> and scores how well it answers. It is slower, so it only re-scores the top {cfg['reranker']['candidate_pool']} candidates and passes the "
           f"best {cfg['reranker']['top_n']} to the LLM. Pattern: cheap retrieval for recall, expensive re-ranking for precision."))
S.append(P("3.6 Guardrails", H2))
S += bullets([
    "<b>Input:</b> personal data (emails, phone numbers, card numbers, PAN, Aadhaar-style numbers) is masked before it reaches the LLM or the logs; obvious prompt-injection phrases are blocked.",
    f"<b>Relevance gate:</b> if the best article's cosine is below {THR}, the question is refused and the LLM is never called (saves cost, blocks unrelated questions).",
    "<b>Output:</b> the prompt tells the model to answer only from the supplied articles and to reply NOT_IN_KB otherwise; every citation must point to an article that was actually retrieved.",
])

# ---- 4. architecture
S.append(PageBreak())
S.append(P("4. System design", H1))
S.append(figure(FIG["arch"], "Architecture of the assistant: guardrails, parallel retrieval, fusion, re-ranking, grounded generation and output checks."))
mods = [["Module", "Role"],
        ["config.py, config/config.yaml", "Single source of truth for every tunable (models, thresholds, k1, b, RRF k, domain text)"],
        ["ingest.py, text.py", "Load and validate the corpus and test queries; shared tokenizer with stop-word removal and light stemming"],
        ["retrievers/lexical.py", "TF-IDF (scikit-learn) and BM25 written by hand so every term of the formula is visible"],
        ["embeddings.py, retrievers/dense.py", "Sentence embeddings, cosine search, on-disk cache keyed by model and article text"],
        ["retrievers/hybrid.py, reranked.py", "RRF fusion with provenance (which retriever ranked what); cross-encoder re-ranking wrapper"],
        ["guardrails.py, generation.py, pipeline.py", "PII masking, injection check, prompt, Groq client with fallback model, output validation, request log"],
        ["metrics.py, run_stage*.py", "hit@1, recall@3, MRR; one script per stage so each step is reproducible"],
        ["app.py, ui_helpers.py", "Streamlit chat UI with a colour-coded evidence panel"]]
S.append(table(mods, [5.2 * cm, W - 5.2 * cm]))
S.append(caption("tab", "Main modules and their responsibilities."))
S.append(P("4.1 Engineering practices", H2))
eng = [["Quality", "How it is implemented"],
       ["Configurable", "All settings in one YAML file; the domain can be swapped by changing configuration, not code"],
       ["Reproducible", "Fixed seed, requirements file with a freeze step, one script per stage that prints and saves its results"],
       ["Secure", "API key only in .env (git-ignored); personal data masked before the LLM and before logging"],
       ["Resilient", "Fallback LLM, then an extractive answer marked 'degraded' if the LLM is unavailable; no crash"],
       ["Observable", "One JSON line per request: status, sources, scores, latency (query already masked)"],
       ["Tested", "Automated pytest suite covering tokenizer, BM25 against a hand calculation, RRF, re-ranker wrapper, guardrails, pipeline and UI helpers"],
       ["Documented", "README with setup and run instructions for every stage, plus written design notes"]]
S.append(table(eng, [3.2 * cm, W - 3.2 * cm]))
S.append(caption("tab", "Enterprise-style engineering practices."))
S += shot("09_project_structure.png", "Project structure in the editor.", "VS Code Explorer showing config, data, src/helpdesk, tests, docs, results")
S += shot("01_app_overview.png", "The assistant after the example 'Error code (keyword)': answer with coloured citation, guardrail strip and the Evidence panel.",
          "Click 'Error code (keyword)' in the sidebar, capture the whole browser window")

# ---- 5. experiment design
S.append(PageBreak())
S.append(P("5. Experiment design", H1))
cats = pd.Series([a["category"] for a in corpus]).value_counts()
S.append(P(f"<b>Corpus.</b> {len(corpus)} synthetic articles: " + ", ".join(f"{esc(k)} ({v})" for k, v in cats.items()) + f". Each article is {min(len(a['text'].split()) for a in corpus)} to {max(len(a['text'].split()) for a in corpus)} words, so all articles are short and similar in length."))
qt = pd.DataFrame(queries)
qt["set"] = qt.get("set", "core")
piv = qt.groupby(["type", "set"]).size().unstack(fill_value=0)
rows = [["Query type", "core", "extended", "total"]] + [[esc(t), int(piv.loc[t].get("core", 0)), int(piv.loc[t].get("extended", 0)), int(piv.loc[t].sum())] for t in piv.index]
S.append(P(f"<b>Test queries.</b> {len(queries)} labelled queries: {n_core} in the <i>core</i> set and {n_ext} in an <i>extended</i> set. "
           "The extended queries were added after the first vector-search results, because the core queries were too easy to tell the pipelines apart (the vector search already scored high on them); "
           "they are reported separately so that the extension cannot hide anything. Each answerable query has the correct article id(s) marked; "
           f"{n_oos} queries are deliberately out of scope."))
S.append(table(rows, [5 * cm, 3 * cm, 3 * cm, 3 * cm]))
S.append(caption("tab", "Number of test queries by type and set."))
S.append(P("<b>Metrics.</b> <b>hit@1</b>: was a correct article ranked first? <b>recall@3</b>: share of the correct articles found in the top 3. "
           "<b>MRR</b> (mean reciprocal rank): 1 divided by the rank of the first correct article, averaged over queries (1.0 = always first). "
           "Out-of-scope queries are excluded from these metrics and evaluated by the guardrails instead."))
S.append(P("<b>Pipelines compared.</b> BM25; dense vectors; hybrid (RRF of the two); dense + re-rank; hybrid + re-rank. "
           "The re-ranked pipelines re-score the top 20 candidates."))

# ---- 6. results
S.append(P("6. Retrieval results", H1))
if ev is None:
    S += note_missing("stage5_evaluation.csv")
else:
    rows = [["Pipeline", "hit@1", "recall@3", "MRR"]] + [[NAMES[m], f"{overall.loc[m, 'hit@1']:.3f}", f"{overall.loc[m, 'recall@3']:.3f}", f"{overall.loc[m, 'mrr']:.3f}"] for m in overall.index]
    S.append(table(rows, [6.5 * cm, 3 * cm, 3 * cm, 3 * cm]))
    S.append(caption("tab", f"Retrieval quality on all {n_ans} answerable queries."))
    S.append(figure(FIG["overall"], "MRR by pipeline."))
    bs = ev.groupby(["set", "method"])["mrr"].mean().unstack().reindex(columns=methods)
    rows = [["Query set"] + [NAMES[m] for m in methods]] + [[esc(s)] + [f"{bs.loc[s, m]:.3f}" for m in methods] for s in bs.index]
    S.append(table(rows, [2.6 * cm] + [(W - 2.6 * cm) / len(methods)] * len(methods)))
    S.append(caption("tab", "MRR for the core queries and for the queries added later (extended)."))
    rows = [["Query type"] + [NAMES[m] for m in methods]] + [[esc(t)] + [f"{by_type.loc[t, m]:.3f}" for m in methods] for t in by_type.index]
    S.append(table(rows, [2.6 * cm] + [(W - 2.6 * cm) / len(methods)] * len(methods)))
    S.append(caption("tab", "MRR by query type."))
    S.append(figure(FIG["by_type"], "MRR by query type and pipeline."))

    # data-driven findings
    m = ev.pivot(index=["qid", "query"], columns="method", values="mrr")
    S.append(P("6.1 What the numbers say", H2))
    f = []
    if {"bm25", "dense"} <= set(m.columns) and "paraphrase" in by_type.index and "keyword" in by_type.index:
        f.append(f"<b>Each retriever has a home ground.</b> On keyword queries BM25 scored {by_type.loc['keyword', 'bm25']:.3f} and dense {by_type.loc['keyword', 'dense']:.3f}; "
                 f"on paraphrase queries BM25 scored {by_type.loc['paraphrase', 'bm25']:.3f} and dense {by_type.loc['paraphrase', 'dense']:.3f}.")
    if "hybrid" in m.columns:
        best_single = m[["bm25", "dense"]].max(axis=1)
        f.append(f"<b>Plain RRF fusion did not beat the best single retriever here.</b> Hybrid MRR was {overall.mrr['hybrid']:.3f} against {overall.mrr['dense']:.3f} for dense and "
                 f"{overall.mrr['bm25']:.3f} for BM25. It was better than both on {int((m['hybrid'] > best_single).sum())} queries, equal to the better one on "
                 f"{int((m['hybrid'] == best_single).sum())}, and below it on {int((m['hybrid'] < best_single).sum())}.")
    if {"dense", "dense+rerank"} <= set(m.columns):
        d = by_type["dense+rerank"] - by_type["dense"]
        gain = overall.mrr["dense+rerank"] - overall.mrr["dense"]
        verdict = "helped on average" if gain > 1e-9 else "did not help on average" if gain < -1e-9 else "made no difference on average"
        f.append(f"<b>Re-ranking {verdict}</b> (dense MRR {overall.mrr['dense']:.3f} to {overall.mrr['dense+rerank']:.3f}), and the effect differs by query type: "
                 + "; ".join(f"{esc(t)} {d[t]:+.3f}" for t in d.index) + " (change in MRR by query type).")
    if {"hybrid+rerank", "dense+rerank"} <= set(m.columns):
        tie = abs(overall.mrr["hybrid+rerank"] - overall.mrr["dense+rerank"]) < 1e-9
        f.append("<b>Hybrid + re-rank and dense + re-rank " + ("scored the same MRR" if tie else
                 f"scored {overall.mrr['hybrid+rerank']:.3f} and {overall.mrr['dense+rerank']:.3f}") + ".</b> "
                 "Hybrid + re-rank is the default for a design reason (keyword search stays as a safety net for exact terms), not because the data proves it is better.")
    S += bullets(f)
    if "hybrid" in m.columns and overall.mrr["hybrid"] <= overall.mrr["dense"]:
        S.append(P(
            f"<b>Why plain fusion did not help.</b> RRF rewards agreement. A wrong article that both retrievers rank moderately can outscore the right article that only one "
            f"retriever ranks first (for example 1/{RRFK + 1} + 1/{RRFK + 3} = {1 / (RRFK + 1) + 1 / (RRFK + 3):.4f} against 1/{RRFK + 1} = {1 / (RRFK + 1):.4f}). "
            "On a small, clean corpus the vector search alone is already strong, so the keyword list adds noise about as often as it adds signal. "
            "A re-ranker, which reads the query and article together, repairs many of these cases. On a larger or messier knowledge base the balance may differ; this was not tested."))
        worse = m[m["hybrid"] < m[["bm25", "dense"]].max(axis=1)].reset_index()
        if len(worse):
            rows = [["Query", "BM25", "Dense", "Hybrid", "Hybrid + re-rank"]]
            for _, r in worse.iterrows():
                rows.append([esc(r["query"]), f"{r['bm25']:.2f}", f"{r['dense']:.2f}", f"{r['hybrid']:.2f}", f"{r.get('hybrid+rerank', float('nan')):.2f}"])
            S.append(table(rows, [8.2 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, W - 13.6 * cm]))
            S.append(caption("tab", "Queries where hybrid (RRF) ranked the correct article lower than the better single retriever (MRR per query)."))
    diff = m[m.nunique(axis=1) > 1].reset_index()
    if len(diff):
        rows = [["Query"] + [NAMES[x].split(" (")[0] for x in methods]] + [[esc(r["query"])[:70]] + [f"{r[x]:.2f}" for x in methods] for _, r in diff.iterrows()]
        S.append(table(rows, [6.3 * cm] + [(W - 6.3 * cm) / len(methods)] * len(methods), font=7.6))
        S.append(caption("tab", "Every query on which the five pipelines did not all agree (MRR per query)."))
    S += shot("02_paraphrase.png", "A paraphrase question ('I cannot remember my login credentials') answered from the right article; the evidence panel shows how it was found.",
              "Click 'Different words, same meaning', capture answer + Evidence panel")

# ---- 7. guardrails
S.append(PageBreak())
S.append(P("7. Guardrails and end-to-end behaviour", H1))
if e2e is None:
    S += note_missing("stage6_end_to_end.csv")
else:
    e2e["cited"] = e2e["cited"].fillna("")
    lab = e2e[e2e.in_scope & e2e.type.isin(["keyword", "paraphrase", "mixed"])].copy()
    refused_set = ["refused_out_of_scope", "no_answer", "refused_injection"]

    def outcome(r):
        if r.status in refused_set:
            return "declined"
        if r.status == "uncited":
            return "answered, no valid citation"
        return "answered with relevant citation" if r.cites_relevant else "answered, wrong citation"

    lab["outcome"] = lab.apply(outcome, axis=1)
    oos = e2e[~e2e.in_scope]
    gate_n, llm_n = int((oos.status == "refused_out_of_scope").sum()), int((oos.status == "no_answer").sum())
    leaks = oos[~oos.status.isin(refused_set)]
    S.append(P(f"The full pipeline was run on all {len(e2e)} queries (the {len(queries)} test queries plus {len(val_queries)} separate validation queries written to check the gate), "
               f"with the LLM <b>{esc(cfg['generation']['model'])}</b>."))
    ct = pd.crosstab(e2e.in_scope.map({True: "in scope", False: "out of scope"}), e2e.status)
    rows = [["Query group"] + list(ct.columns)] + [[esc(i)] + [int(v) for v in ct.loc[i]] for i in ct.index]
    S.append(table(rows, [3 * cm] + [(W - 3 * cm) / len(ct.columns)] * len(ct.columns), font=7.6))
    S.append(caption("tab", "Final status of every query, by whether it was in scope."))
    oc = lab.outcome.value_counts()
    S.append(P("<b>Labelled in-scope queries (" + str(len(lab)) + "):</b> " + ", ".join(f"{int(v)} {esc(k)}" for k, v in oc.items()) + "."))
    S.append(P(f"<b>Out-of-scope queries ({len(oos)}):</b> {gate_n + llm_n} refused ({gate_n} by the relevance gate, {llm_n} by the LLM answering NOT_IN_KB), "
               f"{len(leaks)} answered." + ("" if not len(leaks) else " Leaks: " + "; ".join(esc(q) for q in leaks["query"]) + ".")))
    trap = oos[oos.status == "no_answer"]
    if len(trap):
        S.append(P("<b>Why one layer is not enough.</b> The questions that passed the gate but were declined by the LLM were: " +
                   "; ".join(f"<i>{esc(r['query'])}</i> (best cosine {r['cosine']:.2f})" for _, r in trap.iterrows()) +
                   ". They sound like helpdesk questions, so their similarity is above the gate; no single threshold can separate them from real questions without refusing real users. "
                   "The prompt rule (answer only from the articles, otherwise NOT_IN_KB) is the second layer that catches them."))
    declined = lab[lab.outcome == "declined"]
    if len(declined):
        S.append(P("<b>Declined in-scope queries:</b> " + "; ".join(f"<i>{esc(r['query'])}</i>" for _, r in declined.iterrows()) +
                   ". A decline can be the correct behaviour when the article does not actually contain the requested detail, and the test labels here are retrieval labels, not answer labels."))
    if "latency_ms" in e2e.columns:
        ans = e2e[e2e.status.isin(["answered", "uncited", "degraded"])].latency_ms.dropna()
        gate = e2e[e2e.status == "refused_out_of_scope"].latency_ms.dropna()
        if len(ans) and len(gate):
            S.append(P(f"<b>Latency.</b> Median {ans.median() / 1000:.1f} s for an LLM-answered question; median {gate.median():.0f} ms for a question refused by the gate (no LLM call)."))
if cal is not None:
    rows = [["Gate threshold", "Test: in-scope kept", "Test: out-of-scope refused", "Validation: in-scope kept", "Validation: out-of-scope refused"]]
    for _, r in cal.iterrows():
        mark = " *" if abs(r.threshold - THR) < 1e-9 else ""
        rows.append([f"{r.threshold:.2f}{mark}", f"{r.cal_in_kept:.0%}", f"{r.cal_oos_refused:.0%}", f"{r.val_in_kept:.0%}", f"{r.val_oos_refused:.0%}"])
    S.append(table(rows, [3 * cm] + [(W - 3 * cm) / 4] * 4, font=7.8))
    S.append(caption("tab", f"Effect of the relevance-gate threshold (* = configured value {THR}). The threshold was chosen on the test queries, so the validation columns are the fairer check."))
if "gate" in FIG:
    S.append(figure(FIG["gate"], "Best vector cosine of each query: in-scope queries sit above the gate, most out-of-scope queries below it, and a few near-domain traps sit in between."))
S += shot("03_pii_masking.png", "Personal data is masked before the LLM; the guardrail strip shows what was masked.", "Click 'Personal data (PII)', capture answer + guardrail chips")
S += shot("04_request_log.png", "The request log stores the masked query, never the raw email address.", "Show the last line of logs/requests.jsonl in VS Code or a terminal (query contains [EMAIL])")
S += shot("05_out_of_scope.png", "An out-of-scope question is refused by the relevance gate without calling the LLM.", "Click 'Out of scope', capture the answer + Evidence panel")
S += shot("06_near_domain_trap.png", "A near-domain trap (personal Gmail) passes the gate but the grounded prompt makes the model answer 'not in the knowledge base'.", "Click 'Near-domain trap', capture the result")
S += shot("07_injection_blocked.png", "A prompt-injection attempt is blocked before any retrieval.", "Click 'Prompt injection', capture the result")
if e2e is not None:
    S.append(P("7.1 Example answers", H2))
    rows = [["Query", "Outcome", "Answer (shortened)"]]
    for q in ["Q-01", "Q-06", "Q-22", "V-12", "Q-16"]:
        r = e2e[e2e.qid == q]
        if len(r):
            r = r.iloc[0]
            rows.append([esc(r["query"]), esc(r["status"]), esc(str(r["answer"])[:330] if str(r["answer"]) != "nan" else "")])
    S.append(table(rows, [4.6 * cm, 2.8 * cm, W - 7.4 * cm], font=7.6))
    S.append(caption("tab", "Sample outputs (keyword, paraphrase, out of scope and a near-domain trap)."))

# ---- 8. testing
S.append(P("8. Testing", H1))
S.append(P("An automated pytest suite checks the pieces that are easy to get subtly wrong: the tokenizer (error codes stay whole), BM25 against a hand-calculated score, "
           "RRF against a hand-calculated example, caching of embeddings, the re-ranker wrapper, PII masking (including a Luhn check for card numbers), injection detection, "
           "citation parsing, graceful degradation when the LLM fails, and that the request log never contains a raw email address."))
S += shot("08_pytest.png", "All automated tests passing.", "Terminal: python -m pytest -q showing all tests passed")
S += shot("10_stage6_summary.png", "End-to-end evaluation summary from run_stage6.py.", "Terminal: the status counts and summary lines from python run_stage6.py", optional=True)

# ---- 9. limitations
S.append(P("9. Limitations and threats to validity", H1))
S += bullets([
    f"<b>Small, synthetic data.</b> 40 short articles and {len(queries)} + {len(val_queries)} queries written by the author. Differences of one or two queries are not statistically meaningful, and no confidence intervals were computed.",
    "<b>Test set extended after seeing results.</b> The extended queries were added after the first vector-search run; they are reported separately, but the author's choices still shaped the test.",
    f"<b>Thresholds tuned on small sets.</b> The gate value {THR} sits between observed scores of a handful of out-of-scope queries; the margin is narrow (see the gate figure).",
    "<b>Citation relevance is a proxy.</b> 'Cites a relevant article' does not prove the answer text is correct. No human or LLM judge was used.",
    "<b>Regex guardrails are a first line only.</b> Unusual personal-data formats or a determined injection attempt can get through.",
    "<b>LLM output is not fully deterministic</b> (low temperature, hosted free tier), so end-to-end numbers can vary slightly between runs. Retrieval numbers are deterministic.",
    "<b>The re-ranker was trained on web-search data</b>, not conversational helpdesk text, which may explain why it does not help every paraphrase query.",
])

# ---- 10. next steps and learnings
S.append(P("10. Next steps", H1))
S += bullets([
    "<b>Graph RAG for relational questions.</b> Hybrid search matches text. Questions that follow relationships between entities (for example which suppliers' candidates are linked to which open vacancies and unpaid timesheets in a workforce-management system) need a knowledge graph; the retrieval layer built here can serve as the text half of such a system.",
    "<b>A query router (agent).</b> Decide per question whether to use text search, a graph query, or a tool call.",
    "<b>Better evaluation.</b> A larger and messier corpus, human-labelled answers, and an LLM-as-judge check for groundedness.",
    "<b>Fine-tune or swap the re-ranker</b> on helpdesk-style data; tune fusion weights on a held-out set.",
])
S.append(P("11. What I learned", H1))
S += bullets([
    "Keyword and semantic search fail in opposite ways, and measuring both on labelled queries is the only way to know which matters for a given corpus.",
    "A clever component (RRF fusion) does not automatically beat a strong baseline; a result that contradicts the expectation is still a finding.",
    "Guardrails work best in layers, because a similarity threshold cannot catch questions that merely sound relevant.",
    "Small engineering habits (configuration, logging, tests, masked logs) are what make a prototype believable as an enterprise component.",
])

# ---- references
S.append(P("References", H2))
S += bullets([
    "Cormack, Clarke and Buettcher (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR.",
    "Robertson and Zaragoza (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval.",
    "Reimers and Gurevych (2019). Sentence-BERT: sentence embeddings using Siamese BERT-networks. EMNLP.",
    "Lewis et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.",
])

# ---- optional appendix: decision log (OFF by default: it may contain internal notes; reproduction steps live in the README)
if args.include_decision_log and log_path.exists():
    S.append(PageBreak())
    S.append(P("Appendix A. Decision log (extract)", H1))
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m_ = re.match(r"\|\s*(D\d+)\s*\|(.+?)\|(.+)\|\s*$", line)
        if m_:
            rows.append((int(m_.group(1)[1:]), m_.group(1), m_.group(2).strip(), m_.group(3).strip()))
    rows.sort()
    if rows:
        S.append(table([["#", "Decision", "Why"]] + [[r[1], esc(r[2]), esc(r[3])] for r in rows], [1.1 * cm, 6 * cm, W - 7.1 * cm], font=7.4))


# ------------------------------------------------------------------ page furniture and build
def furniture(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 8); canv.setFillColor(colors.HexColor("#6b7280"))
    canv.drawString(2 * cm, 1.2 * cm, clean(f"{meta['course']}  |  {meta['author']}"))
    canv.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canv.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title=clean(meta["title"]), author=clean(meta["author"]))
doc.build(S, onFirstPage=furniture, onLaterPages=furniture)

# ------------------------------------------------------------------ summary for the user
try:
    from pypdf import PdfReader
    pages = len(PdfReader(str(OUT)).pages)
except Exception:
    pages = "?"
print(f"\nReport written: {OUT}  ({pages} pages)")
if missing_inputs:
    print("\nMissing result files (run the experiments first):")
    for n in missing_inputs:
        print(f"  - {n}")
if missing_shots:
    print(f"\n{len(missing_shots)} screenshot(s) still missing; red placeholders were inserted:")
    for n, w in missing_shots:
        print(f"  - {SHOTS / n}   ({w})")
if "YOUR" in str(meta["author"]) + str(meta["student_id"]):
    print("\nEdit report/report_meta.yaml: author or student id is still a placeholder.")
if args.strict and (missing_shots or missing_inputs):
    sys.exit(1)
