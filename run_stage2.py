"""Stage 2: compare TF-IDF and BM25 on the labelled test queries.

  python run_stage2.py                  evaluation table + per-type summary + saves a CSV
  python run_stage2.py --sweep          also try different k1 / b values for BM25
  python run_stage2.py --query "text"   see both retrievers' top results for any query
"""
import argparse
import sys
sys.path.insert(0, "src")
import pandas as pd
from helpdesk.config import load_config, resolve_path, PROJECT_ROOT
from helpdesk.logging_setup import setup_logging
from helpdesk.ingest import load_corpus, load_test_queries
from helpdesk.retrievers import TfidfRetriever, BM25Retriever
from helpdesk.metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", action="store_true")
ap.add_argument("--query")
args = ap.parse_args()

cfg = load_config()
setup_logging(cfg)
docs = load_corpus(resolve_path(cfg, "corpus"))
queries = load_test_queries(resolve_path(cfg, "test_queries"), [d.id for d in docs])
title = {d.id: d.title for d in docs}
lex, bm = cfg["lexical"], cfg["retrieval"]["bm25"]
k = cfg["retrieval"]["top_k"]

tfidf = TfidfRetriever(docs, lex)
bm25 = BM25Retriever(docs, lex, bm["k1"], bm["b"])

if args.query:
    for r in (tfidf, bm25):
        print(f"\n[{r.name}] {args.query}")
        hits = r.search(args.query, k)
        for h in hits or []:
            print(f"  {h.rank}. {h.doc_id}  {h.score:.3f}  {title[h.doc_id]}")
        if not hits:
            print("  (no article shares any word with this query)")
    sys.exit()

rows = evaluate(tfidf, queries, k) + evaluate(bm25, queries, k)
df = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_colwidth", 48)

print("\nPer-query results (top3 = best three article ids)")
wide = df.pivot(index=["qid", "type", "query", "relevant"], columns="method", values="top3").reset_index()
print(wide.to_string(index=False))

print("\nSummary by query type (mean)")
summary = df.groupby(["type", "method"])[["hit@1", "recall@3", "mrr"]].mean().round(3)
print(summary.to_string())
print("\nOverall (mean over answerable queries)")
print(df.groupby("method")[["hit@1", "recall@3", "mrr"]].mean().round(3).to_string())

out = PROJECT_ROOT / cfg["paths"]["results_dir"] / "stage2_lexical_results.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved {out}")

print("\nOut-of-scope queries: top BM25 score (preview for the guardrail threshold in Stage 5)")
for q in queries:
    if q.expect_refusal:
        h = bm25.search(q.query, 1)
        print(f"  {q.query!r}: {h[0].score:.3f} ({h[0].doc_id})" if h else f"  {q.query!r}: no match")

if args.sweep:
    print("\nBM25 sweep: overall MRR for different k1 (rows) and b (columns)")
    grid = {}
    for k1 in (0.0, 0.5, 1.2, 1.5, 2.0, 3.0):
        grid[k1] = {b: pd.DataFrame(evaluate(BM25Retriever(docs, lex, k1, b), queries, k))["mrr"].mean()
                    for b in (0.0, 0.25, 0.5, 0.75, 1.0)}
    print(pd.DataFrame(grid).T.round(3).to_string())
