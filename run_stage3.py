"""Stage 3: add the vector retriever and compare it with the keyword retrievers.

  python run_stage3.py                      full comparison (needs the embedding model: first run downloads ~90 MB)
  python run_stage3.py --query "text"       top results from all three retrievers for any query
  python run_stage3.py --backend hashing    offline dry run to check the pipeline (NOT semantic; do not report these numbers)
"""
import argparse
import sys
import numpy as np
sys.path.insert(0, "src")
import pandas as pd
from helpdesk.config import load_config, resolve_path, PROJECT_ROOT
from helpdesk.logging_setup import setup_logging
from helpdesk.ingest import load_corpus, load_test_queries
from helpdesk.embeddings import build_encoder
from helpdesk.retrievers import TfidfRetriever, BM25Retriever, DenseRetriever
from helpdesk.metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--query")
ap.add_argument("--backend")
args = ap.parse_args()

cfg = load_config()
setup_logging(cfg)
docs = load_corpus(resolve_path(cfg, "corpus"))
queries = load_test_queries(resolve_path(cfg, "test_queries"), [d.id for d in docs])
title = {d.id: d.title for d in docs}
lex, bm, k = cfg["lexical"], cfg["retrieval"]["bm25"], cfg["retrieval"]["top_k"]

encoder = build_encoder(cfg, args.backend)
dense = DenseRetriever(docs, encoder, PROJECT_ROOT / cfg["retrieval"]["embedding"]["cache_dir"])
tfidf = TfidfRetriever(docs, lex)
bm25 = BM25Retriever(docs, lex, bm["k1"], bm["b"])
retrievers = [tfidf, bm25, dense]
label = "HASHING DRY RUN (not semantic)" if args.backend == "hashing" else "real embeddings"

if args.query:
    for r in retrievers:
        print(f"\n[{r.name}] {args.query}")
        for h in r.search(args.query, k):
            print(f"  {h.rank}. {h.doc_id}  {h.score:.3f}  {title[h.doc_id]}")
    sys.exit()

df = pd.DataFrame(sum((evaluate(r, queries, k) for r in retrievers), []))
pd.set_option("display.width", 220, "display.max_colwidth", 46)
print(f"\n=== Stage 3 ({label}) ===")
print("\nPer-query top3")
print(df.pivot(index=["qid", "type", "query", "relevant"], columns="method", values="top3")[["bm25", "dense"]]
        .reset_index().to_string(index=False))

print("\nSummary by query type (mean)")
print(df.groupby(["type", "method"])[["hit@1", "recall@3", "mrr"]].mean().round(3).to_string())
print("\nOverall")
print(df.groupby("method")[["hit@1", "recall@3", "mrr"]].mean().round(3).to_string())

m = df.pivot(index=["qid", "query"], columns="method", values="mrr")
print("\nWhere dense and BM25 disagree (MRR per query) -> the reason to fuse them in Stage 4")
diff = m[m["dense"] != m["bm25"]][["bm25", "dense"]]
print(diff.to_string() if len(diff) else "  (none)")

out = PROJECT_ROOT / cfg["paths"]["results_dir"] / ("stage3_dry_run.csv" if args.backend == "hashing" else "stage3_lexical_vs_dense.csv")
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved {out}")

print("\nTop-1 cosine score (preview for the guardrail threshold in Stage 5)")
inscope = [dense.search(q.query, 1)[0].score for q in queries if not q.expect_refusal]
print(f"  answerable queries : min {min(inscope):.3f}  median {np.median(inscope):.3f}  max {max(inscope):.3f}")
for q in queries:
    if q.expect_refusal:
        h = dense.search(q.query, 1)[0]
        print(f"  out of scope {q.query!r}: {h.score:.3f} ({h.doc_id})")
