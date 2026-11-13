"""Stage 4: RRF fusion of BM25 + dense, compared with each alone.

  python run_stage4.py                 full comparison; saves results/stage4_hybrid_results.csv
  python run_stage4.py --query "text"  fused result with which retriever ranked each article where
  python run_stage4.py --sweep         effect of the RRF constant k (expect a small effect)
  python run_stage4.py --backend hashing   offline pipeline check only (NOT semantic; never report)
"""
import argparse
import sys
sys.path.insert(0, "src")
import pandas as pd
from helpdesk.config import load_config, resolve_path, PROJECT_ROOT
from helpdesk.logging_setup import setup_logging
from helpdesk.ingest import load_corpus, load_test_queries
from helpdesk.embeddings import build_encoder
from helpdesk.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from helpdesk.metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--query")
ap.add_argument("--sweep", action="store_true")
ap.add_argument("--backend")
args = ap.parse_args()

cfg = load_config()
setup_logging(cfg)
docs = load_corpus(resolve_path(cfg, "corpus"))
queries = load_test_queries(resolve_path(cfg, "test_queries"), [d.id for d in docs])
title = {d.id: d.title for d in docs}
R, k = cfg["retrieval"], cfg["retrieval"]["top_k"]

bm25 = BM25Retriever(docs, cfg["lexical"], R["bm25"]["k1"], R["bm25"]["b"])
dense = DenseRetriever(docs, build_encoder(cfg, args.backend), PROJECT_ROOT / R["embedding"]["cache_dir"])
fusion = R["fusion"]
hybrid = HybridRetriever([bm25, dense], fusion["rrf_k"], R["candidate_pool"], fusion.get("weights"))

if args.query:
    print(f"\n[hybrid RRF] {args.query}")
    for x in hybrid.search_detailed(args.query, k):
        print(f"  {x['rank']}. {x['doc_id']}  rrf={x['rrf_score']:.4f}  {title[x['doc_id']]}")
        print(f"       ranked by: {x['ranks']}")
    sys.exit()

label = "HASHING DRY RUN (not semantic)" if args.backend == "hashing" else "real embeddings"
df = pd.DataFrame(sum((evaluate(r, queries, k) for r in (bm25, dense, hybrid)), []))
pd.set_option("display.width", 230, "display.max_colwidth", 44)
print(f"\n=== Stage 4 ({label}) ===")
print("\nPer-query top3")
print(df.pivot(index=["qid", "set", "type", "query", "relevant"], columns="method", values="top3")
        [["bm25", "dense", "hybrid"]].reset_index().to_string(index=False))

cols = ["hit@1", "recall@3", "mrr"]
print("\nOverall, by query set (core = original 15 answerable; extended = added after Stage 3)")
print(df.groupby(["set", "method"])[cols].mean().round(3).to_string())
allrows = df.groupby("method")[cols].mean().round(3)
print("\nOverall, all answerable queries")
print(allrows.to_string())
print("\nBy query type, all queries")
print(df.groupby(["type", "method"])[cols].mean().round(3).to_string())

m = df.pivot(index=["qid", "query"], columns="method", values="mrr")
best_single = m[["bm25", "dense"]].max(axis=1)
worst_single = m[["bm25", "dense"]].min(axis=1)
print("\nHybrid vs the individual retrievers (per query, by MRR)")
print(f"  hybrid better than BOTH single retrievers : {(m['hybrid'] > best_single).sum()}")
print(f"  hybrid equal to the best single           : {(m['hybrid'] == best_single).sum()}")
print(f"  hybrid below the best but above the worst : {((m['hybrid'] < best_single) & (m['hybrid'] > worst_single)).sum()}")
print(f"  hybrid at or below the worst              : {((m['hybrid'] <= worst_single) & (m['hybrid'] < best_single)).sum()}")
loss = m[m["hybrid"] < best_single]
print("\nQueries where hybrid is worse than the best single retriever")
print(loss.round(3).to_string() if len(loss) else "  (none)")

out = PROJECT_ROOT / cfg["paths"]["results_dir"] / ("stage4_dry_run.csv" if args.backend == "hashing" else "stage4_hybrid_results.csv")
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved {out}")

if args.sweep:
    print("\nRRF constant k vs overall MRR / hit@1 (all answerable queries)")
    for kk in (1, 10, 30, 60, 100, 300):
        r = pd.DataFrame(evaluate(HybridRetriever([bm25, dense], kk, R["candidate_pool"], fusion.get("weights")), queries, k))
        print(f"  rrf_k={kk:<4} mrr={r['mrr'].mean():.3f}  hit@1={r['hit@1'].mean():.3f}")
