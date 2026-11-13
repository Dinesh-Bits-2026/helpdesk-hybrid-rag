"""Stage 5: add the re-ranker and run the full evaluation of every pipeline.

  python run_stage5.py                 full evaluation; saves CSV and charts to results/
  python run_stage5.py --query "text"  show each stage: BM25, dense, fused, re-ranked
  python run_stage5.py --backend hashing --rerank-backend overlap   offline pipeline check only (never report)
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
from helpdesk.rerank import Reranker, build_scorer
from helpdesk.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankedRetriever
from helpdesk.metrics import evaluate, hit_at_k

ap = argparse.ArgumentParser()
ap.add_argument("--query")
ap.add_argument("--backend")
ap.add_argument("--rerank-backend")
ap.add_argument("--no-charts", action="store_true")
args = ap.parse_args()

cfg = load_config()
setup_logging(cfg)
docs = load_corpus(resolve_path(cfg, "corpus"))
queries = load_test_queries(resolve_path(cfg, "test_queries"), [d.id for d in docs])
title = {d.id: d.title for d in docs}
R, k = cfg["retrieval"], cfg["retrieval"]["top_k"]
RR = cfg["reranker"]

bm25 = BM25Retriever(docs, cfg["lexical"], R["bm25"]["k1"], R["bm25"]["b"])
dense = DenseRetriever(docs, build_encoder(cfg, args.backend), PROJECT_ROOT / R["embedding"]["cache_dir"])
fz = R["fusion"]
hybrid = HybridRetriever([bm25, dense], fz["rrf_k"], R["candidate_pool"], fz.get("weights"))
reranker = Reranker(build_scorer(cfg, args.rerank_backend), docs)
dense_rr = RerankedRetriever(dense, reranker, RR["candidate_pool"])
hybrid_rr = RerankedRetriever(hybrid, reranker, RR["candidate_pool"])
pipelines = [bm25, dense, hybrid, dense_rr, hybrid_rr]

if args.query:
    q = args.query
    print(f"\nQuery: {q}")
    for r in (bm25, dense):
        print(f"\n[{r.name}]")
        for h in r.search(q, 5):
            print(f"  {h.rank}. {h.doc_id}  {h.score:.3f}  {title[h.doc_id]}")
    print("\n[hybrid RRF]")
    for x in hybrid.search_detailed(q, 5):
        print(f"  {x['rank']}. {x['doc_id']}  rrf={x['rrf_score']:.4f}  {title[x['doc_id']]}   ranked by {x['ranks']}")
    print("\n[hybrid + re-rank]  (first_stage = rank before re-ranking)")
    for x in hybrid_rr.search_detailed(q, 5):
        print(f"  {x['rank']}. {x['doc_id']}  score={x['score']:.3f}  {title[x['doc_id']]}   first_stage={x['first_stage_rank']}")
    sys.exit()

label = "DRY RUN (not real)" if (args.backend == "hashing" or args.rerank_backend == "overlap") else "real models"
df = pd.DataFrame(sum((evaluate(r, queries, k) for r in pipelines), []))
order = [p.name for p in pipelines]
pd.set_option("display.width", 240, "display.max_colwidth", 40)
cols = ["hit@1", "recall@3", "mrr"]
print(f"\n=== Stage 5 ({label}) ===")


def table(group):
    t = df.groupby(group + ["method"])[cols].mean().round(3)
    return t.reindex(order, level="method") if len(group) == 0 else t


print("\nOverall, all answerable queries")
print(df.groupby("method")[cols].mean().round(3).reindex(order).to_string())
print("\nBy query set (core = original; extended = added after Stage 3)")
print(df.groupby(["set", "method"])[cols].mean().round(3).to_string())
print("\nBy query type")
print(df.groupby(["type", "method"])[cols].mean().round(3).to_string())

m = df.pivot(index=["qid", "query"], columns="method", values="mrr")[order]
print("\nPer-query MRR where the pipelines are not all equal")
print(m[m.nunique(axis=1) > 1].round(2).to_string())

print("\nHead to head by MRR (queries where the two differ)")
for a, b in (("hybrid+rerank", "dense"), ("hybrid+rerank", "hybrid"), ("dense+rerank", "dense")):
    print(f"  {a:14} vs {b:8}: better on {(m[a] > m[b]).sum():2d}, worse on {(m[a] < m[b]).sum():2d}, same on {(m[a] == m[b]).sum():2d}")

print("\nCandidate recall: how often does the fused list CONTAIN a correct article within the top N?")
ans = [q for q in queries if not q.expect_refusal]
for n in (3, 5, 10, 20):
    r = np.mean([hit_at_k([h.doc_id for h in hybrid.search(q.query, n)], q.relevant_ids, n) for q in ans])
    print(f"  hybrid top {n:<2}: {r:.3f}")

print("\nGuardrail preview: top-1 score of answerable vs out-of-scope queries (threshold is chosen in Stage 6)")
rows = []
for q in queries:
    rows.append({"qid": q.id, "refuse": q.expect_refusal, "query": q.query,
                 "dense_cos": dense.search(q.query, 1)[0].score,
                 "rerank": hybrid_rr.search(q.query, 1)[0].score})
g = pd.DataFrame(rows)
for col in ("dense_cos", "rerank"):
    a, o = g[~g.refuse][col], g[g.refuse][col]
    print(f"  {col:9} answerable: min {a.min():.3f} median {a.median():.3f} | out of scope: max {o.max():.3f} "
          f"| gap (lowest answerable - highest out of scope) = {a.min() - o.max():+.3f}")
print(g[g.refuse][["qid", "query", "dense_cos", "rerank"]].round(3).to_string(index=False))
low = g[~g.refuse].nsmallest(3, "rerank")[["qid", "query", "dense_cos", "rerank"]].round(3)
print("\nThree lowest-scoring ANSWERABLE queries (the ones a threshold would wrongly refuse first)")
print(low.to_string(index=False))

res = PROJECT_ROOT / cfg["paths"]["results_dir"]
res.mkdir(exist_ok=True)
tag = "stage5_dry_run" if label.startswith("DRY") else "stage5_evaluation"
df.to_csv(res / f"{tag}.csv", index=False)
g.to_csv(res / f"{tag}_guardrail_scores.csv", index=False)
print(f"\nSaved {res / (tag + '.csv')}")

if not args.no_charts and not label.startswith("DRY"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"bm25": "#9aa5b1", "dense": "#2a6fdb", "hybrid": "#e08a1e", "dense+rerank": "#1c9a6b", "hybrid+rerank": "#7a3db8"}
    overall = df.groupby("method")["mrr"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(order, overall.values, color=[colors[o] for o in order])
    for i, v in enumerate(overall.values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("MRR (higher is better)"); ax.set_title("Retrieval quality by pipeline (all answerable queries)")
    plt.xticks(rotation=15); plt.tight_layout(); fig.savefig(res / "chart_mrr_overall.png", dpi=160); plt.close(fig)
    bt = df.groupby(["type", "method"])["mrr"].mean().unstack().reindex(columns=order)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.16
    for j, o in enumerate(order):
        ax.bar(np.arange(len(bt)) + (j - 2) * w, bt[o].values, w, label=o, color=colors[o])
    ax.set_xticks(np.arange(len(bt))); ax.set_xticklabels(bt.index)
    ax.set_ylim(0, 1.1); ax.set_ylabel("MRR"); ax.set_title("Retrieval quality by query type"); ax.legend(fontsize=8, ncol=3)
    plt.tight_layout(); fig.savefig(res / "chart_mrr_by_type.png", dpi=160); plt.close(fig)
    print(f"Saved charts to {res}")
