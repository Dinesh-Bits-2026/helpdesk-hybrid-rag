"""Stage 6: the full assistant (guardrails + retrieval + Groq answer + output checks).

  python run_stage6.py --calibrate          choose / check the 'I don't know' threshold (no LLM, no API key needed)
  python run_stage6.py --ask "question"     one question through the whole pipeline
  python run_stage6.py                      end-to-end evaluation on all test queries (uses Groq, ~40 calls)
  Add --no-llm to use the extractive fallback instead of Groq.
  Offline pipeline check only: --backend hashing --rerank-backend overlap --no-llm   (never report these numbers)
"""
import argparse
import json
import sys
import time
import numpy as np
sys.path.insert(0, "src")
import pandas as pd
from helpdesk.config import load_config, resolve_path, PROJECT_ROOT
from helpdesk.logging_setup import setup_logging
from helpdesk.ingest import load_corpus, load_test_queries
from helpdesk.pipeline import build_assistant

ap = argparse.ArgumentParser()
ap.add_argument("--ask")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--no-llm", action="store_true")
ap.add_argument("--qids", help="comma-separated ids to run only a subset, e.g. Q-04,Q-10")
ap.add_argument("--backend")
ap.add_argument("--rerank-backend")
ap.add_argument("--pipeline")
ap.add_argument("--sleep", type=float, default=2.0, help="seconds between Groq calls (free-tier rate limit)")
args = ap.parse_args()

cfg = load_config()
setup_logging(cfg)
docs = load_corpus(resolve_path(cfg, "corpus"))
queries = load_test_queries(resolve_path(cfg, "test_queries"), [d.id for d in docs])
validation = json.loads((PROJECT_ROOT / "data" / "guardrail_validation.json").read_text(encoding="utf-8"))
assistant, parts = build_assistant(cfg, docs, args.backend, args.rerank_backend, not args.no_llm, args.pipeline)
dense = parts["dense"]
thr = cfg["guardrails"]["min_dense_cosine"]
dry = args.backend == "hashing" or args.rerank_backend == "overlap"
pd.set_option("display.width", 220, "display.max_colwidth", 70)

if args.ask:
    r = assistant.answer(args.ask)
    print(f"\nStatus : {r.status}   (model: {r.model or '-'}, best cosine: {r.top_dense_cosine}, PII masked: {r.pii_types or 'none'})")
    print(f"Answer : {r.answer}")
    for s in r.sources:
        print(f"  source {s['id']}  {s['title']}  (score {s['score']})")
    print(f"Latency: {r.latency_ms}")
    sys.exit()

if args.calibrate:
    cal = [(q.query, not q.expect_refusal) for q in queries]
    val = [(v["query"], v["in_scope"]) for v in validation]
    score = lambda qs: np.array([dense.scores(q).max() for q, _ in qs])
    cs, vs = score(cal), score(val)
    cin, vin = np.array([x for _, x in cal]), np.array([x for _, x in val])
    print(f"\n=== Guardrail calibration ({'DRY RUN - not real' if dry else 'real embeddings'}) ===")
    print(f"Calibration set: {cin.sum()} in-scope + {(~cin).sum()} out-of-scope   | Validation set: {vin.sum()} + {(~vin).sum()} (written separately, used only to check)")
    print(f"Configured threshold: {thr}\n")
    print("threshold | calibration: in-scope kept / out-of-scope refused | validation: in-scope kept / out-of-scope refused")
    sweep = []
    for t in sorted({round(x, 2) for x in np.arange(0.15, 0.37, 0.02)} | {thr}):
        row = lambda sc, inn: (f"{(sc[inn] >= t).mean():5.0%} / {(sc[~inn] < t).mean():5.0%}")
        mark = "  <- configured" if abs(t - thr) < 1e-9 else ""
        print(f"  {t:.2f}    |        {row(cs, cin)}                  |        {row(vs, vin)}{mark}")
        sweep.append({"threshold": t, "cal_in_kept": (cs[cin] >= t).mean(), "cal_oos_refused": (cs[~cin] < t).mean(),
                      "val_in_kept": (vs[vin] >= t).mean(), "val_oos_refused": (vs[~vin] < t).mean()})
    resdir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    resdir.mkdir(exist_ok=True)
    pre = "stage6_dry_" if dry else "stage6_"
    pd.DataFrame(sweep).to_csv(resdir / f"{pre}calibration.csv", index=False)
    pd.DataFrame({"in_scope": vin, "cosine": vs, "query": [q for q, _ in val]}).to_csv(resdir / f"{pre}validation_scores.csv", index=False)
    print(f"\nSaved {resdir / (pre + 'calibration.csv')} and {pre}validation_scores.csv (used by build_report.py)")
    print("\nValidation queries and their best cosine (sorted)")
    out = pd.DataFrame({"in_scope": vin, "cosine": vs.round(3), "query": [q for q, _ in val]}).sort_values("cosine")
    print(out.to_string(index=False))
    print("\nIn-scope queries with the lowest cosine on the calibration set (closest to being wrongly refused)")
    low = pd.DataFrame({"cosine": cs[cin].round(3), "query": [q for (q, x) in cal if x]}).nsmallest(4, "cosine")
    print(low.to_string(index=False))
    sys.exit()

# ---- end-to-end evaluation ----
items = [(q.id, q.type, q.query, not q.expect_refusal, set(q.relevant_ids)) for q in queries]
items += [(v["id"], "validation", v["query"], v["in_scope"], set()) for v in validation]
if args.qids:
    wanted = {x.strip() for x in args.qids.split(",")}
    items = [it for it in items if it[0] in wanted]
rows = []
for qid, typ, text, in_scope, rel in items:
    r = assistant.answer(text)
    called_llm = r.model != ""
    rows.append({"qid": qid, "type": typ, "in_scope": in_scope, "query": text, "status": r.status,
                 "cosine": r.top_dense_cosine, "cited": ",".join(r.cited_ids), "model": r.model, "latency_ms": r.latency_ms.get("total"),
                 "cites_relevant": bool(rel & set(r.cited_ids)), "answer": r.answer, "raw_answer": r.raw_answer})
    if called_llm and not args.no_llm:
        time.sleep(args.sleep)
df = pd.DataFrame(rows)
refused = df.status.isin(["refused_out_of_scope", "no_answer", "refused_injection"])
print(f"\n=== Stage 6 end-to-end ({'DRY RUN - not real' if dry else 'real models'}; LLM: {'extractive fallback' if args.no_llm else cfg['generation']['model']}) ===")
print("\nStatus counts")
print(pd.crosstab(df.in_scope.map({True: "in-scope", False: "out-of-scope"}), df.status).to_string())
ans = df[df.in_scope & df.type.isin(["keyword", "paraphrase", "mixed"])].copy()
def outcome(x):
    if x.status in ("refused_out_of_scope", "no_answer", "refused_injection"):
        return "declined"
    if x.status == "uncited":
        return "uncited"
    return "ok" if x.cites_relevant else "wrong_citation"
ans["outcome"] = ans.apply(outcome, axis=1)
print(f"\nIn-scope labelled queries ({len(ans)}): " + ", ".join(f"{k} {v}" for k, v in ans.outcome.value_counts().items()))
print("  ok = answered and cited a relevant article | uncited = answered but no valid citation | declined = refused / NOT_IN_KB")
oos = df[~df.in_scope]
if len(oos):
    print(f"Out-of-scope queries ({len(oos)}): correctly refused {refused[oos.index].mean():.0%} "
          f"(by the gate: {(oos.status == 'refused_out_of_scope').sum()}, by the LLM saying NOT_IN_KB: {(oos.status == 'no_answer').sum()})")
leak = oos[~refused[oos.index]]
print("\nOut-of-scope queries that got an answer (guardrail leaks)")
print(leak[["qid", "query", "cosine", "status"]].to_string(index=False) if len(leak) else "  (none)")
print("\nIn-scope queries that did NOT end as 'ok'")
bad = ans[ans.outcome != "ok"]
print(bad[["qid", "query", "cosine", "status", "cited"]].to_string(index=False) if len(bad) else "  (none)")
if (bad.outcome == "uncited").any():
    print("\nRaw model text for uncited answers (to see what format the model used)")
    for _, x in bad[bad.outcome == "uncited"].iterrows():
        print(f"  {x['qid']}: {x['raw_answer'][:230]!r}")
res = PROJECT_ROOT / cfg["paths"]["results_dir"]
res.mkdir(exist_ok=True)
out = res / ("stage6_dry_run.csv" if dry else "stage6_end_to_end.csv")
df.to_csv(out, index=False)
print(f"\nSaved {out}")
print("\nThree sample answers")
for _, x in df[df.in_scope & ~refused].head(3).iterrows():
    print(f"  Q: {x['query']}\n  A: {x['answer']}\n")
