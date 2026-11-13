"""Stage 1 smoke check: load config, corpus and test queries, print a summary."""
import sys
from collections import Counter
sys.path.insert(0, "src")
from helpdesk.config import load_config, resolve_path
from helpdesk.logging_setup import setup_logging
from helpdesk.ingest import load_corpus, load_test_queries

cfg = load_config()
setup_logging(cfg)
docs = load_corpus(resolve_path(cfg, "corpus"))
queries = load_test_queries(resolve_path(cfg, "test_queries"), [d.id for d in docs])

print("\nArticles per category:")
for cat, n in Counter(d.category for d in docs).items():
    print(f"  {cat:<18} {n}")
print("\nTest queries per type:")
for t, n in Counter(q.type for q in queries).items():
    print(f"  {t:<14} {n}")
print("\nStage 1 OK")
