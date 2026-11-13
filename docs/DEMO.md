# 2-minute demo path

Start the app with `streamlit run app.py` and click the sidebar examples in this order.

| # | Example | What to point out |
|---|---|---|
| 1 | **Error code (keyword)** | Exact term "809" found by keyword search; green "Keyword + semantic" card; coloured citation pill matches the card |
| 2 | **Different words, same meaning** | No shared words with the article, so the card is blue ("Semantic only"): vector search rescues what BM25 misses |
| 3 | **Mixed question** | Both signals agree; re-ranker orders the candidates |
| 4 | **Personal data (PII)** | "Personal data masked" chip; the request log stores `[EMAIL]`, never the address |
| 5 | **Out of scope** | Refused by the relevance gate in milliseconds; the LLM was never called |
| 6 | **Near-domain trap** | Sounds like helpdesk but is not in the knowledge base; passes the gate, so the grounded prompt answers "not in the knowledge base" |
| 7 | **Prompt injection** | Blocked before any retrieval |

Then open "Advanced settings" and switch the search method to **Keywords only** and ask example 2 again to show the difference.
