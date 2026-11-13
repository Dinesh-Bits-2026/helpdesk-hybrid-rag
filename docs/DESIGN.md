# Design notes

## Goal
A helpdesk assistant that handles both exact-term lookups ("VPN error 809") and questions that share no words with the answer ("I cannot remember my login credentials"), cites its sources, and refuses what it cannot answer.

## Pipeline
1. **Input guardrails**: mask personal data; block obvious prompt-injection phrases.
2. **Relevance gate**: best cosine similarity below a threshold means refuse, with no LLM call.
3. **Retrieval**: BM25 (own implementation) and sentence-embedding search each return their top 20.
4. **Fusion**: Reciprocal Rank Fusion, `score = sum of 1 / (k + rank)`; ranks, not scores, because BM25 and cosine scores are on different scales.
5. **Re-ranking**: a cross-encoder re-scores the 20 candidates; the top 3 go to the LLM.
6. **Generation**: the LLM answers only from those articles, cites ids, or replies `NOT_IN_KB`.
7. **Output checks**: citations must be retrieved ids; invented ones are removed.
8. **Logging**: one JSON line per request with the masked query, status, sources and latency.

## Key decisions and why

| Decision | Reason |
|---|---|
| Everything tunable in one YAML file | Configurable and reproducible; the domain can be swapped without code changes |
| BM25 written by hand (numpy) | Every term of the formula is visible and testable against a hand calculation; one fewer dependency |
| Light built-in stemmer; digits never stemmed | Plural and -ing matching without a heavy NLP dependency; error codes such as `0x80070005` must stay exact |
| RRF over weighted score sum | No score normalisation needed |
| Re-ranker over a wide pool (20, not 10) | RRF can bury an article only one retriever found; a wide pool lets the re-ranker rescue it |
| Default pipeline hybrid + re-rank | Ties dense + re-rank on every metric; chosen so keyword search remains a safety net for exact terms. Switchable in config |
| Gate on dense cosine, not on the re-ranker score | The cross-encoder's top score barely separates in-scope from out-of-scope questions and its scale varies by library version |
| Layered guardrails | A similarity threshold cannot catch questions that only sound relevant; the grounded prompt handles those |
| LLM failure falls back to an extractive answer marked `degraded` | Graceful degradation instead of a crash |
| Request log stores only the masked query | Observability without leaking personal data |

## Evaluation method
- Labelled queries by type (keyword, paraphrase, mixed, out of scope), each with its correct article id(s).
- Metrics: hit@1, recall@3, MRR on answerable queries; out-of-scope queries are judged by the guardrails.
- A *core* set and an *extended* set are reported separately; the extended set was added after the first vector-search results because the core queries could not tell the pipelines apart.
- A separate 20-query validation file checks the gate threshold, which was chosen on the test queries.

## Known failure modes
- Plain RRF can rank a wrong article that both retrievers like above the right article that only one ranks first.
- Near-domain out-of-scope questions pass the similarity gate; only the grounded prompt stops them.
- LLM output can omit or reformat citations (for example full-width brackets); the parser normalises these but cannot recover a missing citation.

## Extension points
- New domain: replace `data/corpus.json`, the queries, and the `domain` block in the config.
- New retriever: implement `search(query, k)` returning ranked `Hit` objects and add it to the fusion list.
- New generator: implement `generate(question, sources)` and pass it to `HelpdeskAssistant`.
