# Pipeline Reference

Core philosophy: **every accuracy technique has a measurable delta.** Run
`scripts/run_eval` with each layer toggled on/off and report the improvement as a
number. That ablation table is what separates this from a RAG tutorial.

---

## Phase 1 — Ingestion & indexing  (`src/ingestion/`)

| Store    | Chunk strategy                  | Why |
|----------|---------------------------------|-----|
| statute  | split by `Section N`            | preserves legal structure |
| case_law | paragraph + 200-token overlap   | courts write in argument blocks |
| contract | split by clause heading         | clause-level retrieval beats fixed-size |

Each store is embedded (`all-MiniLM-L6-v2`, swappable for `legal-bert`) and stored
as: dense vectors (FAISS) + BM25 keyword index + chunk metadata. Swapping the
embedding model and re-running retrieval eval gives the **embedding-precision delta**.

## Phase 2 — MultiQuery expansion  (`src/retrieval/multiquery.py`)

The LLM rewrites each query into 4 framings (literal / procedural / remedy /
precedent). Legal synonyms sit far apart in embedding space, so variants retrieve
complementary chunks. Typical lift: **+12–15% context recall**.

## Phase 3 — Hybrid retrieval + routing  (`src/retrieval/`)

- **Router** (`router.py`) — the trained intent classifier picks the store.
- **Hybrid** (`hybrid.py`) — BM25 (nails `IPC 302`, `Section 73`) + dense
  (semantic intent), every variant fused with **Reciprocal Rank Fusion** (`fusion.py`).
- **Rerank** (`rerank.py`) — cross-encoder `bge-reranker` re-scores the top-20 →
  top-5. Typical lift: **+8–11% MRR@5**.

## Phase 4 — Guardrail stack  (`src/guardrails/`)

1. **Input filter** — reject out-of-scope queries (intent classifier).
2. **Context sufficiency** — refuse if `< 2` chunks or `max_cosine < 0.45` (or the
   learned relevance scorer flags it) → returns *"insufficient information"* instead
   of hallucinating.
3. **Output faithfulness** — score the answer with the hallucination classifier;
   if `faithfulness < 0.75`, **auto-regenerate** with a stricter grounded prompt.
   This retry loop is the key differentiator.

## Phase 5 — Evaluation  (`src/evaluation/`)

Four RAGAS metrics on the test set, computed at each ablation stage:

| Metric            | Target | Measures |
|-------------------|--------|----------|
| Faithfulness      | > 0.88 | claims grounded in retrieved docs |
| Context recall    | > 0.84 | retrieval found the right chunks |
| Answer relevancy  | > 0.82 | answer on-topic to the question |
| Context precision | > 0.79 | retrieved chunks actually relevant |

`scripts/run_eval` prints the per-stage table + deltas + a Wilson 95% CI on the
incorrect-answer rate.

> **Offline vs real numbers.** With no deps/keys the project runs on a deterministic
> stub (hashing embeddings + echo LLM) purely as a smoke test — those numbers are
> NOT representative. For real, resume-grade numbers:
> `pip install -r requirements.txt`, set an LLM key in `.env`, unset `USE_STUB_LLM`,
> then `python -m scripts.run_eval`.

---

## The two ML models  (`src/ml/`)

### 1. Query Intent Classifier — supervised text classification
TF-IDF + Logistic Regression (or DistilBERT). Routes `statute / case_law /
contract / out_of_scope`. Train + eval: `python -m scripts.train_intent`
→ precision/recall/F1 per class.

### 2. Hallucination Risk Classifier — weak supervision
Trained on RAGAS faithfulness labels (`faithful ≥ τ → 1`). Distils an expensive
LLM-judge into a fast local logistic-regression scorer over cheap features
(cosine support, token/numeric overlap, hedge ratio). Powers Gate 3 at **zero API
cost in the hot path**. Train + eval: `python -m scripts.train_hallucination`
→ ROC-AUC + F1.

---

## Stack

| Layer        | Tool |
|--------------|------|
| Orchestration| custom pipeline (LangChain-compatible) |
| Embeddings   | sentence-transformers (`all-MiniLM` / `legal-bert`) |
| Vector store | FAISS (+ numpy fallback) |
| Keyword      | `rank_bm25` |
| Reranker     | `BAAI/bge-reranker-base` (CrossEncoder) |
| ML models    | scikit-learn (intent + hallucination) |
| LLM          | Anthropic Claude / OpenAI / local — provider-agnostic |
| Evaluation   | RAGAS (+ offline proxy) |
| UI           | Streamlit |
