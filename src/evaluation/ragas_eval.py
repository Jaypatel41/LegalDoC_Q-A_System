"""Evaluation metrics.

Two modes:
  * REAL RAGAS  — when `ragas` is installed AND a real LLM is configured
    (USE_STUB_LLM != 1). Computes faithfulness / context_recall /
    answer_relevancy / context_precision with the genuine RAGAS judges.
  * PROXY       — offline estimators (no API) so the ablation table is
    reproducible anywhere. Clearly labelled as proxy in the report.

Proxy definitions (token/embedding based):
  context_recall   : fraction of ground-truth content tokens covered by the
                     union of retrieved contexts.
  context_precision: fraction of retrieved contexts that overlap the ground truth.
  answer_relevancy : cosine(answer, question).
  faithfulness     : 1 - hallucination_risk(answer, contexts)  (our ML model).
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from ..config import env
from ..embeddings import cosine, get_embedder
from ..ml.hallucination_classifier import get_hallucination_classifier

_STOP = set("a an the of to in on for and or is are was were be by with as at "
            "that this which shall any such not no it its from under into".split())


def _content_tokens(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if t not in _STOP and len(t) > 2}


def _recall(ground_truth: str, contexts: List[str]) -> float:
    gt = _content_tokens(ground_truth)
    if not gt:
        return 0.0
    ctx = set()
    for c in contexts:
        ctx |= _content_tokens(c)
    return len(gt & ctx) / len(gt)


def _precision(ground_truth: str, contexts: List[str]) -> float:
    gt = _content_tokens(ground_truth)
    if not contexts or not gt:
        return 0.0
    hits = 0
    for c in contexts:
        ct = _content_tokens(c)
        # a context is "relevant" if it shares a meaningful fraction with GT
        if ct and len(gt & ct) / len(gt) >= 0.10:
            hits += 1
    return hits / len(contexts)


def proxy_metrics(question: str, answer: str, contexts: List[str],
                  ground_truth: str) -> Dict[str, float]:
    emb = get_embedder()
    qv, av = emb.encode([question])[0], emb.encode([answer])[0]
    hc = get_hallucination_classifier()
    return {
        "faithfulness": round(1.0 - hc.risk(answer, contexts), 3),
        "context_recall": round(_recall(ground_truth, contexts), 3),
        "answer_relevancy": round(max(0.0, cosine(qv, av)), 3),
        "context_precision": round(_precision(ground_truth, contexts), 3),
    }


def _use_real_ragas() -> bool:
    if env("USE_STUB_LLM", "0") == "1":
        return False
    try:
        import ragas  # noqa: F401
        return True
    except Exception:
        return False


def evaluate_records(records: List[Dict]) -> Dict[str, float]:
    """records: [{question, answer, contexts, ground_truth}, ...]"""
    if _use_real_ragas():
        try:
            return _real_ragas(records)
        except Exception as e:  # pragma: no cover
            print(f"[eval] RAGAS failed ({e}); using proxy metrics.")
    rows = [proxy_metrics(r["question"], r["answer"], r["contexts"],
                          r["ground_truth"]) for r in records]
    keys = ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]
    return {k: round(float(np.mean([row[k] for row in rows])), 3) for k in keys}


def _real_ragas(records: List[Dict]) -> Dict[str, float]:  # pragma: no cover
    """RAGAS 0.4.x. Uses the configured LLM (Groq/OpenAI/local, via LangChain's
    OpenAI-compatible ChatOpenAI) as the judge and a local sentence-transformer
    for the embedding-based metric. Throttled via RunConfig to respect free-tier
    rate limits."""
    import os
    import numpy as np
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    from ragas.run_config import RunConfig

    base_url = (os.environ.get("LOCAL_BASE_URL")
                or "https://api.openai.com/v1")
    api_key = (os.environ.get("LOCAL_API_KEY")
               or os.environ.get("OPENAI_API_KEY") or "")
    model = os.environ.get("LOCAL_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # bypass_n=True -> RAGAS loops single completions instead of requesting n>1,
    # which Groq (and most non-OpenAI OpenAI-compatible APIs) rejects.
    judge = LangchainLLMWrapper(ChatOpenAI(
        model=model, base_url=base_url, api_key=api_key,
        temperature=0.0, max_retries=6, timeout=120), bypass_n=True)
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}))

    ds = EvaluationDataset.from_list([{
        "user_input": r["question"],
        "response": r["answer"],
        "retrieved_contexts": r["contexts"] or [""],
        "reference": r["ground_truth"],
    } for r in records])

    metrics = [Faithfulness(), LLMContextRecall(),
               ResponseRelevancy(), LLMContextPrecisionWithReference()]
    # low worker count + backoff keeps us under Groq's ~30 req/min free tier
    run_config = RunConfig(max_workers=2, timeout=180, max_retries=10, max_wait=90)
    result = evaluate(dataset=ds, metrics=metrics, llm=judge, embeddings=emb,
                      run_config=run_config)
    df = result.to_pandas()

    def avg(metric) -> float:
        col = metric.name
        return round(float(np.nanmean(df[col])), 3) if col in df.columns else float("nan")

    return {
        "faithfulness": avg(metrics[0]),
        "context_recall": avg(metrics[1]),
        "answer_relevancy": avg(metrics[2]),
        "context_precision": avg(metrics[3]),
    }


def metrics_mode() -> str:
    return "RAGAS" if _use_real_ragas() else "PROXY (offline)"
