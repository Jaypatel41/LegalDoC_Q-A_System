"""Cross-encoder reranking. The fused top-20 is re-scored by a cross-encoder
(BAAI/bge-reranker), which reads (query, chunk) jointly rather than comparing
independent embeddings — much higher precision at the top. Returns top-k_final.

This stage alone typically adds +8-11% MRR@5. When the cross-encoder isn't
installed it falls back to a cosine re-score so the pipeline still runs."""
from __future__ import annotations

from typing import List

from ..config import CFG
from ..embeddings import cosine, get_embedder
from ..schema import Retrieved

_RERANKER = None
_LOAD_FAILED = False


def _get_reranker():
    global _RERANKER, _LOAD_FAILED
    if _RERANKER is not None or _LOAD_FAILED:
        return _RERANKER
    try:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(CFG["retrieval"]["rerank"]["model"])
    except Exception as e:  # pragma: no cover
        print(f"[rerank] cross-encoder unavailable, using cosine fallback ({e})")
        _LOAD_FAILED = True
    return _RERANKER


def rerank(query: str, results: List[Retrieved]) -> List[Retrieved]:
    rcfg = CFG["retrieval"]["rerank"]
    top_k = rcfg["top_k_final"]
    if not rcfg["enabled"] or not results:
        return results[:top_k]

    model = _get_reranker()
    if model is not None:
        pairs = [(query, r.chunk.text) for r in results]
        scores = model.predict(pairs)
        for r, s in zip(results, scores):
            r.rerank_score = float(s)
    else:
        emb = get_embedder()
        qv = emb.encode([query])[0]
        cvs = emb.encode([r.chunk.text for r in results])
        for r, cv in zip(results, cvs):
            r.rerank_score = cosine(qv, cv)

    return sorted(results, key=lambda r: -r.rerank_score)[:top_k]
