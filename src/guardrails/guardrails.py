"""Three-gate guardrail stack — the layer that makes this production-grade.

  Gate 1  Input filter        : reject out-of-scope / non-legal queries
                                 (uses the trained intent classifier).
  Gate 2  Context sufficiency : if retrieval is too weak (few chunks or low
                                 similarity / low learned relevance) refuse to
                                 answer instead of hallucinating.
  Gate 3  Output faithfulness : after generation, score groundedness with the
                                 hallucination classifier; if risky, signal the
                                 pipeline to regenerate with a stricter prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..config import CFG
from ..embeddings import cosine, get_embedder
from ..ml.hallucination_classifier import get_hallucination_classifier
from ..schema import Retrieved
from ..retrieval.router import Route


@dataclass
class GateResult:
    passed: bool
    reason: str = ""
    score: Optional[float] = None


# ---------------------------------------------------------------- Gate 1
def input_filter(route: Route) -> GateResult:
    if not CFG.path("guardrails.input_filter.enabled", True):
        return GateResult(True)
    if not route.in_scope:
        return GateResult(False, "Query is outside the legal knowledge base "
                                 "(classified out-of-scope).", route.confidence)
    return GateResult(True, score=route.confidence)


# ---------------------------------------------------------------- Gate 2
def context_sufficiency(query: str, results: List[Retrieved]) -> GateResult:
    cfg = CFG["guardrails"]["context_sufficiency"]
    if not cfg["enabled"]:
        return GateResult(True)
    if len(results) < int(cfg["min_chunks"]):
        return GateResult(False, "Too few relevant chunks retrieved.",
                          float(len(results)))

    emb = get_embedder()
    if emb.is_fallback:
        # offline hashing embeddings: cosine isn't calibrated to min_cosine, so
        # enforce only the chunk-count gate to keep the demo usable.
        return GateResult(True, score=1.0)
    qv = emb.encode([query])[0]
    cvs = emb.encode([r.chunk.text for r in results])
    max_cos = max(cosine(qv, cv) for cv in cvs)

    if cfg.get("use_learned_scorer", True):
        # learned relevance: low hallucination-risk of the top context => relevant
        hc = get_hallucination_classifier()
        risk = hc.risk(query, [results[0].chunk.text])
        learned_ok = risk < 0.85
        if max_cos < float(cfg["min_cosine"]) and not learned_ok:
            return GateResult(False, "Retrieved context insufficiently related "
                                     "to the query.", max_cos)
        return GateResult(True, score=max_cos)

    if max_cos < float(cfg["min_cosine"]):
        return GateResult(False, "Retrieved context insufficiently related to "
                                 "the query.", max_cos)
    return GateResult(True, score=max_cos)


# ---------------------------------------------------------------- Gate 3
def output_faithfulness(answer: str, contexts: List[str]) -> GateResult:
    cfg = CFG["guardrails"]["output_faithfulness"]
    if not cfg["enabled"]:
        return GateResult(True)
    hc = get_hallucination_classifier()
    risk = hc.risk(answer, contexts)               # P(hallucinated)
    faithfulness = 1.0 - risk
    if faithfulness < float(cfg["min_faithfulness"]):
        return GateResult(False, "Generated answer not sufficiently grounded; "
                                 "regenerating with stricter prompt.", faithfulness)
    return GateResult(True, score=faithfulness)
