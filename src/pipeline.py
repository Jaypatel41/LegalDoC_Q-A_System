"""End-to-end Legal QA RAG orchestrator.

    route -> [Gate1 input] -> multiquery -> hybrid retrieve -> rerank
          -> [Gate2 sufficiency] -> generate -> [Gate3 faithfulness -> regen]

Stage toggles (AblationFlags) let scripts/run_eval.py turn each accuracy layer
on/off and measure the RAGAS delta per layer — the project's core selling point.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .config import CFG
from .generation.generator import citations, generate
from .guardrails import guardrails as G
from .retrieval import hybrid, multiquery, rerank as rr
from .retrieval.router import route as route_query
from .schema import PipelineResult


@dataclass
class AblationFlags:
    multiquery: bool = True
    hybrid: bool = True      # if False -> dense-only
    rerank: bool = True
    guardrails: bool = True  # input + sufficiency + faithfulness gates

    @classmethod
    def baseline(cls) -> "AblationFlags":
        return cls(multiquery=False, hybrid=False, rerank=False, guardrails=False)


def _now() -> float:
    return time.perf_counter()


class LegalRAG:
    def __init__(self, flags: AblationFlags | None = None):
        self.flags = flags or AblationFlags()

    def answer(self, query: str) -> PipelineResult:
        f = self.flags
        res = PipelineResult(query=query, answer="")
        t = _now()

        # ---- route -------------------------------------------------------
        route = route_query(query)
        res.route = route.store
        res.confidence = route.confidence
        res.timings_ms["route"] = (_now() - t) * 1000

        # ---- Gate 1: input filter ---------------------------------------
        if f.guardrails:
            g1 = G.input_filter(route)
            if not g1.passed:
                res.blocked = True
                res.block_reason = g1.reason
                res.answer = ("This question appears to be outside the legal "
                              "knowledge base, so it was not answered.")
                return res
            stores = route.stores
        else:
            # no routing guard: search everything
            stores = [s["name"] for s in CFG["stores"]]

        # ---- MultiQuery --------------------------------------------------
        t = _now()
        queries = multiquery.expand(query) if f.multiquery else [query]
        res.sub_queries = queries
        res.timings_ms["multiquery"] = (_now() - t) * 1000

        # ---- retrieval ---------------------------------------------------
        t = _now()
        if f.hybrid:
            results = hybrid.retrieve(stores, queries)
        else:
            results = self._dense_only(stores, query)
        res.timings_ms["retrieve"] = (_now() - t) * 1000

        # ---- rerank ------------------------------------------------------
        t = _now()
        if f.rerank and results:
            results = rr.rerank(query, results)
        else:
            results = results[: CFG["retrieval"]["rerank"]["top_k_final"]]
        res.timings_ms["rerank"] = (_now() - t) * 1000

        res.contexts = [r.chunk.text for r in results]
        res.citations = citations(results)

        # ---- Gate 2: context sufficiency --------------------------------
        if f.guardrails:
            g2 = G.context_sufficiency(query, results)
            if not g2.passed:
                res.blocked = True
                res.block_reason = g2.reason
                res.answer = CFG["generation"]["insufficient_msg"]
                res.confidence = min(res.confidence, g2.score or 0.0)
                return res

        # ---- generation --------------------------------------------------
        t = _now()
        answer = generate(query, results, strict=False)
        res.timings_ms["generate"] = (_now() - t) * 1000

        # ---- Gate 3: output faithfulness + regeneration -----------------
        if f.guardrails:
            g3 = G.output_faithfulness(answer, res.contexts)
            res.faithfulness = g3.score
            res.hallucination_risk = 1.0 - (g3.score or 0.0)
            max_regen = int(CFG.path("guardrails.output_faithfulness.max_regenerations", 1))
            if not g3.passed and max_regen > 0:
                t = _now()
                answer = generate(query, results, strict=True)
                res.regenerated = True
                g3b = G.output_faithfulness(answer, res.contexts)
                res.faithfulness = g3b.score
                res.hallucination_risk = 1.0 - (g3b.score or 0.0)
                res.timings_ms["regenerate"] = (_now() - t) * 1000

        res.answer = answer
        # blended confidence: routing certainty x groundedness
        ground = res.faithfulness if res.faithfulness is not None else 0.7
        res.confidence = round(float(route.confidence) * float(ground), 3)
        return res

    # dense-only path for the ablation baseline
    def _dense_only(self, stores, query):
        from .embeddings import get_embedder
        from .ingestion.indexer import get_index
        from .schema import Retrieved
        emb = get_embedder()
        qv = emb.encode([query])[0]
        pool = []
        for store in stores:
            idx = get_index(store)
            for i, s in idx.dense_search(qv, CFG["retrieval"]["top_k_dense"]):
                pool.append(Retrieved(chunk=idx.chunks[i], score=s, dense_score=s))
        pool.sort(key=lambda r: -r.score)
        return pool[: CFG["retrieval"]["top_k_merged"]]


def answer(query: str, flags: AblationFlags | None = None) -> PipelineResult:
    return LegalRAG(flags).answer(query)
