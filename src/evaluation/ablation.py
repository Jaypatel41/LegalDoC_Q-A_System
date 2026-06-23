"""Ablation runner. Executes the pipeline at progressive accuracy stages and
records the RAGAS (or proxy) metric at each, producing the delta-per-layer table
that is the heart of the resume story.

Stages:
  baseline       : dense-only retrieval, single query, no guardrails
  +multiquery    : add 4-variant MultiQuery expansion
  +hybrid_rerank : add BM25 hybrid + RRF fusion + cross-encoder rerank
  +guardrails    : add the 3-gate guardrail stack (incl. faithfulness regen)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

from ..config import CFG
from ..pipeline import AblationFlags, LegalRAG
from . import ragas_eval

STAGES = {
    "baseline":       AblationFlags(multiquery=False, hybrid=False, rerank=False, guardrails=False),
    "+multiquery":    AblationFlags(multiquery=True,  hybrid=False, rerank=False, guardrails=False),
    "+hybrid_rerank": AblationFlags(multiquery=True,  hybrid=True,  rerank=True,  guardrails=False),
    "+guardrails":    AblationFlags(multiquery=True,  hybrid=True,  rerank=True,  guardrails=True),
}


def load_testset() -> List[Dict]:
    path = Path(CFG["paths"]["eval_set"])
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion (incorrect-answer rate)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_stage(name: str, flags: AblationFlags, testset: List[Dict]) -> Dict:
    rag = LegalRAG(flags)
    records, route_hits, halluc = [], 0, 0
    incorrect = 0
    for ex in testset:
        res = rag.answer(ex["question"])
        records.append({
            "question": ex["question"],
            "answer": res.answer,
            "contexts": res.contexts,
            "ground_truth": ex["ground_truth"],
        })
        if ex.get("route") and res.route == ex["route"]:
            route_hits += 1
        if res.hallucination_risk is not None and res.hallucination_risk > 0.5:
            halluc += 1
    metrics = ragas_eval.evaluate_records(records)
    n = len(testset)

    # "incorrect" proxy: faithfulness below target OR recall very low
    for r, ex in zip(records, testset):
        m = ragas_eval.proxy_metrics(r["question"], r["answer"], r["contexts"],
                                     r["ground_truth"])
        if m["faithfulness"] < 0.75 or m["context_recall"] < 0.3:
            incorrect += 1
    lo, hi = _wilson_ci(incorrect, n)

    return {
        "stage": name,
        "metrics": metrics,
        "route_accuracy": round(route_hits / n, 3),
        "hallucination_rate": round(halluc / n, 3),
        "incorrect_rate": round(incorrect / n, 3),
        "incorrect_ci95": [round(lo, 3), round(hi, 3)],
        "n": n,
    }


def run_ablation() -> List[Dict]:
    testset = load_testset()
    return [run_stage(name, flags, testset) for name, flags in STAGES.items()]
