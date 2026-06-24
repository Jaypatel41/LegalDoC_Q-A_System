"""Feature engineering for the hallucination risk classifier.

Given an (answer, retrieved_contexts) pair, we extract cheap, interpretable
signals that correlate with groundedness — no LLM call at inference time:

  - max / mean cosine(answer, context)        : semantic support
  - token overlap (answer ∩ context) / |ans|  : lexical grounding
  - n_contexts, answer_len                     : coverage / verbosity
  - numeric_overlap                            : do numbers in the answer appear
                                                 in the context? (cite-checking)
  - hedge_ratio                                : "may/insufficient/unclear" markers
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

from ..embeddings import cosine, get_embedder

_HEDGES = ("insufficient", "not sure", "unclear", "cannot", "no information",
           "unable", "i don't", "not found")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def extract(answer: str, contexts: List[str]) -> np.ndarray:
    emb = get_embedder()
    ans = answer or ""
    ctx_join = " \n ".join(contexts) if contexts else ""

    av = emb.encode([ans])[0]
    if contexts:
        cvs = emb.encode(contexts)
        cos = [cosine(av, cv) for cv in cvs]
        max_cos, mean_cos = max(cos), float(np.mean(cos))
    else:
        max_cos = mean_cos = 0.0

    ans_tokens = set(ans.lower().split())
    ctx_tokens = set(ctx_join.lower().split())
    overlap = len(ans_tokens & ctx_tokens) / (len(ans_tokens) + 1e-6)

    ans_nums = set(_NUM_RE.findall(ans))
    ctx_nums = set(_NUM_RE.findall(ctx_join))
    num_overlap = (len(ans_nums & ctx_nums) / (len(ans_nums) + 1e-6)
                   if ans_nums else 1.0)

    hedge = sum(ans.lower().count(h) for h in _HEDGES)
    hedge_ratio = hedge / (len(ans.split()) + 1e-6)

    return np.array([
        max_cos, mean_cos, overlap, num_overlap,
        len(contexts), len(ans.split()) / 100.0, hedge_ratio,
    ], dtype="float32")


FEATURE_NAMES = [
    "max_cosine", "mean_cosine", "token_overlap", "numeric_overlap",
    "n_contexts", "answer_len_100", "hedge_ratio",
]
