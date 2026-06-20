"""Reciprocal Rank Fusion (RRF). Merges multiple ranked lists (dense, BM25, and
one list per MultiQuery variant) into a single robust ranking without needing to
calibrate heterogeneous score scales.

    RRF_score(d) = sum_over_lists  1 / (k + rank_of_d_in_list)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..config import CFG


def rrf(ranked_lists: List[List[str]], k: int | None = None) -> List[Tuple[str, float]]:
    """ranked_lists: list of lists of chunk-ids (best first). Returns fused ranking."""
    k = k or int(CFG.path("retrieval.fusion.rrf_k", 60))
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
