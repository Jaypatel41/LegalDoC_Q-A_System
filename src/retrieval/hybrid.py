"""Hybrid retrieval: BM25 (keyword) + dense (semantic), per routed store, fused
across all MultiQuery variants with Reciprocal Rank Fusion.

BM25 nails exact legal citations (IPC 302, Section 73); dense retrieval captures
semantic intent. Combining them beats either alone — measurable as MRR@5 / recall
delta in the ablation."""
from __future__ import annotations

from typing import Dict, List

from ..config import CFG
from ..embeddings import cosine, get_embedder
from ..ingestion.indexer import get_index
from ..schema import Chunk, Retrieved
from .fusion import rrf


def _search_store(store: str, queries: List[str]) -> Dict[str, Retrieved]:
    """Run hybrid search for every query variant in one store, fuse with RRF."""
    idx = get_index(store)
    if not idx.chunks:
        return {}
    emb = get_embedder()
    rcfg = CFG["retrieval"]
    ranked_lists: List[List[str]] = []
    pool: Dict[str, Chunk] = {}
    raw_scores: Dict[str, Dict[str, float]] = {}

    qvecs = emb.encode(queries)
    for qi, q in enumerate(queries):
        # dense
        if rcfg["hybrid"]["enabled"] or True:
            dense = idx.dense_search(qvecs[qi], rcfg["top_k_dense"])
            dlist = []
            for cid_i, score in dense:
                c = idx.chunks[cid_i]
                pool[c.id] = c
                dlist.append(c.id)
                raw_scores.setdefault(c.id, {})["dense"] = max(
                    raw_scores.get(c.id, {}).get("dense", 0.0), score)
            ranked_lists.append(dlist)
        # bm25
        if rcfg["hybrid"]["enabled"]:
            bm = idx.bm25_search(q, rcfg["top_k_bm25"])
            blist = []
            for cid_i, score in bm:
                c = idx.chunks[cid_i]
                pool[c.id] = c
                blist.append(c.id)
                raw_scores.setdefault(c.id, {})["bm25"] = max(
                    raw_scores.get(c.id, {}).get("bm25", 0.0), score)
            ranked_lists.append(blist)

    fused = rrf(ranked_lists)[: rcfg["top_k_merged"]]
    out: Dict[str, Retrieved] = {}
    for cid, fscore in fused:
        c = pool[cid]
        rs = raw_scores.get(cid, {})
        out[cid] = Retrieved(chunk=c, score=fscore,
                             bm25_score=rs.get("bm25", 0.0),
                             dense_score=rs.get("dense", 0.0))
    return out


def retrieve(stores: List[str], queries: List[str]) -> List[Retrieved]:
    """Search each routed store, merge results across stores by fused score."""
    merged: Dict[str, Retrieved] = {}
    for store in stores:
        for cid, r in _search_store(store, queries).items():
            if cid not in merged or r.score > merged[cid].score:
                merged[cid] = r
    results = sorted(merged.values(), key=lambda r: -r.score)
    return results[: CFG["retrieval"]["top_k_merged"]]
