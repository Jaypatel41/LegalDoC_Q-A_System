"""Per-store index = dense vectors (FAISS or numpy) + BM25 keyword index + chunks.
Persisted under data/index/<store>/. Falls back gracefully when faiss/rank_bm25
are not installed so the project runs anywhere."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..config import CFG
from ..embeddings import get_embedder
from ..schema import Chunk
from .chunker import chunk_document
from .loader import load_store


class StoreIndex:
    def __init__(self, store: str):
        self.store = store
        self.dir = Path(CFG["paths"]["index_dir"]) / store
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self._faiss = None
        self._bm25 = None
        self._tokenized: List[List[str]] = []

    # ----------------------------------------------------------------- build
    def build(self) -> "StoreIndex":
        emb = get_embedder()
        docs = load_store(self.store)
        self.chunks = []
        for fname, text in docs:
            self.chunks.extend(chunk_document(self.store, fname, text))
        if not self.chunks:
            print(f"[index] no documents for store '{self.store}'")
            return self
        texts = [c.text for c in self.chunks]
        self.embeddings = emb.encode(texts)
        self._build_bm25(texts)
        self._build_faiss()
        self.save()
        print(f"[index] {self.store}: {len(self.chunks)} chunks indexed")
        return self

    def _build_bm25(self, texts: List[str]) -> None:
        self._tokenized = [t.lower().split() for t in texts]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized)
        except Exception:
            self._bm25 = None  # falls back to overlap scoring in hybrid

    def _build_faiss(self) -> None:
        try:
            import faiss
            dim = self.embeddings.shape[1]
            idx = faiss.IndexFlatIP(dim)  # cosine (vectors normalized)
            idx.add(self.embeddings)
            self._faiss = idx
        except Exception:
            self._faiss = None  # numpy fallback in dense_search

    # ------------------------------------------------------------------ search
    def dense_search(self, qvec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        k = min(k, len(self.chunks))
        if self._faiss is not None:
            scores, idxs = self._faiss.search(qvec.reshape(1, -1), k)
            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        sims = self.embeddings @ qvec
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        toks = query.lower().split()
        if self._bm25 is not None:
            scores = self._bm25.get_scores(toks)
        else:  # simple token-overlap fallback
            qset = set(toks)
            scores = np.array([len(qset & set(doc)) for doc in self._tokenized],
                              dtype="float32")
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    # ------------------------------------------------------------------- io
    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "embeddings.npy", self.embeddings)
        with open(self.dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False)
        with open(self.dir / "bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self._bm25, "tokenized": self._tokenized}, f)

    def load(self) -> "StoreIndex":
        if not (self.dir / "chunks.json").exists():
            raise FileNotFoundError(
                f"Index for '{self.store}' missing. Run: python -m scripts.build_index")
        with open(self.dir / "chunks.json", encoding="utf-8") as f:
            self.chunks = [Chunk(**d) for d in json.load(f)]
        self.embeddings = np.load(self.dir / "embeddings.npy")
        with open(self.dir / "bm25.pkl", "rb") as f:
            blob = pickle.load(f)
            self._bm25 = blob["bm25"]
            self._tokenized = blob["tokenized"]
        self._build_faiss()
        return self


def build_all() -> None:
    for s in CFG["stores"]:
        StoreIndex(s["name"]).build()


_CACHE: dict[str, StoreIndex] = {}


def get_index(store: str) -> StoreIndex:
    if store not in _CACHE:
        _CACHE[store] = StoreIndex(store).load()
    return _CACHE[store]
