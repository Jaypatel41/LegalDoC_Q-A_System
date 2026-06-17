"""Embedding wrapper. Uses sentence-transformers when available, else a
deterministic hashing-bag-of-words fallback so the pipeline runs with zero
downloads. Swap embeddings.model vs embeddings.legal_model in config to measure
the retrieval-precision delta (a number for your resume)."""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from .config import CFG


class Embedder:
    def __init__(self) -> None:
        ecfg = CFG["embeddings"]
        self.model_name = ecfg["legal_model"] if ecfg["use_legal_model"] else ecfg["model"]
        self.device = ecfg.get("device", "cpu")
        self._model = None
        self.dim = 384  # fallback dim
        self._load()
        # True when running on the offline hashing fallback (cosine values are
        # not calibrated to the real-model thresholds in config).
        self.is_fallback = self._model is None

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self._model.get_sentence_embedding_dimension()
        except Exception as e:  # pragma: no cover
            print(f"[embeddings] using hashing fallback ({e})")
            self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")
        return np.vstack([self._hash_embed(t) for t in texts]).astype("float32")

    # --- deterministic offline fallback (hashed bag of tokens) ---
    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec


_EMB: Embedder | None = None


def get_embedder() -> Embedder:
    global _EMB
    if _EMB is None:
        _EMB = Embedder()
    return _EMB


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
