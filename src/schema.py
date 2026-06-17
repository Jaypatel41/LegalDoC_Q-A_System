"""Shared data structures used across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    id: str
    text: str
    store: str                       # statute | case_law | contract
    source: str                      # filename
    title: str = ""                  # e.g. "Section 302" or clause heading
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class PipelineResult:
    query: str
    answer: str
    contexts: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    route: str = ""
    sub_queries: List[str] = field(default_factory=list)
    faithfulness: Optional[float] = None
    hallucination_risk: Optional[float] = None
    confidence: float = 0.0
    blocked: bool = False
    block_reason: str = ""
    regenerated: bool = False
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
