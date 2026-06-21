"""Intent-based data router. Uses the trained intent classifier to decide which
store(s) to search. 'out_of_scope' short-circuits the pipeline (input guardrail).

Returns the routed store plus the full probability distribution (used for the
confidence score reported to the user)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..config import CFG
from ..ml.intent_classifier import get_intent_classifier, heuristic_predict


@dataclass
class Route:
    store: str
    confidence: float
    proba: Dict[str, float]
    in_scope: bool

    @property
    def stores(self) -> List[str]:
        # primary store; statute & case_law are cross-searched for legal questions
        if self.store in ("statute", "case_law"):
            return [self.store, "case_law" if self.store == "statute" else "statute"]
        return [self.store]


def route(query: str) -> Route:
    use_ml = CFG.path("guardrails.input_filter.use_ml_classifier", True)
    if use_ml:
        clf = get_intent_classifier()
        proba = clf.predict_proba(query)
        store = max(proba, key=proba.get)
        conf = proba[store]
    else:
        store = heuristic_predict(query)
        proba = {store: 1.0}
        conf = 1.0
    return Route(store=store, confidence=conf, proba=proba,
                 in_scope=store != "out_of_scope")
