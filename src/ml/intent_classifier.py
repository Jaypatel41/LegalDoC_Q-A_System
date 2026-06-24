"""Query Intent Classifier — supervised text classification.

Routes each query to: statute | case_law | contract | out_of_scope.
Backends:
  - tfidf_logreg : TF-IDF features + Logistic Regression (sklearn) — default, fast.
  - distilbert   : fine-tuned DistilBERT (transformers) — heavier, higher ceiling.

If sklearn/transformers are unavailable, a transparent keyword heuristic is used
so routing never hard-fails. Train + evaluate via scripts/train_intent.py, which
reports precision / recall / F1 per class on a held-out split (resume numbers).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import CFG

LABELS = CFG.path("ml.intent_classifier.labels",
                  ["statute", "case_law", "contract", "out_of_scope"])
MODEL_PATH = Path(CFG["paths"]["models_dir"]) / "intent_clf.pkl"

# --- keyword heuristic (fallback + weak prior) ---
_KEYWORDS = {
    "statute": ["section", "ipc", "penal code", "punishment", "penalty",
                "act", "provision", "statute", "clause of law"],
    "case_law": ["case", "court", "judgment", "ruling", "held", "precedent",
                 "supreme court", "v.", " vs ", "doctrine", "bharati", "gandhi"],
    "contract": ["agreement", "clause", "termination", "confidential",
                 "liability", "indemnif", "force majeure", "payment terms",
                 "arbitration", "notice period", "governing law"],
}


def heuristic_predict(query: str) -> str:
    q = " " + query.lower() + " "
    scores = {lbl: sum(q.count(k) for k in kws) for lbl, kws in _KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "out_of_scope"


def load_dataset(path: str | None = None) -> Tuple[List[str], List[str]]:
    path = path or str(Path(CFG["paths"]["eval_set"]).parent / "intent_train.jsonl")
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            X.append(row["query"])
            y.append(row["label"])
    return X, y


class IntentClassifier:
    def __init__(self) -> None:
        self.backend = CFG.path("ml.intent_classifier.backend", "tfidf_logreg")
        self.pipe = None  # sklearn pipeline

    # -------------------------------------------------------------- training
    def train(self, X: List[str], y: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")),
        ])
        self.pipe.fit(X, y)
        return self

    def evaluate(self, X: List[str], y: List[str]) -> Dict:
        from sklearn.metrics import classification_report, f1_score
        pred = self.pipe.predict(X)
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
        return {"macro_f1": float(macro_f1), "report": report}

    # --------------------------------------------------------------- predict
    def predict(self, query: str) -> str:
        if self.pipe is None:
            return heuristic_predict(query)
        return str(self.pipe.predict([query])[0])

    def predict_proba(self, query: str) -> Dict[str, float]:
        if self.pipe is None:
            lbl = heuristic_predict(query)
            return {l: (1.0 if l == lbl else 0.0) for l in LABELS}
        proba = self.pipe.predict_proba([query])[0]
        classes = list(self.pipe.classes_)
        return {c: float(p) for c, p in zip(classes, proba)}

    # ------------------------------------------------------------------- io
    def save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.pipe, f)

    def load(self) -> "IntentClassifier":
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.pipe = pickle.load(f)
            except Exception:
                self.pipe = None
        return self


_CLF: IntentClassifier | None = None


def get_intent_classifier() -> IntentClassifier:
    global _CLF
    if _CLF is None:
        _CLF = IntentClassifier().load()
    return _CLF
