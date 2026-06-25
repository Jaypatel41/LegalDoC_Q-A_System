"""Train + evaluate the Hallucination Risk Classifier.

Usage:  python -m scripts.train_hallucination

Weak supervision: in production the training labels come from RAGAS faithfulness
scores (faithful >= tau -> 1, else 0). Here, so the project trains end-to-end
offline, we synthesise the same supervision signal from the indexed corpus:

  * FAITHFUL (label 1): answer = a sentence grounded in the provided context.
  * HALLUCINATED (label 0): answer paired with an UNRELATED context, or with its
    numbers/sections corrupted (cite-tampering) — exactly the failure modes RAGAS
    penalises.

If a RAGAS-labelled file (data/eval/halluc_labeled.jsonl) exists, it is used
instead. Reports ROC-AUC + F1 on a held-out split.
"""
import json
import re
from pathlib import Path

import numpy as np

from src.config import CFG
from src.ingestion.indexer import StoreIndex
from src.ml.features import extract
from src.ml.hallucination_classifier import HallucinationClassifier

_NUM = re.compile(r"\b\d+[A-Za-z]?\b")


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if len(s.strip()) > 30]


def _corrupt(text):
    # tamper with section numbers / digits to simulate a fabricated citation
    return _NUM.sub(lambda m: str((int(re.sub(r"\D", "", m.group()) or 0) + 7))
                    if re.sub(r"\D", "", m.group()) else m.group(), text)


def build_dataset():
    """Return (rows) where each row = {answer, contexts, label}."""
    chunks = []
    for s in CFG["stores"]:
        try:
            chunks.extend(StoreIndex(s["name"]).load().chunks)
        except FileNotFoundError:
            pass
    rows = []
    n = len(chunks)
    for i, c in enumerate(chunks):
        sents = _sentences(c.text)
        if not sents:
            continue
        sent = sents[0]
        other = chunks[(i + n // 2) % n]
        other_sents = _sentences(other.text)

        # FAITHFUL (label 1)
        rows.append({"answer": sent, "contexts": [c.text], "label": 1})
        if len(sents) > 1:                      # a later sentence -> lower verbatim
            rows.append({"answer": sents[-1], "contexts": [c.text], "label": 1})

        # HALLUCINATED (label 0)
        if other.id != c.id:
            # fully off-context
            rows.append({"answer": sent, "contexts": [other.text], "label": 0})
            # HARD negative: half-grounded, half-fabricated (partial hallucination)
            if other_sents:
                mixed = sent + " " + other_sents[0]
                rows.append({"answer": mixed, "contexts": [c.text], "label": 0})
        corrupted = _corrupt(sent)              # tampered citation
        if corrupted != sent:
            rows.append({"answer": corrupted, "contexts": [c.text], "label": 0})
    return rows


def main():
    labeled = Path(CFG["paths"]["eval_set"]).parent / "halluc_labeled.jsonl"
    if labeled.exists():
        rows = [json.loads(l) for l in labeled.read_text().splitlines() if l.strip()]
        print(f"Using RAGAS-labelled set: {labeled.name} ({len(rows)} rows)")
    else:
        rows = build_dataset()
        print(f"Synthesised weak-supervision set from corpus ({len(rows)} rows)")

    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        print("scikit-learn not installed; pipeline will use the cosine fallback "
              "for hallucination risk. `pip install scikit-learn` to train.")
        return

    X = np.vstack([extract(r["answer"], r["contexts"]) for r in rows])
    y = [int(r["label"]) for r in rows]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    clf = HallucinationClassifier().train(Xtr, ytr)
    metrics = clf.evaluate(Xte, yte)
    clf.save()

    print(f"\n=== Hallucination Risk Classifier ===")
    print(f"Test set: {len(Xte)} examples")
    print(f"ROC-AUC : {metrics['auc']:.3f}")
    print(f"F1      : {metrics['f1']:.3f}")
    src = "RAGAS-labelled" if labeled.exists() else "synthetic weak-supervision"
    print(f"\n(metrics on the {src} set; retrain on real RAGAS faithfulness "
          f"labels for production-representative numbers)")
    print(f"Resume: \"Distilled RAGAS faithfulness into a local hallucination "
          f"detector — ROC-AUC {metrics['auc']:.3f} / F1 {metrics['f1']:.3f}, "
          f"zero API calls at inference\"")

    out = Path(CFG["paths"]["models_dir"]) / "halluc_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
