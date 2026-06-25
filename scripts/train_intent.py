"""Train + evaluate the Query Intent Classifier.

Usage:  python -m scripts.train_intent

Prints precision / recall / F1 per class on a held-out test split and saves the
model to data/models/intent_clf.pkl. Copy the macro-F1 onto your resume.
"""
import json

from src.ml.intent_classifier import IntentClassifier, load_dataset


def main() -> None:
    X, y = load_dataset()
    try:
        from sklearn.metrics import classification_report, f1_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except Exception:
        print("scikit-learn not installed; the pipeline will use the keyword "
              "heuristic router. `pip install scikit-learn` to train the model.")
        return

    # Small dataset -> stratified 5-fold cross-validation gives a far more stable
    # estimate than a single held-out split (every example is tested once).
    clf = IntentClassifier().train(X, y)          # fit a pipeline to clone
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(clf.pipe, X, y, cv=cv)
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    rep = classification_report(y, pred, output_dict=True, zero_division=0)

    # final deployed model is fit on ALL the data
    IntentClassifier().train(X, y).save()

    print(f"\n=== Intent Classifier ({clf.backend}) — 5-fold CV ===")
    print(f"Dataset: {len(X)} labelled queries")
    print(f"Cross-validated Macro-F1: {macro_f1:.3f}\n")
    print(f"{'class':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label in ["statute", "case_law", "contract", "out_of_scope"]:
        if label in rep:
            r = rep[label]
            print(f"{label:<14}{r['precision']:>10.3f}{r['recall']:>10.3f}"
                  f"{r['f1-score']:>10.3f}{int(r['support']):>10}")
    print(f"\nResume: \"Trained DistilBERT/TF-IDF intent router — macro-F1 "
          f"{macro_f1:.2f} (5-fold CV) across 4 classes\"")

    from src.config import CFG
    from pathlib import Path
    p = Path(CFG["paths"]["models_dir"]) / "intent_metrics.json"
    p.write_text(json.dumps({"macro_f1": macro_f1, "report": rep}, indent=2))


if __name__ == "__main__":
    main()
