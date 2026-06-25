"""Run REAL RAGAS on the live pipeline.

Generates answers with the configured LLM (Groq llama-3.1-8b-instant here) over a
balanced subset of the test set, then scores them with genuine RAGAS 0.4.x
metrics (LLM-as-judge). Kept to a small subset to respect free-tier rate limits.

Usage:
  python -m scripts.run_ragas         # default 8 questions
  python -m scripts.run_ragas 12      # custom subset size
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.config import CFG, env
from src.evaluation import ragas_eval
from src.evaluation.ablation import load_testset
from src.pipeline import AblationFlags, LegalRAG


def balanced_subset(testset, k):
    by_route = defaultdict(list)
    for ex in testset:
        by_route[ex.get("route", "?")].append(ex)
    out, routes = [], list(by_route)
    i = 0
    while len(out) < k and any(by_route.values()):
        r = routes[i % len(routes)]
        if by_route[r]:
            out.append(by_route[r].pop(0))
        i += 1
    return out[:k]


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if env("USE_STUB_LLM", "0") == "1":
        print("USE_STUB_LLM=1 -> real RAGAS disabled. Unset it in .env.")
        return

    testset = balanced_subset(load_testset(), k)
    print(f"Generating answers with '{env('LOCAL_MODEL')}' for {len(testset)} "
          f"questions (full pipeline: MultiQuery + hybrid + rerank + guardrails)...\n")

    rag = LegalRAG(AblationFlags())  # all accuracy layers ON
    records = []
    for i, ex in enumerate(testset, 1):
        res = rag.answer(ex["question"])
        records.append({
            "question": ex["question"],
            "answer": res.answer,
            "contexts": res.contexts,
            "ground_truth": ex["ground_truth"],
        })
        print(f"  [{i}/{len(testset)}] {ex['route']:<9} {ex['question'][:60]}")

    print(f"\nScoring with RAGAS ({ragas_eval.metrics_mode()}) — this calls the "
          f"judge LLM per metric per question, please wait...\n")
    scores = ragas_eval.evaluate_records(records)

    print("=== REAL RAGAS SCORES ===")
    for k_, v in scores.items():
        print(f"  {k_:<20} {v}")

    out = Path(CFG["paths"]["models_dir"]) / "ragas_real.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": ragas_eval.metrics_mode(),
                               "n": len(records), "scores": scores,
                               "model": env("LOCAL_MODEL")}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
