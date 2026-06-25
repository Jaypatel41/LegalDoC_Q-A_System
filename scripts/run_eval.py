"""Run the full ablation evaluation and print the resume table.

Usage:
  USE_STUB_LLM=1 python -m scripts.run_eval     # offline proxy metrics
  python -m scripts.run_eval                     # real RAGAS (needs LLM key)

Outputs:
  * console ablation table (metric per stage + delta)
  * data/models/ablation_results.json
"""
import json
from pathlib import Path

from src.config import CFG
from src.evaluation import ragas_eval
from src.evaluation.ablation import run_ablation

METRICS = ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]


def main():
    print(f"Metrics mode: {ragas_eval.metrics_mode()}")
    print("Running ablation (this calls the LLM per question per stage)...\n")
    results = run_ablation()

    # ---- ablation table ----
    hdr = f"{'stage':<16}" + "".join(f"{m[:13]:>15}" for m in METRICS)
    print(hdr)
    print("-" * len(hdr))
    prev = None
    for row in results:
        m = row["metrics"]
        line = f"{row['stage']:<16}" + "".join(f"{m[k]:>15.3f}" for k in METRICS)
        print(line)
        if prev is not None:
            delta = "".join(f"{(m[k]-prev[k]):>+15.3f}" for k in METRICS)
            print(f"{'  Δ vs prev':<16}{delta}")
        prev = m

    print("\n--- operational metrics ---")
    print(f"{'stage':<16}{'route_acc':>12}{'halluc_rate':>14}"
          f"{'incorrect':>12}{'ci95':>16}")
    for row in results:
        ci = row["incorrect_ci95"]
        print(f"{row['stage']:<16}{row['route_accuracy']:>12.3f}"
              f"{row['hallucination_rate']:>14.3f}{row['incorrect_rate']:>12.3f}"
              f"{f'[{ci[0]:.3f},{ci[1]:.3f}]':>16}")

    # ---- resume deltas ----
    base, final = results[0]["metrics"], results[-1]["metrics"]
    print("\n=== RESUME NUMBERS ===")
    print(f"context_recall  baseline {base['context_recall']:.3f} -> "
          f"final {final['context_recall']:.3f}  "
          f"(+{(final['context_recall']-base['context_recall'])*100:.1f}%)")
    print(f"faithfulness    baseline {base['faithfulness']:.3f} -> "
          f"final {final['faithfulness']:.3f}")
    fi = results[0]["incorrect_rate"]; li = results[-1]["incorrect_rate"]
    lci = results[-1]["incorrect_ci95"]
    print(f"incorrect-rate  baseline {fi*100:.1f}% -> final {li*100:.1f}% "
          f"(95% CI {lci[0]*100:.1f}-{lci[1]*100:.1f}%)")

    out = Path(CFG["paths"]["models_dir"]) / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": ragas_eval.metrics_mode(),
                               "results": results}, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
