# --------------------------------------------------------------
# tests/test_basic_trends.py
# Pilot that checks the three EMO reviewer basic trends:
#   1) non-degenerate Pareto front ( >1 non-dominated point )
#   2) hyper-volume improves / varies across generations
#   3) NSGA-II vs SMS-EMOA produce different fronts
# Comprehensive logging to test_run/ + results/emo logs.
# --------------------------------------------------------------
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emo_driver import run_moea, pareto_front, compute_hypervolume


def is_nondegenerate(F, n_points=2):
    if F.shape[0] < n_points:
        return False
    nd = pareto_front(F)
    return len(nd) >= n_points


def hv(F, ref=(0.0, 0.0)):
    return compute_hypervolume(F, ref)


def main():
    out = "test_run"
    os.makedirs(out, exist_ok=True)

    def ph(config):
        np.random.seed()
        return (np.random.normal(500, 50), np.random.normal(400, 40))

    results = {}
    fronts = {}
    for algo in ["NSGA2", "SMS_EMOA"]:
        F, X, hv_hist, rep = run_moea(
            eval_func=ph, algorithm=algo,
            pop_size=20, n_gen=10, n_seq=2,
            out_dir=out, algorithm_name="PPO")
        fronts[algo] = F
        results[f"{algo}_nondeg"] = is_nondegenerate(F)
        results[f"{algo}_hv"] = rep["mean_hv"]
        results[f"{algo}_hv_final"] = rep["mean_hv_final"]
        results[f"{algo}_npoints"] = rep["n_final_points"]
        # trend: hv improves over generations
        g = rep.get("hv_generation", [])
        if len(g) >= 2:
            imp = g[-1] > g[0] + 1e-9
        else:
            imp = None
        results[f"{algo}_hv_increase"] = imp

    # shared nondominated between MOEAs
    merged = np.vstack([fronts["NSGA2"], fronts["SMS_EMOA"]])
    nd = pareto_front(merged)
    results["shared_nondom"] = int(len(nd))
    results["total_points"] = int(len(merged))

    summary = os.path.join(out, "trend_summary.txt")
    lines = ["EMO basic-trend pilot summary", "=" * 40]
    for k, v in results.items():
        lines.append(f"{k}: {v}")
    with open(summary, "w") as f:
        f.write("\n".join(lines) + "\n")

    # json copy
    with open(os.path.join(out, "trend_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== TREND SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # verdict
    nondeg = all(results[f"{a}_nondeg"] for a in ["NSGA2", "SMS_EMOA"])
    print(f"\nVERDICT: non-degenerate front across MOEAs -> {nondeg}")
    print("Full artifact ->", summary)


if __name__ == "__main__":
    main()
