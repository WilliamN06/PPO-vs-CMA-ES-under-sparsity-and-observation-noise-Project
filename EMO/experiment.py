# --------------------------------------------------------------
# experiment.py
# --------------------------------------------------------------
# High-level EMO experiment manager. CLI:
#   python experiment.py --mode pilot
#   python experiment.py --mode optuna --algo PPO --trials 20
#   python experiment.py --mode moea  --algo NSGA2 --n-gen 10
# --------------------------------------------------------------

import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emo_driver import run_moea
from optuna_emo_search import run_optuna_emo, PPO_DOMAIN, CMA_DOMAIN

DEFAULT_LOG = "results/emo"


def placeholder_eval(config):
    np.random.seed(int(config.get("_seed", 0)))
    return (np.random.normal(500, 50), np.random.normal(400, 40))


def run_pilot():
    outs = {}
    for algo in ["NSGA2", "SMS_EMOA"]:
        F, X, hv, rep = run_moea(
            eval_func=placeholder_eval,
            algorithm=algo,
            pop_size=20, n_gen=10, n_seq=2,
            out_dir=DEFAULT_LOG,
            algorithm_name="PPO")
        # Run again for CMA-ES to showcase both base algorithms
        F2, X2, hv2, rep2 = run_moea(
            eval_func=placeholder_eval,
            algorithm=algo,
            pop_size=20, n_gen=10, n_seq=2,
            out_dir=DEFAULT_LOG,
            algorithm_name="CMA-ES")
        outs[f"{algo}_PPO"] = rep["mean_hv"]
        outs[f"{algo}_CMA"] = rep2["mean_hv"]
    print("\n=== PILOT SUMMARY ===")
    for k, v in outs.items():
        print(f"  {k}: mean_HV={v:.4f}")


def run_optuna_cli(args):
    run_optuna_emo(study_name=args.study,
                   algorithm_name=args.algo,
                   n_trials=args.trials,
                   jobs=args.jobs,
                   weights=(args.w1, args.w2))


def main(argv=None):
    ap = argparse.ArgumentParser(description="EMO experiment manager")
    ap.add_argument("--mode", choices=["pilot", "optuna"], default="pilot")
    ap.add_argument("--algo", default="PPO", choices=["PPO", "CMA-ES"])
    ap.add_argument("--study", default="emo_ppo")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=1.0)
    args = ap.parse_args(argv)

    if args.mode == "pilot":
        run_pilot()
    else:
        run_optuna_cli(args)


if __name__ == "__main__":
    main()
