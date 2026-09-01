# --------------------------------------------------------------
# emo_driver.py
# --------------------------------------------------------------
# Runs a chosen MOEA (NSGA-II or SMS-EMOA) on EMO_Problem and
# writes a CSV report useful for the pilot/trend test.
# Comprehensive logging of hypervolume per generation + final front.
# --------------------------------------------------------------

import os
import csv
import time
import json
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from emo_problem import (EMO_Problem, VAR_NAMES_PPO, VAR_NAMES_CMA,
                         BOUNDS_PPO_LO, BOUNDS_PPO_HI,
                         BOUNDS_CMA_LO, BOUNDS_CMA_HI)


def make_problem(eval_func, algorithm_name="PPO"):
    """Return an EMO_Problem with bounds set for the chosen algorithm."""
    if algorithm_name.upper() == "PPO":
        var_names = VAR_NAMES_PPO
        xl, xu = BOUNDS_PPO_LO, BOUNDS_PPO_HI
    else:
        var_names = VAR_NAMES_CMA
        xl, xu = BOUNDS_CMA_LO, BOUNDS_CMA_HI

    problem = EMO_Problem(n_var=len(var_names),
                          minimize=False,
                          eval_func=eval_func)
    problem.xl = xl
    problem.xu = xu
    return problem, var_names


def make_algorithm(name, pop_size, seed):
    name = name.upper()
    if name == "NSGA2":
        return NSGA2(pop_size=pop_size, seed=seed)
    if name in ("SMS_EMOA", "SMS-EMOA", "SMS"):
        return SMSEMOA(pop_size=pop_size, seed=seed)
    raise ValueError(f"Unsupported algorithm: {name}")


def pareto_front(F):
    """Return indices of non-dominated points (maximisation is handled by caller)."""
    nds = NonDominatedSorting()
    if F.shape[0] == 0:
        return np.array([], dtype=int)
    # For maximisation, use -F internally? We'll compute on raw F where
    # pymoo assumes minimisation by default. Here we negate.
    return nds.do(-F, only_non_dominated_front=True).flatten()


def compute_hypervolume(F, ref_point):
    # Maximization problem: negate so that HV indicator (built for
    # minimization) yields the maximization hypervolume correctly.
    F_neg = -np.asarray(F, dtype=float)
    ref_neg = -np.array(ref_point, dtype=float)
    hv = HV(ref_point=ref_neg)
    return float(hv(F_neg))


def run_moea(eval_func,
             algorithm="NSGA2",
             pop_size=30,
             n_gen=30,
             n_seq=1,
             out_dir=".",
             algorithm_name="PPO",
             ref_point=(0.0, 0.0),
             logger=None):
    """
    Execute the MOEA, log HV across generations, and write CSV + JSON reports.
    Returns (concatenated_front, hv_history, reports) where reports is a dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    problem, var_names = make_problem(eval_func, algorithm_name)
    ref_point = np.array(ref_point, dtype=float)

    all_F = []
    all_X = []
    hv_seq = []           # per-generation hv (average over seeds when available)
    hv_by_seed = []
    times = []

    for seq in range(n_seq):
        print(f"\n=== MOEA run {seq+1}/{n_seq} ({algorithm.upper()}) ===")
        t0 = time.time()
        alg = make_algorithm(algorithm, pop_size, seed=42 + seq)
        res = minimize(problem, alg, ("n_gen", n_gen),
                       seed=42 + seq, save_history=True, verbose=False)
        elapsed = time.time() - t0

        F = res.F
        X = res.X
        all_F.append(F)
        all_X.append(X)

        # per-run HV + history
        hv_val = compute_hypervolume(F, ref_point)
        hv_by_seed.append(hv_val)
        print(f"Final hyper-volume (seed {seq+1}): {hv_val:.4f}  [{elapsed:.1f}s]")

        # generation history
        run_hv = []
        if hasattr(res, "history"):
            for entry in res.history:
                pop = getattr(entry, "pop", None)
                if pop is None:
                    run_hv.append(np.nan)
                    continue
                f_hist = pop.get("F")
                if f_hist is not None and len(f_hist) > 0:
                    try:
                        run_hv.append(compute_hypervolume(f_hist, ref_point))
                    except Exception:
                        run_hv.append(np.nan)
                else:
                    run_hv.append(np.nan)
        hv_seq.append(run_hv)
        times.append(elapsed)

    # ----- aggregate front (union + non-dominated) ----------------
    F_all = np.vstack(all_F) if len(all_F) else np.empty((0, 2))
    X_all = np.vstack(all_X) if len(all_X) else np.empty((0, problem.n_var))
    if len(F_all):
        nd_idx = pareto_front(F_all)
        front_F = F_all[nd_idx]
        front_X = X_all[nd_idx]
    else:
        front_F = np.empty((0, 2))
        front_X = np.empty((0, problem.n_var))

    # ----- build per-generation mean HV (final-friendly) -----------
    max_len = max((len(h) for h in hv_seq), default=0)
    mean_hv = []
    for g in range(max_len):
        vals = [h[g] for h in hv_seq if g < len(h) and not np.isnan(h[g])]
        mean_hv.append(float(np.mean(vals)) if vals else np.nan)

    report = {
        "algorithm": algorithm.upper(),
        "algorithm_name": algorithm_name,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "n_seq": n_seq,
        "n_final_points": int(len(front_F)),
        "n_final_objectives": 2,
        "front_F": front_F.tolist(),
        "front_X": front_X.tolist(),
        "ref_point": ref_point.tolist(),
        "hv_by_seed": hv_by_seed,
        "mean_hv": float(np.mean(hv_by_seed)) if hv_by_seed else None,
        "mean_hv_final": float(mean_hv[-1]) if mean_hv else None,
        "hv_generation": mean_hv,
        "times_sec": times,
        "var_names": var_names,
    }

    # CSV report
    csv_path = os.path.join(out_dir, f"emo_report_{algorithm_name}_{algorithm.upper()}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", report["algorithm"], "algo_name", algorithm_name])
        w.writerow(["pop_size", pop_size, "n_gen", n_gen, "n_seq", n_seq])
        w.writerow(["n_final_points", report["n_final_points"]])
        w.writerow(["mean_hv", report["mean_hv"]])
        w.writerow(["generation"] + list(range(1, max_len + 1)))
        w.writerow(["hv"] + [f"{v:.4f}" if v is not None else "" for v in mean_hv])

    # JSON report
    json_path = os.path.join(out_dir, f"emo_front_{algorithm_name}_{algorithm.upper()}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    if logger:
        logger.info(f"MOEA {algorithm} on {algorithm_name}: mean_hv={report['mean_hv']:.4f}, "
                    f"n_points={report['n_final_points']}")
    print(f"[emo_driver] Reports -> {csv_path} , {json_path}")
    return front_F, front_X, mean_hv, report


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    def placeholder_eval(config):
        np.random.seed()
        return (np.random.normal(500, 50), np.random.normal(400, 40))

    os.makedirs("./test_run", exist_ok=True)
    F, X, hv, rep = run_moea(eval_func=placeholder_eval, algorithm="NSGA2",
                             pop_size=20, n_gen=10, n_seq=2,
                             out_dir="./test_run", algorithm_name="PPO")
    print("Front size:", F.shape[0], "mean HV:", rep["mean_hv"])
