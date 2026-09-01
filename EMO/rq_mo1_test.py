"""
rq_mo1_test.py - COMPLETE, CORRECT implementation.

Tests RQ-MO1: "Does a single hyperparameter configuration jointly optimize
noise-tolerance and sparsity-tolerance, or is there an inherent trade-off?"

This uses the REAL PPO/CMA-ES training infrastructure from the fixed_runs /
research-internship codebase:
    src.core.experiment.ExperimentConfig
    src.algorithms.ppo_runner.run_ppo
    src.algorithms.cmaes_runner.run_cmaes
    src.environments.wrappers.get_shared_obs_stats, get_sparsity_thresholds

A MOEA (NSGA-II / SMS-EMOA) searches the hyperparameter space. Each candidate
config is evaluated by ACTUALLY training the base algorithm under the two
stressors, producing two objectives:
    f1 = noise-tolerance   = mean return over sigma in {0.1, 0.3, 0.5}
    f2 = sparsity-tolerance = mean return over rho in {S1-S3 (-> known keys)}
The front's existence + hypervolume answers whether a trade-off exists.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

EMO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EMO_ROOT))

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Real infrastructure
from src.core.experiment import ExperimentConfig
from src.algorithms.ppo_runner import run_ppo
from src.algorithms.cmaes_runner import run_cmaes
from src.environments.wrappers import get_shared_obs_stats, get_sparsity_thresholds

# ------------------------------------------------  CONFIG  ------------------------------------------------
# These MUST match the real runners' expectations. Loud-domain mapping.
NOISE_LEVELS = [0.3, 0.5, 0.7]      # f1 inputs
SPARSITY_LEVELS = ["medium", "sparse", "very_sparse"]  # f2 inputs (real key names)
N_INNER_SEEDS = 2                    # inner seeds per config (increase for rigor)
ENV_NAME = "HalfCheetah-v4"

PPO_VAR_NAMES = ["learning_rate", "n_steps", "batch_size", "n_epochs",
                 "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef"]

PPO_BOUNDS_LO = [1e-5, 512, 32, 3, 0.95, 0.90, 0.1, 0.0, 0.3]
PPO_BOUNDS_HI = [1e-3, 4096, 256, 12, 0.999, 0.99, 0.3, 0.05, 0.7]

CMA_VAR_NAMES = ["sigma_init", "popsize"]
CMA_BOUNDS_LO = [0.5, 10]
CMA_BOUNDS_HI = [2.0, 40]


# ------------------------------------------------  PROBLEM  ------------------------------------------------
class EMOHyperSearchProblem(Problem):
    """n_var-dimensional continuous problem over base-alg hyperparameters.

    eval_fn(cfg_dict) -> (f1, f2)  (both higher-is-better).
    pymoo minimises by default, so we wrap the evaluator.
    """
    def __init__(self, var_names, bounds_lo, bounds_hi, eval_fn):
        self.var_names = list(var_names)
        self.cfg_bounds = {n: (lo, hi) for n, lo, hi in zip(var_names, bounds_lo, bounds_hi)}
        self.eval_fn = eval_fn
        super().__init__(
            n_var=len(var_names), n_obj=2, n_constr=0,
            xl=np.array(bounds_lo, dtype=float),
            xu=np.array(bounds_hi, dtype=float),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        for row in x:
            cfg = {n: float(row[i]) for i, n in enumerate(self.var_names)}
            f1, f2 = self.eval_fn(cfg)
            F.append([-f1, -f2])   # pymoo minimises -> negate for maximisation
        out["F"] = np.array(F, dtype=float)


# ------------------------------------------------  EVALUATORS  ------------------------------------------------
class Evaluator:
    def __init__(self, algorithm, shared_dir, n_inner_seeds=2, env=ENV_NAME):
        self.algorithm = algorithm
        self.env = env
        self.n_inner_seeds = n_inner_seeds
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.obs_stats = get_shared_obs_stats(env, self.shared_dir, logger=None)
        self.thresholds = get_sparsity_thresholds(env, self.shared_dir, logger=None)

    def _make_config(self, cfg_dict, seed):
        c = ExperimentConfig()
        c.host = "isca"
        c.env_names = [self.env]
        c.algorithms = [self.algorithm]
        c.seeds = [seed]
        c.ppo_total_timesteps = 200_000          # short eval for MOEA pilot
        c.cmaes_generations = 60
        c.cmaes_population_size = 20
        c.cmaes_hidden_dims = [64, 64]
        c.ppo_hidden_dims = [64, 64]
        c.n_eval_episodes = 5
        c.shared_dir = self.shared_dir
        c.resume = False                          # never resume for search trials
        c.threshold_override = self.thresholds
        if self.algorithm == "PPO":
            # NOTE: do NOT set c.optuna_params here. The real ppo_runner
            # overwrites the typed fields below with raw (possibly float)
            # values from optuna_params dict, which breaks int-only params
            # like n_steps/batch_size. We set the typed fields directly.
            c.ppo_learning_rate = float(cfg_dict.get("learning_rate", 3e-4))
            c.ppo_n_steps = int(round(cfg_dict.get("n_steps", 2048)))
            c.ppo_batch_size = int(round(cfg_dict.get("batch_size", 64)))
            c.ppo_n_epochs = int(round(cfg_dict.get("n_epochs", 10)))
            c.ppo_gamma = float(cfg_dict.get("gamma", 0.99))
            c.ppo_gae_lambda = float(cfg_dict.get("gae_lambda", 0.95))
            c.ppo_clip_range = float(cfg_dict.get("clip_range", 0.2))
            c.ppo_ent_coef = float(cfg_dict.get("ent_coef", 0.01))
            c.ppo_vf_coef = float(cfg_dict.get("vf_coef", 0.5))
        else:
            # Same for CMA-ES: set typed fields directly, don't set cmaes_optuna_params.
            c.cmaes_initial_sigma = float(cfg_dict.get("sigma_init", 1.0))
            if "popsize" in cfg_dict:
                c.cmaes_population_size = int(round(cfg_dict["popsize"]))
        return c

    def _score_noise_tol(self, cfg_dict):
        vals = []
        for seed in range(self.n_inner_seeds):
            for sig in NOISE_LEVELS:
                c = self._make_config(cfg_dict, seed)
                if self.algorithm == "PPO":
                    r = run_ppo(self.env, sig, "dense", self.obs_stats, seed, c,
                                logger=None, sparsity_thresholds=self.thresholds)
                else:
                    r = run_cmaes(self.env, sig, "dense", self.obs_stats, seed, c,
                                  logger=None, sparsity_thresholds=self.thresholds)
                vals.append(float(r.get("final_return", r.get("clean_return", 0))))
        return float(np.mean(vals))

    def _score_sparsity_tol(self, cfg_dict):
        vals = []
        for seed in range(self.n_inner_seeds):
            for sp in SPARSITY_LEVELS:
                c = self._make_config(cfg_dict, seed)
                if self.algorithm == "PPO":
                    r = run_ppo(self.env, 0.0, sp, self.obs_stats, seed, c,
                                logger=None, sparsity_thresholds=self.thresholds)
                else:
                    r = run_cmaes(self.env, 0.0, sp, self.obs_stats, seed, c,
                                  logger=None, sparsity_thresholds=self.thresholds)
                vals.append(float(r.get("final_return", r.get("clean_return", 0))))
        return float(np.mean(vals))

    def __call__(self, cfg_dict):
        f1 = self._score_noise_tol(cfg_dict)
        f2 = self._score_sparsity_tol(cfg_dict)
        print(f"[EVAL {self.algorithm}] f1={f1:.1f} f2={f2:.1f}", flush=True)
        return f1, f2


# ------------------------------------------------  MOEA HELPERS  ------------------------------------------------
def pareto_front_indices(F):
    if F.shape[0] == 0:
        return np.array([], dtype=int)
    nds = NonDominatedSorting()
    return nds.do(-F, only_non_dominated_front=True).flatten()


def compute_hv(F, ref=(0.0, 0.0)):
    return float(HV(ref_point=np.array(ref)).calc(-F))


# ------------------------------------------------  MAIN  ------------------------------------------------
def run_rq_mo1(algorithm="PPO", moea="NSGA2", pop=20, n_gen=10, n_seq=2,
               out_dir="results/rq_mo1", scale_eval=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if algorithm.upper() == "PPO":
        var_names, lo, hi = PPO_VAR_NAMES, PPO_BOUNDS_LO, PPO_BOUNDS_HI
    else:
        var_names, lo, hi = CMA_VAR_NAMES, CMA_BOUNDS_LO, CMA_BOUNDS_HI

    # shared stats live under EMO/shared
    shared_dir = EMO_ROOT / "shared"
    evaluator = Evaluator(algorithm.upper(), shared_dir,
                          n_inner_seeds=N_INNER_SEEDS * scale_eval)

    problem = EMOHyperSearchProblem(var_names, lo, hi, evaluator)

    if moea.upper() == "NSGA2":
        alg = NSGA2(pop_size=pop, seed=42)
    else:
        alg = SMSEMOA(pop_size=pop, seed=42)

    print(f"\n{'='*60}\nRQ-MO1: {algorithm} | {moea} | pop={pop} gen={n_gen} seq={n_seq}\n{'='*60}")

    all_F, all_X = [], []
    hv_by_seed = []
    for seq in range(n_seq):
        res = minimize(problem, alg, ("n_gen", n_gen), seed=42 + seq,
                       verbose=True)
        f = -res.F            # back to maximisation space (f1, f2)
        all_F.append(f)
        all_X.append(res.X)
        hv_by_seed.append(compute_hv(f))
        print(f"[seq {seq+1}] final HV={hv_by_seed[-1]:.2f}")

    F_all = np.vstack(all_F)
    X_all = np.vstack(all_X)
    nd = pareto_front_indices(F_all)
    front_F = F_all[nd]
    front_X = X_all[nd]

    n_nondom = int(len(nd))
    tradeoff = n_nondom >= 2
    hv_mean = float(np.mean(hv_by_seed))

    verdict = {
        "algorithm": algorithm.upper(),
        "moea": moea,
        "n_final_points": int(len(F_all)),
        "n_nondominated": n_nondom,
        "non_degenerate": tradeoff,
        "tradeoff_exists": tradeoff,
        "mean_hypervolume": hv_mean,
        "front": front_F.tolist(),
        "front_vars": front_X.tolist(),
        "interpretation": (
            "TRADE-OFF EXISTS: Pareto front has multiple non-dominated configs; "
            "no single config jointly optimizes both." if tradeoff else
            "NO TRADE-OFF: A single config dominates both objectives (front degenerate)."
        ),
    }

    with open(out_dir / f"rq_mo1_verdict_{algorithm}.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n>>> RQ-MO1 VERDICT [{algorithm}]:")
    print(f"    non-degenerate ({n_nondom} front pts): {tradeoff}")
    print(f"    mean HV: {hv_mean:.2f}")
    print(f"    {verdict['interpretation']}")
    print(f"    saved -> {out_dir / f'rq_mo1_verdict_{algorithm}.json'}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["PPO", "CMA-ES"], default="PPO")
    ap.add_argument("--moea", choices=["NSGA2", "SMS_EMOA"], default="NSGA2")
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--gen", type=int, default=10)
    ap.add_argument("--seq", type=int, default=2)
    ap.add_argument("--scale-eval", type=int, default=1)
    ap.add_argument("--out", default="results/rq_mo1")
    args = ap.parse_args()

    v = run_rq_mo1(algorithm=args.algo, moea=args.moea, pop=args.pop,
                   n_gen=args.gen, n_seq=args.seq, out_dir=args.out,
                   scale_eval=args.scale_eval)
    return 0 if v["tradeoff_exists"] else 1


if __name__ == "__main__":
    sys.exit(main())