# --------------------------------------------------------------
# emo_problem.py
# --------------------------------------------------------------
# Multi-objective problem definition for the EMO hyper-parameter search.
# Two objectives (both maximised):
#   f1 = noise-tolerance   = mean return across sigma in {0.1, 0.3, 0.5}
#   f2 = sparsity-tolerance= mean return across rho in {S1, S2, S3}
#
# Decision variables = hyper-parameters of the base algorithm.
#   PPO:    [lr, clip_eps, n_epochs, batch_size, ent_coef,
#            gae_lambda, vf_coef, max_grad_norm]
#   CMA-ES: [sigma0, pop_mult, c1, cmu, cs, cc, weight_scheme]
#
# The evaluator (callable `eval_func`) must return a 2-element
# numpy array [f1, f2] (higher = better).
# --------------------------------------------------------------

import numpy as np
from pymoo.core.problem import Problem


class EMO_Problem(Problem):
    def __init__(self,
                 n_var=8,                 # number of decision vars
                 minimize=False,
                 eval_func=None):
        xl = np.array([1e-4, 0.1, 1, 16, 1e-4, 0.95, 1e-3, 0.1])
        xu = np.array([1e-2, 0.5, 20, 256, 1e-1, 0.999, 1.0, 5.0])

        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_constr=0,
                         type_var=np.dtype(np.float64),
                         xl=xl,
                         xu=xu)

        self.minimize = minimize
        self.eval_func = eval_func

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        for i in range(x.shape[0]):
            # ------------------------------------------------------
            # CALL YOUR ACTUAL TRAINING FUNCTION HERE.
            # The placeholder below returns synthetic numbers so the
            # code runs without errors; replace with your evaluator.
            #   cfg = dict(zip(VAR_NAMES_PPO, x[i]))
            #   f1, f2 = self.eval_func(cfg)
            # ------------------------------------------------------
            np.random.seed(0)
            f1 = np.random.normal(loc=500, scale=50)
            f2 = np.random.normal(loc=400, scale=40)
            F.append([f1, f2])
        out["F"] = np.array(F, dtype=float)


VAR_NAMES_PPO = ["lr", "clip_eps", "n_epochs", "batch_size",
                 "ent_coef", "gae_lambda", "vf_coef", "max_grad_norm"]

VAR_NAMES_CMA = ["sigma0", "pop_mult", "c1", "cmu", "cs", "cc", "weight_scheme"]

BOUNDS_PPO_LO = np.array([1e-4, 0.1, 1, 16, 1e-4, 0.95, 1e-3, 0.1])
BOUNDS_PPO_HI = np.array([1e-2, 0.5, 20, 256, 1e-1, 0.999, 1.0, 5.0])
BOUNDS_CMA_LO = np.array([0.01, 4, 0.01, 0.01, 1e-4, 1e-3, 1e-4])
BOUNDS_CMA_HI = np.array([1.0, 64, 0.5, 1.0, 0.2, 0.5, 1.0])
