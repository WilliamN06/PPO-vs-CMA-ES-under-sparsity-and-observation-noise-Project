# --------------------------------------------------------------
# optuna_emo_search.py
# --------------------------------------------------------------
# Optuna-based multi-objective search for the EMO problem.
#   * Each Optuna trial samples a hyper-parameter configuration.
#   * The configuration is plugged into the EMO evaluator (which runs the
#     base algorithm under the two stressors) to get [f1, f2].
#   * A scalarised objective (e.g. hypervolume of a 3-point / k-point front)
#     or a single aggregated scalar is optimised, with comprehensive logging.
#
# Two modes:
#   scalar mode: maximise w1*f1 + w2*f2 (use Tchebycheff weights)
#   (future: Optuna's MultiObjectiveSampler)
# --------------------------------------------------------------

import argparse
import json
import os
import time
import numpy as np
import optuna
from pathlib import Path

try:
    from optuna.samplers import TPESampler
except Exception:
    TPESampler = None

EMO_ROOT = Path(__file__).resolve().parent

PPO_DOMAIN = {
    "lr":          (1e-4, 1e-2, "log"),
    "clip_eps":    (0.1, 0.5, "float"),
    "n_epochs":    (1, 20, "int"),
    "batch_size":  (16, 256, "int"),
    "ent_coef":    (1e-4, 1e-1, "log"),
    "gae_lambda":  (0.9, 0.999, "float"),
    "vf_coef":     (0.1, 1.0, "float"),
    "max_grad_norm": (0.1, 5.0, "float"),
}

CMA_DOMAIN = {
    "sigma0": (0.1, 2.0, "float"),
    "pop_mult": (4, 64, "int"),
    "c1": (0.01, 0.5, "float"),
    "cmu": (0.1, 1.0, "float"),
    "cs": (0.1, 0.5, "float"),
    "cc": (0.1, 0.5, "float"),
    "weight_scheme": ("flat", "inverse", "adaptive"),
}


def evaluate_trial_config(config, algorithm_name="PPO", n_inner_seeds=2):
    """
    Evaluate a hyper-parameter configuration, returning [noise_tol, sparsity_tol].

    REPLACE the synthetic body with your real training pipeline:
        f1 = mean return over sigma in {0.1, 0.3, 0.5}
        f2 = mean return over rho in {S1, S2, S3}
    The `config` dict holds the sampled hyper-parameters.
    """
    # --- synthetic proxy objective (replace me) ----
    rng = np.random.RandomState(int(config.get("_seed", 0)))
    # noise tolerance benefits from low lr / high ent exploration; sparsity
    # tolerance benefits from larger batch ... constructed so a real trade-off
    # (non-degenerate front) can emerge on synthetic data.
    lr = config.get("lr", 3e-4)
    ent = config.get("ent_coef", 0.01)
    n_steps_like = config.get("batch_size", 64)

    noise_tol = 300 + 300 * np.clip(lr / 1e-3, 0, 1) - 80 * ent * np.log1p(lr * 1e4)
    sparsity_tol = 300 + 120 * np.log1p(n_steps_like / 16) - 200 * np.clip(lr / 5e-3, 0, 1)
    noise_tol += rng.normal(0, 15)
    sparsity_tol += rng.normal(0, 15)
    return float(noise_tol), float(sparsity_tol)


def tchebycheff_value(f, ref, weights):
    """Weighted Tchebycheff distance to a reference point (minimised)."""
    return float(np.max(weights * (ref - np.array(f))))


def emo_objective(trial, domain, algorithm_name="PPO", n_inner_seeds=2,
                  weights=(1.0, 1.0), ref=None):
    """
    Optuna trial scalarisAL-objective (maximised). Samples a config, evaluates
    the two objectives, and returns a negated-augmented Tchebycheff value so
    Optuna maximises it.
    """
    config = {}
    for name, spec in domain.items():
        if isinstance(spec, tuple) and len(spec) > 0 and isinstance(spec[0], str):
            # categorical domain (e.g. weight_scheme = ("flat","inverse","adaptive"))
            config[name] = trial.suggest_categorical(name, list(spec))
            continue
        lo, hi, typ = spec
        if typ == "int":
            config[name] = trial.suggest_int(name, lo, hi)
        elif typ == "log":
            config[name] = trial.suggest_float(name, lo, hi, log=True)
        else:
            config[name] = trial.suggest_float(name, lo, hi)

    config["_seed"] = trial.number

    fs = []
    for s in range(n_inner_seeds):
        config["_seed"] = trial.number * 100 + s
        f1, f2 = evaluate_trial_config(config, algorithm_name, n_inner_seeds)
        fs.append((f1, f2))
    f_mean = (
        float(np.mean([x[0] for x in fs])),
        float(np.mean([x[1] for x in fs])),
    )

    if ref is None:
        ref = np.array([600.0, 600.0])
    tcb = tchebycheff_value(f_mean, ref, np.array(weights))
    return float(-tcb)          # Optuna maximises


def run_optuna_emo(study_name="emo_ppo",
                   algorithm_name="PPO",
                   n_trials=20,
                   n_inner_seeds=2,
                   weights=(1.0, 1.0),
                   jobs=1,
                   log_dir="results/optuna"):
    if algorithm_name.upper() == "PPO":
        domain = PPO_DOMAIN
    else:
        domain = CMA_DOMAIN

    log_dir = EMO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    dir_kwargs = dict(direction="maximize")
    if TPESampler is not None:
        dir_kwargs["sampler"] = TPESampler(seed=0)

    study = optuna.create_study(
        study_name=study_name,
        storage=None,
        load_if_exists=False,
        **dir_kwargs,
    )

    print(f"[EMO-OPTUNA] study={study_name} algo={algorithm_name} "
          f"trials={n_trials} inner_seeds={n_inner_seeds} jobs={jobs}")

    def obj(trial):
        return emo_objective(trial, domain, algorithm_name,
                             n_inner_seeds, weights=weights)

    t0 = time.time()
    study.optimize(obj, n_trials=n_trials, n_jobs=jobs, show_progress_bar=False)
    elapsed = time.time() - t0

    # ---- logging -------------------------------------------------
    best = study.best_trial
    records = []
    for t in study.trials:
        records.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })

    out_json = log_dir / f"{study_name}_log.json"
    with open(out_json, "w") as f:
        json.dump({"study": study_name, "algorithm_name": algorithm_name,
                   "n_trials": n_trials, "elapsed_sec": elapsed,
                   "trials": records, "best": best.params,
                   "best_value": best.value}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    hp_file = EMO_ROOT / "config" / "hyperparams.json"
    hp_file.parent.mkdir(parents=True, exist_ok=True)
    with open(hp_file, "w") as f:
        json.dump({algorithm_name: best.params}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"[EMO-OPTUNA] best value={best.value:.4f}, params={best.params}")
    print(f"[EMO-OPTUNA] log -> {out_json}")
    print(f"[EMO-OPTUNA] best config -> {hp_file}")
    return study, best.params


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="emo_ppo")
    ap.add_argument("--algo", default="PPO", choices=["PPO", "CMA-ES"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=1.0)
    args = ap.parse_args()
    run_optuna_emo(study_name=args.study,
                   algorithm_name=args.algo,
                   n_trials=args.trials,
                   jobs=args.jobs,
                   weights=(args.w1, args.w2))
