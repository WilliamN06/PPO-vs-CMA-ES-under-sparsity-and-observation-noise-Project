"""Fixed experiment configuration - handles None values properly with resume support"""
from pathlib import Path
import sys
import json
import time
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.utils.mujoco_patch import apply_mujoco_patch
from src.utils.utils import DiagnosticLogger, ProgressTracker, ResultSaver, make_task_id
from src.environments.wrappers import get_shared_obs_stats, get_sparsity_thresholds
from src.algorithms.ppo_runner import run_ppo
from src.algorithms.cmaes_runner import run_cmaes

apply_mujoco_patch()


def load_best_params_file(config) -> dict:
    """Load per-env best params from optuna_file into a nested dict
    {env: {'ppo': {...}, 'cmaes': {...}}} stored on config.best_params."""
    optuna_file = getattr(config, 'optuna_file', None)
    if not optuna_file or not Path(optuna_file).exists():
        return {}
    with open(optuna_file) as f:
        data = json.load(f)
    config.best_params = data
    print(f"[OPTUNA] Loaded best params from {optuna_file}", flush=True)
    return data


class ExperimentConfig:
    def __init__(self):
        self.name = "experiment"
        self.host = "isca"
        self.env_names = ["HalfCheetah-v4"]
        self.noise_levels = [0.3, 0.5, 0.7]
        self.sparsity_levels = ['medium', 'sparse', 'very_sparse']
        self.algorithms = ["PPO", "CMA-ES"]
        self.seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.n_jobs = 8
        self.budget_episodes = 5_000
        self.ppo_total_timesteps = 5_000_000
        self.cmaes_episode_length = 1000
        self.cmaes_hidden_dims = [64, 64]  # MATCHES PPO (was [64, 32])
        self.cmaes_population_size = 20  # 20 x 250 x 1000 = 5M env steps = PPO budget (budget-matched)
        self.cmaes_generations = 250  # Must be explicitly set (was 227)
        self.cmaes_initial_sigma = 1.0
        self.n_eval_episodes = 20
        self.output_dir = Path("results/experiment")
        self.shared_dir = Path("shared")
        self.optuna_file = None
        self.optuna_params = {}
        self.cmaes_optuna_params = {}
        self.resume = True
        
        self.ppo_learning_rate = 3e-4
        self.ppo_n_steps = 2048
        self.ppo_batch_size = 64
        self.ppo_n_epochs = 10
        self.ppo_gamma = 0.99
        self.ppo_gae_lambda = 0.95
        self.ppo_clip_range = 0.2
        self.ppo_ent_coef = 0.01  # standard exploration bonus (was 0.0)
        self.ppo_vf_coef = 0.5
        self.ppo_max_grad_norm = 0.5
        self.ppo_hidden_dims = [64, 64]  # now actually used (was [64, 32])


def run_single_task(task, config, logger=None):
    try:
        best = getattr(config, 'best_params', None) or {}
        env_entry = best.get(task['env_name'], {})
        config.optuna_params = env_entry.get('ppo', {}) if env_entry else {}
        config.cmaes_optuna_params = env_entry.get('cmaes', {}) if env_entry else {}
        fn = run_ppo if task['algorithm'] == 'PPO' else run_cmaes
        return fn(task['env_name'], task['noise_std'], task['sparsity_level'],
                  task['obs_stats'], task['seed'], config, logger,
                  task.get('sparsity_thresholds'))
    except Exception as e:
        print(f"[ERROR] Task failed: {e}", flush=True)
        traceback.print_exc()
        raise


def run_experiment(config):
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = DiagnosticLogger(log_dir, config.name, level="INFO")

    logger.info("=" * 70, "EXPERIMENT")
    logger.info(f"EXPERIMENT: {config.name}", "EXPERIMENT")
    logger.info(f"Start: {datetime.now().isoformat()}", "EXPERIMENT")
    logger.info(f"Workers: {config.n_jobs}", "EXPERIMENT")
    logger.info(f"PPO timesteps: {config.ppo_total_timesteps:,}", "BUDGET")
    logger.info(f"CMA-ES generations: {config.cmaes_generations}", "BUDGET")
    logger.info(f"CMA-ES population: {config.cmaes_population_size}", "BUDGET")
    logger.info(f"CMA-ES hidden: {config.cmaes_hidden_dims}", "BUDGET")
    logger.info("=" * 70, "EXPERIMENT")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load optuna best params (if optuna_file set and exists)
    load_best_params_file(config)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(config), f, indent=2, default=str)

    env_stats = {}
    sparsity_thresholds = {}
    for env_name in config.env_names:
        env_stats[env_name] = get_shared_obs_stats(env_name, config.shared_dir, logger=logger)
        sparsity_thresholds[env_name] = get_sparsity_thresholds(env_name, config.shared_dir, logger=logger)

    result_saver = ResultSaver(output_dir, config.name, host=config.host, logger=logger)

    # Tasks are generated SEED-MAJOR: every condition for a given seed is
    # created before the next seed, so completion follows seed order.
    tasks = []
    for env_name in config.env_names:
        obs_stats = env_stats[env_name]
        thresholds = sparsity_thresholds[env_name]
        for seed in config.seeds:
            for noise_std in config.noise_levels:
                for sparsity_level in config.sparsity_levels:
                    for algorithm in config.algorithms:
                        tasks.append({
                            'env_name': env_name,
                            'noise_std': noise_std,
                            'sparsity_level': sparsity_level,
                            'algorithm': algorithm,
                            'seed': seed,
                            'obs_stats': obs_stats,
                            'sparsity_thresholds': thresholds,
                            # Budget goes into the task id so runs trained with
                            # different budgets are never treated as identical.
                            'ppo_total_timesteps': config.ppo_total_timesteps,
                            'cmaes_generations': config.cmaes_generations,
                        })

    logger.info(f"Total tasks: {len(tasks)}", "EXPERIMENT")

    # Check for already completed tasks (resume support)
    pending = []
    skipped_count = 0
    for task in tasks:
        task_id = make_task_id(task)
        task['task_id'] = task_id
        if result_saver.run_exists(task_id):
            result_saver.mark_skipped(task, task_id)
            skipped_count += 1
            continue
        pending.append((task, task_id))
    
    logger.info(f"Pending: {len(pending)}, Skipped (already completed): {skipped_count}", "EXPERIMENT")

    if len(pending) == 0:
        logger.info("All tasks already completed! Nothing to do.", "EXPERIMENT")
        return result_saver.get_results()

    progress = ProgressTracker(len(tasks), log_interval=1, logger=logger)
    progress.set_progress_file(output_dir / "progress.txt")

    start_time = time.time()
    completed = 0
    error_count = 0

    def execute(task, task_id):
        try:
            result = run_single_task(task, config, None)
            return task, result, None
        except Exception as e:
            return task, None, str(e)

    print(f"\n[START] Running {len(pending)} tasks with {config.n_jobs} workers", flush=True)
    print(f"[START] Skipped {skipped_count} already completed tasks", flush=True)

    # SEED-MAJOR EXECUTION: all conditions for seed N finish before seed N+1's
    # tasks are submitted, so results are produced in seed order.
    pending_by_seed = {}
    for task, task_id in pending:
        pending_by_seed.setdefault(task['seed'], []).append((task, task_id))

    with ThreadPoolExecutor(max_workers=config.n_jobs) as executor:
        for seed in sorted(pending_by_seed):
            seed_tasks = pending_by_seed[seed]
            print(f"[SEED] seed={seed}: {len(seed_tasks)} conditions", flush=True)
            future_to_task = {executor.submit(execute, task, task_id): (task, task_id)
                              for task, task_id in seed_tasks}

            for future in as_completed(future_to_task):
                task, task_id = future_to_task[future]
                try:
                    task_result, result, error = future.result()
                    completed += 1

                    if error:
                        error_count += 1
                        print(f"[ERROR] Task {task_id} failed: {error[:100]}", flush=True)
                        error_result = {
                            'env_name': task['env_name'], 'noise_std': task['noise_std'],
                            'sparsity_level': task['sparsity_level'], 'algorithm': task['algorithm'],
                            'seed': task['seed'], 'final_return': np.nan,
                            'error': str(error), 'task_id': task_id
                        }
                        progress.update(error_result, "FAILED")
                    else:
                        result['env_name'] = task['env_name']
                        result['noise_std'] = task['noise_std']
                        result['sparsity_level'] = task['sparsity_level']
                        result['algorithm'] = task['algorithm']
                        result['seed'] = task['seed']
                        success = result_saver.save_result(result, task_id)
                        progress.update(result, "COMPLETE" if success else "PARTIAL")

                except Exception as e:
                    error_count += 1
                    print(f"[ERROR] Task {task_id} crashed: {str(e)[:100]}", flush=True)
                    error_result = {
                        'env_name': task['env_name'], 'noise_std': task['noise_std'],
                        'sparsity_level': task['sparsity_level'], 'algorithm': task['algorithm'],
                        'seed': task['seed'], 'final_return': np.nan, 'error': str(e), 'task_id': task_id
                    }
                    progress.update(error_result, "FAILED")

    print("\n" + "=" * 70, flush=True)
    print(f"EXPERIMENT COMPLETE: {config.name}", flush=True)
    print(f"Completed: {completed - error_count}", flush=True)
    print(f"Failed: {error_count}", flush=True)
    print(f"Skipped: {skipped_count}", flush=True)
    print("=" * 70, flush=True)

    result_saver.save_complete()
    result_saver.save_validation_report()
    logger.info("EXPERIMENT COMPLETE", "EXPERIMENT")

    return result_saver.get_results()
