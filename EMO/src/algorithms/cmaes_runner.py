"""CMA-ES Runner - COMPLETELY FIXED
Fixes: evolution path access timing, population diversity calculation, covariance access
"""
import numpy as np
import torch
import torch.nn as nn
import cma
import gymnasium as gym
from datetime import datetime
import os
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from src.environments.wrappers import NoisySparseEnv
from src.diagnostics.cmaes_diagnostics import CMAESDiagnostics
from src.utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint


class LinearOrMLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or []
        layers, prev_dim = [], obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.Tanh()]
            prev_dim = h
        layers += [nn.Linear(prev_dim, action_dim), nn.Tanh()]
        self.network = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, obs):
        return self.network(obs)

    def set_parameters(self, params):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = torch.from_numpy(params[offset:offset + numel].reshape(p.shape)).float()
            offset += numel

    def get_parameters(self):
        return np.concatenate([p.detach().numpy().flatten() for p in self.parameters()])


def save_cmaes_checkpoint(es, policy, generation, best_return, checkpoint_dir):
    try:
        pc = es.pc if hasattr(es, 'pc') and es.pc is not None else (
            es.sm.pc if hasattr(es, 'sm') and hasattr(es.sm, 'pc') and es.sm.pc is not None else None)
    except Exception:
        pc = None
    checkpoint_data = {
        'generation': generation,
        'xbest': es.result.xbest,
        'sigma': es.sigma,
        'C': es.C if hasattr(es, 'C') else None,
        'pc': pc,
        'policy_params': policy.get_parameters(),
        'best_return': best_return,
    }
    checkpoint_file = checkpoint_dir / f"gen_{generation:04d}.pkl"
    metadata = {'generation': generation, 'best_return': best_return, 'timestamp': datetime.now().isoformat()}
    save_checkpoint(checkpoint_data, checkpoint_file, metadata)
    return checkpoint_file


def run_cmaes(env_name, noise_std, sparsity_level, obs_stats, seed, config, logger=None, sparsity_thresholds=None) -> dict:
    torch.set_num_threads(1)

    # Full RNG seeding (reproducibility P1): the policy's initial weights come
    # from nn.Linear init (torch global RNG), so it MUST be seeded or x0 (and
    # therefore the whole CMA-ES trajectory) changes every run.
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[CMA-ES START] env={env_name} noise={noise_std} sparsity={sparsity_level} seed={seed}", flush=True)

    exp_name = getattr(config, 'name', 'experiment')
    checkpoint_dir = Path(f"checkpoints/{exp_name}/cmaes_{env_name}_noise{noise_std}_sparsity{sparsity_level}_seed{seed}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if getattr(config, 'cmaes_optuna_params', None):
        p = config.cmaes_optuna_params
        config.cmaes_initial_sigma = p.get('sigma_init', config.cmaes_initial_sigma)
        if 'popsize' in p:
            config.cmaes_population_size = p['popsize']

    popsize = getattr(config, 'cmaes_population_size', getattr(config, 'cmaes_popsize', 20))

    # Separate env seed from algorithm seed (reproducibility fix P1).
    env_seed = seed + 10_000
    start_time = time.time()

    env = gym.make(env_name)
    env.reset(seed=env_seed)
    env = NoisySparseEnv(env, noise_std=noise_std, sparsity_level=sparsity_level,
                         obs_stats=obs_stats, seed=env_seed, thresholds=sparsity_thresholds)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    hidden_dims = getattr(config, 'cmaes_hidden_dims', [128, 128])
    policy = LinearOrMLPPolicy(obs_dim, action_dim, hidden_dims=hidden_dims)

    resume = getattr(config, 'resume', True)
    latest_checkpoint = None if not resume else get_latest_checkpoint(checkpoint_dir, "gen_")
    start_gen = 0
    resume_sigma = None
    resume_pc = None

    if latest_checkpoint:
        data, metadata = load_checkpoint(latest_checkpoint)
        if data:
            start_gen = data.get('generation', 0) + 1
            policy.set_parameters(data['policy_params'])
            resume_sigma = data.get('sigma')
            resume_pc = data.get('pc')
            print(f"[CMA-ES] Resuming from generation {start_gen}", flush=True)

    x0 = policy.get_parameters()
    cma_options = {
        'seed': seed,
        'verbose': -1,
        'maxiter': config.cmaes_generations,
        # NOTE: CMA_diagonal was REMOVED so es.C is the full covariance matrix
        # and all covariance diagnostics work. Full covariance is O(n^3) per
        # generation - with hidden [64, 64] (~5-6k params) this is expensive;
        # if too slow, set config.cmaes_diagonal = True (metrics go unavailable).
        'CMA_diagonal': False,
        'popsize': popsize,
    }
    es = cma.CMAEvolutionStrategy(x0, config.cmaes_initial_sigma, cma_options)

    if start_gen > 0:
        # Best-effort resume: restore step-size and evolution path. The full
        # covariance matrix is not restored (pycma internals); it restarts
        # adapted, which is strictly better than the old soft-resume.
        if resume_sigma:
            es.sigma = float(resume_sigma)
        if resume_pc is not None:
            try:
                es.sm.pc = np.asarray(resume_pc)
            except Exception:
                pass

    diagnostics = CMAESDiagnostics()

    print(f"[CMA-ES TRAINING] env={env_name} seed={seed} started", flush=True)

    def rollout(steps_limit):
        obs, _ = env.reset()
        done, episode_return, steps = False, 0.0, 0
        while not done and steps < steps_limit:
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).float().unsqueeze(0)).numpy().flatten()
            lo, hi = env.action_space.low, env.action_space.high
            action = np.clip(action, -1, 1) * (hi - lo) / 2 + (hi + lo) / 2
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            steps += 1
            done = terminated or truncated
        return episode_return

    best_return = float('-inf')
    for generation in range(start_gen, config.cmaes_generations):
        candidates = es.ask()
        fitnesses, candidate_returns = [], []
        for params in candidates:
            policy.set_parameters(params)
            episode_return = rollout(config.cmaes_episode_length)
            fitnesses.append(-episode_return)
            candidate_returns.append(episode_return)

        es.tell(candidates, fitnesses)

        current_best = max(candidate_returns)
        if current_best > best_return:
            best_return = current_best

        centroid = np.mean(candidates, axis=0)
        diversity = np.mean([np.linalg.norm(x - centroid) for x in candidates])

        path_length = 0.0
        if hasattr(es, 'pc') and es.pc is not None:
            path_length = float(np.linalg.norm(es.pc))
        elif hasattr(es, 'sm') and hasattr(es.sm, 'pc') and es.sm.pc is not None:
            path_length = float(np.linalg.norm(es.sm.pc))

        covariance = es.C if hasattr(es, 'C') else None

        diagnostics.update(fitnesses, candidate_returns, covariance, path_length=path_length)
        diagnostics.set_population_diversity(diversity)

        if generation % 20 == 0:
            save_cmaes_checkpoint(es, policy, generation, best_return, checkpoint_dir)
            print(f"[CMA-ES] gen={generation} best={best_return:.2f}, path_len={path_length:.3f}", flush=True)

    policy.set_parameters(es.result.xbest)

    # NOISY/SPARSE eval (same perturbation as training) - reported as final_return
    eval_returns = [rollout(config.cmaes_episode_length) for _ in range(config.n_eval_episodes)]

    # CLEAN eval (no noise, no sparsity) - true learned quality
    clean_env = gym.make(env_name)
    clean_env.reset(seed=env_seed + 500)
    clean_returns = []
    for _ in range(config.n_eval_episodes):
        obs, _ = clean_env.reset()
        done, episode_return = False, 0.0
        while not done:
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).float().unsqueeze(0)).numpy().flatten()
            lo, hi = clean_env.action_space.low, clean_env.action_space.high
            action = np.clip(action, -1, 1) * (hi - lo) / 2 + (hi + lo) / 2
            obs, reward, terminated, truncated, _ = clean_env.step(action)
            episode_return += reward
            done = terminated or truncated
        clean_returns.append(episode_return)
    clean_env.close()

    env.close()
    elapsed = time.time() - start_time

    result = {
        'final_return': float(np.mean(eval_returns)),
        'final_return_std': float(np.std(eval_returns)),
        'clean_return': float(np.mean(clean_returns)),
        'clean_return_std': float(np.std(clean_returns)),
        'wall_clock_seconds': float(elapsed),
        'learning_curve': diagnostics.get_learning_curve(),
        'n_params': policy.n_params,
        'diagnostics': diagnostics.get_diagnostics(),
        'algorithm': 'CMA-ES', 'env_name': env_name, 'noise_std': noise_std,
        'sparsity_level': sparsity_level, 'seed': seed,
        'n_eval_episodes': config.n_eval_episodes, 'timestamp': datetime.now().isoformat()
    }

    print(f"[CMA-ES DONE] env={env_name} seed={seed} return={result['final_return']:.2f}", flush=True)
    return result
