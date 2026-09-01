"""PPO Runner - FIXED"""
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np
from datetime import datetime
import os
import time
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from src.diagnostics.ppo_diagnostics import PPODiagnosticCallback
from src.environments.wrappers import NoisySparseEnv
from src.utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint


class CheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_dir, save_freq=100000, max_checkpoints=2, verbose=0):
        super().__init__(verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_count = 0

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            self._save_checkpoint()
        return True

    def _save_checkpoint(self):
        self.checkpoint_count += 1
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.num_timesteps:06d}"
        self.model.save(str(checkpoint_file))
        metadata = {'step': self.num_timesteps, 'timestamp': datetime.now().isoformat()}
        save_checkpoint({'step': self.num_timesteps}, checkpoint_file.with_suffix('.pkl'), metadata)
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.zip"))
        if len(checkpoints) > self.max_checkpoints:
            for old in checkpoints[:-self.max_checkpoints]:
                old.unlink()
                pkl_file = old.with_suffix('.pkl')
                if pkl_file.exists():
                    pkl_file.unlink()


def run_ppo(env_name, noise_std, sparsity_level, obs_stats, seed, config, logger=None, sparsity_thresholds=None) -> dict:
    torch.set_num_threads(1)

    # Full RNG seeding (reproducibility P1): seed torch/numpy explicitly so the
    # policy init and any non-SB3 RNG use are deterministic alongside SB3's own
    # seed=seed handling.
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[PPO START] env={env_name} noise={noise_std} sparsity={sparsity_level} seed={seed}", flush=True)

    exp_name = getattr(config, 'name', 'experiment')
    checkpoint_dir = Path(f"checkpoints/{exp_name}/ppo_{env_name}_noise{noise_std}_sparsity{sparsity_level}_seed{seed}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume is disabled for hyperparameter-search trials (optuna) so a stale
    # checkpoint from a previous identical-seed run can never contaminate a
    # fresh trial. Main runs keep resume enabled for crash recovery.
    resume = getattr(config, 'resume', True)
    latest_checkpoint = None if not resume else get_latest_checkpoint(checkpoint_dir, "checkpoint")
    resume_from = None
    start_step = 0

    if latest_checkpoint:
        data, metadata = load_checkpoint(latest_checkpoint)
        if data:
            start_step = data.get('step', 0)
            resume_from = str(latest_checkpoint).replace('.pkl', '.zip')
            if not Path(resume_from).exists():
                resume_from = None
                start_step = 0

    if getattr(config, 'optuna_params', None):
        p = config.optuna_params
        config.ppo_learning_rate = p.get('learning_rate', config.ppo_learning_rate)
        config.ppo_n_steps = p.get('n_steps', config.ppo_n_steps)
        config.ppo_batch_size = p.get('batch_size', config.ppo_batch_size)
        config.ppo_n_epochs = p.get('n_epochs', config.ppo_n_epochs)
        config.ppo_gamma = p.get('gamma', config.ppo_gamma)
        config.ppo_gae_lambda = p.get('gae_lambda', config.ppo_gae_lambda)
        config.ppo_clip_range = p.get('clip_range', config.ppo_clip_range)
        config.ppo_ent_coef = p.get('ent_coef', config.ppo_ent_coef)
        config.ppo_vf_coef = p.get('vf_coef', config.ppo_vf_coef)

    # Separate env seed from algorithm seed so env and algo stochasticity are
    # distinguishable (reproducibility fix P1).
    env_seed = seed + 10_000
    start_time = time.time()

    def make_env():
        env = gym.make(env_name)
        env.reset(seed=env_seed)
        return NoisySparseEnv(env, noise_std=noise_std, sparsity_level=sparsity_level,
                              obs_stats=obs_stats, seed=env_seed, thresholds=sparsity_thresholds)

    def make_clean_env():
        env = gym.make(env_name)
        env.reset(seed=env_seed + 500)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env.seed(seed)

    if resume_from and Path(resume_from).exists():
        model = PPO.load(resume_from, env=vec_env)
    else:
        hidden = list(getattr(config, 'ppo_hidden_dims', [64, 64]))
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=config.ppo_learning_rate, n_steps=config.ppo_n_steps,
            batch_size=config.ppo_batch_size, n_epochs=config.ppo_n_epochs,
            gamma=config.ppo_gamma, gae_lambda=config.ppo_gae_lambda,
            clip_range=config.ppo_clip_range, ent_coef=config.ppo_ent_coef,
            vf_coef=config.ppo_vf_coef, max_grad_norm=config.ppo_max_grad_norm,
            policy_kwargs={'net_arch': dict(pi=hidden, vf=hidden), 'activation_fn': nn.Tanh},
            verbose=0, seed=seed, device='cpu'
        )

    diag_callback = PPODiagnosticCallback(verbose=0, logger=logger)
    checkpoint_callback = CheckpointCallback(checkpoint_dir, save_freq=100000)

    print(f"[PPO TRAINING] env={env_name} seed={seed} started", flush=True)

    remaining_steps = config.ppo_total_timesteps - start_step
    if remaining_steps > 0:
        model.learn(
            total_timesteps=remaining_steps,
            callback=[diag_callback, checkpoint_callback],
            reset_num_timesteps=False
        )

    # NOISY/SPARSE eval (same perturbation as training) - reported as final_return
    eval_env = make_env()
    eval_returns = []
    for i in range(config.n_eval_episodes):
        # Distinct reset seed per episode for independent initial states.
        obs, _ = eval_env.reset(seed=10_000 + seed * 1_000 + i)
        done, episode_return = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_return += reward
            done = terminated or truncated
        eval_returns.append(episode_return)
    eval_env.close()

    # CLEAN eval (no noise, no sparsity) - true learned quality
    clean_env = make_clean_env()
    clean_returns = []
    for i in range(config.n_eval_episodes):
        obs, _ = clean_env.reset(seed=20_000 + seed * 1_000 + i)
        done, episode_return = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = clean_env.step(action)
            episode_return += reward
            done = terminated or truncated
        clean_returns.append(episode_return)
    clean_env.close()

    vec_env.close()
    elapsed = time.time() - start_time

    result = {
        'final_return': float(np.mean(eval_returns)),
        'final_return_std': float(np.std(eval_returns)),
        'clean_return': float(np.mean(clean_returns)),
        'clean_return_std': float(np.std(clean_returns)),
        'wall_clock_seconds': float(elapsed),
        'diagnostics': diag_callback.get_diagnostics(),
        'learning_curve': diag_callback.get_learning_curve(),
        'algorithm': 'PPO', 'env_name': env_name, 'noise_std': noise_std,
        'sparsity_level': sparsity_level, 'seed': seed,
        'n_eval_episodes': config.n_eval_episodes, 'timestamp': datetime.now().isoformat()
    }

    print(f"[PPO DONE] env={env_name} seed={seed} return={result['final_return']:.2f}", flush=True)
    return result
