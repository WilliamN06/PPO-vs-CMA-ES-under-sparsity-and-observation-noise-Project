"""
Environment Wrapper - STRUCTURAL Goal-Radius Sparsity
Reward is based on |v_x| (forward velocity) threshold.
NO TERMINAL SPARSITY.
FIXED: Always adds 'episode' to info when episode ends (terminated, truncated, or max steps)
"""
import json
from pathlib import Path
import numpy as np
import gymnasium as gym


# Index of the forward x-velocity component (v_x) in the observation vector.
# These envs expose obs = qpos[1:] + qvel (x position excluded), so obs[0] is
# the root z-height, NOT the velocity:
#   HalfCheetah-v4 (17): obs[0]=z, obs[1]=theta, ..., obs[8]=dx
#   Hopper-v4      (11): obs[0]=z, obs[1]=theta, ..., obs[5]=dx
#   Walker2d-v4    (17): obs[0]=z, obs[1]=theta, ..., obs[8]=dx
VX_INDEX = {
    "HalfCheetah-v4": 8,
    "Hopper-v4": 5,
    "Walker2d-v4": 8,
}


class NoisySparseEnv(gym.Wrapper):
    def __init__(self, env, noise_std=0.0, sparsity_level="dense", 
                 obs_stats=None, seed=None, thresholds=None):
        super().__init__(env)
        self.noise_std = noise_std
        self.sparsity_level = sparsity_level
        self.obs_stats = obs_stats or {}
        self._rng = np.random.default_rng(seed)
        self._total_steps = 0
        self._zero_reward_steps = 0
        self._step_count = 0
        self._episode_return = 0.0
        self._max_episode_steps = 1000  # MuJoCo default
        
        self.sparsity_thresholds = {
            "dense": 0.0,
            "medium": 0.5,
            "sparse": 1.0,
            "very_sparse": 2.0,
        }
        if thresholds is not None:
            self.sparsity_thresholds.update(thresholds)
        if sparsity_level not in self.sparsity_thresholds:
            # Fail loudly instead of silently behaving like "dense".
            raise ValueError(
                f"Unknown sparsity_level '{sparsity_level}'. Valid levels: "
                f"{sorted(self.sparsity_thresholds)}"
            )
        self._threshold = self.sparsity_thresholds[sparsity_level]

        env_id = getattr(getattr(self.env, 'spec', None), 'id', None)
        if env_id in VX_INDEX:
            self._vx_index = VX_INDEX[env_id]
        else:
            self._vx_index = 0
            print(f"[WARN] No v_x index for env id '{env_id}', defaulting to 0", flush=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._total_steps = 0
        self._zero_reward_steps = 0
        self._step_count = 0
        self._episode_return = 0.0
        if self.noise_std > 0:
            obs = self._add_noise(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += 1
        self._step_count += 1
        original_reward = reward
        self._episode_return += float(reward)

        # Sparsity decision must use the CLEAN observation, not the noisy one,
        # so observation noise does not corrupt the v_x used to zero rewards.
        clean_obs = obs
        if self.noise_std > 0:
            obs = self._add_noise(obs)

        # STRUCTURAL GOAL-RADIUS SPARSITY
        v_x = clean_obs[self._vx_index] if len(clean_obs) > self._vx_index else 0.0
        
        if self._threshold > 0:
            if abs(v_x) <= self._threshold:
                reward = 0.0
                self._zero_reward_steps += 1

        # FIXED: ALWAYS add episode info when episode ends
        # Check for episode end via terminated, truncated, OR max steps
        episode_ended = terminated or truncated or self._total_steps >= self._max_episode_steps
        
        if episode_ended:
            # Add episode data for callback
            info['episode'] = {
                'r': self._episode_return,
                'l': self._total_steps
            }
            # Reset episode return for next episode
            self._episode_return = 0.0
            
            # If truncated due to max steps, mark as truncated
            if self._total_steps >= self._max_episode_steps:
                truncated = True
        
        # Always add these for debugging
        info['episode_return'] = self._episode_return
        info['original_reward'] = original_reward
        info['sparsity_ratio'] = self._zero_reward_steps / max(self._total_steps, 1)
        info['sparsity_level'] = self.sparsity_level
        info['v_x'] = float(v_x)
        info['v_x_threshold'] = self._threshold
        info['total_steps'] = self._total_steps
        
        return obs, reward, terminated, truncated, info

    def _add_noise(self, obs):
        noise = self._rng.normal(0, self.noise_std, size=obs.shape)
        if self.obs_stats.get('std') is not None:
            noise = noise * self.obs_stats['std']
        return (obs + noise).astype(obs.dtype)


def calibrate_sparsity_thresholds(env_name: str, n_episodes: int = 100, seed: int = 0) -> dict:
    from src.utils.mujoco_patch import apply_mujoco_patch
    apply_mujoco_patch()
    
    print(f"[SPARSITY] Calibrating {env_name}...", flush=True)
    env = gym.make(env_name)
    env.reset(seed=seed)

    vx_index = VX_INDEX.get(env_name, 0)
    vx_values = []
    for _ in range(max(n_episodes, 10)):
        obs, _ = env.reset()
        done = False
        while not done:
            vx_values.append(obs[vx_index] if len(obs) > vx_index else 0.0)
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    
    abs_vx = np.abs(np.array(vx_values))
    thresholds = {
        "dense": 0.0,
        "medium": float(np.percentile(abs_vx, 70)),
        "sparse": float(np.percentile(abs_vx, 90)),
        "very_sparse": float(np.percentile(abs_vx, 99)),
    }
    
    print(f"[SPARSITY] {env_name}: medium={thresholds['medium']:.3f}, sparse={thresholds['sparse']:.3f}, very_sparse={thresholds['very_sparse']:.3f}", flush=True)
    return thresholds


def get_sparsity_thresholds(env_name: str, shared_dir: Path, logger=None, n_episodes: int = 100) -> dict:
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    # v2: recalibrated on the correct forward-velocity index, so the stale
    # (wrong-index) thresholds from earlier runs are not reused.
    thresholds_file = shared_dir / f"{env_name}_sparsity_thresholds_v2.json"
    
    if thresholds_file.exists():
        with open(thresholds_file) as f:
            return json.load(f)
    
    thresholds = calibrate_sparsity_thresholds(env_name, n_episodes)
    with open(thresholds_file, 'w') as f:
        json.dump(thresholds, f, indent=2)
    if logger:
        logger.info(f"Saved sparsity thresholds to {thresholds_file}", "SPARSITY")
    return thresholds


def get_shared_obs_stats(env_name: str, shared_dir: Path, logger=None, n_episodes: int = 100) -> dict:
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    stats_file = shared_dir / f"{env_name}_obs_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            raw = json.load(f)
        return {'std': np.array(raw['std']), 'mean': np.array(raw['mean'])}
    
    from src.utils.mujoco_patch import apply_mujoco_patch
    apply_mujoco_patch()
    env = gym.make(env_name)
    env.reset(seed=0)
    obs_list = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_list.append(obs)
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    obs_array = np.array(obs_list)
    stats = {'std': np.std(obs_array, axis=0), 'mean': np.mean(obs_array, axis=0)}
    with open(stats_file, 'w') as f:
        json.dump({'std': stats['std'].tolist(), 'mean': stats['mean'].tolist()}, f, indent=2)
    return stats
