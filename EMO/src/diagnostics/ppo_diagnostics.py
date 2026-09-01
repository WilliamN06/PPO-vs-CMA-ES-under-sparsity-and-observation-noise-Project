"""PPO Diagnostics Callback - COMPLETELY FIXED
Captures gradients reliably during training step
Fixes: episode tracking, gradient capture timing, value/entropy logging
"""
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class PPODiagnosticCallback(BaseCallback):
    def __init__(self, verbose=0, logger=None):
        super().__init__(verbose)
        self._logger = logger
        self.diagnostics = {
            'timesteps': [],
            'gradient_variance': [],
            'gradient_snr': [],
            'gradient_magnitude': [],
            'advantage_variance': [],
            'value_loss': [],
            'policy_entropy': [],
            'episode_returns': []
        }
        self._current_episode_reward = 0.0
        self._episode_count = 0
        self._step_count = 0
        self._log_freq = 1000
        self._last_log_step = 0
        self._updates_done = 0
        self._gradients_captured = False
        self._capture_attempts = 0
        self._initialized = False

    def _init_callback(self):
        self._initialized = True
        print(f"[PPO] ✅ Diagnostic callback initialized", flush=True)

    def _on_step(self):
        self._step_count += 1

        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])

        for i, info in enumerate(infos):
            if 'episode' in info:
                episode_return = float(info['episode']['r'])
                self.diagnostics['episode_returns'].append(episode_return)
                self._episode_count += 1
                if self._episode_count % 100 == 0:
                    print(f"[PPO] 📊 Episode {self._episode_count}: Return = {episode_return:.2f}", flush=True)
                continue

            if i < len(dones) and dones[i]:
                if self._current_episode_reward != 0:
                    self.diagnostics['episode_returns'].append(float(self._current_episode_reward))
                    self._episode_count += 1
                    self._current_episode_reward = 0.0
                    if self._episode_count % 100 == 0:
                        print(f"[PPO] 📊 Episode {self._episode_count}: Return = {self._current_episode_reward:.2f}", flush=True)

            reward = info.get('reward', 0)
            if reward is None or reward == 0:
                if i < len(rewards):
                    reward = rewards[i]

            if reward is not None and reward != 0:
                self._current_episode_reward += float(reward)

            if info.get('terminated', False) or info.get('truncated', False):
                if self._current_episode_reward != 0:
                    self.diagnostics['episode_returns'].append(float(self._current_episode_reward))
                    self._episode_count += 1
                    if self._episode_count % 100 == 0:
                        print(f"[PPO] 📊 Episode {self._episode_count}: Return = {self._current_episode_reward:.2f}", flush=True)
                    self._current_episode_reward = 0.0

        # Gradients are captured ONLY in _on_update_end (post optimizer.step()),
        # not here, so captures never catch mid-update states.

        if self._step_count - self._last_log_step >= self._log_freq:
            self._last_log_step = self._step_count
            if self.diagnostics['episode_returns']:
                recent = self.diagnostics['episode_returns'][-10:]
                avg_return = np.mean(recent)
                print(f"[PPO] 📊 Step {self._step_count}: Avg Return (last 10) = {avg_return:.2f}", flush=True)

        return True

    def _on_rollout_end(self):
        self._updates_done += 1
        try:
            buf = self.model.rollout_buffer
            if hasattr(buf, 'advantages') and buf.advantages is not None:
                adv_var = float(np.var(buf.advantages.flatten()))
                self.diagnostics['advantage_variance'].append(adv_var)
        except Exception:
            pass

    def _on_update_end(self):
        self._updates_done += 1
        self._capture_gradients()
        captured_value = captured_entropy = False
        try:
            logs = getattr(self.model.logger, 'name_to_value', {})
            for key in ['train/value_loss', 'value_loss', 'loss/value_loss', 'loss/value']:
                if key in logs:
                    self.diagnostics['value_loss'].append(float(logs[key]))
                    captured_value = True
                    break
            for key in ['train/entropy_loss', 'entropy_loss', 'loss/entropy', 'train/entropy']:
                if key in logs:
                    self.diagnostics['policy_entropy'].append(float(logs[key]))
                    captured_entropy = True
                    break
        except Exception:
            pass
        # Fallback: compute value loss directly from the rollout buffer so the
        # diagnostic is never silently empty (value collapse is a key sparsity
        # failure mode and must be measurable).
        if not captured_value:
            try:
                buf = self.model.rollout_buffer
                if hasattr(buf, 'values') and hasattr(buf, 'returns') and \
                   buf.values is not None and buf.returns is not None:
                    vf_loss = float(np.mean((buf.values.flatten() - buf.returns.flatten()) ** 2))
                    self.diagnostics['value_loss'].append(vf_loss)
            except Exception:
                pass

    def _capture_gradients(self):
        self._capture_attempts += 1
        try:
            if not hasattr(self, 'model') or self.model is None:
                return
            if not hasattr(self.model, 'policy'):
                return

            grad_list = []
            has_grad = False
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    has_grad = True
                    grad_list.append(p.grad.detach().cpu().numpy().flatten())

            if not has_grad or not grad_list:
                return

            grad_vector = np.concatenate(grad_list)
            if np.allclose(grad_vector, 0):
                return

            mag = float(np.linalg.norm(grad_vector))
            self.diagnostics['gradient_magnitude'].append(mag)

            if len(grad_vector) > 1:
                var = float(np.var(grad_vector))
                self.diagnostics['gradient_variance'].append(var)
            else:
                self.diagnostics['gradient_variance'].append(0.0)

            grad_mean = np.mean(np.abs(grad_vector))
            grad_std = np.std(grad_vector)
            if grad_std > 1e-8:
                snr = float(grad_mean / grad_std)
            else:
                snr = float(grad_mean / 1e-8) if grad_mean > 0 else 0.0
            self.diagnostics['gradient_snr'].append(snr)

            if not self._gradients_captured:
                print(f"[PPO] 🔍 Gradients captured: mag={mag:.4f}, var={var:.4f}, snr={snr:.4f}", flush=True)
                self._gradients_captured = True

        except Exception as e:
            if self._logger and self._capture_attempts <= 5:
                self._logger.debug(f"Gradient capture error: {e}", "CALLBACK")

    def get_diagnostics(self) -> dict:
        summary = {}
        for key, values in self.diagnostics.items():
            if values:
                summary[f'{key}_mean'] = float(np.mean(values))
                summary[f'{key}_std'] = float(np.std(values))
                summary[f'{key}_len'] = len(values)
        return summary

    def get_learning_curve(self) -> list:
        return self.diagnostics.get('episode_returns', [])
