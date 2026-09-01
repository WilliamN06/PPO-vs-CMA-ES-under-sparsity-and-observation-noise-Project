"""CMA-ES Diagnostics - COMPLETELY FIXED
Fixes: evolution path access, covariance change variance, condition number, eigenvector cosine
"""
import numpy as np
from scipy.stats import spearmanr


class CMAESDiagnostics:
    def __init__(self):
        self.best_returns = []
        self.rank_stability = []
        self.population_diversity = []
        self.covariance_stability = []
        self.episode_return_variance = []

        self.selection_pressure = []
        self.condition_number = []
        self.first_eigenvector_cosine = []
        self.covariance_change_variance = []
        self.evolution_path_length = []

        self.cov_frob_history = []
        self.prev_ranking = None
        self.prev_covariance = None
        self.prev_eigvecs = None
        self.generation = 0
        self._covariance_initialized = False

    def update(self, fitnesses, candidate_returns, covariance=None, path_length=None):
        best_idx = np.argmin(fitnesses)
        best_return = candidate_returns[best_idx]
        self.best_returns.append(best_return)
        self.generation += 1

        if len(candidate_returns) > 1:
            self.episode_return_variance.append(float(np.var(candidate_returns)))
        else:
            self.episode_return_variance.append(0.0)

        current_ranking = np.argsort(np.argsort(fitnesses))
        if self.prev_ranking is not None and len(self.prev_ranking) == len(current_ranking):
            corr, _ = spearmanr(self.prev_ranking, current_ranking)
            self.rank_stability.append(float(corr) if not np.isnan(corr) else 0.0)
        else:
            self.rank_stability.append(np.nan)
        self.prev_ranking = current_ranking.copy()

        ret_std = np.std(candidate_returns)
        if ret_std > 1e-8:
            sp = (np.max(candidate_returns) - np.mean(candidate_returns)) / ret_std
            self.selection_pressure.append(float(sp))
        else:
            self.selection_pressure.append(0.0)

        if path_length is not None:
            self.evolution_path_length.append(float(path_length))
        else:
            self.evolution_path_length.append(0.0)

        if covariance is not None:
            if isinstance(covariance, np.ndarray) and covariance.size > 0 and covariance.ndim >= 2:
                if self.prev_covariance is not None:
                    try:
                        if covariance.shape == self.prev_covariance.shape:
                            frob = float(np.linalg.norm(covariance - self.prev_covariance, ord='fro'))
                            self.covariance_stability.append(frob)
                            self.cov_frob_history.append(frob)
                        else:
                            self.covariance_stability.append(0.0)
                            self.cov_frob_history.append(0.0)
                    except Exception:
                        self.covariance_stability.append(0.0)
                        self.cov_frob_history.append(0.0)
                else:
                    self.covariance_stability.append(0.0)
                    self.cov_frob_history.append(0.0)

                if len(self.cov_frob_history) >= 2:
                    window = min(10, len(self.cov_frob_history))
                    var = float(np.var(self.cov_frob_history[-window:]))
                    self.covariance_change_variance.append(var)
                else:
                    self.covariance_change_variance.append(0.0)

                try:
                    jitter = 1e-12 * np.trace(covariance) / covariance.shape[0]
                    reg_cov = covariance + jitter * np.eye(covariance.shape[0])
                    cond = np.linalg.cond(reg_cov)
                    self.condition_number.append(float(cond) if np.isfinite(cond) else float('inf'))
                except:
                    self.condition_number.append(float('inf'))

                try:
                    jitter = 1e-12 * np.trace(covariance) / covariance.shape[0]
                    reg_cov = covariance + jitter * np.eye(covariance.shape[0])
                    eigvals, eigvecs = np.linalg.eigh(reg_cov)
                    first_eigvec = eigvecs[:, -1]
                    if self.prev_eigvecs is not None:
                        if np.dot(first_eigvec, self.prev_eigvecs) < 0:
                            first_eigvec = -first_eigvec
                        cos_sim = abs(np.dot(first_eigvec, self.prev_eigvecs))
                        self.first_eigenvector_cosine.append(float(cos_sim) if not np.isnan(cos_sim) else 0.0)
                    else:
                        self.first_eigenvector_cosine.append(1.0)
                    self.prev_eigvecs = first_eigvec.copy()
                except:
                    self.first_eigenvector_cosine.append(0.0)

                self.prev_covariance = covariance.copy()
                self._covariance_initialized = True
            else:
                self.covariance_stability.append(0.0)
                self.cov_frob_history.append(0.0)
                self.covariance_change_variance.append(0.0)
                self.condition_number.append(float('inf'))
                self.first_eigenvector_cosine.append(0.0)
        else:
            self.covariance_stability.append(0.0)
            self.cov_frob_history.append(0.0)
            self.covariance_change_variance.append(0.0)
            self.condition_number.append(float('inf'))
            self.first_eigenvector_cosine.append(0.0)

    def set_population_diversity(self, diversity):
        self.population_diversity.append(float(diversity))

    def get_diagnostics(self) -> dict:
        return {
            'rank_stability_mean': float(np.nanmean(self.rank_stability)) if self.rank_stability else 0.0,
            'rank_stability_std': float(np.nanstd(self.rank_stability)) if self.rank_stability else 0.0,
            'rank_stability_len': len(self.rank_stability),
            'episode_return_variance_mean': float(np.mean(self.episode_return_variance)) if self.episode_return_variance else 0.0,
            'episode_return_variance_std': float(np.std(self.episode_return_variance)) if self.episode_return_variance else 0.0,
            'episode_return_len': len(self.episode_return_variance),
            'population_diversity_mean': float(np.mean(self.population_diversity)) if self.population_diversity else 0.0,
            'population_diversity_len': len(self.population_diversity),
            'covariance_stability_mean': float(np.mean(self.covariance_stability)) if self.covariance_stability else 0.0,
            'covariance_stability_len': len(self.covariance_stability),
            'selection_pressure_mean': float(np.mean(self.selection_pressure)) if self.selection_pressure else 0.0,
            'selection_pressure_std': float(np.std(self.selection_pressure)) if self.selection_pressure else 0.0,
            'selection_pressure_len': len(self.selection_pressure),
            'evolution_path_length_mean': float(np.mean(self.evolution_path_length)) if self.evolution_path_length else 0.0,
            'evolution_path_length_std': float(np.std(self.evolution_path_length)) if self.evolution_path_length else 0.0,
            'evolution_path_length_len': len(self.evolution_path_length),
            'covariance_change_variance_mean': float(np.mean(self.covariance_change_variance)) if self.covariance_change_variance else 0.0,
            'covariance_change_variance_len': len(self.covariance_change_variance),
            'condition_number_mean': float(np.mean([c for c in self.condition_number if np.isfinite(c)])) if self.condition_number else 0.0,
            'condition_number_len': len(self.condition_number),
            'first_eigenvector_cosine_mean': float(np.mean(self.first_eigenvector_cosine)) if self.first_eigenvector_cosine else 0.0,
            'first_eigenvector_cosine_len': len(self.first_eigenvector_cosine),
            'generations': self.generation,
            'best_return': self.best_returns[-1] if self.best_returns else 0.0,
        }

    def get_learning_curve(self) -> list:
        return self.best_returns
