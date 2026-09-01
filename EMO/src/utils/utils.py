import os
import sys
import json
import pickle
import time
try:
    import resource
except ImportError:  # Windows / non-POSIX fallback
    resource = None
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd


class DiagnosticLogger:
    LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

    def __init__(self, log_dir: Path, name: str = "experiment", level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.level = self.LEVELS.get(level, 1)
        self.log_file = self.log_dir / f"{name}.log"
        self.error_file = self.log_dir / f"{name}_errors.log"
        for f in [self.log_file, self.error_file]:
            if not f.exists():
                f.touch()

    def log(self, message: str, level: str = "INFO", category: str = "GENERAL"):
        if self.LEVELS.get(level, 1) < self.level:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] [{level}] [{category}] {message}\n"
        with open(self.log_file, 'a') as f:
            f.write(entry)
            f.flush()
            os.fsync(f.fileno())
        if level in ["ERROR", "CRITICAL"]:
            with open(self.error_file, 'a') as f:
                f.write(entry)
                f.flush()
                os.fsync(f.fileno())

    def debug(self, msg, category="GENERAL"): self.log(msg, "DEBUG", category)
    def info(self, msg, category="GENERAL"): self.log(msg, "INFO", category)
    def warning(self, msg, category="GENERAL"): self.log(msg, "WARNING", category)
    def error(self, msg, category="GENERAL"): self.log(msg, "ERROR", category)
    def critical(self, msg, category="GENERAL"): self.log(msg, "CRITICAL", category)


class ResourceMonitor:
    @staticmethod
    def snapshot() -> Dict:
        if resource is None:
            return {'max_rss_mb': 0.0, 'user_time_s': 0.0, 'system_time_s': 0.0}
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            'max_rss_mb': usage.ru_maxrss / 1024,
            'user_time_s': usage.ru_utime,
            'system_time_s': usage.ru_stime,
        }

    @staticmethod
    def log_snapshot(logger, label: str = ""):
        snap = ResourceMonitor.snapshot()
        if logger:
            logger.info(f"{label} rss={snap['max_rss_mb']:.0f}MB "
                        f"user={snap['user_time_s']:.1f}s sys={snap['system_time_s']:.1f}s", "RESOURCE")
        return snap


def retry_with_backoff(fn, max_attempts: int = 3, base_delay: float = 2.0, logger=None, label: str = ""):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if logger:
                logger.warning(f"{label} attempt {attempt}/{max_attempts} failed: {e}", "RETRY")
            if attempt < max_attempts:
                time.sleep(base_delay ** attempt)
    raise last_err


class ProgressTracker:
    STATUS = ["PENDING", "RUNNING", "COMPLETE", "PARTIAL", "FAILED", "SKIPPED"]

    def __init__(self, total_runs: int, log_interval: int = 10, logger=None):
        self.total_runs = total_runs
        self.completed_runs = 0
        self.log_interval = log_interval
        self.start_time = time.time()
        self.results = []
        self.status_counts = {s: 0 for s in self.STATUS}
        self.logger = logger
        self.progress_file = None

    def set_progress_file(self, filepath: Path):
        self.progress_file = Path(filepath)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.progress_file.exists():
            with open(self.progress_file, 'w') as f:
                f.write("task_id,algorithm,env_name,noise,sparsity,seed,return,status,host,timestamp,file_size,has_learning_curve\n")
                f.flush()
                os.fsync(f.fileno())

    def update(self, result: Dict, status: str = "COMPLETE", file_size: int = 0, has_curve: bool = False):
        self.completed_runs += 1
        self.results.append(result)
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        if self.progress_file:
            row = [
                result.get('task_id', 'unknown'),
                result.get('algorithm', 'unknown'),
                result.get('env_name', 'unknown'),
                result.get('noise_std', 0),
                result.get('sparsity_level', 'dense'),
                result.get('seed', 0),
                result.get('final_return', np.nan),
                status,
                result.get('host', 'unknown'),
                datetime.now().isoformat(),
                file_size,
                has_curve,
            ]
            with open(self.progress_file, 'a') as f:
                f.write(",".join(str(x) for x in row) + "\n")
                f.flush()
                os.fsync(f.fileno())
        if self.completed_runs % self.log_interval == 0:
            self.display_progress()

    def display_progress(self):
        elapsed = time.time() - self.start_time
        if self.completed_runs > 0 and elapsed > 0:
            rate = self.completed_runs / elapsed
            remaining = (self.total_runs - self.completed_runs) / rate if rate > 0 else 0
            msg = (f"Progress: {self.completed_runs}/{self.total_runs} | "
                   f"COMPLETE={self.status_counts['COMPLETE']} "
                   f"PARTIAL={self.status_counts['PARTIAL']} "
                   f"FAILED={self.status_counts['FAILED']} "
                   f"SKIPPED={self.status_counts['SKIPPED']} | "
                   f"ETA: {remaining/3600:.1f}h")
            if self.logger: self.logger.info(msg, "PROGRESS")
            print(msg)

    def get_summary(self) -> Dict:
        """FIXED: Returns summary with all keys, including 'skipped'"""
        return {
            'total': self.total_runs,
            'complete': self.status_counts.get('COMPLETE', 0),
            'partial': self.status_counts.get('PARTIAL', 0),
            'failed': self.status_counts.get('FAILED', 0),
            'skipped': self.status_counts.get('SKIPPED', 0),
            'mean_return': np.nan,  # Will be computed if results exist
            'std_return': np.nan,
        }


class AtomicSaver:
    @staticmethod
    def save(data: Any, filepath: Path, verify: bool = True) -> int:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_file = filepath.with_suffix('.tmp')
        backup_file = filepath.with_suffix('.bak')
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            if temp_file.exists(): temp_file.unlink()
            raise IOError(f"Failed to write: {e}")
        if verify:
            try:
                with open(temp_file, 'rb') as f:
                    verified = pickle.load(f)
                if isinstance(data, dict) and isinstance(verified, dict):
                    if 'final_return' in data and 'final_return' in verified:
                        if abs(data['final_return'] - verified['final_return']) > 1e-6:
                            raise ValueError("Data verification failed")
            except Exception as e:
                if temp_file.exists(): temp_file.unlink()
                raise IOError(f"Verification failed: {e}")
        if filepath.exists():
            filepath.rename(backup_file)
        temp_file.rename(filepath)
        if backup_file.exists():
            backup_file.unlink()
        return filepath.stat().st_size

    @staticmethod
    def load(filepath: Path) -> Any:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception:
            backup_file = filepath.with_suffix('.bak')
            if backup_file.exists():
                with open(backup_file, 'rb') as f:
                    return pickle.load(f)
            raise

    @staticmethod
    def save_json(data: Dict, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())


class Validator:
    REQUIRED_FIELDS = ['algorithm', 'env_name', 'seed', 'timestamp']

    @staticmethod
    def validate_result(result: Dict) -> Tuple[List[str], List[str]]:
        errors, warnings = [], []
        if not isinstance(result, dict):
            errors.append("Result is not a dictionary")
            return errors, warnings
        for field in Validator.REQUIRED_FIELDS:
            if field not in result:
                errors.append(f"Missing required field: {field}")
        if 'final_return' in result:
            if not isinstance(result['final_return'], (int, float)):
                errors.append(f"final_return not numeric: {type(result['final_return'])}")
            elif np.isnan(result['final_return']) or np.isinf(result['final_return']):
                warnings.append("final_return is NaN or Inf")
        if 'learning_curve' in result:
            if not isinstance(result['learning_curve'], list):
                warnings.append("learning_curve not a list")
            elif len(result['learning_curve']) == 0:
                warnings.append("learning_curve is empty")
        if 'diagnostics' in result and isinstance(result['diagnostics'], dict):
            algo = result.get('algorithm', '')
            if algo == 'PPO':
                required = ['gradient_variance_mean', 'gradient_snr_mean', 'gradient_magnitude_mean']
                for d in required:
                    if d not in result['diagnostics']:
                        warnings.append(f"Missing PPO diagnostic: {d}")
            elif algo == 'CMA-ES':
                required = ['rank_stability_mean', 'episode_return_variance_mean']
                for d in required:
                    if d not in result['diagnostics']:
                        warnings.append(f"Missing CMA-ES diagnostic: {d}")
                # Learning curve should cover ~all generations (P4 fix)
                curve = result.get('learning_curve', [])
                gens = result['diagnostics'].get('generations', len(curve))
                if isinstance(curve, list) and gens > 0 and len(curve) < gens * 0.9:
                    warnings.append(
                        f"CMA-ES learning curve too short: {len(curve)} < 0.9*{gens}")
        return errors, warnings

    @staticmethod
    def validate_file(filepath: Path) -> Tuple[bool, str]:
        filepath = Path(filepath)
        if not filepath.exists():
            return False, "File does not exist"
        size = filepath.stat().st_size
        try:
            data = AtomicSaver.load(filepath)
            errors, _ = Validator.validate_result(data)
            if errors:
                return False, f"Validation errors: {errors}"
            curve_len = len(data.get('learning_curve', []))
            return True, f"Valid ({size} bytes, curve_len={curve_len})"
        except Exception as e:
            return False, f"Cannot load: {e}"


def make_task_id(task: Dict) -> str:
    task_id = (f"{task['algorithm']}_{task['env_name']}_"
               f"noise{task['noise_std']}_sparsity{task['sparsity_level']}_seed{task['seed']}")
    # Include the training budget so runs trained with different total
    # timesteps / generations are treated as distinct tasks.
    if task.get('ppo_total_timesteps'):
        task_id += f"_t{task['ppo_total_timesteps']}"
    if task.get('cmaes_generations'):
        task_id += f"_g{task['cmaes_generations']}"
    return task_id


class ResultSaver:
    def __init__(self, output_dir: Path, experiment_name: str, host: str = "unknown", logger=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.host = host
        self.logger = logger
        self.results = []
        self.errors = []
        self.progress_file = self.output_dir / "progress.txt"
        self.summary_file = self.output_dir / "summary.json"
        self.validation_report_file = self.output_dir / "validation_report.json"
        if not self.progress_file.exists():
            with open(self.progress_file, 'w') as f:
                f.write("task_id,algorithm,env_name,noise,sparsity,seed,return,status,host,timestamp,file_size,has_learning_curve\n")
                f.flush()
                os.fsync(f.fileno())

    def run_exists(self, task_id: str) -> bool:
        result_file = self.output_dir / f"run_{task_id}" / "result.pkl"
        if not result_file.exists():
            return False
        valid, _ = Validator.validate_file(result_file)
        return valid

    def save_result(self, result: Dict, task_id: str, save_individual: bool = True) -> bool:
        result['host'] = self.host
        result['task_id'] = task_id
        errors, warnings = Validator.validate_result(result)
        if errors:
            self.errors.append({'task_id': task_id, 'errors': errors})
            if self.logger:
                self.logger.error(f"Validation errors for {task_id}: {errors}", "VALIDATION")
            return False
        for warning in warnings:
            if self.logger:
                self.logger.warning(f"{task_id}: {warning}", "VALIDATION")
        self.results.append(result)
        file_size, has_curve = 0, False
        if save_individual:
            run_dir = self.output_dir / f"run_{task_id}"
            run_dir.mkdir(exist_ok=True)
            result_file = run_dir / "result.pkl"
            try:
                file_size = AtomicSaver.save(result, result_file)
                has_curve = len(result.get('learning_curve', [])) > 0
                if self.logger:
                    self.logger.info(f"Saved {task_id} ({file_size} bytes)", "STORAGE")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to save {task_id}: {e}", "STORAGE")
                return False
            metadata = {
                'task_id': task_id,
                'timestamp': datetime.now().isoformat(),
                'algorithm': result.get('algorithm', 'unknown'),
                'env_name': result.get('env_name', 'unknown'),
                'noise_std': result.get('noise_std', 0),
                'sparsity_level': result.get('sparsity_level', 'dense'),
                'seed': result.get('seed', 0),
                'host': self.host,
                'file_size': file_size,
                'has_learning_curve': has_curve,
                'valid': len(errors) == 0
            }
            AtomicSaver.save_json(metadata, run_dir / "metadata.json")
        self._append_progress(result, task_id, status="COMPLETE", file_size=file_size, has_curve=has_curve)
        self._save_summary()
        return True

    def mark_skipped(self, task: Dict, task_id: str):
        self._append_progress(task, task_id, status="SKIPPED", file_size=0, has_curve=True)

    def _append_progress(self, result, task_id, status, file_size, has_curve):
        try:
            row = [
                task_id, result.get('algorithm', 'unknown'), result.get('env_name', 'unknown'),
                result.get('noise_std', 0), result.get('sparsity_level', 'dense'), result.get('seed', 0),
                result.get('final_return', np.nan), status, self.host, datetime.now().isoformat(),
                file_size, has_curve
            ]
            with open(self.progress_file, 'a') as f:
                f.write(",".join(str(x) for x in row) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update progress: {e}", "STORAGE")

    def _save_summary(self):
        if not self.results:
            return
        df = pd.DataFrame(self.results)
        summary = {
            'total_runs': len(self.results),
            'host': self.host,
            'timestamp': datetime.now().isoformat(),
            'algorithms': df['algorithm'].unique().tolist() if 'algorithm' in df else [],
            'environments': df['env_name'].unique().tolist() if 'env_name' in df else [],
            'errors': self.errors,
            'statistics': {}
        }
        if 'final_return' in df:
            summary['statistics'] = {
                'mean_return': float(df['final_return'].mean()),
                'std_return': float(df['final_return'].std()),
                'min_return': float(df['final_return'].min()),
                'max_return': float(df['final_return'].max()),
                'median_return': float(df['final_return'].median()),
                'count': int(df['final_return'].count())
            }
        AtomicSaver.save_json(summary, self.summary_file)

    def save_validation_report(self):
        report = {'timestamp': datetime.now().isoformat(), 'host': self.host,
                   'total_runs': len(self.results), 'errors': self.errors, 'run_validations': {}}
        for run_dir in self.output_dir.glob("run_*"):
            result_file = run_dir / "result.pkl"
            if result_file.exists():
                valid, msg = Validator.validate_file(result_file)
                report['run_validations'][run_dir.name] = {
                    'valid': valid, 'message': msg, 'file_size': result_file.stat().st_size
                }
        AtomicSaver.save_json(report, self.validation_report_file)

    def save_complete(self):
        if self.results:
            try:
                df = pd.DataFrame(self.results)
                slim = df.drop(columns=['learning_curve', 'diagnostics'], errors='ignore')
                slim.to_csv(self.output_dir / "results_complete.csv", index=False)
                if self.logger:
                    self.logger.info(f"Complete results saved: {len(self.results)} runs", "STORAGE")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to save complete: {e}", "STORAGE")

    def get_results(self):
        return self.results
