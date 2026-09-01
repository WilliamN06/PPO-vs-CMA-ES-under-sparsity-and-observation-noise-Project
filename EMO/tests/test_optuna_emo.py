# --------------------------------------------------------------
# tests/test_optuna_emo.py
# Smoke test that the Optuna EMO search runs a couple of trials.
# --------------------------------------------------------------
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optuna_emo_search import run_optuna_emo


def test_fast_optuna():
    study, best = run_optuna_emo(study_name="smoke_test", algorithm_name="PPO",
                                 n_trials=3, n_inner_seeds=1,
                                 log_dir="test_optuna_out")
    assert study.best_trial is not None
    print(f"smoke ok, best={best}")


if __name__ == "__main__":
    test_fast_optuna()
