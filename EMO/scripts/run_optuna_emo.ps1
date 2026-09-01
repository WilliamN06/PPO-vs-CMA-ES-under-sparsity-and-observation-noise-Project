$ErrorActionPreference = "Stop"
$emoDir = "C:\Users\Willi\Documents\Opencode Projects\Research Internship\EMO"
Set-Location $emoDir
Write-Host "=== Running Optuna EMO search ==="
python optuna_emo_search.py --study emo_ppo --algo PPO --trials 20 --jobs 1
Write-Host "`nDone. See config/hyperparams.json & results/optuna"
