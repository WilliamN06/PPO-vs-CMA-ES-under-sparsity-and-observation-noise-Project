#!/bin/bash
#SBATCH --job-name=emo_rq_mo1
#SBATCH --partition=sq
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --output=/lustre/projects/Research_Project-T132310/experiments/emo/logs/rq_mo1_%j.out
#SBATCH --error=/lustre/projects/Research_Project-T132310/experiments/emo/logs/rq_mo1_%j.err
#SBATCH --ntasks=1
#SBATCH --exclude=gpu01,gpu02,gpu03,gpu04,gpu05,gpu06,gpu07,gpu08,gpu09,comp045,comp093,comp118
# NO TIME LIMIT

set -euo pipefail
exec 1>/lustre/projects/Research_Project-T132310/experiments/emo/logs/rq_mo1_${SLURM_JOB_ID}.out 2>&1

cd /lustre/projects/Research_Project-T132310/experiments/emo
mkdir -p logs results/rq_mo1

export MUJOCO_PATH=/lustre/projects/Research_Project-T132310/mujoco/mujoco-3.2.5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PATH/bin
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VENV_PYTHON=/lustre/projects/Research_Project-T132310/experiments/venv/bin/python3

echo "=== RQ-MO1 PILOT: $(date) on $(hostname) ==="

mkdir -p shared

"$VENV_PYTHON" rq_mo1_test.py --algo PPO --moea NSGA2 --pop 20 --gen 6 --seq 2 --out results/rq_mo1
echo "=== PILOT DONE: $(date) ==="
