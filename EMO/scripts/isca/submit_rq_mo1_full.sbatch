#!/bin/bash
#SBATCH --job-name=emo_rq_mo1_full
#SBATCH --partition=sq
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --output=/lustre/projects/Research_Project-T132310/experiments/emo/logs/rq_mo1_full_%j.out
#SBATCH --error=/lustre/projects/Research_Project-T132310/experiments/emo/logs/rq_mo1_full_%j.err
#SBATCH --ntasks=1
#SBATCH --exclude=gpu01,gpu02,gpu03,gpu04,gpu05,gpu06,gpu07,gpu08,gpu09,comp045,comp093,comp118
# NO TIME LIMIT

set -euo pipefail
exec 1>/lustre/projects/Research_Project-T132310/experiments/emo/logs/rq_mo1_full_${SLURM_JOB_ID}.out 2>&1

cd /lustre/projects/Research_Project-T132310/experiments/emo
mkdir -p logs results/rq_mo1_full

export MUJOCO_PATH=/lustre/projects/Research_Project-T132310/mujoco/mujoco-3.2.5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PATH/bin
export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

VENV_PYTHON=/lustre/projects/Research_Project-T132310/experiments/venv/bin/python3

mkdir -p shared

# PPO on both MOEAs
"$VENV_PYTHON" rq_mo1_test.py --algo PPO --moea NSGA2 --pop 30 --gen 20 --seq 3 --out results/rq_mo1_full --scale-eval 2
"$VENV_PYTHON" rq_mo1_test.py --algo PPO --moea SMS_EMOA --pop 30 --gen 20 --seq 3 --out results/rq_mo1_full --scale-eval 2

# CMA-ES
"$VENV_PYTHON" rq_mo1_test.py --algo CMA-ES --moea NSGA2 --pop 30 --gen 15 --seq 3 --out results/rq_mo1_full --scale-eval 2
"$VENV_PYTHON" rq_mo1_test.py --algo CMA-ES --moea SMS_EMOA --pop 30 --gen 15 --seq 3 --out results/rq_mo1_full --scale-eval 2

echo "=== FULL DONE: $(date) ==="
