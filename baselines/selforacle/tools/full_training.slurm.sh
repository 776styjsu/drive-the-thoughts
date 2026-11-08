#!/usr/bin/env bash
# Six independent tasks: SAE seeds 0/1/2, then VAE seeds 0/1/2.
#SBATCH --account=your_hpc_account
#SBATCH --partition=gpu-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --job-name=selforacle_full

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
FULL_TASK="${SLURM_ARRAY_TASK_ID:?Submit as an array}"
FULL_SEED=$((FULL_TASK % 3))
if (( FULL_TASK < 3 )); then FULL_MODEL=sae; else FULL_MODEL=vae; fi
exec runs/selforacle/envs/torch211_cu128/bin/python -m selforacle_baseline.full_training train \
    --manifest "${FULL_TRAINING_ROOT:?Set FULL_TRAINING_ROOT}/manifest.json" \
    --protocol "$FULL_TRAINING_ROOT/protocol.json" --cache-index "$FULL_TRAINING_ROOT/cache/index.json" \
    --model "$FULL_MODEL" --seed "$FULL_SEED" --allow-pending-nominal \
    --output "$FULL_TRAINING_ROOT/models/${FULL_MODEL}_seed${FULL_SEED}"
