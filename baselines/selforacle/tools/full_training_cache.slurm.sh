#!/usr/bin/env bash
#SBATCH --account=your_hpc_account
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=selforacle_rgb_cache

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec baselines/selforacle/.venv/bin/python -m selforacle_baseline.full_training cache \
    --manifest "${FULL_TRAINING_ROOT:?Set FULL_TRAINING_ROOT}/manifest.json" \
    --output "$FULL_TRAINING_ROOT/cache" --workers "${SLURM_CPUS_PER_TASK:-16}"
