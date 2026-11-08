#!/usr/bin/env bash
# Score all six frozen full-training checkpoints on the same prepared windows.
#SBATCH --account=your_hpc_account
#SBATCH --partition=gpu-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=selforacle_full_safety150

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
SAFETY_PYTHON="$PWD/runs/selforacle/envs/torch211_cu128/bin/python"
for SAFETY_MODEL in sae vae; do
    for SAFETY_SEED in 0 1 2; do
        SAFETY_TAG="${SAFETY_MODEL}_seed${SAFETY_SEED}"
        "$SAFETY_PYTHON" -m selforacle_baseline.benchmark_safety evaluate \
            --manifest "${SAFETY_MANIFEST:?Set SAFETY_MANIFEST}" \
            --training-manifest "${SAFETY_FULL_ROOT:?Set SAFETY_FULL_ROOT}/manifest.json" \
            --checkpoint "$SAFETY_FULL_ROOT/models/$SAFETY_TAG/model.pt" \
            --calibration "$SAFETY_FULL_ROOT/models/$SAFETY_TAG/calibration.json" \
            --allow-pending-nominal --device cuda --workers "${SLURM_CPUS_PER_TASK:-4}" \
            --output "${SAFETY_OUTPUT:?Set SAFETY_OUTPUT}/$SAFETY_TAG"
    done
done
