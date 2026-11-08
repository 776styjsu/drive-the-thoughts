#!/usr/bin/env bash
# Submit from the repository root; no GPU or training backend is needed.
#SBATCH --account=your_hpc_account
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=selforacle_audit

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec baselines/selforacle/.venv/bin/python baselines/selforacle/tools/audit_training_data.py \
    --campaign "${NOMINAL_CAMPAIGN:?Set NOMINAL_CAMPAIGN}" \
    --reserved-scenes "${NOMINAL_RESERVED_SCENES:?Set NOMINAL_RESERVED_SCENES}" \
    --output "${NOMINAL_AUDIT_OUTPUT:?Set NOMINAL_AUDIT_OUTPUT}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
