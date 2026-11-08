#!/usr/bin/env bash
# Submit from the drive-the-thoughts repository root.
#SBATCH --account=your_hpc_account
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_40gb|a100_80gb|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=selforacle_gt

set -euo pipefail
source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
module load apptainer
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE="$REQUESTS_CA_BUNDLE"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
TASK_REPO="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$TASK_REPO"
exec alpasim/.venv/bin/python baselines/selforacle/tools/render_nominal.py \
    --campaign "${NOMINAL_CAMPAIGN:?Set NOMINAL_CAMPAIGN}" \
    --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --scenes-per-task "${NOMINAL_SCENES_PER_TASK:-20}" \
    --limit "${NOMINAL_LIMIT:-900}"
