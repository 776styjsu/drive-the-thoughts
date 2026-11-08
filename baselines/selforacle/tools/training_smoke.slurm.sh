#!/usr/bin/env bash
# Submit from the repository root. One 24 GB MIG device, with four CPU cores.
#SBATCH --account=your_hpc_account
#SBATCH --partition=gpu-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=selforacle_smoke

set -euo pipefail
SMOKE_REPO="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$SMOKE_REPO"
export UV_CACHE_DIR="$SMOKE_REPO/.uv-cache"
export UV_PROJECT_ENVIRONMENT="$SMOKE_REPO/runs/selforacle/envs/torch211_cu128"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
"$UV_BIN" sync --project baselines/selforacle --extra cuda --locked \
    --python "$SMOKE_REPO/baselines/selforacle/.venv/bin/python"
exec "$UV_PROJECT_ENVIRONMENT/bin/python" -m selforacle_baseline.smoke run \
    --manifest "${SMOKE_MANIFEST:?Set SMOKE_MANIFEST}" \
    --output "${SMOKE_OUTPUT:?Set SMOKE_OUTPUT}" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" --seed 0 --batch-size 32
