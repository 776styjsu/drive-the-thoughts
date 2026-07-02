#!/usr/bin/env bash
#
# Submit with:
#   sbatch tools/run_top_scenes_alpamayo.slurm.sh
#
# Common overrides:
#   sbatch --export=ALL,CSV_PATH=data/scenes/sim_scenes.csv,TOP_N=100,START_INDEX=100 \
#     tools/run_top_scenes_alpamayo.slurm.sh
#
# Optional array chunking:
#   sbatch --array=0-9 --export=ALL,TOP_N=100,SCENES_PER_TASK=10 \
#     tools/run_top_scenes_alpamayo.slurm.sh

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH -J "alpamayo_run"
#SBATCH --account=your_hpc_account
#SBATCH --output=logs/alpamayo_run_%A_%a.out
#SBATCH --error=logs/alpamayo_run_%A_%a.err

set -euo pipefail

# ---------- Config (override via sbatch --export=ALL,NAME=value,...) ----------
REPO_ROOT="${REPO_ROOT:-<alpasim-source-root>}"
CSV_PATH="${CSV_PATH:-data/scenes/sim_scenes_monitor_resample_actionable.csv}"
TOP_N="${TOP_N:-100}"
START_INDEX="${START_INDEX:-0}"
SIM_STEPS="${SIM_STEPS:-100}"
DEPLOY="${DEPLOY:-local_apptainer}"
LOG_ROOT="${LOG_ROOT:-tutorial_alpamayo_no_monitor}"
SIF_CACHE="${SIF_CACHE:-$REPO_ROOT/sif-cache}"
DRIVER="${DRIVER:-alpamayo1_5}"
TOPOLOGY="${TOPOLOGY:-1gpu}"
NRE_CACHE_SIZE="${NRE_CACHE_SIZE:-20}"
RESUME="${RESUME:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
SETUP_ENV="${SETUP_ENV:-1}"

# Space-separated module names. Override to add site-specific modules, for example:
#   MODULES="apptainer cuda/12.4"
MODULES="${MODULES:-apptainer}"

# For array jobs, each task runs a contiguous chunk from the selected CSV range.
# Example: --array=0-9 --export=ALL,TOP_N=100,SCENES_PER_TASK=10
SCENES_PER_TASK="${SCENES_PER_TASK:-0}"

load_cluster_modules() {
  if ! type module >/dev/null 2>&1; then
    # Common Lmod/Environment Modules initialization path on non-interactive jobs.
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
  fi

  if ! type module >/dev/null 2>&1; then
    echo "Warning: environment modules are not available; checking PATH directly." >&2
    return
  fi

  local module_name
  for module_name in $MODULES; do
    if [[ "$module_name" == "apptainer" ]]; then
      module load apptainer >/dev/null 2>&1 \
        || module load apptainer/1.3.4 >/dev/null 2>&1 \
        || {
          echo "Failed to load Apptainer module: apptainer or apptainer/1.3.4" >&2
          exit 1
        }
    else
      module load "$module_name"
    fi
  done
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

load_cluster_modules

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Apptainer is not available after module loading." >&2
  exit 1
fi

if [[ -f "$HOME/.cargo/env" ]]; then
  # setup_local_env.sh needs cargo for utils_rs; many clusters do not load
  # user shell init files for batch jobs.
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

# Hugging Face auth is read by huggingface_hub from HF_TOKEN or a cached
# `huggingface-cli login`. Prefer passing HF_TOKEN via sbatch --export.
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  mkdir -p "$SLURM_TMPDIR/uv-cache"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SLURM_TMPDIR/uv-cache}"
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_TMPDIR/alpasim-run-top-scenes-${SLURM_JOB_ID:-manual}}"
fi

if [[ "$SETUP_ENV" == "1" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/setup_local_env.sh"
fi

if ! is_non_negative_int "$TOP_N" || [[ "$TOP_N" -eq 0 ]]; then
  echo "TOP_N must be a positive integer" >&2
  exit 1
fi

if ! is_non_negative_int "$START_INDEX"; then
  echo "START_INDEX must be a non-negative integer" >&2
  exit 1
fi

if ! is_non_negative_int "$SCENES_PER_TASK"; then
  echo "SCENES_PER_TASK must be a non-negative integer" >&2
  exit 1
fi

if ! is_non_negative_int "$NRE_CACHE_SIZE" || [[ "$NRE_CACHE_SIZE" -eq 0 ]]; then
  echo "NRE_CACHE_SIZE must be a positive integer" >&2
  exit 1
fi

RUN_TOP_N="$TOP_N"
RUN_START_INDEX="$START_INDEX"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "$SCENES_PER_TASK" -gt 0 ]]; then
  RUN_START_INDEX=$((START_INDEX + SLURM_ARRAY_TASK_ID * SCENES_PER_TASK))

  if [[ "$RUN_START_INDEX" -ge $((START_INDEX + TOP_N)) ]]; then
    echo "No scene assigned to task ${SLURM_ARRAY_TASK_ID}; exiting."
    exit 0
  fi

  RUN_TOP_N="$SCENES_PER_TASK"
  remaining=$((START_INDEX + TOP_N - RUN_START_INDEX))
  if [[ "$RUN_TOP_N" -gt "$remaining" ]]; then
    RUN_TOP_N="$remaining"
  fi
fi

cmd=(
  tools/run_top_scenes_alpamayo.sh
  --csv "$CSV_PATH"
  --top-n "$RUN_TOP_N"
  --start-index "$RUN_START_INDEX"
  --sim-steps "$SIM_STEPS"
  --deploy "$DEPLOY"
  --log-root "$LOG_ROOT"
  --sif-cache "$SIF_CACHE"
  --driver "$DRIVER"
  --topology "$TOPOLOGY"
  --nre-cache-size "$NRE_CACHE_SIZE"
)

if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi

if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  cmd+=(--continue-on-error)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "[$(date)] Running Alpamayo scenes"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  CSV_PATH=$CSV_PATH"
echo "  TOP_N=$TOP_N"
echo "  START_INDEX=$START_INDEX"
echo "  RUN_TOP_N=$RUN_TOP_N"
echo "  RUN_START_INDEX=$RUN_START_INDEX"
echo "  SIM_STEPS=$SIM_STEPS"
echo "  DEPLOY=$DEPLOY"
echo "  LOG_ROOT=$LOG_ROOT"
echo "  SIF_CACHE=$SIF_CACHE"
echo "  DRIVER=$DRIVER"
echo "  TOPOLOGY=$TOPOLOGY"
echo "  NRE_CACHE_SIZE=$NRE_CACHE_SIZE"
echo "  SETUP_ENV=$SETUP_ENV"
echo "  UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-<uv default>}"
echo "Command: ${cmd[*]}"

"${cmd[@]}"

echo "[$(date)] Done."
