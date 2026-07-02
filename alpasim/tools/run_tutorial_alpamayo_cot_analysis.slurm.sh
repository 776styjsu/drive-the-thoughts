#!/usr/bin/env bash
#SBATCH --time=12:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=1   # number of CPU cores per task
#SBATCH --partition=standard   # partition
#SBATCH -J "cot-analysis"   # job name
#SBATCH --account=your_hpc_account   # allocation name
#SBATCH --output=slurm_output/cot_alpamayo_%A_%a.out
#SBATCH --error=slurm_output/cot_alpamayo_%A_%a.err

set -euo pipefail

# ---------- Config (override via sbatch --export=...) ----------
REPO_ROOT="${REPO_ROOT:-<alpasim-source-root>}"
TUTORIAL_ROOT="${TUTORIAL_ROOT:-tutorial_alpamayo}"
SCENE_START="${SCENE_START:-1}"
SCENE_END="${SCENE_END:-50}"
EVERY_NTH="${EVERY_NTH:-10}"
MODEL="${MODEL:-gemini-3.1-pro-preview}"
DELAY="${DELAY:-1.0}"
CAMERA_ID="${CAMERA_ID:-camera_front_wide_120fov}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
NO_IMAGES="${NO_IMAGES:-0}"

# For array jobs: each array task runs a scene chunk.
# Example: sbatch --array=0-9 --export=ALL,SCENES_PER_TASK=5 ...
SCENES_PER_TASK="${SCENES_PER_TASK:-0}"

mkdir -p "$REPO_ROOT/slurm_output"
cd "$REPO_ROOT"

# source setup_local_env.sh

RUN_SCENE_START="$SCENE_START"
RUN_SCENE_END="$SCENE_END"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "$SCENES_PER_TASK" -gt 0 ]]; then
  RUN_SCENE_START=$((SCENE_START + SLURM_ARRAY_TASK_ID * SCENES_PER_TASK))
  RUN_SCENE_END=$((RUN_SCENE_START + SCENES_PER_TASK - 1))

  if [[ "$RUN_SCENE_START" -gt "$SCENE_END" ]]; then
    echo "No scene assigned to task ${SLURM_ARRAY_TASK_ID}; exiting."
    exit 0
  fi

  if [[ "$RUN_SCENE_END" -gt "$SCENE_END" ]]; then
    RUN_SCENE_END="$SCENE_END"
  fi
fi

cmd=(
  <alpasim-source-root>/.venv/bin/python
  tools/run_tutorial_alpamayo_cot_analysis.py
  --tutorial-root "$TUTORIAL_ROOT"
  --scene-start "$RUN_SCENE_START"
  --scene-end "$RUN_SCENE_END"
  --every-nth "$EVERY_NTH"
  --model "$MODEL"
  --delay "$DELAY"
  --camera-id "$CAMERA_ID"
  --log-level "$LOG_LEVEL"
)

if [[ "$NO_IMAGES" == "1" ]]; then
  cmd+=(--no-images)
fi

# Prefer explicit API_KEY export; fallback to GEMINI_API_KEY if present.
if [[ -n "${API_KEY:-}" ]]; then
  cmd+=(--api-key "$API_KEY")
elif [[ -n "${GEMINI_API_KEY:-}" ]]; then
  cmd+=(--api-key "$GEMINI_API_KEY")
fi

echo "Running command: ${cmd[*]}"
"${cmd[@]}"
