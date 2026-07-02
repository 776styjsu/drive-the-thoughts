#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/run_top_scenes_alpamayo.sh [options]

Runs alpasim_wizard on the top N scene_ids from a scenes CSV.

Options:
  -c, --csv PATH            Path to scenes CSV (default: data/scenes/sim_scenes.csv)
  -n, --top-n INT           Number of scenes to run (default: 1)
  -s, --sim-steps INT       runtime.simulation_config.n_sim_steps (default: 200)
  -d, --deploy NAME         Hydra deploy profile (default: local_apptainer)
  -l, --log-root PATH       Root output directory (default: $PWD/tutorial_alpamayo_test)
      --sif-cache PATH      SIF cache directory (default: $PWD/sif-cache)
      --driver CONF         Driver config (default: alpamayo1_5)
      --topology CONF       Topology config (default: 1gpu)
      --nre-cache-size INT  defines.nre_cache_size / sensorsim cache size (default: 20)
      --start-index INT     0-based index into CSV scene rows before taking top N (default: 0)
      --resume              Skip scenes that already have _complete markers and enable runtime autoresume
      --continue-on-error   Continue to next scene if one run fails
      --extra OVERRIDE      Extra Hydra override appended verbatim to alpasim_wizard
                            (repeatable). Also seeded from the EXTRA_OVERRIDES env
                            var (space-separated). Used to enable the consistency
                            monitor, e.g.
                            --extra runtime.simulation_config.consistency_monitor.enabled=true
      --dry-run             Print commands without running them
  -h, --help                Show this help

Example:
  tools/run_top_scenes_alpamayo.sh --top-n 5 --sim-steps 300

Resume example:
  tools/run_top_scenes_alpamayo.sh --top-n 100 --sim-steps 200 --resume
EOF
}

CSV_PATH="data/scenes/sim_scenes.csv"
TOP_N=1
SIM_STEPS=200
DEPLOY="local_apptainer"
LOG_ROOT="$PWD/tutorial_alpamayo"
SIF_CACHE="$PWD/sif-cache"
DRIVER="alpamayo1_5"
TOPOLOGY="1gpu"
NRE_CACHE_SIZE=20
START_INDEX=0
RESUME=0
CONTINUE_ON_ERROR=0
DRY_RUN=0

# Extra Hydra overrides appended verbatim to alpasim_wizard. Seeded from the
# EXTRA_OVERRIDES env var (space-separated) for batch/slurm use, then extended
# by any --extra options.
EXTRA=()
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a EXTRA <<< "${EXTRA_OVERRIDES}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--csv)
      CSV_PATH="$2"
      shift 2
      ;;
    -n|--top-n)
      TOP_N="$2"
      shift 2
      ;;
    -s|--sim-steps)
      SIM_STEPS="$2"
      shift 2
      ;;
    -d|--deploy)
      DEPLOY="$2"
      shift 2
      ;;
    -l|--log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --sif-cache)
      SIF_CACHE="$2"
      shift 2
      ;;
    --driver)
      DRIVER="$2"
      shift 2
      ;;
    --topology)
      TOPOLOGY="$2"
      shift 2
      ;;
    --nre-cache-size)
      NRE_CACHE_SIZE="$2"
      shift 2
      ;;
    --start-index)
      START_INDEX="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --extra)
      EXTRA+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

has_complete_rollout() {
  local run_dir="$1"
  local scene_id="$2"

  if [[ ! -d "$run_dir/rollouts/$scene_id" ]]; then
    return 1
  fi

  find "$run_dir/rollouts/$scene_id" -type f -name "_complete" -print -quit | grep -q .
}

if [[ ! -f "$CSV_PATH" ]]; then
  echo "CSV file not found: $CSV_PATH" >&2
  exit 1
fi

if ! is_non_negative_int "$TOP_N" || [[ "$TOP_N" -eq 0 ]]; then
  echo "--top-n must be a positive integer" >&2
  exit 1
fi

if ! is_non_negative_int "$SIM_STEPS" || [[ "$SIM_STEPS" -eq 0 ]]; then
  echo "--sim-steps must be a positive integer" >&2
  exit 1
fi

if ! is_non_negative_int "$NRE_CACHE_SIZE" || [[ "$NRE_CACHE_SIZE" -eq 0 ]]; then
  echo "--nre-cache-size must be a positive integer" >&2
  exit 1
fi

if ! is_non_negative_int "$START_INDEX"; then
  echo "--start-index must be a non-negative integer" >&2
  exit 1
fi

mapfile -t SCENES < <(
  awk -F, 'NR>1 {print $2}' "$CSV_PATH" \
    | sed '/^$/d' \
    | tail -n +$((START_INDEX + 1)) \
    | head -n "$TOP_N"
)

if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "No scene_ids found (check --csv and --start-index)." >&2
  exit 1
fi

echo "Selected ${#SCENES[@]} scene(s) from $CSV_PATH"

for idx in "${!SCENES[@]}"; do
  scene_id="${SCENES[$idx]}"
  run_no=$((START_INDEX + idx + 1))
  progress_no=$((idx + 1))
  run_dir="$LOG_ROOT/scene_${run_no}_${scene_id}"

  if [[ "$RESUME" -eq 1 ]] && has_complete_rollout "$run_dir" "$scene_id"; then
    echo "[$progress_no/${#SCENES[@]}] scene_id=$scene_id"
    echo "Skipping: found completed rollout in $run_dir"
    continue
  fi

  cmd=(
    alpasim_wizard
    "deploy=${DEPLOY}"
    "topology=${TOPOLOGY}"
    "wizard.sif_caches=[$SIF_CACHE]"
    "wizard.log_dir=${run_dir}"
    "driver=${DRIVER}"
    "defines.nre_cache_size=${NRE_CACHE_SIZE}"
    "runtime.simulation_config.n_sim_steps=${SIM_STEPS}"
    "scenes.scenes_csv=[${CSV_PATH}]"
    "scenes.scene_ids=[${scene_id}]"
  )

  if [[ "$RESUME" -eq 1 ]]; then
    cmd+=("runtime.enable_autoresume=true")
  fi

  if [[ ${#EXTRA[@]} -gt 0 ]]; then
    cmd+=("${EXTRA[@]}")
  fi

  echo "[$progress_no/${#SCENES[@]}] scene_id=$scene_id"
  echo "Command: ${cmd[*]}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi

  if ! uv run "${cmd[@]}"; then
    echo "Run failed for scene_id=$scene_id" >&2
    if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
      exit 1
    fi
  fi
done

echo "Done."
