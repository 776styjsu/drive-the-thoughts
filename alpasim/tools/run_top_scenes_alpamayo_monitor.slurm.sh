#!/usr/bin/env bash
#
# Run the top-N Alpamayo scenes with the online CoT/trajectory consistency
# monitor enabled (F-LLM map_graph approach). Mirrors
# tools/run_top_scenes_alpamayo.slurm.sh, adding the monitor Hydra overrides and
# (optionally) a local qwen vLLM judge server.
#
# Submit with:
#   sbatch tools/run_top_scenes_alpamayo_monitor.slurm.sh
#
# Pick the judge backend with JUDGE (qwen | qwen35 | gpt | kimi):
#   sbatch --export=ALL,JUDGE=gpt,TOP_N=20  tools/run_top_scenes_alpamayo_monitor.slurm.sh
#   sbatch --export=ALL,JUDGE=kimi,TOP_N=20 tools/run_top_scenes_alpamayo_monitor.slurm.sh
#   sbatch --export=ALL,JUDGE=qwen,START_VLLM=1,TOP_N=20 \
#     tools/run_top_scenes_alpamayo_monitor.slurm.sh
# (Set MONITOR_PROVIDER directly to override the JUDGE mapping.)
#
# The judge is an OpenAI-compatible endpoint:
#   - gpt  -> OpenAI GPT-5.5     (needs OPENAI_API_KEY)
#   - kimi -> gateway Kimi K2.5   (needs GENAI_GATEWAY_KEY)
#   - qwen / qwen35 -> a local vLLM server (start it separately with
#     `tools/serve_qwen3_4b_fp8_vllm.sh serve`, or point QWEN3_BASE_URL at one;
#     set START_VLLM=1 to launch one on this job — shares the GPU, keep
#     VLLM_GPU_FRACTION low).
#
# Credentials are read from $REPO_ROOT/.env (or the job environment) and
# forwarded into the runtime container so the in-loop judge can authenticate.

#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu
#SBATCH --mem=24G
#SBATCH -J "alpamayo_monitor_run"
#SBATCH --account=your_hpc_account
#SBATCH --output=logs/alpamayo_monitor_run_%A_%a.out
#SBATCH --error=logs/alpamayo_monitor_run_%A_%a.err

set -euo pipefail

# ---------- Config (override via sbatch --export=ALL,NAME=value,...) ----------
REPO_ROOT="${REPO_ROOT:-<alpasim-source-root>}"
CSV_PATH="${CSV_PATH:-data/scenes/sim_scenes_monitor_resample_actionable.csv}"
TOP_N="${TOP_N:-100}"
START_INDEX="${START_INDEX:-0}"
SIM_STEPS="${SIM_STEPS:-100}"
DEPLOY="${DEPLOY:-local_apptainer}"
LOG_ROOT="${LOG_ROOT:-tutorial_alpamayo_monitor_gpt}"
SIF_CACHE="${SIF_CACHE:-$REPO_ROOT/sif-cache}"
DRIVER="${DRIVER:-alpamayo1_5}"
TOPOLOGY="${TOPOLOGY:-1gpu}"
NRE_CACHE_SIZE="${NRE_CACHE_SIZE:-20}"
RESUME="${RESUME:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
SETUP_ENV="${SETUP_ENV:-1}"

# ---------- Consistency monitor config ----------
# Friendly judge switch. JUDGE selects the backend; an explicit MONITOR_PROVIDER
# wins over JUDGE. gpt = OpenAI GPT-5.5, kimi = gateway Kimi K2.5, qwen/qwen35 =
# local vLLM.
JUDGE="${JUDGE:-qwen}"
if [[ -z "${MONITOR_PROVIDER:-}" ]]; then
  case "$JUDGE" in
    qwen)   MONITOR_PROVIDER="qwen3_4b_fp8" ;;
    qwen35) MONITOR_PROVIDER="qwen35_4b_fp8" ;;
    gpt)    MONITOR_PROVIDER="openai" ;;
    kimi)   MONITOR_PROVIDER="gateway" ;;
    *)
      echo "Unknown JUDGE='$JUDGE' (expected: qwen | qwen35 | gpt | kimi)" >&2
      exit 1
      ;;
  esac
fi
MONITOR_MODEL="${MONITOR_MODEL:-}"        # empty -> provider default
MONITOR_BASE_URL="${MONITOR_BASE_URL:-}"  # empty -> provider default (localhost:8000/v1)
MONITOR_PROMPT="${MONITOR_PROMPT:-center_of_lane_v5}"
MONITOR_MAX_SAMPLES="${MONITOR_MAX_SAMPLES:-3}"
MONITOR_ACCEPT_THRESHOLD="${MONITOR_ACCEPT_THRESHOLD:-3}"
MONITOR_EVERY_N_STEPS="${MONITOR_EVERY_N_STEPS:-1}"
MONITOR_SEED="${MONITOR_SEED:-42}"

# ---------- Optional local vLLM judge server ----------
START_VLLM="${START_VLLM:-0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_FRACTION="${VLLM_GPU_FRACTION:-0.25}"
VLLM_READY_TIMEOUT_S="${VLLM_READY_TIMEOUT_S:-600}"

# Space-separated module names. Override to add site-specific modules.
MODULES="${MODULES:-apptainer}"

# For array jobs, each task runs a contiguous chunk from the selected CSV range.
SCENES_PER_TASK="${SCENES_PER_TASK:-0}"

load_cluster_modules() {
  if ! type module >/dev/null 2>&1; then
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
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  mkdir -p "$SLURM_TMPDIR/uv-cache"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SLURM_TMPDIR/uv-cache}"
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_TMPDIR/alpasim-run-top-scenes-monitor-${SLURM_JOB_ID:-manual}}"
fi

if [[ "$SETUP_ENV" == "1" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/setup_local_env.sh"
fi

# ---------- Judge credentials ----------
# Read a single KEY from $REPO_ROOT/.env unless it is already set (existing env
# wins), stripping surrounding quotes. Mirrors cot_analysis .env handling.
load_env_var() {
  local var="$1" file="$REPO_ROOT/.env" line val
  [[ -n "${!var:-}" ]] && return 0      # already set; keep it
  [[ -f "$file" ]] || return 0
  line=$(grep -E "^[[:space:]]*${var}=" "$file" | tail -n 1) || true
  [[ -n "$line" ]] || return 0
  val="${line#*=}"
  val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
  export "${var}=${val}"
  return 0
}

# $REPO_ROOT/.env is the default source for judge credentials. Load the standard
# judge env vars from it (existing environment wins), then forward the ones that
# end up set into the runtime container (where the in-loop judge runs), so the
# backend is authenticated regardless of which JUDGE is selected.
JUDGE_ENV_VARS=(
  OPENAI_API_KEY OPENAI_BASE_URL
  GENAI_GATEWAY_KEY GENAI_GATEWAY_BASE_URL
  QWEN3_API_KEY QWEN3_BASE_URL
  QWEN35_API_KEY QWEN35_BASE_URL
)
JUDGE_ENV_FORWARD=()
for _v in "${JUDGE_ENV_VARS[@]}"; do
  load_env_var "$_v"
  [[ -n "${!_v:-}" ]] && JUDGE_ENV_FORWARD+=("$_v")
done

# The selected provider's key var, used only for the missing-key warning below.
case "$MONITOR_PROVIDER" in
  openai)        JUDGE_KEY_VAR=OPENAI_API_KEY ;;
  gateway)       JUDGE_KEY_VAR=GENAI_GATEWAY_KEY ;;
  qwen3_4b_fp8)  JUDGE_KEY_VAR=QWEN3_API_KEY ;;
  qwen35_4b_fp8) JUDGE_KEY_VAR=QWEN35_API_KEY ;;
  *)             JUDGE_KEY_VAR="" ;;
esac

# Remote judges (gpt/kimi) call an external API; never auto-start a local vLLM,
# and warn early if their key is missing (the monitor would silently no-op).
if [[ "$MONITOR_PROVIDER" == "openai" || "$MONITOR_PROVIDER" == "gateway" ]]; then
  START_VLLM=0
  if [[ -z "${!JUDGE_KEY_VAR:-}" ]]; then
    echo "WARNING: $JUDGE_KEY_VAR not found in environment or $REPO_ROOT/.env; the " \
      "$MONITOR_PROVIDER judge cannot authenticate and the monitor will no-op." >&2
  fi
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

# ---------- Optionally launch a local qwen vLLM judge server ----------
VLLM_PID=""
cleanup() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date)] Stopping vLLM judge server (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$START_VLLM" == "1" ]]; then
  echo "[$(date)] Starting local qwen vLLM judge server on port $VLLM_PORT (gpu_fraction=$VLLM_GPU_FRACTION)"
  QWEN3_PORT="$VLLM_PORT" VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_FRACTION" \
    tools/serve_qwen3_4b_fp8_vllm.sh serve >"logs/vllm_judge_${SLURM_JOB_ID:-manual}.log" 2>&1 &
  VLLM_PID=$!
  : "${MONITOR_BASE_URL:=http://localhost:${VLLM_PORT}/v1}"
  echo "[$(date)] Waiting up to ${VLLM_READY_TIMEOUT_S}s for vLLM (pid $VLLM_PID) ..."
  deadline=$((SECONDS + VLLM_READY_TIMEOUT_S))
  until curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM judge server exited early; see logs/vllm_judge_${SLURM_JOB_ID:-manual}.log" >&2
      exit 1
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "vLLM judge server did not become ready within ${VLLM_READY_TIMEOUT_S}s" >&2
      exit 1
    fi
    sleep 5
  done
  echo "[$(date)] vLLM judge server is ready."
fi

# ---------- Build monitor Hydra overrides ----------
MON="runtime.simulation_config.consistency_monitor"
EXTRA_ARGS=(
  --extra "${MON}.enabled=true"
  --extra "${MON}.provider=${MONITOR_PROVIDER}"
  --extra "${MON}.prompt=${MONITOR_PROMPT}"
  --extra "${MON}.max_samples=${MONITOR_MAX_SAMPLES}"
  --extra "${MON}.accept_threshold=${MONITOR_ACCEPT_THRESHOLD}"
  --extra "${MON}.monitor_every_n_steps=${MONITOR_EVERY_N_STEPS}"
  --extra "${MON}.seed=${MONITOR_SEED}"
)
if [[ -n "$MONITOR_MODEL" ]]; then
  EXTRA_ARGS+=(--extra "${MON}.model=${MONITOR_MODEL}")
fi
if [[ -n "$MONITOR_BASE_URL" ]]; then
  EXTRA_ARGS+=(--extra "${MON}.base_url=${MONITOR_BASE_URL}")
fi

# Forward the judge credentials into the runtime container (pass-through form:
# a bare VAR becomes `--env VAR=$VAR` in apptainer, using this job's value).
if [[ ${#JUDGE_ENV_FORWARD[@]} -gt 0 ]]; then
  forward_csv=$(IFS=,; echo "${JUDGE_ENV_FORWARD[*]}")
  EXTRA_ARGS+=(--extra "services.runtime.environments=[${forward_csv}]")
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
  "${EXTRA_ARGS[@]}"
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

echo "[$(date)] Running Alpamayo scenes WITH consistency monitor"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  CSV_PATH=$CSV_PATH  TOP_N=$TOP_N  START_INDEX=$START_INDEX"
echo "  RUN_TOP_N=$RUN_TOP_N  RUN_START_INDEX=$RUN_START_INDEX  SIM_STEPS=$SIM_STEPS"
echo "  DEPLOY=$DEPLOY  LOG_ROOT=$LOG_ROOT  DRIVER=$DRIVER"
echo "  MONITOR: judge=$JUDGE provider=$MONITOR_PROVIDER prompt=$MONITOR_PROMPT max_samples=$MONITOR_MAX_SAMPLES"
echo "           accept_threshold=$MONITOR_ACCEPT_THRESHOLD every_n_steps=$MONITOR_EVERY_N_STEPS"
echo "           model=${MONITOR_MODEL:-<provider default>} base_url=${MONITOR_BASE_URL:-<provider default>}"
echo "           forward_env=${JUDGE_ENV_FORWARD[*]:-<none>} START_VLLM=$START_VLLM"
echo "Command: ${cmd[*]}"

"${cmd[@]}"

echo "[$(date)] Done."
