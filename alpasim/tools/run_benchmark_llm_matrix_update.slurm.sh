#!/usr/bin/env bash
#
# Update runs/llm_matrix from the current benchmark.json without overwriting
# existing results until staged outputs have been validated.
#
# Submit:
#   sbatch tools/run_benchmark_llm_matrix_update.slurm.sh
#
# Common overrides:
#   sbatch --export=ALL,START_QWEN35_SERVER=0,QWEN35_BASE_URL=http://host:8000/v1 \
#     tools/run_benchmark_llm_matrix_update.slurm.sh
#   sbatch --export=ALL,APPLY_TO_LIVE=0 tools/run_benchmark_llm_matrix_update.slurm.sh

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH -J "llm_matrix_update"
#SBATCH --account=your_hpc_account
#SBATCH --output=logs/llm_matrix_update_%A.out
#SBATCH --error=logs/llm_matrix_update_%A.err

set -euo pipefail

# Slurm executes a spool-copied script, so BASH_SOURCE may point under
# /var/spool/slurm/slurmd instead of this repository.
DEFAULT_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "${DEFAULT_REPO_ROOT}/tools/run_benchmark_llm_matrix.py" ]]; then
  echo "Could not infer repository root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}." >&2
  echo "Submit from the repository root or pass REPO_ROOT=/path/to/alpasim." >&2
  exit 1
fi

# ---------- Config (override via sbatch --export=ALL,NAME=value,...) ----------
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
BENCHMARK_JSON="${BENCHMARK_JSON:-benchmark.json}"
BENCHMARK_SOURCE_ROOT="${BENCHMARK_SOURCE_ROOT:-$REPO_ROOT}"
LIVE_DIR="${LIVE_DIR:-runs/llm_matrix}"

RUN_ID="${RUN_ID:-llm_matrix_update_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
STAGE_DIR="${STAGE_DIR:-runs/${RUN_ID}}"
GPT_KIMI_STAGE="${GPT_KIMI_STAGE:-${STAGE_DIR}/gpt_kimi}"
QWEN35_STAGE="${QWEN35_STAGE:-${STAGE_DIR}/qwen35}"

N="${N:-3}"
QWEN35_N="${QWEN35_N:-1}"
SEED="${SEED:-42}"
DELAY="${DELAY:-0.0}"
JOBS="${JOBS:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SETUP_ENV="${SETUP_ENV:-1}"
RUN_GPT_KIMI="${RUN_GPT_KIMI:-1}"
RUN_QWEN35="${RUN_QWEN35:-1}"
APPLY_TO_LIVE="${APPLY_TO_LIVE:-1}"

# qwen35 uses the local OpenAI-compatible vLLM endpoint unless you point
# QWEN35_BASE_URL at an existing server and set START_QWEN35_SERVER=0.
START_QWEN35_SERVER="${START_QWEN35_SERVER:-1}"
QWEN35_HOST="${QWEN35_HOST:-127.0.0.1}"
QWEN35_PORT="${QWEN35_PORT:-8000}"
QWEN35_READY_TIMEOUT_S="${QWEN35_READY_TIMEOUT_S:-900}"
QWEN35_GPU_MEMORY_UTILIZATION="${QWEN35_GPU_MEMORY_UTILIZATION:-0.80}"
QWEN35_SERVER_LOG="${QWEN35_SERVER_LOG:-logs/qwen35_llm_matrix_update_${SLURM_JOB_ID:-manual}.log}"

# Space-separated module names. Leave empty if uv/vLLM can run from PATH.
MODULES="${MODULES:-}"

load_cluster_modules() {
  if [[ -z "$MODULES" ]]; then
    return
  fi
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
    module load "$module_name"
  done
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

wait_for_qwen35() {
  local url="http://${QWEN35_HOST}:${QWEN35_PORT}/v1/models"
  local deadline=$((SECONDS + QWEN35_READY_TIMEOUT_S))
  echo "[$(date)] Waiting for qwen35 endpoint: ${url}"
  until curl -sf "$url" >/dev/null 2>&1; do
    if [[ -n "${QWEN35_PID:-}" ]] && ! kill -0 "$QWEN35_PID" 2>/dev/null; then
      echo "qwen35 vLLM server exited early; see ${QWEN35_SERVER_LOG}" >&2
      exit 1
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "qwen35 endpoint was not ready within ${QWEN35_READY_TIMEOUT_S}s" >&2
      exit 1
    fi
    sleep 5
  done
  echo "[$(date)] qwen35 endpoint is ready."
}

cleanup() {
  if [[ -n "${QWEN35_PID:-}" ]] && kill -0 "$QWEN35_PID" 2>/dev/null; then
    echo "[$(date)] Stopping qwen35 vLLM server (pid ${QWEN35_PID})"
    kill "$QWEN35_PID" 2>/dev/null || true
    wait "$QWEN35_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

load_cluster_modules

if [[ -f "$HOME/.cargo/env" ]]; then
  # setup_local_env.sh and vLLM dependency setup may need cargo in batch jobs.
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  mkdir -p "$SLURM_TMPDIR/uv-cache" "$SLURM_TMPDIR/hf-cache"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SLURM_TMPDIR/uv-cache}"
  export HF_HOME="${HF_HOME:-$SLURM_TMPDIR/hf-cache}"
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SLURM_TMPDIR/alpasim-llm-matrix-update-${SLURM_JOB_ID:-manual}}"
fi
UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.hf-cache}"
export UV_CACHE_DIR HF_HOME

if [[ "$SETUP_ENV" == "1" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/setup_local_env.sh"
fi

for value_name in N QWEN35_N JOBS; do
  if ! is_positive_int "${!value_name}"; then
    echo "${value_name} must be a positive integer" >&2
    exit 1
  fi
done

if [[ ! -f "$BENCHMARK_JSON" ]]; then
  echo "Benchmark JSON not found: ${BENCHMARK_JSON}" >&2
  exit 1
fi
if [[ ! -d "$LIVE_DIR" ]]; then
  echo "Live matrix directory not found: ${LIVE_DIR}" >&2
  exit 1
fi

mkdir -p "$GPT_KIMI_STAGE" "$QWEN35_STAGE"
RELIABLE_BENCHMARK="${STAGE_DIR}/benchmark.all_reliable.json"

echo "[$(date)] Preparing reliable-only benchmark"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  BENCHMARK_JSON=$BENCHMARK_JSON"
echo "  LIVE_DIR=$LIVE_DIR"
echo "  STAGE_DIR=$STAGE_DIR"
echo "  RUN_GPT_KIMI=$RUN_GPT_KIMI"
echo "  RUN_QWEN35=$RUN_QWEN35"
echo "  APPLY_TO_LIVE=$APPLY_TO_LIVE"

RELIABLE_BENCHMARK="$RELIABLE_BENCHMARK" BENCHMARK_JSON="$BENCHMARK_JSON" uv run python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["BENCHMARK_JSON"])
out = Path(os.environ["RELIABLE_BENCHMARK"])
payload = json.loads(src.read_text())
if isinstance(payload, dict):
    items = payload.get("results", payload.get("entries", []))
else:
    items = payload
if not isinstance(items, list):
    raise SystemExit(f"Expected benchmark list or results/entries list: {src}")

def is_reliable(item):
    nested = item.get("cot_reliability")
    if isinstance(nested, dict) and "reliable" in nested:
        return bool(nested["reliable"])
    flat = item.get("cot_reliable")
    if isinstance(flat, bool):
        return flat
    if isinstance(flat, str):
        return flat.strip().lower() not in {"false", "no", "0", "unreliable"}
    return True

selected = []
for item in items:
    if not isinstance(item, dict) or not is_reliable(item):
        continue
    normalized = dict(item)
    normalized["cot_reliable"] = True
    selected.append(normalized)

if not selected:
    raise SystemExit("No reliable benchmark entries selected")

if isinstance(payload, dict):
    output_payload = dict(payload)
    if "results" in output_payload:
        output_payload["results"] = selected
    elif "entries" in output_payload:
        output_payload["entries"] = selected
    else:
        output_payload["results"] = selected
else:
    output_payload = selected

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(output_payload, indent=2) + "\n")
print(f"wrote {len(selected)} reliable entries to {out}")
PY

echo "[$(date)] Seeding staged outputs from ${LIVE_DIR}"
if [[ "$RUN_GPT_KIMI" == "1" ]]; then
  cp -p "$LIVE_DIR"/gpt.*.run_*.json "$GPT_KIMI_STAGE"/
  cp -p "$LIVE_DIR"/kimi.*.run_*.json "$GPT_KIMI_STAGE"/
fi
if [[ "$RUN_QWEN35" == "1" ]]; then
  cp -p "$LIVE_DIR/qwen.llm.run_001.json" \
    "$QWEN35_STAGE/qwen.llm.run_001.json"
  cp -p "$LIVE_DIR/qwen.f_llm_map_graph.run_001.json" \
    "$QWEN35_STAGE/qwen.f_llm_map_graph.run_001.json"
fi

common_args=(
  --benchmark-json "$RELIABLE_BENCHMARK"
  --benchmark-source-root "$BENCHMARK_SOURCE_ROOT"
  --fixed-seed
  --seed "$SEED"
  --delay "$DELAY"
  --jobs "$JOBS"
  --log-level "$LOG_LEVEL"
  --uv-cache-dir "$UV_CACHE_DIR"
  --hf-home "$HF_HOME"
)

if [[ "$RUN_GPT_KIMI" == "1" ]]; then
  echo "[$(date)] Running GPT/Kimi matrix update"
  uv run python tools/run_benchmark_llm_matrix.py \
    "${common_args[@]}" \
    -n "$N" \
    --output-dir "$GPT_KIMI_STAGE" \
    --providers gpt,kimi
fi

if [[ "$RUN_QWEN35" == "1" ]]; then
  export QWEN35_API_KEY="${QWEN35_API_KEY:-EMPTY}"
  export QWEN35_BASE_URL="${QWEN35_BASE_URL:-http://${QWEN35_HOST}:${QWEN35_PORT}/v1}"

  if [[ "$START_QWEN35_SERVER" == "1" ]]; then
    echo "[$(date)] Starting qwen35 vLLM server"
    mkdir -p "$(dirname "$QWEN35_SERVER_LOG")"
    QWEN35_PORT="$QWEN35_PORT" QWEN35_HOST="$QWEN35_HOST" \
      tools/serve_qwen35_4b_fp8_vllm.sh serve \
        --gpu-memory-utilization "$QWEN35_GPU_MEMORY_UTILIZATION" \
        >"$QWEN35_SERVER_LOG" 2>&1 &
    QWEN35_PID=$!
  fi
  wait_for_qwen35

  echo "[$(date)] Running qwen35 deterministic matrix update"
  uv run python tools/run_benchmark_llm_matrix.py \
    "${common_args[@]}" \
    -n "$QWEN35_N" \
    --output-dir "$QWEN35_STAGE" \
    --providers qwen
fi

echo "[$(date)] Validating staged outputs"
RUN_GPT_KIMI="$RUN_GPT_KIMI" RUN_QWEN35="$RUN_QWEN35" \
N="$N" QWEN35_N="$QWEN35_N" \
GPT_KIMI_STAGE="$GPT_KIMI_STAGE" QWEN35_STAGE="$QWEN35_STAGE" \
RELIABLE_BENCHMARK="$RELIABLE_BENCHMARK" uv run python - <<'PY'
import json
import os
from pathlib import Path

def load_items(path):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        return payload.get("results", payload.get("entries", []))
    return payload

expected = len(load_items(os.environ["RELIABLE_BENCHMARK"]))
required = []
if os.environ["RUN_GPT_KIMI"] == "1":
    for provider in ("gpt", "kimi"):
        for variant in ("llm", "f_llm_map_graph"):
            for index in range(1, int(os.environ["N"]) + 1):
                required.append(
                    Path(os.environ["GPT_KIMI_STAGE"])
                    / f"{provider}.{variant}.run_{index:03d}.json"
                )
if os.environ["RUN_QWEN35"] == "1":
    for variant in ("llm", "f_llm_map_graph"):
        for index in range(1, int(os.environ["QWEN35_N"]) + 1):
            required.append(
                Path(os.environ["QWEN35_STAGE"])
                / f"qwen.{variant}.run_{index:03d}.json"
            )

failures = []
for path in required:
    if not path.exists():
        failures.append(f"missing: {path}")
        continue
    payload = json.loads(path.read_text())
    results = payload.get("results", payload if isinstance(payload, list) else [])
    if not isinstance(results, list):
        failures.append(f"{path}: results is not a list")
        continue
    clip_ids = [
        item.get("clip_id")
        for item in results
        if isinstance(item, dict) and item.get("clip_id")
    ]
    errors = [
        item.get("clip_id", "<missing>")
        for item in results
        if isinstance(item, dict)
        and (item.get("error") or item.get("evaluation", {}).get("error"))
    ]
    if len(results) != expected:
        failures.append(f"{path}: {len(results)} results, expected {expected}")
    if len(set(clip_ids)) != expected:
        failures.append(
            f"{path}: {len(set(clip_ids))} unique clip_ids, expected {expected}"
        )
    if errors:
        failures.append(f"{path}: {len(errors)} errored result(s): {errors[:5]}")
    print(f"{path}: {len(results)} results")

if failures:
    print("Validation failed:")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print(f"Validation passed for {len(required)} file(s); expected entries={expected}")
PY

if [[ "$APPLY_TO_LIVE" != "1" ]]; then
  echo "[$(date)] APPLY_TO_LIVE=0; staged outputs left at ${STAGE_DIR}"
  exit 0
fi

echo "[$(date)] Applying staged outputs to ${LIVE_DIR}"
BACKUP_DIR="${LIVE_DIR}/backups/${RUN_ID}"
mkdir -p "$BACKUP_DIR"
cp -p "$LIVE_DIR"/*.run_*.json "$BACKUP_DIR"/
if [[ -f "$LIVE_DIR/benchmark.all_reliable.json" ]]; then
  cp -p "$LIVE_DIR/benchmark.all_reliable.json" "$BACKUP_DIR"/
fi
if [[ -f "$LIVE_DIR/manifest.json" ]]; then
  cp -p "$LIVE_DIR/manifest.json" "$BACKUP_DIR"/
fi

if [[ "$RUN_GPT_KIMI" == "1" ]]; then
  cp -p "$GPT_KIMI_STAGE"/gpt.*.run_*.json "$LIVE_DIR"/
  cp -p "$GPT_KIMI_STAGE"/kimi.*.run_*.json "$LIVE_DIR"/
fi
if [[ "$RUN_QWEN35" == "1" ]]; then
  cp -p "$QWEN35_STAGE/qwen.llm.run_001.json" \
    "$LIVE_DIR/qwen.llm.run_001.json"
  cp -p "$QWEN35_STAGE/qwen.f_llm_map_graph.run_001.json" \
    "$LIVE_DIR/qwen.f_llm_map_graph.run_001.json"
fi
cp -p "$RELIABLE_BENCHMARK" "$LIVE_DIR/benchmark.all_reliable.json"

mkdir -p "$LIVE_DIR/source_manifests" "$LIVE_DIR/logs/${RUN_ID}"
if [[ -f "$GPT_KIMI_STAGE/manifest.json" ]]; then
  cp -p "$GPT_KIMI_STAGE/manifest.json" \
    "$LIVE_DIR/source_manifests/${RUN_ID}.gpt_kimi.json"
fi
if [[ -f "$QWEN35_STAGE/manifest.json" ]]; then
  cp -p "$QWEN35_STAGE/manifest.json" \
    "$LIVE_DIR/source_manifests/${RUN_ID}.qwen35.json"
fi
if [[ -d "$GPT_KIMI_STAGE/logs" ]]; then
  cp -a "$GPT_KIMI_STAGE/logs" "$LIVE_DIR/logs/${RUN_ID}/gpt_kimi"
fi
if [[ -d "$QWEN35_STAGE/logs" ]]; then
  cp -a "$QWEN35_STAGE/logs" "$LIVE_DIR/logs/${RUN_ID}/qwen35"
fi

RUN_ID="$RUN_ID" STAGE_DIR="$STAGE_DIR" LIVE_DIR="$LIVE_DIR" \
BENCHMARK_JSON="$BENCHMARK_JSON" BENCHMARK_SOURCE_ROOT="$BENCHMARK_SOURCE_ROOT" \
BACKUP_DIR="$BACKUP_DIR" RELIABLE_BENCHMARK="$RELIABLE_BENCHMARK" \
N="$N" QWEN35_N="$QWEN35_N" SEED="$SEED" DELAY="$DELAY" JOBS="$JOBS" \
uv run python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

live_dir = Path(os.environ["LIVE_DIR"])
benchmark_path = live_dir / "benchmark.all_reliable.json"
benchmark_payload = json.loads(benchmark_path.read_text())
if isinstance(benchmark_payload, dict):
    benchmark_items = benchmark_payload.get("results", benchmark_payload.get("entries", []))
else:
    benchmark_items = benchmark_payload
selected_items = len(benchmark_items)

runs = []
for path in sorted(live_dir.glob("*.run_*.json")):
    payload = json.loads(path.read_text())
    results = payload.get("results", payload if isinstance(payload, list) else [])
    parts = path.name.removesuffix(".json").split(".")
    runs.append(
        {
            "name": path.name.removesuffix(".json"),
            "output": str(path),
            "result_count": len(results) if isinstance(results, list) else None,
            "provider_label": parts[0] if parts else None,
            "variant": parts[1] if len(parts) > 1 else None,
            "repeat": parts[2] if len(parts) > 2 else None,
        }
    )

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "update_id": os.environ["RUN_ID"],
    "benchmark_json": str(Path(os.environ["BENCHMARK_JSON"]).resolve()),
    "benchmark_source_root": str(Path(os.environ["BENCHMARK_SOURCE_ROOT"]).resolve()),
    "subset_path": str(benchmark_path),
    "staging_dir": os.environ["STAGE_DIR"],
    "backup_dir": os.environ["BACKUP_DIR"],
    "reliable_benchmark_staged": os.environ["RELIABLE_BENCHMARK"],
    "selected_items": selected_items,
    "repeats": {
        "gpt": int(os.environ["N"]),
        "kimi": int(os.environ["N"]),
        "qwen": int(os.environ["QWEN35_N"]),
    },
    "base_seed": int(os.environ["SEED"]),
    "fixed_seed": True,
    "delay": float(os.environ["DELAY"]),
    "jobs": int(os.environ["JOBS"]),
    "runs": runs,
}
(live_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote manifest with {len(runs)} run files to {live_dir / 'manifest.json'}")
PY

echo "[$(date)] Done. Live matrix updated."
echo "  backup: ${BACKUP_DIR}"
echo "  stage:  ${STAGE_DIR}"
