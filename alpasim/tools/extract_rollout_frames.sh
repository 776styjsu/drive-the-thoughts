#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/extract_rollout_frames.sh [options]

Extract camera frames, metadata, CoT, trajectory plots, and additional_info.json
from completed Alpasim rollout.asl files.

Options:
  -r, --run-dir PATH        Run directory containing rollouts/ (default: tutorial_alpamayo15_coverage_boost)
  -o, --output-dir PATH     Output directory (default: <run-dir>/extracted_frames)
      --time TIME_US        Extract only one matching time_query_us or time_now_us
      --include-incomplete  Process rollout.asl files even without sibling _complete marker
      --no-additional-info  Skip writing additional_info.json sidecars
      --resume              Skip scenes whose output dir already exists and is non-empty
      --dry-run             Print extraction commands without running them
  -h, --help                Show this help

Examples:
  tools/extract_rollout_frames.sh

  tools/extract_rollout_frames.sh \
    --run-dir tutorial_alpamayo15_coverage_boost \
    --output-dir tutorial_alpamayo15_coverage_boost/extracted_frames

  tools/extract_rollout_frames.sh --time 3900000
EOF
}

RUN_DIR="tutorial_alpamayo15_coverage_boost"
OUTPUT_DIR=""
TIME_US=""
INCLUDE_INCOMPLETE=0
NO_ADDITIONAL_INFO=0
RESUME=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --time)
      TIME_US="$2"
      shift 2
      ;;
    --include-incomplete)
      INCLUDE_INCOMPLETE=1
      shift
      ;;
    --no-additional-info)
      NO_ADDITIONAL_INFO=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
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
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${RUN_DIR}/extracted_frames"
fi

if [[ ! -d "${RUN_DIR}/rollouts" ]]; then
  echo "Missing rollout directory: ${RUN_DIR}/rollouts" >&2
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

mapfile -t ASL_FILES < <(
  if [[ "${INCLUDE_INCOMPLETE}" -eq 1 ]]; then
    find -L "${RUN_DIR}/rollouts" -path '*/rollout.asl' -type f | sort
  else
    find -L "${RUN_DIR}/rollouts" -path '*/_complete' -type f \
      | sort \
      | while read -r complete_marker; do
          rollout_dir="$(dirname "${complete_marker}")"
          asl="${rollout_dir}/rollout.asl"
          [[ -f "${asl}" ]] && printf '%s\n' "${asl}"
        done
  fi
)

if [[ "${#ASL_FILES[@]}" -eq 0 ]]; then
  echo "No rollout.asl files found under ${RUN_DIR}/rollouts" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Run dir: ${RUN_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Found ${#ASL_FILES[@]} rollout(s)."

for asl in "${ASL_FILES[@]}"; do
  rollout_dir="$(dirname "${asl}")"
  scene_dir="$(dirname "${rollout_dir}")"
  scene_id="$(basename "${scene_dir}")"
  scene_output_dir="${OUTPUT_DIR}/${scene_id}"

  if [[ "${RESUME}" -eq 1 && -d "${scene_output_dir}" ]] && [[ -n "$(ls -A "${scene_output_dir}" 2>/dev/null)" ]]; then
    echo
    echo "Skipping ${scene_id} (already extracted)"
    continue
  fi

  cmd=(
    uv run --extra eval python -m alpasim_utils.extract_frame
    "${asl}"
    --output-dir "${scene_output_dir}"
  )

  if [[ -n "${TIME_US}" ]]; then
    cmd+=(--time "${TIME_US}")
  fi

  if [[ "${NO_ADDITIONAL_INFO}" -eq 1 ]]; then
    cmd+=(--no-additional-info)
  fi

  echo
  echo "Extracting ${scene_id}"
  echo "  ASL: ${asl}"
  echo "  Output: ${scene_output_dir}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    continue
  fi

  mkdir -p "${scene_output_dir}"
  "${cmd[@]}"
done

echo
echo "Done. Extracted frames are under ${OUTPUT_DIR}"
