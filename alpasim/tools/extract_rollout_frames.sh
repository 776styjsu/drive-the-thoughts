#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/extract_rollout_frames.sh [options]

Extract camera frames, metadata, CoT, trajectory plots, and additional_info.json
once for every scene_* directory in an AlpaSim run.

Options:
  -r, --run-dir PATH        Directory containing scene_* directories
                            (default: tutorial_alpamayo_ood_hazard_2604)
      --time TIME_US        Extract only one matching time_query_us or time_now_us
      --include-incomplete  Use the newest rollout even if it has no _complete marker
      --no-additional-info  Skip writing additional_info.json sidecars
      --resume              Skip scenes whose extracted_frames dir is already non-empty
      --dry-run             Print extraction commands without running them
  -h, --help                Show this help

Examples:
  tools/extract_rollout_frames.sh

  tools/extract_rollout_frames.sh \
    --run-dir tutorial_alpamayo_ood_hazard_2604 \
    --resume

  tools/extract_rollout_frames.sh --time 3900000
EOF
}

RUN_DIR="tutorial_alpamayo_ood_hazard_2604"
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

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Missing run directory: ${RUN_DIR}" >&2
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

mapfile -t SCENE_DIRS < <(
  find -L "${RUN_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'scene_*' \
    | sort -V
)

if [[ "${#SCENE_DIRS[@]}" -eq 0 ]]; then
  echo "No scene_* directories found under ${RUN_DIR}" >&2
  exit 1
fi

echo "Run dir: ${RUN_DIR}"
echo "Found ${#SCENE_DIRS[@]} scene(s)."

extracted_count=0
skipped_count=0
missing_count=0

for scene_dir in "${SCENE_DIRS[@]}"; do
  scene_id="$(basename "${scene_dir}")"
  rollouts_dir="${scene_dir}/rollouts"
  scene_output_dir="${scene_dir}/extracted_frames"

  if [[ "${RESUME}" -eq 1 && -d "${scene_output_dir}" ]] && [[ -n "$(ls -A "${scene_output_dir}" 2>/dev/null)" ]]; then
    echo
    echo "Skipping ${scene_id} (already extracted)"
    ((skipped_count += 1))
    continue
  fi

  if [[ ! -d "${rollouts_dir}" ]]; then
    echo
    echo "Skipping ${scene_id} (missing ${rollouts_dir})" >&2
    ((missing_count += 1))
    continue
  fi

  if [[ "${INCLUDE_INCOMPLETE}" -eq 1 ]]; then
    mapfile -t ASL_FILES < <(
      find -L "${rollouts_dir}" -path '*/rollout.asl' -type f \
        -printf '%T@ %p\n' \
        | sort -nr \
        | cut -d ' ' -f 2-
    )
  else
    mapfile -t ASL_FILES < <(
      find -L "${rollouts_dir}" -path '*/_complete' -type f \
        -printf '%T@ %h\n' \
        | sort -nr \
        | cut -d ' ' -f 2- \
        | while IFS= read -r completed_rollout_dir; do
            asl_path="${completed_rollout_dir}/rollout.asl"
            [[ -f "${asl_path}" ]] && printf '%s\n' "${asl_path}"
          done
    )
  fi

  if [[ "${#ASL_FILES[@]}" -eq 0 ]]; then
    echo
    echo "Skipping ${scene_id} (no eligible rollout.asl found)" >&2
    ((missing_count += 1))
    continue
  fi

  # A scene may contain old or failed attempts. Extract only the newest eligible
  # rollout so that every scene is processed exactly once.
  asl="${ASL_FILES[0]}"

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
    ((extracted_count += 1))
    continue
  fi

  mkdir -p "${scene_output_dir}"
  "${cmd[@]}"
  ((extracted_count += 1))
done

echo
echo "Done. Extracted ${extracted_count} scene(s); skipped ${skipped_count}; no eligible rollout ${missing_count}."
echo "Frames are under each scene's extracted_frames directory."
