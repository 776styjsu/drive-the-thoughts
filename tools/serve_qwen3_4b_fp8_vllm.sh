#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ID="${QWEN3_MODEL_ID:-Qwen/Qwen3-4B-FP8}"
SERVED_MODEL_NAME="${QWEN3_SERVED_MODEL_NAME:-Qwen/Qwen3-4B-FP8}"
HOST="${QWEN3_HOST:-127.0.0.1}"
PORT="${QWEN3_PORT:-8000}"
HF_HUB_SPEC="${QWEN3_HF_HUB_SPEC:-huggingface_hub>=0.30.0}"
VLLM_SPEC="${QWEN3_VLLM_SPEC:-vllm>=0.9.0}"
NINJA_SPEC="${QWEN3_NINJA_SPEC:-ninja>=1.11.1}"
QWEN3_PYTHON="${QWEN3_PYTHON:-3.12}"
QWEN3_ENV_ROOT="${QWEN3_ENV_ROOT:-${REPO_ROOT}/.qwen3-vllm-env}"
QWEN3_VENV_DIR="${QWEN3_VENV_DIR:-${QWEN3_ENV_ROOT}/venv-managed}"

# Keep large package/model/runtime caches out of quota-limited home directories.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${QWEN3_ENV_ROOT}/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${QWEN3_ENV_ROOT}/uv-python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${QWEN3_ENV_ROOT}/xdg-cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${QWEN3_ENV_ROOT}/xdg-config}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${QWEN3_ENV_ROOT}/pip-cache}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_HOME}/assets}"
export TORCH_HOME="${TORCH_HOME:-${QWEN3_ENV_ROOT}/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${QWEN3_ENV_ROOT}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${QWEN3_ENV_ROOT}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${QWEN3_ENV_ROOT}/cuda-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${QWEN3_ENV_ROOT}/vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${QWEN3_ENV_ROOT}/flashinfer}"
export CARGO_HOME="${CARGO_HOME:-${QWEN3_ENV_ROOT}/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-${QWEN3_ENV_ROOT}/rustup}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${QWEN3_ENV_ROOT}/cargo-target}"
export TMPDIR="${TMPDIR:-${QWEN3_ENV_ROOT}/tmp}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
mkdir -p \
  "${UV_CACHE_DIR}" \
  "${UV_PYTHON_INSTALL_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_ASSETS_CACHE}" \
  "${TORCH_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${VLLM_CACHE_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${CARGO_HOME}" \
  "${RUSTUP_HOME}" \
  "${CARGO_TARGET_DIR}" \
  "${TMPDIR}" \
  "${QWEN3_ENV_ROOT}"

usage() {
  cat <<EOF
Usage:
  $0 env
  $0 setup [uv venv args...]
  $0 pull
  $0 serve [vllm serve args...]

Environment:
  QWEN3_MODEL_ID   Hugging Face model id or local model path. Default: ${MODEL_ID}
  QWEN3_SERVED_MODEL_NAME
                   Model name exposed by vLLM. Default: ${SERVED_MODEL_NAME}
  QWEN3_HOST       vLLM listen host. Default: ${HOST}
  QWEN3_PORT       vLLM listen port. Default: ${PORT}
  QWEN3_HF_HUB_SPEC
                   uv dependency spec for pull. Default: ${HF_HUB_SPEC}
  QWEN3_VLLM_SPEC  uv dependency spec for serve. Default: ${VLLM_SPEC}
  QWEN3_NINJA_SPEC uv dependency spec for FlashInfer JIT builds. Default: ${NINJA_SPEC}
  QWEN3_PYTHON     uv-managed Python version for setup. Default: ${QWEN3_PYTHON}
  QWEN3_ENV_ROOT   Repo-local cache/env root. Default: ${QWEN3_ENV_ROOT}
  QWEN3_VENV_DIR   Persistent vLLM virtualenv. Default: ${QWEN3_VENV_DIR}
  QWEN3_ALLOW_TRANSIENT
                   Set to 1 to allow serve without setup via uv's transient env.
  HF_HOME          Hugging Face cache root. Default: ${HF_HOME}
  FLASHINFER_WORKSPACE_BASE
                   FlashInfer JIT cache base. Default: ${FLASHINFER_WORKSPACE_BASE}
  CARGO_HOME       Cargo cache root. Default: ${CARGO_HOME}
  RUSTUP_HOME      Rustup toolchain root. Default: ${RUSTUP_HOME}
  HF_TOKEN         Optional Hugging Face token.

The cot_analysis provider expects:
  QWEN3_BASE_URL=http://localhost:${PORT}/v1
  QWEN3_API_KEY=EMPTY

For a persistent repo-local vLLM env:
  $0 setup
  $0 serve

To export the same cache settings into your shell:
  source <($0 env)
EOF
}

print_env() {
  local name
  for name in \
    QWEN3_ENV_ROOT \
    QWEN3_VENV_DIR \
    QWEN3_PYTHON \
    UV_CACHE_DIR \
    UV_PYTHON_INSTALL_DIR \
    XDG_CACHE_HOME \
    XDG_CONFIG_HOME \
    PIP_CACHE_DIR \
    HF_HOME \
    HF_HUB_CACHE \
    HF_ASSETS_CACHE \
    TORCH_HOME \
    TORCHINDUCTOR_CACHE_DIR \
    TRITON_CACHE_DIR \
    CUDA_CACHE_PATH \
    VLLM_CACHE_ROOT \
    FLASHINFER_WORKSPACE_BASE \
    CARGO_HOME \
    RUSTUP_HOME \
    CARGO_TARGET_DIR \
    TMPDIR \
    PYTHONNOUSERSITE \
    VLLM_NO_USAGE_STATS
  do
    printf 'export %s=%q\n' "${name}" "${!name}"
  done
}

main() {
  local action="${1:-serve}"
  shift || true

  cd "${REPO_ROOT}"

  case "${action}" in
    env)
      print_env
      ;;
    setup)
      if command -v rustup >/dev/null 2>&1; then
        rustup set profile minimal
      fi
      uv venv --managed-python --python "${QWEN3_PYTHON}" "${QWEN3_VENV_DIR}" "$@"
      uv pip install --python "${QWEN3_VENV_DIR}/bin/python" \
        "${HF_HUB_SPEC}" \
        "${VLLM_SPEC}" \
        "${NINJA_SPEC}"
      ;;
    pull)
      if [[ -x "${QWEN3_VENV_DIR}/bin/hf" ]]; then
        "${QWEN3_VENV_DIR}/bin/hf" download "${MODEL_ID}" "$@"
      else
        uv run --no-project --with "${HF_HUB_SPEC}" \
          hf download "${MODEL_ID}" "$@"
      fi
      ;;
    serve)
      if [[ -x "${QWEN3_VENV_DIR}/bin/vllm" ]]; then
        export PATH="${QWEN3_VENV_DIR}/bin:${PATH}"
        exec "${QWEN3_VENV_DIR}/bin/vllm" serve "${MODEL_ID}" \
          --host "${HOST}" \
          --port "${PORT}" \
          --served-model-name "${SERVED_MODEL_NAME}" \
          "$@"
      fi
      echo "No repo-local vLLM env found at ${QWEN3_VENV_DIR}." >&2
      echo "Run '$0 setup' once to install vLLM under ${QWEN3_ENV_ROOT}." >&2
      if [[ "${QWEN3_ALLOW_TRANSIENT:-0}" != "1" ]]; then
        exit 1
      fi
      echo "QWEN3_ALLOW_TRANSIENT=1 is set; using uv transient env." >&2
      exec uv run --no-project --with "${VLLM_SPEC}" \
        vllm serve "${MODEL_ID}" \
          --host "${HOST}" \
          --port "${PORT}" \
          --served-model-name "${SERVED_MODEL_NAME}" \
          "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "Unknown action: ${action}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
