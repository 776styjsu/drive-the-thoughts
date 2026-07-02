#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
#
# Serve a local Qwen judge model with vLLM behind an OpenAI-compatible API.
#
# The first argument selects the model variant and the env-var prefix used for
# overrides:
#   qwen35 (default) -> RedHatAI/Qwen3.5-4B-FP8-dynamic, QWEN35_* overrides
#   qwen3            -> Qwen/Qwen3-4B-FP8,               QWEN3_* overrides

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VARIANT="qwen35"
if [[ "${1:-}" == "qwen3" || "${1:-}" == "qwen35" ]]; then
  VARIANT="$1"
  shift
fi

# PREFIX_VAR expands e.g. QWEN35_MODEL_ID / QWEN3_MODEL_ID indirectly so both
# variants keep their historical override names.
case "${VARIANT}" in
  qwen35)
    PREFIX="QWEN35"
    DEFAULT_MODEL_ID="RedHatAI/Qwen3.5-4B-FP8-dynamic"
    ;;
  qwen3)
    PREFIX="QWEN3"
    DEFAULT_MODEL_ID="Qwen/Qwen3-4B-FP8"
    ;;
esac

prefixed() {
  # prefixed NAME DEFAULT -> value of ${PREFIX}_NAME or DEFAULT
  local var="${PREFIX}_$1"
  printf '%s' "${!var:-$2}"
}

MODEL_ID="$(prefixed MODEL_ID "${DEFAULT_MODEL_ID}")"
SERVED_MODEL_NAME="$(prefixed SERVED_MODEL_NAME "${MODEL_ID}")"
HOST="$(prefixed HOST 127.0.0.1)"
PORT="$(prefixed PORT 8000)"
HF_HUB_SPEC="$(prefixed HF_HUB_SPEC 'huggingface_hub>=0.30.0')"
VLLM_SPEC="$(prefixed VLLM_SPEC 'vllm>=0.9.0')"
NINJA_SPEC="$(prefixed NINJA_SPEC 'ninja>=1.11.1')"
PYTHON_VERSION="$(prefixed PYTHON 3.12)"
# Both variants share one env root by default; vLLM serves either model.
ENV_ROOT="$(prefixed ENV_ROOT "${QWEN3_ENV_ROOT:-${REPO_ROOT}/.qwen3-vllm-env}")"
VENV_DIR="$(prefixed VENV_DIR "${ENV_ROOT}/venv-managed")"
ALLOW_TRANSIENT="$(prefixed ALLOW_TRANSIENT 0)"

# Keep large package/model/runtime caches out of quota-limited home directories.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ENV_ROOT}/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${ENV_ROOT}/uv-python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ENV_ROOT}/xdg-cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${ENV_ROOT}/xdg-config}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ENV_ROOT}/pip-cache}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_HOME}/assets}"
export TORCH_HOME="${TORCH_HOME:-${ENV_ROOT}/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${ENV_ROOT}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ENV_ROOT}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${ENV_ROOT}/cuda-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${ENV_ROOT}/vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${ENV_ROOT}/flashinfer}"
export CARGO_HOME="${CARGO_HOME:-${ENV_ROOT}/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-${ENV_ROOT}/rustup}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${ENV_ROOT}/cargo-target}"
export TMPDIR="${TMPDIR:-${ENV_ROOT}/tmp}"
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
  "${ENV_ROOT}"

usage() {
  cat <<EOF
Usage:
  $0 [qwen35|qwen3] env
  $0 [qwen35|qwen3] setup [uv venv args...]
  $0 [qwen35|qwen3] pull
  $0 [qwen35|qwen3] serve [vllm serve args...]

Selected variant '${VARIANT}' serves ${DEFAULT_MODEL_ID}; overrides use the
${PREFIX}_* environment variables:
  ${PREFIX}_MODEL_ID          Hugging Face model id or local model path. Default: ${MODEL_ID}
  ${PREFIX}_SERVED_MODEL_NAME Model name exposed by vLLM. Default: ${SERVED_MODEL_NAME}
  ${PREFIX}_HOST              vLLM listen host. Default: ${HOST}
  ${PREFIX}_PORT              vLLM listen port. Default: ${PORT}
  ${PREFIX}_HF_HUB_SPEC       uv dependency spec for pull. Default: ${HF_HUB_SPEC}
  ${PREFIX}_VLLM_SPEC         uv dependency spec for serve. Default: ${VLLM_SPEC}
  ${PREFIX}_NINJA_SPEC        uv dependency spec for FlashInfer JIT builds. Default: ${NINJA_SPEC}
  ${PREFIX}_PYTHON            uv-managed Python version for setup. Default: ${PYTHON_VERSION}
  ${PREFIX}_ENV_ROOT          Repo-local cache/env root. Default: ${ENV_ROOT}
  ${PREFIX}_VENV_DIR          Persistent vLLM virtualenv. Default: ${VENV_DIR}
  ${PREFIX}_ALLOW_TRANSIENT   Set to 1 to allow serve without setup via uv's transient env.
  HF_HOME                     Hugging Face cache root. Default: ${HF_HOME}
  HF_TOKEN                    Optional Hugging Face token.

The cot_analysis provider expects:
  ${PREFIX}_BASE_URL=http://localhost:${PORT}/v1
  ${PREFIX}_API_KEY=EMPTY

For a persistent repo-local vLLM env:
  $0 ${VARIANT} setup
  $0 ${VARIANT} serve

To export the same cache settings into your shell:
  source <($0 ${VARIANT} env)
EOF
}

print_env() {
  local name
  for name in \
    ENV_ROOT \
    VENV_DIR \
    PYTHON_VERSION \
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
      uv venv --managed-python --python "${PYTHON_VERSION}" "${VENV_DIR}" "$@"
      uv pip install --python "${VENV_DIR}/bin/python" \
        "${HF_HUB_SPEC}" \
        "${VLLM_SPEC}" \
        "${NINJA_SPEC}"
      ;;
    pull)
      if [[ -x "${VENV_DIR}/bin/hf" ]]; then
        "${VENV_DIR}/bin/hf" download "${MODEL_ID}" "$@"
      else
        uv run --no-project --with "${HF_HUB_SPEC}" \
          hf download "${MODEL_ID}" "$@"
      fi
      ;;
    serve)
      if [[ -x "${VENV_DIR}/bin/vllm" ]]; then
        export PATH="${VENV_DIR}/bin:${PATH}"
        exec "${VENV_DIR}/bin/vllm" serve "${MODEL_ID}" \
          --host "${HOST}" \
          --port "${PORT}" \
          --served-model-name "${SERVED_MODEL_NAME}" \
          "$@"
      fi
      echo "No repo-local vLLM env found at ${VENV_DIR}." >&2
      echo "Run '$0 ${VARIANT} setup' once to install vLLM under ${ENV_ROOT}." >&2
      if [[ "${ALLOW_TRANSIENT}" != "1" ]]; then
        exit 1
      fi
      echo "${PREFIX}_ALLOW_TRANSIENT=1 is set; using uv transient env." >&2
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
