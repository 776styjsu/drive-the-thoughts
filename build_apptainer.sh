#!/bin/bash

# Build an Apptainer SIF image for AlpaSim — no Docker required.
#
# Usage:
#   ./build_apptainer.sh                # uses default tag
#   SIF_FILE=my_image.sif ./build_apptainer.sh  # custom output filename
#
# By default the host's ~/.netrc is bind-mounted (read-only) into the build
# so that uv/git can access private repos.  Override with NETRC_PATH.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

echo "Building Apptainer image in $REPO_ROOT"
set -x

cleanup() {
    if [ -n "${BUILD_CONTEXT_DIR:-}" ] && [ -d "$BUILD_CONTEXT_DIR" ]; then
        rm -rf "$BUILD_CONTEXT_DIR"
    fi
}

trap cleanup EXIT

find_apptainer_bin() {
    if command -v apptainer >/dev/null 2>&1; then
        command -v apptainer
        return 0
    fi

    if command -v singularity >/dev/null 2>&1; then
        command -v singularity
        return 0
    fi

    if type module >/dev/null 2>&1; then
        module load apptainer >/dev/null 2>&1 || \
            module load apptainer/1.3.4 >/dev/null 2>&1 || true

        if command -v apptainer >/dev/null 2>&1; then
            command -v apptainer
            return 0
        fi
    fi

    return 1
}

# Build target type: "sif" or "sandbox"
BUILD_FORMAT="${BUILD_FORMAT:-sif}"

# Output image path (override with SIF_FILE env var for backward compatibility)
SIF_FILE="${SIF_FILE:-alpasim_base_latest.sif}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/sif-cache}"

if [ "$BUILD_FORMAT" = "sandbox" ]; then
    case "$SIF_FILE" in
        *.sif)
            SIF_FILE="${SIF_FILE%.sif}.sandbox"
            ;;
    esac
fi

case "$SIF_FILE" in
    /*)
        FINAL_IMAGE_PATH="$SIF_FILE"
        ;;
    *)
        FINAL_IMAGE_PATH="$OUTPUT_DIR/$SIF_FILE"
        ;;
esac

IMAGE_NAME="$(basename -- "$FINAL_IMAGE_PATH")"

# Path to .netrc for private-repo authentication (override with NETRC_PATH)
NETRC_PATH="${NETRC_PATH:-$HOME/.netrc}"

echo "Target image path: $FINAL_IMAGE_PATH"
echo "Build format: $BUILD_FORMAT"

# Build args
BUILD_ARGS=()

APPTAINER_BIN="${APPTAINER_BIN:-}"
if [ -z "$APPTAINER_BIN" ]; then
    if ! APPTAINER_BIN="$(find_apptainer_bin)"; then
        echo "ERROR: Could not find Apptainer in PATH." >&2
        echo "Try one of the following and rerun:" >&2
        echo "  module load apptainer/1.3.4" >&2
        echo "  export APPTAINER_BIN=/path/to/apptainer" >&2
        exit 127
    fi
fi

if [ "$(id -u)" -ne 0 ] && [ "${APPTAINER_USE_FAKEROOT:-1}" = "1" ]; then
    BUILD_ARGS+=(--fakeroot)
fi

# Bind .netrc if it exists so %post can authenticate
if [ -f "$NETRC_PATH" ]; then
    echo "Binding $NETRC_PATH for private-repo access"
    BUILD_ARGS+=(--bind "${NETRC_PATH}:/run/netrc:ro")
else
    echo "WARNING: $NETRC_PATH not found — private repos may fail to resolve."
fi

# Set Apptainer temporary and cache directories to the output dir
# This prevents OOM errors on HPC clusters where /tmp is a memory-backed tmpfs.
export APPTAINER_TMPDIR="$OUTPUT_DIR/tmp"
export APPTAINER_CACHEDIR="$OUTPUT_DIR/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Remove stale output before building
mkdir -p "$(dirname -- "$FINAL_IMAGE_PATH")"
rm -rf "$FINAL_IMAGE_PATH"

# Build from a temporary slim context instead of copying the full repo (which may
# include large data/outputs artifacts) into the image.
BUILD_CONTEXT_DIR="$(mktemp -d -p "$APPTAINER_TMPDIR" -t alpasim-apptainer-build-XXXXXX)"

rsync -a \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'data' \
    --exclude 'outputs' \
    --exclude 'tutorial*' \
    --exclude 'tutorial_archive' \
    --exclude 'sif-cache' \
    --exclude 'logs' \
    --exclude '*.sif' \
    --exclude '*.sqsh' \
    --exclude 'src/tools/run-on-ord/runs' \
    "$REPO_ROOT/" "$BUILD_CONTEXT_DIR/"

cd "$BUILD_CONTEXT_DIR"
if [ "$BUILD_FORMAT" = "sandbox" ]; then
    "$APPTAINER_BIN" build --sandbox "${BUILD_ARGS[@]}" "$IMAGE_NAME" Apptainer.def
else
    "$APPTAINER_BIN" build "${BUILD_ARGS[@]}" "$IMAGE_NAME" Apptainer.def
fi
mv "$IMAGE_NAME" "$FINAL_IMAGE_PATH"

echo "=== Done ==="
echo "Image created: $FINAL_IMAGE_PATH"
echo ""
echo "Example usage:"
echo "  $APPTAINER_BIN exec --nv $FINAL_IMAGE_PATH bash -c 'uv run python -m alpasim_controller.server'"
