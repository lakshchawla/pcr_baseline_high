#!/usr/bin/env bash
# Runs the full PCR pipeline, Stage 0 through Stage 2, in sequence:
#
#   Stage 0 (BPA segmentation pretraining)
#     -> Stage 1 (per-part CLIP prompt learning)
#     -> cache_text_anchors.py (builds Stage 2's frozen text-prototype table)
#     -> Stage 2 (supervised backbone finetune)
#
# Intended to run on a training machine, in the "py312" conda environment. Stops on the first
# failure (set -e) rather than continuing on to a stage whose inputs didn't build correctly.
#
# Usage:
#   ./run_full_pipeline.sh
#
# Config paths default to this repo's own configs/ directory; override any of them via
# environment variables if you want to point at different YAML files without editing this script:
#   STAGE0_CONFIG=configs/my_stage0.yaml ./run_full_pipeline.sh
#
# Equivalent orchestration also exists as examples/run_pipeline.py (Python, same stage sequence,
# used elsewhere in this repo) -- this script is the same pipeline as a plain bash driver, for
# running directly on a training machine without needing to reason about that script's own CLI.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-py312}"

# Resolve the repo root as the directory this script itself lives in, so it works regardless of
# the caller's current directory (all config/checkpoint paths below are repo-root-relative,
# matching every example script's own convention).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

STAGE0_CONFIG="${STAGE0_CONFIG:-configs/stage0_bpa_segmentation.yaml}"
STAGE1_CONFIG="${STAGE1_CONFIG:-configs/stage1_relational_prompts.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-configs/stage2_relational_finetune.yaml}"

DRIVER_LOG_DIR="logs/pipeline_runs"
mkdir -p "$DRIVER_LOG_DIR"
DRIVER_LOG="$DRIVER_LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER_LOG"
}

# --- conda activation -------------------------------------------------------------------------
# `conda activate` doesn't work in a non-interactive shell until conda's own shell hook has been
# sourced -- this is the standard fix, not optional boilerplate.
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
log "Activated conda env '$CONDA_ENV' (python: $(command -v python), $(python --version 2>&1))"

if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
fi

run_stage() {
    local name="$1"
    shift
    log "=================================================================="
    log "Starting: $name"
    log "Command: $*"
    log "=================================================================="
    local start_ts=$SECONDS
    "$@"
    local elapsed=$((SECONDS - start_ts))
    log "Finished: $name (${elapsed}s)"
}

run_stage "Stage 0 -- BPA segmentation pretraining" \
    python examples/train_bpa_segmentation.py --config "$STAGE0_CONFIG"

run_stage "Stage 1 -- per-part CLIP prompt learning" \
    python examples/train_relational_prompts.py --config "$STAGE1_CONFIG"

run_stage "Building Stage 2's frozen text-prototype table" \
    python examples/cache_text_anchors.py --config "$STAGE1_CONFIG"

run_stage "Stage 2 -- supervised backbone finetune" \
    python examples/train_relational_finetune.py --config "$STAGE2_CONFIG"

log "=================================================================="
log "Full pipeline complete."
log "=================================================================="
