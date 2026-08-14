#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# TRL SFT training launch script for Minecraft VLM
#
# Usage (single node, 8 GPUs):
#   bash launch.sh --mode train --nproc 8
#
# Usage (multi-node, 2 nodes × 8 GPUs):
#   # Node 0:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash launch.sh --mode train
#   # Node 1:
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash launch.sh --mode train
#
# Debug mode (quick test):
#   bash launch.sh --mode debug
#
# `train`/`debug` fully LOCALIZE the Stage II JSONL shards + referenced images to
# `$LOCAL_DATA_ROOT` on local SSD BEFORE torchrun starts (see
# `localize_stage2_jsonl_and_images` below). Root cause this avoids: with
# `--data_path`/`--image_root` left as `s3://...`, every DataLoader worker on every
# rank streams JSONL rows and reads images LIVE from S3 for the entire training run.
# Confirmed in two real 16-GPU Stage II runs (2026-08-13): a stalled/slow live S3 read
# on just ONE rank starves that rank's forward/backward step, which then blocks every
# other rank's DeepSpeed `ALLREDUCE` (a collective op needs ALL ranks) until NCCL's
# watchdog timeout fires and aborts the whole job -- in both runs the last per-rank log
# line was an S3 credential lookup, followed by ~600s of total silence, then the
# watchdog. Localizing ahead of time means training reads only the local filesystem
# (no live S3 dependency, no per-step network variance) -- `--model_path` is separately
# already downloaded once via `--download_model` before this happens.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── defaults ──
MODE="${MODE:-train}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29400}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        train|debug|preflight) MODE="$1"; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

MODEL_PATH="${MODEL_PATH:-s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B-stage1-16gpu}"
DATA_PATH="${DATA_PATH:-s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-vqa-241102.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-caption-241104.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-grounding-point-embodied-image5.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-grounding-point-embodied.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-grounding-point-gui.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/local-ssd/minecraft-vlp}"
OUTPUT_DIR="${OUTPUT_DIR:-./stage2-qwen35-9b}"
DOWNLOAD_CACHE="${DOWNLOAD_CACHE:-/tmp/qwen35_9b_stage1_cache}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-./stage2-preflight-report.json}"
STAGE2_TRAIN_SAMPLES="${STAGE2_TRAIN_SAMPLES:-261461}"
TOTAL_GPUS=$((NNODES * NPROC))
MAX_STEPS="${MAX_STEPS:-$((STAGE2_TRAIN_SAMPLES / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TOTAL_GPUS)))}"

# W&B: sourced from a git-ignored local file (this repo is PUBLIC on GitHub -- never
# commit a real key). Create trl_sft/.env.wandb with:
#   export WANDB_API_KEY="your-key"
#   export WANDB_PROJECT="minecraft-sft"
#   export WANDB_RUN_NAME="trl-sft-v1"   # optional, defaults to "minecraft-sft-trl"
# NOTE: this only helps for LOCAL runs of this script. For remote koala training jobs,
# this file is NOT synced to S3 (same reason) -- export WANDB_API_KEY explicitly inside
# the `koala submit -c "..."` command instead.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env.wandb" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env.wandb"
fi

# Pre-download every Stage II JSONL shard in `$DATA_PATH` plus every image
# `$IMAGE_ROOT` points to onto local SSD, then REPOINT `$DATA_PATH`/`$IMAGE_ROOT` at the
# local copies -- so torchrun never touches S3 again once training starts. Must run
# independently on EACH node (local SSD is node-local, not shared). No-op if
# `$DATA_PATH` is already local (e.g. a previous invocation on this same pod already
# localized it, or the caller passed local paths directly).
localize_stage2_jsonl_and_images() {
    if [[ "$DATA_PATH" != s3://* ]]; then
        echo "[localize] DATA_PATH is already local, skipping." >&2
        return
    fi
    if ! python3 -c "import s3fs" 2>/dev/null; then
        echo "Missing s3fs: install trl_sft/requirements.txt before using S3 data or weights." >&2
        exit 1
    fi

    mkdir -p "$LOCAL_DATA_ROOT"
    echo "[localize] Downloading Stage II JSONL shard(s) to $LOCAL_DATA_ROOT ..." >&2
    local_paths=()
    IFS=',' read -ra shard_uris <<< "$DATA_PATH"
    for shard_uri in "${shard_uris[@]}"; do
        local_file="$LOCAL_DATA_ROOT/$(basename "$shard_uri")"
        if [ ! -s "$local_file" ]; then
            aws s3 cp "$shard_uri" "$local_file" --only-show-errors
        else
            echo "[localize] $local_file already present, skipping re-download." >&2
        fi
        local_paths+=("$local_file")
    done

    echo "[localize] Syncing referenced images from $IMAGE_ROOT to $LOCAL_DATA_ROOT ..." >&2
    if command -v s5cmd >/dev/null 2>&1; then
        s5cmd sync --exclude "*.jsonl" "${IMAGE_ROOT%/}/*" "$LOCAL_DATA_ROOT/"
    else
        aws s3 sync "${IMAGE_ROOT%/}/" "$LOCAL_DATA_ROOT/" --exclude "*.jsonl" --only-show-errors
    fi

    DATA_PATH="$(IFS=,; echo "${local_paths[*]}")"
    IMAGE_ROOT="$LOCAL_DATA_ROOT"
    echo "[localize] Done. DATA_PATH=$DATA_PATH IMAGE_ROOT=$IMAGE_ROOT" >&2
    echo "[localize] Local image count: $(find "$LOCAL_DATA_ROOT" -type f ! -name '*.jsonl' | wc -l)" >&2
}

case "$MODE" in
    preflight)
        echo "=== Running Stage II data preflight ==="
        # Deliberately reads directly from S3 (NOT localized): preflight's purpose is
        # to validate the authoritative source data itself, independent of any local
        # caching step.
        if ! python3 -c "import s3fs"; then
            echo "Missing s3fs: install trl_sft/requirements.txt before reading S3 Stage II data." >&2
            exit 1
        fi
        python3 train_sft.py \
            --model_path "$MODEL_PATH" \
            --data_path "$DATA_PATH" \
            --data_format jsonl \
            --image_root "$IMAGE_ROOT" \
            --preflight \
            --preflight_report "$PREFLIGHT_REPORT"
        ;;
    debug)
        echo "=== Running DEBUG dry-run ==="
        localize_stage2_jsonl_and_images
        python3 train_sft.py \
            --model_path "$MODEL_PATH" \
            --data_path "$DATA_PATH" \
            --data_format jsonl \
            --image_root "$IMAGE_ROOT" \
            --download_model "$DOWNLOAD_CACHE" \
            --max_turns 2 \
            --debug
        ;;
    train)
        echo "=== Multi-node training ==="
        echo "NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC=$NPROC"
        echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

        localize_stage2_jsonl_and_images
        echo "MODEL_PATH=$MODEL_PATH"
        echo "OUTPUT_DIR=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS TOTAL_GPUS=$TOTAL_GPUS"
        torchrun \
            --nnodes="$NNODES" \
            --nproc_per_node="$NPROC" \
            --node_rank="$NODE_RANK" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            train_sft.py \
                --model_path "$MODEL_PATH" \
                --data_path "$DATA_PATH" \
                --data_format jsonl \
                --image_root "$IMAGE_ROOT" \
                --download_model "$DOWNLOAD_CACHE" \
                --output_dir "$OUTPUT_DIR" \
                --resume_from_checkpoint auto \
                --max_turns 4 \
                --max_seq_length "$MAX_SEQ_LENGTH" \
                --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
                --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
                --gradient_checkpointing \
                --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
                --num_train_epochs 1 \
                --max_steps "$MAX_STEPS" \
                --learning_rate 8e-6 \
                --deepspeed ds_zero2.json \
                --save_steps 200 \
                --logging_steps 10
        ;;
    *)
        echo "Usage: bash launch.sh [preflight|debug|train]"
        exit 1
        ;;
esac
