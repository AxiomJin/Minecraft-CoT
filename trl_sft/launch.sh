#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# TRL SFT training launch script for Minecraft VLM.
#
# Usage:
#   bash launch.sh debug                       # quick 1-GPU smoke test
#   bash launch.sh train --nproc 8              # Stage I/II, single node
#   bash launch.sh stage3 --nproc 8              # Stage III, full-trajectory multi-step loss
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash launch.sh train   # multi-node, node 0
#
# Every mode self-bootstraps its Python env (see `bootstrap_env`) and localizes S3 data
# to local disk BEFORE torchrun starts -- both were found, empirically, to be the two
# most common causes of a training job dying: (a) the koala training image ships NO
# torch/trl/deepspeed at all (only a bare `base` conda env), so `torchrun` is simply not
# on PATH until an env is built; (b) leaving --data_path/--image_root as `s3://...`
# makes every DataLoader worker on every rank stream rows/images LIVE from S3 for the
# whole run -- one slow/stalled read on a single rank blocks that rank's step, which
# then blocks every other rank's collective ops until NCCL's 600s watchdog aborts the
# whole job (confirmed on two real 16-GPU Stage II runs).
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── defaults ──
MODE="${MODE:-train}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29400}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        train|debug|stage3) MODE="$1"; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --attn-impl) ATTN_IMPL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B-stage1-8gpu-20260817}"
DATA_PATH="${DATA_PATH:-s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-vqa-241102.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-caption-241104.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-grounding-point-embodied-image5.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-grounding-point-embodied.jsonl,s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-grounding-point-gui.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/local-ssd/minecraft-vlp}"
LOCAL_PARQUET_ROOT="${LOCAL_PARQUET_ROOT:-/local-ssd/minecraft-text-action}"
OUTPUT_DIR="${OUTPUT_DIR:-./stage2-qwen35-9b}"
DOWNLOAD_CACHE="${DOWNLOAD_CACHE:-/local-ssd/model_cache}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
STAGE2_TRAIN_SAMPLES="${STAGE2_TRAIN_SAMPLES:-261461}"
TOTAL_GPUS=$((NNODES * NPROC))
MAX_STEPS="${MAX_STEPS:-$((STAGE2_TRAIN_SAMPLES / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * TOTAL_GPUS)))}"

# W&B: sourced from a git-ignored local file (this repo is PUBLIC on GitHub -- never
# commit a real key). Create trl_sft/.env.wandb with WANDB_API_KEY/WANDB_PROJECT/
# WANDB_RUN_NAME. Only applies to LOCAL runs of this script; remote koala jobs must
# export WANDB_API_KEY explicitly in the submit command instead.
if [ -f "$SCRIPT_DIR/.env.wandb" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env.wandb"
fi

# Make the training env ready to run `torchrun` (idempotent -- skips whatever's already
# installed). The default koala training image ships NO torch/trl/deepspeed at all, and
# has no `openha` (or similar) conda env baked in -- both confirmed by trial and error,
# so this is the single source of truth going forward instead of re-discovering it via
# failed jobs. requirements.txt has flash-attn commented out (see its own comment) --
# --attn-impl sdpa (the default) does NOT need it installed at all.
bootstrap_env() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "[env] no conda found, assuming torch/trl/deepspeed are already on PATH." >&2
        return
    fi
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    conda env list | grep -q "^sft " || conda create -n sft python=3.10 -y -q
    conda activate sft
    python -c "import torch" 2>/dev/null || \
        pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    python -c "import trl, transformers, deepspeed, datasets, accelerate" 2>/dev/null || \
        pip install -q -r requirements.txt
}

# Pre-download every Stage II JSONL shard in `$DATA_PATH` plus every image
# `$IMAGE_ROOT` points to onto local SSD, then REPOINT `$DATA_PATH`/`$IMAGE_ROOT` at the
# local copies. Must run on EACH node (local SSD is node-local). No-op if `$DATA_PATH`
# is already local.
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
}

# Pre-download `minecraft-text-action-dataset`'s parquet shards (Stage III) onto local
# SSD, then repoint `$DATA_PATH` at the local glob -- same rationale as
# `localize_stage2_jsonl_and_images` above. Each parquet row embeds its own
# `image_bytes` column, so no separate image sync is needed. Expects `$DATA_PATH` to be
# a SINGLE glob (e.g. "s3://.../data/train-*.parquet"), not a comma-separated list.
localize_stage3_parquet_dataset() {
    if [[ "$DATA_PATH" != s3://* ]]; then
        echo "[localize] DATA_PATH is already local, skipping." >&2
        return
    fi

    local s3_dir="${DATA_PATH%/*}/"
    local glob_name="$(basename "$DATA_PATH")"
    mkdir -p "$LOCAL_PARQUET_ROOT"
    echo "[localize] Syncing parquet shards from $s3_dir to $LOCAL_PARQUET_ROOT ..." >&2
    if command -v s5cmd >/dev/null 2>&1; then
        s5cmd sync --include "*.parquet" "${s3_dir}*" "$LOCAL_PARQUET_ROOT/"
    else
        aws s3 sync "$s3_dir" "$LOCAL_PARQUET_ROOT/" --exclude "*" --include "*.parquet" --only-show-errors
    fi

    # Quoted below at the call site -- an UNQUOTED glob here would be expanded by bash
    # into hundreds of positional args before argparse ever sees it (confirmed root
    # cause of a real launch failure: `--data_path` is a single str that does its own
    # internal glob matching, not something the shell should expand for it).
    DATA_PATH="$LOCAL_PARQUET_ROOT/$glob_name"
    echo "[localize] Done. DATA_PATH=$DATA_PATH" >&2
}

case "$MODE" in
    debug)
        echo "=== Running DEBUG dry-run ==="
        bootstrap_env
        localize_stage2_jsonl_and_images
        python3 train_sft.py \
            --model_path "$MODEL_PATH" \
            --data_path "$DATA_PATH" \
            --data_format jsonl \
            --image_root "$IMAGE_ROOT" \
            --download_model "$DOWNLOAD_CACHE" \
            --attn_implementation "$ATTN_IMPL" \
            --max_turns 2 \
            --debug
        ;;
    train)
        echo "=== Stage I/II training: NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC=$NPROC ==="
        bootstrap_env
        localize_stage2_jsonl_and_images
        echo "MODEL_PATH=$MODEL_PATH OUTPUT_DIR=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS TOTAL_GPUS=$TOTAL_GPUS"
        torchrun \
            --nnodes="$NNODES" --nproc_per_node="$NPROC" --node_rank="$NODE_RANK" \
            --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
            train_sft.py \
                --model_path "$MODEL_PATH" \
                --data_path "$DATA_PATH" \
                --data_format jsonl \
                --image_root "$IMAGE_ROOT" \
                --download_model "$DOWNLOAD_CACHE" \
                --output_dir "$OUTPUT_DIR" \
                --resume_from_checkpoint auto \
                --attn_implementation "$ATTN_IMPL" \
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
    stage3)
        : "${MODEL_PATH:?stage3 needs an explicit MODEL_PATH (a Stage II checkpoint)}"
        MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-19456}"          # full trajectories need far more room than Stage I/II's 16384
        PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
        GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
        DATA_PATH="${DATA_PATH_STAGE3:-s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset/data/train-*.parquet}"
        OUTPUT_DIR="${OUTPUT_DIR_STAGE3:-./stage3-output}"
        MAX_STEPS="${MAX_STEPS_STAGE3:-3400}"
        echo "=== Stage III training (full-trajectory multi-step loss): NPROC=$NPROC ==="
        bootstrap_env
        localize_stage3_parquet_dataset
        echo "MODEL_PATH=$MODEL_PATH OUTPUT_DIR=$OUTPUT_DIR MAX_STEPS=$MAX_STEPS"
        torchrun --nproc_per_node="$NPROC" --tee 3 train_sft.py \
            --model_path "$MODEL_PATH" \
            --data_path "$DATA_PATH" \
            --data_format parquet \
            --download_model "$DOWNLOAD_CACHE" \
            --output_dir "$OUTPUT_DIR" \
            --resume_from_checkpoint auto \
            --full_trajectory \
            --attn_implementation "$ATTN_IMPL" \
            --max_seq_length "$MAX_SEQ_LENGTH" \
            --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
            --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
            --gradient_checkpointing \
            --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
            --num_train_epochs 1 \
            --max_steps "$MAX_STEPS" \
            --learning_rate 8e-6 \
            --weight_decay 0.05 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --deepspeed ds_zero2_no_offload.json \
            --save_steps 200 \
            --logging_steps 10
        ;;
    *)
        echo "Usage: bash launch.sh [debug|train|stage3]"
        exit 1
        ;;
esac
