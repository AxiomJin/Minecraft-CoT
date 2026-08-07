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
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── defaults ──
MODE="${1:-train}"
NPROC="${NPROC:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29400}"

MODEL_PATH="s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B"
DATA_PATH="s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset/data/train-*.parquet"
OUTPUT_DIR="./output"
DOWNLOAD_CACHE="/tmp/qwen35_9b_cache"

# W&B (set your own key)
# export WANDB_API_KEY="your-key"
# export WANDB_PROJECT="minecraft-sft"
# export WANDB_RUN_NAME="trl-sft-v1"

case "$MODE" in
    debug)
        echo "=== Running DEBUG dry-run ==="
        python3 train_sft.py \
            --model_path "$MODEL_PATH" \
            --data_path "$DATA_PATH" \
            --max_turns 2 \
            --debug
        ;;
    train)
        echo "=== Multi-node training ==="
        echo "NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC=$NPROC"
        echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

        torchrun \
            --nnodes="$NNODES" \
            --nproc_per_node="$NPROC" \
            --node_rank="$NODE_RANK" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            train_sft.py \
                --model_path "$MODEL_PATH" \
                --data_path "$DATA_PATH" \
                --download_model "$DOWNLOAD_CACHE" \
                --output_dir "$OUTPUT_DIR" \
                --max_turns 4 \
                --max_seq_length 16384 \
                --per_device_batch_size 2 \
                --gradient_accumulation_steps 4 \
                --num_train_epochs 1 \
                --learning_rate 8e-6 \
                --deepspeed ds_zero2.json \
                --save_steps 200 \
                --logging_steps 10
        ;;
    *)
        echo "Usage: bash launch.sh [debug|train]"
        exit 1
        ;;
esac
