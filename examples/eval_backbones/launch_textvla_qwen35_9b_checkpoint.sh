#!/bin/bash
# ============================================================================
# 评测本次修复双重分片bug后重跑得到的 TextVLA-Qwen3.5-9B Stage3 checkpoint
# ── 使用与骨干模型(Qwen2-VL-7B/Qwen2.5-VL-7B/Qwen3.5-9B)及Qwen2-VL版TextVLA
#    checkpoint 完全一致的实验设置
#
# 用法：CKPT=<step> bash launch_textvla_qwen35_9b_checkpoint.sh
#   例：CKPT=820 bash launch_textvla_qwen35_9b_checkpoint.sh   # 最终checkpoint(epoch=1.0)
#   不设 CKPT 则默认用 output_dir 根目录下的最终合并模型(等价于 checkpoint-820,
#   但没有对应的 checkpoint-N 优化器状态子目录负担，可直接当权重目录用)。
#
# 模型信息：
#   基座: 本次修复双重分片bug后重跑的 Qwen3.5-9B Stage2 checkpoint
#         (s3://.../Qwen3.5-9B-stage2-8gpu-20260817/)
#   训练: SFT on CraftJarvis/minecraft-text-action-dataset (~215K 样本), max_steps=820
#   来源: s3://.../TextVLA-qwen35-9b-stage3-from-new-stage2-20260818/
#         (根目录=最终模型, 不含优化器状态; 若指定CKPT则改用对应checkpoint-${CKPT}/子目录)
#   架构: Qwen3_5ForConditionalGeneration (混合线性/全注意力, 需 vllm>=0.17.0)
# ============================================================================
set -o pipefail

BASE_S3="s3://arcwm-code-us-west-2/axiom/model/TextVLA-qwen35-9b-stage3-from-new-stage2-20260818"
if [ -n "${CKPT:-}" ] && [ "${CKPT}" != "820" ]; then
    export MODEL_LOCAL_NAME="textvla-qwen35-9b-ckpt${CKPT}-20260818"
    export MODEL_S3_URI="${BASE_S3}/checkpoint-${CKPT}/"
    export SERVED_MODEL_NAME="eval-textvla-qwen35-9b-ckpt${CKPT}"
else
    # CKPT=820(或未设置)：直接用根目录最终模型，等价于checkpoint-820但没有
    # 优化器状态负担，也不需要额外裁剪 eval_snapshot。
    export MODEL_LOCAL_NAME="textvla-qwen35-9b-ckpt820-20260818"
    export MODEL_S3_URI="${BASE_S3}/"
    export SERVED_MODEL_NAME="eval-textvla-qwen35-9b-ckpt820"
fi
export VLLM_CONDA_ENV="vllm35"   # Qwen3.5 混合线性/全注意力架构，需 vllm>=0.17.0，单独装环境
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"
