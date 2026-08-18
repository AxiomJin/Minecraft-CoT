#!/bin/bash
# ============================================================================
# 评测本次重跑得到的 TextVLA-Qwen2VL-7B checkpoint-400
# ── 使用与骨干模型(Qwen2-VL-7B/Qwen2.5-VL-7B/Qwen3.5-9B)完全一致的实验设置
#
# 模型信息：
#   基座: 本次修复双重分片bug后重跑的 Qwen2-VL-7B Stage2 checkpoint
#   训练: SFT on CraftJarvis/minecraft-text-action-dataset (~215K 样本), step 400/820 (epoch≈0.49)
#   来源: s3://arcwm-code-us-west-2/axiom/model/TextVLA-qwen2vl-7b-stage3-from-new-stage2-20260818/checkpoint-400/
#         (仅拷贝了vLLM serving所需的servable文件，排除了116GB的deepspeed优化器状态)
#   架构: Qwen2VLForConditionalGeneration (Qwen2-VL 原生架构, vllm 0.8.5 直接兼容)
# ============================================================================
set -o pipefail
export MODEL_LOCAL_NAME="textvla-qwen2vl-7b-ckpt400-20260818"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/eval_snapshots/TextVLA-qwen2vl-7b-ckpt400-20260818/"
export SERVED_MODEL_NAME="eval-textvla-qwen2vl-7b-ckpt400"
export VLLM_CONDA_ENV="openha"   # Qwen2-VL 架构，vLLM==0.8.5 原生支持
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"
