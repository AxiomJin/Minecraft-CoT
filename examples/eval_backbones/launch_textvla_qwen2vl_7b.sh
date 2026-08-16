#!/bin/bash
# ============================================================================
# 评测 minecraft-textvla-qwen2vl-7b-2509
# ── 使用与骨干模型(Qwen2-VL-7B/Qwen2.5-VL-7B/Qwen3.5-9B)完全一致的实验设置
#
# 模型信息：
#   基座: Qwen/Qwen2-VL-7B-Instruct
#   训练: SFT on CraftJarvis/minecraft-text-action-dataset (~215K 样本)
#   论文: OpenHA (arXiv:2509.13347)
#   架构: Qwen2VLForConditionalGeneration (Qwen2-VL 原生架构, vllm 0.8.5 直接兼容)
# ============================================================================
set -o pipefail
export MODEL_LOCAL_NAME="minecraft-textvla-qwen2vl-7b-2509"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/minecraft-textvla-qwen2vl-7b-2509/"
export SERVED_MODEL_NAME="eval-minecraft-textvla-qwen2vl-7b-2509"
export VLLM_CONDA_ENV="openha"   # Qwen2-VL 架构，vLLM==0.8.5 原生支持
export REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
source "$(dirname "$0")/run_backbone_eval.sh"
