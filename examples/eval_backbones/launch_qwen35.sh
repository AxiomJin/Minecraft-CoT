#!/bin/bash
set -o pipefail
export MODEL_LOCAL_NAME="Qwen3.5-9B"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B/"
export SERVED_MODEL_NAME="eval-qwen3.5-9b"
export VLLM_CONDA_ENV="vllm35"   # Qwen3.5 混合线性/全注意力架构，需 vllm>=0.17.0，单独装环境
source "$(dirname "$0")/run_backbone_eval.sh"
