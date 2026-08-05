#!/bin/bash
set -o pipefail
export MODEL_LOCAL_NAME="Qwen2.5-VL-7B-Instruct"
export MODEL_S3_URI="s3://arcwm-code-us-west-2/axiom/model/Qwen2.5-VL-7B-Instruct/"
export SERVED_MODEL_NAME="eval-qwen2.5vl-7b-instruct"
export VLLM_CONDA_ENV="openha"   # 原生支持，用主环境(vllm==0.8.5)即可
source "$(dirname "$0")/run_backbone_eval.sh"
