#!/bin/bash
# ============================================================================
# 评测任务统一提交入口 —— 在本机(有 koala CLI 权限)一键提交某个
# launch_<model>.sh 对应的评测 job，保证今后所有评测都调用同一套固定实验设置
# (EVAL_BENCHMARK=mini 是 run_backbone_eval.sh 的默认值，详见其顶部注释)。
#
# 用法：
#   bash submit_eval_job.sh launch_qwen2vl.sh
#   bash submit_eval_job.sh launch_qwen25vl.sh
#   bash submit_eval_job.sh launch_qwen35.sh
#   CKPT=820 bash submit_eval_job.sh launch_textvla_qwen2vl_7b_checkpoint.sh
#   CKPT=820 bash submit_eval_job.sh launch_textvla_qwen35_9b_checkpoint.sh
#
# 前提：本地代码改动要先同步到koala job读取代码的S3路径，比如：
#   s5cmd sync examples/eval_backbones/ \
#       s3://arcwm-code-us-west-2/axiom/code/Minecraft-CoT/examples/eval_backbones/
#
# 可选环境变量：
#   EVAL_BENCHMARK   评测规模，默认继承 run_backbone_eval.sh 的默认值(mini)。
#                    需要跑完整benchmark时显式设 EVAL_BENCHMARK=full。
#   CKPT             仅对 launch_textvla_*_checkpoint.sh 有意义，指定要评测的
#                    训练 step（如 400/600/820）。
#   CODE_S3_URI      代码同步的S3路径，默认 s3://arcwm-code-us-west-2/axiom/code
#   JOB_TAG          job名后缀，默认用当前时间戳，用于区分同一模型的多次提交
# ============================================================================
set -euo pipefail

LAUNCH_SCRIPT="${1:?用法: bash submit_eval_job.sh <launch_script.sh>，例如 launch_qwen35.sh}"
CODE_S3_URI="${CODE_S3_URI:-s3://arcwm-code-us-west-2/axiom/code}"
JOB_TAG="${JOB_TAG:-$(date +%Y%m%d%H%M%S)}"

MODEL_TAG="${LAUNCH_SCRIPT#launch_}"
MODEL_TAG="${MODEL_TAG%.sh}"
MODEL_TAG="${MODEL_TAG//_/-}"   # koala job名仅支持小写字母/数字/连字符，不能有下划线
CKPT_SUFFIX=""
CKPT_EXPORT=""
if [ -n "${CKPT:-}" ]; then
    CKPT_SUFFIX="-ckpt${CKPT}"
    CKPT_EXPORT="export CKPT=${CKPT}; "
fi
BENCH_EXPORT=""
BENCH_SUFFIX=""
if [ -n "${EVAL_BENCHMARK:-}" ]; then
    BENCH_EXPORT="export EVAL_BENCHMARK=${EVAL_BENCHMARK}; "
    BENCH_SUFFIX="-${EVAL_BENCHMARK}"
fi

# koala job名有长度限制(含koala自动追加的 -normal-<timestamp> 后缀)，稳妥截断。
JOB_NAME="axiomjin-eval-${MODEL_TAG}${CKPT_SUFFIX}${BENCH_SUFFIX}-${JOB_TAG}"
JOB_NAME="${JOB_NAME:0:55}"

REMOTE_CMD="set -euo pipefail; export REPO_ROOT=/data/work/run_codes/Minecraft-CoT; ${CKPT_EXPORT}${BENCH_EXPORT}cd /data/work/run_codes/Minecraft-CoT; apt-get update -qq 2>&1 | tail -3 || true; apt-get install -y -qq xvfb 2>&1 | tail -5 || true; bash examples/eval_backbones/${LAUNCH_SCRIPT}"

echo "[submit] job=${JOB_NAME}"
echo "[submit] launch_script=${LAUNCH_SCRIPT} ckpt=${CKPT:-<none>} eval_benchmark=${EVAL_BENCHMARK:-<run_backbone_eval.sh默认值>}"
koala submit -m normal -j "${JOB_NAME}" -g 1 \
    -c "${REMOTE_CMD}" \
    --code "${CODE_S3_URI}:/data/work/run_codes" \
    --large-ssd --s3-log -y
