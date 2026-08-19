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
# ============================================================================
set -euo pipefail

LAUNCH_SCRIPT="${1:?用法: bash submit_eval_job.sh <launch_script.sh>，例如 launch_qwen35.sh}"
CODE_S3_URI="${CODE_S3_URI:-s3://arcwm-code-us-west-2/axiom/code}"

# koala对 -j 传入的前缀名限制为29字符(其后会自动追加 "-<mode>-<timestamp>"
# 拼成完整job名，总长≤52)，不能直接用完整launch脚本名，需要一份简称映射表。
# 未来新增 launch_<model>.sh 时，请在这里补一条对应的短别名。
case "${LAUNCH_SCRIPT}" in
    launch_qwen2vl.sh)                         SHORT_TAG="q2vl" ;;
    launch_qwen25vl.sh)                        SHORT_TAG="q25vl" ;;
    launch_qwen35.sh)                          SHORT_TAG="q35" ;;
    launch_textvla_qwen2vl_7b.sh)              SHORT_TAG="tv-q2vl" ;;
    launch_textvla_qwen2vl_7b_checkpoint.sh)   SHORT_TAG="tv-q2vl" ;;
    launch_textvla_qwen35_9b_checkpoint.sh)    SHORT_TAG="tv-q35" ;;
    *)
        # 未知脚本：用脚本名(去掉launch_/.sh、下划线转连字符)截断到8字符 + 4位hash保证唯一
        SHORT_TAG="${LAUNCH_SCRIPT#launch_}"
        SHORT_TAG="${SHORT_TAG%.sh}"
        SHORT_TAG="${SHORT_TAG//_/-}"
        SHORT_TAG="${SHORT_TAG:0:8}-$(echo -n "${LAUNCH_SCRIPT}" | md5 2>/dev/null || echo -n "${LAUNCH_SCRIPT}" | md5sum | cut -c1-4)"
        SHORT_TAG="${SHORT_TAG:0:13}"
        ;;
esac

CKPT_SUFFIX=""
CKPT_EXPORT=""
if [ -n "${CKPT:-}" ]; then
    CKPT_SUFFIX="-c${CKPT}"
    CKPT_EXPORT="export CKPT=${CKPT}; "
fi
BENCH_EXPORT=""
if [ -n "${EVAL_BENCHMARK:-}" ]; then
    BENCH_EXPORT="export EVAL_BENCHMARK=${EVAL_BENCHMARK}; "
fi

# "axiomjin-eval-"(14字符) + SHORT_TAG + CKPT_SUFFIX，务必控制在29字符以内。
JOB_NAME="axiomjin-eval-${SHORT_TAG}${CKPT_SUFFIX}"
JOB_NAME="${JOB_NAME:0:29}"

REMOTE_CMD="set -euo pipefail; export REPO_ROOT=/data/work/run_codes/Minecraft-CoT; ${CKPT_EXPORT}${BENCH_EXPORT}cd /data/work/run_codes/Minecraft-CoT; apt-get update -qq 2>&1 | tail -3 || true; apt-get install -y -qq xvfb 2>&1 | tail -5 || true; bash examples/eval_backbones/${LAUNCH_SCRIPT}"

echo "[submit] job=${JOB_NAME}"
echo "[submit] launch_script=${LAUNCH_SCRIPT} ckpt=${CKPT:-<none>} eval_benchmark=${EVAL_BENCHMARK:-<run_backbone_eval.sh默认值>}"
koala submit -m normal -j "${JOB_NAME}" -g 1 \
    -c "${REMOTE_CMD}" \
    --code "${CODE_S3_URI}:/data/work/run_codes" \
    --large-ssd --s3-log -y
