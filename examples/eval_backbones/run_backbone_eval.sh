#!/bin/bash
# ============================================================================
# 通用"骨干VLM"评测编排脚本 —— 在同一套InkRL evaluate pipeline下，
# 用完全一致的实验设置评测不同的VLM backbone (Qwen2-VL-7B-Instruct /
# Qwen2.5-VL-7B-Instruct / Qwen3.5-9B 等)。
#
# 用法（由 launch_<model>.sh 设置好下列环境变量后 source 本脚本）：
#   MODEL_LOCAL_NAME     模型短名，如Qwen2-VL-7B-Instruct
#   MODEL_S3_URI          权重所在S3路径，如 s3://arcwm-code-us-west-2/axiom/model/Qwen2-VL-7B-Instruct/
#   SERVED_MODEL_NAMEvllm --served-model-name / rollout --model_id，如 eval-qwen2vl-7b
#   VLLM_CONDA_ENV        起服务用哪个conda env（openha 或 vllm35，二者只是vllm版本不同）
#
# 所有"实验设置"相关的超参数在下方统一定义为常量，三个模型共用，保证可比性。
# ============================================================================
set -o pipefail
# 注意：不用 `set -u`——conda(如 openjdk 包的 activate.d/deactivate.d hook)
# 内部会引用一些未初始化的变量(如 JAVA_HOME_CONDA_BACKUP)，开-u 会导致脚本被杀死。

: "${MODEL_LOCAL_NAME:?must set MODEL_LOCAL_NAME}"
: "${MODEL_S3_URI:?must set MODEL_S3_URI}"
: "${SERVED_MODEL_NAME:?must set SERVED_MODEL_NAME}"
: "${VLLM_CONDA_ENV:=openha}"

# ---- 固定的、三模型共用的评测设置（保证公平对比） -------------------------
export TASK_LIST=${TASK_LIST:-"mine_block:oak_log kill_entity:sheep craft_item:crafting_table"}
export ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK:-5}
export MAX_STEPS_NUM=${MAX_STEPS_NUM:-200}
export MAXIMUM_HISTORY_LENGTH=${MAXIMUM_HISTORY_LENGTH:-3}
export DIFFICULTY=${DIFFICULTY:-zero}
export OUTPUT_MODE=${OUTPUT_MODE:-text_action}
export SYSTEM_MESSAGE_TAG=${SYSTEM_MESSAGE_TAG:-text_action}
export TEMPERATURE=${TEMPERATURE:-0.8}
export TOP_P=${TOP_P:-0.99}
export TOP_K=${TOP_K:--1}
export FPS=${FPS:-20}
export LIMIT_MM_IMAGE=${LIMIT_MM_IMAGE:-5}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
export TP_SIZE=${TP_SIZE:-1}
export GPU_PER_ROLLOUT=${GPU_PER_ROLLOUT:-0.1}
export VLLM_PORT=${VLLM_PORT:-11000}

REPO_ROOT="${REPO_ROOT:-/data/work/run_codes}"
LOCAL_MODEL_DIR="/local-ssd/models/${MODEL_LOCAL_NAME}"
RECORD_ROOT="/local-ssd/eval_output/${MODEL_LOCAL_NAME}"
RESULT_S3_URI="s3://arcwm-code-us-west-2/axiom/eval_results/${MODEL_LOCAL_NAME}"
LOG_DIR="/local-ssd/logs/${MODEL_LOCAL_NAME}"
export MINESTUDIO_DIR="${MINESTUDIO_DIR:-/local-ssd/minestudio}"
mkdir -p "${LOCAL_MODEL_DIR}" "${RECORD_ROOT}" "${LOG_DIR}" "${MINESTUDIO_DIR}"

echo "=============================================================="
echo "[eval] model=${MODEL_LOCAL_NAME} served_name=${SERVED_MODEL_NAME} vllm_env=${VLLM_CONDA_ENV}"
echo "[eval] tasks=${TASK_LIST} rollouts_per_task=${ROLLOUTS_PER_TASK} max_steps=${MAX_STEPS_NUM}"
echo "=============================================================="

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"

# MineStudio(minerl/Malmo) 无头渲染需要 xvfb-run，基础镜像未预装，需系统级安装一次。
if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[setup] installing xvfb (system package, needed by MineStudio headless rendering)"
    apt-get update -qq && apt-get install -y -qq xvfb
fi

# ---------------------------------------------------------------------------
# 1. 准备 openha 评测环境（安装 InkRL / openagents，供rollout_openha.py 使用；
#    该环境仅用 openai client 以 HTTP 方式访问 vLLM 服务，与serve端vllm版本无关）
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"
if ! conda env list | grep -qE "^openha "; then
    echo "[setup] creating conda env: openha"
    conda create -n openha python=3.10 -y
fi
conda activate openha
# 注意：不能用 `import openagents` 判断是否已装好——cwd(=REPO_ROOT)下就有 openagents/
# 源码目录，`python -c` 默认把cwd 加入 sys.path，即使没pip install 也能import 到，
# 从而误判"已安装"。改用一个真实第三方依赖(torch/vllm)来判断是否需要安装。
if ! python -c "import torch, vllm, minestudio" >/dev/null 2>&1; then
    echo "[setup] installing openagents + deps into openha env"
    pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    conda install --channel=conda-forge openjdk=8 -y -q
    pip install -q -e .
fi
# openagents/agents/base.py 顶层无条件 `from sam2.build_sam import build_sam2_camera_predictor`
# （即便 text_action 评测根本不用 grounding/SAM2）。仓库通过 git submodule 引用了一个
# 带camera_predictor 的 SAM2 fork(external/SAM2 -> zhwang4ai/SAM2)，但本环境未初始化
# submodule，因此直接从该 fork 的 GitHub 地址 pip 安装，保证 import 不报错。
if ! python -c "from sam2.build_sam import build_sam2_camera_predictor" >/dev/null 2>&1; then
    echo "[setup] installing sam2 (zhwang4ai/SAM2 fork, for base.py import compatibility)"
    pip install -q "git+https://github.com/zhwang4ai/SAM2.git"
fi
# minestudio 依赖 cuda-python 但未pin版本；pip默认装到cuda-python>=13(namespace 包，
# 移除了 `from cuda import cuda, cudart` 这种旧式扁平API)，导致 minerl 的 gpu_utils.py
# 报 ImportError。显式装回和 CUDA12.x匹配的旧版 API。
if ! python -c "from cuda import cuda, cudart" >/dev/null 2>&1; then
    echo "[setup] pinning cuda-python==12.6.2.post1 for minestudio/minerl gpu_utils compatibility"
    pip install -q "cuda-python==12.6.2.post1"
fi
# MineStudio首次运行会交互式询问是否下载模拟器引擎(Y/N)，在非交互 job 里会直接 EOFError。
# 提前非交互下载好，避免 rollout 阶段卡死。
if ! python -c "
import os
from minestudio.utils import get_mine_studio_dir
jar = os.path.join(get_mine_studio_dir(), 'engine', 'build', 'libs', 'mcprec-6.13.jar')
assert os.path.exists(jar)
" >/dev/null 2>&1; then
    echo "[setup] downloading MineStudio simulator engine to ${MINESTUDIO_DIR}"
    python -c "from minestudio.simulator.entry import download_engine; download_engine()"
fi
echo "[setup] openha env ready: $(python -c 'import torch,vllm; print(f"torch={torch.__version__} vllm={vllm.__version__}")')"

# ---------------------------------------------------------------------------
# 2. 若该模型需要单独的 vllm 版本（如 Qwen3.5需要 vllm>=0.17.0），
#    准备一个独立的 conda env 只用来跑 `vllm serve`。
# ---------------------------------------------------------------------------
if [ "${VLLM_CONDA_ENV}" != "openha" ]; then
    if ! conda env list | grep -qE "^${VLLM_CONDA_ENV} "; then
        echo "[setup] creating conda env: ${VLLM_CONDA_ENV} (vllm==0.17.0 for newer architectures)"
        conda create -n "${VLLM_CONDA_ENV}" python=3.11 -y
    fi
    conda activate "${VLLM_CONDA_ENV}"
    if ! python -c "import vllm" >/dev/null 2>&1; then
        pip install -q --no-cache-dir "vllm==0.17.0"
    fi
    echo "[setup] ${VLLM_CONDA_ENV} env ready: $(python -c 'import vllm;print(vllm.__version__)')"
    conda activate openha
fi

# ---------------------------------------------------------------------------
# 3. 下载模型权重到本地盘（vLLM 需要本地路径）
# ---------------------------------------------------------------------------
if [ ! -f "${LOCAL_MODEL_DIR}/config.json" ]; then
    echo "[download] syncing ${MODEL_S3_URI} -> ${LOCAL_MODEL_DIR}"
    aws s3 sync "${MODEL_S3_URI%/}/" "${LOCAL_MODEL_DIR}/" --no-progress
else
    echo "[download] model already present at ${LOCAL_MODEL_DIR}, skip"
fi

# ---------------------------------------------------------------------------
# 4. 启动 vLLM OpenAI-compatible server（后台）
# ---------------------------------------------------------------------------
VLLM_LOG="${LOG_DIR}/vllm_serve.log"
# vllm==0.8.5 的 --limit-mm-per-prompt 用 key=value 格式(如 image=5)；
# vllm==0.17.0 改成了 JSON 格式(如 '{"image": 5}')，两个版本不兼容，需按serve环境区分。
if [ "${VLLM_CONDA_ENV}" = "openha" ]; then
    LIMIT_MM_ARG="image=${LIMIT_MM_IMAGE}"
else
    LIMIT_MM_ARG="{\"image\": ${LIMIT_MM_IMAGE}}"
fi
echo "[serve] launching vllm serve (env=${VLLM_CONDA_ENV}) -> ${VLLM_LOG}"
conda run --no-capture-output -n "${VLLM_CONDA_ENV}" vllm serve "${LOCAL_MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --limit-mm-per-prompt "${LIMIT_MM_ARG}" \
    --trust-remote-code --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-num-seqs 16 \
    --max-logprobs 20 \
    --max-model-len "${MAX_MODEL_LEN}" \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "[serve] vllm pid=${VLLM_PID}"

echo "[serve] waiting for server to become healthy on :${VLLM_PORT} ..."
READY=0
for i in $(seq 1 90); do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        READY=1
        echo "[serve] server ready after ${i}0s"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[serve][ERROR] vllm process died early, see ${VLLM_LOG}"
        tail -n 200 "${VLLM_LOG}"
        break
    fi
    sleep 10
done

if [ "${READY}" != "1" ]; then
    echo "[serve][FATAL] vllm server never became healthy, aborting eval for ${MODEL_LOCAL_NAME}"
    kill "${VLLM_PID}" 2>/dev/null
    tail -n 300 "${VLLM_LOG}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. 逐任务跑 rollout（openha env，online模式，纯HTTP调用，与vllm版本无关）
# ---------------------------------------------------------------------------
conda activate openha
cd "${REPO_ROOT}"
for TASK in ${TASK_LIST}; do
    echo "------------------------------------------------------------"
    echo "[rollout] task=${TASK} num_rollouts=${ROLLOUTS_PER_TASK}"
    echo "------------------------------------------------------------"
    python examples/rollout_openha.py \
        --output_mode "${OUTPUT_MODE}" \
        --vlm_client_mode online \
        --system_message_tag "${SYSTEM_MESSAGE_TAG}" \
        --model_ips localhost \
        --model_ports "${VLLM_PORT}" \
        --model_id "${SERVED_MODEL_NAME}" \
        --model_path "${LOCAL_MODEL_DIR}" \
        --record_path "${RECORD_ROOT}" \
        --max_steps_num "${MAX_STEPS_NUM}" \
        --maximum_history_length "${MAXIMUM_HISTORY_LENGTH}" \
        --task "${TASK}" \
        --difficulty "${DIFFICULTY}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --top_k "${TOP_K}" \
        --fps "${FPS}" \
        --gpu_per_rollout "${GPU_PER_ROLLOUT}" \
        --num_rollouts "${ROLLOUTS_PER_TASK}" \
        2>&1 | tee -a "${LOG_DIR}/rollout_${TASK//[:,]/_}.log"
done

# ---------------------------------------------------------------------------
# 6. 汇总成功率
# ---------------------------------------------------------------------------
echo "=============================================================="
echo "[summary] aggregating results for ${MODEL_LOCAL_NAME}"
python "${REPO_ROOT}/examples/eval_backbones/aggregate_results.py" \
    --record_path "${RECORD_ROOT}" \
    --model_name "${MODEL_LOCAL_NAME}" \
    --output_json "${RECORD_ROOT}/summary.json"
cat "${RECORD_ROOT}/summary.json"

# ---------------------------------------------------------------------------
# 7. 停止 vLLM，回传结果到 S3
# ---------------------------------------------------------------------------
kill "${VLLM_PID}" 2>/dev/null || true

echo "[upload] syncing results -> ${RESULT_S3_URI}"
aws s3 sync "${RECORD_ROOT}/" "${RESULT_S3_URI}/" --no-progress

echo "[done] ${MODEL_LOCAL_NAME} evaluation finished."
