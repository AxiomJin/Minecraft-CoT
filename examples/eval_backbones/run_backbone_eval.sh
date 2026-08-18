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
# EVAL_BENCHMARK 决定评测规模，用于和原论文(OpenHA, arXiv:2509.13347)的
# benchmark 对齐：
#   smoke (默认) - 3个任务、difficulty=zero 的快速烟雾测试，不对应论文任何数值
#   mini         - 30个代表性任务(Embodied/GUI/Combat各10个，easy/middle/hard均分)，
#                  对应论文 Table 3 的 "mini benchmark"/代表性子集评测规模
#   full         - 全部 800+ 个任务(单一difficulty=normal)，对应论文完整benchmark
# mini/full 模式下会用 build_task_list.py 自动生成 TASK_DIFFICULTY_LIST(逐任务
# 独立难度)，取代下面的 TASK_LIST + 全局 DIFFICULTY 组合。手动设置了
# TASK_LIST或TASK_DIFFICULTY_LIST 的话，其优先级更高（用于调试单个任务）。
export EVAL_BENCHMARK=${EVAL_BENCHMARK:-smoke}
export TASK_LIST=${TASK_LIST:-"mine_block:oak_log kill_entity:sheep craft_item:crafting_table"}
export TASK_DIFFICULTY_LIST=${TASK_DIFFICULTY_LIST:-}
case "${EVAL_BENCHMARK}" in
    mini)
        export ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK:-10}
        ;;
    full)
        export ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK:-3}
        ;;
    *)
        export ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK:-5}
        ;;
esac
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

# mini/full 模式下自动生成逐任务差异化难度的 TASK_DIFFICULTY_LIST（除非已手动
# 指定 TASK_DIFFICULTY_LIST，那样直接尊重用户手动给的列表）。仅依赖标准库，
# 不需要等 conda env 装好。
if [ "${EVAL_BENCHMARK}" != "smoke" ] && [ -z "${TASK_DIFFICULTY_LIST}" ]; then
    echo "[task-list] generating EVAL_BENCHMARK=${EVAL_BENCHMARK} task list via build_task_list.py"
    export TASK_LIST_MANIFEST="${RECORD_ROOT}/task_list_manifest.json"
    TASK_DIFFICULTY_LIST="$(python3 "$(dirname "$0")/build_task_list.py" --scope "${EVAL_BENCHMARK}")"
    export TASK_DIFFICULTY_LIST
    echo "[task-list] $(echo "${TASK_DIFFICULTY_LIST}" | wc -w) tasks, manifest -> ${TASK_LIST_MANIFEST}"
fi

echo "=============================================================="
echo "[eval] model=${MODEL_LOCAL_NAME} served_name=${SERVED_MODEL_NAME} vllm_env=${VLLM_CONDA_ENV}"
echo "[eval] benchmark=${EVAL_BENCHMARK} rollouts_per_task=${ROLLOUTS_PER_TASK} max_steps=${MAX_STEPS_NUM}"
if [ -n "${TASK_DIFFICULTY_LIST}" ]; then
    echo "[eval] tasks(task@difficulty, $(echo "${TASK_DIFFICULTY_LIST}" | wc -w) total)=${TASK_DIFFICULTY_LIST}"
else
    echo "[eval] tasks=${TASK_LIST} difficulty=${DIFFICULTY}"
fi
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
# 注意：pip从 PyPI 装 sam2 的依赖(如iopath)时偶发 502(files.pythonhosted.org 抖动)，
# 之前这里没有重试也没有校验安装结果，网络一抖就导致每个 rollout 一启动就
# `ModuleNotFoundError: No module named 'sam2'`，被 rollout_wrapper 的 try/except
# 静默吞掉——外层 job 显示 Succeeded，但 summary.json 里 per_task 是空的、什么都
# 没跑出来（实测于 koala job axiomjin-eval-ckpt{400,600,820}-fix2-*-20260818）。
# 加重试，且最终仍失败要真正 exit 1，不能悄悄跑出一个"成功但是空"的job。
if ! python -c "from sam2.build_sam import build_sam2_camera_predictor" >/dev/null 2>&1; then
    echo "[setup] installing sam2 (zhwang4ai/SAM2 fork, for base.py import compatibility)"
    for attempt in 1 2 3 4 5; do
        pip install -q "git+https://github.com/zhwang4ai/SAM2.git" && break
        echo "[retry] sam2 pip install failed (attempt ${attempt}/5), retrying in 15s..." >&2
        sleep 15
    done
    if ! python -c "from sam2.build_sam import build_sam2_camera_predictor" >/dev/null 2>&1; then
        echo "[setup][FATAL] sam2 install/import still failing after retries, aborting (would otherwise silently produce 0 rollout results)" >&2
        exit 1
    fi
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

# Checkpoints saved by trl_sft/train_sft.py's transformers>=5 write a
# `extra_special_tokens` (dict-typed attribute-name -> token map) into
# tokenizer_config.json. The much older transformers pinned in this eval
# conda env doesn't understand that schema and unconditionally does
# `special_tokens.keys()` on it in `_set_model_specific_special_tokens()`,
# crashing with `AttributeError: 'list' object has no attribute 'keys'`
# the moment vllm serve tries to load the tokenizer -- even though the
# field is redundant metadata (the tokens it lists are already fully
# defined via tokenizer.json/added_tokens_decoder, so dropping it doesn't
# change tokenization behavior at all; the officially released
# minecraft-textvla-qwen2vl-7b-2509 checkpoint never had this field and
# loads fine). Strip it so any checkpoint trained with a newer
# transformers can still be served here without a manual fix each time.
TOKENIZER_CFG="${LOCAL_MODEL_DIR}/tokenizer_config.json"
if [ -f "${TOKENIZER_CFG}" ] && python3 -c "
import json, sys
with open('${TOKENIZER_CFG}') as f:
    sys.exit(0 if 'extra_special_tokens' in json.load(f) else 1)
"; then
    echo "[compat] removing incompatible 'extra_special_tokens' from tokenizer_config.json"
    python3 -c "
import json
p = '${TOKENIZER_CFG}'
with open(p) as f:
    cfg = json.load(f)
cfg.pop('extra_special_tokens', None)
with open(p, 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
"
fi

# Same story for the image processor config: newer transformers' Qwen2VLProcessor
# saves a single merged `processor_config.json` (nesting image_processor/
# video_processor sub-dicts), but this eval env's older transformers/vllm only
# know how to load the legacy flat `preprocessor_config.json` (min_pixels,
# max_pixels, patch_size, ... at the top level) -- without it vllm's image
# processor loading fails with `OSError: ... does not appear to have a file
# named preprocessor_config.json`. Derive one from processor_config.json's
# image_processor block when it's missing (the officially released
# minecraft-textvla-qwen2vl-7b-2509 checkpoint ships preprocessor_config.json
# directly and needs no fix).
PREPROCESSOR_CFG="${LOCAL_MODEL_DIR}/preprocessor_config.json"
PROCESSOR_CFG="${LOCAL_MODEL_DIR}/processor_config.json"
if [ ! -f "${PREPROCESSOR_CFG}" ] && [ -f "${PROCESSOR_CFG}" ]; then
    echo "[compat] deriving preprocessor_config.json from processor_config.json"
    python3 -c "
import json
src = json.load(open('${PROCESSOR_CFG}'))
ip = src['image_processor']
out = {
    'min_pixels': ip['size']['shortest_edge'],
    'max_pixels': ip['size']['longest_edge'],
    'patch_size': ip['patch_size'],
    'temporal_patch_size': ip['temporal_patch_size'],
    'merge_size': ip['merge_size'],
    'image_mean': ip['image_mean'],
    'image_std': ip['image_std'],
    'image_processor_type': ip['image_processor_type'],
    'processor_class': src['processor_class'],
}
json.dump(out, open('${PREPROCESSOR_CFG}', 'w'), indent=2)
"
fi
# The merged `processor_config.json` itself is still a problem even after the
# derivation above (which only *adds* preprocessor_config.json, it doesn't remove
# the original file): its `video_processor` sub-dict is a plain JSON object with a
# `video_processor_type` key, a schema this eval env's older transformers doesn't
# know how to instantiate into a `BaseVideoProcessor`. `ProcessorMixin.from_pretrained`
# prefers `processor_config.json` over `preprocessor_config.json` when both exist, so
# it still loads the merged file, tries to build `Qwen2VLProcessor(image_processor,
# tokenizer, video_processor, ...)` with the raw dict as `video_processor`, and crashes
# with `TypeError: Received a dict for argument video_processor, but a
# BaseVideoProcessor was expected.` at vllm server startup (confirmed in koala jobs
# axiomjin-eval-ckpt{400,600,820}* on 2026-08-18, all failed ~3-7min in). The
# officially released minecraft-textvla-qwen2vl-7b-2509 checkpoint never had a
# processor_config.json (only the legacy flat preprocessor_config.json) and loads
# fine, so move the merged file out of the way once we no longer need it as a
# derivation source -- everything the old code path needs is already in
# preprocessor_config.json/tokenizer_config.json.
if [ -f "${PROCESSOR_CFG}" ]; then
    echo "[compat] moving aside processor_config.json (incompatible video_processor schema for this eval env)"
    mv "${PROCESSOR_CFG}" "${PROCESSOR_CFG}.bak"
fi

# The model's own config.json has the exact same "newer transformers nests
# everything" problem: transformers>=4.54's Qwen2VLConfig now stores all
# text-backbone fields (vocab_size/hidden_size/num_hidden_layers/rope_theta/...)
# inside a nested `text_config` sub-dict instead of flat on the top-level object.
# `Qwen2VLConfig.__init__` only setattr's flat kwargs onto `self` via
# `super().__init__(**kwargs)` -- fields consumed into `self.text_config` don't
# also become top-level attributes. But vllm==0.8.5's `Qwen2Model.__init__`
# (reused as the language-model backbone inside Qwen2VLForConditionalGeneration)
# does plain flat attribute access like `config.vocab_size` / `config.hidden_size`
# / `config.rope_scaling`, so with a nested-only config it crashes with
# `AttributeError: 'Qwen2VLConfig' object has no attribute 'vocab_size'` right as
# the model weights start loading (confirmed in koala job
# axiomjin-eval-ckpt820-fix-normal-20260818-153150, after the processor_config.json
# fix above got it past the earlier vllm-serve-startup crash). The officially
# released minecraft-textvla-qwen2vl-7b-2509 checkpoint ships the old flat
# schema (text fields directly at top level, only `vision_config` nested) and
# loads fine. Promote every field out of `text_config` back onto the top level
# (without clobbering any same-named top-level key that's already correct, e.g.
# top-level `model_type`/`pad_token_id` must stay "qwen2_vl"/151643, not the
# text-only "qwen2_vl_text"/null shadowed inside `text_config`), and translate
# `rope_parameters` (new key) back into the old `rope_theta` + `rope_scaling`
# pair vllm's decoder layers expect. `text_config`/`vision_config` are left in
# place too -- transformers itself still parses them fine, this only *adds* the
# flat duplicates vllm needs.
MODEL_CFG="${LOCAL_MODEL_DIR}/config.json"
if [ -f "${MODEL_CFG}" ] && python3 -c "
import json, sys
sys.exit(0 if 'text_config' in json.load(open('${MODEL_CFG}')) else 1)
"; then
    echo "[compat] flattening config.json's nested text_config onto top level (vllm 0.8.5 needs flat vocab_size/hidden_size/rope_scaling/...)"
    python3 -c "
import json
p = '${MODEL_CFG}'
cfg = json.load(open(p))
tc = dict(cfg.get('text_config') or {})
rope = tc.pop('rope_parameters', None)
if rope:
    tc.setdefault('rope_theta', rope.get('rope_theta', 1000000.0))
    rope_type = rope.get('type') or rope.get('rope_type') or 'mrope'
    tc.setdefault('rope_scaling', {'type': rope_type, 'mrope_section': rope.get('mrope_section')})
for k, v in tc.items():
    cfg.setdefault(k, v)
json.dump(cfg, open(p, 'w'), indent=2)
"
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
#    优先用 TASK_DIFFICULTY_LIST（"task@difficulty" token序列，逐任务独立难度，
#    由 mini/full benchmark 模式自动生成）；未设置时回退到旧的
#    TASK_LIST + 全局 DIFFICULTY 组合（用于 smoke 模式 / 手动调试单任务）。
# ---------------------------------------------------------------------------
conda activate openha
cd "${REPO_ROOT}"
if [ -n "${TASK_DIFFICULTY_LIST}" ]; then
    for TASK_DIFF in ${TASK_DIFFICULTY_LIST}; do
        TASK="${TASK_DIFF%@*}"
        TASK_DIFFICULTY="${TASK_DIFF##*@}"
        echo "------------------------------------------------------------"
        echo "[rollout] task=${TASK} difficulty=${TASK_DIFFICULTY} num_rollouts=${ROLLOUTS_PER_TASK}"
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
            --difficulty "${TASK_DIFFICULTY}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --top_k "${TOP_K}" \
            --fps "${FPS}" \
            --gpu_per_rollout "${GPU_PER_ROLLOUT}" \
            --num_rollouts "${ROLLOUTS_PER_TASK}" \
            2>&1 | tee -a "${LOG_DIR}/rollout_${TASK//[:,]/_}.log"
    done
else
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
fi

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

# 防止"静默失败"：rollout_wrapper 内部会 catch 所有异常只打印不上抛（例如某个
# 依赖 import 失败/环境初始化失败），导致这个 job 在 shell 层面看起来正常跑完、
# 状态显示 Succeeded，但实际一条有效 rollout 结果都没有产生。汇总后如果总数为0，
# 在这里才真正让 job 失败退出，避免误判"评测已完成"。
OVERALL_TOTAL=$(python3 -c "import json; print(json.load(open('${RECORD_ROOT}/summary.json'))['overall']['total'])")
if [ "${OVERALL_TOTAL}" = "0" ]; then
    echo "[summary][FATAL] 0 rollout results produced (overall.total=0) -- likely a silent setup/import failure upstream, check ${LOG_DIR}/ for tracebacks" >&2
    kill "${VLLM_PID}" 2>/dev/null || true
    aws s3 sync "${RECORD_ROOT}/" "${RESULT_S3_URI}/" --no-progress || true
    exit 1
fi

# ---------------------------------------------------------------------------
# 7. 停止 vLLM，回传结果到 S3
# ---------------------------------------------------------------------------
kill "${VLLM_PID}" 2>/dev/null || true

echo "[upload] syncing results -> ${RESULT_S3_URI}"
aws s3 sync "${RECORD_ROOT}/" "${RESULT_S3_URI}/" --no-progress

echo "[done] ${MODEL_LOCAL_NAME} evaluation finished."
