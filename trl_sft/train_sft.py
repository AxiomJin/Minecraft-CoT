"""
TRL-based SFT training for Minecraft text-action VLM.

Data (two supported layouts, see `build_minecraft_dataset`):
  - parquet trajectories: s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset/
  - jsonl flat QA (e.g. minecraft-vlp): s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/
Model: s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B/ (also works for Qwen2-VL /
    Qwen2.5-VL / Qwen3-VL under s3://arcwm-code-us-west-2/axiom/model/)

Usage:
    torchrun --nproc_per_node=$NPROC train_sft.py \
        --model_path s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B \
        --data_path s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset/data/train-*.parquet \
        --output_dir ./output \
        --max_turns 4 \
        --max_seq_length 16384 \
        --per_device_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 1 \
        --deepspeed ds_zero2.json

    # jsonl / flat-QA layout (e.g. minecraft-vlp), --data_format auto-detected from the
    # ".jsonl" extension; --image_root defaults to the directory containing --data_path:
    torchrun --nproc_per_node=$NPROC train_sft.py \
        --model_path s3://arcwm-code-us-west-2/axiom/model/Qwen2-VL-7B-Instruct \
        --data_path s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-vqa-241102.jsonl \
        --output_dir ./output

    # Stage I (JARVIS-VLA "Minecraft world knowledge" text-only post-training): rows
    # are plain system+user+assistant text QA with no images at all (e.g.
    # minecraft-vlp/mc-qa-*.jsonl, label=["qa","wiki","self-instruct"]). --text_only
    # makes samples carry no "images" key (so SFTTrainer uses its plain-text collator,
    # not the vision one) and --freeze_vision_tower keeps the ViT+adapter frozen while
    # only the LLM backbone is updated. Hyperparameters below match JARVIS-VLA's paper
    # recipe as closely as feasible (LR=5e-6, AdamW beta2=0.95/wd=0, fixed 200-step
    # warmup, grad-norm clip 1.0, global batch=256, DeepSpeed ZeRO-1, max_length=3584,
    # seed=42) -- EXCEPT for two deliberate deviations:
    #   (a) --gradient_accumulation_steps is scaled up from the paper's 4 to compensate
    #       for using 8 GPUs here instead of the paper's 32 (2*16*8 = 2*4*32 = 256, so
    #       the *effective* global batch is identical, training just takes longer
    #       wall-clock);
    #   (b) ds_zero1.json defines an EXPLICIT DeepSpeed-native "WarmupDecayLR" scheduler
    #       with HARDCODED warmup_min_lr=0/warmup_max_lr=5e-6/warmup_num_steps=200/
    #       total_num_steps=1077 (matching --learning_rate/--warmup_steps/--max_steps
    #       below) rather than "auto"-filling those or letting --lr_scheduler_type=cosine
    #       run as a plain HF scheduler under DeepSpeed. Both alternatives were tried
    #       and empirically failed on this exact stack (transformers+deepspeed+
    #       grad-accum=16): the plain-HF-scheduler-under-DeepSpeed path completed warmup
    #       ~5-16x faster than the requested 200 steps, and "auto"-filling
    #       scheduler.params.warmup_num_steps resolved via --warmup_ratio (not
    #       --warmup_steps) and crashed with warmup_num_steps=0 once --warmup_ratio was
    #       forced to 0 to disambiguate the first issue. Hardcoding the scheduler section
    #       sidesteps both: reuse this ds_zero1.json for a *different* max_steps/lr/
    #       warmup_steps combination requires updating it to match. The one remaining
    #       deviation from the paper this introduces: WarmupDecayLR decays LINEARLY
    #       after warmup, not via the paper's cosine schedule.
    torchrun --nproc_per_node=8 train_sft.py \
        --model_path s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B \
        --data_path s3://arcwm-code-us-west-2/axiom/data/minecraft-vlp/mc-qa-250312.jsonl \
        --text_only \
        --freeze_vision_tower \
        --max_seq_length 3584 \
        --per_device_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --num_train_epochs 1 \
        --max_steps 1077 \
        --learning_rate 5e-6 \
        --lr_scheduler_type cosine \
        --warmup_steps 200 \
        --weight_decay 0.0 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-8 \
        --max_grad_norm 1.0 \
        --seed 42 \
        --deepspeed ds_zero1.json \
        --output_dir ./output

NOTE on VLM + TRL:
  - All target models (Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL) are vision-language
    models. TRL's `SFTTrainer` picks the vision-language data collator
    (`DataCollatorForVisionLanguageModeling`) only when a sample dict has a top-level
    "images" (or "image") key.
      * `packing=True` is not supported for VLMs by TRL and will raise at trainer init time.
      * `assistant_only_loss=True` is ALSO not supported for vision datasets -- TRL raises
        `ValueError` for it at trainer init time (`DataCollatorForVisionLanguageModeling`
        has no `assistant_masks` handling at all, it only masks padding).
  - Loss masking strategy actually used here: each sample is split into
    {"prompt": [...history/context...], "completion": [last assistant turn only],
    "images": [...]} (TRL's *conversational prompt-completion* format) and
    `SFTConfig(completion_only_loss=True)` is set below. This IS supported for VLMs
    (`_collate_prompt_completion`) and gives exactly what we want: loss is only computed
    on the target "Action: ..." turn, never on the system prompt / history / image tokens.
"""

from __future__ import annotations

import argparse
import faulthandler
import io
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import concatenate_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)


# ─── hang diagnostics ──────────────────────────────────────────────────────────
#
# A multi-node hang surfaces only as NCCL's `Watchdog caught collective operation
# timeout ... ran for 600000ms`, which says *that* some ranks stopped making
# progress but never *where* in Python they are blocked -- so the root cause
# (data loading? tokenization? a forward op? an actual collective mismatch?)
# stays guesswork. The watchdog below closes that gap: it notices when steps stop
# advancing and dumps every thread's traceback (and its dataloader workers')
# straight to stderr, i.e. into the job log, well before NCCL aborts the process.

_HEARTBEAT: Dict[str, Union[float, int, str]] = {"t": 0.0, "step": -1, "phase": "startup"}


def _rank_tag() -> str:
    return f"rank={os.environ.get('RANK', '?')} local_rank={os.environ.get('LOCAL_RANK', '?')}"


def _child_pids(pid: int) -> List[int]:
    """Direct child PIDs (Linux), used to reach forked dataloader workers."""
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r") as fh:
            return [int(p) for p in fh.read().split()]
    except OSError:
        return []


def _dump_all_stacks(reason: str) -> None:
    """Dump this process' thread stacks, then ask dataloader workers to dump theirs."""
    tag = _rank_tag()
    print(f"\n===== [stall-watchdog] {tag} {reason} =====", file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    for child in _child_pids(os.getpid()):
        print(f"===== [stall-watchdog] {tag} dumping child pid={child} =====", file=sys.stderr, flush=True)
        try:
            # Handler registered in _install_stall_watchdog() and inherited on fork.
            os.kill(child, signal.SIGUSR1)
        except OSError as exc:
            print(f"[stall-watchdog] cannot signal pid={child}: {exc}", file=sys.stderr, flush=True)
    time.sleep(1.0)  # let children flush their tracebacks before we return
    sys.stderr.flush()


def _install_stall_watchdog(threshold_sec: float, max_dumps: int = 3) -> None:
    """Start a daemon thread that dumps Python stacks when training stalls.

    `threshold_sec` should sit comfortably below the NCCL collective timeout
    (default 600s) so the stacks are captured while the process is still alive.
    """
    if threshold_sec <= 0:
        return

    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
    except (AttributeError, ValueError, OSError):  # platform without SIGUSR1 support
        pass

    _HEARTBEAT["t"] = time.time()

    def _loop() -> None:
        dumps, dumped_for_step = 0, None
        while True:
            time.sleep(min(30.0, max(5.0, threshold_sec / 4)))
            age = time.time() - float(_HEARTBEAT["t"])
            if age < threshold_sec:
                continue
            if _HEARTBEAT["step"] != dumped_for_step:  # a fresh stall -> dump again
                dumps, dumped_for_step = 0, _HEARTBEAT["step"]
            if dumps >= max_dumps:
                continue
            dumps += 1
            _dump_all_stacks(
                f"no progress for {age:.0f}s (step={_HEARTBEAT['step']} "
                f"phase={_HEARTBEAT['phase']}, dump {dumps}/{max_dumps})"
            )

    threading.Thread(target=_loop, name="stall-watchdog", daemon=True).start()
    logger.info(f"[stall-watchdog] armed: dumping Python stacks after {threshold_sec:.0f}s without step progress")


class _HeartbeatCallback(TrainerCallback):
    """Feeds the stall watchdog, so a hang can be pinned to a specific step/phase."""

    def _beat(self, state, phase: str) -> None:
        _HEARTBEAT.update(t=time.time(), step=state.global_step, phase=phase)

    def on_train_begin(self, args, state, control, **kwargs):
        self._beat(state, "train_begin")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._beat(state, "step_begin")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._beat(state, "step_end")
        return control

    def on_save(self, args, state, control, **kwargs):
        self._beat(state, "save")
        return control


# ─── s3 helpers ────────────────────────────────────────────────────────────────


def _local_cache_name(s3_or_local_path: str) -> str:
    """Derive a filesystem-safe, model-specific cache dir name from a path."""
    name = s3_or_local_path.rstrip("/").split("/")[-1]
    return name or "model"


def download_from_s3(s3_path: str, local_dir: str, exclude_checkpoints: bool = False) -> str:
    """Download model/dataset from S3 to local disk, skip if already exists.

    `exclude_checkpoints=True` skips any `checkpoint-*/` subdirectory. This matters when
    `s3_path` is a model dir produced by a PREVIOUS run of this script used as the
    starting point for a later stage (e.g. loading a Stage I output dir to start Stage
    II): `--output_dir` (and hence the uploaded model dir) contains both the final
    merged model at the root (config.json/model.safetensors/tokenizer.*, all that
    `from_pretrained` ever reads) AND every intermediate `--save_steps` checkpoint
    (each one a full DeepSpeed ZeRO checkpoint with fp32 optimizer state -- for a 9B
    model that's ~5x the plain model size, PER checkpoint, and `--save_total_limit`
    keeps up to 5 of them). Downloading those checkpoint dirs alongside the root is
    pure waste: verified on the real Qwen3.5-9B-stage1-16gpu output (475GB total, only
    18.8GB of it is the root-level files `from_pretrained` will actually load; the
    other 456GB is 5 checkpoint-*/ dirs full of *_optim_states.pt / mp_rank_*.pt
    shards). Harmless / a no-op for base model dirs that don't have any checkpoint-*
    subdirs in the first place.
    """
    local_dir = Path(local_dir)
    marker = local_dir / ".download_complete"

    if marker.exists():
        logger.info(f"Already downloaded: {local_dir}")
        return str(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {s3_path} -> {local_dir} ...")
    exclude_flag = " --exclude 'checkpoint-*/*'" if exclude_checkpoints else ""
    ret = os.system(f"aws s3 cp --recursive{exclude_flag} {s3_path} {local_dir}/")
    if ret != 0:
        raise RuntimeError(f"aws s3 cp failed (exit={ret}) for {s3_path} -> {local_dir}")
    marker.touch()
    logger.info(f"Download complete: {local_dir}")
    return str(local_dir)


# ─── dataset helpers ──────────────────────────────────────────────────────────


def build_messages(
    conversations: list,
    image_bytes_list: list,
    max_turns: int,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[List[Image.Image]]]:
    """
    Convert a single parquet row into TRL's conversational *prompt-completion* format:
    (prompt, completion, images).

    Only the LAST assistant turn is the actual training target ("Action: ..."); every
    turn before it (system prompt / history / current-step image+instruction) is context
    that must NOT contribute to the loss. Splitting into prompt/completion (rather than a
    single flat `messages` list) lets `SFTConfig(completion_only_loss=True)` mask the
    context out -- this is the only loss-masking mechanism TRL currently supports for
    vision-language datasets (`assistant_only_loss` raises `ValueError` for VLMs).

    Args:
        conversations: list of {role, content[{type, text/image}]}
        image_bytes_list: list of JPEG bytes, one per user turn (with image)
        max_turns: maximum number of (user, assistant) pairs to include
        rng: `random.Random` instance to draw the history-length sample from. Defaults to
            the global `random` module (fine for single-threaded/debug use, but callers
            with per-sample determinism needs -- e.g. `_row_to_trl_sample` -- should
            pass their own local instance instead of relying on/mutating global state).

    Returns:
        (prompt, completion, images):
          - prompt: all turns except the final assistant turn.
          - completion: single-element list containing only the final assistant message.
          - images: flat, ordered list of decoded `PIL.Image` objects matching the
            `{"type": "image"}` placeholders inside `prompt` (this must be passed as the
            top-level "images" key of the dataset sample, NOT embedded in `content`).
    """
    rng = rng if rng is not None else random
    if not conversations or len(conversations) < 2:
        return None, None, None

    # Some rows may start with a stray non-user turn (seen in the original VeOmni
    # preprocessing); drop it so the user/assistant pairing below stays aligned.
    if conversations[0]["role"] != "user":
        conversations = conversations[1:]
    if not conversations or conversations[-1]["role"] != "assistant":
        return None, None, None

    # Count total turns
    total_turns = len(conversations) // 2

    # Random history length: 0 to min(total_turns-1, max_turns)
    max_possible = min(total_turns - 1, max_turns)
    if max_possible < 0:
        return None, None, None
    history_len = rng.randint(0, max_possible)

    # Take the last (history_len + 1) turns
    start_turn = total_turns - (history_len + 1)
    start_idx = start_turn * 2

    selected_convs = conversations[start_idx:]
    # NOTE: this slices `image_bytes_list` by *pair index* (`start_turn`), which matches
    # `minecraft-text-action-dataset`'s parquet schema: exactly one `image_bytes` entry
    # per (user, assistant) trajectory step, regardless of whether that step's user turn
    # actually contains an image placeholder. This is different from the FLAT,
    # encounter-order indexing that `_split_prompt_completion_with_images` below uses for
    # the actual placeholder <-> bytes matching -- see `build_messages_qa` for a dataset
    # format (`minecraft-vlp`) where that flat/no-slicing behavior is what's needed
    # instead.
    selected_image_bytes = image_bytes_list[start_turn:] if image_bytes_list else []

    return _split_prompt_completion_with_images(selected_convs, selected_image_bytes)


def _split_prompt_completion_with_images(
    conversations: List[Dict],
    image_bytes_list: List[bytes],
) -> Tuple[List[Dict], List[Dict], List[Image.Image]]:
    """
    Shared tail used by both `build_messages` (trajectory/parquet rows) and
    `build_messages_qa` (flat QA/jsonl rows): walks `conversations`, keeps
    `{"type": "image"}` placeholders in-place (to preserve text/image interleaving
    order) while decoding the matching entry of the FLAT `image_bytes_list` (one entry
    per placeholder, in encounter order across the whole `conversations` list -- NOT
    per-turn) into a separate flat `images` list. Finally splits off the final
    assistant turn as `completion`; everything before it becomes `prompt` (context that
    `SFTConfig(completion_only_loss=True)` will mask out of the loss).
    """
    messages = []
    images: List[Image.Image] = []
    image_idx = 0

    for conv in conversations:
        role = conv["role"]
        content_list = []

        for item in conv["content"]:
            if item.get("type") == "text":
                content_list.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                if image_idx < len(image_bytes_list):
                    try:
                        img = Image.open(io.BytesIO(image_bytes_list[image_idx])).convert("RGB")
                        content_list.append({"type": "image"})
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to decode image at idx {image_idx}: {e}")
                        content_list.append({"type": "text", "text": "[image]"})
                image_idx += 1
            elif item.get("type") == "point":
                # Grounding rows (e.g. minecraft-vlp/mc-grounding-point-*.jsonl, JARVIS-VLA
                # Stage II spatial-grounding data): the assistant turn's answer is
                # structured `{"point": [[x, y], ...], "label": "..."}` -- x/y are
                # percentages (0-100) of image width/height, not pixels, and a single
                # turn can list multiple points (e.g. "Spot the 2 slot" -> up to ~10
                # points for repeated/ambiguous targets). There is no free-form "text"
                # answer to fall back on, so serialize the coordinates into plain text
                # the LM can actually be trained to generate: "(x, y)" per point,
                # multiple points joined with "; ". This keeps the format simple/
                # deterministic and resolution-independent (matches the 0-100 percentage
                # scale already used by the raw labels).
                points = item.get("point") or []
                coord_text = "; ".join(f"({x:.2f}, {y:.2f})" for x, y in points)
                content_list.append({"type": "text", "text": coord_text})

        # TRL expects roles: "user", "assistant", "system"
        messages.append({"role": role, "content": content_list})

    # Split off the final assistant turn as the `completion`; everything before it
    # becomes the `prompt` (context that `completion_only_loss` will mask out).
    prompt, completion = messages[:-1], [messages[-1]]

    # Defensive invariant check: TRL's collator (`trl.data_utils.prepare_multimodal_messages`,
    # called on `example["prompt"]`/`example["images"]` inside `SFTTrainer`'s
    # `torch_call`) requires the number of `{"type": "image"}` placeholders WITHIN
    # `prompt` to exactly equal `len(images)` (the flat image list accumulated above
    # across the WHOLE conversation, prompt- and completion-side alike) -- otherwise it
    # raises `ValueError: Number of images provided (N) does not match number of image
    # placeholders (M)`. Observed in practice: this crashed two separate live 16-GPU
    # Stage II jobs ~13h in, once a `DataLoader` worker's shuffled iteration order
    # finally landed on the one bad row out of hundreds of thousands. Root cause is a
    # small minority of mislabeled rows in the underlying jsonl data where an image
    # placeholder ends up in the FINAL (assistant) turn -- which becomes `completion`,
    # not `prompt` -- so that image's matching placeholder is invisible to `prompt`'s
    # count even though `images` (built across the whole conversation) still includes
    # it. Checking this invariant up front, at dataset-construction time, converts a
    # "crash the whole distributed job hours into training" failure into a "silently
    # drop this one malformed row" outcome instead.
    num_prompt_placeholders = sum(
        1 for m in prompt for item in m["content"] if item.get("type") == "image"
    )
    if num_prompt_placeholders != len(images):
        logger.warning(
            f"Dropping a sample where prompt-side image placeholders ({num_prompt_placeholders}) "
            f"!= total images ({len(images)}) -- likely an image placeholder landed in the final "
            f"(completion) turn. See this function's docstring/comment for why this is unsafe."
        )
        return None, None, []
    return prompt, completion, images


def _read_bytes(uri: str) -> bytes:
    """Read raw bytes from a local path or any `fsspec`-supported URI (e.g. `s3://...`,
    via the `s3fs` dependency)."""
    import fsspec

    with fsspec.open(uri, "rb") as f:
        return f.read()


def build_messages_qa(
    conversations: list,
    image_paths: list,
    image_root: str,
) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[List[Image.Image]]]:
    """
    Convert one JSONL "flat QA" row (e.g. `minecraft-vlp/*.jsonl`) into the same TRL
    prompt/completion/images format as `build_messages`, but WITHOUT the
    trajectory-specific history-window sampling: these rows are short, independent
    multi-turn Q&A sessions about a small, fixed set of images (declared once in the
    row's "image" field, referenced by whichever `{"type": "image"}` placeholder(s)
    appear anywhere in `conversations`, in encounter order) rather than a long
    sequential trajectory of (image, action) steps -- so there is no "history length"
    to randomly truncate. Keeping the whole conversation also guarantees any image
    placeholder (almost always in the first user turn) always stays inside the
    resulting `prompt` instead of possibly being sliced away.

    Args:
        conversations: list of {role, content[{type, text/image}]}, same schema as
            `build_messages`.
        image_paths: list of paths *relative to `image_root`* (as stored in the row's
            "image" field), one entry per `{"type": "image"}` placeholder in encounter
            order across the whole conversation.
        image_root: directory (local path or any `fsspec` URI, e.g.
            "s3://bucket/prefix") that `image_paths` entries are relative to -- normally
            the directory containing the jsonl file itself; see
            `build_minecraft_dataset`/`_default_image_root`.
    """
    if not conversations or len(conversations) < 2:
        return None, None, None

    if conversations[0]["role"] != "user":
        conversations = conversations[1:]
    if not conversations or conversations[-1]["role"] != "assistant":
        return None, None, None

    image_bytes_list: List[bytes] = []
    for rel_path in image_paths or []:
        uri = f"{image_root.rstrip('/')}/{str(rel_path).lstrip('/')}"
        try:
            image_bytes_list.append(_read_bytes(uri))
        except Exception as e:
            logger.warning(f"Failed to read image {uri}: {e}")
            # Keep a (deliberately undecodable) placeholder so the flat index stays in
            # sync with the placeholders in `conversations`; it will simply fail to
            # decode below and fall back to a "[image]" text placeholder.
            image_bytes_list.append(b"")

    return _split_prompt_completion_with_images(conversations, image_bytes_list)


def build_messages_text_only(conversations: list) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
    """
    Convert one pure-text row (e.g. `minecraft-vlp/mc-qa-*.jsonl`'s
    `label=["qa","wiki","self-instruct"]` rows: a leading `system` turn + a `user`
    question + an `assistant` answer, `image=[]`) into TRL prompt/completion format,
    for JARVIS-VLA's Stage I ("Minecraft world knowledge" text-only post-training --
    see `--text_only`/`--freeze_vision_tower`).

    Unlike `build_messages`/`build_messages_qa`, this does NOT strip a leading
    non-"user" turn: that stripping exists in those two to drop a stray artifact turn
    seen in trajectory/VQA preprocessing, but here a leading turn is normally a
    legitimate system prompt ("You are a helpful assistant...") that must be preserved
    inside `prompt`, not discarded.

    Returns `(prompt, completion)` -- no `images` list, since this data format never
    has any (see `_row_to_trl_sample`/`build_minecraft_dataset` for why the caller must
    omit the "images" key entirely from the resulting sample dict rather than passing
    an empty list, so `SFTTrainer` picks its plain-text collator).

    Any `{"type": "image"}` placeholder encountered is unexpected for this data format
    and gets replaced with a "[image]" text stand-in (with a warning) rather than
    silently producing an inconsistent sample -- if you see that warning, the file
    you're pointing --text_only at probably isn't actually text-only.
    """
    if not conversations or len(conversations) < 2:
        return None, None
    if conversations[-1]["role"] != "assistant":
        return None, None

    messages = []
    for conv in conversations:
        content_list = []
        for item in conv["content"]:
            if item.get("type") == "text":
                content_list.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                logger.warning(
                    "build_messages_text_only: found an image placeholder in a "
                    "--text_only row; replacing with a '[image]' text stand-in. This "
                    "data file may not actually be text-only -- consider --text_only=False."
                )
                content_list.append({"type": "text", "text": "[image]"})
        messages.append({"role": conv["role"], "content": content_list})

    prompt, completion = messages[:-1], [messages[-1]]
    return prompt, completion


# TRL's `SFTTrainer` tokenizes `prompt` alone (with `add_generation_prompt=True`) and
# `prompt + completion` together, then checks that the former is a token-for-token
# PREFIX of the latter to compute `completion_mask` (see
# "Mismatch between tokenized prompt..." warning). For any Qwen3.x-family "thinking"
# chat template, leaving `enable_thinking` unset makes these two renderings literally
# different strings at the assistant turn: `add_generation_prompt=True` alone emits an
# UNCLOSED `<think>\n`, while the full prompt+completion render (assistant turn with
# empty `reasoning_content`) emits a CLOSED `<think>\n\n</think>\n\n`. Text-wise the
# former IS a character-prefix of the latter, but the tokenizer is not prefix-stable
# across that specific `\n`+`\n` boundary (verified against the real Qwen3.5-9B
# tokenizer: "<think>\n" tokenizes with a trailing lone `\n` token, but the same
# characters as a prefix of "<think>\n\n</think>\n\n..." get merged into a single
# double-newline token) -- so `len(prompt_ids)` ends up 2 tokens short of the true
# prompt/completion boundary, and those 2 boilerplate tokens (`</think>` + the
# following blank line) leak into the loss on every single sample. Passing
# `enable_thinking=False` explicitly makes `add_generation_prompt=True` ALSO emit the
# closed `<think>\n\n</think>\n\n` form, so both renderings share the exact same
# literal string at the boundary and tokenize identically -- eliminates the mismatch
# (verified with the real tokenizer).
#
# Only actually apply it (see `_resolve_chat_template_kwargs` below) when the loaded
# processor's own chat template references `enable_thinking` at all (Qwen3.x): passing
# it unconditionally to a template that doesn't (e.g. Qwen2-VL/Qwen2.5-VL) still
# renders byte-identically (Jinja silently ignores unused template variables), but
# `transformers`' `apply_chat_template` does its OWN separate check -- via Jinja
# template introspection (`_get_template_variables`) -- to decide which of its
# `**kwargs` are "real" template variables vs. mistakenly-misplaced `processor_kwargs`;
# `enable_thinking` not appearing in that template's variable list makes it match the
# latter and trips `logger.warning("Kwargs passed to \`processor.__call__\` have to be
# in \`processor_kwargs\` dict, not in \`**kwargs\`")` on EVERY `apply_chat_template`
# call (i.e. once per sample, every step) -- purely cosmetic log spam, but avoiding it
# outright is simplest.
_CHAT_TEMPLATE_KWARGS: Dict = {"enable_thinking": False}


def _resolve_chat_template_kwargs(processor) -> Dict:
    """Return `_CHAT_TEMPLATE_KWARGS` only if `processor`'s own chat template actually
    references `enable_thinking` (Qwen3.x "thinking mode" templates); otherwise `{}`.
    See `_CHAT_TEMPLATE_KWARGS`'s comment above for why this avoids a per-sample
    `transformers` log-spam warning on models (e.g. Qwen2-VL/Qwen2.5-VL) whose template
    doesn't use it. Falls back to the unconditional dict if `processor` is `None`
    (callers that don't have one yet can't detect support either way).
    """
    if processor is None:
        return _CHAT_TEMPLATE_KWARGS
    template = getattr(processor, "chat_template", None)
    if isinstance(template, dict):
        template = "\n".join(v for v in template.values() if isinstance(v, str))
    if isinstance(template, str) and "enable_thinking" in template:
        return _CHAT_TEMPLATE_KWARGS
    return {}


def _row_to_trl_sample(
    sample: Dict,
    idx: int,
    max_turns: int,
    seed: int,
    data_format: str,
    image_root: Optional[str],
    text_only: bool = False,
    processor=None,
    max_seq_length: Optional[int] = None,
) -> Dict:
    """
    Map ONE raw dataset row (parquet trajectory step OR jsonl QA session, per
    `data_format`) to a TRL prompt/completion/images sample.

    This is used as the `function` argument of `datasets.Dataset.map` /
    `datasets.IterableDataset.map` (with `with_indices=True`) rather than being wrapped
    in a hand-rolled `torch.utils.data.IterableDataset` subclass. That distinction
    matters: `trl.SFTTrainer.__init__` only accepts a `train_dataset` that
    `isinstance`-checks against `datasets.Dataset` / `datasets.IterableDataset` --
    a custom `torch.utils.data.IterableDataset` iterates fine on its own but is
    rejected with `TypeError` at trainer-construction time. Chaining `.map()` on the
    object returned by `datasets.load_dataset(...)` keeps the result a genuine
    `datasets.Dataset` / `datasets.IterableDataset`, so it passes that check while all
    the sharding/streaming machinery documented in `build_minecraft_dataset` still
    applies unchanged.

    For `data_format == "parquet"` (`minecraft-text-action-dataset`-style trajectory
    rows), uses a *local* RNG keyed by the sample's own stable "id" (not `idx`, the
    stream-position/row index) so that (a) different ranks/workers never happen to
    reuse the same seed for "the k-th sample they each see locally" -- which they
    otherwise would, since after sharding every rank/worker's local stream position
    resets to 0 -- and (b) we don't mutate the global `random` module state as a side
    effect. For `data_format == "jsonl"` (`minecraft-vlp`-style flat QA rows,
    `build_messages_qa`) there is no history sampling, so no RNG is needed.

    If `text_only=True` (Stage I, see `--text_only`), this bypasses `build_messages`/
    `build_messages_qa` entirely in favor of `build_messages_text_only`, and the
    returned dict has NO "images" key at all -- not even an empty list. That distinction
    matters just as much as the `datasets`-type one above: `SFTTrainer` picks its
    vision-language collator purely based on whether an "images"/"image" column is
    *present* on the dataset (regardless of whether it's populated), so omitting the
    key entirely is what makes Stage I run as ordinary text-only LM fine-tuning instead
    of (incorrectly) going through the vision collator with empty image lists.

    Invalid rows are flagged via `"_keep": False` instead of being dropped here (a
    `.map()` function must return exactly one output row per input row); the caller
    chains `.filter(lambda x: x["_keep"])` afterwards to actually drop them.

    If `processor` and `max_seq_length` are both given, image-bearing samples also get
    a `_keep=False` pre-filter based on their REAL tokenized length (see
    `_exceeds_max_length`) -- this guards against a crash that is otherwise silent
    until it kills a whole multi-GPU job partway through training: `SFTConfig`'s
    `max_length`/`max_seq_length` truncation happens at the raw-token level with no
    awareness of where each image's placeholder-token block starts/ends, so an
    oversized multi-image sample (e.g. `mc-grounding-point-embodied-image5.jsonl`'s
    5-image rows) can get truncated *through the middle* of an image's placeholder
    tokens. The VLM's forward pass then finds the (untruncated) vision-tower feature
    count no longer matches the (truncated) placeholder-token count still in
    `input_ids` and raises `ValueError: Image features and image tokens do not
    match, tokens: N, features: M` -- observed in practice for Qwen2-VL-7B on this
    Stage II data.

    For trajectory (parquet) rows the overflow is resolved by *trimming the sampled
    history* until the sample fits, keeping one output sample per input row. Dropping
    such rows instead is what previously hung multi-node training -- see the long
    comment at the trimming loop for the full mechanism.
    """
    sample_id = sample.get("id", idx)
    chat_template_kwargs = _resolve_chat_template_kwargs(processor)
    if text_only:
        prompt, completion = build_messages_text_only(conversations=sample["conversations"])
        if prompt is None:
            return {"prompt": [], "completion": [], "chat_template_kwargs": {}, "_keep": False}
        return {
            "prompt": prompt,
            "completion": completion,
            "chat_template_kwargs": chat_template_kwargs,
            "_keep": True,
        }

    if data_format == "jsonl":
        prompt, completion, images = build_messages_qa(
            conversations=sample["conversations"],
            image_paths=sample.get("image", []),
            image_root=image_root,
        )
        if prompt is None:
            return {"prompt": [], "completion": [], "images": [], "chat_template_kwargs": {}, "_keep": False}
        if images and processor is not None and max_seq_length is not None:
            if _exceeds_max_length(processor, prompt, completion, images, max_seq_length, chat_template_kwargs):
                return {"prompt": [], "completion": [], "images": [], "chat_template_kwargs": {}, "_keep": False}
    else:
        def _build(turns: int):
            # Fresh rng per attempt so the drawn history length stays a pure function of
            # (seed, sample_id, turns) -- i.e. identical on every rank, never dependent on
            # how many attempts happened to run.
            return build_messages(
                conversations=sample["conversations"],
                image_bytes_list=sample.get("image_bytes", []),
                max_turns=turns,
                rng=random.Random(f"{seed}-{sample_id}"),
            )

        prompt, completion, images = _build(max_turns)
        if prompt is None:
            return {"prompt": [], "completion": [], "images": [], "chat_template_kwargs": {}, "_keep": False}

        # Shrink the sampled history (dropping the OLDEST turns, and with them their
        # images) until the sample fits, instead of discarding the row outright.
        #
        # Dropping rows here is what deadlocked multi-node training: `.filter()` removes a
        # *data-dependent, therefore rank-dependent* number of rows from each rank's shard
        # of the streaming dataset, so ranks stop yielding batches at different steps, run
        # different numbers of micro-batches within one gradient-accumulation window, and
        # end up issuing mismatched NCCL collectives -- some ranks still in
        # `SFTTrainer.compute_loss`'s metrics all-gather while others have already reached
        # the ZeRO optimizer's 1-element overflow all-reduce. That mismatch hangs every
        # rank until the 600s watchdog aborts the job (no Python-level error, which is why
        # it presented only as `Watchdog caught collective operation timeout`).
        #
        # Keeping this map 1:1 (one input row -> one training sample) makes every rank's
        # stream exactly the same length, so the ranks cannot drift apart. It also recovers
        # long-trajectory samples that used to be thrown away: only the surplus history is
        # trimmed, never the current step or its target action.
        if images and processor is not None and max_seq_length is not None:
            turns = max_turns
            while _exceeds_max_length(
                processor, prompt, completion, images, max_seq_length, chat_template_kwargs
            ):
                if turns <= 0:
                    # Even a single turn (current observation + its action) overflows, so
                    # there is nothing left to trim -- drop as a last resort. Vanishingly
                    # rare (one 640x360 frame is ~300 vision tokens), and unlike the
                    # length-based dropping above it is not correlated with trajectory
                    # length, so it does not systematically favour any particular rank.
                    logger.warning(
                        f"Sample {sample_id} exceeds max_seq_length={max_seq_length} even with a "
                        "single turn; dropping it."
                    )
                    return {"prompt": [], "completion": [], "images": [], "chat_template_kwargs": {}, "_keep": False}
                turns -= 1
                prompt, completion, images = _build(turns)
                if prompt is None:
                    return {"prompt": [], "completion": [], "images": [], "chat_template_kwargs": {}, "_keep": False}
    return {
        "prompt": prompt,
        "completion": completion,
        "images": images,
        "chat_template_kwargs": chat_template_kwargs,
        "_keep": True,
    }


def _fast_encoded_length(
    processor,
    rendered: str,
    images: List["Image.Image"],
) -> int:
    """Compute the exact token count `processor(text=[rendered], images=images)` would
    produce, WITHOUT paying for the actual (expensive) pixel resize/rescale/normalize/
    patchify work images go through -- only their (height, width) is needed.

    Relies on `Qwen2VLProcessor`/`Qwen3VLProcessor`'s own public
    `_get_num_multimodal_tokens(image_sizes=...)` (pure arithmetic on image dimensions
    + the vision config's patch_size/merge_size/min_pixels/max_pixels -- see
    `Qwen2VLImageProcessor.get_number_of_image_patches`/`smart_resize`) for the number
    of vision tokens each image expands to, and combines that with a plain-text
    tokenization of `rendered` (which -- pre-expansion -- contains exactly one literal
    image-placeholder token per image, per both models' chat templates:
    `<|vision_start|><|image_pad|><|vision_end|>`) to reconstruct the post-expansion
    total length: `base_len - num_images + sum(num_image_tokens_per_image)`.

    Verified byte-exact against the real `processor(...)` output across both Qwen2-VL
    and Qwen3.5, for 1-5 images and image sizes spanning 1x1 to 7000x50 pixels (see
    dev-time validation script; not shipped as a unit test to keep this file
    dependency-light). Raises if `processor` doesn't support
    `_get_num_multimodal_tokens` (caller falls back to the exact-but-slow path below).
    """
    raw_ids = processor.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    base_len = len(raw_ids)
    if not images:
        return base_len
    image_sizes = [(im.height, im.width) for im in images]
    num_tokens_per_image = processor._get_num_multimodal_tokens(image_sizes=image_sizes)["num_image_tokens"]
    return base_len - len(images) + sum(num_tokens_per_image)


def _exceeds_max_length(
    processor,
    prompt: List[Dict],
    completion: List[Dict],
    images: List["Image.Image"],
    max_seq_length: int,
    chat_template_kwargs: Optional[Dict] = None,
) -> bool:
    """Compute `prompt+completion+images`'s token count under the REAL processor/
    chat-template about to be used for training, and check whether it overflows
    `max_seq_length`.

    This exists purely to pre-filter samples that would otherwise crash training (see
    `_row_to_trl_sample`'s docstring): unlike plain-text overflow (which the collator
    truncates away harmlessly), an overflowing VLM sample risks the truncation point
    landing inside an image's placeholder-token block, which crashes the model's
    forward pass with a tokens/features-count mismatch instead of just losing some
    trailing context. We therefore treat ANY overflow on an image-bearing sample as
    unsafe and drop it, rather than trying to reason about whether this particular
    truncation point happens to fall after the last image (safe) or through one
    (crash) -- the dropped fraction is small and this is far cheaper than debugging a
    mid-run multi-GPU crash.

    Tries the cheap `_fast_encoded_length` path first (see its docstring -- avoids
    redoing the SAME expensive image resize/rescale/normalize/patchify work the
    collator is about to do for real on every image-bearing sample a SECOND time,
    which was silently doubling image-preprocessing CPU cost per sample and capping
    GPU utilization around 30% even after fixing the S3-read bottleneck separately).
    Falls back to the exact-but-slower full `processor(...)` call if the fast path
    raises (e.g. a future/different model class without `_get_num_multimodal_tokens`).

    On any error from BOTH paths (e.g. a malformed/corrupt image), returns True (drop
    defensively) rather than propagating the exception out of a `.map()` call.
    """
    try:
        rendered = processor.apply_chat_template(
            list(prompt) + list(completion),
            tokenize=False,
            add_generation_prompt=False,
            **(chat_template_kwargs or {}),
        )
    except Exception as e:
        logger.warning(f"Length pre-check's apply_chat_template failed for a sample ({e!r}); dropping it defensively.")
        return True

    try:
        total_len = _fast_encoded_length(processor, rendered, images)
    except Exception:
        try:
            encoded = processor(text=[rendered], images=images, return_tensors=None)
            total_len = len(encoded["input_ids"][0])
        except Exception as e:
            logger.warning(f"Length pre-check failed for a sample ({e!r}); dropping it defensively.")
            return True
    return total_len > max_seq_length


def _detect_data_format(data_path: Union[str, List[str]]) -> str:
    """Infer "parquet" vs "jsonl" from the file extension in `data_path` (which may be a
    glob, e.g. "s3://.../train-*.parquet" or "s3://.../*.jsonl", or (see
    `build_minecraft_dataset`) a list of several such globs/paths -- in that case every
    entry is checked and detection fails loudly on a mix of extensions rather than
    silently guessing)."""
    paths = data_path if isinstance(data_path, list) else [data_path]
    formats = {"jsonl" if (".jsonl" in p.lower() or ".json" in p.lower()) else "parquet" for p in paths}
    if len(formats) > 1:
        raise ValueError(f"--data_path mixes parquet and jsonl extensions ({paths!r}); pass --data_format explicitly.")
    return formats.pop()


def _default_image_root(data_path: Union[str, List[str]]) -> str:
    """Directory containing `data_path`'s file(s) -- e.g. for
    "s3://bucket/minecraft-vlp/mc-vqa-241102.jsonl" (or the glob
    "s3://bucket/minecraft-vlp/*.jsonl") this is "s3://bucket/minecraft-vlp". That is
    also where `minecraft-vlp`-style datasets keep their `images/` subdirectory, which
    is what each row's "image" (relative-path) field is rooted at.

    When `data_path` is a list of several files (e.g. combining VQA + Caption +
    Grounding jsonls for JARVIS-VLA Stage II), this uses the FIRST entry's directory --
    only correct if every file lives directly alongside the others under the same
    `<root>/images/...` layout (true for all of `minecraft-vlp`'s files); pass
    `--image_root` explicitly if that doesn't hold."""
    first = data_path[0] if isinstance(data_path, list) else data_path
    return first.rsplit("/", 1)[0]


# Columns actually read anywhere downstream (`_row_to_trl_sample` / `build_messages` /
# `build_messages_qa` / `build_messages_text_only`): the trajectory-id (used only to
# seed the history-sampling RNG, falls back to the row index if absent), the
# conversation itself, and the image path list. Everything else (`label`, `model`,
# `datetime`, `source`, ...) is pure metadata never touched by training code.
_USED_COLUMNS = {"id", "conversations", "image"}


def _load_dataset_multi(builder_name: str, data_path: Union[str, List[str]], streaming: bool):
    """`load_dataset(builder_name, data_files=data_path, split="train", streaming=...)`,
    except when `data_path` is a `list` of several files: those are loaded and reduced
    to `_USED_COLUMNS` ONE FILE AT A TIME, then stitched together with
    `concatenate_datasets`, instead of a single `load_dataset(..., data_files=[...])`
    call across every file at once.

    This matters because `minecraft-vlp`'s jsonl files (needed combined for JARVIS-VLA
    Stage II: VQA + Caption + 3 Grounding files) have wildly different schemas for
    columns training never uses -- e.g. `source` is `{"image_url": str, "points":
    [x,y], ...}` in one file, `{"image_urls": [str, ...], "points": [[x,y], ...],
    "action": str, ...}` in another, and `{"image_url": [str], "points": [{"x":,"y":},
    ...], "bbox": [{"label":, "bbox": [[[...]]]}]}` in a third; one file even has a
    typo'd `datatime` key instead of `datetime`. A single combined `load_dataset(...,
    data_files=[f1, f2, ...])` call makes the streaming JSON reader try to unify ALL
    files' schemas into one global Arrow schema up front (via HF `datasets`'
    `_cast_table`/`table_cast`) -- verified this hard-crashes with `TypeError: Couldn't
    cast array of type struct<...> to struct<...>` as soon as iteration reaches a file
    whose `source` struct-shape disagrees with the one inferred from an earlier file.
    Since pyarrow's per-file JSON schema inference never sees more than one file when
    each is loaded (and trimmed) separately, this side-steps the incompatibility
    entirely -- verified against the real 5-file Stage II combination (261,461 rows
    iterated end-to-end with no error, vs. an immediate crash with the naive combined
    call).
    """
    if not isinstance(data_path, list):
        return load_dataset(builder_name, data_files=data_path, split="train", streaming=streaming)

    per_file = []
    for path in data_path:
        ds = load_dataset(builder_name, data_files=path, split="train", streaming=streaming)
        drop = [c for c in ds.column_names if c not in _USED_COLUMNS]
        if drop:
            ds = ds.remove_columns(drop)
        per_file.append(ds)
    return concatenate_datasets(per_file)


def build_minecraft_dataset(
    data_path: Union[str, List[str]],
    max_turns: int = 4,
    streaming: bool = False,
    seed: int = 42,
    data_format: str = "auto",
    image_root: Optional[str] = None,
    text_only: bool = False,
    processor=None,
    max_seq_length: Optional[int] = None,
):
    """
    Build the Minecraft SFT dataset as a genuine `datasets.Dataset` (`streaming=False`)
    or `datasets.IterableDataset` (`streaming=True`) of TRL prompt-completion samples:
        {"prompt": [...], "completion": [...], "images": [PIL.Image, ...]}
    (see `build_messages`/`build_messages_qa` for why prompt/completion instead of a
    flat `messages` list -- it's what lets `SFTConfig(completion_only_loss=True)` mask
    context/history out of the loss for VLM training).

    Supports two on-disk layouts, auto-detected from `data_path`'s extension (override
    with `data_format="parquet"` or `data_format="jsonl"`):
      - "parquet" (e.g. `minecraft-text-action-dataset`): each row is one trajectory
        with a `conversations` list plus an `image_bytes` list (one raw-JPEG-bytes
        entry per (user, assistant) turn pair); `build_messages` randomly samples a
        suffix history window (`max_turns`) from it. Images are decoded from the
        already-embedded bytes.
      - "jsonl" (e.g. `minecraft-vlp`): each row is a short, independent multi-turn Q&A
        session with an `image` field listing path(s) *relative to `image_root`*
        (default: the directory containing the jsonl file(s), which matches
        `minecraft-vlp`'s layout of `<root>/*.jsonl` + `<root>/images/...`);
        `build_messages_qa` loads those bytes on-the-fly via `fsspec` (works for both
        local paths and `s3://` URIs, the latter via the `s3fs` dependency) and keeps
        the whole conversation (no history sampling -- these rows are already short).

    `text_only=True` (Stage I): for pure-text rows with no images at all -- e.g.
    `minecraft-vlp/mc-qa-*.jsonl`'s `label=["qa","wiki","self-instruct"]` rows, a
    `system` + `user` + `assistant` text QA turn with `image=[]` -- routes every row
    through `build_messages_text_only` instead of `build_messages`/`build_messages_qa`,
    and the resulting samples carry NO "images" key at all (see `_row_to_trl_sample`).
    This is what JARVIS-VLA calls Stage I ("Minecraft world knowledge" text-only
    post-training); combine with `freeze_vision_tower(model)` beforehand to also match
    the paper's "ViT + adapter frozen, only LLM trained" recipe for this stage. Using
    `text_only=True` together with `data_format="parquet"` is unusual (parquet rows are
    trajectories with real images) and logs a warning, but works: any `image_bytes` on
    those rows is simply ignored.

    IMPORTANT: this returns the result of chaining `.map()` / `.filter()` /
    `.remove_columns()` directly on `datasets.load_dataset(...)`'s return value -- NOT a
    hand-rolled `torch.utils.data.IterableDataset` subclass. `trl.SFTTrainer` requires
    `train_dataset` to be a `datasets.Dataset` / `datasets.IterableDataset` instance (it
    raises `TypeError` otherwise at trainer-construction time, even though a custom torch
    `IterableDataset` would iterate correctly on its own -- this bug is what this function
    replaces). Chaining `.map()`/`.filter()` on the loaded dataset keeps the returned
    object a real `datasets`-library type while preserving all sharding behavior:
      - Cross-rank sharding uses `datasets.distributed.split_dataset_by_node`, applied
        *before* `.map()`, so each rank only streams/transforms its own shards
        (instead of every rank pulling the full dataset over the network).
      - Cross-worker sharding (when `dataloader_num_workers > 0`) needs no extra code:
        `.map()`/`.filter()` on an `IterableDataset` just wrap the underlying shard
        iterator rather than replacing it with a single unshardable generator, so
        `datasets.IterableDataset.__iter__`'s own `get_worker_info()`-based splitting
        still kicks in automatically inside each DataLoader worker process.

    When `streaming=False`, the underlying HF dataset is fully materialized and
    random-access (`__getitem__`/`__len__`) is supported as usual for `datasets.Dataset`;
    when `streaming=True` (the default for large S3 datasets), only iteration is
    supported -- `len()` is undefined, matching `datasets.IterableDataset` semantics.

    `processor`/`max_seq_length`: when both are given, image-bearing samples whose REAL
    tokenized length (computed with this exact `processor`) would overflow
    `max_seq_length` are dropped instead of silently truncated -- see
    `_row_to_trl_sample`/`_exceeds_max_length` for why blind token-level truncation is
    unsafe for VLM samples (it can crash training by cutting through an image's
    placeholder-token block). Pass the same `AutoProcessor` instance used to build
    `training_args`/the model so the token counts match exactly. Omit both (the
    default) to skip this check entirely -- e.g. for Stage I `--text_only` data, which
    has no images and thus no risk of this specific crash.
    """
    if data_format == "auto":
        data_format = _detect_data_format(data_path)
    if data_format not in ("parquet", "jsonl"):
        raise ValueError(f"Unknown data_format: {data_format!r} (expected 'parquet', 'jsonl', or 'auto')")
    if text_only and data_format == "parquet":
        logger.warning(
            "--text_only was set together with a parquet (trajectory) data_format; "
            "this is unusual -- --text_only is designed for Stage I text-QA jsonl rows "
            "(e.g. minecraft-vlp/mc-qa-*.jsonl). Any image_bytes on these rows will be "
            "ignored."
        )
    if data_format == "jsonl" and image_root is None:
        image_root = _default_image_root(data_path)

    builder_name = "parquet" if data_format == "parquet" else "json"
    if streaming:
        dataset = _load_dataset_multi(builder_name, data_path, streaming=True)
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
            logger.info(f"Sharded streaming dataset across {world_size} ranks (this rank={rank}).")
        logger.info(f"Dataset loaded in streaming mode (format={data_format}, length unknown ahead of time)")
    else:
        dataset = _load_dataset_multi(builder_name, data_path, streaming=False)
        logger.info(f"Dataset loaded (format={data_format}): {len(dataset)} samples")

    raw_columns = dataset.column_names
    dataset = dataset.map(
        _row_to_trl_sample,
        with_indices=True,
        fn_kwargs={
            "max_turns": max_turns,
            "seed": seed,
            "data_format": data_format,
            "image_root": image_root,
            "text_only": text_only,
            "processor": processor,
            "max_seq_length": max_seq_length,
        },
        remove_columns=raw_columns,
    )
    dataset = dataset.filter(lambda x: x["_keep"])
    dataset = dataset.remove_columns(["_keep"])
    return dataset


# ─── model helpers ─────────────────────────────────────────────────────────────

# Substrings matched (case-insensitively) against each submodule's own leaf name (not
# its full dotted path) to find the vision tower. For the Qwen2-VL / Qwen2.5-VL /
# Qwen3-VL / Qwen3.5-VL family the ViT + patch-merger both live under a single
# submodule literally named "visual" (e.g. `model.visual`, containing `patch_embed`,
# `blocks`, AND `merger`) -- so freezing that one submodule already freezes the
# encoder and the vision-to-text adapter together, matching JARVIS-VLA's Stage I
# recipe of "ViT + adapter frozen, only the LLM backbone trained". The other hints are
# fallbacks for other VLM architectures that split the encoder/adapter differently.
_VISION_SUBMODULE_HINTS = ("visual", "vision_tower", "vision_model", "image_encoder")


def freeze_vision_tower(model: torch.nn.Module) -> None:
    """
    Freeze the vision encoder (+ its adapter/merger into the LLM's embedding space),
    leaving only the language-model backbone trainable. This is JARVIS-VLA's Stage I
    ("Minecraft world knowledge" text-only post-training) recipe: only the LLM is
    updated while the vision tower stays frozen; Stage II then unfreezes everything
    again once real image data (VQA/captioning/grounding) is introduced -- so this
    should only be called for the Stage I / `--text_only` run, not Stage II.

    Walks `model.named_modules()` looking for a submodule whose *own* name (not its
    full dotted path) matches one of `_VISION_SUBMODULE_HINTS`, and sets
    `requires_grad_(False)` on every parameter under it. Only the outermost matching
    submodule per branch is frozen (a submodule nested inside an already-frozen one is
    skipped) to avoid redundant work and confusing double-counting in the log message.

    Raises `RuntimeError` if no matching submodule is found at all -- silently no-op'ing
    here would be far worse than crashing, since it would look like Stage I is running
    correctly while actually training the full model (vision tower included).
    """
    frozen_modules: List[str] = []
    frozen_params = 0

    for name, module in model.named_modules():
        if not name:
            continue
        leaf_name = name.rsplit(".", 1)[-1].lower()
        if not any(hint in leaf_name for hint in _VISION_SUBMODULE_HINTS):
            continue
        if any(name == m or name.startswith(f"{m}.") for m in frozen_modules):
            continue  # nested inside an already-frozen submodule

        n = 0
        for p in module.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n += p.numel()
        if n > 0:
            frozen_modules.append(name)
            frozen_params += n

    if not frozen_modules:
        raise RuntimeError(
            "freeze_vision_tower: could not find any submodule matching "
            f"{_VISION_SUBMODULE_HINTS} (case-insensitive, matched against each "
            "submodule's own leaf name) on this model. Inspect `model.named_modules()` "
            "for this architecture and extend `_VISION_SUBMODULE_HINTS`."
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"freeze_vision_tower: froze {frozen_modules} ({frozen_params:,} params). "
        f"Trainable now: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.1f}%)."
    )


# ─── immutable VLM collator adapter ───────────────────────────────────────────


def _clone_conversation(messages):
    """Clone mutable chat containers while retaining immutable/PIL payload references."""
    if not isinstance(messages, list):
        return messages
    cloned_messages = []
    for message in messages:
        if not isinstance(message, dict):
            cloned_messages.append(message)
            continue
        cloned_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            cloned_message["content"] = [dict(item) if isinstance(item, dict) else item for item in content]
        cloned_messages.append(cloned_message)
    return cloned_messages


class _ImmutableVisionCollatorAdapter:
    """Give TRL's mutating VLM collator disposable sample containers.

    `DataCollatorForVisionLanguageModeling` injects decoded images into prompt content
    and writes the resulting messages back to the supplied example dict. Dataset rows
    must remain pristine because an iterable/dataloader may hand the same Python object
    to the collator again. This adapter copies only the mutable dict/list structure;
    decoded PIL images remain shared references, so it does not duplicate pixel memory.
    It intentionally does not catch or alter collator exceptions.
    """

    def __init__(self, inner_collator):
        self.inner_collator = inner_collator

    def __call__(self, examples):
        working_examples = []
        for example in examples:
            working = dict(example)
            for field in ("messages", "prompt", "completion"):
                if field in working:
                    working[field] = _clone_conversation(working[field])
            if isinstance(working.get("images"), list):
                working["images"] = list(working["images"])
            working_examples.append(working)
        return self.inner_collator(working_examples)


# ─── debug / dry-run helpers ──────────────────────────────────────────────────


def debug_dry_run(args):
    """
    Quick END-TO-END smoke test against a real GPU, using the exact same code path as
    real training:
      1. Download the model, load processor + model.
      2. Build the REAL training dataset via `build_minecraft_dataset` (same function
         `main()` calls) -- this is what actually catches `train_dataset`
         type/format incompatibilities with `SFTTrainer` (e.g. a hand-rolled
         `torch.utils.data.IterableDataset` raises `TypeError` at trainer-construction
         time; a manual `processor(...)`/collator-only test would never hit that code
         path at all).
      3. Construct a real `SFTTrainer` with a bounded `SFTConfig` (controlled by
      `--debug_steps`, no checkpoint saving, no external logging) and call `.train()` -- this exercises

         dataset iteration, the vision-language collator, forward, backward, and an
         optimizer step through the *exact* same trainer code real training uses.
    """
    logger.info("=== DEBUG DRY RUN ===")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")

    # Download model into a cache dir specific to this model (avoids reusing a
    # stale download from a previously-tested model in the same container).
    cache_dir = args.download_model or f"/tmp/{_local_cache_name(args.model_path)}"
    local_model = args.model_path
    if args.model_path.startswith("s3://"):
        local_model = download_from_s3(args.model_path.rstrip("/"), cache_dir, exclude_checkpoints=True)

    logger.info("Loading model & processor...")
    processor = AutoProcessor.from_pretrained(local_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        local_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    if args.freeze_vision_tower:
        freeze_vision_tower(model)

    logger.info("Building dataset (real `build_minecraft_dataset` code path)...")
    dataset = build_minecraft_dataset(
        data_path=args.data_path,
        max_turns=args.max_turns,
        streaming=True,
        seed=args.seed,
        data_format=args.data_format,
        image_root=args.image_root,
        text_only=args.text_only,
        processor=processor,
        max_seq_length=args.max_seq_length,
    )

    sft_config_field_names = {f.name for f in dataclass_fields(SFTConfig)}
    max_len_kwarg = {}
    if "max_length" in sft_config_field_names:
        max_len_kwarg["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_config_field_names:
        max_len_kwarg["max_seq_length"] = args.max_seq_length

    debug_output_dir = os.path.join(args.output_dir, "_debug_dry_run")
    training_args = SFTConfig(
        output_dir=debug_output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.debug_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=1,
        save_strategy="no",
        dataloader_num_workers=args.dataloader_num_workers,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        packing=False,
        completion_only_loss=True,
        seed=args.seed,
        report_to=["none"],
        **max_len_kwarg,
    )

    logger.info("Constructing SFTTrainer (this is where a bad train_dataset type/format would raise)...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.data_collator = _ImmutableVisionCollatorAdapter(trainer.data_collator)
    logger.info(f"Running trainer.train() for max_steps={args.debug_steps} with immutable VLM collator inputs...")
    trainer.train()
    logger.info(f"=== DRY RUN PASSED (SFTTrainer built + trained for {args.debug_steps} steps successfully) ===")


# ─── main training ────────────────────────────────────────────────────────────


def _is_complete_trainer_checkpoint(checkpoint: str) -> bool:
    """Return whether a Trainer checkpoint has its completion state and model artifact."""
    trainer_state_path = os.path.join(checkpoint, "trainer_state.json")
    if not os.path.isfile(trainer_state_path) or os.path.getsize(trainer_state_path) == 0:
        return False
    try:
        with open(trainer_state_path, "r", encoding="utf-8") as state_file:
            json.load(state_file)
    except (OSError, json.JSONDecodeError):
        return False

    model_artifacts = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    return any(os.path.isfile(os.path.join(checkpoint, artifact)) for artifact in model_artifacts)


def _resolve_resume_checkpoint(requested: Optional[str], output_dir: str) -> Optional[str]:
    """Resolve an explicit checkpoint path or the latest complete checkpoint in ``output_dir``."""
    if requested in (None, "none"):
        return None

    if requested == "auto":
        candidates = []
        if os.path.isdir(output_dir):
            for entry in os.scandir(output_dir):
                if entry.is_dir() and entry.name.startswith("checkpoint-"):
                    try:
                        step = int(entry.name.removeprefix("checkpoint-"))
                    except ValueError:
                        continue
                    candidates.append((step, entry.path))
        for _, checkpoint in sorted(candidates, reverse=True):
            if _is_complete_trainer_checkpoint(checkpoint):
                logger.info(f"Resuming exact Trainer state from checkpoint: {checkpoint}")
                return checkpoint
            logger.warning(f"Skipping incomplete checkpoint: {checkpoint}")
        logger.warning("--resume_from_checkpoint=auto found no complete checkpoint; starting a new run.")
        return None

    if not os.path.isdir(requested):
        raise ValueError(f"Resume checkpoint does not exist or is not a directory: {requested}")
    if not _is_complete_trainer_checkpoint(requested):
        raise ValueError(f"Resume checkpoint is incomplete: {requested}")
    logger.info(f"Resuming exact Trainer state from checkpoint: {requested}")
    return requested


def main():
    parser = argparse.ArgumentParser(description="TRL SFT for Minecraft VLM")
    parser.add_argument("--model_path", type=str, required=True, help="S3 or local path to model")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="S3 glob or local path to parquet/jsonl files. May be a comma-separated "
        "list of several such paths/globs (e.g. to combine minecraft-vlp's "
        "mc-vqa-*.jsonl + mc-caption-*.jsonl + mc-grounding-point-*.jsonl into one "
        "dataset for JARVIS-VLA Stage II) -- each entry is passed through as-is (still "
        "supports globs), just don't mix parquet and jsonl in the same list.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "parquet", "jsonl"],
        help="'parquet' (e.g. minecraft-text-action-dataset, embedded image_bytes) or "
        "'jsonl' (e.g. minecraft-vlp, images loaded from an 'image_root' via relative "
        "paths). 'auto' (default) infers from --data_path's extension.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Only used for --data_format=jsonl: base dir/URI that each row's 'image' "
        "relative path(s) are resolved against. Defaults to the directory containing "
        "--data_path (matches minecraft-vlp's <root>/*.jsonl + <root>/images/ layout).",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Stage I mode (JARVIS-VLA's 'Minecraft world knowledge' text-only "
        "post-training): treat every row as plain system+user+assistant text QA with "
        "no images (e.g. minecraft-vlp/mc-qa-*.jsonl), producing samples with no "
        "'images' key so SFTTrainer uses its plain-text collator instead of the "
        "vision-language one. Typically combined with --freeze_vision_tower.",
    )
    parser.add_argument(
        "--freeze_vision_tower",
        action="store_true",
        help="Freeze the vision encoder + adapter/merger, training only the LLM "
        "backbone -- matches JARVIS-VLA's Stage I recipe. Typically combined with "
        "--text_only.",
    )
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="auto",
        help="Checkpoint directory to resume exactly (model, optimizer, scheduler and RNG). "
        "'auto' (default) resumes the latest complete checkpoint-* in --output_dir; "
        "use 'none' to force a new run.",
    )
    parser.add_argument("--max_turns", type=int, default=4, help="Max (user,assistant) pairs per sample")
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Trade compute for activation memory -- recommended for large models / "
        "long max_seq_length, especially when DeepSpeed optimizer-state offload isn't "
        "available (e.g. due to a CUDA-toolkit/torch-build version mismatch preventing "
        "DeepSpeedCPUAdam's JIT compile).",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Explicit total training steps, overriding the built-in dataset-size "
        "estimate below (which is specific to minecraft-text-action-dataset and wrong "
        "for any other dataset -- always pass this for --text_only/--data_format=jsonl runs).",
    )
    parser.add_argument("--learning_rate", type=float, default=8e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Fixed warmup step count. If > 0, this takes precedence over --warmup_ratio "
        "(standard `transformers.TrainingArguments` behavior). JARVIS-VLA's Stage I/II "
        "recipe uses a fixed 200-step warmup rather than a ratio.",
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="HF Trainer default is 0.999; JARVIS-VLA's recipe uses 0.95.",
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of background `DataLoader` worker processes (per rank) used to "
            "prefetch/preprocess samples (image decode/resize, tokenization) while the "
            "GPU trains on the previous batch. With `--image_root` pointed at local "
            "disk (see the training launch script's pre-download-to-/local-ssd step) "
            "and `_exceeds_max_length`'s cheap size-only length pre-check (see its "
            "docstring), the remaining per-sample CPU cost is dominated by the "
            "collator's real image resize/rescale/normalize/patchify -- more workers "
            "lets that run in parallel across CPU cores instead of serializing behind "
            "GPU compute. Bump this further (e.g. 8) if GPU utilization is still low "
            "after confirming images are being read from local disk, not S3."
        ),
    )
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stall_dump_seconds",
        type=float,
        default=180.0,
        help="Diagnose hangs: if no training step completes within this many seconds, "
        "dump every thread's Python traceback (plus the dataloader workers') to "
        "stderr/job log. A multi-node hang otherwise only shows up as NCCL's "
        "'Watchdog caught collective operation timeout ... 600000ms', which never "
        "reveals which line each rank is stuck on. Keep this well below the NCCL "
        "timeout so stacks are captured while the process is still alive. Set 0 to "
        "disable (default: 180).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="GPU smoke/stress run: build the real dataset, construct a real SFTTrainer, train for "
        "--debug_steps, then exit without saving a checkpoint or final model.",
    )
    parser.add_argument(
        "--debug_steps",
        type=int,
        default=2,
        help="Number of optimizer steps for --debug (default: 2).",
    )
    parser.add_argument("--download_model", type=str, default=None, help="Local dir to cache downloaded model")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    # NOTE: TRL's SFTTrainer raises ValueError for packing=True on vision-language
    # models (all supported models here are VLMs), so packing defaults to False and
    # any attempt to force it on is rejected with a clear error instead of a crash
    # deep inside the trainer.
    parser.add_argument("--packing", action="store_true", default=False, help="NOT supported for VLM training; kept for API symmetry")
    parser.add_argument("--no_packing", action="store_false", dest="packing")

    args = parser.parse_args()

    # Arm before anything heavy, so even a hang during dataset/model setup is caught.
    _install_stall_watchdog(args.stall_dump_seconds)

    # Allow --data_path to be a comma-separated list of files/globs (e.g. combining
    # VQA + Caption + Grounding jsonls for JARVIS-VLA Stage II, which don't share a
    # single glob pattern without also pulling in unrelated files like mc-qa-*.jsonl).
    # `build_minecraft_dataset`/`_detect_data_format`/`_default_image_root` all accept
    # either a plain str or a List[str] here.
    if "," in args.data_path:
        args.data_path = [p.strip() for p in args.data_path.split(",") if p.strip()]

    if args.packing:
        raise ValueError(
            "--packing was requested, but TRL's SFTTrainer does not support sequence "
            "packing for vision-language models (Qwen2-VL / Qwen2.5-VL / Qwen3-VL / "
            "Qwen3.5-VL are all VLMs here). Remove --packing."
        )

    # ── seed ──
    set_seed(args.seed)

    # ── debug mode ──
    if args.debug:
        debug_dry_run(args)
        sys.exit(0)

    resume_from_checkpoint = _resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)

    # ── download model ──
    local_model_path = args.model_path
    if args.model_path.startswith("s3://"):
        cache_dir = args.download_model or f"/tmp/{_local_cache_name(args.model_path)}"
        local_model_path = download_from_s3(args.model_path.rstrip("/"), cache_dir, exclude_checkpoints=True)

    # ── load model & processor ──
    logger.info(f"Loading model from {local_model_path} ...")
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        local_model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    if args.freeze_vision_tower:
        freeze_vision_tower(model)

    # ── load dataset ──
    # For large S3 datasets, use streaming to avoid downloading everything.
    # `build_minecraft_dataset` returns a genuine `datasets.IterableDataset` (built via
    # `.map()`/`.filter()` chained on `datasets.load_dataset(...)`), which is required
    # for `SFTTrainer` -- a hand-rolled `torch.utils.data.IterableDataset` subclass is
    # rejected with `TypeError` at trainer-construction time.
    dataset = build_minecraft_dataset(
        data_path=args.data_path,
        max_turns=args.max_turns,
        streaming=True,
        seed=args.seed,
        data_format=args.data_format,
        image_root=args.image_root,
        text_only=args.text_only,
        # Pre-filter oversized multi-image samples using the REAL processor/max_length
        # about to be used for training -- see `_row_to_trl_sample`/`_exceeds_max_length`.
        # `text_only` samples have no images so this is a no-op for Stage I regardless.
        processor=processor,
        max_seq_length=args.max_seq_length,
    )

    # ── training config ──
    total_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps
    # For torchrun, world_size is available via env
    n_gpus = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", 1)))
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        # Compute max_steps from approximate dataset size. This magic number is
        # `minecraft-text-action-dataset`-specific (363 files x ~600 samples each ~=
        # 217800 samples per epoch) -- it does NOT apply to other datasets/formats
        # (e.g. --text_only Stage I data, which has a very different row count). Pass
        # --max_steps explicitly for anything other than the default parquet dataset.
        if args.text_only or args.data_format == "jsonl":
            logger.warning(
                "No --max_steps given and --text_only/--data_format=jsonl is set: "
                "falling back to the minecraft-text-action-dataset-specific sample-count "
                "estimate below, which is almost certainly wrong for this dataset. Pass "
                "--max_steps explicitly."
            )
        approx_dataset_size = 217800
        max_steps = (approx_dataset_size * args.num_train_epochs) // (total_batch_size * n_gpus)

    # `SFTConfig`'s max-sequence-length kwarg was renamed from `max_seq_length` to
    # `max_length` in newer TRL releases. Detect which one the installed TRL expects
    # so this script keeps working across TRL versions.
    sft_config_field_names = {f.name for f in dataclass_fields(SFTConfig)}
    max_len_kwarg = {}
    if "max_length" in sft_config_field_names:
        max_len_kwarg["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_config_field_names:
        max_len_kwarg["max_seq_length"] = args.max_seq_length
    else:
        logger.warning("Neither `max_length` nor `max_seq_length` found on SFTConfig; skipping.")

    # `TrainingArguments.warmup_ratio` was REMOVED as an `__init__` kwarg in some
    # transformers releases (consistent with a "warmup_ratio is deprecated ... removed
    # in v5.2" warning seen on affected versions) in favor of `warmup_steps` alone --
    # passing `warmup_ratio=...` unconditionally would then raise `TypeError` at
    # `SFTConfig(...)` construction time. Detect support the same way `max_length` vs
    # `max_seq_length` is detected above, and only pass whichever of
    # `warmup_steps`/`warmup_ratio` is actually needed: if `--warmup_steps > 0` is
    # requested, always pass that (unambiguous on every version tested); otherwise fall
    # back to `--warmup_ratio`, but only include it in the SFTConfig kwargs if the
    # installed version still supports it (recent versions default warmup_ratio's
    # effect to 0 when omitted, which is the same as passing 0.0 explicitly anyway).
    warmup_kwargs: Dict[str, float] = {"warmup_steps": args.warmup_steps}
    if args.warmup_steps <= 0:
        if "warmup_ratio" in sft_config_field_names:
            warmup_kwargs["warmup_ratio"] = args.warmup_ratio
        else:
            logger.warning(
                "--warmup_ratio was requested but this installed version of "
                "`trl`/`transformers` no longer accepts `warmup_ratio` on `SFTConfig`; "
                "ignoring it. Pass --warmup_steps (an absolute step count) instead."
            )
    logger.info(f"Warmup config: {warmup_kwargs}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        deepspeed=args.deepspeed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        packing=False,  # unsupported for VLMs, see argparse note above
        # Loss must only be computed on the target "Action: ..." completion, not on the
        # prompt (system prompt / history / current-step image+instruction). This is the
        # VLM-supported equivalent of `assistant_only_loss` (which TRL rejects for vision
        # datasets) -- it relies on `build_messages` splitting each sample into
        # {"prompt": ..., "completion": [last assistant turn]} rather than a flat
        # `messages` list.
        completion_only_loss=True,
        seed=args.seed,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
        run_name=os.environ.get("WANDB_RUN_NAME", "minecraft-sft-trl"),
        **warmup_kwargs,
        **max_len_kwarg,
    )

    logger.info(f"Training config: total_batch={total_batch_size}, n_gpus={n_gpus}, max_steps={max_steps}")
    logger.info(
        f"Resolved training_args: warmup_steps={training_args.warmup_steps}, "
        f"warmup_ratio={getattr(training_args, 'warmup_ratio', 'N/A')}, "
        f"max_steps={training_args.max_steps}, learning_rate={training_args.learning_rate}"
    )

    # ── trainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    if not args.text_only:
        trainer.data_collator = _ImmutableVisionCollatorAdapter(trainer.data_collator)
    if args.stall_dump_seconds > 0:
        trainer.add_callback(_HeartbeatCallback())
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    logger.info(f"Training finished. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
