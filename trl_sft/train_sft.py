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

import argparse
import io
import json
import logging
import math
import os
import random
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── s3 helpers ────────────────────────────────────────────────────────────────


def _local_cache_name(s3_or_local_path: str) -> str:
    """Derive a filesystem-safe, model-specific cache dir name from a path."""
    name = s3_or_local_path.rstrip("/").split("/")[-1]
    return name or "model"


def download_from_s3(s3_path: str, local_dir: str) -> str:
    """Download model/dataset from S3 to local disk, skip if already exists."""
    local_dir = Path(local_dir)
    marker = local_dir / ".download_complete"

    if marker.exists():
        logger.info(f"Already downloaded: {local_dir}")
        return str(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {s3_path} -> {local_dir} ...")
    ret = os.system(f"aws s3 cp --recursive {s3_path} {local_dir}/")
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

        # TRL expects roles: "user", "assistant", "system"
        messages.append({"role": role, "content": content_list})

    # Split off the final assistant turn as the `completion`; everything before it
    # becomes the `prompt` (context that `completion_only_loss` will mask out).
    prompt, completion = messages[:-1], [messages[-1]]
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


def _row_to_trl_sample(
    sample: Dict,
    idx: int,
    max_turns: int,
    seed: int,
    data_format: str,
    image_root: Optional[str],
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

    Invalid rows are flagged via `"_keep": False` instead of being dropped here (a
    `.map()` function must return exactly one output row per input row); the caller
    chains `.filter(lambda x: x["_keep"])` afterwards to actually drop them.
    """
    sample_id = sample.get("id", idx)
    if data_format == "jsonl":
        prompt, completion, images = build_messages_qa(
            conversations=sample["conversations"],
            image_paths=sample.get("image", []),
            image_root=image_root,
        )
    else:
        rng = random.Random(f"{seed}-{sample_id}")
        prompt, completion, images = build_messages(
            conversations=sample["conversations"],
            image_bytes_list=sample.get("image_bytes", []),
            max_turns=max_turns,
            rng=rng,
        )
    if prompt is None:
        return {"prompt": [], "completion": [], "images": [], "_keep": False}
    return {"prompt": prompt, "completion": completion, "images": images, "_keep": True}


def _detect_data_format(data_path: str) -> str:
    """Infer "parquet" vs "jsonl" from the file extension in `data_path` (which may be a
    glob, e.g. "s3://.../train-*.parquet" or "s3://.../*.jsonl")."""
    lowered = data_path.lower()
    if ".jsonl" in lowered or ".json" in lowered:
        return "jsonl"
    return "parquet"


def _default_image_root(data_path: str) -> str:
    """Directory containing `data_path`'s file(s) -- e.g. for
    "s3://bucket/minecraft-vlp/mc-vqa-241102.jsonl" (or the glob
    "s3://bucket/minecraft-vlp/*.jsonl") this is "s3://bucket/minecraft-vlp". That is
    also where `minecraft-vlp`-style datasets keep their `images/` subdirectory, which
    is what each row's "image" (relative-path) field is rooted at."""
    return data_path.rsplit("/", 1)[0]


def build_minecraft_dataset(
    data_path: str,
    max_turns: int = 4,
    streaming: bool = False,
    seed: int = 42,
    data_format: str = "auto",
    image_root: Optional[str] = None,
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
    """
    if data_format == "auto":
        data_format = _detect_data_format(data_path)
    if data_format not in ("parquet", "jsonl"):
        raise ValueError(f"Unknown data_format: {data_format!r} (expected 'parquet', 'jsonl', or 'auto')")
    if data_format == "jsonl" and image_root is None:
        image_root = _default_image_root(data_path)

    builder_name = "parquet" if data_format == "parquet" else "json"
    if streaming:
        dataset = load_dataset(builder_name, data_files=data_path, split="train", streaming=True)
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
            logger.info(f"Sharded streaming dataset across {world_size} ranks (this rank={rank}).")
        logger.info(f"Dataset loaded in streaming mode (format={data_format}, length unknown ahead of time)")
    else:
        dataset = load_dataset(builder_name, data_files=data_path, split="train")
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
        },
        remove_columns=raw_columns,
    )
    dataset = dataset.filter(lambda x: x["_keep"])
    dataset = dataset.remove_columns(["_keep"])
    return dataset


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
      3. Construct a real `SFTTrainer` with a tiny `SFTConfig` (`max_steps=2`, no
         checkpoint saving, no external logging) and call `.train()` -- this exercises
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
        local_model = download_from_s3(args.model_path.rstrip("/"), cache_dir)

    logger.info("Loading model & processor...")
    processor = AutoProcessor.from_pretrained(local_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        local_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )

    logger.info("Building dataset (real `build_minecraft_dataset` code path)...")
    dataset = build_minecraft_dataset(
        data_path=args.data_path,
        max_turns=args.max_turns,
        streaming=True,
        seed=args.seed,
        data_format=args.data_format,
        image_root=args.image_root,
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        dataloader_num_workers=0,
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

    logger.info("Running trainer.train() for max_steps=2...")
    trainer.train()
    logger.info("=== DRY RUN PASSED (SFTTrainer built + trained for 2 steps successfully) ===")


# ─── main training ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TRL SFT for Minecraft VLM")
    parser.add_argument("--model_path", type=str, required=True, help="S3 or local path to model")
    parser.add_argument("--data_path", type=str, required=True, help="S3 glob or local path to parquet/jsonl files")
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
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_turns", type=int, default=4, help="Max (user,assistant) pairs per sample")
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=8e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dry-run: build the real dataset, construct a real SFTTrainer, train for "
        "max_steps=2, exit (no checkpoint saved).",
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

    if args.packing:
        raise ValueError(
            "--packing was requested, but TRL's SFTTrainer does not support sequence "
            "packing for vision-language models (Qwen2-VL / Qwen2.5-VL / Qwen3-VL / "
            "Qwen3.5-VL are all VLMs here). Remove --packing."
        )

    set_seed(args.seed)

    # ── debug mode ──
    if args.debug:
        debug_dry_run(args)
        sys.exit(0)

    # ── download model ──
    local_model_path = args.model_path
    if args.model_path.startswith("s3://"):
        cache_dir = args.download_model or f"/tmp/{_local_cache_name(args.model_path)}"
        local_model_path = download_from_s3(args.model_path.rstrip("/"), cache_dir)

    # ── load model & processor ──
    logger.info(f"Loading model from {local_model_path} ...")
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        local_model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )

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
    )

    # ── training config ──
    total_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps
    # For torchrun, world_size is available via env
    n_gpus = int(os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", 1)))
    # Compute max_steps from approximate dataset size
    # 363 files × ~600 samples each ≈ 217800 samples per epoch
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

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        deepspeed=args.deepspeed,
        dataloader_num_workers=2,
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
        **max_len_kwarg,
    )

    logger.info(f"Training config: total_batch={total_batch_size}, n_gpus={n_gpus}, max_steps={max_steps}")

    # ── trainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    logger.info(f"Training finished. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
