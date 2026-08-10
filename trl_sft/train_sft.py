"""
TRL-based SFT training for Minecraft text-action VLM.

Data: s3://arcwm-code-us-west-2/axiom/data/minecraft-text-action-dataset/
Model: s3://arcwm-code-us-west-2/axiom/model/Qwen3.5-9B/

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
    selected_images = image_bytes_list[start_turn:] if image_bytes_list else []

    # Build messages, keeping image placeholders in-place (to preserve
    # text/image interleaving order) and collecting the actual decoded
    # images into a separate flat list.
    messages = []
    images: List[Image.Image] = []
    image_idx = 0

    for conv in selected_convs:
        role = conv["role"]
        content_list = []

        for item in conv["content"]:
            if item.get("type") == "text":
                content_list.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                if image_idx < len(selected_images):
                    try:
                        img = Image.open(io.BytesIO(selected_images[image_idx])).convert("RGB")
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


def _row_to_trl_sample(sample: Dict, idx: int, max_turns: int, seed: int) -> Dict:
    """
    Map ONE raw parquet row to a TRL prompt/completion/images sample.

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

    Uses a *local* RNG keyed by the sample's own stable "id" (not `idx`, the
    stream-position/row index) so that (a) different ranks/workers never happen to
    reuse the same seed for "the k-th sample they each see locally" -- which they
    otherwise would, since after sharding every rank/worker's local stream position
    resets to 0 -- and (b) we don't mutate the global `random` module state as a
    side effect.

    Invalid rows are flagged via `"_keep": False` instead of being dropped here (a
    `.map()` function must return exactly one output row per input row); the caller
    chains `.filter(lambda x: x["_keep"])` afterwards to actually drop them.
    """
    sample_id = sample.get("id", idx)
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


def build_minecraft_dataset(
    data_path: str,
    max_turns: int = 4,
    streaming: bool = False,
    seed: int = 42,
):
    """
    Build the Minecraft SFT dataset as a genuine `datasets.Dataset` (`streaming=False`)
    or `datasets.IterableDataset` (`streaming=True`) of TRL prompt-completion samples:
        {"prompt": [...], "completion": [...], "images": [PIL.Image, ...]}
    (see `build_messages` for why prompt/completion instead of a flat `messages` list --
    it's what lets `SFTConfig(completion_only_loss=True)` mask context/history out of the
    loss for VLM training). Images are decoded from bytes on-the-fly (not cached).

    IMPORTANT: this returns the result of chaining `.map()` / `.filter()` /
    `.remove_columns()` directly on `datasets.load_dataset(...)`'s return value -- NOT a
    hand-rolled `torch.utils.data.IterableDataset` subclass. `trl.SFTTrainer` requires
    `train_dataset` to be a `datasets.Dataset` / `datasets.IterableDataset` instance (it
    raises `TypeError` otherwise at trainer-construction time, even though a custom torch
    `IterableDataset` would iterate correctly on its own -- this bug is what this function
    replaces). Chaining `.map()`/`.filter()` on the loaded dataset keeps the returned
    object a real `datasets`-library type while preserving all sharding behavior:
      - Cross-rank sharding uses `datasets.distributed.split_dataset_by_node`, applied
        *before* `.map()`, so each rank only streams/transforms its own parquet shards
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
    if streaming:
        dataset = load_dataset("parquet", data_files=data_path, split="train", streaming=True)
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
            logger.info(f"Sharded streaming dataset across {world_size} ranks (this rank={rank}).")
        logger.info("Dataset loaded in streaming mode (length unknown ahead of time)")
    else:
        dataset = load_dataset("parquet", data_files=data_path, split="train")
        logger.info(f"Dataset loaded: {len(dataset)} samples")

    raw_columns = dataset.column_names
    dataset = dataset.map(
        _row_to_trl_sample,
        with_indices=True,
        fn_kwargs={"max_turns": max_turns, "seed": seed},
        remove_columns=raw_columns,
    )
    dataset = dataset.filter(lambda x: x["_keep"])
    dataset = dataset.remove_columns(["_keep"])
    return dataset


# ─── debug / dry-run helpers ──────────────────────────────────────────────────


def debug_dry_run(args):
    """
    Quick dry-run against a real GPU:
      1. Download the model, load processor + model.
      2. Pull one sample from the dataset, build (prompt, completion, images).
      3. Run the SAME `DataCollatorForVisionLanguageModeling` that `SFTTrainer` would
         use, on a mini-batch of 2 samples -- this is what actually catches
         VLM/packing/dataset-format incompatibilities, unlike a hand-rolled
         processor(...) call which can silently skip the real training code path.
      4. Run a forward + backward pass (no optimizer step) to make sure gradients flow.
    """
    logger.info("=== DEBUG DRY RUN ===")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")

    # Download model into a cache dir specific to this model (avoids reusing a
    # stale download from a previously-tested model in the same container).
    cache_dir = args.download_model or f"/tmp/{_local_cache_name(args.model_path)}"
    local_model = download_from_s3(args.model_path.rstrip("/"), cache_dir)

    logger.info("Loading model & processor...")
    processor = AutoProcessor.from_pretrained(local_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        local_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )

    # Load a couple of samples
    ds = load_dataset("parquet", data_files=args.data_path, split="train", streaming=True)
    it = iter(ds)
    samples = [next(it) for _ in range(2)]

    examples = []
    for sample in samples:
        prompt, completion, images = build_messages(
            conversations=sample["conversations"],
            image_bytes_list=sample.get("image_bytes", []),
            max_turns=args.max_turns,
        )
        if prompt is None:
            continue
        examples.append({"prompt": prompt, "completion": completion, "images": images})

    if not examples:
        raise RuntimeError("Could not build any valid (prompt, completion, images) example from the dataset sample.")

    for i, ex in enumerate(examples):
        n_img = len(ex["images"])
        n_turns = len(ex["prompt"]) + len(ex["completion"])
        logger.info(f"  sample[{i}]: turns={n_turns}, images={n_img}")

    # ── this is the part that actually exercises the SFTTrainer code path ──
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    # `completion_only_loss=True` mirrors the training config below: loss must only be
    # computed on the target "Action: ..." completion, never on prompt/history/images.
    collator = DataCollatorForVisionLanguageModeling(
        processor, max_length=args.max_seq_length, completion_only_loss=True
    )
    logger.info("Running DataCollatorForVisionLanguageModeling on the mini-batch...")
    batch = collator(examples)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"  {k}: {v.shape} ({v.dtype})")
        else:
            logger.info(f"  {k}: {type(v).__name__}")

    device = next(model.parameters()).device
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    logger.info("Running forward + backward pass...")
    outputs = model(**batch)
    if outputs.loss is None:
        raise RuntimeError("Model forward pass did not return a loss; check labels/collator output.")
    logger.info(f"Loss: {outputs.loss.item():.4f}")
    outputs.loss.backward()
    logger.info("Backward pass OK (gradients computed).")
    logger.info("=== DRY RUN PASSED ===")


# ─── main training ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TRL SFT for Minecraft VLM")
    parser.add_argument("--model_path", type=str, required=True, help="S3 or local path to model")
    parser.add_argument("--data_path", type=str, required=True, help="S3 glob or local path to parquet files")
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
    parser.add_argument("--debug", action="store_true", help="Dry-run: load 1 sample, collate, forward+backward, exit")
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
