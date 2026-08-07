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
    "images" (or "image") key. Because of that:
      * dataset samples MUST expose {"messages": [...], "images": [...]} -- images must
        NOT be embedded inside `content` blocks (only `{"type": "image"}` placeholders are
        embedded there; the actual `PIL.Image` objects live in the separate "images" list).
      * `packing=True` is not supported for VLMs by TRL and will raise at trainer init time.
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
) -> Tuple[Optional[List[Dict]], Optional[List[Image.Image]]]:
    """
    Convert a single parquet row into TRL-compatible (messages, images).

    Args:
        conversations: list of {role, content[{type, text/image}]}
        image_bytes_list: list of JPEG bytes, one per user turn (with image)
        max_turns: maximum number of (user, assistant) pairs to include

    Returns:
        (messages, images):
          - messages: OpenAI-chat-format messages. Image content blocks only carry
            `{"type": "image"}` placeholders (no payload) so that TRL's
            `prepare_multimodal_messages` fills them in from `images`, in order.
          - images: flat, ordered list of decoded `PIL.Image` objects matching the
            placeholders above (this must be passed as the top-level "images" key of
            the dataset sample, NOT embedded in `content`).
    """
    if not conversations or len(conversations) < 2:
        return None, None

    # Count total turns
    total_turns = len(conversations) // 2

    # Random history length: 0 to min(total_turns-1, max_turns)
    max_possible = min(total_turns - 1, max_turns)
    if max_possible < 0:
        return None, None
    history_len = random.randint(0, max_possible)

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

    return messages, images


class MinecraftParquetDataset(torch.utils.data.IterableDataset):
    """
    Streaming-first PyTorch `IterableDataset` over the Minecraft parquet shards.

    Yields TRL-compatible samples: {"messages": [...], "images": [PIL.Image, ...]}.
    Images are decoded from bytes on-the-fly (not cached).

    When `streaming=False`, the underlying HF dataset is fully materialized and
    random-access (`__getitem__`/`__len__`) is also supported; when `streaming=True`
    (the default for large S3 datasets), only iteration is supported -- `len()` is
    undefined, matching `datasets.IterableDataset` semantics.

    For multi-process distributed training, each rank shards the stream by
    `RANK`/`WORLD_SIZE` (env vars set by `torchrun`) so ranks don't train on
    duplicate data.
    """

    def __init__(
        self,
        data_path: str,
        max_turns: int = 4,
        streaming: bool = False,
        seed: int = 42,
    ):
        self.max_turns = max_turns
        self.seed = seed
        self.streaming = streaming

        if streaming:
            self.dataset = load_dataset(
                "parquet",
                data_files=data_path,
                split="train",
                streaming=True,
            )
            logger.info("Dataset loaded in streaming mode (length unknown ahead of time)")
        else:
            self.dataset = load_dataset(
                "parquet",
                data_files=data_path,
                split="train",
            )
            logger.info(f"Dataset loaded: {len(self.dataset)} samples")

    def _to_sample(self, sample, idx: int) -> Optional[Dict]:
        random.seed(self.seed + idx)
        messages, images = build_messages(
            conversations=sample["conversations"],
            image_bytes_list=sample.get("image_bytes", []),
            max_turns=self.max_turns,
        )
        if messages is None:
            return None
        return {"messages": messages, "images": images}

    def __iter__(self):
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        for idx, sample in enumerate(self.dataset):
            if idx % world_size != rank:
                continue
            record = self._to_sample(sample, idx)
            if record is None:
                continue
            yield record

    def __len__(self) -> int:
        if self.streaming:
            raise TypeError(
                "MinecraftParquetDataset has no defined length while streaming=True "
                "(it is a torch IterableDataset over a streamed HF dataset)."
            )
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        if self.streaming:
            raise TypeError(
                "Random access (__getitem__) is not supported while streaming=True; "
                "iterate the dataset instead."
            )
        sample = self.dataset[idx]
        record = self._to_sample(sample, idx)
        if record is None:
            record = {"messages": [{"role": "user", "content": [{"type": "text", "text": "fallback"}]}], "images": []}
        return record


# ─── debug / dry-run helpers ──────────────────────────────────────────────────


def debug_dry_run(args):
    """
    Quick dry-run against a real GPU:
      1. Download the model, load processor + model.
      2. Pull one sample from the dataset, build (messages, images).
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
        messages, images = build_messages(
            conversations=sample["conversations"],
            image_bytes_list=sample.get("image_bytes", []),
            max_turns=args.max_turns,
        )
        if messages is None:
            continue
        examples.append({"messages": messages, "images": images})

    if not examples:
        raise RuntimeError("Could not build any valid (messages, images) example from the dataset sample.")

    for i, ex in enumerate(examples):
        n_img = len(ex["images"])
        n_turns = len(ex["messages"])
        logger.info(f"  sample[{i}]: turns={n_turns}, images={n_img}")

    # ── this is the part that actually exercises the SFTTrainer code path ──
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    collator = DataCollatorForVisionLanguageModeling(processor, max_length=args.max_seq_length)
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
    # For large S3 datasets, use streaming to avoid downloading everything
    dataset = MinecraftParquetDataset(
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
