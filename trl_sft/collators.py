"""Model helpers and VLM collator adapters for SFT training.

- `freeze_vision_tower`: freeze ViT + adapter (JARVIS-VLA Stage I recipe).
- `_ImmutableVisionCollatorAdapter`: give TRL's mutating VLM collator disposable
  sample containers so dataset rows stay pristine.
- `_MultiStepVLMCollator`: TRL's VLM collator subclass that masks non-assistant
  tokens to -100, training on every assistant turn (Stage III full-trajectory).
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

logger = logging.getLogger(__name__)

# ─── model helpers ─────────────────────────────────────────────────────────────

# Substrings matched (case-insensitively) against each submodule's own leaf name to
# find the vision tower. For Qwen2-VL/Qwen2.5-VL/Qwen3-VL/Qwen3.5-VL, ViT + adapter
# both live under one submodule named "visual", so freezing it matches JARVIS-VLA's
# Stage I recipe ("ViT + adapter frozen, only LLM trained"). Other hints are fallbacks
# for architectures that split encoder/adapter differently.
_VISION_SUBMODULE_HINTS = ("visual", "vision_tower", "vision_model", "image_encoder")


def freeze_vision_tower(model: torch.nn.Module) -> None:
    """
    Freeze the vision encoder + adapter, leaving only the LLM backbone trainable --
    JARVIS-VLA's Stage I recipe (Stage II unfreezes everything again once real image
    data is introduced, so only call this for `--text_only` runs).

    Walks `model.named_modules()` for a submodule whose own name matches
    `_VISION_SUBMODULE_HINTS` and freezes every parameter under it (skipping submodules
    already nested inside a frozen one). Raises `RuntimeError` if nothing matches --
    silently no-op'ing would look like Stage I ran correctly while actually training
    the full model.
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


class ImmutableVisionCollatorAdapter:
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


class MultiStepVLMCollator(DataCollatorForVisionLanguageModeling):
    """TRL's VLM collator, but only assistant turns contribute to the loss.

    The parent already injects images, renders the chat template, expands image tokens
    and pads; its `_collate_language_modeling` sets `labels = input_ids` (loss on every
    token except padding). We only override that labels step: assistant turns are
    delimited by `<|im_start|>assistant` ... `<|im_end|>`, so every token outside those
    spans (system/user/image/padding) is masked to -100. This reproduces JARVIS-VLA
    Stage III, where every "Action: ..." turn is a training target.
    """

    def _collate_language_modeling(self, examples):
        output = super()._collate_language_modeling(examples)
        tok = self.processor.tokenizer
        im_start = tok.convert_tokens_to_ids("<|im_start|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        assistant_id = tok.convert_tokens_to_ids("assistant")
        labels = output["input_ids"].clone()
        for b, row in enumerate(output["input_ids"].tolist()):
            in_assistant = False
            for t, tok_id in enumerate(row):
                if tok_id == im_start and t + 1 < len(row) and row[t + 1] == assistant_id:
                    in_assistant = True
                if not in_assistant:
                    labels[b, t] = -100
                if tok_id == im_end:
                    in_assistant = False
        labels[output["attention_mask"] == 0] = -100
        output["labels"] = labels
        return output
