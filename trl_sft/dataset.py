"""Dataset construction for Minecraft SFT: message building, tokenization, and
dataset loading for parquet trajectory and jsonl flat-QA formats.

Supports two on-disk layouts:
  - parquet trajectories (e.g. minecraft-text-action-dataset): each row is one
    trajectory with embedded `image_bytes` per turn.
  - jsonl flat QA (e.g. minecraft-vlp): each row is a short independent Q&A
    session with image paths relative to an `image_root`.

Also supports `text_only` mode (Stage I, no images) and `full_trajectory` mode
(Stage III, multi-step loss on every assistant turn).
"""

from __future__ import annotations

import io
import logging
import random
from typing import Dict, List, Optional, Tuple, Union

from datasets import concatenate_datasets, load_dataset
from PIL import Image

logger = logging.getLogger(__name__)


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

    Only the LAST assistant turn is the training target ("Action: ..."); everything
    before it is context that must NOT contribute to the loss. Splitting into
    prompt/completion (rather than a flat `messages` list) lets
    `SFTConfig(completion_only_loss=True)` mask the context out -- the only
    loss-masking TRL currently supports for VLM datasets (`assistant_only_loss` raises
    for VLMs).

    Args:
        conversations: list of {role, content[{type, text/image}]}
        image_bytes_list: list of JPEG bytes, one per user turn (with image)
        max_turns: maximum number of (user, assistant) pairs to include
        rng: `random.Random` for the history-length sample. Defaults to the global
            `random` module; callers needing per-sample determinism (e.g.
            `_row_to_trl_sample`) should pass their own instance.

    Returns:
        (prompt, completion, images): `prompt` is every turn except the final
        assistant turn; `completion` is a single-element list with just that turn;
        `images` is the flat, ordered list of decoded `PIL.Image`s matching `prompt`'s
        `{"type": "image"}` placeholders (pass as the sample's top-level "images" key,
        not embedded in `content`).
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
    # per (user, assistant) trajectory step. See
    # `build_messages_qa` for the FLAT/no-slicing indexing used by the jsonl format.
    selected_image_bytes = image_bytes_list[start_turn:] if image_bytes_list else []

    return _split_prompt_completion_with_images(selected_convs, selected_image_bytes)


def _split_prompt_completion_with_images(
    conversations: List[Dict],
    image_bytes_list: List[bytes],
) -> Tuple[List[Dict], List[Dict], List[Image.Image]]:
    """
    Shared tail used by `build_messages` (parquet trajectory rows) and
    `build_messages_qa` (flat QA/jsonl rows): walks `conversations`, keeps
    `{"type": "image"}` placeholders in-place while decoding the matching entry of the
    flat `image_bytes_list` (one entry per placeholder, in encounter order across the
    whole conversation) into a separate `images` list. Finally splits off the final
    assistant turn as `completion`, with everything before it as `prompt`.
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

    # TRL's collator requires #`{"type":"image"}` placeholders WITHIN `prompt` to equal
    # `len(images)`, or it raises mid-training. A small minority of source rows put an
    # image placeholder in the FINAL (assistant) turn -- which becomes `completion`, not
    # `prompt` -- so that placeholder goes uncounted while `images` still includes it.
    # Caught two real 16-GPU jobs crashing ~13h in on the one bad row out of hundreds of
    # thousands; checking here converts that into a silent per-row drop instead.
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


def _build_full_trajectory(
    conversations: List[Dict],
    image_bytes_list: List[bytes],
) -> Tuple[Optional[List[Dict]], Optional[List[Image.Image]]]:
    """One parquet trajectory row -> full (messages, images) for multi-step loss.

    Mirrors `_split_prompt_completion_with_images` (same content normalization + image
    decode) but keeps the WHOLE conversation: no prompt/completion split, so every
    assistant "Action: ..." turn stays a training target for `_MultiStepVLMCollator`.
    """
    if not conversations or len(conversations) < 2:
        return None, None
    if conversations[0]["role"] != "user":
        conversations = conversations[1:]
    if not conversations or conversations[-1]["role"] != "assistant":
        return None, None

    messages, images, image_idx = [], [], 0
    for conv in conversations:
        content_list = []
        for item in conv["content"]:
            if item.get("type") == "text":
                content_list.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image":
                if image_idx < len(image_bytes_list):
                    try:
                        images.append(Image.open(io.BytesIO(image_bytes_list[image_idx])).convert("RGB"))
                        content_list.append({"type": "image"})
                    except Exception as e:
                        logger.warning(f"Failed to decode image at idx {image_idx}: {e}")
                        content_list.append({"type": "text", "text": "[image]"})
                image_idx += 1
        messages.append({"role": conv["role"], "content": content_list})
    return messages, images


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


# TRL's `SFTTrainer` checks that tokenized `prompt` is a token-for-token PREFIX of
# tokenized `prompt+completion` to compute `completion_mask`. For Qwen3.x "thinking"
# chat templates, leaving `enable_thinking` unset breaks that: `add_generation_prompt=
# True` alone emits an unclosed `<think>\n`, while the full render's assistant turn
# emits a closed `<think>\n\n</think>\n\n` -- character-prefix but NOT token-prefix
# (verified: the tokenizer merges that boundary differently), so 2 boilerplate tokens
# leak into the loss on every sample. `enable_thinking=False` makes both renderings
# emit the same closed form, eliminating the mismatch.
#
# Only applied (see `_resolve_chat_template_kwargs`) when the processor's template
# actually references `enable_thinking` (Qwen3.x): passing it to a template that
# doesn't (Qwen2-VL/Qwen2.5-VL) renders identically but trips a per-sample
# `transformers` "kwargs not in processor_kwargs" warning -- avoided by gating on it.
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
    full_trajectory: bool = False,
) -> Dict:
    """
    Map ONE raw dataset row (parquet trajectory step OR jsonl QA session, per
    `data_format`) to a TRL prompt/completion/images sample.

    Used as the `function` arg of `datasets.Dataset.map`/`IterableDataset.map` (with
    `with_indices=True`), not a hand-rolled `torch.utils.data.IterableDataset`:
    `SFTTrainer.__init__` requires `train_dataset` to `isinstance`-check against
    `datasets.Dataset`/`IterableDataset`, and chaining `.map()` on `load_dataset(...)`'s
    return value preserves that (plus SFTTrainer/Accelerate's sharding).

    For parquet rows, uses a *local* RNG keyed by the sample's stable "id" (not `idx`)
    so process/worker sharding can't change a sample's history choice, and the global
    `random` module state isn't mutated as a side effect.

    `text_only=True` (Stage I) bypasses `build_messages`/`build_messages_qa` for
    `build_messages_text_only`, and the returned dict has NO "images" key at all --
    `SFTTrainer` picks its vision collator purely based on whether that key is
    *present*, so omitting it (not just leaving it empty) is what keeps Stage I on the
    plain-text path.

    Invalid rows get `"_keep": False` (a `.map()` fn must return one row per input row);
    the caller chains `.filter(lambda x: x["_keep"])` to actually drop them.

    If `processor`/`max_seq_length` are given, image-bearing samples are also
    length-checked (`_exceeds_max_length`) -- otherwise `SFTConfig`'s raw-token-level
    truncation can land inside an image's placeholder-token block, and the VLM forward
    pass crashes with a tokens/features mismatch (observed for real on Qwen2-VL-7B).
    For trajectory rows the fix is to trim the sampled history until it fits, instead
    of dropping the row outright.
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

    if full_trajectory:
        # Stage III multi-step loss: keep the whole parquet trajectory as a flat
        # `messages` list (not prompt/completion) so `_MultiStepVLMCollator` trains on
        # every assistant turn. Oversized trajectories are dropped (same rationale as
        # the prompt/completion path below: truncation through a vision-token block
        # crashes the forward pass).
        messages, images = _build_full_trajectory(sample["conversations"], sample.get("image_bytes", []))
        if messages is None:
            return {"messages": [], "images": [], "chat_template_kwargs": {}, "_keep": False}
        if images and processor is not None and max_seq_length is not None:
            if _exceeds_max_length(processor, messages[:-1], [messages[-1]], images, max_seq_length, chat_template_kwargs):
                return {"messages": [], "images": [], "chat_template_kwargs": {}, "_keep": False}
        return {"messages": messages, "images": images, "chat_template_kwargs": chat_template_kwargs, "_keep": True}

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
        # images) until the sample fits, instead of discarding the row outright. This
        # keeps the common parquet path 1:1 (one input row -> one training sample),
        # reduces rank-to-rank differences in effective shard lengths, and recovers
        # long-trajectory samples that used to be thrown away: only surplus history is
        # trimmed, never the current step or its target action.
        #
        # This is complementary to, but distinct from, process sharding. The dataset
        # returned by `build_minecraft_dataset` must remain UNSHARDED here; SFTTrainer /
        # Accelerate performs process-level sharding exactly once when preparing the
        # DataLoader. Manually calling `split_dataset_by_node` before that caused every
        # rank's stream to be divided by world_size a second time (N/world_size^2), which
        # was the actual cause of the deterministic step-53/105 exhaustion and mismatched
        # NCCL collectives. Rare malformed or irreducibly oversized rows may still be
        # filtered, so callers should keep a small margin below the theoretical epoch end.
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
    produce, without paying for the actual (expensive) pixel resize/rescale/normalize/
    patchify work -- only each image's (height, width) is needed.

    Uses `Qwen2VLProcessor`/`Qwen3VLProcessor`'s own `_get_num_multimodal_tokens(...)`
    (pure arithmetic on image size + patch_size/merge_size/min_pixels/max_pixels) for
    each image's vision-token count, combined with a plain-text tokenization of
    `rendered` (pre-expansion, one placeholder token per image) to reconstruct the
    post-expansion length: `base_len - num_images + sum(num_image_tokens_per_image)`.
    Verified byte-exact against real `processor(...)` output for 1-5 images across a
    range of sizes. Raises if `processor` lacks `_get_num_multimodal_tokens` (caller
    falls back to the exact-but-slow path).
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
    chat-template about to be used, and check whether it overflows `max_seq_length`.

    Pre-filters samples that would otherwise crash training mid-run: an overflowing
    VLM sample risks the truncation point landing inside an image's placeholder-token
    block, which crashes the forward pass with a tokens/features mismatch (unlike
    plain-text overflow, which the collator truncates away harmlessly). Any overflow on
    an image-bearing sample is therefore treated as unsafe and dropped.

    Tries the cheap `_fast_encoded_length` path first (avoids redoing the same
    expensive image preprocessing the collator is about to do for real -- this was
    silently capping GPU utilization around 30%), falling back to the exact
    `processor(...)` call if that raises. Any error from both paths drops the sample
    defensively rather than propagating out of a `.map()` call.
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
    except when `data_path` is a `list`: those files are loaded and trimmed to
    `_USED_COLUMNS` ONE AT A TIME then stitched with `concatenate_datasets`, instead of
    a single `load_dataset(..., data_files=[...])` call.

    Needed because `minecraft-vlp`'s jsonl files (combined for JARVIS-VLA Stage II)
    have wildly different schemas for columns training never uses (e.g. `source`'s
    struct shape differs per file, one file even has a typo'd `datatime` key). A single
    combined `load_dataset` call tries to unify ALL files into one Arrow schema up
    front and hard-crashes (`TypeError: Couldn't cast array...`) the moment it hits a
    disagreeing file; loading each file separately never exposes pyarrow to more than
    one schema at a time, sidestepping this entirely (verified against the real 5-file
    Stage II combination).
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
    full_trajectory: bool = False,
):
    """
    Build the Minecraft SFT dataset as a genuine `datasets.Dataset` (`streaming=False`)
    or `datasets.IterableDataset` (`streaming=True`). Samples are either
    `{"prompt": [...], "completion": [...], "images": [...]}` (default -- lets
    `SFTConfig(completion_only_loss=True)` mask context out of the loss) or, with
    `full_trajectory=True`, `{"messages": [...], "images": [...]}` for
    `_MultiStepVLMCollator`'s per-step masking instead.

    Supports two on-disk layouts, auto-detected from `data_path`'s extension (override
    with `data_format=`):
      - "parquet" (e.g. `minecraft-text-action-dataset`): each row is one trajectory
        (`conversations` + one `image_bytes` entry per turn pair). `build_messages`
        randomly samples a suffix history window (`max_turns`), or (with
        `full_trajectory=True`) `_build_full_trajectory` keeps the whole thing.
      - "jsonl" (e.g. `minecraft-vlp`): each row is a short, independent Q&A session
        with an `image` field listing path(s) relative to `image_root` (default: the
        directory containing the jsonl file). `build_messages_qa` loads those on-the-fly
        via `fsspec` and keeps the whole (already-short) conversation.

    `text_only=True` (Stage I, no images): routes every row through
    `build_messages_text_only` instead, and the resulting samples carry NO "images" key
    at all -- `SFTTrainer` picks its vision collator purely based on whether that key
    is *present*. Combine with `freeze_vision_tower(model)` to match JARVIS-VLA's Stage
    I recipe (ViT+adapter frozen, only LLM trained).

    IMPORTANT: returns the result of chaining `.map()`/`.filter()`/`.remove_columns()`
    directly on `datasets.load_dataset(...)`'s return value -- NOT a hand-rolled
    `torch.utils.data.IterableDataset` (which `SFTTrainer` rejects with `TypeError` at
    construction time). The dataset is deliberately returned FULL and UNSHARDED:
    SFTTrainer/Accelerate performs the one and only process-level shard while preparing
    the DataLoader; calling `split_dataset_by_node` here would divide each stream by
    world_size a second time (deterministic early exhaustion + mismatched collectives).

    `processor`/`max_seq_length`: when both given, image-bearing samples whose REAL
    tokenized length would overflow `max_seq_length` are dropped instead of silently
    truncated (see `_exceeds_max_length` for why blind truncation crashes VLM training).
    Pass the same `AutoProcessor` used for the model. Omit both to skip this check
    (e.g. for `--text_only` data, which has no images and thus no risk of this crash).
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
        # Return the complete HF IterableDataset. SFTTrainer/Accelerate shards it once
        # while preparing the DataLoader. Pre-sharding here caused double sharding:
        # N/world_size^2 samples per process and deterministic early stream exhaustion.
        logger.info(
            f"Dataset loaded in streaming mode (format={data_format}, length unknown ahead of time); "
            "process sharding is delegated to SFTTrainer/Accelerate"
        )
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
            "full_trajectory": full_trajectory,
        },
        remove_columns=raw_columns,
    )
    dataset = dataset.filter(lambda x: x["_keep"])
    dataset = dataset.remove_columns(["_keep"])
    return dataset
