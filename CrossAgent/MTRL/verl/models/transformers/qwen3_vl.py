# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qwen3-VL adapter for verl.

Unlike Qwen2-VL/Qwen2.5-VL, Qwen3-VL:
  1. Uses "interleaved" M-RoPE (frequencies interleaved as [T,H,W,T,H,W,...,T,T]
     instead of "chunked" [T,T,...,H,H,...,W,W,...]). The interleaving itself is
     performed inside `Qwen3VLTextRotaryEmbedding.forward` (i.e. once `cos`/`sin`
     are produced by the rotary embedding module they can be applied with the
     regular (non-multimodal) `apply_rotary_pos_emb`), so no special
     `apply_multimodal_rotary_pos_emb` is required here.
  2. Encodes temporal information for videos via injected timestamp tokens
     rather than absolute `t` position ids (see `get_rope_index` below,
     ported from `Qwen3VLModel.get_rope_index` in transformers).
  3. Uses the new unified `ALL_ATTENTION_FUNCTIONS` attention-interface
     (`Qwen3VLTextAttention.forward` looks up the implementation by
     `config._attn_implementation` instead of dispatching to a dedicated
     `Qwen3VLFlashAttention2` subclass), so the Ulysses patch below replaces
     `Qwen3VLTextAttention.forward` directly instead of a `FlashAttention2`
     subclass' `forward`.
  4. `Qwen3VLForConditionalGeneration.forward` fully delegates multimodal
     embedding fusion (incl. "DeepStack" visual feature injection) and rope
     computation to the inner `Qwen3VLModel`, so `forward_for_ppo` below only
     needs to swap the final `lm_head` + loss computation for verl's fused
     log_probs/entropy kernel -- no need to duplicate the embedding-fusion
     logic like the Qwen2-VL adapter does.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)

from .qwen2_vl import flash_attention_forward


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Same as `transformers.models.qwen2_vl.modeling_qwen2_vl.repeat_kv` (also identical to
    `transformers.models.llama.modeling_llama.repeat_kv`); duplicated here so this adapter doesn't
    depend on `repeat_kv` continuing to be (re-)exported from the qwen2_vl modeling module."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,  # noqa: ARG001 - kept for call-site compatibility
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen3-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.

    Ported from `Qwen3VLModel.get_rope_index` (single-example, no batch dim), see:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py

    Note: `second_per_grid_ts` is accepted (and ignored) purely so this function is a drop-in
    replacement for `verl.models.transformers.qwen2_vl.get_rope_index` at call sites. Qwen3-VL does
    not use it: video temporal information is encoded via injected timestamp tokens instead of
    absolute `t` position ids, so every video's `grid_thw` is first split into one-frame-per-entry
    (mirroring `Qwen3VLModel.get_rope_index`) and `t_index` is always 0.
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")

    if video_grid_thw is not None:
        # Qwen3-VL separates video frames with timestamp tokens, so each frame is treated
        # as an independent "image" of temporal size 1 when computing position ids.
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)  # (3, seqlen)
        image_index, video_index = 0, 0
        input_ids = input_ids[attention_mask == 1]
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # t_index is always 0: Qwen3-VL uses injected timestamp tokens (not absolute t
            # position ids) to encode temporal information, incl. for videos.
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def ulysses_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Ulysses-SP-aware replacement for `Qwen3VLTextAttention.forward`.

    Qwen3-VL (unlike Qwen2-VL) has no dedicated `*FlashAttention2` subclass to monkey-patch:
    the attention implementation is resolved dynamically via `ALL_ATTENTION_FUNCTIONS`, and the
    (interleaved-)M-RoPE `cos`/`sin` are already computed by `Qwen3VLTextRotaryEmbedding` and
    applied with the plain (non-multimodal) `apply_rotary_pos_emb`. So we only need to replace
    `forward` itself, insert the Ulysses all-to-all around flash-attention, and keep q_norm/k_norm
    identical to the original implementation.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        validate_ulysses_config(query_states.size(1), ulysses_sp_size)

        n_rep = max(ulysses_sp_size // key_states.size(1), 1)
        key_states = _repeat_kv(key_states, n_rep)
        value_states = _repeat_kv(value_states, n_rep)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    attn_output, _ = flash_attention_forward(
        self,
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        is_causal=self.is_causal,
    )

    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


@dataclass
class Qwen3VLCausalLMOutputForPPO:
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


def forward_for_ppo(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> Union[Tuple, Qwen3VLCausalLMOutputForPPO]:
    """
    Copy-paste of `Qwen3VLForConditionalGeneration.forward`, with the final `lm_head` + loss
    computation swapped for verl's `FusedLinearForPPO` log_probs/entropy kernel.

    Unlike the Qwen2-VL adapter, we don't need to re-implement multimodal-embedding fusion or
    rope-index caching here: `Qwen3VLForConditionalGeneration.forward` itself already just calls
    `self.model(...)` (which internally handles image/video/DeepStack fusion + `get_rope_index`),
    so we do the same and only replace what comes after.
    """
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_for_ppo, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        rope_deltas=outputs.rope_deltas,
    )
