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
Qwen3.5 adapter for verl.

Qwen3.5 (`transformers>=5.2.0`, model_type `"qwen3_5"` / `"qwen3_5_moe"`) is architecturally very
different from Qwen2-VL/Qwen2.5-VL/Qwen3-VL:

  1. Native early-fusion multimodal training: text/image/video tokens are processed by the *same*
     backbone from the start (no separate late-fusion "project vision tower output, then concat"
     step at the top level -- the fusion still happens by scattering vision embeddings into the
     token sequence, but the backbone itself is natively multimodal-aware, e.g. `mm_token_type_ids`
     is threaded through every layer).
  2. Hybrid attention backbone: decoder layers alternate between `Qwen3_5Attention` (standard full
     causal attention with M-RoPE) and `Qwen3_5LinearAttention` / "Gated DeltaNet" (a *linear*,
     chunk-recurrent attention with **no standard KV cache** and an inherent sequential dependency
     along the time axis), per `config.text_config.layer_types` (e.g. every 4th layer is full
     attention, default `full_attention_interval=4`).

Design decision (explicitly agreed with the user): this adapter intentionally does **not** attempt
to support verl's Ulysses sequence-parallelism for Qwen3.5. Splitting the sequence dimension across
GPUs (as Ulysses SP does) breaks the causal recurrence of the Gated-DeltaNet linear-attention layers
-- correctly supporting SP for this would require a dedicated ring/halo-exchange communication
scheme (similar to what's needed for Mamba/RWKV-style SSMs), which is out of scope here. Training
should use plain FSDP sharding (no `ulysses_sequence_parallel_size > 1`) for this model family.

Since Ulysses SP is unsupported, this adapter is much smaller than the qwen2_vl/qwen3_vl ones: we
don't need a standalone `get_rope_index` either, since `Qwen3_5ForConditionalGeneration.forward`
(like Qwen3-VL) fully delegates to the inner `Qwen3_5Model`, whose `forward` internally computes
M-RoPE position ids itself (via `compute_3d_position_ids`) as long as `mm_token_type_ids` is passed
through -- which the processor already returns alongside `input_ids`/`image_grid_thw`, so verl's
dataset/rollout code just needs to forward it, not recompute it.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch


@dataclass
class Qwen3_5CausalLMOutputForPPO:
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
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> Union[Tuple, Qwen3_5CausalLMOutputForPPO]:
    """
    Copy-paste of `Qwen3_5ForConditionalGeneration.forward`, with the final `lm_head` + loss
    computation swapped for verl's `FusedLinearForPPO` log_probs/entropy kernel -- same pattern as
    the Qwen3-VL adapter's `forward_for_ppo`.

    NOTE: `mm_token_type_ids` must be forwarded from the processor output (it's what
    `Qwen3_5Model.forward` uses internally to compute M-RoPE position ids via
    `compute_3d_position_ids`, given `position_ids=None`); verl's dataset/rollout code should pass
    it through as-is rather than trying to recompute rope position ids externally like it does for
    Qwen2-VL/Qwen3-VL.
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
        mm_token_type_ids=mm_token_type_ids,
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

    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        rope_deltas=outputs.rope_deltas,
    )
