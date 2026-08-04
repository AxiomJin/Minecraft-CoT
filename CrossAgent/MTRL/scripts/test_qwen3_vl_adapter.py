"""
Structural smoke test for verl's Qwen3-VL adapter (verl/models/transformers/qwen3_vl.py).

This builds a tiny randomly-initialized Qwen3VLForConditionalGeneration (no real checkpoint
download needed) and checks:
  1. `apply_monkey_patch` correctly dispatches on model_type == "qwen3_vl" and patches
     `Qwen3VLTextAttention.forward` / `Qwen3VLForConditionalGeneration.forward`.
  2. A forward pass with fabricated multimodal inputs (random pixel_values matching
     image_grid_thw) runs end-to-end through the patched `forward_for_ppo` and produces
     `log_probs`/`entropy` of the expected shape.
  3. The standalone `get_rope_index` produces positionids of shape (3, seq_len) without
     crashing, using a minimal fake processor (no real tokenizer/checkpoint needed).

Run with: python scripts/test_qwen3_vl_adapter.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# MTRL root, for `import verl`

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextAttention,
)

from verl.models.transformers.monkey_patch import apply_monkey_patch


def build_tiny_config() -> Qwen3VLConfig:
    text_config = dict(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]},
    )
    vision_config = dict(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=64,
        num_position_embeddings=64,
        deepstack_visual_indexes=[1],
    )
    return Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=1000,
        video_token_id=1001,
        vision_start_token_id=1002,
        vision_end_token_id=1003,
    )


class _FakeTokenizer:
    _map = {"<|image_pad|>": 1000, "<|video_pad|>": 1001, "<|vision_start|>": 1002}

    def convert_tokens_to_ids(self, tok):
        return self._map[tok]


class _FakeImageProcessor:
    merge_size = 2


class _FakeProcessor:
    __class__ = type("Qwen3VLProcessor", (), {})  # so `__class__.__name__ == "Qwen3VLProcessor"`
    tokenizer = _FakeTokenizer()
    image_processor = _FakeImageProcessor()


def test_get_rope_index():
    from verl.models.transformers.qwen3_vl import get_rope_index

    # sequence: [vision_start, image_pad]*prod(grid)/merge^2, text...]
    image_grid_thw = torch.tensor([[1, 4, 4]])  # -> llm grid (1, 2, 2) = 4 image tokens
    input_ids = torch.tensor([1002, 1000, 1000, 1000, 1000, 5, 6, 7])
    attention_mask = torch.ones_like(input_ids)

    position_ids = get_rope_index(
        _FakeProcessor(),
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
    )
    assert position_ids.shape == (3, input_ids.shape[0]), position_ids.shape
    print("[OK] get_rope_index ->", position_ids.tolist())


def test_forward_for_ppo():
    config = build_tiny_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLForConditionalGeneration(config).to(device=device, dtype=torch.bfloat16)
    model.eval()

    # use_remove_padding=False: no real flash-attn build available in this sandbox (see
    # test_ulysses_flash_attn_forward_shapes for an isolated test of that path with a stub).
    apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=False, use_fused_kernels=True)
    assert Qwen3VLForConditionalGeneration.forward.__name__ == "forward_for_ppo"
    print("[OK] apply_monkey_patch patched forward_for_ppo")

    # 1 image of grid (t=1, h=4, w=4) -> merge_size=2 -> 1*2*2=4 image tokens
    image_grid_thw = torch.tensor([[1, 4, 4]], device=device)
    num_patches = int(image_grid_thw.prod(-1).item())  # 16 raw patches before spatial merge
    patch_dim = config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
    pixel_values = torch.randn(num_patches, patch_dim, device=device, dtype=torch.bfloat16)

    input_ids = torch.tensor(
        [[1002, 1000, 1000, 1000, 1000, 5, 6, 7]],  # vision_start + 4 image tokens + 3 text tokens
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            temperature=1.0,
        )

    assert out.log_probs.shape == (1, input_ids.shape[1]), out.log_probs.shape
    assert out.entropy.shape == (1, input_ids.shape[1]), out.entropy.shape
    assert torch.isfinite(out.log_probs).all(), "log_probs contains NaN/Inf"
    assert torch.isfinite(out.entropy).all(), "entropy contains NaN/Inf"
    print(f"[OK] forward_for_ppo -> log_probs{tuple(out.log_probs.shape)}, entropy{tuple(out.entropy.shape)}, device={device}")


def test_forward_without_fused_kernels_matches_shapes():
    """Sanity check: with use_remove_padding/use_fused_kernels both off, no monkey-patch is applied
    at all (plain HF sdpa/eager attention, no flash-attn dependency) -- the original HF forward
    (loss/logits) should still run without shape errors. This isolates "does get_rope_index /
    multimodal embedding fusion / rope even work" from "is the Ulysses flash-attn patch correct",
    since a real flash-attn build isn't available in this sandbox (prebuilt wheel ABI mismatch)."""
    config = build_tiny_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLForConditionalGeneration(config).to(device=device, dtype=torch.bfloat16)
    model.eval()

    apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=False, use_fused_kernels=False)

    image_grid_thw = torch.tensor([[1, 4, 4]], device=device)
    num_patches = int(image_grid_thw.prod(-1).item())
    patch_dim = config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
    pixel_values = torch.randn(num_patches, patch_dim, device=device, dtype=torch.bfloat16)
    input_ids = torch.tensor([[1002, 1000, 1000, 1000, 1000, 5, 6, 7]], device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    assert out.logits.shape == (1, input_ids.shape[1], config.text_config.vocab_size), out.logits.shape
    print(f"[OK] original forward (sdpa/eager, unpatched) -> logits{tuple(out.logits.shape)}")


def test_ulysses_flash_attn_forward_shapes():
    """Isolated shape/logic test for `ulysses_flash_attn_forward` that stubs out the actual
    flash-attn kernel call (`flash_attention_forward`), since no working flash-attn build is
    available in this sandbox (prebuilt wheel ABI mismatch against this torch build). This still
    exercises: q_norm/k_norm, `apply_rotary_pos_emb` with the (interleaved-)mrope cos/sin produced
    by `Qwen3VLTextRotaryEmbedding`, the KV cache update call-site, and the final reshape/o_proj --
    i.e. everything in `ulysses_flash_attn_forward` except the actual attention kernel math."""
    import verl.models.transformers.qwen3_vl as qwen3_vl_adapter

    config = build_tiny_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLForConditionalGeneration(config).to(device=device, dtype=torch.bfloat16)
    model.eval()

    def _stub_flash_attention_forward(module, query_states, key_states, value_states, attention_mask, **kwargs):
        # query/key/value are (batch, seqlen, num_heads, head_dim) here (already transposed
        # back by the caller); just return something of the expected shape.
        return query_states.reshape(*query_states.shape[:-2], -1), None

    original = qwen3_vl_adapter.flash_attention_forward
    qwen3_vl_adapter.flash_attention_forward = _stub_flash_attention_forward
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    original_attn_forward = Qwen3VLTextAttention.forward
    try:
        Qwen3VLTextAttention.forward = qwen3_vl_adapter.ulysses_flash_attn_forward

        image_grid_thw = torch.tensor([[1, 4, 4]], device=device)
        num_patches = int(image_grid_thw.prod(-1).item())
        patch_dim = (
            config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
        )
        pixel_values = torch.randn(num_patches, patch_dim, device=device, dtype=torch.bfloat16)
        input_ids = torch.tensor([[1002, 1000, 1000, 1000, 1000, 5, 6, 7]], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

        assert out.logits.shape == (1, input_ids.shape[1], config.text_config.vocab_size), out.logits.shape
        print(f"[OK] ulysses_flash_attn_forward (stubbed kernel) -> logits{tuple(out.logits.shape)}")
    finally:
        qwen3_vl_adapter.flash_attention_forward = original
        Qwen3VLTextAttention.forward = original_attn_forward


if __name__ == "__main__":
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    test_get_rope_index()
    test_forward_without_fused_kernels_matches_shapes()
    test_ulysses_flash_attn_forward_shapes()
    test_forward_for_ppo()
    print("\nALL TESTS PASSED")

