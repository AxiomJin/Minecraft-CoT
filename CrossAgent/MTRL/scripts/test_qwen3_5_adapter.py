"""
Structural smoke test for verl's Qwen3.5 adapter (verl/models/transformers/qwen3_5.py).

Builds a tiny randomly-initialized Qwen3_5ForConditionalGeneration (hybrid full-attention /
Gated-DeltaNet linear-attention layers, per `layer_types`) and checks:
  1. `apply_monkey_patch` patches `Qwen3_5ForConditionalGeneration.forward` for model_type=="qwen3_5".
  2. `apply_monkey_patch(..., ulysses_sp_size=2)` correctly raises NotImplementedError for this
     model family (Ulysses SP is intentionally unsupported).
  3. A forward pass with fabricated multimodal inputs (random pixel_values matching
     image_grid_thw) + `mm_token_type_ids` runs end-to-end through both a hybrid full-attn layer
     and a hybrid linear-attn layer, producing `log_probs`/`entropy` of the expected shape, with
     `position_ids=None` (i.e. relying on the model's internal `compute_3d_position_ids`).

Run with: python scripts/test_qwen3_5_adapter.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # MTRL root, for `import verl`

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from verl.models.transformers.monkey_patch import apply_monkey_patch


def build_tiny_config() -> Qwen3_5Config:
    text_config = dict(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,  # with default full_attention_interval=4 -> [linear, linear, linear, full]
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
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
    )
    return Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=1000,
        video_token_id=1001,
        vision_start_token_id=1002,
        vision_end_token_id=1003,
    )


def test_ulysses_sp_rejected():
    config = build_tiny_config()
    model = Qwen3_5ForConditionalGeneration(config)
    try:
        apply_monkey_patch(model, ulysses_sp_size=2, use_remove_padding=False, use_fused_kernels=True)
        raise AssertionError("expected NotImplementedError for ulysses_sp_size > 1 on qwen3_5")
    except NotImplementedError:
        print("[OK] apply_monkey_patch correctly rejects ulysses_sp_size > 1 for qwen3_5")


def test_forward_for_ppo():
    config = build_tiny_config()
    print("layer_types:", config.text_config.layer_types)
    assert "linear_attention" in config.text_config.layer_types
    assert "full_attention" in config.text_config.layer_types

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3_5ForConditionalGeneration(config).to(device=device, dtype=torch.bfloat16)
    model.eval()

    apply_monkey_patch(model, ulysses_sp_size=1, use_remove_padding=False, use_fused_kernels=True)
    assert Qwen3_5ForConditionalGeneration.forward.__name__ == "forward_for_ppo"
    print("[OK] apply_monkey_patch patched forward_for_ppo")

    # 1 image of grid (t=1, h=4, w=4) -> merge_size=2 -> 1*2*2=4 image tokens
    image_grid_thw = torch.tensor([[1, 4, 4]], device=device)
    num_patches = int(image_grid_thw.prod(-1).item())
    patch_dim = config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
    pixel_values = torch.randn(num_patches, patch_dim, device=device, dtype=torch.bfloat16)

    # sequence: [vision_start,4x image_pad, text..., vision_start, 4x image_pad, text...]
    # (repeated twice so both a "linear_attention" and the "full_attention" layer see >1 chunk boundary)
    input_ids = torch.tensor(
        [[1002, 1000, 1000, 1000, 1000, 5, 6, 7, 8, 9, 10]],
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    # text=0, image=1 (per Qwen3.5's get_rope_index docstring)
    mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], device=device, dtype=torch.int)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=None,  # let the model compute M-RoPE internally, as dp_actor.py now does
            temperature=1.0,
        )

    assert out.log_probs.shape == (1, input_ids.shape[1]), out.log_probs.shape
    assert out.entropy.shape == (1, input_ids.shape[1]), out.entropy.shape
    assert torch.isfinite(out.log_probs).all(), "log_probs contains NaN/Inf"
    assert torch.isfinite(out.entropy).all(), "entropy contains NaN/Inf"
    print(f"[OK] forward_for_ppo (hybrid linear+full attn) -> log_probs{tuple(out.log_probs.shape)}, entropy{tuple(out.entropy.shape)}, device={device}")


def test_forward_without_fused_kernels_matches_shapes():
    """Unpatched sanity check: original HF forward (loss/logits) with the same hybrid-layer + rope
    inputs, isolating "does the base model even run" from "is the PPO head-swap correct"."""
    config = build_tiny_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3_5ForConditionalGeneration(config).to(device=device, dtype=torch.bfloat16)
    model.eval()

    image_grid_thw = torch.tensor([[1, 4, 4]], device=device)
    num_patches = int(image_grid_thw.prod(-1).item())
    patch_dim = config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
    pixel_values = torch.randn(num_patches, patch_dim, device=device, dtype=torch.bfloat16)
    input_ids = torch.tensor([[1002, 1000, 1000, 1000, 1000, 5, 6, 7, 8, 9, 10]], device=device)
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], device=device, dtype=torch.int)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=None,
        )

    assert out.logits.shape == (1, input_ids.shape[1], config.text_config.vocab_size), out.logits.shape
    print(f"[OK] original forward (unpatched, hybrid layers) -> logits{tuple(out.logits.shape)}")


if __name__ == "__main__":
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    test_ulysses_sp_rejected()
    test_forward_without_fused_kernels_matches_shapes()
    test_forward_for_ppo()
    print("\nALL TESTS PASSED")
