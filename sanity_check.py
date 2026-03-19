"""
sanity_check.py — DCM Integration Test
=======================================
Minimal integration test that verifies all DCM components handshake correctly
WITHOUT requiring a full training run or real data.

Tests:
    1. SSM Encoder compresses token embeddings → fixed-length latents z₀
    2. Diffuser adds noise (forward) and denoises (reverse) → ẑ₀
    3. Qwen LoRA Head accepts denoised memory as soft prefixes
    4. Full pipeline produces a valid probability distribution for next token
    5. No OOM or tensor shape mismatches on target hardware

Usage:
    python sanity_check.py                 # Runs on available GPU(s)
    python sanity_check.py --cpu           # Force CPU (slower but works anywhere)
    python sanity_check.py --skip_qwen     # Test encoder + diffuser only (no model download)
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_ssm_encoder(device: torch.device):
    """Test that DCM_SSMEncoder compresses variable-length input to fixed latents."""
    from dcm_model import DCMConfig, DCM_SSMEncoder

    separator("Test 1: SSM Encoder")

    cfg = DCMConfig(
        ssm_input_dim=256,   # Small dims for testing
        latent_dim=256,
        ssm_state_dim=16,
        ssm_num_layers=2,
        num_latent_vectors=8,
    )
    encoder = DCM_SSMEncoder(cfg).to(device)

    B, L, D = 2, 100, cfg.ssm_input_dim  # 100 tokens → 8 latent vectors
    x = torch.randn(B, L, D, device=device)

    z0 = encoder(x)

    assert z0.shape == (B, cfg.num_latent_vectors, cfg.latent_dim), \
        f"Expected shape {(B, cfg.num_latent_vectors, cfg.latent_dim)}, got {z0.shape}"

    print(f"  Input:  {x.shape}  (B={B}, L={L}, D={D})")
    print(f"  Output: {z0.shape} (B={B}, M={cfg.num_latent_vectors}, D={cfg.latent_dim})")
    print(f"  z0 stats: mean={z0.mean():.4f}, std={z0.std():.4f}")
    print("  PASSED")
    return cfg


def test_diffuser(device: torch.device, cfg):
    """Test forward + reverse diffusion on latent vectors."""
    from dcm_model import DCM_LatentDiffuser

    separator("Test 2: Latent Diffuser")

    diffuser = DCM_LatentDiffuser(cfg).to(device)

    B, M, D = 2, cfg.num_latent_vectors, cfg.latent_dim
    z0 = torch.randn(B, M, D, device=device)

    # Forward diffusion
    t = torch.randint(0, cfg.diffusion_steps, (B,), device=device)
    z_t, noise = diffuser.q_sample(z0, t)
    assert z_t.shape == z0.shape, f"q_sample shape mismatch: {z_t.shape} vs {z0.shape}"
    print(f"  Forward diffusion: z0 {z0.shape} → z_t {z_t.shape} at t={t.tolist()}")

    # Reverse diffusion (single step prediction)
    z0_pred = diffuser.predict_z0(z_t, t)
    assert z0_pred.shape == z0.shape, f"predict_z0 shape mismatch"
    print(f"  Reverse prediction: z_t → ẑ₀ {z0_pred.shape}")

    # Diffusion loss
    loss, z0_pred2 = diffuser.diffusion_loss(z0)
    assert loss.dim() == 0, "Loss should be scalar"
    print(f"  Diffusion MSE loss: {loss.item():.4f}")

    # Full reverse sampling from noise
    z_noise = torch.randn(B, M, D, device=device)
    z_sampled = diffuser.sample(z_noise, num_steps=10)
    assert z_sampled.shape == z0.shape, f"Sample shape mismatch"
    print(f"  Full reverse sample (10 steps): {z_sampled.shape}")
    print("  PASSED")


def test_full_pipeline_no_qwen(device: torch.device):
    """
    Test encoder + diffuser integration without loading Qwen.
    Verifies the data flow: tokens → z₀ → zₜ → ẑ₀.
    """
    from dcm_model import DCMConfig, DCM_SSMEncoder, DCM_LatentDiffuser

    separator("Test 3: Encoder + Diffuser Pipeline (no Qwen)")

    cfg = DCMConfig(
        ssm_input_dim=256,
        latent_dim=256,
        ssm_state_dim=16,
        ssm_num_layers=2,
        num_latent_vectors=8,
        denoiser_hidden_dim=128,
        denoiser_num_layers=2,
    )

    encoder = DCM_SSMEncoder(cfg).to(device)
    diffuser = DCM_LatentDiffuser(cfg).to(device)

    # Simulate 100 random token embeddings as context
    B, L = 2, 100
    token_embeds = torch.randn(B, L, cfg.ssm_input_dim, device=device)

    # Encode → latents
    z0 = encoder(token_embeds)
    print(f"  Encoded {L} tokens → {z0.shape[1]} latent vectors")

    # Diffusion round-trip
    loss, z0_pred = diffuser.diffusion_loss(z0)
    print(f"  Diffusion loss: {loss.item():.4f}")
    print(f"  ẑ₀ shape: {z0_pred.shape}, stats: mean={z0_pred.mean():.4f}, std={z0_pred.std():.4f}")

    # Simulate what would happen when feeding to LLM:
    # z0_pred serves as (B, M, D) soft prompt prefix
    fake_token_embeds = torch.randn(B, 50, cfg.latent_dim, device=device)  # continuation
    combined = torch.cat([z0_pred, fake_token_embeds], dim=1)
    print(f"  Combined prefix+tokens: {combined.shape} (M={z0_pred.shape[1]} + L=50)")

    # Verify shapes flow correctly
    assert combined.shape == (B, cfg.num_latent_vectors + 50, cfg.latent_dim)
    print("  PASSED — all shapes validated for LLM injection")


def test_full_pipeline_with_qwen(device: torch.device):
    """
    Full end-to-end test WITH Qwen loaded in 4-bit.
    Synthesizes 100 random tokens, encodes context, diffuses memory,
    and verifies the LLM produces a probability distribution for token 101.
    """
    from dcm_model import DCMConfig, DiffusionContextModel

    separator("Test 4: Full Pipeline with Qwen (4-bit)")

    cfg = DCMConfig(
        num_latent_vectors=8,       # Small for testing
        denoiser_hidden_dim=512,
        denoiser_num_layers=2,
        lora_r=8,                   # Smaller LoRA for test
    )

    print("  Loading DiffusionContextModel (this downloads Qwen if needed)...")
    t0 = time.time()
    model = DiffusionContextModel(cfg, device_map="auto")
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Synthesize 100 random tokens
    vocab_size = model.decoder.tokenizer.vocab_size
    B = 1  # Single sample for memory safety
    context_ids = torch.randint(0, vocab_size, (B, 100), device=device)
    continuation_ids = torch.randint(0, vocab_size, (B, 20), device=device)
    continuation_labels = torch.randint(0, vocab_size, (B, 20), device=device)

    print(f"  Context: {context_ids.shape}, Continuation: {continuation_ids.shape}")

    # Full forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            context_ids=context_ids,
            continuation_ids=continuation_ids,
            continuation_labels=continuation_labels,
        )

    logits = outputs["logits"]
    print(f"  Output logits: {logits.shape}")
    assert logits.shape[0] == B and logits.shape[1] == 20 and logits.dim() == 3, \
        f"Expected logits shape (1, 20, V), got {logits.shape}"

    # Verify probability distribution for the "101st token" (last position)
    last_logits = logits[:, -1, :]  # (B, model_vocab_size)
    probs = F.softmax(last_logits, dim=-1)

    assert torch.allclose(probs.sum(dim=-1), torch.ones(B, device=probs.device), atol=1e-3), \
        "Probabilities don't sum to 1!"

    top5_probs, top5_ids = probs.topk(5, dim=-1)
    tokenizer = model.decoder.tokenizer
    print(f"\n  Probability distribution for next token (top 5):")
    for i in range(5):
        token_str = tokenizer.decode([top5_ids[0, i].item()])
        print(f"    #{i+1}: '{token_str}' (id={top5_ids[0, i].item()}) → p={top5_probs[0, i].item():.4f}")

    print(f"\n  Loss total:     {outputs['loss'].item():.4f}")
    print(f"  Loss AR (CE):   {outputs['loss_ar'].item():.4f}")
    print(f"  Loss Diffusion: {outputs['loss_diffusion'].item():.4f}")
    print("  PASSED — full pipeline produces valid next-token distribution")


def test_memory_report(device: torch.device):
    """Report GPU memory usage after tests."""
    separator("GPU Memory Report")
    if device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"    Allocated: {allocated:.2f} GB / {total:.2f} GB")
            print(f"    Reserved:  {reserved:.2f} GB / {total:.2f} GB")
    else:
        print("  Running on CPU — no GPU memory to report")


def main():
    parser = argparse.ArgumentParser(description="DCM Sanity Check")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--skip_qwen", action="store_true",
                        help="Skip Qwen loading (test encoder + diffuser only)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda:0")
    print(f"DCM Sanity Check — device: {device}")

    try:
        # Test 1: SSM Encoder
        cfg = test_ssm_encoder(device)

        # Test 2: Diffuser
        test_diffuser(device, cfg)

        # Test 3: Encoder + Diffuser integration
        test_full_pipeline_no_qwen(device)

        # Test 4: Full pipeline with Qwen (optional)
        if not args.skip_qwen:
            test_full_pipeline_with_qwen(device)
        else:
            separator("Test 4: SKIPPED (--skip_qwen)")
            print("  Qwen loading skipped. Encoder + Diffuser verified.")

        # Memory report
        test_memory_report(device)

        separator("ALL TESTS PASSED")
        print("  DCM components handshake correctly.")
        print("  Ready for training run.\n")

    except Exception as e:
        separator("TEST FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
