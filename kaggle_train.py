"""
kaggle_train.py — DCM training loop for Kaggle dual-T4 GPUs
============================================================
Uses single-process model parallelism: Qwen is sharded across both T4s via
device_map="auto", while the encoder/diffuser live on cuda:0.

This is NOT DDP — both GPUs serve one model copy. The two T4s give us 32GB
total VRAM, enough for 4-bit Qwen (7GB) + encoder (52M) + diffuser (32M)
+ activations + optimizer states.

Key optimizations for T4 memory:
    - Qwen loaded in 4-bit NF4 quantization
    - LoRA adapters (only ~0.5% of Qwen params trainable)
    - Gradient accumulation to reach effective batch sizes
    - Mixed precision (fp16) via torch.cuda.amp
    - Gradient checkpointing on encoder

Usage (Kaggle notebook):
    !python kaggle_train.py --use_synthetic --max_steps 50 --log_every 5
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dcm_model import DCMConfig, DiffusionContextModel
from dcm_data import LongTextIterableDataset, SyntheticLongTextDataset


# ---------------------------------------------------------------------------
#  Hyperparameters
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DCM Kaggle Training")
    # Data
    p.add_argument("--data_dir", type=str, default="/kaggle/input/gutenberg",
                    help="Directory with .txt files for training")
    p.add_argument("--use_synthetic", action="store_true",
                    help="Use synthetic random data (for testing)")
    p.add_argument("--context_len", type=int, default=1024)
    p.add_argument("--continuation_len", type=int, default=512)

    # Training
    p.add_argument("--batch_size", type=int, default=1,
                    help="Micro batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="Accumulation steps (effective batch = batch_size * accum)")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="/kaggle/working/dcm_checkpoints")

    # Model
    p.add_argument("--lambda_diffusion", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
#  Learning rate scheduler (cosine with linear warmup)
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, max_steps: int, peak_lr: float) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup:
        return peak_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
#  Main training function
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "CPU"

    print("=" * 60)
    print("  DCM — Diffusion Context Model — Kaggle Training")
    print(f"  GPUs: {num_gpus} x {device_name}")
    print(f"  Mode: Model parallelism (Qwen sharded across all GPUs)")
    print("=" * 60)

    # ----- Config -----
    cfg = DCMConfig(
        lambda_diffusion=args.lambda_diffusion,
        lora_r=args.lora_r,
        num_latent_vectors=64,
    )

    # ----- Model -----
    # device_map="auto" shards Qwen across all available GPUs.
    # Encoder and diffuser are placed on the same device as Qwen's first layer.
    print("Loading DCM model (4-bit Qwen + SSM encoder + diffuser)...")
    model = DiffusionContextModel(cfg, device_map="auto")

    # Collect trainable params
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_trainable:,}")

    # ----- Data -----
    if args.use_synthetic:
        print("Using SYNTHETIC data for testing.")
        vocab_size = model.decoder.tokenizer.vocab_size
        dataset = SyntheticLongTextDataset(
            vocab_size=vocab_size,
            context_len=args.context_len,
            continuation_len=args.continuation_len,
            num_samples=args.max_steps * args.batch_size * args.gradient_accumulation_steps * 2,
        )
    else:
        print(f"Loading data from: {args.data_dir}")
        dataset = LongTextIterableDataset(
            data_dir=args.data_dir,
            tokenizer=model.decoder.tokenizer,
            context_len=args.context_len,
            continuation_len=args.continuation_len,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Mixed precision scaler (fp16 on T4)
    scaler = GradScaler("cuda")

    # ----- Training Loop -----
    print(f"\nStarting training...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Max steps: {args.max_steps}")
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_loss_ar = 0.0
    running_loss_diff = 0.0
    start_time = time.time()

    model.train()
    data_iter = iter(dataloader)

    while global_step < args.max_steps:
        # Fetch batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        context_ids = batch["context_ids"]
        continuation_ids = batch["continuation_ids"]
        continuation_labels = batch["continuation_labels"]

        # Forward pass with mixed precision
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                context_ids=context_ids,
                continuation_ids=continuation_ids,
                continuation_labels=continuation_labels,
            )
            loss = outputs["loss"] / args.gradient_accumulation_steps

        # Backward with scaled gradients
        scaler.scale(loss).backward()

        # Track metrics (use unscaled loss for logging)
        running_loss += outputs["loss"].item()
        running_loss_ar += outputs["loss_ar"].item()
        running_loss_diff += outputs["loss_diffusion"].item()

        micro_step += 1

        # Optimizer step after accumulation
        if micro_step % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # LR scheduling
            lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                n = args.log_every * args.gradient_accumulation_steps
                avg_loss = running_loss / n
                avg_ar = running_loss_ar / n
                avg_diff = running_loss_diff / n
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed

                # GPU memory
                mem_alloc = torch.cuda.memory_allocated(0) / 1024**3 if num_gpus > 0 else 0
                mem_total = torch.cuda.get_device_properties(0).total_mem / 1024**3 if num_gpus > 0 else 0

                print(
                    f"Step {global_step}/{args.max_steps} | "
                    f"Loss: {avg_loss:.4f} (AR: {avg_ar:.4f}, Diff: {avg_diff:.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {steps_per_sec:.2f} steps/s | "
                    f"GPU0: {mem_alloc:.1f}/{mem_total:.1f} GB"
                )
                running_loss = 0.0
                running_loss_ar = 0.0
                running_loss_diff = 0.0

            # Checkpointing
            if global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                print(f"Saving checkpoint to {save_path}")
                os.makedirs(save_path, exist_ok=True)
                # Save trainable components only (encoder, diffuser, LoRA, memory_proj)
                torch.save({
                    "global_step": global_step,
                    "encoder": model.encoder.state_dict(),
                    "diffuser": model.diffuser.state_dict(),
                    "memory_proj": model.decoder.memory_proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                }, os.path.join(save_path, "dcm_training_state.pt"))
                # Save LoRA adapter separately
                model.decoder.model.save_pretrained(os.path.join(save_path, "lora_adapter"))

    # ----- Final save -----
    print("Training complete!")
    final_path = os.path.join(args.output_dir, "checkpoint-final")
    os.makedirs(final_path, exist_ok=True)
    torch.save({
        "global_step": global_step,
        "encoder": model.encoder.state_dict(),
        "diffuser": model.diffuser.state_dict(),
        "memory_proj": model.decoder.memory_proj.state_dict(),
    }, os.path.join(final_path, "dcm_training_state.pt"))
    model.decoder.model.save_pretrained(os.path.join(final_path, "lora_adapter"))
    print(f"Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
