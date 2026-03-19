"""
kaggle_train.py — DCM training loop for Kaggle dual-T4 GPUs
============================================================
Uses HuggingFace Accelerate for DDP across 2x NVIDIA T4 (16GB each).

Key optimizations for T4 memory:
    - Qwen loaded in 4-bit NF4 quantization
    - LoRA adapters (only ~0.5% of params trainable)
    - Gradient accumulation to reach effective batch sizes
    - Mixed precision (bf16/fp16) for all custom modules
    - Gradient checkpointing on the SSM encoder

Usage (Kaggle notebook):
    !accelerate launch --num_processes=2 kaggle_train.py

Or configure via accelerate config first:
    !accelerate config  # select multi-GPU, 2 processes, bf16
    !accelerate launch kaggle_train.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
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
                    help="Per-device micro batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="Accumulation steps (effective batch = batch_size * num_gpus * accum)")
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
    return peak_lr * 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))


# ---------------------------------------------------------------------------
#  Main training function
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Initialize Accelerator (handles DDP, mixed precision, device placement)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",  # T4 supports fp16; bf16 via software emulation
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    set_seed(args.seed)

    accelerator.print("=" * 60)
    accelerator.print("  DCM — Diffusion Context Model — Kaggle Training")
    accelerator.print(f"  Devices: {accelerator.num_processes} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    accelerator.print("=" * 60)

    # ----- Config -----
    cfg = DCMConfig(
        lambda_diffusion=args.lambda_diffusion,
        lora_r=args.lora_r,
        num_latent_vectors=64,
    )

    # ----- Model -----
    accelerator.print("Loading DCM model (4-bit Qwen + SSM encoder + diffuser)...")
    model = DiffusionContextModel(cfg, device_map="auto")

    # Freeze base Qwen (only LoRA + encoder + diffuser are trainable)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            if accelerator.is_main_process:
                accelerator.print(f"  Trainable: {name} | {param.numel():,} params")

    total_trainable = sum(p.numel() for p in trainable_params)
    accelerator.print(f"Total trainable parameters: {total_trainable:,}")

    # ----- Data -----
    if args.use_synthetic:
        accelerator.print("Using SYNTHETIC data for testing.")
        vocab_size = model.decoder.tokenizer.vocab_size
        dataset = SyntheticLongTextDataset(
            vocab_size=vocab_size,
            context_len=args.context_len,
            continuation_len=args.continuation_len,
            num_samples=args.max_steps * args.batch_size * args.gradient_accumulation_steps,
        )
    else:
        accelerator.print(f"Loading data from: {args.data_dir}")
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

    # ----- Prepare with Accelerate -----
    # Note: We don't prepare the full model since Qwen uses device_map="auto".
    # Instead, we prepare encoder + diffuser + optimizer + dataloader.
    model.encoder, model.diffuser, optimizer, dataloader = accelerator.prepare(
        model.encoder, model.diffuser, optimizer, dataloader
    )

    # ----- Training Loop -----
    accelerator.print("\nStarting training...")
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
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

        with accelerator.accumulate(model.encoder, model.diffuser):
            # Move to device
            context_ids = batch["context_ids"]
            continuation_ids = batch["continuation_ids"]
            continuation_labels = batch["continuation_labels"]

            # Forward pass
            outputs = model(
                context_ids=context_ids,
                continuation_ids=continuation_ids,
                continuation_labels=continuation_labels,
            )

            loss = outputs["loss"]

            # Backward
            accelerator.backward(loss)

            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()

            # LR scheduling
            lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()

        # Track metrics
        running_loss += outputs["loss"].item()
        running_loss_ar += outputs["loss_ar"].item()
        running_loss_diff += outputs["loss_diffusion"].item()

        if accelerator.sync_gradients:
            global_step += 1

            # Logging
            if global_step % args.log_every == 0 and accelerator.is_main_process:
                avg_loss = running_loss / args.log_every
                avg_ar = running_loss_ar / args.log_every
                avg_diff = running_loss_diff / args.log_every
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed

                accelerator.print(
                    f"Step {global_step}/{args.max_steps} | "
                    f"Loss: {avg_loss:.4f} (AR: {avg_ar:.4f}, Diff: {avg_diff:.4f}) | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {steps_per_sec:.2f} steps/s"
                )
                running_loss = 0.0
                running_loss_ar = 0.0
                running_loss_diff = 0.0

            # Checkpointing
            if global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.print(f"Saving checkpoint to {save_path}")
                accelerator.save_state(save_path)

    # ----- Final save -----
    accelerator.print("Training complete!")
    final_path = os.path.join(args.output_dir, "checkpoint-final")
    accelerator.save_state(final_path)
    accelerator.print(f"Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
