# DCM — Diffusion Context Model

Replaces the **O(n^2) KV-cache** of autoregressive Transformers with a **continuous, O(n)-scaling semantic memory** built on latent diffusion. Context tokens are compressed into a fixed-length sequence of continuous latent vectors via a State Space Model, refined through a denoising diffusion process, then injected as soft prompts into a frozen LLM.

---

## Architecture

```
                         DCM — Full Pipeline
  ═══════════════════════════════════════════════════════════

  CONTEXT TOKENS (B, L_ctx)          CONTINUATION TOKENS (B, L_cont)
        │                                      │
        ▼                                      │
  ┌───────────────┐                            │
  │  Qwen Embed   │  (shared embedding layer)  │
  │  Layer (frozen)│                            │
  └──────┬────────┘                            │
         │ (B, L_ctx, 3584)                    │
         ▼                                     │
  ┌─────────────────────────────┐              │
  │      DCM_SSMEncoder         │              │
  │  ┌────────────────────────┐ │              │
  │  │  Linear Projection     │ │              │
  │  │  D_in → D_latent       │ │              │
  │  └──────────┬─────────────┘ │              │
  │             ▼               │              │
  │  ┌────────────────────────┐ │              │
  │  │  SSM Block x2          │ │              │
  │  │  (Selective Scan,      │ │              │
  │  │   O(n) recurrence)     │ │              │
  │  └──────────┬─────────────┘ │              │
  │             ▼               │              │
  │  ┌────────────────────────┐ │              │
  │  │  Learned Query Pooling │ │              │
  │  │  L tokens → M latents  │ │              │
  │  │  (1024 → 64 vectors)   │ │              │
  │  └──────────┬─────────────┘ │              │
  └─────────────┼───────────────┘              │
                │ z0 (B, M, D)                 │
                ▼                              │
  ┌─────────────────────────────┐              │
  │     DCM_LatentDiffuser      │              │
  │                             │              │
  │  TRAIN:                     │              │
  │   z0 ──noise──► z_t         │              │
  │   z_t ─denoise─► z0_pred    │              │
  │   loss = MSE(z0, z0_pred)   │              │
  │                             │              │
  │  INFERENCE:                 │              │
  │   noise ──sample──► z0_pred │              │
  │   (50-step reverse DDIM)    │              │
  └─────────────┬───────────────┘              │
                │ z0_pred (B, M, D)            │
                │ "denoised memory"            │
                ▼                              ▼
  ┌───────────────────────────────────────────────────┐
  │              QwenLoRAHead                         │
  │                                                   │
  │  z0_pred ──► memory_proj ──► mem_embeds (B,M,H)  │
  │                                     │             │
  │  continuation_ids ──► embed ──► tok_embeds (B,L,H)│
  │                                     │             │
  │        ┌────────────────────────────┘             │
  │        ▼                                          │
  │  [ mem_embeds ; tok_embeds ]  ──► Qwen2.5-7B     │
  │   (B, M+L, H)                    (4-bit NF4,     │
  │                                    LoRA q/v_proj) │
  │        │                                          │
  │        ▼                                          │
  │  logits[:, M:, :]  ──► next-token prediction     │
  └───────────────────────────────────────────────────┘
                │
                ▼
  ┌─────────────────────────────┐
  │        HYBRID LOSS          │
  │                             │
  │  L = L_AR + lambda * L_diff │
  │      │            │         │
  │      │            └─ MSE(z0, z0_pred)
  │      └─ CrossEntropy(logits, labels)
  └─────────────────────────────┘
```

### Key Design Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Memory injection | **Option A: Prefix-tuning** (soft prompts prepended to embeddings) | No architectural surgery on Qwen; works with any decoder-only LLM |
| Context encoder | **Selective SSM** (Mamba-style linear recurrence) | O(n) scaling, no self-attention quadratic cost |
| Diffusion target | **x0-prediction** (predict clean latents directly) | More stable gradients than epsilon-prediction for this use case |
| Quantization | **4-bit NF4** via bitsandbytes | Required to fit Qwen 7B on T4 16GB GPUs |
| Fine-tuning | **LoRA** on q_proj, v_proj | 0.03% trainable params; teaches Qwen to read injected memory |

---

## File Map

```
dcm/
├── dcm_model.py          # Core architecture (682 lines)
│   ├── DCMConfig              — Central dataclass config for all components
│   ├── _SelectiveSSMBlock     — Single SSM layer (selective scan recurrence)
│   ├── DCM_SSMEncoder         — Stacked SSM + learned query pooling → z0
│   ├── _SinusoidalTimeEmbedding — Timestep encoding for diffusion
│   ├── _DenoisingMLP          — Residual MLP that predicts z0 from z_t
│   ├── DCM_LatentDiffuser     — Noise schedule + forward/reverse diffusion
│   ├── AbstractDecoderHead    — ABC for swappable LLM backends
│   ├── QwenLoRAHead           — 4-bit Qwen + LoRA + prefix memory injection
│   └── DiffusionContextModel  — Master orchestrator (forward, generate)
│
├── dcm_data.py           # Data pipeline (186 lines)
│   ├── LongTextIterableDataset  — Streams .txt → rolling (context, continuation) windows
│   ├── SyntheticLongTextDataset — Random tokens for testing
│   └── build_dataloader()       — Factory with sensible defaults
│
├── kaggle_train.py       # Training loop (261 lines)
│   ├── parse_args()             — CLI args for all hyperparameters
│   ├── get_lr()                 — Cosine decay with linear warmup
│   └── main()                   — Accelerate DDP loop, hybrid loss, checkpointing
│
├── sanity_check.py       # Integration tests (270 lines)
│   ├── test_ssm_encoder()              — Shape validation for encoder
│   ├── test_diffuser()                 — Forward/reverse diffusion round-trip
│   ├── test_full_pipeline_no_qwen()    — Encoder+diffuser without downloading Qwen
│   ├── test_full_pipeline_with_qwen()  — End-to-end: 100 tokens → probability dist
│   └── test_memory_report()            — GPU memory usage
│
├── kaggle_notebook.py    # Copy-paste cells for Kaggle (103 lines)
├── aws_scaling_plan.md   # Phase 2: p4d.24xlarge scaling guide
├── requirements.txt      # Python dependencies
└── .gitignore
```

---

## Quick Start (Kaggle)

**Prerequisites:** Kaggle notebook with **GPU T4 x2** and **Internet** enabled.

```python
# Cell 1 — Install + clone
!pip install -q transformers accelerate peft bitsandbytes
!git clone https://github.com/locke-inc/dcm.git /kaggle/working/dcm
import sys; sys.path.insert(0, "/kaggle/working/dcm")

# Cell 2 — Sanity check (no model download)
%cd /kaggle/working/dcm
!python sanity_check.py --skip_qwen

# Cell 3 — Full sanity check (downloads Qwen ~15GB)
!python sanity_check.py

# Cell 4 — Test training loop (synthetic data)
!python kaggle_train.py --use_synthetic --max_steps 50 --log_every 5

# Cell 5 — Real training with DDP
!accelerate launch kaggle_train.py \
    --data_dir /kaggle/input/YOUR_DATASET/ \
    --batch_size 1 --gradient_accumulation_steps 8 \
    --max_steps 5000 --output_dir /kaggle/working/dcm_checkpoints
```

See `kaggle_notebook.py` for full cell-by-cell instructions including accelerate config.

---

## Configuration

All hyperparameters live in `DCMConfig` (`dcm_model.py:30`). Key defaults:

```
SSM Encoder:     ssm_input_dim=3584, ssm_state_dim=64, ssm_num_layers=2
Latent Space:    num_latent_vectors=64, latent_dim=3584
Diffuser:        diffusion_steps=1000, beta=[0.0001, 0.02], denoiser_layers=4
LLM:             Qwen/Qwen2.5-7B-Instruct, 4-bit NF4, LoRA r=16 on q/v_proj
Training:        lambda_diffusion=1.0 (equal weight AR + diffusion losses)
```

Override at construction: `cfg = DCMConfig(num_latent_vectors=128, lora_r=32)`

---

## Training Data Flow

```
  long_document.txt
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Tokenize → concatenate into rolling buffer             │
  │                                                         │
  │  ◄──── context_len ────►◄── continuation_len ──►│       │
  │  [ tok tok tok ... tok ] [ tok tok tok ... tok ] [label] │
  │        context_ids           continuation_ids            │
  │                                                         │
  │  Slide by stride, yield next window                     │
  └─────────────────────────────────────────────────────────┘
```

- Zero padding waste (continuous stream, not padded batches)
- `context_ids` → SSM encoder → latent memory
- `continuation_ids` → AR prediction conditioned on memory
- `continuation_labels` = `continuation_ids` shifted by 1

---

## Hybrid Loss

```
L_total = L_AR + lambda * L_diffusion

L_AR        = CrossEntropy(logits, next_token_labels)
              Only computed over continuation positions (memory prefix masked with -100)

L_diffusion = MSE(z0, z0_pred)
              z0 = true latents from encoder
              z0_pred = denoiser output from noisy z_t at random timestep
```

---

## Device Handling

Qwen loads via `device_map="auto"` (HuggingFace places shards across GPUs). Custom modules (encoder, diffuser, memory_proj) are explicitly `.to()` the same device as Qwen's parameters. Input tensors are moved in `DiffusionContextModel.forward()`.

If you hit device errors, check:
- `dcm_model.py:589` — `self._device` detection
- `dcm_model.py:461` — `memory_proj` placement
- `dcm_model.py:614-617` — input tensor movement

---

## Phase 2: AWS Scaling

See `aws_scaling_plan.md` for full details. Summary:

| | Kaggle (Phase 1) | AWS p4d.24xlarge (Phase 2) |
|---|---|---|
| GPUs | 2x T4 (16GB) | 8x A100 (40GB) |
| Precision | 4-bit NF4 + fp16 | Full bf16 |
| Context | 1,024 tokens | 16K–65K tokens |
| Distributed | Accelerate DDP | PyTorch FSDP |
| SSM scan | Sequential loop | Triton parallel scan kernel |
| Budget | Free | $2,500 strict |
