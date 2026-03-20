# DCM — Diffusion Context Model

Replaces the **O(n^2) KV-cache** of autoregressive Transformers with a **continuous, O(n)-scaling semantic memory**. Context tokens are compressed into a fixed-length sequence of continuous latent vectors via a State Space Model, then injected as soft prompts into a frozen LLM. A diffusion process regularizes the latent space during training.

---

## Architecture

```
                    DCM — Full Pipeline (Path B: Conditional Diffusion)
  ══════════════════════════════════════════════════════════════════════

  CONTEXT TOKENS (B, L_ctx)             CONTINUATION TOKENS (B, L_cont)
        │                                         │
        ▼                                         │
  ┌───────────────┐                               │
  │  Qwen Embed   │  (shared embedding layer)     │
  │  Layer (frozen)│                               │
  └──────┬────────┘                               │
         │ (B, L_ctx, D)                          │
         ▼                                        │
  ┌─────────────────────────────────────┐         │
  │         DCM_SSMEncoder              │         │
  │  ┌────────────────────────┐         │         │
  │  │  Linear Projection     │         │         │
  │  │  D_in → D_latent       │         │         │
  │  └──────────┬─────────────┘         │         │
  │             ▼                       │         │
  │  ┌────────────────────────┐         │         │
  │  │  SSM Block x2          │         │         │
  │  │  (Selective Scan,      │         │         │
  │  │   O(n) recurrence)     │         │         │
  │  └──────────┬─────────────┘         │         │
  │             │                       │         │
  │       ┌─────┴─────┐                │         │
  │       ▼           ▼                │         │
  │  ┌──────────┐ ┌──────────────┐     │         │
  │  │  Query   │ │  Mean Pool   │     │         │
  │  │  Pooling │ │  → cond_proj │     │         │
  │  │  L → M   │ │  L → 1       │     │         │
  │  └────┬─────┘ └──────┬───────┘     │         │
  │       │              │             │         │
  └───────┼──────────────┼─────────────┘         │
          │              │                       │
     z0 (B,M,D)    c (B, cond_dim)              │
     [train target]  [bottleneck]                │
          │              │                       │
          ▼              ▼                       │
  ┌─────────────────────────────────┐            │
  │     DCM_LatentDiffuser          │            │
  │                                 │            │
  │  TRAIN:                         │            │
  │   z0 ──noise──► z_t             │            │
  │   denoiser(z_t, t, c) → z0_pred│            │
  │   loss = MSE(z0, z0_pred)      │            │
  │                                 │            │
  │  INFERENCE:                     │            │
  │   noise + c ──sample──► z0_pred │            │
  │   (50-step conditional DDIM)    │            │
  └─────────────┬───────────────────┘            │
                │ z0_pred (B, M, D)              │
                │ "context-conditioned memory"   │
                ▼                                ▼
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

### What each component does

| Component | Role | Training | Inference |
|-----------|------|----------|-----------|
| **SSM Encoder** | Compresses L context tokens into z0 (M latent vectors) + c (small conditioning vector) in O(n). | Produces z0 (target) + c (conditioning) | Produces c (conditioning for diffuser) |
| **Diffuser** | Context-conditioned denoiser with AdaLN. Learns to reconstruct z0 from noise given only the small conditioning vector c. | z0 + noise → z_t, denoiser(z_t, t, c) → z0_pred | noise + c → sample → z0_pred (context-specific memory) |
| **Information bottleneck** | c (cond_dim=256) is much smaller than z0 (M×D). Forces encoder to compress the *essence* of context into c; diffuser expands it back into rich memory. | Gradient from both L_AR and L_diff shapes c | c is all the diffuser needs to reconstruct memory |
| **LoRA + memory_proj** | Teaches Qwen to attend to the injected memory prefix vectors. | Trained | Used |
| **Prefix injection** | Prepends memory as soft prompts to token embeddings before Qwen's forward pass. | Used | Used |

### Key Design Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Memory injection | **Prefix-tuning** (soft prompts prepended to embeddings) | No architectural surgery on Qwen; works with any decoder-only LLM |
| Context encoder | **Selective SSM** (Mamba-style linear recurrence) | O(n) scaling, no self-attention quadratic cost |
| Diffusion conditioning | **AdaLN** (Adaptive Layer Norm, from DiT) | Per-layer scale/shift from c modulates denoiser — lightweight, proven in image diffusion |
| Information bottleneck | **c ∈ ℝ^{cond_dim}** (small conditioning vector) | Forces encoder to learn what's essential; diffuser expands c back into M-vector memory |
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
├── download_data.py      # Fetches Gutenberg books from Kaggle metadata CSV
├── test_generate.py      # Generation test: same prompt, different contexts
├── kaggle_notebook.py    # Copy-paste cells for Kaggle (103 lines)
├── aws_scaling_plan.md   # Phase 2: p4d.24xlarge scaling guide
├── requirements.txt      # Python dependencies
└── .gitignore
```

---

## Quick Start (Kaggle)

**Prerequisites:** Kaggle notebook with **GPU T4 x2** and **Internet** enabled.
Add the dataset **mateibejan/15000-gutenberg-books** to your notebook.

```python
# Cell 1 — Install + clone
!pip install -q transformers accelerate peft bitsandbytes requests
!git clone https://github.com/locke-inc/dcm.git /kaggle/working/dcm
import sys; sys.path.insert(0, "/kaggle/working/dcm")

# Cell 2 — Download training data (20 Gutenberg books)
!python /kaggle/working/dcm/download_data.py

# Cell 3 — Sanity check (no model download)
%cd /kaggle/working/dcm
!python sanity_check.py --skip_qwen

# Cell 4 — Proof-of-concept training (~50 min on 2x T4)
!python kaggle_train.py \
    --data_dir /kaggle/working/gutenberg_texts/ \
    --max_steps 1000 \
    --log_every 50 \
    --context_len 128 \
    --continuation_len 128 \
    --latent_dim 512 \
    --num_latent_vectors 16 \
    --denoiser_hidden_dim 256 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 50 \
    --learning_rate 3e-4

# Cell 5 — Test generation (same prompt, different contexts)
!python /kaggle/working/dcm/test_generate.py
```

### Fast POC vs Full-Scale Config

The POC config shrinks dimensions to validate the concept on free T4s:

| Parameter | Full Scale | POC | Why |
|-----------|-----------|-----|-----|
| `context_len` | 1024 | 128 | SSM scan is O(n) sequential loop — 8x fewer iterations |
| `latent_dim` | 3584 | 512 | Each SSM step + diffuser ops ~7x cheaper |
| `continuation_len` | 512 | 128 | Qwen processes 144 vs 576 positions |
| `num_latent_vectors` | 64 | 16 | Smaller prefix + diffusion target |
| `gradient_accumulation_steps` | 8 | 2 | 4x fewer forward passes per optimizer step |

Full-scale training requires the Phase 2 AWS setup (see below), primarily because
the SSM scan at `dcm_model.py:126` is a Python `for` loop that needs a Triton
parallel scan kernel to run efficiently at `context_len=1024+`.

### Evaluating the Generation Test

`test_generate.py` feeds the **same generic prompt** through four wildly different contexts
(alien veterinarian, medieval bakery, sentient AI, underwater city) plus a no-context control.

| Result | Meaning |
|--------|---------|
| **STRONG PASS** | Each output references its specific context theme (aliens, herbs, code, water) |
| **WEAK PASS** | Outputs differ from each other and from the control, but themes are vague |
| **FAIL** | All outputs are basically the same generic text regardless of context |

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
              Gradient flows: Qwen LoRA ← memory_proj ← z0_pred ← diffuser ← encoder
              This is the primary signal that teaches the encoder what to compress.

L_diffusion = MSE(z0, z0_pred)
              z0 = true latents from encoder
              z0_pred = denoiser output from noisy z_t at random timestep
              Regularizes the latent space — forces encoder to produce representations
              that survive noise corruption and can be reconstructed by the denoiser.
```

Both losses train the encoder jointly. L_AR teaches it *what information to preserve*
(whatever helps Qwen predict the next token). L_diffusion teaches it *how to structure
the latent space* (smooth, denoising-friendly representations).

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
