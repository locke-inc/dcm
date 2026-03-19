# DCM AWS Scaling Plan — p4d.24xlarge

## Target Instance

| Spec | Value |
|------|-------|
| Instance | p4d.24xlarge |
| GPUs | 8x NVIDIA A100 40GB |
| GPU Memory | 320 GB total |
| vCPUs | 96 |
| RAM | 1.1 TB |
| Network | 400 Gbps EFA |
| On-Demand Price | ~$32.77/hr |
| **Budget** | **$2,500 strict** |
| **Max Runtime** | **~76 hours** |

---

## 1. Software Stack

### CUDA / Driver Requirements
- **CUDA**: 12.1+ (required for PyTorch 2.x + Triton)
- **cuDNN**: 8.9+
- **NCCL**: 2.18+ (multi-GPU communication)
- **Driver**: 535+ (comes with AWS Deep Learning AMI)

### Recommended AMI
Use **AWS Deep Learning AMI (Ubuntu 22.04)** — comes pre-installed with:
- CUDA 12.1, cuDNN 8.9, NCCL 2.18
- PyTorch 2.2+, Triton
- EFA drivers for NVLink/NVSwitch

### Python Dependencies
```
torch>=2.2.0
transformers>=4.38.0
accelerate>=0.27.0
peft>=0.8.0
bitsandbytes>=0.42.0       # Can switch to full precision on A100
triton>=2.2.0
datasets>=2.17.0
tensorboard
deepspeed>=0.13.0           # For ZeRO Stage 3 if needed
flash-attn>=2.5.0           # Flash Attention 2 for A100
```

---

## 2. Model Configuration Changes (Kaggle → AWS)

### Quantization
- **Kaggle**: 4-bit NF4 (required for T4 16GB)
- **AWS**: **Full bf16** — A100 has 40GB per GPU, 320GB total. No quantization needed. This dramatically improves training quality.

### Precision
- **Kaggle**: Mixed fp16 (T4 has limited bf16 support)
- **AWS**: **Native bf16** — A100 has hardware bf16 support. Set `torch.set_default_dtype(torch.bfloat16)`.

### Sequence Lengths
| Parameter | Kaggle (T4) | AWS (A100) |
|-----------|-------------|------------|
| context_len | 1,024 | 16,384 – 65,536 |
| continuation_len | 512 | 2,048 – 4,096 |
| num_latent_vectors | 64 | 256 – 512 |
| batch_size (per GPU) | 1 | 4 – 8 |
| gradient_accumulation | 8 | 2 – 4 |
| effective_batch | 16 | 64 – 256 |

### SSM Encoder Scaling
- Increase `ssm_num_layers` from 2 → 4–6
- Increase `ssm_state_dim` from 64 → 128
- With 64K context, the O(n) SSM encoder is critical — a Transformer encoder would OOM

### LoRA → Full Fine-Tuning
With sufficient VRAM, consider removing LoRA and fine-tuning all of Qwen's attention layers. This gives better gradient flow at the cost of more memory. FSDP handles the sharding.

---

## 3. Distributed Training Strategy

### FSDP (Fully Sharded Data Parallel)
Replace Accelerate DDP with **PyTorch FSDP** for memory-efficient sharding across 8 GPUs:

```python
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_sharding_strategy: FULL_SHARD     # ZeRO-3 equivalent
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_cpu_ram_efficient_loading: true
mixed_precision: bf16
num_machines: 1
num_processes: 8
```

### Why FSDP over DeepSpeed ZeRO
- Native PyTorch (no external dependency)
- Better Triton kernel compatibility
- Simpler checkpoint management
- Comparable memory efficiency to ZeRO-3

### Gradient Checkpointing
Enable on both the SSM encoder and Qwen transformer layers:
```python
model.encoder.gradient_checkpointing_enable()
model.decoder.model.gradient_checkpointing_enable()
```

---

## 4. Triton Optimizations

### SSM Parallel Scan Kernel
The sequential scan loop in `_SelectiveSSMBlock` is the primary bottleneck at scale. Replace with a fused Triton kernel:

```python
# Pseudocode for Triton parallel scan
@triton.jit
def parallel_scan_kernel(
    A_bar_ptr, B_bar_ptr, x_ptr, h_ptr, y_ptr,
    B, L, D, N, BLOCK_SIZE: tl.constexpr
):
    """
    Blelloch-style parallel prefix scan for SSM recurrence.
    Reduces O(L) sequential steps to O(log L) parallel steps.
    """
    # 1. Up-sweep (reduce): combine adjacent pairs
    # 2. Down-sweep: propagate prefix sums
    # Each element: (A_bar[t], B_bar[t] * x[t]) forms a monoid
    # Composition: (a1, b1) * (a2, b2) = (a1*a2, a2*b1 + b2)
```

Expected speedup: **5-15x** for long sequences (L > 8K).

### Flash Attention 2
For the Qwen decoder's self-attention and the pooling cross-attention:
```python
from flash_attn import flash_attn_func
# Replaces F.scaled_dot_product_attention with IO-aware tiling
```

### Fused LayerNorm + SiLU
Use `triton.ops.fused_layernorm` to eliminate memory-bound kernel launches in the denoiser MLP.

---

## 5. Data Preparation Strategy

### Dataset
For 64K+ context training, use:
1. **The Pile (deduplicated)** — 800GB diverse text
2. **RedPajama-v2** — 30T tokens with quality filtering
3. **StarCoder data** — long code repositories
4. **ArXiv papers** — naturally long documents

### Preprocessing Pipeline
```bash
# 1. Download and filter
python scripts/download_data.py --dataset pile --output /data/raw/

# 2. Tokenize and pack into binary format
python scripts/tokenize_pack.py \
    --input /data/raw/ \
    --output /data/tokenized/ \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --max_seq_len 65536 \
    --num_workers 64

# 3. Create memory-mapped files for streaming
python scripts/create_memmap.py \
    --input /data/tokenized/ \
    --output /data/memmap/
```

### Data Loading Optimization
- Use **memory-mapped** numpy arrays (no RAM bottleneck on 1.1TB machine)
- Pre-tokenized binary format eliminates tokenization overhead
- Shuffle at the file level, not sample level, to preserve long-range coherence

---

## 6. Budget Estimation

### Training Plan

| Phase | Hours | Cost |
|-------|-------|------|
| Data prep + setup | 2 | $65 |
| Sanity checks + debugging | 3 | $98 |
| Short training run (1K steps, tuning) | 5 | $164 |
| Main training run (50K steps) | 50 | $1,639 |
| Evaluation + checkpointing | 4 | $131 |
| **Contingency (15%)** | **8** | **$262** |
| **Total** | **72** | **$2,359** |

### Cost Controls
1. **Use Spot Instances**: p4d.24xlarge spot price is ~$12-15/hr (60% savings). Implement checkpoint saving every 500 steps for fault tolerance.
2. **Auto-shutdown**: Set a CloudWatch alarm to terminate if GPU utilization < 10% for 15 minutes.
3. **Budget alarm**: Set AWS Budgets alert at $2,000 and $2,400.

### Spot Instance Resilience
```bash
# Launch with spot + fallback
aws ec2 run-instances \
    --instance-type p4d.24xlarge \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"20.00","SpotInstanceType":"persistent"}}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=DCM}]'
```

Training script must:
- Save checkpoints to S3 every 500 steps
- Auto-resume from latest checkpoint on restart
- Use `accelerate launch --main_process_port=29500` for clean restarts

---

## 7. Evaluation Plan

### Metrics
1. **Perplexity** on held-out long documents (8K, 16K, 32K, 64K)
2. **Diffusion reconstruction MSE** (z₀ vs ẑ₀) — should decrease over training
3. **Memory compression ratio**: tokens compressed per latent vector
4. **Inference latency**: time-to-first-token with vs without diffusion memory
5. **Needle-in-haystack**: inject a fact at position N, test retrieval at position N+K

### Baseline Comparisons
- Qwen2.5-7B vanilla (no DCM, standard KV-cache) — perplexity at various lengths
- Qwen2.5-7B + simple mean-pooling memory (no diffusion) — ablation
- DCM with different diffusion step counts at inference

---

## 8. Quick-Start Commands

```bash
# SSH into instance
ssh -i dcm-key.pem ubuntu@<instance-ip>

# Setup
cd /home/ubuntu
git clone <dcm-repo> && cd dcm
pip install -r requirements.txt

# Prepare data (parallel tokenization on 96 cores)
python scripts/prepare_data.py --num_workers 64

# Configure Accelerate for 8x A100 FSDP
accelerate config  # Select FSDP, 8 processes, bf16

# Run sanity check first
python sanity_check.py

# Launch training
accelerate launch kaggle_train.py \
    --data_dir /data/memmap/ \
    --context_len 16384 \
    --continuation_len 2048 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_steps 50000 \
    --output_dir /home/ubuntu/checkpoints/
```
