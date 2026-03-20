"""
dcm_model.py — Diffusion Context Model (DCM)
=============================================
Replaces the O(n²) KV-cache of autoregressive Transformers with a continuous,
O(n)-scaling semantic memory built on latent diffusion.

Architecture:
    1. DCM_SSMEncoder   — Compresses token embeddings via linear SSM into fixed-length latents z₀
    2. DCM_LatentDiffuser — Forward/reverse diffusion on z₀ (x₀-prediction)
    3. QwenLoRAHead      — 4-bit Qwen2.5-7B + LoRA, receives denoised memory as soft prefixes
    4. DiffusionContextModel — Master orchestrator
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

@dataclass
class DCMConfig:
    """Central configuration for the entire DCM pipeline."""
    # SSM Encoder
    ssm_input_dim: int = 3584          # Qwen2.5-7B hidden size
    ssm_state_dim: int = 64            # Internal SSM state dimension
    ssm_num_layers: int = 2            # Number of stacked SSM blocks
    num_latent_vectors: int = 64       # How many latent vectors z₀ to produce
    latent_dim: int = 3584             # Dimension of each latent vector (match Qwen hidden)

    # Diffuser
    diffusion_steps: int = 1000        # Total noise schedule steps T
    beta_start: float = 0.0001         # Linear schedule start
    beta_end: float = 0.02             # Linear schedule end
    denoiser_hidden_dim: int = 1024    # Hidden size inside denoising MLP
    denoiser_num_layers: int = 4       # Depth of denoising MLP
    time_embed_dim: int = 256          # Sinusoidal timestep embedding dim

    # Decoder / LLM
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # Conditioning (Path B: conditional diffusion)
    cond_dim: int = 256                # Conditioning vector dimension (information bottleneck)

    # Training
    lambda_diffusion: float = 1.0      # Weight of L_diffuser_MSE in hybrid loss


# ===========================================================================
#  1. DCM_SSMEncoder — O(n) linear SSM context compressor
# ===========================================================================

class _SelectiveSSMBlock(nn.Module):
    """
    A single selective-scan SSM block using a *parallel scan* friendly
    linear recurrence formulation.

    Given input x of shape (B, L, D):
        Δ = softplus(linear_dt(x))          — input-dependent step size
        A_bar = exp(A * Δ)                  — discretised state transition
        B_bar = Δ * linear_B(x)            — discretised input matrix
        h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]   (scan)
        y[t] = linear_C(x[t]) @ h[t] + D * x[t]

    We implement the scan as a sequential loop here for clarity / portability.
    On CUDA this can later be swapped for a fused Triton kernel (Phase 2).
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Projections (all input-dependent → "selective")
        self.linear_dt = nn.Linear(d_model, d_model, bias=True)
        self.linear_B = nn.Linear(d_model, d_state, bias=False)
        self.linear_C = nn.Linear(d_model, d_state, bias=False)
        self.linear_D = nn.Parameter(torch.ones(d_model))  # skip connection

        # Learnable log-space diagonal state matrix A  (d_model, d_state)
        # Initialised to negative values so exp(A*Δ) < 1 → stable
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(
            -A_log.unsqueeze(0).expand(d_model, -1).clone()  # (d_model, d_state)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        # Input-dependent discretisation
        dt = F.softplus(self.linear_dt(x))          # (B, L, D)
        A = -torch.exp(self.A_log.float())           # (D, N) — always negative
        B_input = self.linear_B(x)                    # (B, L, N)
        C_input = self.linear_C(x)                    # (B, L, N)

        # Discretise:  A_bar = exp(A * dt),  B_bar = dt * B_input
        # A_bar: (B, L, D, N)
        A_bar = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, D, N)
        B_bar = dt.unsqueeze(-1) * B_input.unsqueeze(2)  # (B, L, D, N)

        # Sequential scan (replaced by parallel scan / Triton in Phase 2)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)  # (B, D, N)
            y_t = (C_input[:, t].unsqueeze(1) * h).sum(dim=-1)          # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + self.linear_D * residual
        return y


class DCM_SSMEncoder(nn.Module):
    """
    Encodes a variable-length token embedding sequence into:
        1. z₀ ∈ ℝ^{M × D}  — full latent memory (diffusion training target)
        2. c  ∈ ℝ^{cond_dim} — small conditioning vector (information bottleneck)

    The conditioning vector c is what the diffuser uses to reconstruct
    context-specific memory from noise at inference time.

    Pipeline:
        token_embeddings → [SSM blocks] → query_pool → z₀
                                        → mean_pool → c
    """

    def __init__(self, cfg: DCMConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.ssm_input_dim, cfg.latent_dim)

        self.ssm_blocks = nn.ModuleList([
            _SelectiveSSMBlock(cfg.latent_dim, cfg.ssm_state_dim)
            for _ in range(cfg.ssm_num_layers)
        ])

        self.output_norm = nn.LayerNorm(cfg.latent_dim)

        # Learnable queries for pooling (like Perceiver cross-attention but cheaper)
        self.latent_queries = nn.Parameter(
            torch.randn(1, cfg.num_latent_vectors, cfg.latent_dim) * 0.02
        )
        # Simple linear attention pooling: score = q @ k^T, then weighted sum
        self.pool_proj_k = nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False)
        self.pool_proj_v = nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False)

        # Conditioning head: mean-pool SSM output → small bottleneck vector
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )

    def forward(self, token_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embeddings: (B, L, ssm_input_dim) from the base LLM's embedding layer
        Returns:
            z0: (B, num_latent_vectors, latent_dim) — the true latents
            c:  (B, cond_dim) — conditioning vector for the diffuser
        """
        x = self.input_proj(token_embeddings)  # (B, L, latent_dim)

        for block in self.ssm_blocks:
            x = block(x)  # (B, L, latent_dim)

        x = self.output_norm(x)

        # Conditioning vector: mean-pool → project through bottleneck
        c = self.cond_proj(x.mean(dim=1))  # (B, cond_dim)

        # Compress L → num_latent_vectors via learned query pooling
        keys = self.pool_proj_k(x)    # (B, L, D)
        values = self.pool_proj_v(x)  # (B, L, D)
        queries = self.latent_queries.expand(x.size(0), -1, -1)  # (B, M, D)

        # Scaled dot-product (not self-attention — this is a cross-pool op)
        scale = math.sqrt(self.cfg.latent_dim)
        attn = torch.bmm(queries, keys.transpose(1, 2)) / scale  # (B, M, L)
        attn = F.softmax(attn, dim=-1)
        z0 = torch.bmm(attn, values)  # (B, M, D)

        return z0, c


# ===========================================================================
#  2. DCM_LatentDiffuser — Discrete-time diffusion on continuous latents
# ===========================================================================

class _SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps → (B, dim) embeddings."""
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _DenoisingMLP(nn.Module):
    """
    Context-conditional MLP denoiser with AdaLN (Adaptive Layer Norm).

    Predicts x₀ (the clean latent) from noisy input zₜ, timestep t,
    and a conditioning vector c that carries context information.

    AdaLN modulation (from DiT): the conditioning vector predicts per-layer
    scale and shift parameters that modulate the LayerNorm output. This lets
    the denoiser produce different z₀ predictions for different contexts.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, time_dim: int,
                 num_layers: int, cond_dim: int):
        super().__init__()
        self.time_embed = _SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Context conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: concat(z_t, time, cond) → hidden
        self.input_proj = nn.Linear(latent_dim + hidden_dim * 2, hidden_dim)

        # Residual MLP blocks
        layers = []
        for _ in range(num_layers):
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(layers)
        self.num_blocks = num_layers

        # AdaLN: per-block scale and shift predicted from conditioning
        self.adaLN_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 2)
            for _ in range(num_layers)
        ])

        # Output projection back to latent_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: (B, M, latent_dim) — noisy latents
            t:   (B,) — integer timesteps
            c:   (B, cond_dim) — context conditioning vector
        Returns:
            z0_pred: (B, M, latent_dim) — predicted clean latents (x₀-prediction)
        """
        B, M, D = z_t.shape

        # Time conditioning
        t_emb = self.time_embed(t)         # (B, time_dim)
        t_emb = self.time_proj(t_emb)      # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, M, -1)  # (B, M, hidden_dim)

        # Context conditioning
        c_emb = self.cond_proj(c)          # (B, hidden_dim)
        c_emb = c_emb.unsqueeze(1).expand(-1, M, -1)  # (B, M, hidden_dim)

        # Concat noisy latents with time + context embeddings
        h = torch.cat([z_t, t_emb, c_emb], dim=-1)  # (B, M, D + 2*hidden_dim)
        h = self.input_proj(h)                        # (B, M, hidden_dim)

        # Residual blocks with AdaLN modulation
        for i in range(self.num_blocks):
            base = i * 4
            residual = h
            h = self.blocks[base](h)      # LayerNorm (base normalization)

            # AdaLN: conditioning predicts per-layer scale and shift
            scale, shift = self.adaLN_projs[i](c_emb).chunk(2, dim=-1)
            h = h * (1 + scale) + shift   # Modulate

            h = self.blocks[base + 1](h)  # Linear
            h = self.blocks[base + 2](h)  # SiLU
            h = self.blocks[base + 3](h)  # Linear
            h = h + residual              # Residual skip

        return self.output_proj(h)


class DCM_LatentDiffuser(nn.Module):
    """
    Discrete-time latent diffusion operating on continuous vectors z₀.

    Forward process:   z₀ → zₜ  (add noise at timestep t)
    Reverse process:   zₜ → ẑ₀  (denoise via learned MLP, x₀-prediction)

    Uses a linear beta schedule with configurable endpoints.
    """

    def __init__(self, cfg: DCMConfig):
        super().__init__()
        self.cfg = cfg

        # ----- Noise schedule (precomputed, registered as buffers) -----
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # ----- Denoising network (context-conditioned) -----
        self.denoiser = _DenoisingMLP(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.denoiser_hidden_dim,
            time_dim=cfg.time_embed_dim,
            num_layers=cfg.denoiser_num_layers,
            cond_dim=cfg.cond_dim,
        )

    # ---- Forward diffusion (add noise) ----

    def q_sample(
        self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: sample zₜ ~ q(zₜ | z₀).

        Args:
            z0:    (B, M, D) clean latents
            t:     (B,)      integer timesteps in [0, T)
            noise: optional pre-sampled Gaussian noise
        Returns:
            z_t:   (B, M, D) noisy latents
            noise: (B, M, D) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(z0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]            # (B,1,1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        z_t = sqrt_alpha * z0 + sqrt_one_minus * noise
        return z_t, noise

    # ---- Reverse diffusion (denoise) ----

    def predict_z0(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Predict clean latents ẑ₀ from noisy zₜ, conditioned on context c."""
        return self.denoiser(z_t, t, c)

    def diffusion_loss(self, z0: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the diffusion training loss (MSE between z₀ and ẑ₀).

        Args:
            z0: (B, M, D) clean latents from encoder
            c:  (B, cond_dim) conditioning vector from encoder

        Returns:
            loss:    scalar MSE
            z0_pred: (B, M, D) predicted clean latents (for downstream use)
        """
        B = z0.size(0)
        t = torch.randint(0, self.cfg.diffusion_steps, (B,), device=z0.device)

        z_t, _ = self.q_sample(z0, t)
        z0_pred = self.predict_z0(z_t, t, c)

        loss = F.mse_loss(z0_pred, z0)
        return loss, z0_pred

    @torch.no_grad()
    def sample(self, z_t: torch.Tensor, c: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """
        Full reverse sampling (DDIM-style) for inference, conditioned on c.

        Args:
            z_t: (B, M, D) starting noise
            c:   (B, cond_dim) conditioning vector from encoder
            num_steps: number of denoising steps
        """
        step_indices = torch.linspace(
            self.cfg.diffusion_steps - 1, 0, num_steps, dtype=torch.long,
            device=z_t.device
        )

        for i, t_val in enumerate(step_indices):
            t = t_val.expand(z_t.size(0))
            z0_pred = self.predict_z0(z_t, t, c)

            if i < len(step_indices) - 1:
                # Move to the next (less noisy) step
                t_next = step_indices[i + 1]
                alpha_t = self.alphas_cumprod[t_val]
                alpha_next = self.alphas_cumprod[t_next]

                # DDIM-like deterministic step
                z_t = (
                    torch.sqrt(alpha_next / alpha_t) * z_t
                    + (torch.sqrt(1 - alpha_next) - torch.sqrt(
                        (alpha_next * (1 - alpha_t)) / alpha_t
                    )) * z0_pred
                )
            else:
                z_t = z0_pred

        return z_t


# ===========================================================================
#  3. AbstractDecoderHead — Swappable LLM interface
# ===========================================================================

class AbstractDecoderHead(nn.Module, abc.ABC):
    """
    Abstract base class for any decoder-only LLM head.
    Enforces a consistent interface so the base model can be swapped
    (e.g., Qwen → LLaMA → Mistral) without touching the rest of DCM.
    """

    @abc.abstractmethod
    def get_embedding_layer(self) -> nn.Module:
        """Return the token embedding layer for encoding input_ids."""
        ...

    @abc.abstractmethod
    def forward_with_memory(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run a forward pass with injected continuous memory.

        Args:
            input_ids: (B, L) token IDs for the continuation sequence
            memory:    (B, M, D) denoised latent vectors (soft prompts)
            labels:    (B, L) optional labels for cross-entropy loss

        Returns:
            logits: (B, L, vocab_size)
            loss:   scalar CE loss if labels provided, else None
        """
        ...


# ===========================================================================
#  4. QwenLoRAHead — 4-bit Qwen + LoRA + Prefix Memory Injection
# ===========================================================================

class QwenLoRAHead(AbstractDecoderHead):
    """
    Wraps Qwen2.5-7B-Instruct with:
        - bitsandbytes 4-bit quantization (NF4)
        - LoRA on q_proj / v_proj
        - Prefix-tuning style memory injection (Option A)

    The denoised memory vectors ẑ₀ are treated as "soft prompt" embeddings
    and prepended to the token embeddings before passing into the Qwen model.
    Qwen's self-attention naturally attends over these extra prefix positions.
    """

    def __init__(self, cfg: DCMConfig, device_map: str = "auto"):
        super().__init__()
        self.cfg = cfg
        self._load_quantized_model(device_map)
        self._apply_lora()

        # Projection to align denoised memory dim → Qwen's hidden dim
        # (Identity if latent_dim == Qwen hidden, but kept for flexibility)
        # Must be on the same device as the quantized model weights
        self._model_device = next(self.model.parameters()).device
        self.memory_proj = nn.Linear(cfg.latent_dim, self.hidden_size, bias=False).to(self._model_device)

    def _load_quantized_model(self, device_map: str):
        """Load Qwen in 4-bit NF4 quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.base_model_name,
            trust_remote_code=True,
        )
        self.hidden_size = self.model.config.hidden_size

    def _apply_lora(self):
        """Apply LoRA adapters to Qwen's attention projections."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def get_embedding_layer(self) -> nn.Module:
        """Return Qwen's token embedding layer."""
        # Access the base model embedding through the PEFT wrapper
        return self.model.get_input_embeddings()

    def forward_with_memory(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prefix-tuning style injection:
            1. Embed input_ids → token_embeds (B, L, H)
            2. Project memory   → mem_embeds  (B, M, H)
            3. Concat [mem_embeds ; token_embeds] → (B, M+L, H)
            4. Forward through Qwen with inputs_embeds (bypass token embedding)
            5. Slice logits to only return positions corresponding to real tokens
        """
        # Token embeddings
        embed_layer = self.get_embedding_layer()
        token_embeds = embed_layer(input_ids)  # (B, L, H)

        # Project denoised memory to match Qwen hidden size
        mem_embeds = self.memory_proj(memory)  # (B, M, H)

        # Prepend memory as soft prompts
        combined = torch.cat([mem_embeds, token_embeds], dim=1)  # (B, M+L, H)

        M = memory.size(1)
        L = input_ids.size(1)

        # Build attention mask: all positions attend to everything (causal handled internally)
        attention_mask = torch.ones(
            combined.size(0), M + L,
            device=combined.device, dtype=combined.dtype
        )

        # Shift labels: memory positions have no labels (use -100 to ignore)
        shifted_labels = None
        if labels is not None:
            memory_ignore = torch.full(
                (labels.size(0), M), -100,
                device=labels.device, dtype=labels.dtype
            )
            shifted_labels = torch.cat([memory_ignore, labels], dim=1)  # (B, M+L)

        # Forward through Qwen
        outputs = self.model(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            labels=shifted_labels,
        )

        # Slice to only return logits for real token positions
        logits = outputs.logits[:, M:, :]  # (B, L, vocab_size)
        loss = outputs.loss  # Already computed with shifted_labels ignoring memory positions

        return logits, loss


# ===========================================================================
#  5. DiffusionContextModel — Master Orchestrator
# ===========================================================================

class DiffusionContextModel(nn.Module):
    """
    Top-level DCM module.

    Forward pass (training):
        1. Encode context tokens → z₀           (SSM Encoder)
        2. Noise z₀ → zₜ, denoise zₜ → ẑ₀      (Diffuser — also returns MSE loss)
        3. Inject ẑ₀ as prefix, predict next token (Qwen LoRA Head — returns CE loss)
        4. Combine: L_total = L_AR + λ · L_diffusion

    The context tokens and continuation tokens come from different parts of the
    same long document, split by the DataLoader.
    """

    def __init__(self, cfg: DCMConfig, device_map: str = "auto"):
        super().__init__()
        self.cfg = cfg
        self.decoder = QwenLoRAHead(cfg, device_map=device_map)

        # Determine the device Qwen landed on so encoder/diffuser match
        self._device = next(self.decoder.model.parameters()).device
        self.encoder = DCM_SSMEncoder(cfg).to(self._device)
        self.diffuser = DCM_LatentDiffuser(cfg).to(self._device)

    def get_embedding_layer(self) -> nn.Module:
        return self.decoder.get_embedding_layer()

    def forward(
        self,
        context_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
        continuation_labels: torch.Tensor,
    ) -> dict:
        """
        Full training forward pass.

        Args:
            context_ids:        (B, L_ctx)  — preceding context token IDs
            continuation_ids:   (B, L_cont) — continuation token IDs (model input)
            continuation_labels:(B, L_cont) — shifted labels for AR loss

        Returns:
            dict with keys: loss, loss_ar, loss_diffusion, logits
        """
        # --- Step 1: Encode context → latents z₀ + conditioning c ---
        device = self._device
        context_ids = context_ids.to(device)
        continuation_ids = continuation_ids.to(device)
        continuation_labels = continuation_labels.to(device)

        embed_layer = self.get_embedding_layer()
        context_embeds = embed_layer(context_ids)         # (B, L_ctx, H)
        z0, c = self.encoder(context_embeds)              # (B, M, D), (B, cond_dim)

        # --- Step 2: Conditional diffusion (denoise z_t → z0_pred, conditioned on c) ---
        loss_diff, z0_pred = self.diffuser.diffusion_loss(z0, c)  # scalar, (B, M, D)

        # --- Step 3: AR prediction conditioned on denoised memory ---
        logits, loss_ar = self.decoder.forward_with_memory(
            input_ids=continuation_ids,
            memory=z0_pred.detach() if not self.training else z0_pred,
            labels=continuation_labels,
        )

        # --- Step 4: Hybrid loss ---
        loss_total = loss_ar + self.cfg.lambda_diffusion * loss_diff

        return {
            "loss": loss_total,
            "loss_ar": loss_ar,
            "loss_diffusion": loss_diff,
            "logits": logits,
        }

    @torch.no_grad()
    def generate_with_memory(
        self,
        context_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        diffusion_steps: int = 50,
    ) -> torch.Tensor:
        """
        Inference: encode context → conditioning vector c, then sample
        context-specific memory from noise via conditional diffusion,
        then autoregressively generate tokens.
        """
        self.eval()
        device = self._device
        context_ids = context_ids.to(device)
        prompt_ids = prompt_ids.to(device)

        # Encode context → conditioning vector
        embed_layer = self.get_embedding_layer()
        context_embeds = embed_layer(context_ids)
        z0, c = self.encoder(context_embeds)

        # Conditional diffusion: sample memory from noise, guided by c
        B, M, D = z0.shape
        z_t = torch.randn(B, M, D, device=device)
        memory = self.diffuser.sample(z_t, c, num_steps=diffusion_steps)

        # Greedy / temperature sampling
        generated = prompt_ids
        for _ in range(max_new_tokens):
            logits, _ = self.decoder.forward_with_memory(
                input_ids=generated, memory=memory
            )
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
