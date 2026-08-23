"""nano-infer's from-scratch Qwen2.5 forward pass.

Built one component at a time, each verified against HuggingFace before the next
is added (Phase 1). No transformers.generate(), no HF model classes — we load the
raw safetensors weights onto our own code. HF is used only to obtain the weights.

Growth log (append as components land):
  - [x] QwenConfig + weight loader
  - [x] token embedding
  - [x] RMSNorm
  - [x] RoPE
  - [ ] grouped-query attention
  - [ ] SwiGLU MLP
  - [ ] full block / stack / final norm + tied logits
  - [ ] greedy decode (no cache)
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from . import config as cfg


@dataclass(frozen=True)
class QwenConfig:
    """Architecture constants — read from the model on 2026-08-20, not assumed."""
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_layers: int = 24
    num_q_heads: int = 14
    num_kv_heads: int = 2
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0     # Qwen2.5 default
    tie_word_embeddings: bool = True

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim      # 896

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim     # 128


def load_weights(dtype: torch.dtype = cfg.DTYPE, device: str = cfg.DEVICE) -> dict:
    """Read every tensor from the safetensors checkpoint into a name->tensor dict,
    moved to our dtype/device. This is the raw material; our modules index into it."""
    path = snapshot_download(cfg.MODEL_NAME, allow_patterns=["*.safetensors"])
    files = glob.glob(os.path.join(path, "*.safetensors"))
    weights: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(f, framework="pt", device=device) as st:
            for key in st.keys():
                weights[key] = st.get_tensor(key).to(dtype)
    return weights


# --- component 1: token embedding ------------------------------------------

def embed_tokens(input_ids: torch.Tensor, weights: dict) -> torch.Tensor:
    """Look up each token id's row in the embedding table.

    input_ids : [batch, seq]  (integer token ids)
    returns   : [batch, seq, hidden]  (the token's learned vector)

    That's the whole operation — indexing rows of model.embed_tokens.weight.
    No math, just a gather. It turns discrete token ids into the continuous
    vectors the rest of the transformer operates on.
    """
    table = weights["model.embed_tokens.weight"]      # [vocab, hidden]
    return table[input_ids]                            # fancy-indexing gather


# --- component 2: RMSNorm ---------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square normalization (no mean-subtraction, no bias).

    x      : [..., hidden]
    weight : [hidden]   learned per-dimension rescale
    returns: [..., hidden]  same shape, magnitude-normalized

        rms = sqrt(mean(x^2) + eps)
        out = (x / rms) * weight

    Dtype choreography matches HF exactly: the normalize is done in fp32 for
    stability, cast back to the input dtype, THEN multiplied by the weight. Skip
    the fp32 step and the parity diff balloons — the formula alone isn't enough.
    """
    in_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)     # mean of squares
    x = x * torch.rsqrt(variance + eps)                # 1/rms, in fp32
    return weight * x.to(in_dtype)                      # back to fp16, then rescale


# --- component 3: RoPE (rotary position embedding) -------------------------

def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device: str = cfg.DEVICE, dtype: torch.dtype = cfg.DTYPE):
    """Precompute cos/sin tables for positions 0..seq_len-1.

    Returns cos, sin each of shape [seq_len, head_dim]. Each of the head_dim/2
    frequency pairs spins at its own rate theta_i = theta^(-2i/head_dim) — fast
    hands (small i) and slow hands (large i). The tables are duplicated across the
    two halves (cat(freqs, freqs)) to match HF's rotate_half layout below.

    Computed in fp32 for a clean trig result, then cast to fp16 like HF does.
    """
    # inv_freq[i] = 1 / theta^(2i/head_dim), i = 0..head_dim/2-1   -> [head_dim/2]
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (i / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)   # [seq]
    freqs = torch.outer(pos, inv_freq)                 # [seq, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)            # [seq, head_dim] (duplicated)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF/Llama layout: pair dim i with dim i+head_dim/2. Takes the second half,
    negates it, and moves it to the front: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor):
    """Rotate the query and key vectors by their positions.

    q, k    : [batch, heads, seq, head_dim]
    cos,sin : [seq, head_dim]  -> broadcast over batch and heads
    returns : rotated q, k of the same shapes

        x_rotated = x * cos + rotate_half(x) * sin
    """
    cos = cos.unsqueeze(0).unsqueeze(0)                 # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
