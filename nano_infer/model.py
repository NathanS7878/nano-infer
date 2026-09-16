"""nano-infer's from-scratch Qwen2.5 forward pass.

Built one component at a time, each verified against HuggingFace before the next
is added (Phase 1). No transformers.generate(), no HF model classes — we load the
raw safetensors weights onto our own code. HF is used only to obtain the weights.

Growth log (append as components land):
  - [x] QwenConfig + weight loader
  - [x] token embedding
  - [x] RMSNorm
  - [x] RoPE
  - [x] grouped-query attention
  - [x] SwiGLU MLP
  - [x] full block / stack / final norm + tied logits
  - [x] greedy decode (no cache)
"""
from __future__ import annotations

import glob
import os
import functools
import json
from dataclasses import dataclass, field

import contextlib

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

from . import config as cfg


@functools.lru_cache(maxsize=None)
def checkpoint_config(model_name: str) -> dict:
    """The selected checkpoint's config.json, validated against what this engine
    actually implements.

    Everything the forward pass depends on is checked here, and anything it does
    not implement raises. The failure this prevents is the quiet one: a model
    that loads, runs, and produces fluent text through an architecture detail
    the engine ignores -- RoPE scaling, an untied output head, sliding-window
    attention -- exactly like the adjacent-pair RoPE bug in ROADMAP #12.
    """
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    path = try_to_load_from_cache(model_name, "config.json")
    if not isinstance(path, str):
        path = hf_hub_download(model_name, "config.json")
    with open(path, encoding="utf-8") as f:
        c = json.load(f)

    problems = []
    if c.get("model_type") != "qwen2":
        problems.append(f"model_type {c.get('model_type')!r} (engine implements qwen2)")
    if c.get("hidden_act") != "silu":
        problems.append(f"hidden_act {c.get('hidden_act')!r} (engine implements SwiGLU/silu)")
    if c.get("rope_scaling"):
        problems.append(f"rope_scaling {c['rope_scaling']!r} (engine implements plain RoPE)")
    if not c.get("tie_word_embeddings", False):
        problems.append("untied lm_head (engine reuses the embedding as the output head)")
    if c.get("use_sliding_window", False):
        problems.append("sliding-window attention (engine attends to the full context)")
    if c["hidden_size"] % c["num_attention_heads"]:
        problems.append("hidden_size not divisible by num_attention_heads")
    if c["num_attention_heads"] % c["num_key_value_heads"]:
        problems.append("query heads not a multiple of KV heads")
    if problems:
        raise ValueError(f"{model_name} is not supported by this engine: "
                         + "; ".join(problems))
    return c


def _arch(key: str):
    """Default factory reading one field from the SELECTED model's config.json at
    construction time, so QwenConfig() always describes the weights load_weights()
    will actually load."""
    def factory():
        c = checkpoint_config(cfg.MODEL_NAME)
        if key == "head_dim":
            return c["hidden_size"] // c["num_attention_heads"]
        return c[key]
    return factory


@dataclass(frozen=True)
class QwenConfig:
    """Architecture constants, read from the selected checkpoint's config.json.

    Until 2026-09-16 these were hard-coded 0.5B values (read once from the model
    on 2026-08-20). They now come from config.MODEL_NAME's own config.json, which
    for the default 0.5B model yields exactly those constants -- pinned by
    tests/test_model_select.py -- and for any other model raises if the
    architecture needs something the engine does not implement.
    """
    vocab_size: int = field(default_factory=_arch("vocab_size"))
    hidden_size: int = field(default_factory=_arch("hidden_size"))
    intermediate_size: int = field(default_factory=_arch("intermediate_size"))
    num_layers: int = field(default_factory=_arch("num_hidden_layers"))
    num_q_heads: int = field(default_factory=_arch("num_attention_heads"))
    num_kv_heads: int = field(default_factory=_arch("num_key_value_heads"))
    head_dim: int = field(default_factory=_arch("head_dim"))
    rms_norm_eps: float = field(default_factory=_arch("rms_norm_eps"))
    rope_theta: float = field(default_factory=_arch("rope_theta"))
    tie_word_embeddings: bool = field(default_factory=_arch("tie_word_embeddings"))

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim      # 896 on 0.5B, 1536 on 1.5B

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim     # 128 on 0.5B, 256 on 1.5B


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


def apply_rope_positions(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor):
    """RoPE where every sequence in the batch sits at its OWN position.

    q, k    : [batch, heads, n, head_dim]
    cos,sin : [batch, n, head_dim]  -> broadcast over heads only

    Phase 1's apply_rope assumes one shared position range for the whole batch,
    which holds while every sequence advances in lockstep. Continuous batching
    breaks that: a sequence admitted 40 steps ago is at position 60 while its
    neighbour is at position 3. Each row must be rotated by its own position.
    """
    cos = cos.unsqueeze(1)                              # [b, 1, n, head_dim]
    sin = sin.unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# --- component 4: grouped-query attention ----------------------------------

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: make each of the few KV heads serve n_rep query heads by repeating it.

    x : [batch, num_kv_heads, seq, head_dim]  ->  [batch, num_kv_heads*n_rep, seq, head_dim]

    The reshape order maps query head h to KV head h // n_rep, so query heads
    0..6 share KV head 0, and 7..13 share KV head 1 (n_rep = 14/2 = 7). This is
    memory, not compute: we don't store 14 KV heads, only 2 — that's the whole
    point of GQA, and why the KV cache is 7x smaller than it would be otherwise.
    """
    if n_rep == 1:
        return x
    b, n_kv, seq, hd = x.shape
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, seq, hd)
            .reshape(b, n_kv * n_rep, seq, hd))


def attention(x: torch.Tensor, weights: dict, layer: int, cos: torch.Tensor,
              sin: torch.Tensor, cf: "QwenConfig") -> torch.Tensor:
    """Full self-attention for one layer (input already RMSNorm'd by the caller).

    x : [batch, seq, hidden]  ->  returns [batch, seq, hidden]
    """
    b, seq, _ = x.shape
    p = f"model.layers.{layer}.self_attn."

    # 1. project to Q, K, V. Qwen puts a bias on all three (o_proj has none).
    q = F.linear(x, weights[p + "q_proj.weight"], weights[p + "q_proj.bias"])  # [b,seq,896]
    k = F.linear(x, weights[p + "k_proj.weight"], weights[p + "k_proj.bias"])  # [b,seq,128]
    v = F.linear(x, weights[p + "v_proj.weight"], weights[p + "v_proj.bias"])  # [b,seq,128]

    # 2. split into heads: q has 14 heads, k/v have 2. -> [b, heads, seq, head_dim]
    q = q.view(b, seq, cf.num_q_heads, cf.head_dim).transpose(1, 2)   # [b,14,seq,64]
    k = k.view(b, seq, cf.num_kv_heads, cf.head_dim).transpose(1, 2)  # [b, 2,seq,64]
    v = v.view(b, seq, cf.num_kv_heads, cf.head_dim).transpose(1, 2)  # [b, 2,seq,64]

    # 3. rotate Q and K by position (RoPE). V is never rotated.
    q, k = apply_rope(q, k, cos, sin)

    # 4. GQA: expand the 2 KV heads up to 14 so every query head has a partner.
    n_rep = cf.num_q_heads // cf.num_kv_heads
    k = repeat_kv(k, n_rep)                                           # [b,14,seq,64]
    v = repeat_kv(v, n_rep)

    # 5. attention scores: every query dotted with every key, scaled by 1/sqrt(d).
    scale = cf.head_dim ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale                       # [b,14,seq,seq]

    # 6. causal mask: token i may only attend to j <= i (no peeking ahead).
    mask = torch.triu(torch.full((seq, seq), float("-inf"), device=x.device,
                                 dtype=torch.float32), diagonal=1)
    scores = scores.float() + mask                                   # upcast for stable softmax

    # 7. softmax over keys (in fp32, like HF), then blend the values.
    probs = torch.softmax(scores, dim=-1).to(x.dtype)                # [b,14,seq,seq]
    out = probs @ v                                                  # [b,14,seq,64]

    # 8. merge heads back and apply the output projection (no bias).
    out = out.transpose(1, 2).reshape(b, seq, cf.q_dim)              # [b,seq,896]
    return F.linear(out, weights[p + "o_proj.weight"])


# --- component 5: SwiGLU MLP ------------------------------------------------

def mlp(x: torch.Tensor, weights: dict, layer: int,
        use_kernels: bool = False) -> torch.Tensor:
    """SwiGLU feed-forward. No biases on any projection.

    x : [batch, seq, hidden] -> [batch, seq, hidden]

        gate = x @ Wgate      up = x @ Wup       (both hidden -> intermediate)
        hidden = silu(gate) * up                 (the gated valve)
        out    = hidden @ Wdown                  (intermediate -> hidden)

    silu(z) = z * sigmoid(z). The silu(gate) term acts as a smooth, learned gate
    on `up`: where gate is very negative the valve nearly closes, where positive
    it opens. Fusing this elementwise step is the Phase 3 SwiGLU kernel.
    """
    p = f"model.layers.{layer}.mlp."
    gate = _qlinear(x, weights[p + "gate_proj.weight"])             # [b,seq,4864]
    up = _qlinear(x, weights[p + "up_proj.weight"])                 # [b,seq,4864]
    # kernel 2 when opted in; the default keeps Phase 1 on the reference path.
    hidden = (_k().swiglu_forward(gate, up) if use_kernels
              else F.silu(gate) * up)
    return _qlinear(hidden, weights[p + "down_proj.weight"])         # [b,seq,896]


# --- assembly: block, full forward, greedy decode --------------------------

def decoder_block(x: torch.Tensor, weights: dict, layer: int, cos: torch.Tensor,
                  sin: torch.Tensor, cf: "QwenConfig") -> torch.Tensor:
    """One transformer block, pre-norm with two residual connections.

    The original signal flows straight through; attention and the MLP each add a
    refinement onto it, so a block only has to learn a small nudge, not rebuild
    the whole representation. That's what lets 24 of them stack.
    """
    ln = f"model.layers.{layer}."
    residual = x
    h = rms_norm(x, weights[ln + "input_layernorm.weight"], cf.rms_norm_eps)
    x = residual + attention(h, weights, layer, cos, sin, cf)         # add attn refinement

    residual = x
    h = rms_norm(x, weights[ln + "post_attention_layernorm.weight"], cf.rms_norm_eps)
    x = residual + mlp(h, weights, layer)                             # add mlp refinement
    return x


def forward(input_ids: torch.Tensor, weights: dict, cf: "QwenConfig") -> torch.Tensor:
    """Full forward pass: token ids -> logits over the vocabulary.

    input_ids : [batch, seq]  ->  logits [batch, seq, vocab_size]

    No KV cache — the whole sequence is recomputed each call. Deliberately the
    slow, obviously-correct version (Phase 1). Phase 2 adds the cache.
    """
    b, seq = input_ids.shape
    x = embed_tokens(input_ids, weights)                             # [b,seq,hidden]
    cos, sin = build_rope_cache(seq, cf.head_dim, cf.rope_theta,
                                device=input_ids.device, dtype=x.dtype)
    for i in range(cf.num_layers):
        x = decoder_block(x, weights, i, cos, sin, cf)
    x = rms_norm(x, weights["model.norm.weight"], cf.rms_norm_eps)   # final norm

    # tied embeddings: the output projection reuses embed_tokens.weight [vocab,hidden].
    # F.linear(x, W) = x @ W.T -> [b, seq, vocab].
    return F.linear(x, weights["model.embed_tokens.weight"])


@torch.no_grad()
def greedy_decode(input_ids: torch.Tensor, weights: dict, cf: "QwenConfig",
                  max_new_tokens: int) -> torch.Tensor:
    """Greedy decode with NO cache: recompute the whole sequence every step, take
    the argmax of the last position, append, repeat.

    input_ids : [batch, seq]  ->  returns [batch, max_new_tokens]

    Step t recomputes seq+t tokens, so total work grows with the SQUARE of the
    output length. That is the cost Phase 2's KV cache exists to remove.
    """
    ids = input_ids
    generated = []
    for _ in range(max_new_tokens):
        logits = forward(ids, weights, cf)                           # [b,seq,vocab]
        next_ids = logits[:, -1].argmax(dim=-1)                      # [b]
        generated.append(next_ids)
        ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
    return torch.stack(generated, dim=1)                             # [b, new_tokens]


# ===========================================================================
# Phase 2 — KV cache. Everything above is the Phase 1 reference implementation
# and is deliberately left untouched: it is what the cached path is graded
# against. The functions below add prefill/decode paths alongside it.
# ===========================================================================

# ---------------------------------------------------------------------------
# Phase 3: optional custom CUDA kernels
#
# A module-level switch rather than a parameter threaded through eight
# functions. The rule from `model.py`'s layering (see ROADMAP.md) is that the
# PHASE 1 FUNCTIONS ARE THE ANSWER KEY and are never optimized in place, so
# `rms_norm`, `apply_rope`, `attention` and `forward` deliberately do NOT
# consult this flag. Only the Phase 2 cached/paged paths do, which keeps a
# pure-PyTorch reference available in the same process for comparison.
#
# `mlp` is shared by Phase 1 and Phase 2, so it takes an explicit keyword that
# defaults to False rather than reading the global — Phase 1's call site is
# unchanged by construction.
# ---------------------------------------------------------------------------

_KERNELS_ENABLED = False


def _qlinear(x: torch.Tensor, w, bias=None) -> torch.Tensor:
    """Linear over a weight that may be an fp16 tensor OR a packed quantized one.

    The packed branch routes to the fused dequant-matmul kernel, which unpacks
    in registers — no fp16 copy of the weight is ever created, which is the only
    way the memory saving is real. Plain tensors fall through to F.linear, so
    every existing call site is unchanged when weights are not quantized.

    Bias is applied after: the kernel computes x @ W^T only, and q/k/v carry a
    bias in this model while o_proj and the MLP do not.
    """
    from . import quant as _q
    if _q.is_packed(w):
        if isinstance(w, _q.Int4Tensor):
            y = _k().int4_matmul(x, w.packed, w.scale, w.zero, w.shape[1], w.group)
        else:
            y = _k().int8_matmul(x, w.q, w.scale)
        return y if bias is None else y + bias
    return F.linear(x, w, bias)



def kernels_enabled() -> bool:
    return _KERNELS_ENABLED


def set_kernels(enabled: bool) -> bool:
    """Turn the custom kernels on/off. Returns the previous setting.

    Compiles on first enable (~40 s), then cached. Raises if CUDA or the
    toolchain is unavailable, rather than silently falling back — a benchmark
    that quietly measured the PyTorch path while claiming kernels would be
    worse than no benchmark.
    """
    global _KERNELS_ENABLED
    previous = _KERNELS_ENABLED
    if enabled:
        from nano_infer import kernels as _k
        _k.load()
    _KERNELS_ENABLED = bool(enabled)
    return previous


@contextlib.contextmanager
def using_kernels(enabled: bool = True):
    """Scoped version of set_kernels, so a benchmark can A/B in one process."""
    previous = set_kernels(enabled)
    try:
        yield
    finally:
        set_kernels(previous)


def _k():
    from nano_infer import kernels as _mod
    return _mod.load()


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Kernel 1 if enabled, else the Phase 1 reference."""
    if _KERNELS_ENABLED:
        return _k().rmsnorm_forward(x, weight, eps)
    return rms_norm(x, weight, eps)


def _rope(q, k, cos, sin, per_sequence: bool):
    """Kernel 3 if enabled, else the Phase 1 / Phase 2 reference.

    Both position layouts go to the same kernel: it takes cos/sin as either
    [n, head_dim] (shared range) or [batch, n, head_dim] (per sequence).
    """
    if _KERNELS_ENABLED:
        return _k().rope_forward(q, cos, sin), _k().rope_forward(k, cos, sin)
    return (apply_rope_positions(q, k, cos, sin) if per_sequence
            else apply_rope(q, k, cos, sin))


def attention_cached(x: torch.Tensor, weights: dict, layer: int,
                     cos: torch.Tensor, sin: torch.Tensor, cf: "QwenConfig",
                     cache: "KVCache", start_pos: int) -> torch.Tensor:
    """Attention that reads and writes the KV cache.

    x         : [batch, n, hidden]   n = prompt length (prefill) or 1 (decode)
    start_pos : absolute position of x[0] in the sequence
    cos, sin  : [n, head_dim] rotations for exactly those positions

    Prefill (n > 1) needs a causal mask so a token cannot see its future. Decode
    (n == 1) needs NO mask at all: there is a single query and every cached
    position is, by construction, in its past.
    """
    b, n, _ = x.shape
    p = f"model.layers.{layer}.self_attn."

    q = _qlinear(x, weights[p + "q_proj.weight"], weights[p + "q_proj.bias"])
    k = _qlinear(x, weights[p + "k_proj.weight"], weights[p + "k_proj.bias"])
    v = _qlinear(x, weights[p + "v_proj.weight"], weights[p + "v_proj.bias"])

    q = q.view(b, n, cf.num_q_heads, cf.head_dim).transpose(1, 2)     # [b,14,n,64]
    k = k.view(b, n, cf.num_kv_heads, cf.head_dim).transpose(1, 2)    # [b, 2,n,64]
    v = v.view(b, n, cf.num_kv_heads, cf.head_dim).transpose(1, 2)

    # RoPE first, THEN cache: the rotation is part of the frozen past.
    q, k = _rope(q, k, cos, sin, per_sequence=False)

    cache.append(layer, k, v, start_pos)
    total = start_pos + n
    k_all, v_all = cache.view(layer, total)                           # [b,2,total,64]

    n_rep = cf.num_q_heads // cf.num_kv_heads
    k_all = repeat_kv(k_all, n_rep)                                   # [b,14,total,64]
    v_all = repeat_kv(v_all, n_rep)

    scale = cf.head_dim ** -0.5
    scores = (q @ k_all.transpose(-1, -2)) * scale                    # [b,14,n,total]
    scores = scores.float()

    if n > 1:
        # causal mask over the [n, total] window: query i (absolute start_pos+i)
        # may attend to key j only when j <= start_pos + i.
        q_pos = torch.arange(start_pos, total, device=x.device).unsqueeze(1)
        k_pos = torch.arange(total, device=x.device).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

    probs = torch.softmax(scores, dim=-1).to(x.dtype)
    out = probs @ v_all                                               # [b,14,n,64]
    out = out.transpose(1, 2).reshape(b, n, cf.q_dim)
    return _qlinear(out, weights[p + "o_proj.weight"])


def decoder_block_cached(x: torch.Tensor, weights: dict, layer: int,
                         cos: torch.Tensor, sin: torch.Tensor, cf: "QwenConfig",
                         cache: "KVCache", start_pos: int) -> torch.Tensor:
    """Same pre-norm block as Phase 1, using the cached attention path."""
    ln = f"model.layers.{layer}."
    residual = x
    h = _rms(x, weights[ln + "input_layernorm.weight"], cf.rms_norm_eps)
    x = residual + attention_cached(h, weights, layer, cos, sin, cf, cache, start_pos)

    residual = x
    h = _rms(x, weights[ln + "post_attention_layernorm.weight"], cf.rms_norm_eps)
    x = residual + mlp(h, weights, layer, use_kernels=_KERNELS_ENABLED)
    return x


def forward_cached(input_ids: torch.Tensor, weights: dict, cf: "QwenConfig",
                   cache: "KVCache", start_pos: int, rope: tuple) -> torch.Tensor:
    """Run the stack over `input_ids`, updating the cache. Returns LAST-position
    logits only: [batch, vocab].

    Computing logits for every position (as Phase 1's forward does) is pure waste
    during generation — only the last position picks the next token, and the vocab
    is 151,936 wide. At batch 32 / prompt 34 that is 330 MB of logits computed to
    use 9.7 MB of them.
    """
    cos_all, sin_all = rope
    n = input_ids.shape[1]
    cos = cos_all[start_pos:start_pos + n]
    sin = sin_all[start_pos:start_pos + n]

    x = embed_tokens(input_ids, weights)
    for i in range(cf.num_layers):
        x = decoder_block_cached(x, weights, i, cos, sin, cf, cache, start_pos)
    x = x[:, -1:]                                                     # last position only
    x = _rms(x, weights["model.norm.weight"], cf.rms_norm_eps)
    return F.linear(x, weights["model.embed_tokens.weight"])[:, 0]    # [b, vocab]


@torch.no_grad()
def generate_cached(input_ids: torch.Tensor, weights: dict, cf: "QwenConfig",
                    max_new_tokens: int, cache: "KVCache | None" = None):
    """Greedy decode with a KV cache: one prefill over the prompt, then one
    cheap decode step per new token.

    input_ids : [batch, seq]  ->  returns [batch, max_new_tokens]
    """
    from .cache import KVCache

    b, seq = input_ids.shape
    if cache is None:
        cache = KVCache(cf.num_layers, b, cf.num_kv_heads,
                        seq + max_new_tokens, cf.head_dim,
                        dtype=weights["model.norm.weight"].dtype,
                        device=input_ids.device)

    rope = build_rope_cache(seq + max_new_tokens, cf.head_dim, cf.rope_theta,
                            device=input_ids.device,
                            dtype=weights["model.norm.weight"].dtype)

    # --- prefill: whole prompt at once, compute-bound ---
    logits = forward_cached(input_ids, weights, cf, cache, 0, rope)
    next_ids = logits.argmax(dim=-1)                                  # [b]
    generated = [next_ids]

    # --- decode: one token at a time against the cache, memory-bound ---
    for step in range(1, max_new_tokens):
        pos = seq + step - 1
        logits = forward_cached(next_ids.unsqueeze(1), weights, cf, cache, pos, rope)
        next_ids = logits.argmax(dim=-1)
        generated.append(next_ids)

    cache.length = seq + max_new_tokens - 1
    return torch.stack(generated, dim=1)                              # [b, new_tokens]


# ---------------------------------------------------------------------------
# Paged KV cache path (Phase 2 step 2)
# ---------------------------------------------------------------------------

def attention_paged(x: torch.Tensor, weights: dict, layer: int,
                    cos: torch.Tensor, sin: torch.Tensor, cf: "QwenConfig",
                    cache: "PagedKVCache", plan, allowed,
                    lengths=None) -> torch.Tensor:
    """Attention against a paged cache.

    Same math as attention_cached; the difference is that K/V are scattered into
    blocks and gathered back through a block table rather than living in one
    contiguous span.

    `plan` (slot indices) and `allowed` (the combined causal + padding mask) are
    computed once per step by the caller and shared across all 24 layers — they
    depend only on positions, not on layer contents.
    """
    b, n, _ = x.shape
    p = f"model.layers.{layer}.self_attn."

    q = _qlinear(x, weights[p + "q_proj.weight"], weights[p + "q_proj.bias"])
    k = _qlinear(x, weights[p + "k_proj.weight"], weights[p + "k_proj.bias"])
    v = _qlinear(x, weights[p + "v_proj.weight"], weights[p + "v_proj.bias"])

    q = q.view(b, n, cf.num_q_heads, cf.head_dim).transpose(1, 2)
    k = k.view(b, n, cf.num_kv_heads, cf.head_dim).transpose(1, 2)
    v = v.view(b, n, cf.num_kv_heads, cf.head_dim).transpose(1, 2)

    q, k = _rope(q, k, cos, sin, per_sequence=True)

    cache.append(layer, k, v, plan)
    scale = cf.head_dim ** -0.5

    # Kernel 4 — decode only. Prefill (n > 1) still needs the masked multi-query
    # path: the kernel handles exactly one query token against the cached past,
    # which is what makes the causal mask unnecessary rather than optional.
    if _KERNELS_ENABLED and n == 1 and lengths is not None:
        out = _k().decode_attention_forward(
            q[:, :, 0, :].contiguous(),      # [b, q_heads, head_dim]
            cache.k[layer], cache.v[layer],  # the pool, walked in place
            plan.read, lengths, scale)       # no gather, no repeat_kv
        out = out.reshape(b, n, cf.q_dim)
        return _qlinear(out, weights[p + "o_proj.weight"])

    k_all, v_all = cache.gather(layer, plan)                          # [b,kvh,L,hd]

    n_rep = cf.num_q_heads // cf.num_kv_heads
    k_all = repeat_kv(k_all, n_rep)
    v_all = repeat_kv(v_all, n_rep)

    scores = (q @ k_all.transpose(-1, -2)).float() * scale            # [b,14,n,L]
    scores = scores.masked_fill(~allowed, float("-inf"))

    probs = torch.softmax(scores, dim=-1).to(x.dtype)
    out = (probs @ v_all).transpose(1, 2).reshape(b, n, cf.q_dim)
    return _qlinear(out, weights[p + "o_proj.weight"])


def forward_paged(input_ids: torch.Tensor, weights: dict, cf: "QwenConfig",
                  cache: "PagedKVCache", seq_ids: list, start_positions,
                  rope: tuple) -> torch.Tensor:
    """Run the stack over `input_ids` against a paged cache. Returns [batch, vocab]."""
    cos_all, sin_all = rope
    b, n = input_ids.shape
    # Per-sequence absolute positions: [batch, n]. Indexing the rope tables with
    # this (rather than slicing one shared range) is what lets sequences at
    # different positions share a batch — the requirement continuous batching adds.
    pos = start_positions.unsqueeze(1) + torch.arange(n, device=input_ids.device)
    cos = cos_all[pos]                                  # [b, n, head_dim]
    sin = sin_all[pos]
    lengths = start_positions + n

    for i, s in enumerate(seq_ids):
        cache.ensure_capacity(s, int(lengths[i]))
        cache.lengths[s] = int(lengths[i])

    # Slot indices and the attention mask depend only on positions, so compute
    # them once here and reuse across all 24 layers.
    plan = cache.plan(seq_ids, start_positions, n, lengths)
    L = plan.read.shape[1]
    q_pos = start_positions.view(b, 1, 1, 1) + torch.arange(
        n, device=input_ids.device).view(1, 1, n, 1)
    k_pos = torch.arange(L, device=input_ids.device).view(1, 1, 1, L)
    allowed = (k_pos <= q_pos) & plan.mask.view(b, 1, 1, L)

    x = embed_tokens(input_ids, weights)
    for i in range(cf.num_layers):
        ln = f"model.layers.{i}."
        residual = x
        h = _rms(x, weights[ln + "input_layernorm.weight"], cf.rms_norm_eps)
        x = residual + attention_paged(h, weights, i, cos, sin, cf, cache,
                                       plan, allowed, lengths)
        residual = x
        h = _rms(x, weights[ln + "post_attention_layernorm.weight"],
                 cf.rms_norm_eps)
        x = residual + mlp(h, weights, i, use_kernels=_KERNELS_ENABLED)

    x = _rms(x[:, -1:], weights["model.norm.weight"], cf.rms_norm_eps)
    return F.linear(x, weights["model.embed_tokens.weight"])[:, 0]


@torch.no_grad()
def generate_paged(input_ids: torch.Tensor, weights: dict, cf: "QwenConfig",
                   max_new_tokens: int, cache: "PagedKVCache | None" = None,
                   block_size: int = 16, free_when_done: bool = True):
    """Greedy decode against a paged KV cache. Returns [batch, max_new_tokens]."""
    from .cache import PagedKVCache

    b, seq = input_ids.shape
    dtype = weights["model.norm.weight"].dtype
    total = seq + max_new_tokens

    owned = cache is None
    if owned:
        blocks_per_seq = (total + block_size - 1) // block_size
        cache = PagedKVCache(cf.num_layers, b * blocks_per_seq + 4, block_size,
                             cf.num_kv_heads, cf.head_dim,
                             dtype=dtype, device=input_ids.device)

    seq_ids = [cache.add_sequence() for _ in range(b)]
    rope = build_rope_cache(total, cf.head_dim, cf.rope_theta,
                            device=input_ids.device, dtype=dtype)

    start = torch.zeros(b, dtype=torch.long, device=input_ids.device)
    logits = forward_paged(input_ids, weights, cf, cache, seq_ids, start, rope)
    next_ids = logits.argmax(dim=-1)
    generated = [next_ids]

    for step in range(1, max_new_tokens):
        start = torch.full((b,), seq + step - 1, dtype=torch.long,
                           device=input_ids.device)
        logits = forward_paged(next_ids.unsqueeze(1), weights, cf, cache,
                               seq_ids, start, rope)
        next_ids = logits.argmax(dim=-1)
        generated.append(next_ids)

    out = torch.stack(generated, dim=1)
    if free_when_done:
        for s in seq_ids:
            cache.remove_sequence(s)
    return out
