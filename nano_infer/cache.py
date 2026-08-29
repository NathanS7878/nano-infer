"""KV cache — the storage layer that makes decode cheap.

In attention, every token produces a Key and a Value. Once computed they never
change: a token's K/V depend only on itself and the tokens before it, and the
past is frozen. So recomputing them every decode step (what Phase 1 does) is pure
waste. The cache computes each token's K/V once and reads them back thereafter.

The measured Phase 1 cost this removes: at batch 32, every decode step reprocessed
~5,120 token-positions instead of 32, making the engine 13x slower than the
HuggingFace baseline (see SUMMARY.md §5).

Note on what is stored: K is cached AFTER RoPE has been applied. RoPE rotates a
key by its position, and that rotation is frozen once applied — exactly the same
"the past does not change" property the cache relies on. V is never rotated.
"""
from __future__ import annotations

import torch

from . import config as cfg


class KVCache:
    """Contiguous per-sequence KV cache, preallocated to max_seq.

    Layout per tensor: [num_layers, batch, num_kv_heads, max_seq, head_dim]

    Preallocating the whole [max_seq] extent is the simple approach, and it is
    deliberately the FIRST version: it wastes memory for short sequences (a
    sequence using 50 of 512 slots still holds all 512) and cannot share pages
    between sequences. The paged cache replaces exactly that weakness.

    Only num_kv_heads (2) are stored, not num_q_heads (14) — GQA's payoff. At
    batch 32 / max_seq 512 this cache is 24*32*2*512*64*2 bytes * 2 (K and V)
    = 804 MB; under full multi-head attention it would be 7x that, 5.6 GB, which
    does not fit in the RTX 3070's 8 GB alongside the weights.
    """

    def __init__(self, num_layers: int, batch: int, num_kv_heads: int,
                 max_seq: int, head_dim: int,
                 dtype: torch.dtype = cfg.DTYPE, device: str = cfg.DEVICE):
        shape = (num_layers, batch, num_kv_heads, max_seq, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_seq = max_seq
        self.batch = batch
        self.length = 0          # positions currently filled (same for all rows)

    def append(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor,
               start: int) -> None:
        """Write new K/V at positions [start, start + n) for one layer.

        k_new, v_new : [batch, num_kv_heads, n, head_dim]
        """
        n = k_new.shape[2]
        if start + n > self.max_seq:
            raise ValueError(
                f"KV cache overflow: writing {n} at {start} exceeds max_seq={self.max_seq}")
        self.k[layer, :, :, start:start + n] = k_new
        self.v[layer, :, :, start:start + n] = v_new

    def view(self, layer: int, length: int):
        """Read back the filled prefix for one layer: [batch, kv_heads, length, head_dim]."""
        return self.k[layer, :, :, :length], self.v[layer, :, :, :length]

    def bytes_allocated(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def bytes_used(self, length: int) -> int:
        """How much of the allocation actually holds live data."""
        per_pos = self.k[0, :, :, 0].numel() * self.k.element_size()
        return per_pos * length * self.k.shape[0] * 2

    def __repr__(self) -> str:
        mb = self.bytes_allocated() / 1024**2
        return (f"KVCache(batch={self.batch}, max_seq={self.max_seq}, "
                f"len={self.length}, alloc={mb:.1f} MB)")
