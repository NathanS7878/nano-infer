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


# ===========================================================================
# Paged KV cache
# ===========================================================================

class SlotPlan:
    """Precomputed slot indices for one decode/prefill step, shared by all layers.

    write : [batch*n]        where this step's new K/V go
    read  : [batch, max_len] where each sequence's cached K/V live
    mask  : [batch, max_len] True where the slot holds real data (not padding)
    """
    __slots__ = ("write", "read", "mask")

    def __init__(self, write: torch.Tensor, read: torch.Tensor, mask: torch.Tensor):
        self.write = write
        self.read = read
        self.mask = mask


class BlockAllocator:
    """Free-list allocator over a fixed pool of physical blocks.

    The same idea an operating system uses for memory pages: a sequence does not
    get a contiguous reservation, it gets a list of blocks that may sit anywhere
    in the pool, and blocks return to the free list when the sequence finishes.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise MemoryError(
                f"out of KV blocks: requested {n}, {len(self._free)} free "
                f"of {self.num_blocks}")
        return [self._free.pop() for _ in range(n)]

    def release(self, blocks: list[int]) -> None:
        self._free.extend(blocks)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)


class PagedKVCache:
    """KV cache split into fixed-size blocks drawn from one shared pool.

    Storage is flattened to *slots* — one slot per cacheable token position:

        k[layer] : [num_blocks * block_size, num_kv_heads, head_dim]

    A sequence's logical position p lives at physical slot

        block_table[seq][p // block_size] * block_size + (p % block_size)

    which makes both writing and gathering a single vectorized index operation.

    Why this beats the contiguous cache: there, every sequence reserves max_seq
    slots up front, so batch 32 x max_seq 512 permanently holds 16,384 slots even
    if the sequences are 50 tokens long, and no sequence can use another's spare
    room. Here all sequences draw from one pool, so the pool is sized by total
    expected tokens. The only waste is internal fragmentation in each sequence's
    last, partly-filled block — at most block_size-1 slots per sequence.

    Cost: gathering a sequence's KV means an indexed read that materializes a
    copy each step. A production engine avoids that with an attention kernel that
    walks the block table in-place (Phase 3). In pure PyTorch, expect paging to
    trade a little speed for a lot of memory flexibility.
    """

    def __init__(self, num_layers: int, num_blocks: int, block_size: int,
                 num_kv_heads: int, head_dim: int,
                 dtype: torch.dtype = cfg.DTYPE, device: str = cfg.DEVICE):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device

        slots = num_blocks * block_size
        shape = (num_layers, slots, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

        self.allocator = BlockAllocator(num_blocks)
        self.block_tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}
        self._next_id = 0

        # Cached device-side block table. Rebuilding it from Python lists on every
        # call was measured at 2.26 ms vs 0.07 ms for the actual indexed read —
        # 97% of paging's cost was this, not data movement. It only changes when
        # blocks are allocated or freed, so it is rebuilt on demand, not per layer.
        self._table: torch.Tensor | None = None
        self._table_ids: tuple | None = None
        self._table_dirty = True

    # --- sequence lifecycle -------------------------------------------------

    def add_sequence(self) -> int:
        seq_id = self._next_id
        self._next_id += 1
        self.block_tables[seq_id] = []
        self.lengths[seq_id] = 0
        return seq_id

    def remove_sequence(self, seq_id: int) -> None:
        """Return every block this sequence held to the free list."""
        self.allocator.release(self.block_tables.pop(seq_id))
        self.lengths.pop(seq_id)
        self._table_dirty = True

    def ensure_capacity(self, seq_id: int, total_tokens: int) -> None:
        """Grow this sequence's block table so it can hold `total_tokens`."""
        need = (total_tokens + self.block_size - 1) // self.block_size
        have = len(self.block_tables[seq_id])
        if need > have:
            self.block_tables[seq_id].extend(self.allocator.allocate(need - have))
            self._table_dirty = True

    # --- addressing ---------------------------------------------------------

    def _device_table(self, seq_ids: list[int]) -> torch.Tensor:
        """Device-side [batch, max_blocks] block table, rebuilt only when the
        block assignment actually changed."""
        key = tuple(seq_ids)
        if not self._table_dirty and self._table_ids == key and self._table is not None:
            return self._table

        max_blocks = max(len(self.block_tables[s]) for s in seq_ids)
        rows = [self.block_tables[s] + [0] * (max_blocks - len(self.block_tables[s]))
                for s in seq_ids]
        # Build once on the host, transfer once — not one small copy per sequence.
        self._table = torch.tensor(rows, dtype=torch.long, device=self.device)
        self._table_ids = key
        self._table_dirty = False
        return self._table

    def _slots(self, seq_ids: list[int], positions: torch.Tensor) -> torch.Tensor:
        """Map [batch, n] logical positions to physical slot indices [batch, n]."""
        table = self._device_table(seq_ids)
        logical = positions // self.block_size
        offset = positions % self.block_size
        physical = torch.gather(table, 1, logical)
        return physical * self.block_size + offset

    # --- data ---------------------------------------------------------------

    def plan(self, seq_ids: list[int], start_positions: torch.Tensor, n: int,
             lengths: torch.Tensor) -> "SlotPlan":
        """Compute the write and read slot indices for one step, ONCE.

        These indices are identical for all 24 layers — only the payload differs —
        so computing them per layer repeated ~48 index computations per decode
        step for no reason. Hoisting this out of the layer loop is the second half
        of the paging optimization; see PROGRESS.md for the measured before/after.
        """
        pos = start_positions.unsqueeze(1) + torch.arange(n, device=self.device)
        write = self._slots(seq_ids, pos).reshape(-1)

        max_len = int(lengths.max())
        rpos = torch.arange(max_len, device=self.device).unsqueeze(0).expand(
            len(seq_ids), max_len)
        mask = rpos < lengths.unsqueeze(1)
        read = self._slots(seq_ids, rpos * mask)
        return SlotPlan(write=write, read=read, mask=mask)

    def append(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor,
               plan: "SlotPlan") -> None:
        """Scatter new K/V into the pool. k_new, v_new: [batch, kv_heads, n, head_dim]."""
        kn = k_new.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
        vn = v_new.transpose(1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
        self.k[layer].index_copy_(0, plan.write, kn)
        self.v[layer].index_copy_(0, plan.write, vn)

    def gather(self, layer: int, plan: "SlotPlan"):
        """Read back cached K/V: [batch, num_kv_heads, max_len, head_dim]."""
        k = self.k[layer][plan.read].transpose(1, 2)
        v = self.v[layer][plan.read].transpose(1, 2)
        return k, v

    # --- accounting ---------------------------------------------------------

    def bytes_allocated(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def fragmentation(self) -> dict:
        """Internal fragmentation: slots held vs slots actually holding tokens."""
        held = sum(len(bt) for bt in self.block_tables.values()) * self.block_size
        used = sum(self.lengths.values())
        return {
            "slots_held": held,
            "slots_used": used,
            "wasted_slots": held - used,
            "efficiency_pct": (used / held * 100) if held else 100.0,
            "blocks_used": self.allocator.num_used,
            "blocks_free": self.allocator.num_free,
        }

    def __repr__(self) -> str:
        mb = self.bytes_allocated() / 1024**2
        return (f"PagedKVCache(blocks={self.num_blocks}x{self.block_size}, "
                f"seqs={len(self.block_tables)}, used={self.allocator.num_used}, "
                f"alloc={mb:.1f} MB)")
