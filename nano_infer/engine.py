"""Continuous batching — keep the batch full instead of waiting for stragglers.

Static batching takes N requests, runs them together, and does not start the next
group until the LAST one finishes. Real requests do not finish together: if one
generates 200 tokens and seven generate 20, those seven slots sit occupied for
180 steps producing nothing. The batch is held hostage by its slowest member.

Continuous batching evicts a sequence the moment it completes, returns its cache
blocks to the free list, and admits a waiting request into the freed slot on the
next step. The batch stays full, so the per-step weight-streaming cost (measured
flat at ~38 ms regardless of batch size, see SUMMARY.md) is amortized over as
many sequences as possible.

This is where the paged cache earns its keep: sequences of different lengths come
and go independently and their blocks are recycled. A contiguous cache that
reserves max_seq per slot cannot do this.

Both policies live here so the comparison runs on identical code paths and
identical hardware, with only the admission rule changing.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from . import model as M
from .cache import PagedKVCache


@dataclass
class Request:
    """One inference request and its live decoding state."""
    req_id: int
    prompt_ids: torch.Tensor          # [1, prompt_len]
    max_new_tokens: int
    eos_id: int | None = None

    seq_id: int | None = None         # slot in the paged cache
    position: int = 0                 # next cache position to write
    next_token: int | None = None     # token to feed on the coming step
    generated: list[int] = field(default_factory=list)
    finished: bool = False

    admitted_step: int = -1
    finished_step: int = -1

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_ids.shape[1])

    @property
    def complete(self) -> bool:
        return self.finished or len(self.generated) >= self.max_new_tokens


@dataclass
class RunStats:
    policy: str
    wall_s: float
    steps: int
    useful_tokens: int
    slot_steps: int                   # occupied batch slots x decode steps
    max_batch: int
    num_requests: int = 0             # each contributes one token from prefill

    @property
    def tokens_per_sec(self) -> float:
        return self.useful_tokens / self.wall_s if self.wall_s else 0.0

    @property
    def decode_tokens(self) -> int:
        """Wanted tokens produced by DECODE steps, excluding the one token each
        request gets from its prefill. slot_steps counts decode steps only, so
        both sides of the utilization ratio must exclude prefill."""
        return max(self.useful_tokens - self.num_requests, 0)

    @property
    def slot_utilization(self) -> float:
        """Fraction of spent decode capacity that produced a wanted token."""
        return self.decode_tokens / self.slot_steps * 100 if self.slot_steps else 0.0

    @property
    def wasted_slot_steps(self) -> int:
        return self.slot_steps - self.decode_tokens


class ContinuousBatchingEngine:
    """Runs a stream of requests under either admission policy."""

    def __init__(self, weights: dict, cf, num_blocks: int = 512,
                 block_size: int = 16, max_batch: int = 8, max_seq: int = 1024,
                 device=None):
        self.weights = weights
        self.cf = cf
        self.block_size = block_size
        self.max_batch = max_batch
        self.dtype = weights["model.norm.weight"].dtype
        self.device = device or weights["model.norm.weight"].device
        self.num_blocks = num_blocks
        self.rope = M.build_rope_cache(max_seq, cf.head_dim, cf.rope_theta,
                                       device=self.device, dtype=self.dtype)

    # --- primitives ---------------------------------------------------------

    def _new_cache(self) -> PagedKVCache:
        return PagedKVCache(self.cf.num_layers, self.num_blocks, self.block_size,
                            self.cf.num_kv_heads, self.cf.head_dim,
                            dtype=self.dtype, device=self.device)

    def _prefill(self, cache: PagedKVCache, req: Request) -> None:
        """Prefill one request and take its first generated token.

        Prefilling per request (rather than batching prompts) keeps this simple
        and avoids padding ragged prompts. A production engine batches or chunks
        prefills; that is recorded as a limitation rather than hidden.
        """
        req.seq_id = cache.add_sequence()
        start = torch.zeros(1, dtype=torch.long, device=self.device)
        logits = M.forward_paged(req.prompt_ids, self.weights, self.cf, cache,
                                 [req.seq_id], start, self.rope)
        tok = int(logits.argmax(dim=-1))
        req.next_token = tok
        req.generated.append(tok)
        req.position = req.prompt_len
        if req.eos_id is not None and tok == req.eos_id:
            req.finished = True

    def _decode_step(self, cache: PagedKVCache, active: list) -> None:
        """One decode step across every active sequence, each at its own position."""
        tokens = torch.tensor([[r.next_token] for r in active], dtype=torch.long,
                              device=self.device)
        starts = torch.tensor([r.position for r in active], dtype=torch.long,
                              device=self.device)
        logits = M.forward_paged(tokens, self.weights, self.cf, cache,
                                 [r.seq_id for r in active], starts, self.rope)
        nxt = logits.argmax(dim=-1).tolist()
        for r, t in zip(active, nxt):
            r.next_token = int(t)
            r.generated.append(int(t))
            r.position += 1
            if r.eos_id is not None and t == r.eos_id:
                r.finished = True

    # --- policies -----------------------------------------------------------

    @torch.no_grad()
    def run_continuous(self, requests: list) -> RunStats:
        """Admit a waiting request as soon as any slot frees."""
        cache = self._new_cache()
        pending = deque(requests)
        active: list = []
        steps = 0
        slot_steps = 0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while pending or active:
            while len(active) < self.max_batch and pending:
                req = pending.popleft()
                req.admitted_step = steps
                self._prefill(cache, req)
                active.append(req)

            if not active:
                break
            self._decode_step(cache, active)
            steps += 1
            slot_steps += len(active)

            still: list = []
            for r in active:
                if r.complete:
                    r.finished_step = steps
                    cache.remove_sequence(r.seq_id)
                else:
                    still.append(r)
            active = still
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        return RunStats("continuous", wall, steps,
                        sum(len(r.generated) for r in requests),
                        slot_steps, self.max_batch, len(requests))

    @torch.no_grad()
    def run_static(self, requests: list) -> RunStats:
        """Process fixed groups; a group runs until its LAST member finishes.

        Slots whose sequence already completed keep being decoded. That wasted
        work is exactly what continuous batching removes, so it is reproduced
        faithfully here rather than optimized away.
        """
        steps = 0
        slot_steps = 0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(requests), self.max_batch):
            group = requests[i:i + self.max_batch]
            cache = self._new_cache()
            for req in group:
                req.admitted_step = steps
                self._prefill(cache, req)

            longest = max(r.max_new_tokens for r in group)
            for _ in range(longest - 1):
                self._decode_step(cache, group)      # finished rows keep running
                steps += 1
                slot_steps += len(group)

            for req in group:
                req.finished_step = steps
                cache.remove_sequence(req.seq_id)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        # Count only the tokens the caller actually asked for.
        wanted = sum(min(len(r.generated), r.max_new_tokens) for r in requests)
        return RunStats("static", wall, steps, wanted, slot_steps,
                        self.max_batch, len(requests))
