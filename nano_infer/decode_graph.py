"""Static, synchronisation-free decode — and CUDA graphs on top of it.

WHY THIS FILE EXISTS
--------------------
Phase 3 made decode attention 2.6x faster and end-to-end throughput did not
move (ROADMAP Gotcha #29). The decode step cost ~20 ms whether the batch was 1
or 32 and whether the context was 33 or 1025 tokens — invariances no GPU-bound
loop can show. Two host-side costs were measured inside `forward_paged`:

  1. ~3,200 aten dispatches per step, only 169 of them matmuls. The rest are
     views, transposes and reshapes that launch no GPU work but still pay the
     full Python -> dispatcher -> kernel-selection path on the host.
  2. 65 GPU->host SYNCHRONISATIONS per step at batch 32 (3 at batch 1): two
     `int(lengths[i])` per sequence to drive the block allocator, plus
     `int(lengths.max())` to size the read table. A sync stops the host from
     queuing ahead, so it serialises the loop on top of costing a round trip.

Both exist because the step recomputes, on every call, bookkeeping whose answer
was already known before decode began.

THE OBSERVATION THAT REMOVES BOTH
---------------------------------
In `generate_paged` every sequence's final length is known up front: prompt
length + max_new_tokens. So:

  * allocate every KV block BEFORE decode starts — then each sequence's slot
    table for positions [0, final_len) is fixed for the whole generation, and
    the read table can be built once as a static [batch, final_len] tensor;
  * derive everything else ON THE GPU inside the step:
        lengths = pos + 1
        write   = read.gather(1, pos)          (this step's slot)
        mask    = arange(final_len) < lengths  (what attention may see)
    None of these needs the host to look at a value;
  * let the step feed itself: it writes its argmax into its own input buffer,
    records it into a preallocated output buffer, and advances `pos` in place.

After that, a decode step has no host inputs at all — no H2D copy, no sync, no
allocator call. That is the precondition for a CUDA graph, which replays a
recorded sequence of kernel launches without re-entering Python or the
dispatcher: ~3,200 dispatches become one replay.

TWO MODES, MEASURED SEPARATELY
------------------------------
`use_graph=False` runs the static, sync-free step eagerly. `use_graph=True`
captures that same step as a graph. Keeping both is the point: the eager-static
mode isolates the win from deleting syncs and bookkeeping, and the graph mode
adds the win from deleting dispatch. Reporting one number for both would hide
which one mattered — the same discipline as the kernel benchmarks.

The step reuses `model.attention_paged` unchanged, so the graphed step runs the
same attention code (kernel or PyTorch) as the eager `forward_paged`. The only
structural difference is that the read table is `final_len` wide from the first
decode step, instead of growing to the current maximum length.

WHAT THIS DOES NOT COVER, stated up front
-----------------------------------------
  * Static batches only. `engine.py`'s continuous batching changes batch
    composition every time a request joins or leaves, so it would need one graph
    per batch size with padding — what production engines do, not done here.
  * Capture is not free. It runs warmup iterations plus one recorded step, and
    it is paid per `generate_paged_static` call here because the graph binds
    this call's cache and slot table. Benchmarks report it separately rather
    than hiding it inside a steady-state number.
  * Greedy, fixed length. No EOS early-exit: stopping on a token would need the
    host to read a value, which is exactly the synchronisation removed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import model as M
from .cache import PagedKVCache, SlotPlan


class _StaticDecodeState:
    """Every tensor the decode step reads or writes, allocated exactly once.

    A CUDA graph records memory ADDRESSES, not values, so every input must live
    at a fixed address for the lifetime of the graph and be updated in place.
    """

    def __init__(self, cache: PagedKVCache, seq_ids: list, first_ids: torch.Tensor,
                 prompt_len: int, max_new_tokens: int, final_len: int):
        dev = first_ids.device
        b = first_ids.shape[0]

        # the slot every position [0, final_len) of every sequence will occupy.
        # Built once: all blocks were allocated before this, so it never changes.
        rpos = torch.arange(final_len, device=dev).unsqueeze(0).expand(b, final_len)
        self.read = cache._slots(seq_ids, rpos).contiguous()       # [b, final_len]
        self.k_pos = torch.arange(final_len, device=dev)           # [final_len]

        self.ids = first_ids.clone()                               # [b] input token
        self.pos = torch.full((b,), prompt_len, dtype=torch.long, device=dev)
        self.step = torch.ones(1, dtype=torch.long, device=dev)    # output column
        self.out = torch.zeros(b, max_new_tokens, dtype=torch.long, device=dev)
        self.out[:, 0] = first_ids
        self.record: list | None = None      # see _decode_step; eager only

    def reset(self, first_ids: torch.Tensor, prompt_len: int) -> None:
        """Re-arm for a new prompt IN PLACE. The graph recorded these buffers'
        addresses, so reallocating any of them would leave the graph writing to
        memory this object no longer reads."""
        self.ids.copy_(first_ids)
        self.pos.fill_(prompt_len)
        self.step.fill_(1)
        self.out.zero_()
        self.out[:, 0] = first_ids

    def snapshot(self):
        return self.ids.clone(), self.pos.clone(), self.step.clone(), self.out.clone()

    def restore(self, snap):
        ids, pos, step, out = snap
        self.ids.copy_(ids)
        self.pos.copy_(pos)
        self.step.copy_(step)
        self.out.copy_(out)


def _decode_step(st: _StaticDecodeState, weights: dict, cf: "M.QwenConfig",
                 cache: PagedKVCache, cos_all: torch.Tensor,
                 sin_all: torch.Tensor) -> None:
    """One greedy decode step over static buffers. No host reads, no allocator.

    Mirrors the loop body of `model.forward_paged` for n == 1, with the
    bookkeeping that forward_paged does on the host re-derived on the device.
    """
    b = st.ids.shape[0]
    L = st.read.shape[1]
    pos = st.pos

    lengths = pos + 1                                              # [b]
    write = st.read.gather(1, pos.unsqueeze(1)).reshape(-1)        # [b]
    mask = st.k_pos.unsqueeze(0) < lengths.unsqueeze(1)            # [b, L]
    plan = SlotPlan(write=write, read=st.read, mask=mask)
    allowed = ((st.k_pos.view(1, 1, 1, L) <= pos.view(b, 1, 1, 1))
               & mask.view(b, 1, 1, L))

    p1 = pos.unsqueeze(1)                                          # [b, 1]
    cos = cos_all[p1]                                              # [b, 1, hd]
    sin = sin_all[p1]

    x = M.embed_tokens(st.ids.unsqueeze(1), weights)               # [b, 1, H]
    for i in range(cf.num_layers):
        ln = f"model.layers.{i}."
        residual = x
        h = M._rms(x, weights[ln + "input_layernorm.weight"], cf.rms_norm_eps)
        x = residual + M.attention_paged(h, weights, i, cos, sin, cf, cache,
                                         plan, allowed, lengths)
        residual = x
        h = M._rms(x, weights[ln + "post_attention_layernorm.weight"],
                   cf.rms_norm_eps)
        x = residual + M.mlp(h, weights, i, use_kernels=M.kernels_enabled())

    x = M._rms(x, weights["model.norm.weight"], cf.rms_norm_eps)
    logits = F.linear(x, weights["model.embed_tokens.weight"])[:, 0]
    nxt = logits.argmax(dim=-1)
    # Test observability only: eager runs can record each step's logits so a
    # divergence can be attributed (tests/_drift.py). A host-side list append
    # would record nothing under graph replay, so capture refuses to run with it.
    if st.record is not None:
        st.record.append(logits.detach().float().clone())

    # the step feeds itself: every write below is in place at a fixed address
    st.out.index_copy_(1, st.step, nxt.unsqueeze(1))
    st.ids.copy_(nxt)
    st.pos.add_(1)
    st.step.add_(1)


class DecodeGraphRunner:
    """Owns everything one decode graph binds to, so capture is paid ONCE.

    Capturing per call cost 0.12-0.25 s -- about 28% of a 64-token batch-32 call
    (bench/decode_graph.py). None of it depends on the prompt: the graph binds
    the KV pool, the read table, the rope tables and the state buffers, all of
    which are determined by (batch, prompt_len, max_new_tokens). So a runner
    captures on its first `generate` and every later call of the same shape runs
    prefill + replay only. Production engines do the same at startup, one graph
    per batch-size bucket; here the bucket is exact, with no padding, which is a
    limitation rather than a design choice.

    WHY REUSING THE KV POOL WITHOUT ZEROING IT IS SAFE. A new prompt's prefill
    overwrites positions [0, prompt_len). Every decode step writes its own slot
    before that slot can be read -- the kernel's loop is bounded by `lengths`
    and the PyTorch path masks by it. So no stale value from a previous prompt
    is ever read. tests/test_decode_graph.py checks this with DIFFERENT prompts
    on one runner against fresh one-shot runs, because a leak would otherwise
    produce plausible text.

    The graph also bakes in which attention/MLP path was active when it was
    captured (kernels or PyTorch), and the weight tensors' addresses. Calling
    with the kernel setting flipped raises; replacing tensors inside `weights`
    after capture is unsupported and would be silently ignored.
    """

    def __init__(self, weights: dict, cf: "M.QwenConfig", batch: int,
                 prompt_len: int, max_new_tokens: int, use_graph: bool = True,
                 block_size: int = 16, warmup: int = 3, device=None):
        dev = (torch.device(device) if device is not None
               else weights["model.norm.weight"].device)
        dtype = weights["model.norm.weight"].dtype
        self.weights, self.cf = weights, cf
        self.batch, self.prompt_len, self.max_new = batch, prompt_len, max_new_tokens
        self.use_graph, self.warmup = use_graph, warmup
        self.final_len = prompt_len + max_new_tokens

        blocks_per_seq = (self.final_len + block_size - 1) // block_size
        self.cache = PagedKVCache(cf.num_layers, batch * blocks_per_seq + 4,
                                  block_size, cf.num_kv_heads, cf.head_dim,
                                  dtype=dtype, device=dev)
        self.seq_ids = [self.cache.add_sequence() for _ in range(batch)]
        # THE key move: allocate the whole generation's blocks now, so no decode
        # step ever has to ask the host how long a sequence has become.
        for s in self.seq_ids:
            self.cache.ensure_capacity(s, self.final_len)
        self.cos_all, self.sin_all = M.build_rope_cache(
            self.final_len, cf.head_dim, cf.rope_theta, device=dev, dtype=dtype)

        self.state = None
        self.graph = None
        self.captured_with_kernels = None
        self.captures = 0

    def _step(self):
        _decode_step(self.state, self.weights, self.cf, self.cache,
                     self.cos_all, self.sin_all)

    def _capture(self):
        if self.state.record is not None:
            raise RuntimeError("logit recording is eager-only: a graph replays "
                               "kernels, not the host-side list append")
        # Warm up on a side stream (cuBLAS handles, kernel JIT, allocator), then
        # capture. Both run the step for real, advancing ids/pos/step and writing
        # KV at the current slot, so state is restored after each. The KV they
        # wrote sits at the slot the first real step overwrites before reading.
        #
        # The synchronize() calls are load-bearing (ROADMAP #35). Warmup runs on
        # a SIDE stream, but the prefill and the snapshot clones it restores from
        # were queued on the DEFAULT stream. An earlier version called
        # side.synchronize() -- which waits on the wrong stream -- and it passed
        # at batch 1 purely on timing, then gathered out of bounds at batch 4.
        st = self.state
        snap = st.snapshot()
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self._step()
                st.restore(snap)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        torch.cuda.synchronize()
        st.restore(snap)
        self.captured_with_kernels = M.kernels_enabled()
        self.captures += 1

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, timings: dict | None = None):
        """Greedy-decode `input_ids` ([batch, prompt_len]). Returns a NEW
        [batch, max_new_tokens] tensor -- a clone, because the runner's output
        buffer is overwritten by the next call."""
        import time

        def _mark():
            torch.cuda.synchronize()
            return time.perf_counter()

        if tuple(input_ids.shape) != (self.batch, self.prompt_len):
            raise ValueError(
                f"runner was built for [{self.batch}, {self.prompt_len}], got "
                f"{list(input_ids.shape)}; a graph's shapes are fixed at capture")
        if self.graph is not None and M.kernels_enabled() != self.captured_with_kernels:
            raise RuntimeError(
                "graph was captured with kernels "
                f"{'on' if self.captured_with_kernels else 'off'} and would replay "
                "that path regardless; build a separate runner for the other")

        t0 = _mark() if timings is not None else None
        # prefill stays eager: its shape depends on the prompt, it runs once per
        # call, and it is compute-bound rather than dispatch-bound
        start = torch.zeros(self.batch, dtype=torch.long, device=input_ids.device)
        logits = M.forward_paged(input_ids, self.weights, self.cf, self.cache,
                                 self.seq_ids, start, (self.cos_all, self.sin_all))
        first = logits.argmax(dim=-1)
        t1 = None
        if timings is not None:
            t1 = _mark()
            timings["prefill_s"] = t1 - t0

        if self.max_new <= 1:
            if timings is not None:
                timings.update(capture_s=0.0, decode_s=0.0, decode_steps=0)
            return first.unsqueeze(1)

        if self.state is None:
            self.state = _StaticDecodeState(self.cache, self.seq_ids, first,
                                            self.prompt_len, self.max_new,
                                            self.final_len)
        else:
            self.state.reset(first, self.prompt_len)

        steps = self.max_new - 1
        capture_s = 0.0
        if self.use_graph:
            if self.graph is None:
                self._capture()
                if timings is not None:
                    t2 = _mark()
                    capture_s, t1 = t2 - t1, t2
            for _ in range(steps):
                self.graph.replay()
        else:
            for _ in range(steps):
                self._step()

        out = self.state.out.clone()
        if timings is not None:
            timings["capture_s"] = capture_s
            timings["decode_s"] = _mark() - t1
            timings["decode_steps"] = steps
        return out


@torch.no_grad()
def generate_paged_static(input_ids: torch.Tensor, weights: dict,
                          cf: "M.QwenConfig", max_new_tokens: int,
                          use_graph: bool = True, block_size: int = 16,
                          warmup: int = 3, timings: dict | None = None):
    """One-shot greedy decode with a static, sync-free decode step.

    Same contract as `model.generate_paged`: returns [batch, max_new_tokens].

    use_graph=False  run the static step eagerly (isolates the sync/bookkeeping win)
    use_graph=True   capture it as a CUDA graph and replay it

    Builds a fresh DecodeGraphRunner every call, so it pays capture every call.
    To serve repeated prompts of one shape, keep a DecodeGraphRunner instead.

    If `timings` is a dict, it is filled with host-side wall times (seconds,
    synchronised at phase boundaries only) for prefill, capture and decode, so a
    benchmark can report capture cost instead of burying it.
    """
    b, seq = input_ids.shape
    runner = DecodeGraphRunner(weights, cf, b, seq, max_new_tokens,
                               use_graph=use_graph, block_size=block_size,
                               warmup=warmup, device=input_ids.device)
    return runner.generate(input_ids, timings=timings)
