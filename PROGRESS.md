# Progress log

A running, dated log of what actually got done in each phase. The git history is
the fine-grained record; this is the human-readable summary.

---

## Phase 0 — Ground truth ✅ COMPLETE (2026-08-20)

Goal: build the ruler (benchmark harness) and the answer key (reference logits)
*before* building the engine, so every later speed claim is measurable and every
optimization is checked against correct output.

- **2026-08-20** — Environment stood up: Miniconda, conda env `nano-infer`
  (Python 3.12, conda-forge), PyTorch 2.6.0+cu124 verified seeing the RTX 3070
  (compute capability sm_86, bf16 supported). HuggingFace stack installed for
  weights/tokenizer download only.
- **2026-08-20** — `HARDWARE.md` recorded: RTX 3070, 8 GB, **448 GB/s** theoretical
  peak memory bandwidth (the denominator for all Phase 3 bandwidth-utilization
  numbers). Repo initialized, scaffolding committed.

- **2026-08-20** — `bench/harness.py` written (CUDA-synchronized timing, warmup,
  TTFT / inter-token / tokens-sec, generic over a generate_fn so the same ruler
  measures our engine later). HF baseline captured to `results/phase0_baseline.*`.

### Acceptance criteria — ALL MET
- [x] Benchmark harness stable across 3 runs (variance 1.0–2.3%, all < 3%).
- [x] Reference logits captured from HuggingFace and saved to disk (1.47 MB).
- [x] HuggingFace `generate()` baseline row in the results table.

### Baseline (RTX 3070, Qwen2.5-0.5B-Instruct, fp16, 128 new tokens)

| Batch | Tokens/sec | TTFT (ms) | Inter-token (ms) |
|---|---|---|---|
| 1  | 19.5  | 63.0 | 51.3 |
| 4  | 76.9  | 59.9 | 51.9 |
| 16 | 313.3 | 67.3 | 50.9 |
| 32 | 621.5 | 68.3 | 51.4 |

**Key observation (the project's thesis, measured):** inter-token latency is flat
(~51 ms) across batch sizes, so tokens/sec scales ~linearly with batch. Decode is
memory-bound — the per-step cost is loading weights from VRAM, paid once per step
regardless of batch — so batch-1 serving badly underuses the GPU. This is the
waste Phases 2–3 attack.

---

## Phase 1 — Correct but slow ✅ COMPLETE (2026-08-20)

Rebuilt Qwen2.5-0.5B's forward pass from scratch in plain PyTorch — no HF model
classes, no generate(). Each component verified against HF before the next was
added (embedding, RMSNorm, RoPE, GQA attention, SwiGLU MLP), then assembled into
the decoder block (pre-norm, two residuals), 24-layer stack, final norm, and tied
logits. Greedy decode with no cache (deliberately slow).

- **2026-08-20** — All five components bit-identical to HF eager (0.00e+00). Full
  forward bit-identical to HF eager and to the fixture. `nano_infer/model.py` +
  `tests/test_model.py`.

### Debugging story (for the WRITEUP)

The acceptance test first failed on two prompts with token divergences. Neither
was a bug in our code:
1. **Prompt 2 diverged at step 0.** Cause: the fixture was captured with HF's
   *default SDPA* (fused) attention; our engine mirrors *eager*. SDPA differs
   from eager by ~0.1 logit over 24 layers — enough to flip a near-tie
   ('Certainly'=23.906 vs '```'=23.875). Our forward was proven bit-identical to
   HF eager on all 5 prompts.
2. **Prompt 4 diverged at step 9.** Cause: the fixture used a *KV cache*; our
   Phase 1 engine recomputes with *no cache*. In fp16, cache vs no-cache round
   differently (~0.04 logit); HF's own eager model diverges from *itself* at the
   same step 9 when run both ways.

Fix (not to the engine — to the reference): regenerate the fixture with the same
procedure the engine uses — **eager attention, no cache**. Lesson: a correctness
answer key must be produced by a deterministic reference implementation matching
the engine under test, not an optimized kernel or a different decode path.

### Acceptance criteria — ALL MET
- [x] Token-for-token identical to HF greedy decode, 5 prompts × 50 tokens.
- [x] Max abs logit diff < 1e-3 in fp16 (actually 0.00e+00 vs HF eager).

### Measured "before" (bench/phase1_nocache.py)

Forward-pass cost vs sequence length — **two regimes**, knee at ~512–1024 tokens:

| seq | 32 | 128 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| one forward | 38.5 ms | 37.0 ms | 39.7 ms | 77.9 ms | 223.4 ms | 692.1 ms |

Flat to 512 (weight-streaming-bound: ~1 GB of fp16 weights dominates), then
O(seq²) attention takes over (2048→4096 = 3.1×, approaching quadratic 4×).
~38 ms to stream 988 MB is ~6% of the 448 GB/s peak — the headroom Phase 3 targets.

Head-to-head vs HF baseline, 128 new tokens:

| Batch | ours (tok/s) | HF (tok/s) | ratio |
|---|---|---|---|
| 1 | 25.4 | 22.5 | **1.13×** |
| 4 | 76.4 | 88.6 | 0.86× |
| 16 | 50.5 | 356.5 | 0.14× |
| 32 | 53.3 | 712.4 | **0.07×** |

**We beat HF at batch 1** (recompute is nearly free in the weight-bound regime;
our loop has less Python overhead than `generate()`), and lose 13× at batch 32
(recompute becomes the bottleneck — our own scaling flattens, 50.5→53.3 from
batch 16→32, as the GPU crosses from memory-bound to compute-bound). This is the
gap Phase 2's KV cache closes. Full analysis in [SUMMARY.md](SUMMARY.md).

---

## Phase 2 — KV cache and batching (in progress)

### Step 1 — Contiguous KV cache + prefill/decode split ✅ (2026-08-20)

`nano_infer/cache.py` (KVCache) + cached paths in `model.py` (`attention_cached`,
`decoder_block_cached`, `forward_cached`, `generate_cached`). Phase 1's code is
left untouched as the reference implementation.

**Correctness.** Proving the cache correct required separating two things:
- *Cache logic*: given the same input, cached hidden states are **bit-identical**
  to Phase 1 on all 5 prompts (`torch.equal`). The cache is exactly right.
- *Token output*: 1 divergence in 250 tokens (0.4%), at prompt 4 step 9, where
  the reference top-1/top-2 gap is **0.0234** — a verified near-tie.

Root cause of that near-tie flip, isolated by experiment: the cache's whole
purpose is to process one token per step instead of the whole sequence, which
changes the shape of every matmul in the decode path. Projecting the *same*
hidden state as `[1,36,896]` vs `[1,1,896]` against the 151936x896 output matrix
differs by **7.81e-03** — cuBLAS picks different kernels for different shapes and
they accumulate in different orders. Invisible on a confident token, decisive on
a coin-flip. **Different matmul shapes are inherent to caching, not a bug.**

**Results** (`bench/phase2_cache.py`, 128 new tokens):

| Batch | Phase 1 no cache | **Phase 2 KV cache** | HF generate() | vs Phase 1 | vs HF |
|---|---|---|---|---|---|
| 1 | 25.4 | **26.7** | 22.4 | 1.05x | 1.19x |
| 4 | 76.4 | **104.5** | 86.8 | 1.37x | 1.20x |
| 16 | 50.5 | **410.1** | 344.7 | 8.12x | 1.19x |
| 32 | 53.3 | **831.6** | 686.9 | **15.60x** | 1.21x |

Prefill vs decode — the two phases have opposite bottlenecks, measured:

| Batch | prefill ms | prefill tok/s | decode ms/step | decode tok/s |
|---|---|---|---|---|
| 1 | 41.7 | 1,006 | 37.3 | 26.8 |
| 4 | 45.2 | 3,717 | 38.4 | 104.1 |
| 16 | 61.0 | 11,013 | 38.3 | 418.3 |
| 32 | 72.7 | 18,485 | 38.7 | 826.8 |

**Decode is flat at ~38 ms/step across a 32x batch range** — memory-bound, the
per-step cost is streaming weights + cache out of VRAM regardless of batch.
**Prefill scales 18x in throughput** for a 1.7x time increase — compute-bound with
headroom. Prefill moves tokens ~22x more efficiently than decode (18,485 vs 827
tok/s at batch 32), which is precisely why skipping a prefill via prefix-cache
routing (MiniDynamo) is worth so much.

**Phase 3 headroom, quantified:** decode at batch 1 takes 37.3 ms to stream ~988 MB
of weights = **26.5 GB/s, or 5.9% of the RTX 3070's 448 GB/s peak**. That gap is
PyTorch's unfused memory round-trips, and it is exactly what the custom kernels target.

### Step 2 — Paged KV cache ✅ (2026-08-20)

`PagedKVCache` + `BlockAllocator` in `cache.py`; `attention_paged` / `forward_paged`
/ `generate_paged` in `model.py`. Storage is flattened to slots
(`[num_blocks*block_size, kv_heads, head_dim]` per layer) so a logical position
maps to a physical slot by one vectorized index:
`block_table[seq][p // block_size] * block_size + (p % block_size)`.

**Correctness:** paged prefill is **bit-identical** (0.00e+00) to the contiguous
cache — paging changes where bytes live, not what is computed. Decode shows the
same single near-tie divergence (1/250) as the contiguous path. Allocator tested
for allocate / exhaust / release / reuse; uneven-length batches verified.

**Memory** (8 sequences, lengths 50–400, block=16, vs contiguous max_seq=512):

| | slots |
|---|---|
| contiguous reserved | 4,096 |
| paged held | 1,072 (67 blocks) |
| actually used | 1,025 |

**3.8x fewer slots held, 4.4% internal fragmentation** — waste is bounded by one
partial block per sequence, versus `max_seq` per sequence.

### Debugging story #2: paging was 4.5x too slow, and the cause was not what I predicted

Prediction before measuring: paging would be somewhat slower than contiguous,
because gathering scattered blocks materializes a copy each step. Measured at
batch 32: **0.22x** — paging cost 78% of throughput. Far worse than "somewhat."

Profiling the two suspects separately (batch 32, length 160, per call):

| | before |
|---|---|
| `_slots` — rebuild block table from Python lists | **2.263 ms** |
| indexed read — the actual gather | 0.071 ms |
| contiguous slice (no copy at all) | 0.018 ms |

**The gather was never the problem.** The data movement cost 0.071 ms against a
plain slice's 0.018 ms — both negligible. 97% of the cost was rebuilding the
block-table tensor from Python lists, which happened on *every layer of every
step* (24 x 128 = 3,072 times per generation, twice each for append and gather).

Two fixes:
1. **Cache the device-side block table**, rebuilding only when blocks are actually
   allocated or freed. `_slots`: 2.263 -> 0.295 ms (**7.7x**).
2. **Hoist slot computation out of the layer loop.** The slot indices and
   attention mask depend only on positions, not layer contents, so they are
   identical for all 24 layers. Computed once per step now (`SlotPlan`).

Result:

| Batch | paged before | paged after | gain | cost vs contiguous |
|---|---|---|---|---|
| 1 | 15.5 | **23.2** | 1.50x | 0.59x -> **0.86x** |
| 4 | 53.1 | **92.2** | 1.74x | 0.51x -> **0.85x** |
| 16 | 135.5 | **369.3** | 2.73x | 0.32x -> **0.89x** |
| 32 | 183.7 | **664.9** | **3.62x** | 0.22x -> **0.90x** |

Paging now costs ~10% instead of 78%, for 3.8x better memory efficiency.
**Lesson: profile before optimizing.** The intuitive culprit (memory traffic from
scattered gathers) was 3% of the cost; the unglamorous one (Python running inside
the per-layer loop) was 97%.

Current standing (128 new tokens; contiguous bs=32 read 736.4 this run vs 828.4
earlier — run-to-run variance, the paged/contiguous *ratio* is the stable metric):

| Batch | Phase 1 no cache | Phase 2 contiguous | Phase 2 paged | HF |
|---|---|---|---|---|
| 1 | 25.4 | **27.0** | 23.2 | 22.7 |
| 4 | 76.4 | **107.8** | 92.2 | 88.4 |
| 16 | 50.5 | **417.2** | 369.3 | 355.3 |
| 32 | 53.3 | **736.4** | 664.9 | 638.3 |

### Step 3 — Continuous batching ✅ (2026-08-20) — PHASE 2 COMPLETE

`nano_infer/engine.py`: `Request`, `RunStats`, `ContinuousBatchingEngine` with
both admission policies on identical code paths, so only the scheduling rule
differs between the two measurements.

**What continuous batching required of the model.** Sequences in a batch now sit
at *different* absolute positions (one admitted 40 steps ago is at position 60
while its neighbour is at 3). Phase 1's `apply_rope` assumes a single shared
position range, so `apply_rope_positions` was added, and `forward_paged` now
indexes the rope tables per sequence instead of slicing one range. The paged
cache already handled per-sequence block tables and the padding mask.

**Why a separate benchmark.** The fixed-batch table cannot show this benefit at
all: it gives every sequence the same output length, so nothing finishes early
and the two policies are identical by construction. The win only exists with
varied output lengths, so `bench/phase2_continuous.py` runs a 24-request stream
with lengths drawn from a skewed distribution (mostly short, a few long).

**Results** (24 requests, lengths 16–126, total 1018 tokens, max_batch 8):

| Policy | Wall time (s) | Decode steps | Tokens | Tokens/sec | Slot utilization | Wasted slot-steps |
|---|---|---|---|---|---|---|
| Static batching | 13.89 | 270 | 1018 | 73.3 | 46.0% | 1166 |
| **Continuous batching** | **8.98** | 163 | 1018 | **113.4** | **100.0%** | **0** |

**1.55x throughput, 35.4% less wall time, identical output.** Static spent 1,166
slot-steps decoding sequences that had already finished — 54% of its capacity —
because a group cannot retire until its longest member does. Continuous batching
wasted none: every decode step produced a token someone asked for.

**Correctness** (tests/test_continuous.py, 4 tests): each request generates
exactly what it generates when run alone, including requests admitted mid-flight
next to sequences at unrelated positions; static and continuous produce identical
tokens; all cache blocks return to the free list once the stream drains.

*Metric note:* slot utilization first read 116.7%, which is impossible. Cause:
`useful_tokens` counted each request's prefill token while `slot_steps` counted
only decode steps. Fixed by excluding prefill from both sides of the ratio.

**Limitation (recorded, not hidden):** prefill is done one request at a time
rather than batched or chunked, to avoid padding ragged prompts. A production
engine batches prefills; at high admission rates that would matter here.

### Phase 2 acceptance — ALL MET
- [x] Paged KV cache with fixed blocks, per-sequence block table, free-block allocator.
- [x] Prefill/decode split, with the opposite bottlenecks measured.
- [x] Continuous batching, measured on a dynamic request stream.
- [x] Output still matches Phase 1 (1 near-tie divergence in 250 tokens, root-caused).
- [x] Benchmark table vs the HF baseline at every batch size — ahead at all of them.

---

## Phase 3 — Custom CUDA kernels (in progress)

### Toolchain (2026-08-20)

`nvcc` 12.4.131 installed via **conda-forge, not the official NVIDIA installer** —
the official one bundles a display driver and would have downgraded this
machine's newer 610.62 driver for no benefit. MSVC 14.44 (VS 2022 Build Tools)
as the host compiler, ninja 1.13 for the build. Three environment quirks, all
handled in `nano_infer/kernels/__init__.py` and documented in HARDWARE.md:

1. ninja lands in the env's `Scripts/` dir, which is not on PATH; torch shells
   out to `ninja` by name.
2. CUDA 12.4 rejects the newer MSVC 14.44 as unsupported —
   `-allow-unsupported-compiler`.
3. conda-forge puts import libraries in `Library/lib`, torch expects
   `$CUDA_HOME/lib/x64` — `LNK1181: cannot open input file 'cudart.lib'` until an
   explicit `/LIBPATH` is added.

The CUDA math libraries are required even though our kernels never call them:
torch's `ATen/cuda/CUDAContextLight.h` includes `cusparse.h`.

### Kernel 1 — Fused RMSNorm ✅

**Prediction first** (per the working style): Nathan predicted memory-bound
before any measurement, correctly. RMSNorm moves ~3,584 bytes per 896-wide row
and does ~3,600 FLOPs — about **1 FLOP/byte against this card's ~364 FLOP/byte
roofline ridge**, so it sits 364x below the ridge and memory is the only thing
that matters.

Design: one thread block per row; strided (coalesced) loads; warp-tree reduction
via `__shfl_down_sync` (threads trade values directly through registers, never
touching memory); cross-warp combine through shared memory; single write.

**Correctness, and a tolerance that was wrong.** The first parity test used a
1e-3 absolute bound and failed at 0.00195. Measuring instead of loosening showed
why: **99.997% of elements were bit-identical**, the rest differed by exactly
**one ULP** — the smallest difference fp16 can represent — and the failing
element had magnitude 2.19, where one ULP *is* 0.00195. The tolerance was wrong,
not the kernel. Parity is now stated in hardware units: max ULP distance <= 2,
>= 99% of elements exact, max relative error <= 4 eps. Stricter and more
meaningful than any absolute number, and it cannot quietly widen.

**Optimization: vectorized loads.** v1 loaded one 2-byte half per thread per
iteration, leaving each thread with only 2 bytes in flight. Switching to `float4`
(16 bytes = 8 halves) raised memory-level parallelism 8x for identical arithmetic:

| Shape | v1 scalar loads | v2 float4 loads |
|---|---|---|
| 4096x896 | 48.8% of peak | **54.3%** |
| 16384x896 | 66.7% of peak (299 GB/s) | **75.5%** (338 GB/s) |

**Final results** (`bench/kernel_rmsnorm.py`, fp16, peak 448 GB/s):

| Shape | PyTorch | Ours | Speedup | GB/s | % of peak | PyTorch % of peak |
|---|---|---|---|---|---|---|
| 1x896 | 197.9 us | 30.3 us | 6.53x | 0.1 | 0.0% | 0.0% |
| 1088x896 | 199.7 us | 31.4 us | 6.36x | 124 | 27.7% | 4.4% |
| 4096x896 | 367.5 us | 60.3 us | 6.09x | 243 | 54.3% | 8.9% |
| 16384x896 | 1329.0 us | 173.6 us | **7.66x** | **338** | **75.5%** | 9.9% |

Scored against the same compulsory-traffic ideal, PyTorch reaches at most 9.9% of
peak — it moves roughly 7x the necessary bytes across five separate kernels.

Two honest caveats:
- **At small shapes the win is launch overhead, not bandwidth.** A 1x896 row is
  3.6 KB — far too little to fill 46 SMs. The 6.5x there comes from making one
  call instead of seven, not from memory efficiency.
- **75.5% is good, not maxed.** The remaining gap is a genuine target, not
  rounding.

### Kernel 2 — fused SwiGLU ✅ (2026-09-01)

`silu(gate) * up`, elementwise over `[rows, 4864]` fp16 tensors.

**The prediction came first, from byte-counting, before any code was written.**
PyTorch runs this as two kernels: `F.silu(gate)` (read gate 2B, write tmp 2B)
then `tmp * up` (read tmp 2B, read up 2B, write out 2B) = **10 B/elem**. The
compulsory minimum is read gate + read up + write out = **6 B/elem**. Ceiling:
**10/6 = 1.67x**. Kernel 1 got 7.66x because it collapsed *five* round trips;
this collapses two. The number is smaller for a reason that is fully known in
advance.

**Measured** (`bench/kernel_swiglu.py`, fp16, peak 448 GB/s, width 4864):

| Shape | PyTorch | Ours | Speedup | GB/s | % of peak | PyTorch @ actual traffic |
|---|---|---|---|---|---|---|
| 1x4864 | 107.0 us | 58.4 us | 1.83x | 0.5 | 0.1% | 0.1% |
| 34x4864 | 105.4 us | 58.5 us | 1.80x | 17 | 3.8% | 3.5% |
| 1088x4864 | 158.0 us | 100.5 us | 1.57x | 316 | 70.6% | 74.8% |
| 4096x4864 | 518.0 us | 312.5 us | 1.66x | 383 | 85.4% | 85.9% |
| 16384x4864 | 1965.6 us | 1192.8 us | **1.65x** | **401** | **89.5%** | 90.5% |

**1.65x measured vs 1.67x predicted — within 1%.** The byte-counting model of the
machine is correct.

**Finding: PyTorch's kernels are not inefficient, they just run twice.** Re-scored
against the 10 B/elem it actually moves, PyTorch reaches **90.5% of peak** — the
same as ours at 89.5%. Both saturate the memory system. The whole 1.65x is one
deleted round trip.

**Finding: this simpler kernel beats kernel 1's bandwidth utilization (89.5% vs
75.5%).** SwiGLU is a barrier-free grid-stride loop; RMSNorm needs a block-wide
reduction with two `__syncthreads()` per row, over only 1792 bytes of work per
block. Synchronization, not arithmetic, is what costs a memory-bound kernel its
last stretch of peak.

**Parity: 0 ULP, 100.000% exact on every shape** — including saturating inputs
(`gate = +/-60000`, where `exp(-z)` overflows to `inf` and the result must reach
zero by division, not `NaN`) and real MLP activations from layers 0, 12, 23.
Elementwise means no reduction, so no reordering: mirroring ATen's cast sequence
(promote to fp32, divide, cast back to half, *then* multiply by `up` in half)
reproduces it bit-for-bit. This also proves the accurate `expf` is in use — a
stray `__expf` or `--use_fast_math` would have surfaced here while still passing
any absolute tolerance loose enough to be called "close enough."

**Honest caveat:** below ~1000 rows the numbers measure dispatch overhead, not
the GPU. Our flat ~58 us at both 1x4864 and 32x4864 is Python + pybind +
`empty_like` cost; 32x more data at identical time is the proof.

**Refactor:** `PYBIND11_MODULE` moved out of `rmsnorm.cu` into a new
`kernels/bindings.cpp`. One `.cu` file could own the module; two cannot. Kernels
3 and 4 now add a declaration and a `def()` line there.

- [x] Kernel 2 — fused SwiGLU
- [ ] Kernel 3 — RoPE
- [ ] Kernel 4 — decode fused attention (online softmax)
- [ ] End-to-end tokens/sec improvement over Phase 2
