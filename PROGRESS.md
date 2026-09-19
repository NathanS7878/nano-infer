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

### Kernel 3 — fused RoPE (2026-09-01)

`x_rot = x * cos + rotate_half(x) * sin`, where `rotate_half([x1,x2]) = [-x2,x1]`
pairs dim i with i + head_dim/2.

**Byte count first, again.** PyTorch runs five kernels; per element of x (E
elements at 2 bytes, cos/sin small enough to stay in L2): `-x2` = 2E, the `cat`
= 4E, `x*cos` = 4E, `rotated*sin` = 4E, the add = 6E. **Total 20E.** Ours reads x
once and writes once: **4E**. Predicted ceiling **5.00x**.

Three times SwiGLU's ceiling, and the reason is the lesson: **6E of PyTorch's 20E
go to `rotate_half`, an op that computes nothing.** It is pure plumbing — it
exists only to get the operand into a layout the next elementwise kernel can
consume, and a fused kernel does that with an index offset. The most profitable
thing to fuse is usually not the expensive math, it is the data movement wrapped
around it.

**Measured** (`bench/kernel_rope.py`, fp16, peak 448 GB/s):

| Shape | Elements | PyTorch | Ours | Speedup | GB/s | % of peak |
|---|---|---|---|---|---|---|
| 32x14x1x64 (decode b32, q) | 28,672 | 285.7 us | 59.7 us | 4.78x | 1.9 | 0.4% |
| 32x14x34x64 (prefill b32) | 974,848 | 148.3 us | 30.1 us | 4.94x | 130 | 29.0% |
| 32x14x512x64 | 14,680,064 | 835.8 us | 166.1 us | 5.03x | 354 | 78.9% |
| 32x14x2048x64 | 58,720,256 | 3231.5 us | 598.3 us | **5.40x** | **393** | **87.6%** |

**5.03x at the first genuinely bandwidth-bound size vs a 5.00x prediction.**

**The 5.40x beat the ceiling, so it got checked instead of celebrated.** Scoring
each side against the traffic it actually moves:

| Shape | ours % peak | PyTorch % peak | efficiency ratio | speedup |
|---|---|---|---|---|
| 32x14x34x64 | 29.0% | 29.3% | 0.99 | 4.94x |
| 32x14x512x64 | 78.9% | 78.4% | 1.01 | 5.03x |
| 32x14x2048x64 | 87.6% | 81.1% | **1.08** | 5.40x |

At every size but the largest both implementations are equally efficient per byte
and the speedup is the traffic ratio exactly. At 117 MB tensors PyTorch's chain
falls to 81.1% of peak while ours holds 87.6%: **5.00 x 1.08 = 5.40**. That is
not our kernel beating physics, it is PyTorch's chain doing worse than its own
earlier self — plausibly broadcast index arithmetic plus allocator pressure from
five 117 MB temporaries (effect measured, cause not proven). **Byte counting
predicts the floor of a fusion win, not a bound on it.**

**At the sizes the engine actually runs, the win is not bandwidth.** q is
[b,14,n,64] and at decode n=1, so batch 32 is 28,672 elements = **57 KB**, far too
little to fill 46 SMs. The 4.78x there is five launches becoming one, and the
0.4%-of-peak column says so. Both regimes are in the table deliberately.

**Correctness: the failure mode that does not announce itself.** RoPE can be
wrong in a way that still runs. HF/Llama pairs dim i with i+head_dim/2; the
original paper's diagram pairs adjacent dims. Both give finite, plausible
tensors, and a model built on the wrong one emits fluent text while being quietly
wrong about position. Nothing raises. So the pairing is asserted directly: with
cos=0, sin=1 the transform collapses to exactly `rotate_half`, which separates
the conventions unambiguously, and the test also asserts the adjacent-pair answer
is NOT produced. A second, reference-independent property test checks that RoPE
preserves each head vector's norm (it is a rotation) — that would catch a kernel
matching PyTorch because both were wrong. Max relative norm drift 1.87e-04.

**Parity: 0 ULP, 100.000% exact** on all 7 shapes across BOTH position paths —
shared positions (Phase 1) and per-sequence positions (Phase 2 step 3). A kernel
indexing cos/sin by row rather than by (batch, position) passes the first and
fails the second, so both are tested. Also verified non-contiguous (attention()
makes q/k by transposing a view, so that is the normal case) and on real q/k from
layers 0, 12, 23.

### Kernel 4 — fused decode attention with online softmax (2026-09-02)

Kernels 1–3 were fusions: same arithmetic, fewer round trips, ceiling predictable
by counting bytes, and all three landed within 1% of that prediction. **This one
changes the algorithm, and it is the one that missed its prediction badly — which
made it the most informative kernel in the phase.**

The reference materializes a `[batch, heads, 1, L]` score matrix, softmaxes it,
then multiplies by V. Online softmax never materializes it, streaming keys in
tiles while carrying three running values per (sequence, query head):

```
m   running max of scores       l   running sum of exp(s-m)     acc  running sum of exp(s-m)*v

per tile:  m_new = max(m, max(s_tile));   corr = exp(m - m_new)
           l   = l   * corr + sum(exp(s_tile - m_new))
           acc = acc * corr + sum_j exp(s_j - m_new) * v_j
out = acc / l
```

`corr` is the whole trick: a softmax needs the max of the entire row, a streaming
kernel does not have it yet, so when a later tile raises the max everything
already accumulated is retroactively rescaled. **The recurrence was prototyped in
Python before any CUDA was written** — which also measured what dropping `corr`
costs (2.881 absolute error on a late max jump), giving the regression test a
signature to look for rather than a guess.

#### The prediction, and the miss

Byte count per sequence per layer, L cached positions, GQA 14 q / 2 kv heads:

| | traffic |
|---|---|
| `cache.gather` | 1024L |
| `repeat_kv` — materializes a 7× copy of K and V | **4096L** |
| `q @ kᵀ` | 3612L |
| float / mask / softmax / cast round trips | 392L |
| `probs @ v` | 3612L |
| **PyTorch total** | **~12736L** |
| **Ours** — read K once, read V once | **512L** |

Predicted ceiling ~24×. **Measured 2.48×.** Every previous kernel landed within
1% of its byte count; this one came in at a tenth of it. Byte counting predicts
the ceiling only when the kernel is actually bandwidth-bound — and this one is not.

#### Profiling the miss: it is latency-bound, not bandwidth-bound

The obvious suspect was duplicated reads: one block per (sequence, query head)
means the 7 query heads sharing a KV head each stream that KV independently. The
decisive experiment holds the KV pool fixed (batch 32, L 1024, 2 KV heads) and
varies only how many query heads share it, so compulsory traffic is constant:

| query heads | blocks | time | µs per query head |
|---|---|---|---|
| 2 | 64 | 281.8 µs | 140.9 |
| 4 | 128 | 282.4 µs | 70.6 |
| 8 | 256 | 292.0 µs | 36.5 |
| 14 | 448 | 510.2 µs | 36.4 |

**From 2 to 8 query heads the work quadruples and the wall time does not move.**
That is not bandwidth saturation and it is not the duplicated reads — it is the
signature of a dependency chain with nothing to overlap it. The online-softmax
recurrence is inherently sequential across tiles, each tile costs several
`__syncthreads()`, and at 128 threads per block there were only 4 warps per block
to hide the memory latency between barriers. Kernel 2's lesson (barriers, not
arithmetic, cost a memory-bound kernel its peak) returning with real teeth.

#### The fix that followed from the diagnosis

If latency is the problem, the lever is warps per SM and tiles per sequence — so
block size, which is also the tile width. Sweeping it:

| case | 64 | 128 | 256 | 512 | 1024 | best |
|---|---|---|---|---|---|---|
| b32 L512 | 264.0 | 268.3 | 217.1 | **199.1** | 232.9 | 512 |
| b32 L2048 | 970.8 | 936.6 | 749.5 | **668.7** | 749.0 | 512 |
| b1 L1024 | 278.0 | 185.7 | 129.6 | 93.1 | **51.6** | 1024 |

Batch 1 wants the widest block available and gains **3.6×** from it, for the
reason the diagnosis predicts: with 14 blocks total there is nothing else on the
SM, so all the latency hiding has to come from inside the block. The shipped
heuristic picks 512 when there are many blocks and 1024 when there are few, and
reproduces the measured optimum on every case in the sweep. Overall: **7.9% →
11.1% of peak, 2.25× → 2.48×.**

#### Final results, with the two wins separated

The kernel does two independent things, so reporting one number would hide which
mattered. Three implementations are timed — the Phase 2 paged path, the same
path handed pre-gathered KV, and ours:

| Shape | Phase 2 paged | Pre-gathered | Ours | vs paged | vs pre-gathered | GB/s | % of peak |
|---|---|---|---|---|---|---|---|
| b1 L128 | 375.9 µs | 306.8 µs | 32.6 µs | **11.53×** | 9.41× | 2.0 | 0.4% |
| b32 L128 | 379.4 µs | 321.7 µs | 62.2 µs | 6.10× | 5.17× | 33.7 | 7.5% |
| b32 L512 | 513.2 µs | 418.1 µs | 200.5 µs | 2.56× | 2.09× | 41.8 | 9.3% |
| b32 L1024 | 901.5 µs | 745.5 µs | 396.2 µs | 2.28× | 1.88× | 42.3 | 9.5% |
| b32 L2048 | 1755.3 µs | 1364.6 µs | 677.0 µs | **2.59×** | 2.02× | **49.6** | **11.1%** |

At batch 32 with context ≥ 512: **1.99× from the fusion** (online softmax plus
never materializing `repeat_kv`'s 7× copy) and **1.24× from reading the paged
block table in place** (removing the gather copy Phase 2 pays every step).

#### What is still wrong, stated plainly

**11.1% of peak is not a good number.** Phase 2's decode was at 5.9%, so this
roughly doubles it, but the kernel is still latency-bound rather than
bandwidth-bound and the diagnosis above says exactly why. Two optimizations are
identified and not done:

1. **Split-K (flash-decoding proper).** Partition L across several blocks, each
   producing a partial `(m, l, acc)`, then combine. That adds the parallelism the
   scaling experiment showed is missing, and it is what real flash-decoding does
   for exactly this case — long context, few sequences.
2. **Collapse the 7 query heads sharing a KV head into one block.** They stream
   the same KV; one block could read it once and serve all seven.

The GB/s column also counts each KV element once, though the kernel issues up to
7 reads of it. It therefore measures *useful* bandwidth, not bus traffic, and is
reported that way rather than as an efficiency claim.

#### Correctness: the first kernel that cannot be bit-identical

The reference rounds the QK product to fp16, softmaxes in fp32, then rounds the
probabilities back to fp16 before accumulating. An online algorithm cannot do
that last step at all — it does not know the normaliser until every key has been
seen. So bit-identity is structurally impossible, and the ULP bar that governed
kernels 1–3 had to be replaced rather than loosened:

> **Against an fp64 ground truth, our error must be no larger than the fp16
> reference's.**

That is a harder test than "close to PyTorch", because it cannot be satisfied by
being wrong in the same direction as the reference. Measured across all shapes,
our error is **0.48–1.00×** the reference's — as accurate or better everywhere.

**Third metric lesson.** The ULP counts against the reference look alarming (up
to 2420) while the absolute difference is a flat 1.95e-03 — one fp16 ULP at
magnitude 2. Attention output components pass through zero, and near zero fp16
resolution becomes enormously fine, so ULP distance explodes where the absolute
error is negligible. ULP was exactly the right metric for kernel 1 and is the
wrong one here; absolute error was wrong for kernel 1 and is right here. **Neither
metric is universal — the right one depends on whether the value distribution
spans zero.**

#### The debugging story: the oracle was the broken thing

The ground-truth test was first written the obvious way — upcast the GPU tensors
to float64 and reuse the reference. It reported our kernel and PyTorch as *equally
wrong* by 0.18, on outputs of magnitude ~0.4. Two independent implementations
agreeing closely with each other and not with the oracle indicts the oracle, so
the oracle got tested:

**`torch.softmax` in float64 on CUDA returns incorrect results on this machine
for any tensor with more than one row.** Measured on torch 2.6.0+cu124 / RTX
3070: at `[16, 513]` the elementwise error against CPU is 1.2e-02 and rows sum to
**0.68 instead of 1.0**, while `[1, 513]` is exact to 1.7e-18. fp32 and fp16 are
unaffected, and a uniform input is unaffected at any size.

The ground truth now runs on the CPU, and `test_fp64_softmax_on_cuda_is_unreliable`
pins the bug so nobody "simplifies" it back onto the GPU — it fails if torch ever
fixes it. The reusable lesson: **when a new implementation disagrees with a
trusted reference, the possible culprits include the instrument you are measuring
with.** Two implementations agreeing with each other and not with the oracle is
the tell.

- [x] Kernel 2 — fused SwiGLU
- [x] Kernel 3 — RoPE
### Phase 3 wrap-up — wiring and end-to-end (2026-09-02)

Four correct kernels are not the acceptance criterion; end-to-end tokens/sec is.

**The wiring.** A module-level flag (`model.set_kernels` / `model.using_kernels`)
rather than a parameter threaded through eight functions. The layering rule holds:
**the Phase 1 functions are the answer key and do not consult the flag**, so
`rms_norm`, `apply_rope`, `attention` and `forward` stay on the reference path and
a pure-PyTorch comparison remains available in the same process. `mlp` is shared
between Phase 1 and Phase 2, so it takes an explicit `use_kernels` keyword
defaulting to `False` — Phase 1's call site is unchanged by construction, and a
test asserts both of these rather than trusting them.

#### Correctness: what actually changes when the kernels go in

| check | result |
|---|---|
| batch 1, 40 tokens | **0 / 40 differ** |
| batch 4, 160 tokens | 12 / 160 differ; first divergence top-1/top-2 gap **0.0000** |
| kernel 4 routing | **0 calls during prefill, 24 during decode** (one per layer) |
| prefill logit drift | 4.10e-02 with all kernels, **0.00e+00** with RMSNorm alone reverted |

Three things worth pulling out.

**A divergence count is a cascade, not a tally.** 12 differing tokens out of 160
is one sequence flipping at step 28 and every later token in that sequence
following. The first divergence had a top-1/top-2 gap of **exactly 0.0000** — a
true tie, the strongest possible form of the near-tie standard from Gotcha #2.

**Only two of the four kernels are bit-identical, and the first draft of the test
assumed three were.** SwiGLU and RoPE are 0 ULP / 100% exact — elementwise, no
reduction, no reordering. **RMSNorm is not, and cannot be**: it sums 896 squares
through a warp-tree reduction while PyTorch uses its own order, and floating-point
addition is not associative. Its bar has always been ≤ 2 ULP with ≥ 99% exact
(Gotcha #3); what was new here is watching ~1 ULP per layer compound over 24
layers into a 4.10e-02 logit shift — the same scale Gotcha #1 documents for
eager-vs-SDPA.

**That drift was attributed, not assumed.** Holding the kernel flag on while
routing only RMSNorm back to the reference drops the difference to **exactly
zero**. That single experiment proves three things at once: RMSNorm is the whole
source, SwiGLU and RoPE contribute nothing, and prefill never reaches the decode
kernel. Kernel 4's routing is *also* asserted structurally by counting calls
rather than inferred from numerics — conflating a routing bug with rounding is
exactly how a real bug hides.

#### End-to-end: measured, but not yet a headline number

**The honest expectation, written down before running it:** kernels 1–3 are
launch-bound at decode sizes (RoPE's q at batch 32 is 57 KB; RMSNorm's rows are
896 wide), so their 7.66× / 1.65× / 5.03× microbenchmark ratios cannot transfer.
Kernel 4 is the only one touching a big tensor and it runs at 11.1% of peak.

Three runs of the same A/B, same process, kernels off vs on:

| batch | run 1 | run 2 | run 3 |
|---|---|---|---|
| 1 | 3.29× | 2.88× | 2.48× |
| 4 | 2.60× | 1.85× | 2.69× |
| 16 | 2.45× | 2.83× | 2.93× |
| 32 | 2.06× | 1.96× | 2.31× |

**Consistently 1.85–3.29×, never below 1.8×** — better than expected, and the
direction is robust. But **the precise numbers are not trustworthy yet**, and
saying so is the point:

- run-to-run spread reached **16–40%** against this project's 3% bar;
- the GPU was at **36–53% utilization** and holding 5.8 of 8 GB for other
  processes (Wallpaper Engine, Edge, Steam) throughout;
- decisively, `bench/phase2_cache.py` — the benchmark that originally produced
  the recorded 831/665 tok/s figures — **now times out after 10 minutes** on the
  same machine. The environment, not the code, changed.

An A/B ratio is more robust than an absolute figure here, because both sides
shared the same contention in the same process. But a 3× claim resting on runs
that disagree by 30% with each other is not a measurement, so it is recorded as
provisional and the caveat is written into `results/phase3_end_to_end.md` itself
— a warning that only ever appeared on stdout does not survive being pasted into
a README.

**Phase 3 is functionally complete and its headline number is pending one clean
re-run on an idle GPU.**

- [x] Kernel 4 — decode fused attention (online softmax)
#### The clean run: 2.33x end to end

Re-run on an idle GPU (2% utilization, 350 MiB in use), harness-matched
methodology (3 runs after 2 discarded warmups):

| Batch | PyTorch tok/s | Custom kernels tok/s | Speedup |
|---|---|---|---|
| 1 | 25.1 | 58.6 | **2.34x** |
| 4 | 97.6 | 225.9 | **2.32x** |
| 16 | 376.8 | 905.1 | **2.40x** |
| 32 | 745.1 | **1739.1** | **2.33x** |

**2.32-2.40x, flat across a 32x range of batch sizes**, and three independent
clean runs (3, 5 and 9 repeats) all agreed on 2.2-2.4x. Contrast the contended
runs, which gave 1.85-3.29x and disagreed with each other by 30%: the contention
was not adding noise around a true value so much as producing numbers that were
not measurements at all.

For context against the Phase 2 table, re-measured in the same clean window:
Phase 2 paged is 724.1 tok/s at batch 32 and HF `generate()` is 714.6, so the
kernel path at 1739.1 is **2.4x HuggingFace** on the same hardware, model and
prompts.

The Phase 2 figures reproduced within their documented variance (contiguous
853.2 vs 831.6 recorded, paged 724.1 vs 664.9, HF 714.6 vs 686.9), which
retroactively confirms that the 10-minute timeout during the contended window
was environmental. No correction to the Phase 2 table was needed.

**Why 2.33x is higher than predicted.** The expectation written down beforehand
was a modest gain, because kernels 1-3 are launch-bound at decode sizes and
kernel 4 runs at 11.1% of peak. The gain is larger than that reasoning suggested,
and the honest reading is that *launch-bound* was the operative word: a decode
step issues five kernels per RoPE, five per RMSNorm and three per SwiGLU, 24
times per token. Collapsing those into one launch each removes a per-step CPU
cost that no bandwidth argument captures. The microbenchmarks measured the wrong
thing for this regime -- they measured bandwidth on large tensors, when what the
engine actually gains at decode is launches avoided.

#### A methodology fix found while chasing the 3% bar

Adding runs made the reported spread *worse*: 6.4% at 5 runs, 13.9% at 9 runs,
on an idle GPU doing identical work. The harness's stability metric is
`(max - min) / median`, which is **sample-size dependent** -- more runs sample
more of the tail, so the number grows even when the distribution has not moved.

A "variance < 3%" claim is therefore only reproducible if the run count is quoted
with it. `bench/phase3_end_to_end.py` now reports both that spread (at
harness-matched defaults, so it is comparable to the 3% bar) and the coefficient
of variation `stdev / mean`, which is sample-size stable and is the number to use
when comparing runs that used different repeat counts. On the clean run: spread
8.9% at 3 runs, cv 4.5%.

- [x] Kernels wired into `model.py` behind a flag; correctness verified end to end
- [x] End-to-end tokens/sec: **2.33x** at batch 32 (2.4x vs HuggingFace), clean run
- [x] **PHASE 3 COMPLETE**

---

## Phase 4 — Quantization

### Step 1: schemes + quality measurement (2026-09-04)

Phase 0's discipline applied again: build the measuring instrument first,
validate it on a known-good configuration, and only then use it to judge
anything. So this step is the quantization reference plus `bench/perplexity.py`,
with no kernel at all — `quantize_dequantize` round-trips a weight through the
quantized grid and hands back fp16, which is deliberately useless for speed and
exactly right for isolating the **quality** cost.

**What is quantized, and why the headline is not 4×.** Only the 2-D projection
matrices (q/k/v/o/gate/up/down): 715.7 MB of 988.1 MB, **72.4%**. The embedding
is *tied* to the output head in this model, so an error there is applied twice —
once looking a token up and again producing logits — and it is the most
quality-sensitive tensor in the network. Leaving it fp16 caps the win, and three
different numbers get conflated in most write-ups, so all three are reported:

| | INT8 | INT4 (g128) |
|---|---|---|
| quantized tensors alone | 2.00× | 3.82× |
| **whole model** | **1.57×** | **2.15×** |
| effective bits/weight | 8.01 | 4.19 |

INT4 group-wise is **4.19 bits/weight, not 4.0**: each group of 128 weights
carries an fp16 scale and a uint8 zero, so 64 packed bytes become 67. A test
pins that figure rather than letting "4-bit" stand in for it.

### Quality, with an error bar

WikiText-2 test split, 16 windows × 512 tokens = 8,176 predicted tokens, run
through the Phase 1 reference forward so the cache is not part of the
measurement. Every configuration sees identical windows in identical order.

| Precision | Perplexity | vs fp16 | Model | Compression | bits/wt |
|---|---|---|---|---|---|
| fp16 | 22.4164 | — | 988 MB | 1.00× | 16.00 |
| **INT8** | **22.2838** | **−0.59%** | 631 MB | 1.57× | 8.01 |
| INT4 g32 | 25.6464 | +14.41% * | 485 MB | 2.04× | 4.75 |
| INT4 g64 | 25.9810 | +15.90% * | 468 MB | 2.11× | 4.38 |
| INT4 g128 | 27.1170 | +20.97% * | 460 MB | 2.15× | 4.19 |

**The baseline is 22.4164 ± 3.45% (one standard error).** A perplexity delta
without an error bar is not interpretable, and this one earns its keep
immediately: INT8's −0.59% is *smaller than the sampling error*, so the honest
statement is **"INT8 is lossless within measurement precision"** — not "INT8
improved the model", which is what the raw sign would have suggested. The `*`
marks deltas larger than one standard error; only the INT4 rows have them.

**INT4 costs 14–21% perplexity, and that is worse than published INT4 results.**
Stated plainly rather than buried, per spec rule 5. Two reasons, both real:

1. **This is round-to-nearest with no calibration.** GPTQ and AWQ spend a
   calibration pass compensating quantization error against actual activations;
   RTN just rounds. That gap is most of the difference.
2. **0.5B is a small model.** Quantization robustness comes largely from
   redundancy, and a 0.5B model has far less of it than the 7B+ models INT4
   results are usually quoted on.

**The group-size curve is the useful part.** Going 128 → 32 recovers a third of
the loss (21.1% → 14.5%) for 0.56 more bits/weight, and *costs* whole-model
compression (2.15× → 2.04×) because the metadata grows. That is the real
trade-off surface, and it is why the kernel takes the group size as a parameter
rather than baking in 128.

Qualitatively, INT8 continuations track fp16 closely while INT4 diverges into
different-but-fluent text — which is exactly the failure mode that makes a
perplexity number necessary. Greedy decoding can look fine while the
distribution underneath has measurably flattened.

### A bug the correctness bar caught immediately

The first version computed the scale in fp32, chose codes against it, then
**stored the scale in fp16**. Round-to-nearest guarantees an error of at most
half a quantization step, so the test asserted exactly that — from first
principles, not as a tolerance — and it failed on 535 elements.

The cause was real: at `q = 127`, an fp16 scale rounding of 2⁻¹¹ relative shifts
the reconstruction by ~0.06 of a step, enough to break the half-step guarantee.
The fix is to **round the scale to fp16 before choosing codes**, so quantization
and dequantization agree on the same grid — which is also what the kernel will
have to do. Free accuracy, found by refusing to widen a bound that was derived
rather than guessed.

The final bound in the test is `half_step + |reconstruction| × 2⁻¹¹`, both terms
derived: round-to-nearest gives s/2, and the dequantized value is itself stored
in fp16.

- [x] INT8 per-channel symmetric + INT4 group-wise asymmetric, with tests
- [x] Perplexity harness on WikiText-2, with standard errors
- [x] Quality measured: INT8 lossless within error, INT4 +14-21%
### Step 2: the fused dequant-matmul kernel, and the two ceilings it did not reach

The kernel does what the spec demands: unpacks INT4 in registers between the
load and the multiply, so **no fp16 copy of a weight ever crosses the memory
bus**. A test measures that rather than asserting it by inspection — a
dequantize-then-cuBLAS implementation would allocate `out × in × 2` bytes, and
would move *more* bytes than plain fp16, defeating the entire purpose.

Design: one **warp per output row**, batch accumulators held in registers with
the tile size a compile-time constant. The obvious alternative — one block per
(batch row, output row) — re-reads the whole weight matrix once per batch row,
which at batch 32 would be 32× the compulsory traffic and would destroy the
saving before it started.

#### The prediction was right about the mechanism and wrong about the crossover

Predicted: quantization is a *decode* optimization; at *prefill* the weight is
reused across many rows, intensity climbs, and it should stop paying. Prefill
did behave exactly that way (0.04–0.09×, catastrophic — cuBLAS on tensor cores).

But the crossover is not decode-versus-prefill. **It is around batch 4–8, well
inside decode:**

| batch | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ours (INT4, 4864×896) | 34.4 | 41.1 | 54.6 | 83.6 | 146.1 | 269.3 µs |
| fp16 (cuBLAS) | 60.1 | 61.6 | 60.1 | 60.3 | 61.6 | 63.0 µs |

Our time scales with batch. **cuBLAS's is flat.** That flatness is the entire
explanation: cuBLAS is still weight-streaming-bound at batch 32, so extra rows
of x ride along nearly free on tensor cores (HMMA). Our kernel does BT × 8
scalar fp32 FFMAs per 4-byte weight load, so once there is enough batch to leave
the memory-bound regime, we are competing against tensor cores with scalar math
and lose by roughly their throughput ratio.

**That was checked, not assumed.** Splitting batch 32 into four BT=8 tiles gives
0.97× — no better — so register pressure and unrolling are not the cause. A
kernel that won here would have to dequantize into fp16 fragments and issue HMMA
itself, which is a different and much larger kernel.

| shape | INT8 wins up to | INT4 wins up to | best (batch 1) |
|---|---|---|---|
| q_proj / o_proj 896×896 | batch 16 | batch 16 | 1.75× / 1.73× |
| gate/up 4864×896 | batch 4 | batch 4 | 1.79× / 1.75× |
| down_proj 896×4864 | batch 4 | batch 4 | 1.84× / 1.77× |

#### The second ceiling that was not reached, and how it announced itself

**INT8 and INT4 measure the same speed** — 1.79× vs 1.75×, 1.84× vs 1.77×.
INT4 moves *half* the bytes of INT8. If either were bandwidth-bound, INT4 would
be roughly twice as fast. They are indistinguishable, which is the tell.

Scoring against peak bandwidth confirms it:

| shape | INT8 | INT4 |
|---|---|---|
| q_proj 896×896 | 5.4% of peak | 2.7% |
| gate 4864×896 | 28.9% | 14.1% |
| down 896×4864 | 29.8% | 14.3% |

And the giveaway: **~33 µs is a floor that appears at every shape**, including
one 8× smaller than another. That is launch and dispatch overhead, exactly the
same floor kernels 1–3 hit (~58 µs there). So the batch-1 win is real but it is
**not** the memory-traffic win the byte count predicted — it is one custom
kernel launch beating cuBLAS's overhead at small shapes. The 2× and 4× ceilings
were never approached, and quoting them as achieved would be wrong.

This is Gotcha #17 for the third time: **a byte-counted ceiling only binds when
the kernel is actually bandwidth-bound.** Kernels 1–3 hit their predictions
because they were. Decode attention missed by 10× because it was latency-bound.
This one misses because it is launch-bound at small batch and tensor-core-bound
at large.

#### What this actually means for the engine

Weight-only quantization on this GPU buys **VRAM always** (2.15× smaller model),
**latency for single-stream decode** (1.8× at batch 1), and **costs throughput
for batched decode** (0.23× at batch 32). Those are different products.

There is a real trade-off to state plainly: the VRAM saving only materialises if
the fp16 weights are *not* also resident. Falling back to fp16 above the
crossover means keeping both copies, which gives up the memory win to keep the
speed. A batch-1 latency-oriented deployment would take the quantized path
throughout; a throughput-oriented one would not use INT4 on this hardware at
all — INT8 at least stays lossless.

Combined with the quality result from step 1 — INT8 lossless within measurement
error, INT4 +21% perplexity — **INT8 is the configuration worth shipping at this
model size**, and INT4 is the one that demonstrates the technique.

#### A bug the kernel tests found in the step-1 quantizer

A test built a deliberately *constant* group (all 128 weights equal) to check
that per-group scales were being used. It failed with the kernel returning
exactly zero — and so did the reference, which located the bug upstream: a
constant group has `hi == lo`, so `scale = 0`, and the obvious guard
(substitute 1.0) maps every element to code 0 and reconstructs the whole group
as **zero**, silently destroying it.

Fixed by widening the range so the constant lands on a grid point: `lo > 0` →
range `[lo, 2lo]` reconstructs via code 15; `lo < 0` → `[lo, 0]` via code 0 and
zero-point 15; `lo == 0` → zero, correctly. Constant groups now reconstruct
**exactly** (0.00e+00 error) at every magnitude tested. Degenerate cases are
where guards get written carelessly, which is why the test used one.

- [x] Fused dequant-matmul kernel (unpack in registers, never materialize)
- [x] Speed measured, crossover found at batch 4-8
### Step 3: packed weights end to end, and the acceptance table

The engine now runs on packed weights: `pack_weights` keeps `Int8Tensor` /
`Int4Tensor` objects and `model._qlinear` routes them to the fused kernel, so
**no fp16 copy of a projection is ever resident**. A test asserts that by name
rather than trusting it, because the memory claim is false the moment one
survives.

**The routing decision, stated rather than hidden.** The kernel wins at batch
1–4 and loses above it. Two responses were available:

- **(a) always quantized** — the VRAM saving is real; batched decode is slower
- **(b) quantized below the crossover, fp16 above** — best speed, but both
  copies must be resident, so the memory saving evaporates

This project takes **(a)**, because the memory reduction is the claim INT4
actually delivers on this hardware, and (b) would keep the headline number while
quietly making it untrue. The throughput cost appears in the table instead of
being engineered around.

#### The acceptance table

Idle GPU (0% utilization, 349 MiB), prompt 32, 64 new tokens, median of 3 runs.
All rows on packed weights.

| Precision | Weights | Compression | bits/wt | tok/s (batch 1) | tok/s (batch 32) | Peak VRAM | Perplexity | vs fp16 |
|---|---|---|---|---|---|---|---|---|
| fp16 | 988 MB | 1.00× | 16.00 | 57.8 | **1679.5** | 1030 MiB | 22.42 | — |
| **INT8** | 631 MB | 1.57× | 8.01 | **69.0** | 1325.1 | 682 MiB | 22.28 | −0.59% |
| INT4 | 460 MB | 2.15× | 4.19 | 67.7 | 840.3 | **519 MiB** | 27.12 | +20.97% |

Relative throughput: INT8 **1.17× at batch 1, 0.77× at batch 32**; INT4 **1.16×
and 0.50×**. Peak VRAM 1030 → 519 MiB, a **1.99×** reduction.

Two things worth reading off it.

**The end-to-end batch-1 win (1.17×) is far smaller than the kernel's 1.8×.**
Both numbers are correct. Only 72.4% of the weights are quantized — the tied
embedding stays fp16 — and a decode step also spends time in attention, the KV
cache, RMSNorm, RoPE and SwiGLU, none of which quantization touches. Amdahl,
measured rather than assumed.

**INT4 is not faster than INT8 anywhere**, at either batch size, despite holding
half the bits. That is the launch-bound finding from step 2 surviving into the
end-to-end number: neither scheme is bandwidth-bound at these shapes, so halving
the bytes buys nothing in time. INT4's entire advantage over INT8 here is
**VRAM**, and it pays 21% perplexity for it.

#### The conclusion this phase actually supports

**INT8 is the configuration worth shipping at this model size.** It is lossless
within measurement error, 1.57× smaller, faster at batch 1, and costs 23%
throughput at batch 32. INT4 doubles the memory saving and buys nothing in speed
while costing 21% perplexity — on a 0.5B model with round-to-nearest and no
calibration, that is the wrong trade unless VRAM is the binding constraint.

That is a narrower claim than "we implemented INT4 and got 2.15× compression",
and it is the one the measurements support.

- [x] Wire into model.py + acceptance table (size, tokens/sec, perplexity, VRAM)
- [x] **PHASE 4 COMPLETE**

---

## Phase 5 — Make it legible (2026-09-04)

- **README.md** rewritten around the benchmark table: HF vs Phase 1 vs Phase 2
  vs Phase 3 vs INT8, at batch 1 and batch 32, every row citing the script that
  reproduces it. Plus a mermaid architecture diagram (request -> cache ->
  kernels -> token), the four-kernel table with % of peak and predicted-vs-actual
  ceilings, the quantization trade-off table, and a limitations section.
- **WRITEUP.md** written, ~1,100 words: "The ceiling that wasn't". Three kernels
  hit a byte-counted ceiling within 1%; decode attention missed it by 10x. The
  controlled experiment (hold the KV pool fixed, vary only how many query heads
  share it -- 2 to 8 heads quadruples the work and wall time does not move)
  showed it was latency-bound rather than bandwidth-bound, and the block-size
  sweep that followed took it from 7.9% to 11.1% of peak, 3.6x at batch 1.
  Closes with the same pattern appearing twice more (RoPE beating its ceiling,
  INT4 and INT8 measuring identically).
- **vLLM row: blocked, and labelled as such.** No Windows wheels; the sdist
  fails to unpack under Windows path-length limits (verified with
  `pip install --dry-run vllm`). Recorded as Gotcha #27 with the honest options.
- **Still open:** the MiniDynamo cross-link placeholder in README.md, and the
  reciprocal link from MiniDynamo.

- [x] README with the benchmark table and methodology
- [x] Architecture diagram
- [x] WRITEUP.md
- [x] Limitations section
- [ ] MiniDynamo cross-link (needs the real URL)
- [ ] vLLM row (blocked on platform; see Gotcha #27)

## 2026-09-04 — Kernel 4b: head-group fusion, and what it revealed

### The measurement that started it

The repo had decode attention recorded at **11.1% of peak bandwidth**, called it
the weakest number in the project, and named two unimplemented fixes. Before
writing either, I checked the number itself.

`11.1%` counts **compulsory** bytes — each KV element once. But the kernel
launched one block per (sequence, query head), and with GQA 14q/2kv seven blocks
each read the same KV rows. So there are two possible denominators and only one
had ever been computed.

The experiment holds block count and per-block work **exactly** constant (448
blocks, 512 threads, same dot products, same barriers) and varies only how many
query heads share a KV head, which changes only the distinct footprint:

| kv heads | n_rep | distinct | issued | us | distinct GB/s | issued GB/s | % peak |
|---|---|---|---|---|---|---|---|
| 14 | 1 | 234.9 MB | 234.9 MB | 1199.9 | 195.8 | 195.8 | 43.7% |
| 7 | 2 | 117.4 MB | 234.9 MB | 875.3 | 134.2 | 268.3 | 59.9% |
| 2 | 7 | 33.6 MB | 234.9 MB | 729.3 | 46.0 | 322.1 | **71.9%** |
| 1 | 14 | 16.8 MB | 234.9 MB | 694.2 | 24.2 | 338.4 | 75.5% |

Wall time **flattens** once n_rep >= 7 — the duplicate reads are cache hits, so
DRAM was never the limit. But a cache hit still costs a load instruction and an
issue slot, and against issued traffic the kernel was at **71.9% of peak**. It
was never leaving 89% of the card unused. It was a load path near its ceiling
carrying 7x more traffic than the algorithm needs.

That inverts the fix. "Go faster" was not available. "Ask for less" was.
Recorded as Gotcha #28. Reproduce: `python -m bench.kernel_attention --sharing`.

### Kernel 4b

One block per (sequence, **KV** head). The R = n_rep query heads sharing it ride
along: each K element is loaded once into a register and fed to R dot products,
each V element loaded once and fed to R accumulators. Arithmetic intensity goes
0.5 -> 3.5 FLOP/byte. m, l and the correction factor become length-R register
arrays, and one `block_reduce_vec<R>` reduces all R values with the barrier cost
of a single reduction — otherwise the fusion would pay R times the
synchronisation it exists to amortise.

Two implementation notes worth keeping:
- `corr` and `l` have to be **staged through shared memory** before anything
  indexes them by a thread-varying index. Every thread holds identical copies
  (they are outputs of block-wide reductions), but a dynamic index into a
  register array spills the array to local memory and undoes the point.
- The tile's slot indices are staged in shared once per tile. Without that, the
  PV phase re-reads `slots[]` from global on every ngrp-strided step, and each is
  a dependent load standing in front of a V read.

| Shape | Phase 2 paged | Pre-gathered | Per-query-head | Grouped | Fusion win | vs paged | GB/s | % peak |
|---|---|---|---|---|---|---|---|---|
| b1 L128 | 373.8 | 305.2 | 32.8 | 35.0 | 0.94x | 11.40x | 2.0 | 0.4% |
| b32 L128 | 380.2 | 310.0 | 61.4 | 38.9 | 1.58x | 9.77x | 53.9 | 12.0% |
| b32 L512 | 514.5 | 415.7 | 200.1 | 80.5 | 2.48x | 6.39x | 104.1 | 23.2% |
| b32 L1024 | 899.1 | 743.7 | 352.0 | 136.6 | 2.58x | 6.58x | 122.8 | 27.4% |
| b32 L2048 | 1701.1 | 1355.0 | 674.0 | **247.0** | **2.73x** | **6.89x** | **135.9** | **30.3%** |

**11.1% -> 30.3% of peak. 2.52x -> 6.89x vs the Phase 2 paged path.**

Correctness first, as always. Error against the fp64 CPU ground truth is
*identical* to the per-query-head kernel's on all 8 shapes tested, and both are
0.42-1.00x the fp16 reference's — so the fusion is free numerically, not a
trade. Per-head separation is asserted directly with a delta-softmax
construction (deviation 0.000e+00). 130 tests pass.

### Two sweeps behind the heuristics

**Crossover (grouped speedup vs per-query-head, by block count):** the grouped
kernel launches n_rep times fewer blocks, so it has a floor. It loses below 8
blocks and wins above, growing with block count:

| blocks | 2 | 4 | 6 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| L2048 | 0.64x | 0.64x | 0.64x | 1.06x | 1.82x | 2.55x | 2.92x |

**Block size:** not a constant, but a constant TOTAL thread count. Measured best
was 1024 at 8/16/32 blocks, 512 at 64, 256 at 128 — that is `blocks x threads ~
32768` in every row, about 22 warps/SM on 46 SMs, roughly half the 48-warp max.
Dividing a fixed budget reproduces the measured optimum or comes within 9%
everywhere. Gotcha #31.

### The result that matters more than the kernel

**A 2.6x kernel moved end-to-end `generate_paged` by 0.99-1.03x.** Measured back
to back in one process, both arms forced with `fuse_heads`, call count verified
at 1512 per run (24 layers x 63 decode steps).

| case | per-head tok/s | grouped tok/s | speedup | decode ms/step |
|---|---|---|---|---|
| b32 p32 g64 | 1572.8 | 1550.0 | 0.985x | 20.2 |
| b32 p512 g64 | 1024.8 | 1057.1 | 1.032x | 20.1 |
| b32 p1024 g64 | 624.9 | 622.5 | 0.996x | 19.9 |
| b8 p1024 g64 | 307.4 | 307.1 | 0.999x | 18.4 |
| b1 p1024 g64 | 53.2 | 54.4 | 1.022x | 17.5 |

Look at the last column rather than the speedups. **Decode step time is ~20 ms
regardless of batch size (1 vs 32 — 32x the work, +7%) and regardless of context
length (33 vs 1025).** No GPU-bound loop can be invariant to both. And the
attention kernel alone, timed on those exact tensors, costs 0.44 ms/step at
context 33 versus 8.33 ms at context 1025 — a 7.9 ms difference that simply does
not appear in the total.

The cause, and it is countable rather than inferred: **~3,200 aten dispatches per
decode step**, the same 3,218 at context 33 and at 1025. Only **169** are
`linear`. The rest are metadata: `as_strided` x660, `view` x393, `transpose`
x289, `reshape` x245, `select` x185. At a few microseconds of CPU dispatch each,
that is the entire 20 ms.

**The decode loop is CPU-dispatch-bound. The GPU is idle for most of it.** Which
is Gotcha #4 of this project recurring one level up — I optimized the fastest
part of a step I had never measured end to end. Recorded as Gotcha #29 and
promoted to the immediate next action in ROADMAP.md; CUDA graphs are the obvious
fix, since the decode shape is fixed across steps.

Also recorded: `torch.profiler` could not be made to give a trustworthy
GPU-utilization ratio here (summing `self_device_time_total` over
`key_averages()` double-counts, and profiling inflates the CPU side enough that
GPU-busy-per-step exceeds unprofiled wall time — both attempts returned
113-423%). Its *counts* are reliable; its timings in this regime are not. The
invariance experiments settled it without a profiler. Gotcha #30.

One test-writing lesson too (Gotcha #32): the first per-head-separation test used
a score gap of 7.5, which leaves 3.4% of the softmax mass off-target — enough to
move the output by 0.137 and fail its own 0.02 bar. The kernel was right; the
fixture's premise was not. Widening the gap to 3125 made the selection exact.

## 2026-09-06 — Published, and the two history rewrites before it

Not engineering, but part of the artifact (CLAUDE.md rule 6), so recorded.

- MIT LICENSE added.
- All commits re-authored from `Iceboy66 <98899dragons@gmail.com>` to
  `NathanS7878 <988dragons@gmail.com>`, so GitHub attributes the history to the
  account MiniDynamo lives on. Verified content-identical: same HEAD tree hash,
  same commit count, empty diff against a backup ref.
- The `Co-Authored-By: Claude ...` trailer stripped from the 29 commits carrying
  it, at Nathan's request, and force-pushed. Again content-identical.
- Published at https://github.com/NathanS7878/nano-infer. Nathan did every GitHub
  step himself; nothing was pushed from the assistant session.
- Backup refs deleted after the push landed.

## 2026-09-16 — MiniDynamo reciprocal link

MiniDynamo's README now links back here (MiniDynamo commit `1d2a1aa`), so the
"router down to the CUDA kernel" story reads in both directions. That repo had
unrelated uncommitted work in flight (`router/src/main.rs`, `router/src/router.rs`,
untracked `router/src/dashboard.html`); only README.md was staged. MiniDynamo had
no local git identity, so a commit would have been authored as Iceboy66 — its
local `user.name`/`user.email` are now set to NathanS7878 to match.

## 2026-09-16 — CUDA-graph decode: removing the host from the loop

### What was actually in the step

Gotcha #29 had counted ~3,200 aten dispatches per decode step. Before designing
a fix, the GPU->host synchronisations were counted too, with
`torch.cuda.set_sync_debug_mode("warn")`:

| batch | syncs per decode step | where |
|---|---|---|
| 1 | 3 | `forward_paged:629`, `:630`, `cache.plan:247` |
| 32 | **65** | 32 x `int(lengths[i])` at :629, 32 x at :630, 1 x `int(lengths.max())` |

The first attempt reported 5 syncs over 10 steps. Python's default warning
filter shows each call site once; `warnings.simplefilter("always")` gave the
real count. (Gotcha #34.)

Both costs exist for one reason: the step recomputes bookkeeping whose answer is
known before decode starts. In `generate_paged` every sequence's final length is
prompt + max_new_tokens.

### The design (`nano_infer/decode_graph.py`)

- Allocate every KV block before decode; build the `[batch, final_len]` read
  table ONCE.
- Derive per-step state on the GPU: `lengths = pos + 1`, write slot =
  `read.gather(1, pos)`, mask = `arange < lengths`.
- The step writes its argmax into its own input buffer, records it via
  `index_copy_` at a device-side step counter, and advances `pos` in place.

So a decode step has no host inputs at all — no H2D copy, no sync, no allocator
call — which is the precondition for a CUDA graph. It reuses
`model.attention_paged` unchanged, so it runs the same attention code as eager.

`use_graph=False` runs this static step eagerly; `use_graph=True` captures it.
Keeping both separates the sync/bookkeeping win from the dispatch win.

Skipped on purpose: hand-cutting the view/transpose/reshape ops first. They
launch no GPU work, so a graph erases their host cost entirely; the eager static
mode is the control that shows what the graph adds.

### Correctness first

| check | result |
|---|---|
| graph replay vs eager static step, 5 shapes x kernels off/on | **bitwise equal, 10/10** |
| static graph vs `generate_paged`, kernels ON, 5 shapes | **token-identical, 5/5** |
| same, kernels OFF, batch 32 p32 g64 | 2 of 32 sequences diverge — both near-ties |
| step under `set_sync_debug_mode("error")`, kernels off/on | **no sync raised** |

The kernels-off divergence was attributed, not bounded (Gotcha #20). The one
structural difference is that the read table is `final_len` wide from the first
step instead of growing. Holding everything else fixed:
- width 33 twice: bitwise identical (not nondeterminism);
- width 33 vs 96: layer-0 attention moves by 1.5e-05 to 6.1e-05 (fp16 rounding
  from a different `q @ K^T` shape, Gotcha #2);
- decode kernel at width 33 vs 96: **bitwise identical** — it reads `lengths` and
  never materialises scores, so kernels-on parity is structural, not luck.
The flips: seq 17 at step 1, reference top-1/top-2 gap **0.0000** (exact tie);
seq 21 at step 62, gap **0.0156** — both below the 0.02-0.04 logit shift the
width causes. (Gotcha #37.)

One bug on the way (Gotcha #35): warmup ran on a side stream after
`side.synchronize()`, which waits on the WRONG stream. The prefill and state
snapshots were still queued on the default stream, so warmup restored from
unwritten clones -> garbage `pos` -> out-of-bounds gather. Passed at batch 1 on
timing, asserted at batch 4. Fixed with `torch.cuda.synchronize()` at the capture
boundaries.

20 new tests in `tests/test_decode_graph.py`; 150 passing overall.

### The measurement (`bench/decode_graph.py`)

**Contended GPU: 23% utilization and 1,353 MiB at start** (Wallpaper Engine,
browsers, Steam, Discord, Spotify). The benchmark was built for that: all six
arms run round-robin inside every repeat, so background load hits each arm
equally and the RATIOS hold. The absolutes are not publishable and are not in
the README table. cv stayed at 0.2-6.9%.

Decode ms/step:

| batch | prompt | paged off | paged on | static off | static on | graph off | graph on | graph vs paged (on) |
|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 49.36 | 20.14 | 46.78 | 18.62 | 7.38 | **3.67** | **5.49x** |
| 4 | 32 | 48.90 | 21.12 | 48.69 | 18.72 | 7.27 | **3.95** | **5.35x** |
| 16 | 32 | 49.08 | 22.16 | 48.35 | 19.20 | 8.43 | **4.05** | **5.47x** |
| 32 | 32 | 51.89 | 23.29 | 48.13 | 19.42 | 9.67 | **4.41** | **5.28x** |
| 32 | 512 | 50.84 | 22.48 | 47.99 | 19.24 | 17.82 | **5.66** | **3.97x** |
| 8 | 1024 | 49.08 | 21.79 | 47.32 | 19.34 | 10.98 | **5.17** | **4.22x** |

End to end, including prefill AND capture, batch 32 p32 g64: 1,359.6 -> 4,669.2
tok/s with kernels on (3.43x).

Capture: 0.12-0.14 s kernels on, 0.22-0.25 s kernels off (more ops to record),
paid every call. At batch 32 p32 g64 with kernels on that is 0.124 s of a
0.439 s call — 28%.

### What it showed

**1. Syncs were cheap; dispatch was the cost.** Static (all 65 syncs removed)
vs paged: 1.00-1.20x. Graph vs paged: 5.3-6.7x at short context. Counting a
cost is not measuring its weight. (Gotcha #34.)

**2. Gotcha #29 drew the wrong lesson, and this is the correction.** Under the
old `paged` engine, all kernels on vs all off is **2.21-2.45x on decode**. #29
claimed the Phase 3 kernels were optimizing a part of the step nobody waited on,
and the roadmap said the end-to-end kernel gain was "mostly prefill". Both wrong.
What had been flat was kernel 4 vs 4b — the same number of launches. In a
host-bound loop **a fusion pays through the launches it deletes, not the GPU time
it saves**: RMSNorm and SwiGLU each collapse a chain of PyTorch ops into one
launch, 4b replaces one launch with one launch. (Gotcha #33.)

**3. Decode is now GPU-bound, and specifically weight-streaming-bound.** Weights
read per step, computed from the checkpoint shapes: 24 x 29.8 MB of layer
matrices + 272.3 MB tied lm_head = **987.9 MB** (the lm_head is 28% of it).

| engine | batch | ms/step | weight GB/s | % of 448 GB/s peak |
|---|---|---|---|---|
| graph+kernels | 1 | 3.67 | 269.2 | **60.1%** |
| graph+kernels | 4 | 3.95 | 250.1 | 55.8% |
| graph+kernels | 32 | 4.41 | 224.0 | 50.0% |
| paged+kernels | 1 | 20.14 | 49.1 | 10.9% |
| paged+kernels | 32 | 23.29 | 42.4 | 9.5% |

Floor at peak: 2.21 ms/step. Weights-only, so it understates total traffic
(KV reads excluded), and it was measured contended. And the signature changed:
step time now GROWS with context (batch 32 kernels off, 9.67 -> 17.82 ms from
p32 to p512), the invariance that proved #29 is gone. Kernels on vs off under
graphs is 1.84-2.19x at short context and **3.15x at batch 32 p512**, where the
attention kernel is finally what the engine waits on. (Gotcha #36.)

### Prediction question for Nathan, before the next experiment

Decode is now streaming ~1 GB of weights at ~60% of peak. INT4 stores those
weights in about a quarter of the bytes. Phase 4 measured the INT4 matmul
kernel LOSING to cuBLAS above batch 4 — in the host-bound regime. Under graphs,
at batch 1, do you expect INT4 decode to be faster or slower than fp16, and by
roughly how much? Commit to a number before looking at the next entry.

## 2026-09-16 (cont.) — Quantization under CUDA graphs, and capture paid once

### Quantization's speed story, re-asked

Every Phase 4 speed number came from the eager engine, now known to be
host-bound. Gotcha #26 had leaned on that without knowing it: "INT8 and INT4
measure the SAME speed ... ~33 us floor per call."

**Safety first.** The dequant-matmul kernels have no host reads in their launch
path; with INT8 and with INT4 weights the static decode step raised no sync under
`set_sync_debug_mode("error")`, and eager-static == `generate_paged` and graph ==
eager-static, token for token, at batches 1, 4 and 32.

**Prediction, written before running.** Weight bytes per step: fp16 987.9 MB,
INT8 630.7 MB, INT4 459.6 MB (the tied lm_head stays fp16), so byte ceilings
1.57x and 2.15x. Phase 4 put the kernel at 2.7-29.8% of peak, so expected well
under ceiling: **~1.3x INT8, ~1.5x INT4 at batch 1; INT4 slower than fp16 at
batch 32.**

**Measured** (`bench/quant_graph.py`, contended GPU 26% / 1,317 MiB, arms
round-robin; decode ms/step; speedup vs fp16 on the same engine):

| batch | precision | eager ms | eager vs fp16 | graph ms | **graph vs fp16** | graph weight % peak |
|---|---|---|---|---|---|---|
| 1 | fp16 | 20.04 | 1.00x | 3.67 | 1.00x | 60.1% |
| 1 | int8 | 17.95 | 1.12x | 2.72 | **1.35x** | 51.8% |
| 1 | int4 | 17.20 | 1.16x | 2.59 | **1.42x** | 39.7% |
| 4 | fp16 | 20.85 | 1.00x | 3.91 | 1.00x | 56.3% |
| 4 | int8 | 17.54 | 1.19x | 3.58 | 1.09x | 39.3% |
| 4 | int4 | 17.49 | 1.19x | 4.77 | **0.82x** | 21.5% |
| 32 | fp16 | 21.89 | 1.00x | 4.39 | 1.00x | 50.3% |
| 32 | int8 | 20.35 | 1.08x | 16.54 | **0.27x** | 8.5% |
| 32 | int4 | 32.37 | 0.68x | 28.18 | **0.16x** | 3.6% |

**Scoring the prediction.** Batch 1: predicted 1.3x / 1.5x, measured 1.35x /
1.42x — held. Batch 32 direction — held. NOT predicted: INT4 already losing at
batch 4, and the size of the batch-32 loss (0.16x).

**What it means.**
- The host overhead was diluting the quantized kernel's loss ~3x. Eager INT4 at
  batch 32 looked like 0.68x; graphed it is 0.16x. Removing the host sped fp16 up
  5x and could not speed up a GPU-slow kernel.
- At batch 1 graphs finally let quantization pay. INT8 reached 86% of its byte
  ceiling, INT4 66%. INT4 vs INT8 is only 1.05x despite streaming 1.37x fewer
  bytes (39.7% vs 51.8% of peak on their own bytes): #26's "INT4 cannot turn
  fewer bytes into speed" survives; its "same speed" was partly the host floor.
- The tied lm_head is unquantized and is 59% of INT4's weight traffic.
- **The tensor-core quantized matmul is now the top kernel item**, not an
  optional extra.

Recorded as Gotcha #38; #25 and #26 annotated in place rather than rewritten.

### Capture paid once: DecodeGraphRunner

Capture was 0.12-0.25 s per call. None of what a graph binds depends on the
prompt — KV pool, read table, rope tables, state buffers all follow from
(batch, prompt_len, max_new_tokens) — so `DecodeGraphRunner` captures on its
first `generate` and serves every later prompt of that shape with prefill +
replay. `generate_paged_static` became a one-shot wrapper around it.

It reuses the KV pool without zeroing. Safe because prefill overwrites
[0, prompt_len) and every decode step writes its slot before any read (the
kernel's loop is bounded by `lengths`; the PyTorch path masks by it). That is an
argument, so it is also a test: three DIFFERENT prompts through one runner must
equal fresh one-shot runs, kernels on and off, and an earlier returned tensor
must not change afterwards (`generate` returns a clone). Plus: capture happens
once (0.126 s then 0.000 s), and the runner raises if the kernel setting differs
from the one it captured with, or if the shape is wrong. 155 tests passing.

`bench/decode_graph_runner.py` (contended, 29% / 1,289 MiB), whole-call tok/s,
each call a different prompt:

| batch | new tokens | eager paged | one-shot graph | reused runner | runner vs one-shot |
|---|---|---|---|---|---|
| 1 | 64 | 1.00x | 3.33x | **4.90x** | 1.47x |
| 32 | 64 | 1.00x | 3.26x | **4.62x** | 1.42x |
| 32 | 16 | 1.00x | **1.61x** | **3.61x** | **2.24x** |

Capture is a fixed cost per call, so the shorter the generation the more of the
graph's win it eats: at 16 tokens one-shot graphs are barely worth it. Gotcha #39.

### Not done, and why

- **Clean-GPU numbers.** Every run today was on a contended desktop. Ratios are
  sound by construction; absolutes wait for Nathan to close the GPU-using apps.
- **Tensor-core quantized matmul.** Now clearly the right next kernel, and a
  large one. Per CLAUDE.md, Nathan should predict memory- vs compute-bound
  before it is written, so it was not started unilaterally.
- **Scaling to 1.5B.** Needs a ~3 GB model download, which needs Nathan's
  go-ahead.

## 2026-09-16 (cont.) — Clean rerun, Nathan's prediction, 1.5B downloaded

### Clean rerun

Nathan closed the GPU-using apps. Verified before running rather than assumed:
utilization 0-2% across three samples (was 23-29%), but 950 MiB still held by
processes that remained resident and idle (Wallpaper Engine, Edge, Discord, a
Steam helper) -- above #18's 500 MiB target. Utilization is what competes for
the card during timing, so the run went ahead, and the benchmarks' strict
"contended" stamp was left as-is rather than loosened to make it pass. Nothing
else ran during it: no kernel compile, no download.

| measure | contended | clean |
|---|---|---|
| graph vs eager decode, b1 p32 (kernels on) | 5.49x | **5.19x** (17.88 -> 3.45 ms) |
| graph vs eager decode, b4 p32 | 5.35x | **5.18x** |
| graph vs eager decode, b16 p32 | 5.47x | **5.02x** |
| graph vs eager decode, b32 p32 | 5.28x | **4.94x** (20.43 -> 4.14 ms) |
| graph vs eager decode, b32 p512 | 3.97x | **3.85x** |
| graph vs eager decode, b8 p1024 | 4.22x | **3.81x** |
| syncs removed only (static vs eager) | 1.00-1.20x | 1.01-1.18x |
| kernels on vs off under graphs, b32 p512 | 3.15x | 3.13x |
| INT8 / INT4 graph vs fp16, b1 | 1.35x / 1.42x | 1.34x / 1.42x |
| INT8 / INT4 graph vs fp16, b4 | 1.09x / 0.82x | 1.10x / 0.82x |
| INT8 / INT4 graph vs fp16, b32 | 0.27x / 0.16x | 0.27x / 0.16x |
| reused runner vs eager, b32 g64 | 4.62x | 4.38x |
| reused runner vs eager, b32 g16 | 3.61x | 3.51x |
| graphed fp16 b1, weight bandwidth | 60.1% | 64.4% |

Every ratio landed within a few percent of its contended measurement -- the
round-robin design doing its job. Absolutes went into the README headline table
beside the HF row: reused-runner decode is **259.9 tok/s at batch 1 and 6,909.0
at batch 32**, which is 9.67x the HF row. That HF row is from an earlier
session, and in this run the eager kernels engine measured ~9% below its own
older Phase 3 row, so the README gives the same-session ratio (4.38x) as the
firmer claim and states the cross-session uncertainty.

### Nathan's prediction for the tensor-core quantized matmul

Recorded before any kernel code existed: **compute-bound, and it will beat
graphed fp16 decode at batch 32.** The bar, from the clean rerun: **4.14-4.15
ms/step.**

My own reading, recorded separately so both can be scored: the tensor-core
multiply itself will be fast, but every weight tile must first be dequantized
through shared memory (see below), so I expect the kernel to be limited by that
dequant traffic rather than by compute, and to land near fp16 rather than
clearly beat it. We will see.

### Design so far

Tensor cores are driven through `<mma.h>` (`nvcuda::wmma`): fixed-shape register
fragments and one instruction, `mma_sync(D, A, B, C) = A.B + C`.

- Sub-byte integer IMMA exists (4-bit `s4`/`u4` fragments loaded straight from
  packed memory), but it multiplies 4-bit by 4-bit -- it would need activations
  quantized too. Rejected: that is a different quality trade, unmeasured here.
- fp16 HMMA fragments (m16n16k16) fit. But a fragment's element layout is
  undocumented; the only supported ways to fill one are `load_matrix_sync` from
  memory and `fill_fragment`. So a quantized weight tile has to exist in memory
  before it can be loaded. CLAUDE.md forbids materializing the dequantized
  matrix in GLOBAL memory, so the plan is one 16x16 tile at a time in SHARED
  memory.
- Step 1 (written, not yet compiled, deliberately not referenced by the build
  while the clean benchmarks were running): `hmma_matmul_f16`, a plain fp16
  X.W^T on tensor cores with both operands loaded from global memory. Its job
  is only to pin the undocumented layout semantics (row/col major, ldm) against
  torch. Tests use rectangular shapes and structured row/column scales so a
  layout error cannot pass as rounding.

### Qwen2.5-1.5B-Instruct downloaded

With Nathan's permission: 3.10 GB from huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
(model.safetensors 3,087.5 MB, plus config, generation config and tokenizer).
Not yet loaded; architecture to be read from its config.json, not assumed.

## 2026-09-16 (cont.) — Tensor-core quantized matmul: a dead end, proven

### The plan

INT4 decode is 0.16x fp16 at batch 32 under graphs (#38), matching the ~8x gap
between this card's scalar fp32 (~20.3 TFLOP/s) and fp16 tensor-core
(~163 TFLOP/s) throughput. So the accumulation had to move to tensor cores. The
only device-side interface is `nvcuda::wmma` (`<mma.h>` -> `crt/mma.h[pp]`):
fixed-shape register fragments and `mma_sync(D, A, B, C) = A.B + C`.

Rejected up front: the sub-byte integer path (4-bit `s4`/`u4` fragments). It
multiplies 4-bit by 4-bit, which would require quantized activations -- a
different, unmeasured quality trade.

Chosen: fp16 `m16n16k16` fragments. Their element layout is undocumented and the
only supported ways to fill one are `load_matrix_sync` from memory and
`fill_fragment`, so a quantized tile would be dequantized into shared memory and
loaded from there -- one tile per worker, never the matrix in global memory.

### Step 1 -- prove the interface before building on it

Before any dequantization: a plain fp16 X.W^T on fragments, both operands loaded
from global memory, compared with torch. Tests used rectangular shapes and
structured row/column scales so a layout error could not pass as rounding.

Two compile issues on the way, both recorded because they will recur:
- torch's extension build defines `__CUDA_NO_HALF_CONVERSIONS__`, which compiles
  out every numeric `__half` constructor. fp16 zero was built from raw bits
  (`__half_raw` with `.x = 0`).
- the 3-D launch dimension type is `dim3`.

**Result: 63-90% error on every shape** (e.g. batch 32 x gate_proj: fragment
error 1.263e+02 against a 1.408e+02 output scale; cuBLAS 5.08e-02). Finite and
plausibly sized -- the signature a layout error was expected to have.

### It was not layout

With every dimension 16, a single tile removes `ldm` from the question, leaving
A row/col x B row/col x store row/col, plus whether `mma_sync` tolerates the
accumulator aliasing its own C input: 16 interpretations. The probe ran all of
them against ten candidate products on small-integer inputs (exact in fp16):

- **0 of 16 match any candidate.**
- X = I, W = I gives 3 nonzero cells, not an identity.
- A single 1 at x[0,1], x[1,0], x[0,15] or x[15,0] produces IDENTICAL outputs --
  the result does not depend on where the input is.
- Output values like 1.19e-7 and 6.56e-7 are fp16 subnormals: small raw integers
  reinterpreted as half-floats.
- **All-zero inputs give nonzero output** (up to 1.3e-4 in one run, 2.0 in
  another), and with an aliased accumulator not the same output twice.
- **y(2X) != 2 y(X).** Not bilinear, so not a multiply under any layout.

The type definitions agree once read carefully: the m16n16k16 fp16 matrix
fragment derives from `__frag_base<__half, 16>` -- 16 storage elements, not 256
-- and the accumulator holds 8. These are not dense 16x16 matrices, and the
header, marked proprietary and "internal ... must not be used directly",
documents no element semantics.

### Decision

Stopped. Reverse-engineering an undocumented internal by probing bit patterns
could not produce a claim this repo could defend. The probe was moved OUT of the
engine build into `nano_infer/kernels/experimental/hmma_probe.cu`, built only by
`bench/hmma_probe.py` as its own extension, so the negative result stays
reproducible (hard rule 3) without the engine carrying code that does not work.
The step-1 tests were removed -- they tested an external interface, not this
repo's code. `bindings.cpp` and `kernels/__init__.py` are byte-identical to
before the attempt; 155 tests still pass.

**Nathan's prediction** (compute-bound; beats graphed fp16 at 4.14 ms/step, batch
32) **was not tested** -- neither confirmed nor refuted. My counter-prediction
(dequant-traffic-bound, near fp16) is equally untested.

### What remains, with ceilings

- Dequantize into a bounded fp16 workspace, then cuBLAS: supported APIs, but it
  moves ~2.26x fp16's weight bytes (read packed, write fp16, stream fp16), so it
  is capped near 0.44x fp16 when memory-bound. Better than 0.16x, never better
  than fp16.
- Accept the trade: INT4 is 1.42x at batch 1 and halves VRAM; it loses above
  batch 1, and the README says so.

**Lesson (Gotcha #40): prove the interface before building on it.** Step 1
tested only the layout, which is why this surfaced as one small, attributable
experiment rather than as a mysterious error inside a quantized kernel, where it
would have looked like a dequantization bug.
