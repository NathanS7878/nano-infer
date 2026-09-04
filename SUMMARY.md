# nano-infer — Project Summary

A single-GPU LLM inference engine written from scratch: custom forward pass, KV
cache, CUDA kernels, and INT8/INT4 quantization. This document is the complete
record — what it is, how it was built, what was measured, what was learned, and
what went wrong along the way.

**Author:** Nathan (Iceboy66) · **Started:** 2026-08-20 · **Status:** Phases 0–2 complete, Phase 3 in progress

> **Headline (RTX 3070, Qwen2.5-0.5B-Instruct, fp16, 128 new tokens):** at batch 32
> the engine reaches **831.6 tok/s — 15.6× faster than its own no-cache version and
> 1.21× faster than HuggingFace `generate()`** — with output verified against a
> frozen HuggingFace reference. On a realistic request stream with varied output
> lengths, continuous batching adds a further **1.55×** over static batching by
> lifting slot utilization from 46% to 100%. Full method and caveats below.

---

## 1. What this is and why it exists

The goal, in one sentence: **load an open-weights transformer, generate tokens
without `model.generate()`, vLLM, TensorRT-LLM, or FlashAttention, and beat a
naive PyTorch baseline by a measured, honestly-reported margin.**

This is the GPU-worker layer beneath **MiniDynamo**, a distributed
KV-cache-aware LLM inference *router* (Rust + Python, modeled on NVIDIA Dynamo).
The two projects tell one coherent story across the serving stack:

| Layer | Project | Question it answers |
|---|---|---|
| Distributed routing | MiniDynamo | *Which worker* should run this request? |
| Single-GPU execution | **nano-infer** | What does that worker *actually do* on the GPU? |

They share one central idea at two scales. MiniDynamo routes a request to the
worker that already holds its **prefix** in cache, so an expensive prefill is
skipped *across requests*. nano-infer uses a KV cache so the frozen past is not
recomputed *within* a request. Same insight — cached work is reusable because the
past never changes — applied at the cluster level and at the kernel level.

### Rules the project holds itself to

1. HuggingFace is used for **weights and tokenizer download only**. The forward
   pass, sampling loop, and cache are ours.
2. No vLLM, TensorRT-LLM, FlashAttention, or xformers. We benchmark *against*
   them; we do not import them.
3. Every performance claim needs a number, a methodology, and a hardware spec —
   reproducible by a script in this repo.
4. **Correctness gates every optimization.** Numerical parity tests run before
   speed is ever measured.
5. **Report where it gets worse.** Quantization degrades quality; small batches
   waste the GPU. Say so, with numbers.
6. Commit incrementally with real messages. The git history is part of the artifact.

---

## 2. Hardware and environment

Every number in this document came from one machine. Full detail in
[HARDWARE.md](HARDWARE.md).

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3070, 8 GB GDDR6, Ampere **sm_86**, 46 SMs |
| **Theoretical peak memory bandwidth** | **448 GB/s** (256-bit bus × 14 Gbps) |
| FP16 tensor peak | ~163 TFLOP/s |
| CPU / RAM | Intel i5-8600K (6C/6T) · 31.9 GB |
| OS | Windows 10 Home 19045 |
| Python / PyTorch | 3.12 (conda-forge) · torch 2.6.0+cu124 |
| transformers | 5.15.1 (weights + tokenizer only) |

The 448 GB/s figure is the denominator for every bandwidth-utilization claim in
Phase 3. The **roofline ridge** is ~364 FLOP/byte (163e12 / 448e9): any operation
doing less arithmetic than that per byte moved is memory-bound on this card. LLM
decode sits far below the ridge — which is why decode is memory-bound, and why
the whole optimization strategy targets memory traffic rather than math.

**Known gap:** `nvcc` (the CUDA compiler) is not installed. The torch wheel
bundles the CUDA *runtime*, which is enough to run PyTorch's GPU kernels, but
compiling our own `.cu` files in Phase 3 requires the toolkit. First task of Phase 3.

### Model

`Qwen/Qwen2.5-0.5B-Instruct` — architecture read from the checkpoint, not assumed:

| Property | Value | Consequence |
|---|---|---|
| Layers | 24 | |
| hidden_size | 896 | |
| intermediate_size | 4864 | SwiGLU width |
| Attention heads | **14 query / 2 KV** | GQA: 7 query heads per KV head |
| head_dim | 64 | |
| vocab_size | 151,936 | logits vector length |
| rms_norm_eps | 1e-6 | |
| rope_theta | 1,000,000 | supports 32k context |
| **tie_word_embeddings** | **True** | no `lm_head` — output projection reuses the embedding matrix |
| Params | 494,032,768 | "0.5B" is rounded up |

GQA is visible directly in the weight shapes: `q_proj` outputs 896 (14×64) while
`k_proj`/`v_proj` output only 128 (2×64). The KV cache is 7× smaller than it
would be under full multi-head attention. Qwen also puts a **bias on q/k/v** but
not on `o_proj` or any MLP projection — a detail that would silently corrupt
output if assumed rather than checked.

---

## 3. Phase 0 — Ground truth ✅

**Goal:** build the ruler and the answer key *before* building the engine, so
every later speed claim is measurable and every optimization is checked.

### Deliverable 1: the benchmark harness (`bench/harness.py`)

Measures **TTFT** (time to first token), **inter-token latency**, and
**tokens/sec** at batch sizes 1/4/16/32. Two correctness rules are baked in:

- **`torch.cuda.synchronize()` before every timer stop.** GPU kernel launches are
  asynchronous — the Python call returns while the GPU is still working. Stopping
  the clock without synchronizing measures *how long it took to ask*, not how long
  the work took. This is the classic way to publish impossibly-fast fake numbers.
- **Warmup runs, discarded.** The first calls pay one-time costs (CUDA context
  creation, allocator warmup, kernel autotuning).

The harness is generic over a `generate_fn(input_ids, max_new_tokens)` callable,
so the same ruler measures HuggingFace and our engine under identical conditions.

### Deliverable 2: the correctness answer key (`tests/`)

`tests/capture_reference.py` runs HF in a manual greedy loop and freezes: the
prompt token ids, the 50 greedily-decoded continuation tokens, the **full fp16
logit vector** at the first decode step, and per-step top-5 (id, logit) pairs —
for 5 varied prompts. 1.47 MB, committed to the repo.

`tests/test_parity.py` verifies the fixture is well-formed and — critically —
that a fresh HF run **reproduces it exactly**. That determinism check is what
makes the answer key trustworthy.

### Baseline results (Qwen2.5-0.5B, fp16, 128 new tokens)

| Batch | Tokens/sec | TTFT (ms) | Inter-token (ms) | Variance |
|---|---|---|---|---|
| 1 | 19.5 | 63.0 | 51.3 | 2.3% |
| 4 | 76.9 | 59.9 | 51.9 | 1.2% |
| 16 | 313.3 | 67.3 | 50.9 | 2.3% |
| 32 | 621.5 | 68.3 | 51.4 | 1.0% |

**Acceptance: PASS** — variance 1.0–2.3%, all under the 3% bar.

### Finding #1: the baseline measured the project's own thesis

**Inter-token latency is flat (~51 ms) whether serving 1 sequence or 32**, so
tokens/sec scales almost linearly with batch (19.5 → 621.5, a 32× batch giving a
31.9× throughput). The reason: each decode step's dominant cost is hauling the
model's weights out of VRAM, and that haul is paid **once per step regardless of
batch size**. Thirty-two sequences share one haul.

The corollary is Rule 5 made visible: **at batch 1 the GPU is almost entirely
wasted** — the full weight-streaming cost is paid to produce a single token. The
gap between the batch-1 and batch-32 rows is the inefficiency the rest of the
project exists to reclaim.

---

## 4. Phase 1 — Correct but slow ✅

**Goal:** rebuild the entire forward pass in plain PyTorch, loading raw
safetensors onto our own code. No HF model classes, no `generate()`. Greedy
decode with **no cache** — recompute the whole sequence every step. Deliberately
the slow, obviously-correct version.

### Method: one component at a time, each verified before the next

Correctness was accrued incrementally so a bug would surface the moment the piece
containing it was added, rather than as a wall of wrong logits 300 lines later.
Each component was checked against the equivalent HuggingFace internal output.

| # | Component | What it does | Max abs diff vs HF |
|---|---|---|---|
| 1 | Token embedding | Gather each token id's row from a (151936, 896) table | **0.00e+00** |
| 2 | RMSNorm | `x / sqrt(mean(x²)+eps) * weight`; no mean-subtraction, no bias | **0.00e+00** |
| 3 | RoPE | Rotate Q and K by position × frequency | **0.00e+00** |
| 4 | GQA attention | Q/K/V → heads → RoPE → repeat_kv → scores → causal mask → softmax → blend → o_proj | **0.00e+00** |
| 5 | SwiGLU MLP | `silu(gate) * up`, then down-projection | **0.00e+00** |
| — | **Full forward** (24 layers + final norm + tied logits) | | **0.00e+00** |

### The components, briefly

**RoPE** encodes word order. Attention's dot products are position-blind on their
own. RoPE rotates each token's Q and K by an angle proportional to its position,
which makes the dot product of a query at position *m* with a key at position *n*
depend only on `cos((m−n)·θ)` — **absolute positions cancel, relative distance
survives.** The 64-dim head vector is split into 32 pairs, each rotating at its
own frequency (fast and slow "clock hands"), so every position gets a unique
fingerprint without wrapping out to 32k tokens.

**GQA** is the memory optimization. `repeat_kv` expands the 2 stored KV heads up
to 14 so every query head has a partner (query head *h* uses KV head *h*//7).
This is a memory decision, not a compute one: we store 2 heads of K and V, not
14, which is exactly why the KV cache is 7× smaller.

**Residual connections** are what let 24 layers stack. Each block computes a
small *refinement* that is added onto the signal (`x = residual + attention(...)`)
rather than replacing it, so a block never has to rebuild the whole representation.

### Finding #2: fp16 is deterministic, and that is a testing superpower

I predicted a small nonzero difference at RMSNorm ("it does real arithmetic"),
then at RoPE ("lots of multiply-adds"), then at attention. **All three came back
bit-identical.** Floating-point is deterministic: the same operations, in the same
order, at the same precision, produce the same bits. Every rounding error we make,
HF makes identically.

The corollary matters for later: nonzero diffs appear when an implementation
**diverges in structure**, not merely when it does arithmetic. That is what the
1e-3 tolerance is really reserved for — Phase 3, when our fused kernels stop
mirroring PyTorch op-for-op.

The dtype choreography had to be matched exactly, though. RMSNorm normalizes in
**fp32**, casts back to fp16, *then* multiplies by the learned weight. Getting the
formula right but the cast order wrong would have produced a real error.

### Finding #3 (the debugging story): the answer key was wrong, not the engine

The Phase 1 acceptance test — greedy-decode 5 prompts × 50 tokens, require
token-for-token identity — failed twice. Neither failure was a bug in our code.

**Failure A — prompt 2 diverged at step 0.** Our top-2 logits were
`'Certainly'=23.9062` and `` '```' ``=23.8750; the fixture's were
`` '```' ``=23.9062 and `'Certainly'=23.9062`. A near-tie, broken differently.

Diagnosis: the fixture had been captured with HuggingFace's **default SDPA**
attention (a fused kernel), while our engine mirrors **eager** attention (explicit
softmax). Measured difference: our forward was `0.00e+00` vs HF eager on all five
prompts, but ~0.06–0.13 logit vs HF SDPA — small per-layer rounding accumulated
over 24 layers, enough to flip a coin-flip token.

**Failure B — prompt 4 diverged at step 9.** Diagnosis: the fixture used a **KV
cache**; our Phase 1 engine has none and recomputes the full sequence. Verified by
running HF's *own* eager model both ways: **it diverges from itself at exactly
step 9**, with a steady ~0.04 logit difference between the cached and uncached
paths.

**The fix was to the reference, not the engine:** regenerate the fixture using the
same procedure the engine under test uses — eager attention, no cache.

> **Lesson:** a correctness answer key must be produced by a *deterministic
> reference implementation matching the engine under test* — not by an optimized
> kernel, and not by a different decode path. Otherwise a benign near-tie reads as
> a false failure, and the temptation is to loosen the tolerance until the test
> passes, which destroys the test's value.

This has a direct consequence for Phase 2: adding a KV cache may legitimately flip
near-tied tokens relative to Phase 1's no-cache output. "Output still matches
Phase 1 exactly" needs to be evaluated with that in mind, and any divergence
proven to be a near-tie rather than a bug.

### Acceptance: PASS

All 5 prompts, 50 tokens each, **token-for-token identical to HuggingFace**. Full
forward bit-identical (0.00e+00) to HF eager. 9 tests green.

```
[0] OK  'What is the capital of France?'       -> 'The capital of France is Paris...'
[1] OK  'If a train travels 60 miles...'       -> 'To calculate the average speed...'
[2] OK  'Write a one-line Python function...'  -> "Certainly! Here's a one-line..."
[3] OK  'List three primary colors.'           -> 'Three primary colors are red, blue, and yellow...'
[4] OK  'Explain in one sentence why...'       -> 'The sky is blue because...'
```

---

## 5. Measuring the Phase 1 engine — the honest "before"

Reproduce with `python -m bench.phase1_nocache`. Results in
`results/phase1_nocache.json`.

### Measurement A: cost of one forward pass vs sequence length

| Sequence length | One forward pass | µs per token |
|---|---|---|
| 32 | 38.5 ms | 1203.4 |
| 64 | 37.7 ms | 588.8 |
| 128 | 37.0 ms | 289.2 |
| 256 | 39.9 ms | 155.8 |
| 512 | 39.7 ms | 77.5 |
| 1024 | 77.9 ms | 76.0 |
| 2048 | 223.4 ms | 109.1 |
| 4096 | 692.1 ms | 169.0 |

### Finding #4: there are two regimes, and the knee is around 512–1024 tokens

I expected this curve to grow quadratically from the start. **It does not.** From
32 to 512 tokens the forward pass costs a flat ~38–40 ms — processing 16× more
tokens for free.

The reason is Finding #1 again: at batch 1 and short sequences the pass is
**bound by streaming ~1 GB of fp16 weights out of VRAM** (plus roughly 170 kernel
launches across 24 layers). The attention math over 32 vs 512 positions is noise
next to that fixed cost. Only past ~1024 tokens does attention's O(seq²) term
dominate — and then it bites hard: 2048→4096 costs 3.1×, converging on the 4× of
true quadratic scaling.

At 448 GB/s theoretical peak, streaming 988 MB of weights should take ~2.2 ms. We
measure ~38 ms — roughly **6% of peak bandwidth**. That gap is PyTorch's
per-operation overhead and unfused memory round-trips, and it is precisely what
Phase 3's custom kernels target.

### Measurement B: head-to-head vs the HuggingFace baseline

128 new tokens, identical prompt, CUDA-synchronized, 2 runs, 1 warmup.

| Batch | nano-infer Phase 1 (tok/s) | HF `generate()` (tok/s) | Ratio | Wall clock |
|---|---|---|---|---|
| 1 | **25.4** | 22.5 | **1.13×** | 5.0 s vs 5.7 s |
| 4 | 76.4 | 88.6 | 0.86× | 6.7 s vs 5.8 s |
| 16 | 50.5 | 356.5 | 0.14× | 40.5 s vs 5.7 s |
| 32 | 53.3 | 712.4 | **0.07×** | 76.8 s vs 5.7 s |

### Finding #5: no-cache wins at batch 1 and collapses at batch 32

At **batch 1 our from-scratch engine is 13% faster than HuggingFace** despite
recomputing the entire sequence every step. Two reasons: (a) as Finding #4 shows,
recompute is nearly free in the weight-bound regime, and (b) our decode loop is a
tight `forward → argmax → append`, while `generate()` carries logits processors,
stopping criteria, and cache bookkeeping in Python on every step.

At **batch 32 we are 13× slower.** Once the batch is large, recompute stops being
free: every step processes batch × sequence token-positions (32 × ~160 ≈ 5,120)
against HF's cached 32. The engine crosses from memory-bound into compute-bound,
and the redundant work becomes the entire cost.

Our own scaling shows the crossover clearly: 25.4 → 76.4 tok/s from batch 1→4
(near-linear, still memory-bound), then **50.5 → 53.3** from batch 16→32 —
essentially flat. The "extra sequences are free" property has evaporated because
the GPU is now saturated with redundant arithmetic instead of waiting on memory.

**This is exactly the gap Phase 2 closes**, and it is why the KV cache is the
right next step rather than jumping to custom kernels: the largest win available
is algorithmic, not hardware-level.

---

## 6. Phase 2 — KV cache and batching ✅

**Goal:** remove the redundant recomputation, algorithmically, before touching a
single GPU kernel. Phase 1's code stays intact as the reference implementation —
the cached path is added alongside it, never replacing it.

### Step 1 — Contiguous KV cache + prefill/decode split ✅

`nano_infer/cache.py` holds a preallocated `[layers, batch, kv_heads, max_seq,
head_dim]` tensor pair. K is cached **after RoPE** — the rotation is part of the
frozen past. Only the 2 KV heads are stored, not 14 query heads: at batch 32 /
max_seq 512 that is **192 MB instead of 1,344 MB**, GQA's payoff in bytes, and
the difference between fitting and not fitting in 8 GB alongside the weights.

Prefill and decode became separate code paths because they want different things.
Prefill runs the whole prompt at once and needs a causal mask. **Decode needs no
mask at all** — there is a single query and every cached position is, by
construction, already in its past.

One further optimization: `forward_cached` projects only the *last* position to
logits. Computing all positions (as Phase 1 does) is waste during generation —
at batch 32 with a 34-token prompt that is 330 MB of logits computed to use 9.7 MB.

### Finding #6: proving a cache correct means separating two questions

The Phase 2 acceptance test failed at first, and untangling it produced the most
useful result of the phase.

**Question 1 — is the cache logic right?** Given the same input, does the cached
path compute the same hidden states? **Yes, bit-identically** (`torch.equal`) on
all five prompts. The cache is exactly correct.

**Question 2 — does the output match token-for-token?** Not quite: **1 divergence
in 250 tokens (0.4%)**, at prompt 4 step 9, where the reference's own top-1/top-2
logit gap was **0.0234** — a coin flip.

The cause is not a bug, and isolating it was the interesting part. The cache's
entire purpose is to process **one token per step instead of the whole sequence**,
which changes the **shape of every matmul in the decode path**. Projecting the
*same* hidden state as `[1,36,896]` versus `[1,1,896]` against the 151936×896
output matrix differs by **7.81e-03**, because cuBLAS selects different kernels
for different shapes and those kernels accumulate in different orders.

> **Lesson:** different matmul shapes are *inherent* to caching, so a real
> inference engine cannot be bit-identical to its own uncached reference. The
> right correctness standard is: prove the logic exact where shapes match, then
> require that any token divergence be a demonstrated near-tie and that the
> divergence rate stay negligible — rather than loosening a tolerance until the
> test goes green.

### Results

| Batch | Phase 1 no cache | **Phase 2 KV cache** | HF `generate()` | vs Phase 1 | vs HF |
|---|---|---|---|---|---|
| 1 | 25.4 | **26.7** | 22.4 | 1.05× | 1.19× |
| 4 | 76.4 | **104.5** | 86.8 | 1.37× | 1.20× |
| 16 | 50.5 | **410.1** | 344.7 | 8.12× | 1.19× |
| 32 | 53.3 | **831.6** | 686.9 | **15.60×** | 1.21× |

**15.6× over our own Phase 1 at batch 32, and ahead of HuggingFace at every batch
size** — from an algorithmic change alone, with no custom kernels yet. That
ordering was deliberate: the largest available win was algorithmic, so it came first.

### Finding #7: the two phases have opposite bottlenecks, measured

| Batch | prefill ms | prefill tok/s | decode ms/step | decode tok/s |
|---|---|---|---|---|
| 1 | 41.7 | 1,006 | 37.3 | 26.8 |
| 4 | 45.2 | 3,717 | 38.4 | 104.1 |
| 16 | 61.0 | 11,013 | 38.3 | 418.3 |
| 32 | 72.7 | 18,485 | 38.7 | 826.8 |

**Decode is flat at ~38 ms/step across a 32× range of batch sizes** — memory-bound,
because the per-step cost is streaming weights and cache out of VRAM and that is
paid once per step no matter how many sequences ride along. **Prefill throughput
scales 18×** for a 1.7× increase in time — compute-bound, with the GPU actually
busy doing arithmetic.

Prefill moves tokens roughly **22× more efficiently than decode** (18,485 vs 827
tok/s at batch 32). That ratio is the economic argument for prefix-cache routing:
skipping a prefill is worth far more than speeding up a decode step, which is
exactly the bet MiniDynamo makes at the cluster level.

### Phase 3 headroom, quantified

Decode at batch 1 takes 37.3 ms to stream ~988 MB of fp16 weights — about
**26.5 GB/s, or 5.9% of this card's 448 GB/s peak**. The other 94% is lost to
PyTorch's per-operation overhead and unfused memory round-trips. That is the
target Phase 3's custom kernels aim at, now measured rather than assumed.

### Step 2 — Paged KV cache ✅

The contiguous cache reserves `max_seq` slots per sequence, so a 50-token
sequence still holds all 512 and no sequence can borrow another's spare room.
Paging fixes that the way an operating system does: one shared pool of
fixed-size **blocks**, a **block table** per sequence mapping logical positions
to physical blocks, and a **free-list allocator** handing out blocks on demand.

Storage is flattened to slots — `[num_blocks * block_size, kv_heads, head_dim]`
per layer — so a logical position resolves in one vectorized index:

```
slot = block_table[seq][p // block_size] * block_size + (p % block_size)
```

**Correctness:** paged prefill is **bit-identical** (0.00e+00) to the contiguous
cache. Paging changes where bytes live, not what is computed.

**Memory** (8 sequences of length 50–400, block_size 16, vs contiguous max_seq 512):
**3.8× fewer slots held** (1,072 vs 4,096), with **4.4% internal fragmentation**.
Waste is bounded by one partly-filled block per sequence rather than `max_seq`
per sequence.

### Finding #8 (the best debugging story): profile before optimizing

I predicted paging would be somewhat slower than contiguous, because gathering
scattered blocks materializes a copy every step. Measured at batch 32: **0.22×**
— paging cost 78% of throughput. Far worse than "somewhat," so it was worth
finding out why rather than accepting it.

Timing the two suspects separately (batch 32, length 160, per call):

| Operation | Time |
|---|---|
| `_slots` — rebuild block table from Python lists | **2.263 ms** |
| indexed read — the actual scattered gather | 0.071 ms |
| contiguous slice (no copy at all) | 0.018 ms |

**The gather was never the problem.** The scattered read cost 0.071 ms against a
plain slice's 0.018 ms — both negligible. **97% of the cost was Python**,
rebuilding the block-table tensor from lists on *every layer of every step*
(24 × 128 = 3,072 times per generation, and twice each for append and gather).

Two fixes followed directly from the measurement:

1. **Cache the device-side block table**, rebuilding only when blocks are actually
   allocated or freed. `_slots`: 2.263 → 0.295 ms, a **7.7×** improvement.
2. **Hoist slot computation out of the layer loop.** Slot indices and the
   attention mask depend only on positions, not on layer contents, so they are
   identical across all 24 layers. Now computed once per step (`SlotPlan`).

| Batch | paged before | paged after | gain | cost vs contiguous |
|---|---|---|---|---|
| 1 | 15.5 | **23.2** | 1.50× | 0.59× → **0.86×** |
| 4 | 53.1 | **92.2** | 1.74× | 0.51× → **0.85×** |
| 16 | 135.5 | **369.3** | 2.73× | 0.32× → **0.89×** |
| 32 | 183.7 | **664.9** | **3.62×** | 0.22× → **0.90×** |

Paging now costs ~10% instead of 78%, buying 3.8× better memory efficiency.

> **Lesson:** the intuitive culprit — memory traffic from scattered gathers — was
> 3% of the cost. The unglamorous one — Python executing inside the per-layer
> loop — was 97%. Profiling took ten minutes and redirected the entire fix.
> Optimizing the gather, as I had planned to, would have achieved nothing.

### Step 3 — Continuous batching ✅

Static batching runs a group of requests together and cannot start the next group
until the **last** member finishes. Real requests do not finish together, so
slots whose sequence completed keep decoding output nobody asked for — the batch
is held hostage by its slowest member. Continuous batching evicts a sequence the
moment it completes, returns its blocks to the free list, and admits a waiting
request into that slot on the next step.

**What this required of the model.** Sequences in a batch now sit at *different*
absolute positions — one admitted 40 steps ago is at position 60 while its
neighbour is at 3. Phase 1's `apply_rope` assumes one shared position range for
the whole batch, so `apply_rope_positions` was added and `forward_paged` now
indexes the rope tables per sequence rather than slicing a single range. The
paged cache already supported per-sequence block tables and padding masks, so it
needed no change — the earlier design paid off here.

**Why this needs its own benchmark.** The fixed-batch table cannot show this
benefit at all: it gives every sequence the same output length, so nothing
finishes early, there is no straggler, and both policies are identical by
construction. The win exists only with varied output lengths, so the measurement
uses a request stream with lengths drawn from a skewed distribution.

**Results** — 24 requests, output lengths 16–126 (total 1,018 tokens), max_batch 8,
both policies on the same engine and hardware:

| Policy | Wall time (s) | Decode steps | Tokens | Tokens/sec | Slot utilization | Wasted slot-steps |
|---|---|---|---|---|---|---|
| Static batching | 13.89 | 270 | 1,018 | 73.3 | 46.0% | 1,166 |
| **Continuous batching** | **8.98** | 163 | 1,018 | **113.4** | **100.0%** | **0** |

**1.55× throughput and 35.4% less wall time for identical output.** The
utilization column is the clearest statement of why: static spent **1,166
slot-steps — 54% of its capacity — decoding sequences that had already
finished**. Continuous batching wasted none. Both policies pay the same prefill
and per-token costs; only the admission rule differs.

This also closes the loop with the prefill/decode measurement above. Decode costs
a flat ~38 ms per step regardless of how many sequences ride along, so an empty
slot is pure waste — the fixed cost is paid whether or not the slot is doing
anything useful. Keeping the batch full is how that fixed cost gets amortized.

**Correctness** (4 tests): every request generates exactly what it generates when
run alone — including requests admitted mid-flight beside sequences at unrelated
positions; static and continuous produce identical tokens; every cache block
returns to the free list once the stream drains.

*A metric bug worth recording:* slot utilization first reported **116.7%**, which
is impossible. `useful_tokens` counted each request's prefill token while
`slot_steps` counted only decode steps. Fixed by excluding prefill from both
sides of the ratio. An impossible number is a gift — it fails loudly instead of
quietly overstating a result.

**Limitation, recorded not hidden:** prefill runs one request at a time rather
than batched or chunked, to avoid padding ragged prompts. A production engine
batches prefills; under a high admission rate that would become the bottleneck here.

### Phase 2 acceptance — all met

- [x] Paged KV cache: fixed blocks, per-sequence block table, free-block allocator
- [x] Prefill/decode split, with their opposite bottlenecks measured
- [x] Continuous batching, measured on a dynamic request stream
- [x] Output still matches Phase 1 (1 near-tie divergence in 250 tokens, root-caused)
- [x] Benchmark table vs HF at every batch size — ahead at all of them

---

## 7. Phase 3 — Custom CUDA kernels (in progress)

The measured target from Phase 2: decode achieves **26.5 GB/s, 5.9% of this
card's 448 GB/s peak**. The other 94% is lost to PyTorch's per-operation overhead
and unfused memory round trips. Phase 3 attacks that directly.

### Toolchain

`nvcc` 12.4 was installed via **conda-forge rather than the official NVIDIA
installer**, deliberately: the official installer bundles a display driver and
would have downgraded this machine's newer 610.62 driver for no benefit. Host
compiler is MSVC 14.44 (VS 2022 Build Tools). Three environment quirks — ninja
not on PATH, CUDA 12.4 rejecting the newer MSVC, and conda's import libraries
sitting in `Library/lib` where torch expects `lib/x64` — are all handled in the
kernel loader and written up in [HARDWARE.md](HARDWARE.md).

### Kernel 1 — Fused RMSNorm ✅

PyTorch runs RMSNorm as **five separate kernels**, each making a full round trip
through VRAM, to perform arithmetic worth about **1 FLOP per byte moved**. With
this card's roofline ridge at ~364 FLOP/byte, that is 364× below the point where
compute could ever matter — so essentially all of that traffic is waste.

The fused kernel does it in one trip: one thread block per row, coalesced strided
loads, a warp-tree reduction via `__shfl_down_sync` (threads trade values
directly through registers, never touching memory), a cross-warp combine through
shared memory, and a single write.

### Finding #10: an absolute tolerance was the wrong instrument

The first parity test used a 1e-3 absolute bound and failed at 0.00195. The
tempting move — the one this project has repeatedly refused — is to widen the
tolerance until it passes. Measuring instead showed:

| Metric | Value |
|---|---|
| Elements bit-identical | **99.997%** |
| Elements differing by exactly 1 ULP | 0.003% |
| Max ULP distance | **2** |
| Max relative error | 1.245e-03 (**1.27× fp16 epsilon**) |

The failing element had magnitude 2.19 — and at that magnitude **one ULP *is*
0.00195**. The kernel was as correct as fp16 permits; the tolerance was measuring
the wrong thing. A different summation order (warp-tree vs PyTorch's) cannot
produce the same last bit, because floating-point addition is not associative.

Parity is now stated in the hardware's own units: **max ULP distance ≤ 2, ≥99% of
elements bit-identical, max relative error ≤ 4ε**. That is *stricter* than the
absolute bound it replaced, and it cannot quietly widen as kernels get worse.

### Optimization: vectorized loads

Version 1 loaded one 2-byte half per thread per iteration, leaving each thread
with just 2 bytes in flight while the memory system wants far wider transactions.
Switching to `float4` loads (16 bytes = 8 halves) raised memory-level parallelism
8× for identical arithmetic:

| Shape | v1 scalar loads | v2 `float4` loads |
|---|---|---|
| 4096×896 | 48.8% of peak | **54.3%** |
| 16384×896 | 66.7% of peak (299 GB/s) | **75.5%** (338 GB/s) |

### Results

| Shape | PyTorch | Ours | Speedup | GB/s | % of peak | PyTorch % of peak |
|---|---|---|---|---|---|---|
| 1×896 | 197.9 µs | 30.3 µs | 6.53× | 0.1 | 0.0% | 0.0% |
| 1088×896 | 199.7 µs | 31.4 µs | 6.36× | 124 | 27.7% | 4.4% |
| 4096×896 | 367.5 µs | 60.3 µs | 6.09× | 243 | 54.3% | 8.9% |
| 16384×896 | 1329.0 µs | 173.6 µs | **7.66×** | **338** | **75.5%** | 9.9% |

Scored against the same compulsory-traffic ideal, PyTorch reaches at most **9.9%
of peak** — it moves roughly 7× the necessary bytes across five kernels. That gap
is the entire thesis of kernel fusion, measured.

**Two caveats stated plainly.** At small shapes the win is *launch overhead*, not
bandwidth: a 1×896 row is 3.6 KB, far too little to fill 46 SMs, and the 6.5×
there comes from making one call instead of seven. And **75.5% is good, not
maxed** — the remaining quarter is a real target, not rounding error.

### Kernel 2 — Fused SwiGLU ✅

Kernel 1 got **7.66×**, and the tempting conclusion is "hand-written CUDA is ~7×
faster than PyTorch." Kernel 2 is the control that shows that conclusion is
wrong. The win is never "CUDA"; the win is exactly the memory traffic removed,
and it can be predicted to within a few percent *before writing any code*.

The op is the gated valve in the middle of the MLP:

```
hidden = silu(gate) * up            silu(z) = z / (1 + e^-z)
```

Counting bytes per output element, PyTorch runs two kernels:

| | traffic |
|---|---|
| `F.silu(gate)` — read gate, write tmp | 4 B/elem |
| `tmp * up` — read tmp, read up, write out | 6 B/elem |
| **PyTorch total** | **10 B/elem** |
| **Compulsory minimum** — read gate, read up, write out | **6 B/elem** |

So the prediction, made before the kernel existed: **10/6 = 1.67×**. Not 7×.
RMSNorm was five round trips collapsed into one; this is two collapsed into one,
and the ratio of removed bytes is the whole story.

**Measured** (`bench/kernel_swiglu.py`, fp16, peak 448 GB/s, width 4864):

| Shape | PyTorch | Ours | Speedup | GB/s | % of peak | PyTorch, actual traffic |
|---|---|---|---|---|---|---|
| 1×4864 | 107.0 µs | 58.4 µs | 1.83× | 0.5 | 0.1% | 0.1% |
| 34×4864 | 105.4 µs | 58.5 µs | 1.80× | 17 | 3.8% | 3.5% |
| 1088×4864 | 158.0 µs | 100.5 µs | 1.57× | 316 | 70.6% | 74.8% |
| 4096×4864 | 518.0 µs | 312.5 µs | 1.66× | 383 | 85.4% | 85.9% |
| 16384×4864 | 1965.6 µs | 1192.8 µs | **1.65×** | **401** | **89.5%** | 90.5% |

**1.65× against a 1.67× prediction — within 1%.** The byte-counting model of the
machine is correct, which is a more valuable result than a bigger number would
have been.

Two things fall out of this that are worth more than the speedup:

**PyTorch's kernels are not inefficient — they just run twice.** Re-scored
against the 10 bytes it actually moves, PyTorch hits **90.5% of peak**, the same
as ours at 89.5%. Both implementations saturate the memory system. The entire
1.65× comes from deleting a round trip, not from out-coding anyone. Any claim
that a fused kernel "beats PyTorch" that cannot name the bytes it removed is a
claim about launch overhead.

**This kernel beats kernel 1's bandwidth utilization (89.5% vs 75.5%) while
being far simpler.** SwiGLU is embarrassingly parallel: a grid-stride loop, no
barriers, every SM independent. RMSNorm needs a block-wide reduction with two
`__syncthreads()` per row, so every thread waits on the slowest in its block,
and a 896-wide fp16 row is only 1792 bytes of work per block. **Synchronization,
not arithmetic, is what costs a memory-bound kernel its last 15% of peak.**

**Parity: 0 ULP, 100.000% exact, on every shape** — including saturating inputs
(`gate = ±60000`, where `exp(-z)` overflows to `inf` and the result must reach
zero by division rather than `NaN`) and real MLP activations from layers 0, 12,
and 23. This is Gotcha #8 again: elementwise means no reduction, no reordering,
so mirroring ATen's cast sequence op-for-op reproduces it bit-for-bit. It also
proves the build is using the accurate `expf` — a stray `--use_fast_math` or
`__expf` would have shown up instantly here, while still passing any absolute
tolerance loose enough to be called "close enough."

**Honest caveat:** below ~1000 rows the numbers measure dispatch overhead, not
the GPU. Our flat ~58 µs at 1×4864 and 32×4864 is Python + pybind + `empty_like`
cost, not memory bandwidth; the kernel itself is idle-fast at those sizes.

### Kernel 3 — Fused RoPE ✅

Rotary position embedding rotates each query/key vector by its position:

```
x_rot = x * cos + rotate_half(x) * sin        rotate_half([x1, x2]) = [-x2, x1]
```

Byte count first, before writing anything. PyTorch runs this as **five** kernels,
per element of `x` (E elements at 2 bytes; `cos`/`sin` are small and stay in L2):

| step | traffic |
|---|---|
| `-x2` — read E/2, write E/2 | 2E |
| `cat((-x2, x1))` — read E, write E | 4E |
| `x * cos` — read E, write E | 4E |
| `rotated * sin` — read E, write E | 4E |
| `t1 + t2` — read 2E, write E | 6E |
| **PyTorch total** | **20E** |
| **Ours** — read x once, write once | **4E** |

**Predicted ceiling: 5.00×.** Three times better than SwiGLU's, and the reason is
the interesting part: **6E of PyTorch's 20E bytes go to `rotate_half`, which
computes nothing.** It is pure plumbing — its only job is to present the operand
in a layout the next elementwise kernel can consume. A fused kernel replaces it
with an index offset. *The most profitable thing to fuse is usually not the
expensive math; it is the data movement wrapped around it.*

**Measured** (`bench/kernel_rope.py`, fp16, peak 448 GB/s):

| Shape | Elements | PyTorch | Ours | Speedup | GB/s | % of peak |
|---|---|---|---|---|---|---|
| 32×14×1×64 (decode b32, q) | 28,672 | 285.7 µs | 59.7 µs | 4.78× | 1.9 | 0.4% |
| 32×14×34×64 (prefill b32) | 974,848 | 148.3 µs | 30.1 µs | 4.94× | 130 | 29.0% |
| 32×14×512×64 | 14,680,064 | 835.8 µs | 166.1 µs | 5.03× | 354 | 78.9% |
| 32×14×2048×64 | 58,720,256 | 3231.5 µs | 598.3 µs | **5.40×** | **393** | **87.6%** |

**5.03× at the first genuinely bandwidth-bound size, against a 5.00× prediction.**

#### The 5.40× is above the ceiling, and that needed explaining

A measurement that beats its own ceiling means the model is wrong somewhere, so
it got checked rather than celebrated. Scoring each implementation against the
traffic it actually moves:

| Shape | ours, % of peak | PyTorch, % of peak | efficiency ratio | speedup |
|---|---|---|---|---|
| 32×14×34×64 | 29.0% | 29.3% | 0.99 | 4.94× |
| 32×14×512×64 | 78.9% | 78.4% | 1.01 | 5.03× |
| 32×14×2048×64 | 87.6% | 81.1% | **1.08** | 5.40× |

At every size but the largest the two implementations are equally efficient per
byte, and the speedup is the traffic ratio and nothing else — exactly as
predicted. At 32×14×2048×64 PyTorch's chain drops to 81.1% of peak while ours
holds 87.6%, and **5.00 × 1.08 = 5.40**. The excess is not our kernel doing
better than physics; it is PyTorch's chain doing worse than its own earlier self.
Most likely cause (measured effect, unproven cause): broadcast index arithmetic
in the two multiplies, plus allocator pressure from five 117 MB temporaries. The
honest summary is that the byte-counting model predicts the *floor* of the win,
not a bound on it.

#### The win at the sizes the engine actually runs is not bandwidth

RoPE operates on `q [b, 14, n, 64]` and `k [b, 2, n, 64]`. At decode `n = 1`, so
batch 32 is 28,672 elements — **57 KB**, nowhere near enough to fill 46 SMs. The
4.78× measured there is **five kernel launches becoming one**, not memory
efficiency, and the 0.4%-of-peak column says so plainly. Both regimes are in the
table on purpose; reporting only the 87.6% row would be the kind of flattering
benchmark this project exists to avoid.

#### Correctness: the failure mode that does not announce itself

RoPE can be wrong in a way that still runs. HF/Llama pairs dim `i` with
`i + head_dim/2`; the original RoPE paper's diagram pairs adjacent dims
`(0,1), (2,3), …`. Both produce finite, plausible tensors. A model built on the
wrong one still emits fluent text and is quietly wrong about position — nothing
raises, no tolerance catches it, and `build_rope_cache` duplicating frequencies
as `cat(freqs, freqs)` only makes sense under the halves convention.

So the pairing is asserted *directly*: with `cos = 0, sin = 1` the transform
collapses to exactly `rotate_half`, which distinguishes the two conventions
unambiguously. The test also asserts the adjacent-pair result is *not* produced.
A second, reference-independent property test checks that RoPE preserves each
head vector's norm — it is a rotation — which would catch a kernel that matched
PyTorch because both were wrong (max relative norm drift 1.87e-04, fp16 rounding).

**Parity: 0 ULP, 100.000% exact** on all 7 shapes, across **both** position
paths: shared positions (Phase 1) and per-sequence positions (Phase 2 step 3,
where one sequence sits at position 60 while its neighbour is at 3). A kernel
that indexed `cos`/`sin` by row instead of by `(batch, position)` passes the
first and fails the second, so both are tested. Also verified on non-contiguous
input — `attention()` produces q/k by transposing a view, so that is the normal
case — and on real q/k from layers 0, 12, 23.

### Kernel 4 — Fused decode attention with online softmax ✅

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

### Phase 3 wrap-up — wiring the kernels in, and what end-to-end actually shows

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

**Consistently 1.85–3.29×, never below 1.8×** — the direction was robust, but
the precise numbers were not, and saying so was the point:

- run-to-run spread reached **16–40%** against this project's 3% bar;
- the GPU was at **36–53% utilization** and holding 5.8 of 8 GB for other
  processes (Wallpaper Engine, Edge, Steam) throughout;
- decisively, `bench/phase2_cache.py` — the benchmark that originally produced
  the recorded 831/665 tok/s figures — **now times out after 10 minutes** on the
  same machine. The environment, not the code, changed.

An A/B ratio is more robust than an absolute figure here, because both sides
shared the same contention in the same process. But a 3× claim resting on runs
that disagree by 30% with each other is not a measurement, so it was recorded as
provisional and the caveat written into `results/phase3_end_to_end.md` itself —
a warning that only ever appeared on stdout does not survive being pasted into a
README.

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


### Phase 3 — complete

All four kernels written, correct, benchmarked, wired in, and verified end to
end at **2.33× over the PyTorch path (2.4× over HuggingFace)** at batch 32.

---

## Phase 4 — Quantization

### Step 1: the schemes, and what they cost in quality

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
| **INT8** | **22.2941** | **−0.55%** | 631 MB | 1.57× | 8.01 |
| INT4 g32 | 25.6600 | +14.47% * | 485 MB | 2.04× | 4.75 |
| INT4 g64 | 26.0086 | +16.02% * | 468 MB | 2.11× | 4.38 |
| INT4 g128 | 27.1472 | +21.10% * | 460 MB | 2.15× | 4.19 |

**The baseline is 22.4164 ± 3.45% (one standard error).** A perplexity delta
without an error bar is not interpretable, and this one earns its keep
immediately: INT8's −0.55% is *smaller than the sampling error*, so the honest
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
| fp16 | 988 MB | 1.00× | 16.00 | 57.9 | **1646.0** | 1030 MiB | 22.42 | — |
| **INT8** | 631 MB | 1.57× | 8.01 | **67.9** | 1274.9 | 682 MiB | 22.29 | −0.55% |
| INT4 | 460 MB | 2.15× | 4.19 | 67.1 | 829.1 | **519 MiB** | 27.15 | +21.10% |

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

---

## 8. What was learned

1. **Decode is memory-bound.** Flat ~51 ms inter-token latency across a 32× range
   of batch sizes is the measurement that proves it, and it explains why batching,
   quantization, and fused kernels all attack *memory traffic* rather than math.
2. **Small batches waste the GPU.** The full weight-streaming cost buys one token.
3. **Quadratic attention cost does not matter until it does.** Below ~512 tokens
   the O(seq²) term is invisible under weight streaming; past ~1024 it dominates.
   Optimizing for it earlier would have been optimizing the wrong thing.
4. **fp16 is deterministic**, so an implementation that mirrors a reference
   op-for-op matches bit-for-bit. Differences signal structural divergence.
5. **A correctness baseline must match the engine's own procedure** — same
   attention implementation, same cache behavior — or near-ties produce false
   failures. (The debugging story, §4 Finding #3.)
6. **Build the measuring tools first.** Every finding above came from Phase 0
   infrastructure that existed before there was anything to measure.
7. **A cache cannot be bit-identical to its uncached reference**, because
   processing one token instead of many changes every matmul's shape and
   therefore its fp16 rounding. Prove the logic exact where shapes match; hold
   token output to "any divergence must be a demonstrated near-tie."
8. **Take the algorithmic win before the hardware win.** The KV cache delivered
   15.6× at batch 32 with no GPU code. The measured 5.9%-of-peak bandwidth then
   tells us where the *next* win lives.
9. **Profile before optimizing — intuition about bottlenecks is unreliable.** The
   paged cache's slowdown was 97% Python overhead inside the layer loop and 3%
   the scattered memory access I had blamed. Fixing what I assumed was wrong
   would have gained nothing.
10. **Some wins are invisible to the wrong benchmark.** Continuous batching shows
    zero improvement on a fixed-batch table, because equal output lengths mean
    nothing ever finishes early. Measuring it required building a workload that
    actually contains the problem it solves.
11. **A metric that reports an impossible value is doing you a favour.** Slot
    utilization above 100% exposed a prefill/decode accounting mismatch that a
    merely-plausible number would have hidden.
12. **Measure the tolerance, don't tune it.** A kernel that failed a 1e-3
    absolute bound turned out to be correct to one ULP — the smallest difference
    fp16 can represent. The fix was to express correctness in the hardware's
    units, which produced a *stricter* test, not a looser one.
13. **For a memory-bound kernel, percentage of peak bandwidth is the honest
    metric.** "7.66× faster than PyTorch" flatters us; "75.5% of 448 GB/s" says
    how much room is actually left.
14. **A fusion's speedup is the bytes it removes, and it is predictable in
    advance.** SwiGLU was predicted at 1.67× by counting round trips before any
    code existed, and measured 1.65×. Re-scored against the traffic it really
    moves, PyTorch hits 90.5% of peak — as efficient as ours. It just runs twice.
    Any "beats PyTorch" claim that cannot name the removed bytes is a claim about
    launch overhead.
15. **Synchronization, not arithmetic, costs a memory-bound kernel its last 15%.**
    SwiGLU reaches 89.5% of peak with a barrier-free grid-stride loop; RMSNorm's
    block-wide reduction, with two `__syncthreads()` over a 1792-byte row, stalls
    at 75.5% despite doing less work per byte.
16. **The most profitable thing to fuse is the plumbing, not the math.** 6 of the
    20 bytes/element PyTorch spends on RoPE go to `rotate_half`, an op that
    computes nothing and exists only to rearrange an operand. Replacing it with
    an index offset is most of the 5× win.
17. **A result that beats its own predicted ceiling means the model is wrong, not
    that the kernel is heroic.** RoPE measured 5.40× against a 5.00× prediction;
    decomposing it showed 5.00× of traffic reduction times a 1.08× efficiency
    gap: at 117 MB tensors PyTorch's chain drops to 81.1% of peak while ours
    holds 87.6%. The byte count predicts the floor of a fusion win, not a bound
    on it.
18. **Some correctness failures do not announce themselves.** RoPE's two pairing
    conventions both produce finite, fluent-looking output; only one is right.
    Assert the convention directly (cos=0, sin=1 collapses the transform to
    `rotate_half`) and add a reference-independent property test (a rotation
    preserves vector norm) rather than trusting an end-to-end check to notice.
19. **Byte counting predicts a ceiling only for a kernel that is actually
    bandwidth-bound.** Kernels 1–3 landed within 1% of their byte-counted
    predictions. Decode attention predicted ~24× and delivered 2.48×, because it
    is latency-bound: holding KV volume fixed and quadrupling the block count
    left wall time flat. The prediction failing is what located the bottleneck.
20. **When a new implementation disagrees with a trusted reference, the
    instrument is a suspect too.** An fp64 "ground truth" said this kernel and
    PyTorch were equally wrong by 0.18. Two independent implementations agreeing
    with each other and not with the oracle indicts the oracle: `torch.softmax`
    in float64 on CUDA is wrong on this machine for any multi-row tensor (rows
    summing to 0.68 instead of 1.0).
21. **No error metric is universal.** ULP distance was exactly right for RMSNorm
    and is badly wrong for attention, where outputs pass through zero and fp16
    resolution becomes enormously fine — 2420 ULP at a flat 1.95e-03 absolute
    difference. Absolute error was wrong for RMSNorm and right here. Choose the
    metric from the value distribution, not from habit.
22. **Attribute a divergence, do not just bound it.** Enabling the kernels moved
    prefill logits by 4.10e-02. Reverting only RMSNorm to the reference — with
    the flag still on — dropped it to exactly zero, proving in one experiment
    that RMSNorm's reduction order was the entire cause, that SwiGLU and RoPE
    contribute nothing, and that prefill never reaches the decode kernel.
23. **Check what else is using the GPU before believing a benchmark.** Three runs
    of the same A/B disagreed by up to 30%, and the Phase 2 benchmark that
    produced this project's recorded figures timed out entirely, because the
    desktop was holding 36–53% of the card. The code had not changed; the
    environment had. On an idle GPU the same A/B gave 2.32–2.40× with the
    batch-size spread collapsing to 0.08×, and Phase 2 reproduced within its
    documented variance.
24. **A variance metric can be sample-size dependent, and this project's was.**
    `(max - min) / median` grows as you add runs, because you sample more of the
    tail — measured 6.4% at 5 runs and 13.9% at 9 on an idle GPU doing identical
    work. "Variance < 3%" is not reproducible without its run count. Report the
    coefficient of variation alongside it when comparing across run counts.
25. **A microbenchmark can measure the wrong thing for the regime you care
    about.** Kernels 1–3 were scored on bandwidth at large tensor sizes, which
    predicted they would contribute nothing at decode. End-to-end they clearly
    do: a decode step issues ~13 elementwise launches per layer, 24 layers per
    token, and collapsing those into 3 removes a per-step CPU cost no bandwidth
    argument captures.
26. **A metric delta without an error bar is not a result.** INT8 measured
    −0.55% perplexity against fp16 — a *lower* number, which reads as an
    improvement. The standard error of the estimate is ±3.45%, so the honest
    claim is "lossless within measurement precision" and nothing more. The same
    error bar is what makes INT4's +21% a real finding rather than a guess.
28. **Two schemes measuring the SAME speed is evidence, not a coincidence.**
    INT4 moves half the bytes of INT8 and ran at the same 1.8×. If either were
    bandwidth-bound, INT4 would be ~2× faster. Identical timings, plus a ~33 µs
    floor at every matrix size, said both were launch-bound — so the byte-counted
    2×/4× ceilings were never in play and quoting them as achieved would be wrong.
30. **Amdahl, measured.** The dequant-matmul is 1.8× in isolation and 1.17×
    end-to-end, because only 72.4% of weights are quantized and a decode step
    also spends time in attention, the cache, and four other kernels. Both
    numbers are correct; only one of them is the product.
29. **A degenerate-case test is worth writing precisely because guards get
    written carelessly.** A constant quantization group (hi == lo) made scale
    zero; the obvious guard substituted 1.0 and reconstructed the entire group
    as zero, destroying it silently. The test that caught it existed to check
    something else.
27. **Quote the compression you actually got, not the one the format implies.**
    "INT4" suggests 4×. The tensors touched shrink 3.82× once group metadata is
    counted (4.19 bits/weight, not 4.0), and the whole model shrinks 2.15×
    because the tied embedding stays fp16. Three different numbers, all true,
    routinely conflated.

---

## 9. Roadmap

| Phase | Status | Content |
|---|---|---|
| 0 — Ground truth | ✅ Complete | Hardware spec, benchmark harness, reference fixture, HF baseline |
| 1 — Correct but slow | ✅ Complete | Full forward pass from scratch, token-for-token match, no cache |
| 2 — KV cache & batching | ✅ Complete | Contiguous cache + prefill/decode split (15.6× at batch 32), paged cache (3.8× memory), continuous batching (1.55× on a request stream) |
| 3 — Custom CUDA kernels | ✅ Complete | RMSNorm 7.7× @ 75.5%; SwiGLU 1.65× @ 89.5%; RoPE 5.03× @ 87.6%; decode attention 2.48× @ 11.1%. **End-to-end 2.33× (2.4× vs HF) at batch 32** |
| 4 — Quantization | ✅ Complete | **INT8: lossless, 1.57× smaller, 1.17× tok/s at batch 1.** INT4: 2.15× smaller, 1.99× less VRAM, +21.1% ppl. Crossover reported |
| 5 — Make it legible | Planned | README benchmark table, architecture diagram, WRITEUP.md, limitations |

### Phase 2 specifics

- **Paged KV cache** — fixed-size blocks, a block table per sequence, a free-block
  allocator. Less fragmentation than one contiguous reservation per sequence, and
  it is the storage layer that MiniDynamo's prefix-affinity routing sits on top of.
- **Prefill / decode split** — prefill processes the whole prompt at once
  (compute-bound); decode does one token at a time against the cache
  (memory-bound). The two phases have opposite bottlenecks and want different code.
- **Continuous batching** — sequences join and leave the running batch as they
  finish, instead of the whole batch waiting for its slowest member.
- **Measurement plan** — the fixed-batch table (1/4/16/32) will not by itself
  capture continuous batching's benefit, because that benefit is about a *dynamic*
  workload. A request-stream simulation with varied output lengths, comparing
  continuous against static batching, is a separate Phase 2 deliverable.

---

## 10. Reproducing everything

```bash
conda create -y -n nano-infer -c conda-forge --override-channels python=3.12
conda activate nano-infer
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers safetensors tokenizers huggingface_hub datasets accelerate numpy pytest
```

| Command | What it does |
|---|---|
| `python -m tests.capture_reference` | Regenerate the correctness fixture (eager, no cache) |
| `python -m pytest tests/ -v` | All parity tests: components, full forward, greedy acceptance |
| `python -m bench.harness` | HuggingFace baseline at batch 1/4/16/32 |
| `python -m bench.phase1_nocache` | Growth curve + head-to-head vs baseline |
| `python -m bench.phase2_cache` | Prefill/decode split + Phase 1 vs Phase 2 vs HF |
| `python -m bench.phase2_continuous` | Static vs continuous batching on a request stream |
| `python -m bench.kernel_rmsnorm` | Fused RMSNorm vs PyTorch, with bandwidth utilization |
| `python -m bench.kernel_swiglu` | Fused SwiGLU vs PyTorch, with bandwidth utilization |
| `python -m bench.kernel_rope` | Fused RoPE vs PyTorch, with bandwidth utilization |
| `python -m bench.kernel_attention` | Fused decode attention; the two wins measured separately |
| `python -m bench.phase3_end_to_end` | End-to-end tokens/sec, kernels off vs on |
| `python -m bench.perplexity` | Quality cost of INT8/INT4 on WikiText-2, with error bars |
| `python -m bench.quant_speed` | Dequant-matmul speed and the batch-size crossover |
| `python -m bench.quant_acceptance` | **The Phase 4 acceptance table**: size, tok/s, perplexity, VRAM |

### Repository layout

```
nano_infer/     config.py, hf_ref.py, model.py, cache.py, engine.py — the engine
bench/          harness.py, phase1_nocache.py — measurement
tests/          capture_reference.py, test_parity.py, test_model.py, fixtures/
results/        committed benchmark outputs
```

### Commit history

The git history is deliberately incremental — one verified component per commit.

```
Phase 1 COMPLETE: assemble full model + greedy decode (matches HF token-for-token)
Phase 1: SwiGLU MLP (verified vs HF, bit-identical)
Phase 1: grouped-query attention (verified vs HF eager)
Phase 1: RoPE (verified vs HF, bit-identical)
Phase 1: RMSNorm (verified vs HF, bit-identical)
Phase 1: model skeleton + token embedding (verified vs HF)
Phase 0: benchmark harness + HF baseline — ground truth complete
Phase 0: capture HF reference fixture and parity tests
Phase 0: scaffold repo and record hardware ground truth
```
