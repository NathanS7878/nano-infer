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

### Remaining in Phase 3

- Kernel 4 — decode fused attention with online softmax (flash-decoding in miniature)
- End-to-end tokens/sec improvement over Phase 2

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

---

## 9. Roadmap

| Phase | Status | Content |
|---|---|---|
| 0 — Ground truth | ✅ Complete | Hardware spec, benchmark harness, reference fixture, HF baseline |
| 1 — Correct but slow | ✅ Complete | Full forward pass from scratch, token-for-token match, no cache |
| 2 — KV cache & batching | ✅ Complete | Contiguous cache + prefill/decode split (15.6× at batch 32), paged cache (3.8× memory), continuous batching (1.55× on a request stream) |
| 3 — Custom CUDA kernels | ◐ In progress | ✅ Fused RMSNorm (7.7×, 75.5% of peak). ✅ Fused SwiGLU (1.65× vs 1.67× predicted, 89.5% of peak). ✅ Fused RoPE (5.03× vs 5.00× predicted, 87.6% of peak). Remaining: decode attention with online softmax |
| 4 — Quantization | Planned | INT8 weight-only, then INT4 group-wise (g=128) + fused dequant-matmul. Perplexity cost measured on WikiText-2 |
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
