# nano-infer — Roadmap & Handoff

**This file is the working state of the project.** It exists so a fresh Claude
Code session can pick the work up cold, with only `CLAUDE.md` (the build spec)
and this file.

- `CLAUDE.md` — the spec: what to build, the hard rules, the phase definitions.
  **Do not change its spec content** (it carries a pointer to this file at the top).
- `ROADMAP.md` (this file) — where we are, what's next, what will bite you. **Keep it updated.**
- `SUMMARY.md` — the portfolio narrative: findings, debugging stories, full results.
- `PROGRESS.md` — the dated running log.

---

## ⚠️ INSTRUCTIONS TO CLAUDE — READ FIRST

1. **Read `CLAUDE.md` before doing anything.** Its six hard rules govern this
   project (no `generate()`, no vLLM/FlashAttention, every claim measured,
   correctness gates speed, report regressions, incremental commits).
2. **Work strictly in phase order.** Do not start a phase before the previous
   one's acceptance criteria pass.
3. **YOU MUST UPDATE THIS FILE.** After every meaningful unit of work — a kernel
   landed, a benchmark run, a phase completed, a gotcha discovered — update:
   - the **Status at a glance** table,
   - the relevant **phase section** (check the box, record the numbers),
   - **Gotchas** if you hit something non-obvious,
   - **Next actions** so the next session starts with a correct first step.
   Update it *in the same commit as the work*, not later. A stale roadmap is
   worse than none, because the next session will trust it.
4. **Also update `SUMMARY.md` and `PROGRESS.md`** when a phase or major step
   completes. SUMMARY.md is the portfolio deliverable; keep its findings list and
   headline numbers current.
5. **Nathan is learning this, not just shipping it.** Explain the *why* before
   writing new code. Before each CUDA kernel, ask him to predict memory-bound vs
   compute-bound, then check against the measurement together.
6. **When something is slower than expected, profile it — do not work around it.**
   Two of this project's best results came from doing that (see Gotchas).
7. **Never loosen a tolerance to make a test pass.** Measure *why* it fails first.
   This has been the source of three separate real findings.

---

## Status at a glance

_Last updated: 2026-09-01, after Phase 3 kernel 3._

| Phase | Status | Headline |
|---|---|---|
| 0 — Ground truth | ✅ Complete | Harness (<3% variance), HF baseline, reference fixture |
| 1 — Correct but slow | ✅ Complete | From-scratch forward pass, token-for-token vs HF |
| 2 — KV cache & batching | ✅ Complete | 831 tok/s @ batch 32 = 15.6× vs Phase 1, 1.21× vs HF |
| 3 — Custom CUDA kernels | ◐ **3 of 4 kernels** | RMSNorm 7.66× @ 75.5%; SwiGLU 1.65× @ 89.5% (predicted 1.67×); RoPE **5.03×** @ **87.6%** (predicted 5.00×) |
| 4 — Quantization | ⬜ Not started | INT8 → INT4 group-wise + fused dequant-matmul |
| 5 — Make it legible | ⬜ Not started | README table, diagram, WRITEUP.md, limitations |

- **Tests:** 59 passing (`python -m pytest tests/ -q`)
- **Commits:** 17 on `main`, clean tree
- **Hardware:** RTX 3070, 8 GB, sm_86, **448 GB/s peak** (the Phase 3 denominator)

---

## Environment — how to run anything

System Python does not exist (Windows Store stubs only). **Always use the conda
env interpreter by full path**, and set `PYTHONPATH` to the repo root:

```bash
PY="C:/Users/Nathan stevens/miniconda3/envs/nano-infer/python.exe"
cd "C:/Users/Nathan stevens/OneDrive/Projects/nano-infer"
PYTHONPATH="$PWD" "$PY" -m pytest tests/ -q
```

PowerShell form:

```powershell
$env:PYTHONPATH="C:\Users\Nathan stevens\OneDrive\Projects\nano-infer"
& "C:\Users\Nathan stevens\miniconda3\envs\nano-infer\python.exe" -m pytest tests/ -q
```

| Command | What it does |
|---|---|
| `-m pytest tests/ -q` | Everything (~80 s) |
| `-m pytest tests/test_kernels.py -v -s` | CUDA kernel parity, prints ULP stats |
| `-m tests.capture_reference` | Regenerate the answer key (**eager + no cache**) |
| `-m bench.harness` | HF baseline, batch 1/4/16/32 |
| `-m bench.phase1_nocache` | Growth curve + Phase 1 vs HF |
| `-m bench.phase2_cache` | Prefill/decode split; Phase 1 vs contiguous vs paged vs HF |
| `-m bench.phase2_continuous` | Static vs continuous batching on a request stream |
| `-m bench.kernel_rmsnorm` | Fused RMSNorm vs PyTorch + bandwidth utilization |
| `-m bench.kernel_swiglu` | Fused SwiGLU vs PyTorch + bandwidth utilization |
| `-m bench.kernel_rope` | Fused RoPE vs PyTorch + bandwidth utilization |

Toolchain is installed and working: nvcc 12.4 (conda-forge), MSVC 14.44, ninja.
Full build notes in `HARDWARE.md`. Kernels JIT-compile on first use (~40 s), then cache.

---

## Repository layout

```
nano_infer/
  config.py      single source of truth: model, fp16 dtype, prompts, paths
  hf_ref.py      HF loading + generate() baseline (the ONLY place HF models are used)
  model.py       the engine. Phase 1 reference + Phase 2 cached/paged paths
  cache.py       KVCache (contiguous), PagedKVCache, BlockAllocator, SlotPlan
  engine.py      Request, RunStats, ContinuousBatchingEngine (static + continuous)
  kernels/
    __init__.py  JIT loader — handles ninja/MSVC/CUDA-libpath quirks
    bindings.cpp PYBIND11_MODULE for all kernels (one .cu cannot own it once
                 there are two — add new kernels here)
    rmsnorm.cu   kernel 1: fused RMSNorm (scalar + float4 vectorized paths)
    swiglu.cu    kernel 2: fused SwiGLU (scalar + float4, grid-stride)
    rope.cu      kernel 3: fused RoPE (both shared- and per-sequence positions)
bench/           harness.py + one benchmark per phase/kernel
tests/           parity + component + cache + paged + continuous + kernel tests
results/         committed JSON + markdown benchmark outputs
```

**`model.py` layering — do not break this:**
- Phase 1 functions (`forward`, `greedy_decode`, `attention`, …) are the
  **reference implementation**. They are the correctness answer key. Never
  "optimize" them; add new paths alongside.
- Phase 2 adds `*_cached` (contiguous) and `*_paged` (paged) paths.
- Phase 3 kernels are drop-in replacements validated against Phase 1's math.

---

## Gotchas that will bite you

These were all expensive to discover. Read before debugging anything.

1. **The reference fixture must match the engine's exact procedure.**
   `tests/capture_reference.py` uses **eager attention + `use_cache=False`**.
   HF's default SDPA differs from eager by ~0.1 logit over 24 layers, and
   cache-vs-no-cache by ~0.04 — either flips a near-tied token. If you regenerate
   the fixture, keep both settings.

2. **A cache cannot be bit-identical to its uncached reference.** Processing one
   token instead of a whole sequence changes every matmul's shape, cuBLAS picks a
   different kernel, and fp16 rounds differently (measured: 7.81e-03 on the *same*
   hidden state projected `[1,36,896]` vs `[1,1,896]`). **Standard used:** prove
   the logic exact where shapes match (`torch.equal` on hidden states), then
   require any token divergence to be a *demonstrated near-tie*. Currently 1
   divergence in 250 tokens (0.4%), reference top-1/top-2 gap 0.0234.

3. **Kernel parity is measured in ULP, not absolute error.** A 1e-3 absolute
   bound failed at 0.00195 — but 99.997% of elements were bit-identical and the
   rest differed by exactly one ULP, at a magnitude where one ULP *is* 0.00195.
   Bar is now: **max ULP ≤ 2, ≥99% exact, max relative error ≤ 4ε**. See
   `tests/test_kernels.py::compare`.

4. **Profile before optimizing.** The paged cache first ran at 0.22× throughput.
   The blamed cause (scattered gather copies) was **3%** of the cost; Python
   rebuilding the block table inside the per-layer loop was **97%**. Fixes gave
   3.62×. Reusable lesson: anything Python-side inside the 24-layer loop runs
   3,072× per generation.

5. **CUDA build quirks** (all auto-handled in `kernels/__init__.py`, documented in
   `HARDWARE.md`): ninja is not on PATH; CUDA 12.4 rejects MSVC 14.44 without
   `-allow-unsupported-compiler`; conda puts import libs in `Library/lib` where
   torch expects `lib/x64` (`LNK1181` on `cudart.lib`).

6. **Do NOT install the official NVIDIA CUDA toolkit.** It bundles a display
   driver and would downgrade this machine's newer 610.62 driver. Use conda-forge.

7. **`transformers` 5.x moved `rope_theta`** into `config.rope_parameters["rope_theta"]`.
   Top-level `config.rope_theta` no longer exists.

8. **fp16 is deterministic.** Mirroring a reference op-for-op gives bit-identical
   results. A nonzero diff means the *structure* diverged — that is information,
   not noise.

9. **Predict a fusion's speedup by counting bytes before writing it.** SwiGLU:
   PyTorch moves 10 B/elem (two kernels, one temporary), compulsory minimum is
   6 B/elem, so the ceiling is 1.67×. Measured 1.65×. If a kernel lands far from
   its byte-counted prediction, something else is happening — chase it. Corollary:
   at PyTorch's *actual* traffic it hits 90.5% of peak, so it is not inefficient,
   it just runs twice. Never claim a fusion "beats PyTorch" without naming the
   bytes removed.

10. **Barriers cost more than arithmetic in a memory-bound kernel.** Simpler
    SwiGLU (grid-stride, no `__syncthreads()`) reaches 89.5% of peak; RMSNorm's
    block-wide reduction over a 1792-byte row stalls at 75.5%. When kernel 1's
    remaining 25% gets revisited, the block/row mapping is the suspect, not the
    math.

11. **`PYBIND11_MODULE` lives in `kernels/bindings.cpp`, not in a `.cu`.** Only one
    translation unit may define it. Adding a kernel = new `.cu` + a declaration
    and a `def()` line in `bindings.cpp` + an entry in `sources` in
    `kernels/__init__.py`.

12. **RoPE pairs dim `i` with `i + head_dim/2`, NOT adjacent dims.** The
    adjacent-pair convention (from the original RoPE paper's diagram) produces a
    finite, plausible tensor and a model that emits fluent text while being
    quietly wrong about position. Nothing raises. `build_rope_cache` duplicates
    frequencies as `cat(freqs, freqs)` precisely because of this. Asserted
    directly in `test_rope_pairing_is_halves_not_adjacent` (cos=0, sin=1 collapses
    the transform to exactly `rotate_half`). **Anything touching RoPE — kernel 4
    included — must keep that test green.**

13. **RoPE has TWO position paths and a kernel can pass one while failing the
    other.** `apply_rope` takes `cos/sin` as `[n, head_dim]` (one shared range);
    `apply_rope_positions` takes `[batch, n, head_dim]` (per-sequence, which
    continuous batching requires). A kernel indexing by flat row rather than by
    `(batch, position)` passes the first and fails the second. Both are tested.

14. **A measurement that beats its predicted ceiling means the model is wrong,
    not that the kernel is heroic.** RoPE hit 5.40× against a 5.00× byte-counted
    ceiling. Decomposing: 5.00× traffic reduction × 1.08× because at 117 MB
    tensors PyTorch's chain falls to 81.1% of peak while ours holds 87.6%. Byte
    counting predicts the *floor* of a fusion win, not a bound on it. Always
    score both sides against their own actual traffic before believing a ratio.

---

## Progress detail

### Phase 0 — Ground truth ✅

Harness with `torch.cuda.synchronize()` before every timer stop and discarded
warmups. Variance 1.0–2.3% (bar: <3%). Reference fixture: 5 prompts × 50 tokens,
full step-0 logits + per-step top-5, 1.47 MB, committed.

HF baseline (fp16, 128 new tokens): **19.5 / 76.9 / 313.3 / 621.5 tok/s** at batch
1/4/16/32. Inter-token latency **flat at ~51 ms** across all batch sizes → decode
is memory-bound; batch 1 wastes the GPU.

### Phase 1 — Correct but slow ✅

Full forward pass from raw safetensors: embedding, RMSNorm, RoPE, GQA attention,
SwiGLU, decoder block (pre-norm, 2 residuals), 24 layers, final norm, tied logits.
Greedy decode, no cache. **Every component bit-identical (0.00e+00) to HF eager**;
acceptance met token-for-token on all 5 prompts.

Measured "before": forward cost is **flat ~38–40 ms from seq 32→512**
(weight-streaming-bound), then bends quadratic (1024: 77.9 ms, 4096: 692 ms).
vs HF: **1.13× at batch 1** (we win), **0.07× at batch 32** (13× slower).

Architecture (read from checkpoint, never assume): 24 layers, hidden 896,
intermediate 4864, **14 query / 2 KV heads (GQA)**, head_dim 64, vocab 151,936,
rms_eps 1e-6, rope_theta 1e6, **tied embeddings**, bias on q/k/v only.

### Phase 2 — KV cache and batching ✅

**Step 1 — contiguous cache + prefill/decode split.** K cached *after* RoPE.
Decode needs no causal mask. `forward_cached` projects only the last position.

| Batch | Phase 1 | Phase 2 | HF | vs P1 | vs HF |
|---|---|---|---|---|---|
| 1 | 25.4 | 26.7 | 22.4 | 1.05× | 1.19× |
| 32 | 53.3 | **831.6** | 686.9 | **15.60×** | 1.21× |

Prefill vs decode: **decode flat ~38 ms/step across a 32× batch range**
(memory-bound); prefill throughput scales 18× (compute-bound). Prefill moves
tokens ~22× more efficiently than decode → the economic case for prefix-cache
routing (MiniDynamo).

**Step 2 — paged cache.** Blocks + per-sequence block table + free-list allocator.
Bit-identical to contiguous. **3.8× fewer slots held, 4.4% internal
fragmentation.** After the profiling fix: 664.9 tok/s at batch 32 (~10% cost vs
contiguous, was 78%).

**Step 3 — continuous batching.** Required per-sequence RoPE positions
(`apply_rope_positions`). 24 requests, lengths 16–126:

| Policy | Wall | Tokens/sec | Slot utilization | Wasted slot-steps |
|---|---|---|---|---|
| Static | 13.89 s | 73.3 | 46.0% | 1,166 |
| **Continuous** | **8.98 s** | **113.4** | **100.0%** | **0** |

**1.55× throughput, 35.4% less wall time, identical output.**

### Phase 3 — Custom CUDA kernels ◐ (1 of 4)

**Target, measured:** decode achieves 26.5 GB/s = **5.9% of the 448 GB/s peak**.

**Kernel 1 — fused RMSNorm ✅.** One block per row, coalesced loads, warp-tree
reduction via `__shfl_down_sync`, cross-warp combine in shared memory, single
write. Optimized from 66.7% → **75.5% of peak** by switching scalar 2-byte loads
to `float4` (8 halves).

| Shape | PyTorch | Ours | Speedup | GB/s | % peak |
|---|---|---|---|---|---|
| 4096×896 | 367.5 µs | 60.3 µs | 6.09× | 243 | 54.3% |
| 16384×896 | 1329.0 µs | 173.6 µs | **7.66×** | **338** | **75.5%** |

PyTorch reaches at most 9.9% of peak on the same compulsory-traffic ideal.
Caveats stated: at small shapes the win is launch overhead, not bandwidth; 75.5%
is good but not maxed.

**Kernel 2 — fused SwiGLU ✅.** Grid-stride loop, `float4` vectorized from the
start, no barriers. **Ceiling predicted at 1.67× from byte-counting before any
code was written** (PyTorch 10 B/elem via a temporary vs 6 B/elem compulsory);
**measured 1.65×.**

| Shape | PyTorch | Ours | Speedup | GB/s | % peak | PyTorch @ actual traffic |
|---|---|---|---|---|---|---|
| 1088×4864 | 158.0 µs | 100.5 µs | 1.57× | 316 | 70.6% | 74.8% |
| 4096×4864 | 518.0 µs | 312.5 µs | 1.66× | 383 | 85.4% | 85.9% |
| 16384×4864 | 1965.6 µs | 1192.8 µs | **1.65×** | **401** | **89.5%** | 90.5% |

Parity **0 ULP, 100% exact** on every shape (elementwise ⇒ no reduction ⇒ no
reordering), including `gate = ±60000` saturation and real layer-0/12/23 MLP
activations. Two findings recorded as Gotchas #9 and #10.

**Kernel 3 — fused RoPE ✅.** One thread owns a rotation *pair* `(j, j+half)`, so
each element is read once and the rotation costs an index offset rather than a
tensor. Handles both position paths. **Predicted ceiling 5.00×** (PyTorch 20 B/elem
across five kernels vs 4 B/elem compulsory); **measured 5.03×** at the first
bandwidth-bound size.

| Shape | Elements | PyTorch | Ours | Speedup | GB/s | % peak |
|---|---|---|---|---|---|---|
| 32×14×1×64 (decode b32) | 28,672 | 285.7 µs | 59.7 µs | 4.78× | 1.9 | 0.4% |
| 32×14×34×64 (prefill b32) | 974,848 | 148.3 µs | 30.1 µs | 4.94× | 130 | 29.0% |
| 32×14×512×64 | 14.7 M | 835.8 µs | 166.1 µs | 5.03× | 354 | 78.9% |
| 32×14×2048×64 | 58.7 M | 3231.5 µs | 598.3 µs | **5.40×** | **393** | **87.6%** |

**6E of PyTorch's 20E is `rotate_half`, which computes nothing** — the win is
mostly deleted plumbing, not deleted math. **At the sizes the engine actually
runs this is launch-bound, not bandwidth-bound**: decode b32 q is 57 KB, and the
4.78× there is five launches becoming one (0.4% of peak says so). The 5.40×
overshoot is explained in Gotcha #14. Parity 0 ULP on all 7 shapes, both position
paths, non-contiguous input, and real q/k from layers 0/12/23; pairing asserted
directly (Gotcha #12) plus a reference-independent norm-preservation test.

---

## Next actions

### ▶ IMMEDIATE: Phase 3, kernel 4 — decode fused attention (the hard one)

Single query token against the whole cached KV, **online softmax** so the full
attention matrix is never materialized. Flash-decoding in miniature — write it
from scratch, do not copy FlashAttention. This is the kernel that targets the
**5.9%-of-peak decode** number directly, and it is the one that will actually
move end-to-end tokens/sec: kernels 1–3 all operate on tensors that are tiny at
decode time, while this one streams the entire KV cache.

**This kernel differs from 1–3 in kind, not just difficulty.** Those were
fusions — the win was byte-counted in advance and the reference was a few
elementwise ops. This one changes the *algorithm*: PyTorch materializes a
`[heads, 1, seq]` score matrix, softmaxes it, then multiplies by V. Online
softmax never materializes it, keeping a running max and running sum and
rescaling the accumulator as it goes. So:

1. **Ask Nathan to predict** the byte count again — but note it is harder here,
   because the score matrix is small at decode (one float per position per head)
   while the KV cache is large. Work out which term dominates. Also ask for the
   arithmetic intensity: this is the one kernel in the set with a real matmul in
   it, so it is the first where compute-bound is even a candidate answer.
2. **Get the online-softmax recurrence right on paper before writing CUDA.**
   Running max `m`, running sum `l`, rescale the accumulator by
   `exp(m_old - m_new)` at each block. Prototype the recurrence in Python against
   a plain softmax first — debugging it inside a kernel costs far more.
3. **Read the paged block table in place.** Paging currently pays a gather copy
   to build a contiguous KV view; this kernel should index blocks directly and
   remove it. That is a second, independent win — measure it separately from the
   softmax fusion rather than blurring both into one number.
4. Numerics: softmax needs the max subtraction for stability and the reference
   accumulates in fp32. Mirror the reference's cast points as with kernels 1–3,
   but **expect this to be the first kernel that is NOT 0 ULP** — the reduction
   order genuinely differs, exactly as it did for RMSNorm. Measure the ULP
   spread and report it; do not assume it, and do not loosen the bar (Gotcha #3).
5. Add `attention_decode.cu`; declare + `def()` in `bindings.cpp`; add to
   `sources` in `kernels/__init__.py`.
6. Parity in `tests/test_kernels.py`; microbenchmark `bench/kernel_attention.py`.
7. Commit, update this file + PROGRESS.md + SUMMARY.md.

### Then: Phase 3 wrap-up

- Wire kernels into `model.py` behind a flag (keep PyTorch paths for comparison).
- **End-to-end tokens/sec vs Phase 2** — the acceptance criterion. Individual
  kernel speedups do not automatically move the end-to-end number; measure it.
- Re-run `tests/test_cache.py` and `test_continuous.py` to confirm kernels did not
  change output beyond the documented near-tie.

### Phase 4 — Quantization

INT8 weight-only → INT4 group-wise (groups of 128, per-group scale + zero point).
Fused dequant-matmul: unpack in registers, **never materialize dequantized weights
in global memory** — that defeats the entire purpose. Measure perplexity on a
held-out WikiText-2 slice at fp16 vs INT8 vs INT4, plus side-by-side generations.
Acceptance table: model size GB, tokens/sec, perplexity, peak VRAM.

### Phase 5 — Make it legible

README opening with the benchmark table (nano-infer vs HF vs vLLM, same hardware/
model/prompts). One architecture diagram. `WRITEUP.md`, 800–1200 words on one
non-obvious lesson — **the strongest candidate is Gotcha #4** (paged cache: blamed
memory traffic, was 97% Python, found by profiling, 3.62× fix). A limitations
section.

### Known open items

- **MiniDynamo cross-link is a placeholder** in README.md (`https://github.com/`
  *(link TBD)*). Needs the real URL, and a reciprocal link from MiniDynamo.
- **Prefill is one request at a time** in `engine.py` (avoids padding ragged
  prompts). A production engine batches or chunks prefills; at high admission
  rates this would bottleneck. Recorded as a limitation, not hidden.
- **vLLM comparison row** for the Phase 5 README table has not been run.
- Benchmarks show run-to-run variance (contiguous batch-32 read 828 and 736
  tok/s on different runs). Prefer ratios over absolutes where possible, and
  state the variance.

---

## Reminder

**Update this file as part of the work, in the same commit.** Status table, phase
section, gotchas, next actions. The next session will trust whatever is here.
