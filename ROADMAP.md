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

_Last updated: 2026-09-01, after Phase 3 kernel 2._

| Phase | Status | Headline |
|---|---|---|
| 0 — Ground truth | ✅ Complete | Harness (<3% variance), HF baseline, reference fixture |
| 1 — Correct but slow | ✅ Complete | From-scratch forward pass, token-for-token vs HF |
| 2 — KV cache & batching | ✅ Complete | 831 tok/s @ batch 32 = 15.6× vs Phase 1, 1.21× vs HF |
| 3 — Custom CUDA kernels | ◐ **2 of 4 kernels** | RMSNorm 7.66× @ 75.5% of peak; SwiGLU 1.65× @ **89.5% of peak** (predicted 1.67×) |
| 4 — Quantization | ⬜ Not started | INT8 → INT4 group-wise + fused dequant-matmul |
| 5 — Make it legible | ⬜ Not started | README table, diagram, WRITEUP.md, limitations |

- **Tests:** 41 passing (`python -m pytest tests/ -q`)
- **Commits:** 16 on `main`, clean tree
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

---

## Next actions

### ▶ IMMEDIATE: Phase 3, kernel 3 — fused RoPE

Applied to the QKV projection output. Teaches indexing and layout — there is no
reduction and no fusion-of-round-trips story here, so the interesting part is
getting the memory access pattern right on a 4-D `[batch, heads, seq, head_dim]`
view.

1. **Ask Nathan to predict** memory- vs compute-bound, and the byte-counted
   ceiling, *before* writing it — that worked well on kernel 2. Setup: reads q
   (or k), reads `cos`/`sin`, writes the rotated result. PyTorch runs this as
   several elementwise kernels plus a `torch.cat` for `rotate_half`, and that
   `cat` materializes a full extra tensor — count what it costs.
2. Must match `rotate_half`'s pairing: dim `i` with `i + head_dim/2` (**not**
   adjacent pairs — this is the single most common way to get RoPE wrong).
3. Must support the **per-sequence position path** added in Phase 2 step 3
   (`apply_rope_positions`), not just a contiguous 0..seq range. Continuous
   batching depends on it.
4. Add `rope.cu`; declare + `def()` in `kernels/bindings.cpp`; add to `sources`
   in `kernels/__init__.py`.
5. Parity test in `tests/test_kernels.py` with the existing `assert_parity`.
   Expect 0 ULP as with SwiGLU *if* the cast order is mirrored exactly —
   `cos`/`sin` are fp32 in the reference, so check where the cast back to fp16
   happens.
6. Microbenchmark `bench/kernel_rope.py` mirroring `kernel_swiglu.py`, reporting
   GB/s and % of 448 peak, plus the predicted-vs-measured ceiling.
7. Commit, update this file + PROGRESS.md + SUMMARY.md.

### Then: kernel 4 — decode fused attention (the hard one)

Single query token against the whole cached KV, **online softmax** so the full
attention matrix is never materialized. Flash-decoding in miniature — write it
from scratch, do not copy FlashAttention. This is the kernel that targets the
5.9%-of-peak decode number directly, and it should read the **paged** block table
in place (removing the gather copy that paging currently pays).

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
