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

_Last updated: 2026-09-04, Phase 5 done + kernel 4b (head-group fusion) landed._

| Phase | Status | Headline |
|---|---|---|
| 0 — Ground truth | ✅ Complete | Harness (<3% variance), HF baseline, reference fixture |
| 1 — Correct but slow | ✅ Complete | From-scratch forward pass, token-for-token vs HF |
| 2 — KV cache & batching | ✅ Complete | 831 tok/s @ batch 32 = 15.6× vs Phase 1, 1.21× vs HF |
| 3 — Custom CUDA kernels | ✅ Complete | 4/4 kernels + **4b head-group fusion**. Decode attention **6.89× vs Phase 2 paged, 30.3% of peak** (was 2.52× / 11.1%). End-to-end 2.19–2.35× vs PyTorch |
| 4 — Quantization | ✅ Complete | **INT8 lossless, 1.57× smaller, 1.17× tok/s @ b1.** INT4 2.15× smaller, **1.99× less VRAM**, +21.1% ppl |
| 5 — Make it legible | ◐ **Nearly done** | README + benchmark table + mermaid diagram + limitations ✅, WRITEUP.md ✅. **Open: MiniDynamo link, vLLM row (blocked)** |

- **Tests:** 130 passing (`python -m pytest tests/ -q`)
- **Commits:** 26 on `main`, clean tree
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
| `-m bench.kernel_attention` | Decode attention; fusion, gather and head-group wins separated |
| `-m bench.kernel_attention --sharing` | **Is it DRAM-bound or issue-bound?** The experiment that reframed 11.1% |
| `-m bench.phase3_end_to_end` | **The Phase 3 acceptance number**: tokens/sec, kernels off vs on |
| `-m bench.perplexity` | Quality cost of INT8/INT4 on WikiText-2, with error bars |
| `-m bench.quant_speed` | Dequant-matmul speed; **the batch-size crossover** |
| `-m bench.quant_acceptance` | **The Phase 4 acceptance table** |

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
  quant.py       INT8 per-channel + INT4 group-wise; the quality reference
  kernels/
    __init__.py  JIT loader — handles ninja/MSVC/CUDA-libpath quirks
    bindings.cpp PYBIND11_MODULE for all kernels (one .cu cannot own it once
                 there are two — add new kernels here)
    rmsnorm.cu   kernel 1: fused RMSNorm (scalar + float4 vectorized paths)
    swiglu.cu    kernel 2: fused SwiGLU (scalar + float4, grid-stride)
    rope.cu      kernel 3: fused RoPE (both shared- and per-sequence positions)
    attention_decode.cu
                 kernel 4: online-softmax decode attention (one block per
                 query head) AND kernel 4b, head-group fused (one block per
                 KV head, n_rep query heads share every K/V read). The
                 dispatcher picks on block count; fuse_heads forces either.
    quant_matmul.cu
                 Phase 4: fused INT8/INT4 dequant-matmul, unpacked in registers
                 kernel 4: online-softmax decode attention, walks the paged
                 slot table in place. Block size is a tuned knob (see #17)
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

15. **`torch.softmax` in float64 on CUDA is WRONG on this machine** for any
    tensor with more than one row. torch 2.6.0+cu124 / RTX 3070: at `[16, 513]`
    the elementwise error vs CPU is 1.2e-02 and **rows sum to 0.68 instead of
    1.0**; `[1, 513]` is exact; fp32/fp16 unaffected. Any fp64 "ground truth"
    must be computed **on the CPU**. Pinned by
    `tests/test_kernels.py::test_fp64_softmax_on_cuda_is_unreliable`, which fails
    if torch ever fixes it. This cost real time: it made a correct kernel look
    wrong by 0.18. **Two independent implementations agreeing with each other and
    disagreeing with the oracle indicts the oracle.**

16. **No error metric is universal — pick it from the value distribution.** ULP
    distance was exactly right for RMSNorm (Gotcha #3) and is badly wrong for
    attention: outputs pass through zero, fp16 resolution near zero is enormously
    fine, so a flat 1.95e-03 absolute difference reads as 2420 ULP. Attention
    parity is stated against an **fp64 ground truth** instead: our error must be
    ≤ the fp16 reference's. That bar cannot be satisfied by being wrong in the
    same direction as PyTorch, which a plain "close to the reference" test would
    allow.

17. **Byte counting predicts a ceiling only for a kernel that is actually
    bandwidth-bound.** Kernels 1–3 landed within 1% of their predictions; decode
    attention predicted ~24× and delivered 2.48×. The decisive experiment: hold
    the KV pool fixed and vary only how many query heads share it — 2→8 query
    heads quadruples the blocks and leaves wall time **flat** (281.8 → 292.0 µs).
    Not bandwidth-bound, not duplicated-read-bound: **latency-bound**, because the
    online-softmax recurrence is sequential across tiles and each tile costs
    several `__syncthreads()`. The fix that followed (block size 128 → 512/1024)
    gave 7.9% → 11.1% of peak, and **3.6× at batch 1**, where there are only 14
    blocks so all latency hiding must come from inside the block.

---

18. **Check what else is using the GPU BEFORE trusting any benchmark on this
    machine.** The desktop (Wallpaper Engine, Edge, Steam) can hold 36–53% GPU
    utilization and ~5.8 of 8 GB. Under that load, three runs of the same A/B
    disagreed by up to 30% and `bench/phase2_cache.py` — which produced this
    repo's recorded 831/665 tok/s — **timed out after 10 minutes**. Nothing in
    the code had changed. **Idle** (0–2% util, ~350 MiB) the same A/B gives
    2.32–2.40× and Phase 2 reproduces within its documented variance.
    `bench/phase3_end_to_end.py` queries `nvidia-smi` and stamps what it saw
    into `results/*.md` itself, because a warning printed only to stdout does
    not survive being pasted into a README. **Target before publishing a number:
    utilization ~0%, memory under ~500 MiB.**

19. **Only TWO of the four kernels are bit-identical.** SwiGLU and RoPE are
    0 ULP / 100% exact (elementwise, no reduction, no reordering). **RMSNorm is
    not and cannot be** — warp-tree reduction order vs PyTorch's, bar ≤ 2 ULP
    (Gotcha #3) — and over 24 layers that compounds to a **4.10e-02** logit
    shift. Decode attention is not bit-identical either (Gotcha #16). A test
    that assumes "kernels 1–3 are exact" will fail; the first draft of
    `tests/test_end_to_end_kernels.py` did exactly that.

20. **Attribute a divergence, do not merely bound it.** Keeping the kernel flag
    ON while reverting *only* RMSNorm to the reference dropped that 4.10e-02 to
    **exactly zero** — proving in one experiment that RMSNorm was the whole
    cause, that SwiGLU/RoPE contribute nothing, and that prefill never reaches
    the decode kernel. Cheaper and far more convincing than a tolerance.

21. **The project's stability metric is sample-size dependent.** `bench/harness.py`
    uses `(max - min) / median`, which GROWS as you add runs — measured 6.4% at
    5 runs and 13.9% at 9, on an idle GPU doing identical work. So "variance
    < 3%" only means something with its run count attached, and adding repeats
    to "get a cleaner number" makes the reported figure worse.
    `bench/phase3_end_to_end.py` defaults to the harness's warmup 2 / runs 3 so
    the figure is comparable, and also reports **cv = stdev/mean**, which is
    sample-size stable. Use the cv when comparing runs with different repeats.

22. **WikiText-2 needs a namespaced dataset id now.** `load_dataset("wikitext",
    ...)` raises `HfUriError` on huggingface_hub >= 1.0, which requires
    `namespace/name`. Use `Salesforce/wikitext`, same data.

23. **Round the quantization scale to its STORED precision before choosing
    codes.** Computing a scale in fp32, picking codes against it, then storing
    the scale in fp16 breaks the half-step error guarantee — measured 535
    violations, because at q=127 an fp16 scale rounding shifts the
    reconstruction by ~0.06 of a step. Quantize with exactly the value
    dequantization will use. Free accuracy.

24. **Windows writes `results/*.md` as cp1252 unless told otherwise.** A `Δ` in
    a table header crashes `write_text`. Pass `encoding="utf-8"` on every
    artifact write.

25. **A quantized matmul beats cuBLAS only at batch 1-4 on this GPU.** Measured
    crossover: INT8/INT4 win up to batch 4 on the 4864x896 and 896x4864
    projections (batch 16 on 896x896), and lose hard above it — 0.23x at batch
    32, 0.04-0.09x at prefill. cuBLAS is FLAT across batch 1-32 because it is
    still weight-bound there and rides tensor cores; a scalar-fp32 dequant
    kernel cannot follow. Verified not to be register pressure: splitting batch
    32 into four BT=8 tiles gives 0.97x.

26. **INT8 and INT4 measure the SAME speed here, which proves neither is
    bandwidth-bound.** ~1.8x both, at 2.7-29.8% of peak, with an identical
    ~33 us floor at every matrix size (launch/dispatch overhead, the same floor
    kernels 1-3 hit at ~58 us). INT4 moves half of INT8's bytes; if bandwidth
    bound it would be ~2x faster. The byte-counted 2x/4x ceilings were never
    approached -- do not quote them as achieved.

27. **vLLM cannot be installed on this machine, and that is a documented dead
    end rather than an untried task.** vLLM publishes no Windows wheels; `pip
    install vllm` falls back to the sdist, which fails to unpack under Windows
    path-length limits (verified with `pip install --dry-run vllm`). Its
    supported platform is Linux. The README's vLLM row is deliberately empty and
    labelled -- do not fill it with a number from other hardware. To get one
    honestly, run the repo under WSL2 or on a Linux box and say so.

28. **"% of peak" is a ratio, and the numerator must be the bytes the hardware
    ACTUALLY MOVED — not the bytes your algorithm deserved to move.** Decode
    attention was recorded at 11.1% of peak and treated as the weakest number in
    the repo. That figure counts each KV element ONCE (compulsory bytes), but
    the kernel launched one block per query head, so with GQA 14q/2kv seven
    blocks each read the same KV rows. Against the traffic it actually issued it
    was at **71.9% of peak**. `bench/kernel_attention.py --sharing` is the
    experiment: hold block count and per-block work exactly constant, vary only
    how many query heads share a KV head, and wall time goes FLAT once n_rep ≥ 7
    (1199.9 → 729.3 → 694.2 µs for n_rep 1 → 7 → 14 while issued bytes stay at
    235 MB). Flat means the duplicates are cache hits, so DRAM was never the
    limit — but a cache hit still costs a load instruction and an issue slot.
    **Report both numbers, or the gap between them will read as inefficiency
    when it is actually the optimization you have not done yet.**

29. **THE DECODE LOOP IS CPU-DISPATCH-BOUND, NOT GPU-BOUND. This invalidates
    the intuition behind most of Phase 3.** Making decode attention 2.6× faster
    moved end-to-end `generate_paged` by 0.99–1.03×, i.e. not at all — measured
    back to back in one process with the call count verified at 1512. The
    evidence is two invariances that no GPU-bound loop can show:
      - step time is ~20 ms at batch 1 AND at batch 32 (32× the work, +7%);
      - step time is ~20 ms at context 33 AND at context 1025, even though the
        attention kernel alone costs 0.44 ms vs 8.33 ms per step on those exact
        tensors.
    The cause: **~3,200 aten dispatches per decode step** (identical count at
    context 33 and 1025), of which only 169 are `linear`. The rest is CPU-side
    bookkeeping — `as_strided` ×660, `view` ×393, `transpose` ×289, `reshape`
    ×245, `select` ×185. At a few µs of dispatch each that is the entire 20 ms.
    This is Gotcha #4 one level up. **Before optimizing any kernel further,
    measure whether the engine is waiting on the GPU at all.**

30. **Do not trust `torch.profiler` for a GPU-utilization RATIO here.** Summing
    `self_device_time_total` over `key_averages()` double-counts (parents plus
    children) and gives >100% utilization; and profiling inflates the CPU side
    so much that GPU-busy-per-step measured under the profiler is not comparable
    to unprofiled wall time. Both were tried and both produced 113–423%
    "utilization". The *counts* it reports are trustworthy; the timings, in this
    CPU-bound regime, are not. The invariance experiments above are what settled
    it, and they need no profiler.

31. **A "grouped" kernel trades traffic for parallelism, so it has a block-count
    floor.** Kernel 4b launches n_rep times FEWER blocks. Measured crossover on
    46 SMs: it loses below 8 blocks (0.64× at batch 1, which has 2) and wins
    above, growing to 2.9× at 64 blocks. Its optimal block size is also not a
    constant but a constant TOTAL thread count — `blocks × threads ≈ 32768`
    (~22 warps/SM, about half the 48-warp max) reproduces the measured optimum
    or comes within 9% at every point of a batch 4–64 × L 128–2048 sweep.

32. **A hand-built "obviously correct" kernel test can be measuring its own
    construction.** The first per-head-separation test for kernel 4b used a
    score gap of 7.5, which leaves 3.4% of the softmax mass on the non-target
    positions — enough to move the output by 0.137 and fail a 0.02 bar. The
    kernel was right; the test's premise ("this softmax is a delta") was not.
    Widening the gap to 3125 made the selection exact and the deviation 0.000.
    **Compute what your fixture actually implies before believing it indicts the
    code.**

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

### Phase 3 — Custom CUDA kernels ✅ (4 kernels + the head-group refit)

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

**Kernel 4 — fused decode attention with online softmax ✅.** Streams keys in
tiles carrying `(m, l, acc)`; never materializes the score matrix, never
materializes `repeat_kv`'s 7× copy, and walks the **paged slot table in place**
so the gather copy disappears. Recurrence prototyped in Python first (which also
measured what dropping the `exp(m_old-m_new)` correction costs: 2.881 — now a
regression test).

| Shape | Phase 2 paged | Pre-gathered | Ours | vs paged | GB/s | % peak |
|---|---|---|---|---|---|---|
| b1 L128 | 375.9 µs | 306.8 µs | 32.6 µs | **11.53×** | 2.0 | 0.4% |
| b32 L512 | 513.2 µs | 418.1 µs | 200.5 µs | 2.56× | 41.8 | 9.3% |
| b32 L2048 | 1755.3 µs | 1364.6 µs | 677.0 µs | **2.59×** | **49.6** | **11.1%** |

At batch 32, context ≥ 512: **1.99× from the fusion, 1.24× from removing the
gather** — measured separately on purpose. **Predicted ~24×, got 2.48×**; the
miss is the finding, see Gotcha #17 — and then see Gotcha #28, which shows the
denominator in "11.1% of peak" was the wrong one and led to kernel 4b below. **11.1% of peak is not good** (Phase 2
decode was 5.9%, so this roughly doubles it) and the two identified, un-done
optimizations are split-K/flash-decoding and collapsing the 7 query heads that
share a KV head into one block. First kernel that cannot be bit-identical —
correctness argued against an fp64 CPU ground truth (Gotchas #15, #16); our error
is 0.48–1.00× the reference's on every shape.

**Kernel 4b — head-group fusion ✅.** One block per (sequence, KV head) instead
of per (sequence, query head), so the n_rep = 7 query heads sharing a KV head
read each K and V element ONCE, into a register, and feed it to all seven. Issued
traffic falls 7×; arithmetic is unchanged, so intensity rises from 0.5 to 3.5
FLOP/byte. Barriers per tile are unchanged (7 vs 6) but each now covers 7× the
work.

Written because Gotcha #28's experiment showed the per-query-head kernel was at
71.9% of peak on *issued* traffic — near its ceiling on a load path carrying 7×
more than the algorithm needs. The fix was never "go faster", it was "ask for
less".

| Shape | Phase 2 paged | Pre-gathered | Per-query-head | **Grouped** | Fusion win | vs paged | GB/s | % peak |
|---|---|---|---|---|---|---|---|---|
| b1 L128 | 373.8 | 305.2 | 32.8 | 35.0 | 0.94× | 11.40× | 2.0 | 0.4% |
| b32 L128 | 380.2 | 310.0 | 61.4 | **38.9** | 1.58× | 9.77× | 53.9 | 12.0% |
| b32 L512 | 514.5 | 415.7 | 200.1 | **80.5** | 2.48× | 6.39× | 104.1 | 23.2% |
| b32 L1024 | 899.1 | 743.7 | 352.0 | **136.6** | 2.58× | 6.58× | 122.8 | 27.4% |
| b32 L2048 | 1701.1 | 1355.0 | 674.0 | **247.0** | **2.73×** | **6.89×** | **135.9** | **30.3%** |

**11.1% → 30.3% of peak; 2.52× → 6.89× vs the Phase 2 paged path.** Numerically
it is not a trade: error against the fp64 CPU ground truth is *identical* to the
per-query-head kernel's on all 8 test shapes, and both are 0.42–1.00× the fp16
reference's. Per-head separation asserted directly (delta-softmax construction,
deviation 0.000e+00), plus block-size invariance and a loud failure if
`fuse_heads=1` is asked for where n_rep == 1.

**What it costs:** n_rep times fewer blocks, so it loses below 8 blocks — 0.64×
at batch 1, which has only 2 for 46 SMs. The dispatcher picks on block count
(Gotcha #31); `fuse_heads` (0 auto / 1 grouped / -1 per-head) forces either, so
the benchmark A/Bs the two rather than reporting whatever auto chose.

**And it buys nothing end to end** — see Gotcha #29. That is the more important
result of the two.

---

## Next actions

### ▶ IMMEDIATE: the decode loop is CPU-bound (Gotcha #29)

**This is now the most valuable open problem in the repo, and it is worth more
than any remaining kernel work.** The engine issues ~3,200 aten dispatches per
decode step; only 169 are matmuls. Step time is invariant to batch size and to
context length, which means the GPU is idle waiting for Python. Every kernel in
Phase 3 is therefore optimizing a part of the step that is not the bottleneck.

Concrete first moves, cheapest first:
1. **Count and cut the metadata ops.** `as_strided` ×660, `view` ×393,
   `transpose` ×289, `reshape` ×245 per step. Much of this is the
   `view/transpose/contiguous` dance around attention; some is per-sequence
   indexing (`select` ×185 at batch 32 vs absent at batch 1 — find that loop).
2. **CUDA graphs for the decode step.** The shape is fixed across steps, which
   is exactly the case graphs exist for; it would collapse ~3,200 dispatches
   into one replay. This is what production engines do and it is the single
   highest-leverage change available.
3. Re-measure the Phase 3 kernels' end-to-end contribution afterwards. They
   should finally show up.

The old measurement to keep honest: `bench/phase3_end_to_end.py` reports
2.19–2.35× kernels-on vs kernels-off. That gap is real but it is mostly RMSNorm
and SwiGLU on the *prefill*, not decode attention.

### Then: finish Phase 5 — two open items

Phases 0-4 are complete and Phase 5 is mostly done: `README.md` now opens with
the benchmark table (every row citing the script that reproduces it), a mermaid
architecture diagram, the quantization trade-off table, and a limitations
section. `WRITEUP.md` is written -- 1,100 words on the decode-attention ceiling
miss, the controlled experiment that diagnosed it as latency-bound, and the
block-size fix that followed.

Two things remain, and **neither is code**:

1. **The MiniDynamo cross-link is still a placeholder** in `README.md` line 7:
   `[MiniDynamo](https://github.com/) *(link TBD)*`. Nathan has the real URL.
   Also add the reciprocal link from MiniDynamo's README back here -- the two
   repos are meant to read as one story ("router down to the CUDA kernel"), and
   that only works if a reader can get from either to the other.

2. **The vLLM row is blocked, not pending** (Gotcha #27). vLLM has no Windows
   wheels and its sdist will not unpack here. Options, in order of honesty:
   (a) leave it labelled as it is now -- already done, and defensible;
   (b) run the repo under WSL2 or on a Linux machine and produce a real row;
   (c) do NOT quote a vLLM number measured on other hardware.

### Optional, if the project continues

**Split-K for kernel 4b.** The head-group fusion (the second of Gotcha #17's two
named fixes) is now done: 30.3% of peak, 6.89× vs Phase 2 paged. **Split-K is
still not**, and it is exactly what the grouped kernel needs at small batch,
where it has too few blocks to fill the card and loses to the per-query-head
path. Partition L across blocks, each producing a partial `(m, l, acc)`, then
combine — real flash-decoding. It would let the grouped kernel win at batch 1
too and retire the dispatcher's block-count fallback.

**A tensor-core quantized matmul.** The INT4/INT8 kernel loses above batch ~4
because it accumulates in scalar fp32 while cuBLAS rides HMMA (Gotcha #25).
Dequantizing into fp16 fragments and issuing tensor-core instructions is the
real fix, and a much larger kernel.

**Scale to a bigger model.** CLAUDE.md suggests Qwen2.5-1.5B or Llama-3.2-1B for
the headline numbers if VRAM allows. Two conclusions here are explicitly
0.5B-specific and would likely change: INT4's 21% perplexity cost (larger models
have more redundancy and quantize better), and the batch-4 crossover.

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
- **The decode loop is CPU-dispatch-bound** (Gotcha #29) — the largest open
  problem, promoted to the immediate next action above.
- **Split-K is still unimplemented**, so kernel 4b falls back to the
  per-query-head kernel below 8 blocks.

---

## Reminder

**Update this file as part of the work, in the same commit.** Status table, phase
section, gotchas, next actions. The next session will trust whatever is here.
