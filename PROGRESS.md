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

---

## Phase 2 — KV cache and batching (not started)
