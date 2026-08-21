# Progress log

A running, dated log of what actually got done in each phase. The git history is
the fine-grained record; this is the human-readable summary.

---

## Phase 0 — Ground truth (in progress)

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

### Acceptance criteria (not yet met)
- [ ] Benchmark harness produces a stable number across 3 runs (variance < 3%).
- [ ] Reference logits captured from HuggingFace and saved to disk.
- [ ] HuggingFace `generate()` baseline row in the results table.
