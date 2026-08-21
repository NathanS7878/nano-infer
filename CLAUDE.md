# nano-infer — build spec

## What this is

A single-GPU LLM inference engine written from scratch, with custom CUDA kernels and
INT8/INT4 quantization. It is a portfolio project for NVIDIA, Microsoft CoreAI, OpenAI,
Databricks, and AWS internship applications.

The author (Nathan) already built **MiniDynamo** — a distributed KV-cache-aware LLM
inference *router* in Rust + Python, modeled on NVIDIA Dynamo. That project handles the
*distributed* layer: routing requests across workers by cached-prefix affinity.

**nano-infer is the layer below it.** MiniDynamo decides *which worker* runs a request.
nano-infer is what a worker actually does on the GPU. Together they tell one coherent
story: "I understand LLM serving from the request router down to the CUDA kernel."

Keep them as separate repos. Cross-link them in both READMEs.

## The one-sentence goal

Load an open-weights transformer, generate tokens **without** `model.generate()`, vLLM,
TensorRT-LLM, or FlashAttention, and beat a naive PyTorch baseline by a measured,
honestly-reported margin.

---

## Hard rules — do not violate these

1. **No `transformers.generate()`.** HuggingFace is allowed for *downloading weights and
   the tokenizer only*. The forward pass, sampling loop, and cache are ours.
2. **No vLLM, TensorRT-LLM, FlashAttention, xformers, or ready-made fused kernels.** We
   benchmark *against* them. We do not import them.
3. **Every performance claim needs a number, a methodology, and a hardware spec.** No
   claim goes in the README that isn't reproducible by running a script in the repo.
4. **Correctness gates every optimization.** A kernel that is fast and wrong is worth
   zero. Numerical parity tests run before speed is ever measured.
5. **Report where it gets worse.** Quantization degrades quality. Small batches waste
   the GPU. Say so, with numbers. This is the single thing that separates this repo from
   the thousands of resume-padding repos it will be compared against.
6. **Commit incrementally with real messages.** The git history is part of the artifact —
   an interviewer may read it. No single "initial commit" dump.

---

## Target model and hardware

- **Dev model:** `Qwen/Qwen2.5-0.5B-Instruct` — small enough to iterate fast, real enough
  to be a genuine transformer (GQA, RoPE, RMSNorm, SwiGLU).
- **Benchmark model:** scale to `Qwen2.5-1.5B-Instruct` or `Llama-3.2-1B` for the headline
  numbers if VRAM allows.
- **Hardware:** author has an NVIDIA GPU on a secondary machine. First task is to record
  the exact card, VRAM, driver, and CUDA toolkit version into `HARDWARE.md`, because every
  benchmark number in this repo is meaningless without it.

Ask before assuming the architecture. Different Qwen/Llama versions differ in RoPE scaling
and attention head grouping.

---

## Phases

Work strictly in order. Do not start a phase until the previous phase's acceptance
criteria pass. Each phase ends with a commit and a note in `PROGRESS.md`.

### Phase 0 — Ground truth (target: 2 days)

Before writing anything clever, build the thing that will tell us if we broke something.

- Record hardware into `HARDWARE.md` (`nvidia-smi`, `nvcc --version`, torch version).
- `bench/harness.py`: a benchmark harness that measures **time-to-first-token (TTFT)**,
  **inter-token latency**, and **tokens/sec**, at batch sizes 1, 4, 16, 32, with proper
  CUDA synchronization and warmup iterations. Naive timing around async CUDA calls is
  the classic beginner error — do not make it.
- `tests/test_parity.py`: given a prompt, capture HuggingFace's logits for the first N
  tokens and store them as the reference fixture everything else is compared against.
- Baseline row in the results table: HuggingFace `generate()`, unmodified.

**Acceptance:** harness produces a stable, repeatable number across three runs (variance
under 3%). Reference logits saved to disk.

### Phase 1 — Correct but slow (target: 3 days)

- `nano_infer/model.py`: the full forward pass in plain PyTorch. Embedding, RMSNorm, RoPE,
  grouped-query attention, SwiGLU MLP, final projection.
- Weight loading from the safetensors checkpoint, mapped onto our own module names.
- Greedy decode loop. No cache yet — recompute the whole sequence each step. This is
  deliberately the slow, obviously-correct version.

**Acceptance:** token-for-token identical output to HuggingFace greedy decode for at least
5 different prompts, 50 tokens each. Max absolute logit difference under 1e-3 in fp16.

### Phase 2 — KV cache and batching (target: 3 days)

- Paged KV cache — fixed-size blocks, a block table per sequence, a free-block allocator.
  Nathan already reasoned about prefix-cache affinity in MiniDynamo; this is the storage
  layer that idea sits on top of.
- Split prefill (compute-bound, whole prompt at once) from decode (memory-bound, one token
  at a time). Understanding *why* these two phases have different bottlenecks is the point.
- Continuous batching: sequences join and leave the running batch as they finish, instead
  of waiting for the slowest one.

**Acceptance:** output still matches Phase 1 exactly. Benchmark table now shows our
tokens/sec at each batch size vs the HF baseline. Expect a large win here — this is the
algorithmic optimization, before we touch a single kernel.

### Phase 3 — Custom CUDA kernels (target: 8 days) ← the centerpiece

This is the phase that makes the project worth doing. Write real `.cu` files, bind them
with `torch.utils.cpp_extension.load` or a small setup.py.

Write these four, in this order — easiest to hardest:

1. **Fused RMSNorm.** The "hello world" of CUDA kernels. One block per row, warp-level
   reduction for the sum of squares, fused multiply by the weight. Teaches reductions and
   shared memory.
2. **Fused SwiGLU.** `silu(gate) * up`, elementwise, fused to avoid a round-trip to global
   memory. Teaches why memory traffic, not math, is usually the bottleneck.
3. **RoPE application kernel.** Fused into the QKV projection output. Teaches indexing
   and layout.
4. **Decode-phase fused attention.** The hard one. Single query token against the whole
   cached KV. Online softmax so you never materialize the full attention matrix. This is
   flash-decoding in miniature — write it yourself, do not copy FlashAttention.

For **each** kernel, in this order:
- Numerical parity test against the PyTorch version first (`tests/test_kernels.py`).
- Then a microbenchmark isolating just that op.
- Then profile it. Report **achieved memory bandwidth as a percentage of the card's
  theoretical peak.** Use `torch.profiler` or Nsight Compute.

That last bullet is the differentiator. Anyone can write a kernel that runs. Being able to
say "this decode kernel hits 78% of peak bandwidth, and it's memory-bound not compute-bound,
here's the arithmetic intensity calculation" is what a real GPU engineer sounds like — and
it is exactly what NVIDIA's project deep-dive round probes for.

**Acceptance:** all four kernels numerically correct. Each has a measured speedup and a
bandwidth-utilization number. End-to-end tokens/sec improved over Phase 2.

### Phase 4 — Quantization (target: 4 days)

- **INT8 weight-only**, then **INT4 group-wise** (groups of 128, per-group scale + zero point).
- A fused dequantize-and-matmul kernel: read packed INT4 weights, unpack in registers,
  multiply in fp16. Never materialize the dequantized weight matrix in global memory —
  the entire point is to cut memory traffic, and materializing it defeats that.
- **Measure the quality cost, do not hide it.** Perplexity on a held-out slice of WikiText-2
  at fp16 vs INT8 vs INT4. Plus a handful of qualitative generations side by side.

**Acceptance:** a table showing, for each precision: model size in GB, tokens/sec,
perplexity, and peak VRAM. The tradeoff should be visible and discussed in plain language.

### Phase 5 — Make it legible (target: 2 days)

The repo is worthless to a recruiter who can't see the result in 30 seconds.

- **README** opening with the benchmark table: nano-infer vs HuggingFace vs vLLM, same
  hardware, same model, same prompts. State the methodology. Link to the script that
  reproduces it.
- **One architecture diagram.** Request in, through the cache, through the kernels, token out.
- **`WRITEUP.md`** — 800–1200 words on one non-obvious thing that was learned. The strongest
  version of this is a failure: a kernel that was slower than PyTorch until you found the
  uncoalesced memory access, with the before/after profile. Interviewers remember the
  engineer who debugged something real. They forget the one who lists features.
- A short **limitations** section. What doesn't work, what wasn't tried, what would come next.

---

## Working style for Claude Code

- Explain the *why* before writing code for anything new — Nathan is learning this, not
  just shipping it. A kernel he can't explain is worse than useless in an interview,
  because he will be asked to explain it.
- After each kernel, ask him to predict whether it's memory-bound or compute-bound, then
  check against the profiler together.
- Prefer many small, verifiable steps over large code dumps.
- When something is slower than expected, **debug it with him rather than working around
  it.** The debugging story is the most valuable output of this entire project.
