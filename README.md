# nano-infer

A single-GPU LLM inference engine written from scratch — custom CUDA kernels and
INT8/INT4 quantization — that loads an open-weights transformer, generates tokens
**without** `model.generate()`, vLLM, TensorRT-LLM, or FlashAttention, and beats a
naive PyTorch baseline by a measured, honestly-reported margin.

It is the GPU-worker layer beneath [MiniDynamo](https://github.com/) *(link TBD)*,
a distributed KV-cache-aware inference router. MiniDynamo decides *which worker*
runs a request; nano-infer is what that worker actually does on the GPU.

> **Status:** Phases 0–1 complete (from-scratch forward pass matches HuggingFace
> token-for-token). **Full project write-up: [SUMMARY.md](SUMMARY.md)** — what was
> built, what was measured, and what went wrong. Running log: [PROGRESS.md](PROGRESS.md).

## Hardware

All numbers in this repo come from one machine: **NVIDIA GeForce RTX 3070** (8 GB,
Ampere sm_86, 448 GB/s peak bandwidth). Full spec: [HARDWARE.md](HARDWARE.md).

## The benchmark table

_Populated in Phase 5. Every row is reproducible by a script in this repo._

| Engine | Model | Batch | Tokens/sec | TTFT | Notes |
|---|---|---|---|---|---|
| _tbd_ | | | | | |

## Rules this project holds itself to

1. HuggingFace is used for **weights + tokenizer download only**. The forward
   pass, sampling loop, and cache are ours.
2. No vLLM / TensorRT-LLM / FlashAttention / xformers. We benchmark *against* them.
3. Every performance claim has a number, a methodology, and a hardware spec.
4. Correctness gates every optimization — numerical parity tests run first.
5. Regressions get reported, with numbers (quantization quality loss, small-batch
   GPU waste).

## Layout

```
nano_infer/     the engine (model, cache, kernels, quantization)
bench/          benchmark harness (TTFT, inter-token latency, tokens/sec)
tests/          parity tests — correctness answer keys
results/        committed benchmark outputs and figures
```
