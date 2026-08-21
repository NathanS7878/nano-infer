# Hardware & Software Environment

Every benchmark number in this repository was produced on the machine described
below. Performance claims are meaningless without this context, so it is recorded
first, before any code that produces a number.

_Recorded: 2026-08-20._

## GPU — NVIDIA GeForce RTX 3070

| Property | Value |
|---|---|
| Architecture | Ampere (GA104), compute capability **sm_86** |
| VRAM | 8.0 GB GDDR6 |
| Streaming Multiprocessors (SMs) | 46 |
| Memory bus width | 256-bit |
| Memory data rate | 14 Gbps (GDDR6) |
| **Theoretical peak memory bandwidth** | **448 GB/s** |
| FP32 compute (peak) | ~20.3 TFLOP/s |
| FP16 tensor-core compute (peak, dense) | ~163 TFLOP/s |
| Driver (NVIDIA-SMI) | 610.62 |
| CUDA UMD (driver max) | 13.3 |
| bf16 supported | yes |

### Theoretical peak memory bandwidth — the number Phase 3 divides by

```
bandwidth = bus_width_bytes  ×  data_rate
          = (256 bits / 8)   ×  14e9 transfers/s
          = 32 bytes         ×  14e9
          = 448 GB/s
```

Phase 3 reports each kernel's **achieved** bandwidth as a percentage of this
448 GB/s figure. A memory-bound kernel that reaches a high fraction of peak is
"done"; one that doesn't has a bug worth hunting (usually uncoalesced access).

### Arithmetic-intensity ridge (roofline, for later)

Compute-to-bandwidth ratio ≈ 163e12 / 448e9 ≈ **364 FLOP/byte** (fp16 tensor).
Any op doing fewer than ~364 FLOP per byte of memory traffic is **memory-bound**
on this card. LLM decode is far below that ridge — hence memory-bound — which is
the whole reason the Phase 3 decode kernel exists.

## CPU / System

| Property | Value |
|---|---|
| CPU | Intel Core i5-8600K @ 3.60 GHz |
| Cores / Threads | 6 / 6 |
| System RAM | 31.9 GB |
| OS | Windows 10 Home, 10.0.19045 (build 19045) |

## Software

| Component | Version |
|---|---|
| Python | 3.12 (conda env `nano-infer`, conda-forge) |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime (bundled in torch wheel) | 12.4 |
| CUDA Toolkit (`nvcc`) | **not yet installed** — required for Phase 3 kernels |
| transformers | 5.15.1 (weights + tokenizer download only) |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| datasets | 5.0.1 |
| accelerate | 1.14.0 |
| numpy | 2.5.2 |

## Reproducing this environment

```bash
conda create -y -n nano-infer -c conda-forge --override-channels python=3.12
conda activate nano-infer
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers safetensors tokenizers huggingface_hub datasets accelerate numpy
```

## Known gaps to close

- **`nvcc` / CUDA Toolkit not installed.** The torch wheel bundles the CUDA
  *runtime* (enough to run GPU tensors), but compiling our own `.cu` kernels in
  Phase 3 needs the *toolkit* compiler. Install before Phase 3.
