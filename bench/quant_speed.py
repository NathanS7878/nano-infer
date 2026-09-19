"""Phase 4: fused dequant-matmul speed, and the batch size where it stops paying.

THE PREDICTION, AND WHERE IT WAS WRONG
--------------------------------------
Written down before measuring: weight-only quantization is a DECODE
optimization. Arithmetic intensity per weight is 1 FLOP/byte at fp16, 2 at INT8,
4 at INT4, all far below this card's ~364 FLOP/byte ridge, so decode is
memory-bound and the ceilings are 2x and 4x. At prefill the weight is reused
across many rows of x, intensity climbs, and quantization should stop helping.

Both halves were right about the *mechanism* and wrong about the *crossover*.
It is not decode-versus-prefill. It is around **batch 4-8**, well inside decode:

    batch:      1      2      4      8     16     32
    ours:    34.5   41.1   55.0   84.3  147.1  270.6  us
    fp16:    59.5   59.9   60.1   60.3   61.2   63.5  us

Our time scales with the batch; cuBLAS's is FLAT. That flatness is the whole
explanation. cuBLAS is still weight-streaming-bound at batch 32 — extra rows of
x are nearly free because they ride along on tensor cores (HMMA). Our kernel
does BT x 8 scalar fp32 FFMAs per 4-byte weight load, so once there is enough
batch to leave the memory-bound regime we are competing against tensor cores
with scalar math, and losing by roughly their throughput ratio.

That is not a bug to fix by tuning. It was checked: splitting batch 32 into four
BT=8 tiles gives 0.97x, so register pressure and unrolling are not the cause.
A kernel that won here would need to do its accumulation on tensor cores, which
means dequantizing into fp16 fragments and issuing HMMA — a different and much
larger kernel.

WHAT THIS MEANS FOR THE ENGINE
------------------------------
Weight-only quantization on this GPU buys **latency for single-stream decode**
and **VRAM always**, and it costs **throughput for batched decode**. Those are
different products, and the honest thing is to report the crossover rather than
quote the batch-1 number and stop.

Run:  python -m bench.quant_speed
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from nano_infer import config as cfg
from nano_infer import kernels
from nano_infer import quant as Q
from nano_infer.model import QwenConfig

# (out, in, label) — the projection shapes the model actually contains, read
# from the selected checkpoint rather than hard-coded. These were 896/4864
# literals until the 1.5B scale-up, at which point they would have silently
# kept measuring the 0.5B shapes under bf16 activations.
_CF = QwenConfig()
SHAPES = [
    (_CF.hidden_size, _CF.hidden_size, "q_proj / o_proj"),
    (_CF.intermediate_size, _CF.hidden_size, "gate_proj / up_proj"),
    (_CF.hidden_size, _CF.intermediate_size, "down_proj"),
]
BATCHES = [1, 2, 4, 8, 16, 32]
PREFILL_ROWS = 1024        # batch 32 x 32-token prompts, flattened


def timeit(fn, runs: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    mod = kernels.load()
    torch.manual_seed(0)

    print(f"\nFused dequant-matmul — RTX 3070, {cfg.DTYPE_NAME} activations, "
          f"weight-only quant")
    print(f"{cfg.DTYPE_NAME} column is torch F.linear (cuBLAS, tensor cores)\n")

    rows = []
    for N, K, label in SHAPES:
        w = torch.randn(N, K, dtype=cfg.DTYPE, device=cfg.DEVICE) * 0.05
        t4 = Q.quantize_int4(w, Q.INT4_GROUP)
        t8 = Q.quantize_int8(w)

        print(f"{label}  [{N} x {K}]")
        print(f"{'batch':>7}{'fp16 us':>10}{'int8 us':>10}{'int4 us':>10}"
              f"{'int8':>8}{'int4':>8}")
        for B in BATCHES + [PREFILL_ROWS]:
            x = torch.randn(B, K, dtype=cfg.DTYPE, device=cfg.DEVICE)
            t_f = timeit(lambda: F.linear(x, w), args.runs, args.warmup)
            t_8 = timeit(lambda: mod.int8_matmul(x, t8.q, t8.scale),
                         args.runs, args.warmup)
            t_4 = timeit(lambda: mod.int4_matmul(x, t4.packed, t4.scale, t4.zero,
                                                 K, Q.INT4_GROUP),
                         args.runs, args.warmup)
            tag = "  <- prefill" if B == PREFILL_ROWS else ""
            rows.append({
                "shape": f"{N}x{K}", "label": label, "batch": B,
                "fp16_us": t_f * 1e6, "int8_us": t_8 * 1e6, "int4_us": t_4 * 1e6,
                "int8_speedup": t_f / t_8, "int4_speedup": t_f / t_4,
            })
            print(f"{B:>7}{t_f*1e6:>10.1f}{t_8*1e6:>10.1f}{t_4*1e6:>10.1f}"
                  f"{t_f/t_8:>7.2f}x{t_f/t_4:>7.2f}x{tag}")
        print()

    # where does it stop paying?
    print("crossover — the largest batch at which each scheme still beats fp16:")
    for N, K, label in SHAPES:
        sub = [r for r in rows if r["shape"] == f"{N}x{K}" and r["batch"] in BATCHES]
        for scheme in ("int8", "int4"):
            winning = [r["batch"] for r in sub if r[f"{scheme}_speedup"] > 1.0]
            best = max((r[f"{scheme}_speedup"] for r in sub), default=0.0)
            edge = max(winning) if winning else 0
            print(f"  {label:<22} {scheme}: wins up to batch "
                  f"{edge if edge else '(never)'}, best {best:.2f}x at batch 1")

    print("\nfp16 is FLAT across batch 1-32 because cuBLAS is still weight-bound "
          "there and\nrides tensor cores; ours scales with batch because it "
          "accumulates in scalar fp32.\nThat is the crossover, and it is a "
          "property of the hardware, not a tuning bug.")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "quant_speed.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "dtype": str(cfg.DTYPE), "group": Q.INT4_GROUP,
        "runs": args.runs, "warmup": args.warmup,
        "results": rows,
    }, indent=2), encoding="utf-8")

    table = ["| Shape | Batch | fp16 (us) | INT8 (us) | INT4 (us) | INT8 | INT4 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        table.append(f"| {r['shape']} | {r['batch']} | {r['fp16_us']:.1f} | "
                     f"{r['int8_us']:.1f} | {r['int4_us']:.1f} | "
                     f"{r['int8_speedup']:.2f}x | {r['int4_speedup']:.2f}x |")
    (cfg.RESULTS_DIR / "quant_speed.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8")
    print("\nSaved quant_speed.json and quant_speed.md")


if __name__ == "__main__":
    main()
