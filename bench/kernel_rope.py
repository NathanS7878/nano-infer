"""Microbenchmark: fused RoPE kernel vs PyTorch, with bandwidth utilization.

THE PREDICTION, MADE BEFORE THE KERNEL EXISTED
----------------------------------------------
PyTorch runs `x * cos + rotate_half(x) * sin` as five kernels. Per element of x
(E elements at 2 bytes; cos/sin are small and stay in L2):

    -x2                 read E/2, write E/2      =  2E bytes
    cat((-x2, x1))      read E,   write E        =  4E
    x * cos             read E,   write E        =  4E
    rotated * sin       read E,   write E        =  4E
    t1 + t2             read 2E,  write E        =  6E
    ---------------------------------------------------
    total                                          20E

Ours reads x once, writes once: 4E. Ceiling = 20/4 = **5x**.

That is 3x better than SwiGLU's ceiling, and the reason is worth stating: of
PyTorch's 20E bytes, 6E are spent on `rotate_half`, which computes NOTHING. It
is pure data movement whose only job is to present the operand in a layout the
next elementwise kernel can consume. A fused kernel replaces it with an index
offset. The most profitable thing to fuse is not the expensive math — it is the
plumbing around it.

THE SECOND QUESTION THIS BENCHMARK ANSWERS
------------------------------------------
A 5x ratio on a microbenchmark does not mean 5x anywhere real. RoPE operates on
q [b, 14, n, 64] and k [b, 2, n, 64] — at decode, n = 1, so batch 32 is
32*14*64 = 28,672 elements = 57 KB. That is far too little to reach peak
bandwidth on a 46-SM card. At the sizes the model actually runs, this kernel is
launch-bound, and the win is "1 launch instead of 5", not bandwidth.

So the table below deliberately spans both regimes, and the summary separates
them. Reporting only the large-shape number would be the kind of flattering
benchmark this project exists to avoid.

Run:  python -m bench.kernel_rope
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from nano_infer import config as cfg
from nano_infer import kernels
from nano_infer import model as M

PEAK_GBS = 448.0                   # RTX 3070 theoretical peak, see HARDWARE.md

COMPULSORY_BYTES_PER_ELEM = 4.0    # read x + write out, fp16
TORCH_BYTES_PER_ELEM = 20.0        # + neg, + cat, + two muls, + the add

# (batch, heads, n, head_dim, label) — heads 14 = query heads, 2 = KV heads (GQA)
CASES = [
    (1,  14, 1,    64, "decode b1, q"),
    (32, 14, 1,    64, "decode b32, q"),
    (32, 2,  1,    64, "decode b32, k (GQA: 2 heads)"),
    (1,  14, 34,   64, "prefill b1, q"),
    (32, 14, 34,   64, "prefill b32, q"),
    (32, 14, 512,  64, "large"),
    (32, 14, 2048, 64, "very large"),
]

# Anything at or above this many elements is genuinely bandwidth-bound on this
# card; below it the measurement is dominated by launch and dispatch cost.
BANDWIDTH_BOUND_ELEMS = 4_000_000


def _sync():
    torch.cuda.synchronize()


def time_fn(fn, x, cos, sin, runs: int, warmup: int) -> float:
    for _ in range(warmup):
        fn(x, cos, sin)
    times = []
    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        fn(x, cos, sin)
        _sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    mod = kernels.load()

    def torch_impl(x, cos, sin):
        # exactly the reference: model.apply_rope on a single tensor
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        return x * c + M.rotate_half(x) * s

    def cuda_impl(x, cos, sin):
        return mod.rope_forward(x, cos, sin)

    print(f"\nFused RoPE microbenchmark — RTX 3070, peak {PEAK_GBS:.0f} GB/s, fp16")
    print(f"predicted ceiling from traffic alone: "
          f"{TORCH_BYTES_PER_ELEM / COMPULSORY_BYTES_PER_ELEM:.2f}x "
          f"({TORCH_BYTES_PER_ELEM:.0f} B/elem -> {COMPULSORY_BYTES_PER_ELEM:.0f} B/elem)\n")
    print(f"{'shape':>22}{'elems':>10}{'PyTorch us':>12}{'ours us':>10}"
          f"{'speedup':>9}{'ours GB/s':>12}{'% peak':>9}")
    print("-" * 84)

    rows_out = []
    for batch, heads, n, head_dim, label in CASES:
        x = torch.randn(batch, heads, n, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
        cos, sin = M.build_rope_cache(n, head_dim, 1e6)

        t_torch = time_fn(torch_impl, x, cos, sin, args.runs, args.warmup)
        t_cuda = time_fn(cuda_impl, x, cos, sin, args.runs, args.warmup)

        elems = batch * heads * n * head_dim
        compulsory = COMPULSORY_BYTES_PER_ELEM * elems
        gbs_cuda = compulsory / t_cuda / 1e9
        gbs_torch_actual = TORCH_BYTES_PER_ELEM * elems / t_torch / 1e9

        rows_out.append({
            "batch": batch, "heads": heads, "n": n, "head_dim": head_dim,
            "label": label, "elems": elems,
            "torch_us": t_torch * 1e6, "cuda_us": t_cuda * 1e6,
            "speedup": t_torch / t_cuda,
            "cuda_gbs": gbs_cuda, "cuda_pct_peak": gbs_cuda / PEAK_GBS * 100,
            "torch_actual_pct_peak": gbs_torch_actual / PEAK_GBS * 100,
            "bandwidth_bound": elems >= BANDWIDTH_BOUND_ELEMS,
        })
        shape = f"{batch}x{heads}x{n}x{head_dim}"
        print(f"{shape:>22}{elems:>10}{t_torch*1e6:>12.1f}{t_cuda*1e6:>10.1f}"
              f"{t_torch/t_cuda:>8.2f}x{gbs_cuda:>12.1f}"
              f"{gbs_cuda/PEAK_GBS*100:>8.1f}%")

    print("-" * 84)
    bb = [r for r in rows_out if r["bandwidth_bound"]]
    lb = [r for r in rows_out if not r["bandwidth_bound"]]
    ceiling = TORCH_BYTES_PER_ELEM / COMPULSORY_BYTES_PER_ELEM

    if bb:
        best = max(bb, key=lambda r: r["cuda_pct_peak"])
        mean_bb = sum(r["speedup"] for r in bb) / len(bb)
        print(f"bandwidth-bound sizes (>= {BANDWIDTH_BOUND_ELEMS:,} elems): "
              f"{mean_bb:.2f}x vs the {ceiling:.2f}x traffic ceiling; "
              f"best {best['cuda_pct_peak']:.1f}% of peak ({best['cuda_gbs']:.0f} GB/s)")
    if lb:
        mean_lb = sum(r["speedup"] for r in lb) / len(lb)
        print(f"model-sized shapes (launch-bound, NOT bandwidth-bound): "
              f"{mean_lb:.2f}x — this is 5 launches becoming 1, not memory efficiency")
    print("the shapes the engine actually runs are all in the second group: "
          "decode b32 q is 28,672 elements = 57 KB.")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "kernel_rope.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "peak_bandwidth_gbs": PEAK_GBS,
        "dtype": str(cfg.DTYPE),
        "runs": args.runs, "warmup": args.warmup,
        "compulsory_bytes_per_elem": COMPULSORY_BYTES_PER_ELEM,
        "torch_bytes_per_elem": TORCH_BYTES_PER_ELEM,
        "predicted_ceiling": ceiling,
        "bandwidth_bound_threshold_elems": BANDWIDTH_BOUND_ELEMS,
        "results": rows_out,
    }, indent=2))

    table = ["| Shape | Elements | PyTorch (us) | Fused kernel (us) | Speedup | GB/s | % of peak |",
             "|---|---|---|---|---|---|---|"]
    for r in rows_out:
        shape = f"{r['batch']}x{r['heads']}x{r['n']}x{r['head_dim']}"
        table.append(f"| {shape} | {r['elems']:,} | {r['torch_us']:.1f} | "
                     f"{r['cuda_us']:.1f} | **{r['speedup']:.2f}x** | "
                     f"{r['cuda_gbs']:.0f} | **{r['cuda_pct_peak']:.1f}%** |")
    (cfg.RESULTS_DIR / "kernel_rope.md").write_text("\n".join(table) + "\n")
    print("\nSaved kernel_rope.json and kernel_rope.md")


if __name__ == "__main__":
    main()
