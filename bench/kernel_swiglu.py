"""Microbenchmark: fused SwiGLU kernel vs PyTorch, with bandwidth utilization.

WHY THE EXPECTED WIN IS SMALL, AND WHY THAT IS THE POINT
--------------------------------------------------------
Kernel 1 (RMSNorm) collapsed five kernels into one and got 7.66x. It is easy to
read that as "hand-written CUDA is ~7x faster than PyTorch." It is not. The win
was exactly the memory traffic removed, and this kernel is the control that
proves it.

PyTorch's `F.silu(gate) * up` is two kernels:

    F.silu(gate)   read gate 2B            write tmp 2B    =  4 B/elem
    tmp * up       read tmp 2B + up 2B     write out 2B    =  6 B/elem
                                                             ----------
                                                             10 B/elem

Ours is one kernel at the compulsory minimum:

    read gate 2B + read up 2B + write out 2B                =  6 B/elem

So the ceiling is 10/6 = 1.67x, predicted before any code was written. If the
measurement lands near it, the model of "speedup = bytes removed" holds and we
understand the machine. If it lands far off, something else is going on and that
is worth chasing.

Both implementations are scored against the SAME 6 B/elem compulsory ideal, so
PyTorch cannot exceed 60% of peak here by construction — it moves 10 bytes to
accomplish 6 bytes of work. The `torch actual` column re-scores PyTorch against
the 10 bytes it really moves, which is the honest measure of whether PyTorch's
own kernels are efficient. (They are. They just run twice.)

Run:  python -m bench.kernel_swiglu
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

PEAK_GBS = 448.0           # RTX 3070 theoretical peak, see HARDWARE.md
INTERMEDIATE = 4864        # Qwen2.5-0.5B MLP width

COMPULSORY_BYTES_PER_ELEM = 6.0    # read gate + read up + write out, fp16
TORCH_BYTES_PER_ELEM = 10.0        # + the silu temporary's round trip

CASES = [
    (1,        "decode, batch 1"),
    (32,       "decode, batch 32"),
    (34,       "prefill, batch 1"),
    (32 * 34,  "prefill, batch 32"),
    (4096,     "large"),
    (16384,    "very large"),
]


def _sync():
    torch.cuda.synchronize()


def time_fn(fn, gate, up, runs: int, warmup: int) -> float:
    for _ in range(warmup):
        fn(gate, up)
    times = []
    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        fn(gate, up)
        _sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    mod = kernels.load()

    def torch_impl(gate, up):
        return F.silu(gate) * up

    def cuda_impl(gate, up):
        return mod.swiglu_forward(gate, up)

    print(f"\nFused SwiGLU microbenchmark — RTX 3070, peak {PEAK_GBS:.0f} GB/s, fp16")
    print(f"predicted ceiling from traffic alone: "
          f"{TORCH_BYTES_PER_ELEM / COMPULSORY_BYTES_PER_ELEM:.2f}x "
          f"({TORCH_BYTES_PER_ELEM:.0f} B/elem -> {COMPULSORY_BYTES_PER_ELEM:.0f} B/elem)\n")
    print(f"{'shape':>16}{'PyTorch us':>13}{'ours us':>10}{'speedup':>9}"
          f"{'ours GB/s':>12}{'% peak':>9}{'torch %':>9}{'torch actual':>14}")
    print("-" * 94)

    rows_out = []
    for rows, label in CASES:
        gate = torch.randn(rows, INTERMEDIATE, dtype=cfg.DTYPE, device=cfg.DEVICE)
        up = torch.randn(rows, INTERMEDIATE, dtype=cfg.DTYPE, device=cfg.DEVICE)

        t_torch = time_fn(torch_impl, gate, up, args.runs, args.warmup)
        t_cuda = time_fn(cuda_impl, gate, up, args.runs, args.warmup)

        n = rows * INTERMEDIATE
        compulsory = COMPULSORY_BYTES_PER_ELEM * n
        gbs_cuda = compulsory / t_cuda / 1e9
        gbs_torch = compulsory / t_torch / 1e9
        gbs_torch_actual = TORCH_BYTES_PER_ELEM * n / t_torch / 1e9

        rows_out.append({
            "rows": rows, "width": INTERMEDIATE, "label": label,
            "torch_us": t_torch * 1e6, "cuda_us": t_cuda * 1e6,
            "speedup": t_torch / t_cuda,
            "cuda_gbs": gbs_cuda, "cuda_pct_peak": gbs_cuda / PEAK_GBS * 100,
            "torch_pct_peak": gbs_torch / PEAK_GBS * 100,
            "torch_actual_pct_peak": gbs_torch_actual / PEAK_GBS * 100,
            "compulsory_bytes": compulsory,
        })
        print(f"{rows:>9}x{INTERMEDIATE:<6}{t_torch*1e6:>13.1f}{t_cuda*1e6:>10.1f}"
              f"{t_torch/t_cuda:>8.2f}x{gbs_cuda:>12.1f}"
              f"{gbs_cuda/PEAK_GBS*100:>8.1f}%{gbs_torch/PEAK_GBS*100:>8.1f}%"
              f"{gbs_torch_actual/PEAK_GBS*100:>13.1f}%")

    best = max(rows_out, key=lambda r: r["cuda_pct_peak"])
    big = [r for r in rows_out if r["rows"] >= 1000]
    print("-" * 94)
    print(f"best bandwidth utilization: {best['cuda_pct_peak']:.1f}% of peak "
          f"({best['cuda_gbs']:.0f} GB/s) at {best['rows']}x{INTERMEDIATE}")
    if big:
        mean_big = sum(r["speedup"] for r in big) / len(big)
        print(f"speedup at bandwidth-bound sizes: {mean_big:.2f}x "
              f"vs the {TORCH_BYTES_PER_ELEM/COMPULSORY_BYTES_PER_ELEM:.2f}x traffic ceiling")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "kernel_swiglu.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "peak_bandwidth_gbs": PEAK_GBS,
        "dtype": str(cfg.DTYPE),
        "runs": args.runs, "warmup": args.warmup,
        "compulsory_bytes_per_elem": COMPULSORY_BYTES_PER_ELEM,
        "torch_bytes_per_elem": TORCH_BYTES_PER_ELEM,
        "predicted_ceiling": TORCH_BYTES_PER_ELEM / COMPULSORY_BYTES_PER_ELEM,
        "results": rows_out,
    }, indent=2))

    table = ["| Shape | PyTorch (us) | Fused kernel (us) | Speedup | GB/s | % of peak |",
             "|---|---|---|---|---|---|"]
    for r in rows_out:
        table.append(f"| {r['rows']}x{r['width']} | {r['torch_us']:.1f} | "
                     f"{r['cuda_us']:.1f} | **{r['speedup']:.2f}x** | "
                     f"{r['cuda_gbs']:.0f} | **{r['cuda_pct_peak']:.1f}%** |")
    (cfg.RESULTS_DIR / "kernel_swiglu.md").write_text("\n".join(table) + "\n")
    print("\nSaved kernel_swiglu.json and kernel_swiglu.md")


if __name__ == "__main__":
    main()
