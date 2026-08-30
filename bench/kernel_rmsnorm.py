"""Microbenchmark: fused RMSNorm kernel vs PyTorch, with bandwidth utilization.

The headline number for a memory-bound kernel is not its speedup — it is what
fraction of the card's peak memory bandwidth it achieves. A speedup says "faster
than the thing I replaced." Bandwidth utilization says "here is how much room is
left," and for RMSNorm, whose arithmetic intensity is ~1 FLOP/byte against this
card's ~364 FLOP/byte roofline ridge, memory is the only thing that matters.

Compulsory traffic for one fused RMSNorm over [rows, hidden] in fp16:

    read  x     rows * hidden * 2 bytes
    write out   rows * hidden * 2 bytes
    read  w     hidden * 2 bytes        (reused every row, stays in L2)
    ------------------------------------
    total    ~= 4 * rows * hidden bytes

Both implementations are scored against that same ideal, so PyTorch's number
comes out low precisely because it moves far more than the compulsory minimum —
five kernels, each with its own round trip. That gap is the thing being removed.

Run:  python -m bench.kernel_rmsnorm
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

PEAK_GBS = 448.0          # RTX 3070 theoretical peak, see HARDWARE.md

# (rows, hidden) — shapes the model actually produces.
CASES = [
    (1, 896,        "decode, batch 1"),
    (32, 896,       "decode, batch 32"),
    (34, 896,       "prefill, batch 1"),
    (32 * 34, 896,  "prefill, batch 32"),
    (4096, 896,     "large"),
    (16384, 896,    "very large"),
]


def _sync():
    torch.cuda.synchronize()


def time_fn(fn, x, w, eps, runs: int, warmup: int) -> float:
    for _ in range(warmup):
        fn(x, w, eps)
    times = []
    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        fn(x, w, eps)
        _sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    mod = kernels.load()
    eps = 1e-6

    def torch_impl(x, w, e):
        return M.rms_norm(x, w, e)

    def cuda_impl(x, w, e):
        return mod.rmsnorm_forward(x, w, e)

    print(f"\nFused RMSNorm microbenchmark — RTX 3070, peak {PEAK_GBS:.0f} GB/s, fp16")
    print(f"{'shape':>16}{'PyTorch us':>13}{'ours us':>10}{'speedup':>9}"
          f"{'ours GB/s':>12}{'% peak':>9}{'torch %':>9}")
    print("-" * 80)

    rows_out = []
    for rows, hidden, label in CASES:
        x = torch.randn(rows, hidden, dtype=cfg.DTYPE, device=cfg.DEVICE)
        w = torch.randn(hidden, dtype=cfg.DTYPE, device=cfg.DEVICE)

        t_torch = time_fn(torch_impl, x, w, eps, args.runs, args.warmup)
        t_cuda = time_fn(cuda_impl, x, w, eps, args.runs, args.warmup)

        compulsory = 4.0 * rows * hidden          # read x + write out, fp16
        gbs_cuda = compulsory / t_cuda / 1e9
        gbs_torch = compulsory / t_torch / 1e9

        rows_out.append({
            "rows": rows, "hidden": hidden, "label": label,
            "torch_us": t_torch * 1e6, "cuda_us": t_cuda * 1e6,
            "speedup": t_torch / t_cuda,
            "cuda_gbs": gbs_cuda, "cuda_pct_peak": gbs_cuda / PEAK_GBS * 100,
            "torch_pct_peak": gbs_torch / PEAK_GBS * 100,
            "compulsory_bytes": compulsory,
        })
        print(f"{rows:>9}x{hidden:<6}{t_torch*1e6:>13.1f}{t_cuda*1e6:>10.1f}"
              f"{t_torch/t_cuda:>8.2f}x{gbs_cuda:>12.1f}"
              f"{gbs_cuda/PEAK_GBS*100:>8.1f}%{gbs_torch/PEAK_GBS*100:>8.1f}%")

    best = max(rows_out, key=lambda r: r["cuda_pct_peak"])
    print("-" * 80)
    print(f"best bandwidth utilization: {best['cuda_pct_peak']:.1f}% of peak "
          f"({best['cuda_gbs']:.0f} GB/s) at {best['rows']}x{best['hidden']}")
    print("small shapes are launch-latency-bound, not bandwidth-bound: a "
          "1x896 row is 3.6 KB, far too little work to fill 46 SMs.")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "kernel_rmsnorm.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "peak_bandwidth_gbs": PEAK_GBS,
        "dtype": str(cfg.DTYPE),
        "runs": args.runs, "warmup": args.warmup,
        "results": rows_out,
    }, indent=2))

    table = ["| Shape | PyTorch (us) | Fused kernel (us) | Speedup | GB/s | % of peak |",
             "|---|---|---|---|---|---|"]
    for r in rows_out:
        table.append(f"| {r['rows']}x{r['hidden']} | {r['torch_us']:.1f} | "
                     f"{r['cuda_us']:.1f} | **{r['speedup']:.2f}x** | "
                     f"{r['cuda_gbs']:.0f} | **{r['cuda_pct_peak']:.1f}%** |")
    (cfg.RESULTS_DIR / "kernel_rmsnorm.md").write_text("\n".join(table) + "\n")
    print("\nSaved kernel_rmsnorm.json and kernel_rmsnorm.md")


if __name__ == "__main__":
    main()
