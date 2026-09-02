"""Phase 3 acceptance: end-to-end tokens/sec, custom kernels on vs off.

This is the phase's actual acceptance criterion, and the number that matters.
Individual kernel speedups do not transfer to it, for reasons already measured:

  - kernels 1-3 are LAUNCH-BOUND at decode sizes. RoPE's q at batch 32 is
    28,672 elements = 57 KB; RMSNorm's rows are 896 wide. Their microbenchmark
    ratios (7.66x, 1.65x, 5.03x) were measured on tensors orders of magnitude
    larger than anything a decode step produces.
  - kernel 4 is the only one touching a big tensor, and it runs at 11.1% of
    peak bandwidth because it is latency-bound (Gotcha #17).

So the honest expectation before running this is a modest gain concentrated in
decode, and the point of the benchmark is to find out how modest. A/B is done in
ONE process with the same weights and cache settings, so the only difference is
the flag.

METHODOLOGY
-----------
Same rules as bench/harness.py: torch.cuda.synchronize() before every timer
stop, warmup iterations discarded, median over repeats. Each configuration is
also run several times to report run-to-run spread, because this machine shares
its GPU with the desktop and the spread is not always small — see the
contention warning printed at the top. A number reported without its variance,
on a card that is also drawing someone's wallpaper, is not a measurement.

Run:  python -m bench.phase3_end_to_end
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

import torch

from nano_infer import config as cfg
from nano_infer import model as M

PROMPT_LEN = 32
NEW_TOKENS = 64
BATCHES = [1, 4, 16, 32]


def gpu_contention() -> dict:
    """Ask the driver what else is using the card. A benchmark run while the
    desktop is busy is not comparable to one run on an idle GPU, and pretending
    otherwise is how bogus numbers get published."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        util, used, total = [int(x) for x in out.split(",")]
        return {"utilization_pct": util, "memory_used_mib": used,
                "memory_total_mib": total}
    except Exception:
        return {}


def make_prompt(batch: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(0)
    return torch.randint(1000, 5000, (batch, PROMPT_LEN), generator=g).to(cfg.DEVICE)


def time_generate(weights, cf, ids, new_tokens, repeats, warmup):
    for _ in range(warmup):
        M.generate_paged(ids, weights, cf, new_tokens)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        M.generate_paged(ids, weights, cf, new_tokens)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    args = ap.parse_args()

    before = gpu_contention()
    print(f"\nPhase 3 end-to-end — {cfg.MODEL_NAME}, fp16, paged cache, greedy")
    print(f"prompt {PROMPT_LEN} tokens, {args.new_tokens} new tokens, "
          f"median of {args.repeats} runs")
    if before:
        print(f"GPU at start: {before['utilization_pct']}% utilization, "
              f"{before['memory_used_mib']}/{before['memory_total_mib']} MiB used")
        if before["utilization_pct"] > 10:
            print("  ** WARNING: the GPU is busy with other work. These numbers "
                  "are contended.\n"
                  "  ** Close GPU-using desktop apps and re-run for a clean "
                  "comparison.")

    weights = M.load_weights()
    cf = M.QwenConfig()
    from nano_infer import kernels
    kernels.load()          # JIT compile once, outside every timed region

    print(f"\n{'batch':>7}{'PyTorch tok/s':>16}{'kernels tok/s':>16}"
          f"{'speedup':>10}{'py spread':>12}{'k spread':>11}")
    print("-" * 74)

    rows = []
    for batch in BATCHES:
        ids = make_prompt(batch)

        with M.using_kernels(False):
            t_off = time_generate(weights, cf, ids, args.new_tokens,
                                  args.repeats, args.warmup)
        with M.using_kernels(True):
            t_on = time_generate(weights, cf, ids, args.new_tokens,
                                 args.repeats, args.warmup)

        med_off, med_on = statistics.median(t_off), statistics.median(t_on)
        tok = batch * args.new_tokens
        tps_off, tps_on = tok / med_off, tok / med_on
        spread_off = (max(t_off) - min(t_off)) / med_off * 100
        spread_on = (max(t_on) - min(t_on)) / med_on * 100

        rows.append({
            "batch": batch, "prompt_len": PROMPT_LEN,
            "new_tokens": args.new_tokens,
            "torch_s": med_off, "kernels_s": med_on,
            "torch_tok_s": tps_off, "kernels_tok_s": tps_on,
            "speedup": tps_on / tps_off,
            "torch_spread_pct": spread_off, "kernels_spread_pct": spread_on,
        })
        print(f"{batch:>7}{tps_off:>16.1f}{tps_on:>16.1f}"
              f"{tps_on / tps_off:>9.2f}x{spread_off:>11.1f}%{spread_on:>10.1f}%")

    after = gpu_contention()
    print("-" * 74)
    best = max(rows, key=lambda r: r["speedup"])
    worst = min(rows, key=lambda r: r["speedup"])
    print(f"best  {best['speedup']:.2f}x at batch {best['batch']}   |   "
          f"worst {worst['speedup']:.2f}x at batch {worst['batch']}")
    max_spread = max(max(r["torch_spread_pct"], r["kernels_spread_pct"]) for r in rows)
    print(f"largest run-to-run spread: {max_spread:.1f}% "
          f"(the harness bar is 3%)")
    if max_spread > 3:
        print("  ** spread exceeds the project's 3% bar — treat these ratios as "
              "indicative,\n  ** not as the headline number, until the GPU is idle.")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "phase3_end_to_end.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "model": cfg.MODEL_NAME, "dtype": str(cfg.DTYPE),
        "prompt_len": PROMPT_LEN, "new_tokens": args.new_tokens,
        "repeats": args.repeats, "warmup": args.warmup,
        "gpu_before": before, "gpu_after": after,
        "results": rows,
    }, indent=2))

    # The caveat travels WITH the table. A markdown file gets pasted into a
    # README; a warning that only ever appeared on stdout does not survive that.
    header = []
    if max_spread > 3 or before.get("utilization_pct", 0) > 10:
        header = [
            f"> **Provisional — contended measurement.** The GPU was at "
            f"{before.get('utilization_pct', '?')}% utilization from other "
            f"processes when this ran, and run-to-run spread reached "
            f"{max_spread:.1f}% against this project's 3% bar. The A/B ratio is "
            f"more robust than the absolute tok/s, since both sides shared the "
            f"same contention, but neither is a headline number until this is "
            f"re-run on an idle GPU.",
            "",
        ]
    table = ["| Batch | PyTorch tok/s | Custom kernels tok/s | Speedup |",
             "|---|---|---|---|"]
    for r in rows:
        table.append(f"| {r['batch']} | {r['torch_tok_s']:.1f} | "
                     f"{r['kernels_tok_s']:.1f} | **{r['speedup']:.2f}x** |")
    (cfg.RESULTS_DIR / "phase3_end_to_end.md").write_text(
        "\n".join(header + table) + "\n")
    print("\nSaved phase3_end_to_end.json and phase3_end_to_end.md")


if __name__ == "__main__":
    main()
