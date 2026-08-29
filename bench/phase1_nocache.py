"""Measure the Phase 1 no-cache engine — the honest "before" number.

Phase 1 is deliberately the slow, obviously-correct version: every decode step
recomputes the entire sequence from scratch. This script quantifies exactly how
expensive that is, so the Phase 2 KV-cache speedup is a measured delta rather
than a claim.

Two measurements:

  1. GROWTH CURVE — time one forward pass at increasing sequence lengths. Per-step
     cost rises with sequence length, so generating N tokens costs ~O(N^2) total.
     This is the algorithmic problem the KV cache removes.

  2. HEAD-TO-HEAD — our engine vs the Phase 0 HuggingFace baseline at matching
     batch sizes, same prompt, same token count, same CUDA-synchronized timing.

Run:  python -m bench.phase1_nocache
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from bench.harness import BENCH_PROMPT, _sync, _time_generate
from nano_infer import config as cfg
from nano_infer import model as M
from nano_infer.hf_ref import encode_prompt, hf_generate, load_hf

GROWTH_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048, 4096]


@torch.no_grad()
def growth_curve(weights, cf, runs: int = 3) -> list[dict]:
    """Time a single forward pass at increasing sequence lengths.

    Two regimes are visible in the result. At short sequences the pass is
    dominated by streaming ~1 GB of weights out of VRAM plus ~170 kernel
    launches, so sequence length barely matters and the curve is FLAT. Past some
    length the O(seq^2) attention term takes over and the curve bends upward.
    Where that knee sits determines where a KV cache actually pays off.
    """
    rows = []
    for seq in GROWTH_LENGTHS:
        ids = torch.randint(0, cf.vocab_size, (1, seq), device=cfg.DEVICE)
        try:
            for _ in range(2):                               # warmup
                M.forward(ids, weights, cf)
            times = []
            for _ in range(runs):
                _sync()
                t0 = time.perf_counter()
                M.forward(ids, weights, cf)
                _sync()
                times.append(time.perf_counter() - t0)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  seq={seq:<5d} OOM — materializing [1,14,{seq},{seq}] scores "
                  f"exceeds 8 GB (itself a Phase 3 motivation)")
            rows.append({"seq_len": seq, "forward_ms": None, "oom": True})
            continue
        ms = statistics.median(times) * 1e3
        per_tok = ms / seq * 1e3
        rows.append({"seq_len": seq, "forward_ms": ms, "us_per_token": per_tok})
        print(f"  seq={seq:<5d} one forward pass: {ms:8.2f} ms   "
              f"({per_tok:7.1f} us/token)")
    return rows


def head_to_head(weights, cf, model, base_ids, batch_sizes, new_tokens,
                 warmup, runs) -> list[dict]:
    """Our no-cache engine vs HF generate(), same conditions."""
    rows = []
    for bs in batch_sizes:
        input_ids = base_ids.repeat(bs, 1)

        def ours(x, n):
            return M.greedy_decode(x, weights, cf, n)

        def hf(x, n):
            return hf_generate(model, x, n)

        for _ in range(warmup):
            ours(input_ids, 4)
            hf(input_ids, 4)

        our_times, hf_times = [], []
        for _ in range(runs):
            our_times.append(_time_generate(ours, input_ids, new_tokens))
            hf_times.append(_time_generate(hf, input_ids, new_tokens))

        our_s = statistics.median(our_times)
        hf_s = statistics.median(hf_times)
        our_tps = bs * new_tokens / our_s
        hf_tps = bs * new_tokens / hf_s
        rows.append({
            "batch_size": bs,
            "new_tokens": new_tokens,
            "ours_tokens_per_sec": our_tps,
            "hf_tokens_per_sec": hf_tps,
            "ours_total_s": our_s,
            "hf_total_s": hf_s,
            "speedup_vs_hf": our_tps / hf_tps,
        })
        print(f"  bs={bs:<3d} ours={our_tps:8.1f} tok/s   HF={hf_tps:8.1f} tok/s   "
              f"ratio={our_tps / hf_tps:5.2f}x  ({our_s:.1f}s vs {hf_s:.1f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(cfg.SEED)
    cf = M.QwenConfig()
    weights = M.load_weights()
    model, tok = load_hf()                      # HF baseline (default sdpa)
    base_ids = encode_prompt(tok, BENCH_PROMPT)

    print("\n1. GROWTH CURVE — cost of ONE forward pass vs sequence length")
    print("   (no cache: every decode step pays this, and it grows)")
    growth = growth_curve(weights, cf)

    torch.cuda.empty_cache()      # the 4096 growth point leaves a big allocation
    print(f"\n2. HEAD-TO-HEAD — {args.new_tokens} new tokens, "
          f"{args.runs} runs, {args.warmup} warmup")
    h2h = head_to_head(weights, cf, model, base_ids, args.batch_sizes,
                       args.new_tokens, args.warmup, args.runs)

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.RESULTS_DIR / "phase1_nocache.json"
    out.write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "model": cfg.MODEL_NAME,
        "dtype": str(cfg.DTYPE),
        "engine": "nano-infer Phase 1 (no KV cache, full recompute per step)",
        "growth_curve": growth,
        "head_to_head": h2h,
    }, indent=2))

    lines = ["| Batch | nano-infer Phase 1 (tok/s) | HF generate() (tok/s) | Ratio |",
             "|---|---|---|---|"]
    for r in h2h:
        lines.append(f"| {r['batch_size']} | {r['ours_tokens_per_sec']:.1f} | "
                     f"{r['hf_tokens_per_sec']:.1f} | {r['speedup_vs_hf']:.2f}x |")
    table = "\n".join(lines)
    (cfg.RESULTS_DIR / "phase1_nocache.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nSaved {out.name} and phase1_nocache.md")


if __name__ == "__main__":
    main()
