"""Measure continuous batching against static batching on a request stream.

Why a separate benchmark: the fixed-batch table (bench/phase2_cache.py) cannot
show continuous batching's benefit at all. That table gives every sequence the
same output length, so nothing ever finishes early and there is no straggler to
wait on — static and continuous behave identically by construction.

The benefit only appears with VARIED output lengths, which is what real traffic
looks like. This script builds a stream of requests whose output lengths are
drawn from a skewed distribution (many short, a few long) and runs the same
requests through both admission policies on the same engine and hardware.

Both policies pay identical prefill and per-token costs. The only difference is
what happens to a slot whose sequence has finished:
  static     — the slot keeps decoding until the whole group is done (wasted)
  continuous — the slot is freed and a waiting request moves in immediately

Run:  python -m bench.phase2_continuous
"""
from __future__ import annotations

import argparse
import json
import random

import torch

from nano_infer import config as cfg
from nano_infer import model as M
from nano_infer.engine import ContinuousBatchingEngine, Request
from nano_infer.hf_ref import encode_prompt, load_hf

PROMPT = "Explain how a transformer neural network works, step by step."


def make_lengths(n: int, seed: int, short: int, long: int, long_frac: float):
    """Skewed output lengths: mostly short replies, a few long ones."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < long_frac:
            out.append(rng.randint(long // 2, long))
        else:
            out.append(rng.randint(short // 2, short))
    return out


def build(prompt_ids, lengths):
    return [Request(req_id=i, prompt_ids=prompt_ids, max_new_tokens=n)
            for i, n in enumerate(lengths)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--short", type=int, default=32)
    ap.add_argument("--long", type=int, default=128)
    ap.add_argument("--long-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(cfg.SEED)
    cf = M.QwenConfig()
    weights = M.load_weights()
    _, tok = load_hf()
    prompt_ids = encode_prompt(tok, PROMPT)

    lengths = make_lengths(args.requests, args.seed, args.short, args.long,
                           args.long_frac)
    total_wanted = sum(lengths)
    print(f"\nRequest stream: {args.requests} requests, max_batch={args.max_batch}")
    print(f"  output lengths: {lengths}")
    print(f"  min {min(lengths)}, max {max(lengths)}, total {total_wanted} tokens")

    blocks = args.max_batch * ((prompt_ids.shape[1] + args.long + 16) // 16 + 2)
    engine = ContinuousBatchingEngine(weights, cf, num_blocks=blocks,
                                      block_size=16, max_batch=args.max_batch,
                                      max_seq=prompt_ids.shape[1] + args.long + 8)

    # warm up CUDA / allocator so neither policy pays first-call costs
    engine.run_continuous(build(prompt_ids, [4] * args.max_batch))

    static = engine.run_static(build(prompt_ids, lengths))
    cont = engine.run_continuous(build(prompt_ids, lengths))

    print(f"\n{'policy':>12}{'wall s':>10}{'steps':>8}{'tokens':>9}"
          f"{'tok/s':>10}{'slot util':>11}")
    print("  " + "-" * 58)
    for s in (static, cont):
        print(f"{s.policy:>12}{s.wall_s:>10.2f}{s.steps:>8}{s.useful_tokens:>9}"
              f"{s.tokens_per_sec:>10.1f}{s.slot_utilization:>10.1f}%")

    speedup = cont.tokens_per_sec / static.tokens_per_sec
    saved = (static.wall_s - cont.wall_s) / static.wall_s * 100
    print(f"\n  continuous batching: {speedup:.2f}x throughput, "
          f"{saved:.1f}% less wall time")
    print(f"  wasted slot-steps: static {static.wasted_slot_steps}, "
          f"continuous {cont.wasted_slot_steps}")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "model": cfg.MODEL_NAME,
        "requests": args.requests,
        "max_batch": args.max_batch,
        "output_lengths": lengths,
        "results": [
            {"policy": s.policy, "wall_s": s.wall_s, "steps": s.steps,
             "useful_tokens": s.useful_tokens, "slot_steps": s.slot_steps,
             "tokens_per_sec": s.tokens_per_sec,
             "slot_utilization_pct": s.slot_utilization}
            for s in (static, cont)
        ],
        "speedup": speedup,
    }
    (cfg.RESULTS_DIR / "phase2_continuous.json").write_text(json.dumps(payload, indent=2))

    table = "\n".join([
        "| Policy | Wall time (s) | Decode steps | Tokens | Tokens/sec | Slot utilization |",
        "|---|---|---|---|---|---|",
        f"| Static batching | {static.wall_s:.2f} | {static.steps} | "
        f"{static.useful_tokens} | {static.tokens_per_sec:.1f} | {static.slot_utilization:.1f}% |",
        f"| **Continuous batching** | **{cont.wall_s:.2f}** | {cont.steps} | "
        f"{cont.useful_tokens} | **{cont.tokens_per_sec:.1f}** | **{cont.slot_utilization:.1f}%** |",
    ])
    (cfg.RESULTS_DIR / "phase2_continuous.md").write_text(table + "\n")
    print("\n" + table)
    print("\nSaved phase2_continuous.json and phase2_continuous.md")


if __name__ == "__main__":
    main()
