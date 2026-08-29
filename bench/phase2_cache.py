"""Measure the Phase 2 KV-cache engine.

Three engines, identical conditions (same prompt, same token count, same
CUDA-synchronized timing):

    Phase 1  nano-infer, no cache — full recompute every step
    Phase 2  nano-infer, KV cache — prefill once, then one token per step
    HF       HuggingFace generate() baseline

Plus a prefill/decode breakdown, which is the conceptual point of Phase 2: the
two phases have opposite bottlenecks. Prefill processes the whole prompt at once
(many tokens, big matmuls, compute-bound). Decode processes one token against the
whole cache (memory-bound — dominated by streaming weights and cache out of VRAM).

Phase 1 numbers are read from results/phase1_nocache.json rather than re-measured,
because the no-cache engine takes 77 s per run at batch 32. Conditions are
identical: same prompt, same 128 new tokens, same timing code.

Run:  python -m bench.phase2_cache
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
from nano_infer.cache import KVCache
from nano_infer.hf_ref import encode_prompt, hf_generate, load_hf


@torch.no_grad()
def prefill_decode_split(weights, cf, base_ids, batch_size, new_tokens, runs=3):
    """Time prefill and decode separately for one batch size."""
    input_ids = base_ids.repeat(batch_size, 1)
    seq = input_ids.shape[1]

    prefill_times, decode_times = [], []
    for _ in range(runs + 1):                      # first iteration is warmup
        cache = KVCache(cf.num_layers, batch_size, cf.num_kv_heads,
                        seq + new_tokens, cf.head_dim)
        rope = M.build_rope_cache(seq + new_tokens, cf.head_dim, cf.rope_theta)

        _sync()
        t0 = time.perf_counter()
        logits = M.forward_cached(input_ids, weights, cf, cache, 0, rope)
        _sync()
        t_prefill = time.perf_counter() - t0

        next_ids = logits.argmax(dim=-1)
        _sync()
        t0 = time.perf_counter()
        for step in range(1, new_tokens):
            logits = M.forward_cached(next_ids.unsqueeze(1), weights, cf, cache,
                                      seq + step - 1, rope)
            next_ids = logits.argmax(dim=-1)
        _sync()
        t_decode = time.perf_counter() - t0

        prefill_times.append(t_prefill)
        decode_times.append(t_decode)

    pre = statistics.median(prefill_times[1:])
    dec = statistics.median(decode_times[1:])
    per_step = dec / max(new_tokens - 1, 1)
    return {
        "batch_size": batch_size,
        "prompt_tokens": seq,
        "prefill_ms": pre * 1e3,
        "prefill_tokens_per_sec": batch_size * seq / pre,
        "decode_total_ms": dec * 1e3,
        "decode_ms_per_step": per_step * 1e3,
        "decode_tokens_per_sec": batch_size * (new_tokens - 1) / dec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    torch.manual_seed(cfg.SEED)
    cf = M.QwenConfig()
    weights = M.load_weights()
    model, tok = load_hf()
    base_ids = encode_prompt(tok, BENCH_PROMPT)

    # Phase 1 numbers, measured previously under identical conditions
    p1_path = cfg.RESULTS_DIR / "phase1_nocache.json"
    p1 = {}
    if p1_path.exists():
        for r in json.loads(p1_path.read_text())["head_to_head"]:
            p1[r["batch_size"]] = r["ours_tokens_per_sec"]

    print(f"\n1. PREFILL vs DECODE — {args.new_tokens} tokens")
    print(f"   {'batch':>6}{'prefill ms':>12}{'prefill tok/s':>15}"
          f"{'decode ms/step':>16}{'decode tok/s':>14}")
    print("   " + "-" * 63)
    splits = []
    for bs in args.batch_sizes:
        s = prefill_decode_split(weights, cf, base_ids, bs, args.new_tokens,
                                 runs=args.runs)
        splits.append(s)
        print(f"   {bs:>6}{s['prefill_ms']:>12.2f}{s['prefill_tokens_per_sec']:>15.0f}"
              f"{s['decode_ms_per_step']:>16.2f}{s['decode_tokens_per_sec']:>14.1f}")

    print(f"\n2. HEAD-TO-HEAD — {args.new_tokens} new tokens")
    rows = []
    for bs in args.batch_sizes:
        input_ids = base_ids.repeat(bs, 1)

        def ours(x, n):
            return M.generate_cached(x, weights, cf, n)

        def hf(x, n):
            return hf_generate(model, x, n)

        for _ in range(args.warmup):
            ours(input_ids, 4)
            hf(input_ids, 4)

        our_t, hf_t = [], []
        for _ in range(args.runs):
            our_t.append(_time_generate(ours, input_ids, args.new_tokens))
            hf_t.append(_time_generate(hf, input_ids, args.new_tokens))

        our_s, hf_s = statistics.median(our_t), statistics.median(hf_t)
        our_tps = bs * args.new_tokens / our_s
        hf_tps = bs * args.new_tokens / hf_s
        p1_tps = p1.get(bs)
        rows.append({
            "batch_size": bs,
            "phase1_tokens_per_sec": p1_tps,
            "phase2_tokens_per_sec": our_tps,
            "hf_tokens_per_sec": hf_tps,
            "speedup_vs_phase1": (our_tps / p1_tps) if p1_tps else None,
            "speedup_vs_hf": our_tps / hf_tps,
        })
        sp1 = f"{our_tps / p1_tps:5.2f}x" if p1_tps else "  n/a"
        print(f"   bs={bs:<3d} Phase2={our_tps:8.1f}  Phase1={p1_tps or float('nan'):8.1f}  "
              f"HF={hf_tps:8.1f}   vs-P1={sp1}  vs-HF={our_tps / hf_tps:5.2f}x")

    lines = ["| Batch | Phase 1 no cache | **Phase 2 KV cache** | HF generate() | "
             "vs Phase 1 | vs HF |", "|---|---|---|---|---|---|"]
    for r in rows:
        p1v = f"{r['phase1_tokens_per_sec']:.1f}" if r["phase1_tokens_per_sec"] else "n/a"
        sp = f"{r['speedup_vs_phase1']:.2f}x" if r["speedup_vs_phase1"] else "n/a"
        lines.append(f"| {r['batch_size']} | {p1v} | **{r['phase2_tokens_per_sec']:.1f}** | "
                     f"{r['hf_tokens_per_sec']:.1f} | {sp} | {r['speedup_vs_hf']:.2f}x |")
    table = "\n".join(lines)

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "phase2_cache.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "model": cfg.MODEL_NAME,
        "dtype": str(cfg.DTYPE),
        "new_tokens": args.new_tokens,
        "prefill_decode_split": splits,
        "head_to_head": rows,
    }, indent=2))
    (cfg.RESULTS_DIR / "phase2_cache.md").write_text(table + "\n")
    print("\n" + table)
    print("\nSaved phase2_cache.json and phase2_cache.md")


if __name__ == "__main__":
    main()
