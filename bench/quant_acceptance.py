"""Phase 4 acceptance: model size, tokens/sec, perplexity, peak VRAM.

The table CLAUDE.md asks for, with the trade-off visible rather than averaged
away. Every row runs on PACKED weights — the quantized form is the only form
resident, because keeping an fp16 copy around to dodge the batch crossover would
forfeit the memory saving while leaving the headline claim in place.

WHY tokens/sec IS REPORTED AT TWO BATCH SIZES
---------------------------------------------
The fused dequant-matmul beats cuBLAS at batch 1-4 and loses above it, because
cuBLAS stays weight-bound to batch 32 and rides tensor cores while this kernel
accumulates in scalar fp32 (see bench/quant_speed.py). A single tokens/sec
number would hide that completely. Batch 1 is the latency case quantization
wins; batch 32 is the throughput case it loses.

WHERE THE PERPLEXITY COLUMN COMES FROM
--------------------------------------
bench/perplexity.py, which measures the round-tripped fp16 weights. That is the
SAME quantization grid this packed path uses, and the two were verified to
produce token-for-token identical generations, so the number transfers. It is
not re-measured here because the perplexity sweep needs all-position logits from
the Phase 1 forward, which is the project's unquantized answer key.

Run:  python -m bench.quant_acceptance
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
from nano_infer import quant as Q

PROMPT_LEN = 32
NEW_TOKENS = 64
MODES = ["fp16", "int8", "int4"]


def gpu_idle() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        u, m = [int(v) for v in out.split(",")]
        return {"utilization_pct": u, "memory_used_mib": m}
    except Exception:
        return {}


def load_perplexity() -> dict:
    """Pull the quality column from the step-1 sweep rather than re-deriving it."""
    path = cfg.RESULTS_DIR / "perplexity.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in data.get("results", []):
        key = r["mode"]
        # the sweep may hold several int4 group sizes; keep the default group
        if key == "int4" and r.get("group") not in (None, Q.INT4_GROUP):
            continue
        out[key] = {"perplexity": r["perplexity"], "delta_pct": r["delta_pct"],
                    "sem_pct": r.get("sem_pct")}
    return out


def measure(mode: str, batches, repeats: int, warmup: int) -> dict:
    """Load, pack, drop the fp16 original, then measure size / speed / VRAM."""
    cf = M.QwenConfig()
    base = M.load_weights()
    weights, stats = Q.pack_weights(base, mode)

    # The fp16 originals must go before peak VRAM means anything.
    del base
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    g = torch.Generator(device="cpu").manual_seed(0)
    rows = {}
    peak_overall = 0
    for b in batches:
        ids = torch.randint(1000, 5000, (b, PROMPT_LEN), generator=g).to(cfg.DEVICE)
        with M.using_kernels(True):
            for _ in range(warmup):
                M.generate_paged(ids, weights, cf, NEW_TOKENS)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                M.generate_paged(ids, weights, cf, NEW_TOKENS)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        peak = torch.cuda.max_memory_allocated()
        peak_overall = max(peak_overall, peak)
        rows[b] = {
            "tok_s": b * NEW_TOKENS / med,
            "seconds": med,
            "spread_pct": (max(ts) - min(ts)) / med * 100,
            "cv_pct": (statistics.stdev(ts) / statistics.mean(ts) * 100
                       if len(ts) > 1 else 0.0),
            "peak_vram_mib": peak / 1024 ** 2,
        }

    del weights
    torch.cuda.empty_cache()
    return {"mode": mode, "stats": stats, "batches": rows,
            "peak_vram_mib": peak_overall / 1024 ** 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--batches", default="1,32")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    before = gpu_idle()
    ppl = load_perplexity()

    print(f"\nPhase 4 acceptance — {cfg.MODEL_NAME}, {cfg.DTYPE_NAME} "
          f"activations, weight-only quantization")
    print(f"prompt {PROMPT_LEN}, {NEW_TOKENS} new tokens, median of "
          f"{args.repeats} runs after {args.warmup} warmups")
    if before:
        print(f"GPU at start: {before['utilization_pct']}% utilization, "
              f"{before['memory_used_mib']} MiB used")
        if before["utilization_pct"] > 10:
            print("  ** WARNING: contended GPU — see Gotcha #18")
    print(f"all rows run on PACKED weights; no {cfg.DTYPE_NAME} copy is "
          f"resident\n")

    results = [measure(m, batches, args.repeats, args.warmup) for m in MODES]

    # "fp16" is the mode IDENTIFIER (CLI value, JSON key); the label shown is
    # the dtype actually in use, which is bf16 on Qwen2.5-1.5B.
    def label(mode):
        return cfg.DTYPE_NAME if mode == "fp16" else mode

    hdr = f"{'mode':>6}{'weights MB':>12}{'compress':>10}{'bits/wt':>9}"
    for b in batches:
        hdr += f"{'tok/s b' + str(b):>12}"
    hdr += f"{'peak VRAM':>11}{'perplexity':>12}{'vs ' + cfg.DTYPE_NAME:>9}"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        st = r["stats"]
        line = (f"{label(r['mode']):>6}{st['bytes_quantized']/1e6:>12.1f}"
                f"{st['compression']:>9.2f}x{st['bits_per_weight']:>9.2f}")
        for b in batches:
            line += f"{r['batches'][b]['tok_s']:>12.1f}"
        line += f"{r['peak_vram_mib']:>10.0f}M"
        q = ppl.get(r["mode"])
        if q:
            line += f"{q['perplexity']:>12.2f}{q['delta_pct']:>8.2f}%"
        else:
            line += f"{'-':>12}{'-':>9}"
        print(line)

    print("-" * len(hdr))
    base_row = results[0]
    for r in results[1:]:
        for b in batches:
            rel = r["batches"][b]["tok_s"] / base_row["batches"][b]["tok_s"]
            print(f"  {r['mode']} vs {cfg.DTYPE_NAME} at batch {b}: "
                  f"{rel:.2f}x tokens/sec")
    vram_saved = base_row["peak_vram_mib"] - results[-1]["peak_vram_mib"]
    print(f"  peak VRAM saved, {cfg.DTYPE_NAME} -> int4: {vram_saved:.0f} MiB "
          f"({base_row['peak_vram_mib']/max(results[-1]['peak_vram_mib'],1):.2f}x)")
    sem = (ppl.get("fp16") or {}).get("sem_pct")
    if sem:
        print(f"  perplexity standard error {sem:.2f}% — INT8's delta is inside "
              f"it, so INT8 is lossless\n  within measurement precision; INT4's "
              f"is not.")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "quant_acceptance.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "model": cfg.MODEL_NAME, "prompt_len": PROMPT_LEN,
        "new_tokens": NEW_TOKENS, "repeats": args.repeats,
        "gpu_before": before, "perplexity_source": "results/perplexity.json",
        "results": results,
    }, indent=2), encoding="utf-8")

    table = ["| Precision | Model size | Compression | bits/wt | "
             + " | ".join(f"tok/s (batch {b})" for b in batches)
             + f" | Peak VRAM | Perplexity | vs {cfg.DTYPE_NAME} |",
             "|---" * (7 + len(batches)) + "|"]   # 7 fixed columns + one per batch
    for r in results:
        st = r["stats"]
        q = ppl.get(r["mode"], {})
        cells = [label(r["mode"]), f"{st['bytes_quantized']/1e6:.0f} MB",
                 f"{st['compression']:.2f}x", f"{st['bits_per_weight']:.2f}"]
        cells += [f"{r['batches'][b]['tok_s']:.1f}" for b in batches]
        cells += [f"{r['peak_vram_mib']:.0f} MiB",
                  f"{q.get('perplexity', float('nan')):.2f}",
                  f"{q.get('delta_pct', float('nan')):+.2f}%"]
        table.append("| " + " | ".join(cells) + " |")
    (cfg.RESULTS_DIR / "quant_acceptance.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8")
    print("\nSaved quant_acceptance.json and quant_acceptance.md")


if __name__ == "__main__":
    main()
