"""What does paying CUDA-graph capture once, instead of every call, buy?

`generate_paged_static` builds a fresh DecodeGraphRunner per call and so pays
capture (0.12-0.25 s) every call -- about 28% of a 64-token batch-32 call in
bench/decode_graph.py. A kept DecodeGraphRunner captures on its first call and
then runs prefill + replay only.

Three arms, round-robin inside every repeat (so desktop GPU load hits each
equally -- see bench/decode_graph.py):

  paged     model.generate_paged, the eager engine
  one-shot  generate_paged_static: capture every call
  runner    one DecodeGraphRunner reused across calls: captured during warmup,
            never again

Each timed call gets a DIFFERENT prompt, so the reused runner is doing real
work and cannot benefit from anything prompt-specific. Whole-call tok/s,
including prefill. Kernels on.

Run:  python -m bench.decode_graph_runner
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
from nano_infer.decode_graph import DecodeGraphRunner, generate_paged_static

CASES = [(1, 32, 64), (32, 32, 64), (32, 32, 16)]


def gpu_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        util, used = [int(x) for x in out.split(",")]
        return {"utilization_pct": util, "memory_used_mib": used}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    before = gpu_state()
    contended = bool(before) and (before["utilization_pct"] > 10
                                  or before["memory_used_mib"] > 500)
    print(f"\nCapture once vs every call -- kernels on; GPU at start "
          f"{before.get('utilization_pct')}% util, {before.get('memory_used_mib')} MiB"
          + ("  ** CONTENDED: ratios robust, absolutes not publishable" if contended else ""))

    weights = M.load_weights()
    cf = M.QwenConfig()
    from nano_infer import kernels
    kernels.load()

    rows = []
    with M.using_kernels(True):
        for batch, plen, gen in CASES:
            g = torch.Generator(device="cpu").manual_seed(batch * 17 + gen)
            prompts = [torch.randint(1000, 5000, (batch, plen), generator=g).to(cfg.DEVICE)
                       for _ in range(args.warmup + args.repeats)]
            runner = DecodeGraphRunner(weights, cf, batch, plen, gen)
            arms = {
                "paged": lambda p: M.generate_paged(p, weights, cf, gen),
                "one-shot": lambda p: generate_paged_static(p, weights, cf, gen),
                "runner": lambda p: runner.generate(p),
            }
            times = {a: [] for a in arms}
            for i, p in enumerate(prompts):
                for name, fn in arms.items():
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fn(p)
                    torch.cuda.synchronize()
                    if i >= args.warmup:
                        times[name].append(time.perf_counter() - t0)

            med = {a: statistics.median(t) for a, t in times.items()}
            print(f"\nbatch {batch}, prompt {plen}, {gen} new tokens "
                  f"(runner captured {runner.captures}x in total)")
            print(f"{'arm':>10}{'call s':>9}{'tok/s':>9}{'vs paged':>10}{'vs one-shot':>13}")
            for a in arms:
                print(f"{a:>10}{med[a]:>9.3f}{batch*gen/med[a]:>9.1f}"
                      f"{med['paged']/med[a]:>9.2f}x{med['one-shot']/med[a]:>12.2f}x")
                rows.append({"batch": batch, "prompt_len": plen, "new_tokens": gen,
                             "arm": a, "call_s": med[a], "tok_s": batch * gen / med[a],
                             "vs_paged": med["paged"] / med[a],
                             "vs_one_shot": med["one-shot"] / med[a]})
            assert runner.captures == 1

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "decode_graph_runner.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "model": cfg.MODEL_NAME, "warmup": args.warmup, "repeats": args.repeats,
        "gpu_at_start": before, "contended": contended, "results": rows,
    }, indent=2), encoding="utf-8")
    print("\nSaved decode_graph_runner.json")


if __name__ == "__main__":
    main()
