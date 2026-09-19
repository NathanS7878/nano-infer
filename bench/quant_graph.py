"""Quantization's speed story, re-measured once the host is out of the loop.

Every Phase 4 speed number was taken with the eager `generate_paged` engine,
which turned out to be host-bound (ROADMAP #29, #36). Two Phase 4 conclusions
leaned on that regime without knowing it:

  * Gotcha #26: INT8 and INT4 ran at the SAME speed, "which proves neither is
    bandwidth-bound" -- explained by a ~33 us floor per call that was, at least
    partly, launch and dispatch overhead.
  * the acceptance table: INT4 decode barely faster than fp16 at batch 1
    (67.1 vs 57.9 tok/s) and much slower at batch 32.

Under CUDA graphs a decode step streams its weights at ~60% of peak (#36), and
INT8/INT4 store the quantized projections in ~1/2 and ~1/4 of the bytes. So
this re-asks Phase 4's speed question in the regime where weight bytes are
what the step waits on.

WHAT GETS QUANTIZED, and why it caps the win: only the 168 projection matrices.
The tied embedding -- which is also the lm_head, read in full every step to
produce logits -- stays fp16 (quant.is_quantizable). So the weight bytes one
decode step streams are:

  fp16   715.7 MB layers + 272.3 MB lm_head = 987.9 MB
  int8   ~358 MB layers  + 272.3 MB lm_head = ~630 MB   (ceiling 1.57x)
  int4   ~188 MB layers  + 272.3 MB lm_head = ~460 MB   (ceiling 2.15x)

computed exactly from the packed tensors below, not from these estimates.

METHODOLOGY -- same as bench/decode_graph.py
  * all precision x engine arms round-robin inside every repeat, so background
    GPU load hits each arm equally and ratios survive a contended desktop;
  * warmup 2, repeats 3, median; cv reported;
  * decode ms/step is steady state: paged = (call - prefill-only call)/steps,
    graph = the decode phase timed inside the call, excluding capture;
  * all three weight sets are resident at once so the arms can interleave, so
    peak VRAM here is meaningless -- Phase 4's acceptance table has that.
  * kernels on throughout: packed weights require the dequant-matmul kernels.

Run:  python -m bench.quant_graph
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
from nano_infer.decode_graph import generate_paged_static

PEAK_GBS = 448.0
PROMPT_LEN = 32
NEW_TOKENS = 64
BATCHES = [1, 4, 32]
PRECISIONS = ("fp16", "int8", "int4")
ENGINES = ("paged", "graph")


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


def streamed_weight_bytes(weights: dict) -> int:
    """Bytes of 2-D weights one decode step reads: every projection (packed or
    not, including scales/zero points) plus the tied lm_head. Norms and biases
    are excluded, matching the 987.9 MB fp16 figure in ROADMAP #36."""
    total = 0
    for name, w in weights.items():
        if Q.is_packed(w):
            total += w.nbytes()
        elif name == "model.embed_tokens.weight" or Q.is_quantizable(name, w):
            total += w.numel() * w.element_size()
    return total


def one_call(engine, weights, cf, ids, gen):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if engine == "paged":
        M.generate_paged(ids, weights, cf, gen)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, None
    timings = {}
    generate_paged_static(ids, weights, cf, gen, use_graph=True, timings=timings)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, timings["decode_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    before = gpu_state()
    contended = bool(before) and (before["utilization_pct"] > 10
                                  or before["memory_used_mib"] > 500)
    print(f"\nQuantization under CUDA graphs -- {cfg.MODEL_NAME}, kernels on, "
          f"prompt {PROMPT_LEN}, {NEW_TOKENS} new tokens")
    print(f"GPU at start: {before.get('utilization_pct')}% util, "
          f"{before.get('memory_used_mib')} MiB"
          + ("  ** CONTENDED: ratios robust, absolutes not publishable" if contended else ""))

    cf = M.QwenConfig()
    base = M.load_weights()
    sets = {"fp16": dict(base)}
    for mode in ("int8", "int4"):
        sets[mode], _ = Q.pack_weights(base, mode)
    streamed = {p: streamed_weight_bytes(sets[p]) for p in PRECISIONS}
    print("weight bytes streamed per decode step: " + ", ".join(
        f"{p} {streamed[p]/1e6:.1f} MB (ceiling {streamed['fp16']/streamed[p]:.2f}x)"
        for p in PRECISIONS))

    from nano_infer import kernels
    kernels.load()

    arms = [(p, e) for p in PRECISIONS for e in ENGINES]
    rows = []
    with M.using_kernels(True):
        for batch in BATCHES:
            g = torch.Generator(device="cpu").manual_seed(batch * 131)
            ids = torch.randint(1000, 5000, (batch, PROMPT_LEN), generator=g).to(cfg.DEVICE)

            def prefill(p):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                M.generate_paged(ids, sets[p], cf, 1)
                torch.cuda.synchronize()
                return time.perf_counter() - t0

            for _ in range(args.warmup):
                for p, e in arms:
                    one_call(e, sets[p], cf, ids, NEW_TOKENS)
                for p in PRECISIONS:
                    prefill(p)

            calls = {a: [] for a in arms}
            decs = {a: [] for a in arms}
            pres = {p: [] for p in PRECISIONS}
            for _ in range(args.repeats):
                for p, e in arms:
                    c, d = one_call(e, sets[p], cf, ids, NEW_TOKENS)
                    calls[(p, e)].append(c)
                    if d is not None:
                        decs[(p, e)].append(d)
                for p in PRECISIONS:
                    pres[p].append(prefill(p))

            steps = NEW_TOKENS - 1
            res = {}
            for p, e in arms:
                call = statistics.median(calls[(p, e)])
                dec = ((call - statistics.median(pres[p])) / steps if e == "paged"
                       else statistics.median(decs[(p, e)]) / steps)
                gbs = streamed[p] / dec / 1e9
                res[(p, e)] = {
                    "tok_s": batch * NEW_TOKENS / call,
                    "decode_ms_step": dec * 1e3,
                    "weight_gbs": gbs,
                    "weight_pct_peak": gbs / PEAK_GBS * 100,
                    "cv_pct": (statistics.stdev(calls[(p, e)])
                               / statistics.mean(calls[(p, e)]) * 100),
                }

            print(f"\nbatch {batch}")
            print(f"{'precision':>10}{'engine':>7}{'tok/s':>9}{'decode ms/step':>16}"
                  f"{'vs ' + cfg.DTYPE_NAME + ' (same eng)':>20}"
                  f"{'weight GB/s':>13}{'% peak':>8}{'cv':>7}")
            for p in PRECISIONS:
                for e in ENGINES:
                    r = res[(p, e)]
                    rel = res[("fp16", e)]["decode_ms_step"] / r["decode_ms_step"]
                    r["speedup_vs_fp16_same_engine"] = rel
                    print(f"{p:>10}{e:>7}{r['tok_s']:>9.1f}{r['decode_ms_step']:>16.2f}"
                          f"{rel:>19.2f}x{r['weight_gbs']:>13.1f}"
                          f"{r['weight_pct_peak']:>7.1f}%{r['cv_pct']:>6.1f}%")
                    rows.append({"batch": batch, "precision": p, "engine": e,
                                 "streamed_weight_mb": streamed[p] / 1e6, **r})

    after = gpu_state()
    print(f"\nGPU at end: {after.get('utilization_pct')}% util, "
          f"{after.get('memory_used_mib')} MiB")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "quant_graph.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "model": cfg.MODEL_NAME, "prompt_len": PROMPT_LEN, "new_tokens": NEW_TOKENS,
        "warmup": args.warmup, "repeats": args.repeats,
        "gpu_at_start": before, "gpu_at_end": after, "contended": contended,
        "streamed_weight_bytes": streamed, "results": rows,
    }, indent=2), encoding="utf-8")
    lines = ["| batch | precision | engine | decode ms/step | vs fp16 (same engine) | weight GB/s | % of peak |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['batch']} | {r['precision']} | {r['engine']} | "
                     f"{r['decode_ms_step']:.2f} | {r['speedup_vs_fp16_same_engine']:.2f}x | "
                     f"{r['weight_gbs']:.1f} | {r['weight_pct_peak']:.1f}% |")
    stamp = (f"\nGPU at start: {before.get('utilization_pct')}% util, "
             f"{before.get('memory_used_mib')} MiB. "
             + ("**Contended: ratios robust (round-robin), absolutes not publishable.**"
                if contended else "Idle GPU."))
    (cfg.RESULTS_DIR / "quant_graph.md").write_text("\n".join(lines) + "\n" + stamp + "\n",
                                                   encoding="utf-8")
    print("Saved quant_graph.json and quant_graph.md")


if __name__ == "__main__":
    main()
