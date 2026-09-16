"""Does removing host overhead finally let the kernels matter end to end?

ROADMAP Gotcha #29: making decode attention 2.6x faster moved end-to-end
throughput by 0.99-1.03x, because the decode step was bound by the host --
~3,200 aten dispatches and 65 GPU->host syncs per step at batch 32 -- not by
the GPU. nano_infer/decode_graph.py removes both. This script measures what
that buys, and whether the Phase 3 kernels show up once it is gone.

THREE ENGINES x TWO KERNEL SETTINGS
-----------------------------------
  paged        model.generate_paged -- the existing engine (host bookkeeping
               inside the step, 65 syncs/step at batch 32)
  static       decode_graph.generate_paged_static(use_graph=False) -- the same
               step with every sync and all per-step bookkeeping removed, run
               eagerly. Isolates the SYNC/BOOKKEEPING win.
  graph        the static step captured as a CUDA graph and replayed. Adds the
               DISPATCH win on top.

Each under kernels off and on. The question the table answers is in the last
columns: the kernels-on/kernels-off ratio per engine. If Gotcha #29 is right,
that ratio should be ~1.0x for decode under `paged` and grow under `graph`.

METHODOLOGY
-----------
  * All six arms run ROUND-ROBIN inside every repeat, not one arm's repeats
    back to back. Background GPU load (Gotcha #18) then hits every arm equally,
    so the RATIOS stay meaningful even on a contended desktop. Absolute tok/s
    from a contended run should not be published; the script stamps
    utilization and memory into the results so that cannot happen silently.
  * Warmup 2, repeats 3, median -- the harness convention, so the spread metric
    is comparable (Gotcha #21). cv = stdev/mean is reported too.
  * end-to-end tok/s = batch * new_tokens / whole call, INCLUDING prefill and,
    for `graph`, capture. That is the user-facing number.
  * decode ms/step is steady-state decode only:
      paged:          (call - prefill-only call) / (new_tokens - 1)
      static, graph:  the decode phase measured inside the call, excluding
                      prefill and capture
    Capture cost is reported in its own column rather than hidden in either.

Run:  python -m bench.decode_graph
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
from nano_infer.decode_graph import generate_paged_static

CASES = [  # (batch, prompt_len, new_tokens)
    (1, 32, 64),
    (4, 32, 64),
    (16, 32, 64),
    (32, 32, 64),
    (32, 512, 64),
    (8, 1024, 64),
]
ENGINES = ("paged", "static", "graph")


def gpu_state() -> dict:
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


def _cv(xs):
    return statistics.stdev(xs) / statistics.mean(xs) * 100 if len(xs) > 1 else 0.0


def run_arm(engine, kernels_on, weights, cf, ids, gen):
    """One call. Returns (whole-call seconds, decode seconds or None, capture s)."""
    with M.using_kernels(kernels_on):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if engine == "paged":
            M.generate_paged(ids, weights, cf, gen)
            torch.cuda.synchronize()
            return time.perf_counter() - t0, None, 0.0
        timings = {}
        generate_paged_static(ids, weights, cf, gen,
                              use_graph=(engine == "graph"), timings=timings)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, timings["decode_s"], timings["capture_s"]


def prefill_only(kernels_on, weights, cf, ids):
    with M.using_kernels(kernels_on):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        M.generate_paged(ids, weights, cf, 1)
        torch.cuda.synchronize()
        return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    before = gpu_state()
    print(f"\nDecode host overhead: paged vs static vs CUDA graph -- "
          f"{cfg.MODEL_NAME}, fp16, greedy")
    print(f"warmup {args.warmup}, repeats {args.repeats}, all arms round-robin per repeat")
    if before:
        print(f"GPU at start: {before['utilization_pct']}% utilization, "
              f"{before['memory_used_mib']}/{before['memory_total_mib']} MiB")
        if before["utilization_pct"] > 10 or before["memory_used_mib"] > 500:
            print("  ** CONTENDED: other apps are using the GPU. Ratios below are "
                  "robust (round-robin);\n  ** absolute tok/s are NOT publishable "
                  "from this run.")

    weights = M.load_weights()
    cf = M.QwenConfig()
    from nano_infer import kernels
    kernels.load()

    arms = [(e, k) for e in ENGINES for k in (False, True)]
    rows = []
    for batch, plen, gen in CASES:
        g = torch.Generator(device="cpu").manual_seed(batch * 7919 + plen)
        ids = torch.randint(1000, 5000, (batch, plen), generator=g).to(cfg.DEVICE)

        for _ in range(args.warmup):
            for e, k in arms:
                run_arm(e, k, weights, cf, ids, gen)
            for k in (False, True):
                prefill_only(k, weights, cf, ids)

        samples = {a: {"call": [], "decode": [], "capture": []} for a in arms}
        pre = {False: [], True: []}
        for _ in range(args.repeats):
            for e, k in arms:
                call, dec, cap = run_arm(e, k, weights, cf, ids, gen)
                samples[(e, k)]["call"].append(call)
                samples[(e, k)]["capture"].append(cap)
                if dec is not None:
                    samples[(e, k)]["decode"].append(dec)
            for k in (False, True):
                pre[k].append(prefill_only(k, weights, cf, ids))

        steps = gen - 1
        res = {}
        for e, k in arms:
            s = samples[(e, k)]
            call = statistics.median(s["call"])
            if e == "paged":
                dec = (call - statistics.median(pre[k])) / steps
            else:
                dec = statistics.median(s["decode"]) / steps
            res[(e, k)] = {
                "tok_s": batch * gen / call,
                "decode_ms_step": dec * 1e3,
                "capture_s": statistics.median(s["capture"]),
                "cv_pct": _cv(s["call"]),
            }

        print(f"\nbatch {batch}, prompt {plen}, {gen} new tokens")
        print(f"{'engine':>8}{'kern':>6}{'tok/s':>9}{'decode ms/step':>16}"
              f"{'capture s':>11}{'cv':>7}{'vs paged(same kern)':>21}")
        for e in ENGINES:
            for k in (False, True):
                r = res[(e, k)]
                base = res[("paged", k)]["decode_ms_step"]
                print(f"{e:>8}{('on' if k else 'off'):>6}{r['tok_s']:>9.1f}"
                      f"{r['decode_ms_step']:>16.2f}{r['capture_s']:>11.3f}"
                      f"{r['cv_pct']:>6.1f}%{base / r['decode_ms_step']:>20.2f}x")
        kr = {e: res[(e, False)]["decode_ms_step"] / res[(e, True)]["decode_ms_step"]
              for e in ENGINES}
        print("  kernels-on speedup on DECODE, per engine: " +
              ", ".join(f"{e} {kr[e]:.2f}x" for e in ENGINES))

        for e, k in arms:
            rows.append({"batch": batch, "prompt_len": plen, "new_tokens": gen,
                         "engine": e, "kernels": k, **res[(e, k)]})

    after = gpu_state()
    print(f"\nGPU at end: {after.get('utilization_pct')}% utilization, "
          f"{after.get('memory_used_mib')} MiB")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "decode_graph.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "model": cfg.MODEL_NAME, "dtype": str(cfg.DTYPE),
        "warmup": args.warmup, "repeats": args.repeats,
        "gpu_at_start": before, "gpu_at_end": after,
        "contended": bool(before) and (before["utilization_pct"] > 10
                                       or before["memory_used_mib"] > 500),
        "results": rows,
    }, indent=2), encoding="utf-8")

    lines = ["| batch | prompt | engine | kernels | tok/s | decode ms/step | capture s | vs paged (decode) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = next(x for x in rows if x["batch"] == r["batch"]
                    and x["prompt_len"] == r["prompt_len"]
                    and x["engine"] == "paged" and x["kernels"] == r["kernels"])
        lines.append(f"| {r['batch']} | {r['prompt_len']} | {r['engine']} | "
                     f"{'on' if r['kernels'] else 'off'} | {r['tok_s']:.1f} | "
                     f"{r['decode_ms_step']:.2f} | {r['capture_s']:.3f} | "
                     f"{base['decode_ms_step'] / r['decode_ms_step']:.2f}x |")
    stamp = (f"\nGPU at start: {before.get('utilization_pct')}% util, "
             f"{before.get('memory_used_mib')} MiB. "
             + ("**Contended run: ratios are robust (arms round-robin), absolute "
                "tok/s are not.**" if before and (before['utilization_pct'] > 10
                                                   or before['memory_used_mib'] > 500)
                else "Idle GPU."))
    (cfg.RESULTS_DIR / "decode_graph.md").write_text(
        "\n".join(lines) + "\n" + stamp + "\n", encoding="utf-8")
    print("Saved decode_graph.json and decode_graph.md")


if __name__ == "__main__":
    main()
