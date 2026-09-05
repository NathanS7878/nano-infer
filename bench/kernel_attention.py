"""Microbenchmark: fused decode attention (online softmax, paged in place).

This kernel targets the number Phase 2 ended on: decode achieving 26.5 GB/s,
5.9% of this card's 448 GB/s peak. Kernels 1-3 could not move it — at decode
they touch tensors of a few tens of KB and are launch-bound. This one streams
the whole KV cache, which is where decode's time actually goes.

TWO INDEPENDENT WINS, MEASURED SEPARATELY
-----------------------------------------
The kernel does two distinct things, and reporting one number for both would
hide which one mattered:

  A. it never materializes the score matrix (online softmax), and never
     materializes repeat_kv's 7x copy of K and V;
  B. it walks the paged slot table in place, removing the gather copy that
     Phase 2's paged cache pays every decode step.

So three implementations are timed:

  torch_paged       gather -> repeat_kv -> scores -> softmax -> blend  (Phase 2)
  torch_contiguous  the same WITHOUT the gather (KV handed over pre-gathered)
  ours              one kernel, no gather, no repeat, no score matrix

  torch_paged / torch_contiguous  isolates win B (the gather)
  torch_contiguous / ours         isolates win A (the fusion)

BYTE COUNT, per sequence, L cached positions, fp16, GQA 14 q / 2 kv heads:

  ours (compulsory)   read K 256L + read V 256L                      =   512L
  torch_contiguous    repeat_kv 4096L + qk 3612L + softmax 392L
                      + pv 3612L                                     = ~11712L
  torch_paged         + gather 1024L                                 = ~12736L

Predicted ceiling vs torch_paged: ~24x. The dominant single term is repeat_kv,
which materializes a 7x copy of K and V so a batched matmul can see 14 KV heads
that GQA deliberately did not store. Kernel 3's lesson, louder: the expensive
thing is the plumbing that reshapes operands to suit a library call.

THE BANDWIDTH NUMBER, AND WHAT IT TURNED OUT TO MEAN
----------------------------------------------------
GB/s below is computed from the COMPULSORY bytes — each KV element counted once.
The per-query-head kernel launches one block per (sequence, query head), so the
7 query heads sharing a KV head each issue their own reads of the same data.
This file used to say that a number well under peak "may mean duplicated reads
rather than inefficiency", and record collapsing the 7 heads into one block as
the obvious next optimization. Both halves of that turned out to be right, and
`--sharing` is the experiment that settled it.

Hold the block count and the per-block work exactly constant, and vary only how
many query heads share a KV head:

    kv heads | n_rep | distinct | issued  |     us | issued GB/s | % of peak
          14 |     1 |  235 MB  | 235 MB  | 1132.3 |       207.4 |     46.3%
           2 |     7 | 33.6 MB  | 235 MB  |  687.9 |       341.4 |     76.2%
           1 |    14 | 16.8 MB  | 235 MB  |  673.8 |       348.6 |     77.8%

Wall time flattens once n_rep >= 7: the duplicate reads ARE absorbed by cache.
But a cache hit still costs a load instruction and issue slots, and against the
bytes it actually ASKS FOR the kernel sits at 76% of peak. The headline "11.1%
of peak" was never 11% of the card — it was a load path near its ceiling
carrying 7x more traffic than the algorithm needs.

So the fix is not to go faster but to ask for less, which is what the grouped
kernel does: one block per (sequence, KV head), each K and V element read once
into a register and fed to all n_rep query heads. Both paths are timed below.

Run:  python -m bench.kernel_attention
      python -m bench.kernel_attention --sharing    (the experiment above)
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

PEAK_GBS = 448.0            # RTX 3070 theoretical peak, see HARDWARE.md

Q_HEADS = 14                # Qwen2.5-0.5B
KV_HEADS = 2
HEAD_DIM = 64

# (batch, cached length, label) — decode always has exactly one query token.
CASES = [
    (1,  128,  "batch 1, short context"),
    (32, 128,  "batch 32, short context"),
    (32, 512,  "batch 32, medium context"),
    (32, 1024, "batch 32, long context"),
    (32, 2048, "batch 32, very long context"),
]


def _sync():
    torch.cuda.synchronize()


def time_fn(fn, runs: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def build_case(batch: int, length: int):
    """A paged KV pool with scrambled slots, plus the pre-gathered contiguous
    view the second reference is handed for free."""
    torch.manual_seed(0)
    slots_total = batch * length
    q = torch.randn(batch, Q_HEADS, HEAD_DIM, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k_pool = torch.randn(slots_total, KV_HEADS, HEAD_DIM,
                         dtype=cfg.DTYPE, device=cfg.DEVICE)
    v_pool = torch.randn(slots_total, KV_HEADS, HEAD_DIM,
                         dtype=cfg.DTYPE, device=cfg.DEVICE)
    # scrambled, as a real allocator's block table would be
    perm = torch.randperm(slots_total, device=cfg.DEVICE)
    slot_table = perm.reshape(batch, length).long()
    lengths = torch.full((batch,), length, device=cfg.DEVICE).long()

    k_contig = k_pool[slot_table].transpose(1, 2).contiguous()   # [b,kv,L,hd]
    v_contig = v_pool[slot_table].transpose(1, 2).contiguous()
    return q, k_pool, v_pool, slot_table, lengths, k_contig, v_contig


def sharing_experiment(mod, args):
    """Is this kernel limited by DRAM traffic, or by the rate it issues loads?

    The two are easy to confuse and the distinction decides which optimization
    is worth writing. The experiment holds the BLOCK COUNT and the PER-BLOCK
    WORK exactly constant — same 448 blocks, same 512 threads, same number of
    dot products, same barriers — and varies only how many query heads share a
    KV head, which changes only how much DISTINCT data those blocks touch.

    If wall time tracks the distinct footprint, DRAM is the limit. If it goes
    flat while the issued traffic stays put, the limit is the load path, and the
    fix is to issue fewer loads rather than to move fewer bytes.
    """
    batch, length, q_heads, head_dim = 32, 2048, 14, 64
    print(f"\nKV-sharing experiment — RTX 3070, peak {PEAK_GBS:.0f} GB/s, fp16")
    print(f"batch {batch}, L {length}, {q_heads} query heads, head_dim {head_dim}"
          f" — {batch * q_heads} blocks in every row, per-query-head kernel\n")
    print(f"{'kv heads':>9}{'n_rep':>7}{'distinct MB':>13}{'issued MB':>11}"
          f"{'us':>10}{'distinct GB/s':>15}{'issued GB/s':>13}{'% peak':>9}")
    print("-" * 87)

    rows = []
    for kv_heads in (14, 7, 2, 1):
        torch.manual_seed(0)
        slots_total = batch * length
        q = torch.randn(batch, q_heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
        k_pool = torch.randn(slots_total, kv_heads, head_dim,
                             dtype=cfg.DTYPE, device=cfg.DEVICE)
        v_pool = torch.randn(slots_total, kv_heads, head_dim,
                             dtype=cfg.DTYPE, device=cfg.DEVICE)
        slot_table = torch.randperm(slots_total,
                                    device=cfg.DEVICE).reshape(batch, length).long()
        lengths = torch.full((batch,), length, device=cfg.DEVICE).long()
        scale = head_dim ** -0.5

        # fuse_heads=-1: the per-query-head kernel, so the block count is
        # batch*q_heads in every row and only the footprint changes
        t = time_fn(lambda: mod.decode_attention_forward(
            q, k_pool, v_pool, slot_table, lengths, scale, 0, -1),
            args.runs, args.warmup)

        issued = 2.0 * batch * length * q_heads * head_dim * 2
        distinct = 2.0 * batch * length * kv_heads * head_dim * 2
        rows.append({
            "kv_heads": kv_heads, "n_rep": q_heads // kv_heads,
            "distinct_mb": distinct / 1e6, "issued_mb": issued / 1e6,
            "us": t * 1e6, "distinct_gbs": distinct / t / 1e9,
            "issued_gbs": issued / t / 1e9,
            "issued_pct_peak": issued / t / 1e9 / PEAK_GBS * 100,
        })
        print(f"{kv_heads:>9}{q_heads // kv_heads:>7}{distinct/1e6:>13.1f}"
              f"{issued/1e6:>11.1f}{t*1e6:>10.1f}{distinct/t/1e9:>15.1f}"
              f"{issued/t/1e9:>13.1f}{issued/t/1e9/PEAK_GBS*100:>8.1f}%")
    print("-" * 87)
    flat = rows[-1]["us"] / rows[-2]["us"]
    print(f"halving the distinct footprint again (n_rep 7 -> 14) changes wall "
          f"time by {(1-flat)*100:+.1f}%:")
    print(f"  the duplicate reads are absorbed by cache, so DRAM is not the "
          f"limit -- but the")
    print(f"  kernel still ISSUES {rows[2]['issued_mb']:.0f} MB and sits at "
          f"{rows[2]['issued_pct_peak']:.1f}% of peak on that traffic.")
    print(f"  Reading each KV row once for all n_rep heads is therefore the "
          f"fix: see the grouped\n  kernel in the main table.")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "kernel_attention_sharing.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "peak_bandwidth_gbs": PEAK_GBS, "batch": batch, "length": length,
        "q_heads": q_heads, "head_dim": head_dim,
        "runs": args.runs, "warmup": args.warmup, "results": rows,
    }, indent=2), encoding="utf-8")
    print("\nSaved kernel_attention_sharing.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sharing", action="store_true",
                    help="run the KV-sharing experiment instead: hold block "
                         "count and per-block work fixed, vary only the "
                         "distinct KV footprint")
    args = ap.parse_args()

    mod = kernels.load()
    if args.sharing:
        return sharing_experiment(mod, args)
    scale = HEAD_DIM ** -0.5
    n_rep = Q_HEADS // KV_HEADS

    print(f"\nFused decode attention microbenchmark — RTX 3070, peak {PEAK_GBS:.0f} GB/s, fp16")
    print(f"GQA {Q_HEADS} query / {KV_HEADS} KV heads, head_dim {HEAD_DIM}, one query token\n")
    print(f"{'shape':>18}{'paged us':>10}{'contig us':>11}{'per-head us':>13}"
          f"{'grouped us':>12}{'group win':>11}{'vs paged':>10}"
          f"{'GB/s':>8}{'% peak':>8}")
    print("-" * 101)

    rows = []
    for batch, length, label in CASES:
        q, k_pool, v_pool, slot_table, lengths, k_contig, v_contig = build_case(batch, length)

        def torch_paged():
            k = M.repeat_kv(k_pool[slot_table].transpose(1, 2), n_rep)
            v = M.repeat_kv(v_pool[slot_table].transpose(1, 2), n_rep)
            s = (q.unsqueeze(2) @ k.transpose(-1, -2)).float() * scale
            p = torch.softmax(s, dim=-1).to(v.dtype)
            return (p @ v).squeeze(2)

        def torch_contiguous():
            k = M.repeat_kv(k_contig, n_rep)
            v = M.repeat_kv(v_contig, n_rep)
            s = (q.unsqueeze(2) @ k.transpose(-1, -2)).float() * scale
            p = torch.softmax(s, dim=-1).to(v.dtype)
            return (p @ v).squeeze(2)

        # fuse_heads: -1 forces one block per (sequence, query head), 1 forces
        # one block per (sequence, KV head). Forced rather than left on auto, so
        # this table is an A/B of the two kernels rather than a record of what
        # the dispatcher happened to choose for each shape.
        def per_head():
            return mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                                lengths, scale, 0, -1)

        def grouped():
            return mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                                lengths, scale, 0, 1)

        t_paged = time_fn(torch_paged, args.runs, args.warmup)
        t_contig = time_fn(torch_contiguous, args.runs, args.warmup)
        t_perhead = time_fn(per_head, args.runs, args.warmup)
        t_grouped = time_fn(grouped, args.runs, args.warmup)
        t_ours = min(t_perhead, t_grouped)   # what the auto dispatcher picks

        # compulsory: read K and V once each, for every sequence
        compulsory = 2.0 * batch * length * KV_HEADS * HEAD_DIM * 2
        gbs = compulsory / t_ours / 1e9

        rows.append({
            "batch": batch, "length": length, "label": label,
            "paged_us": t_paged * 1e6, "contig_us": t_contig * 1e6,
            "per_head_us": t_perhead * 1e6, "grouped_us": t_grouped * 1e6,
            "grouped_win": t_perhead / t_grouped,
            "ours_us": t_ours * 1e6,
            "speedup_vs_paged": t_paged / t_ours,
            "speedup_vs_contig": t_contig / t_ours,
            "gather_cost": t_paged / t_contig,
            "ours_gbs": gbs, "ours_pct_peak": gbs / PEAK_GBS * 100,
            "compulsory_bytes": compulsory,
        })
        print(f"{f'b{batch} L{length}':>18}{t_paged*1e6:>10.1f}"
              f"{t_contig*1e6:>11.1f}{t_perhead*1e6:>13.1f}{t_grouped*1e6:>12.1f}"
              f"{t_perhead/t_grouped:>10.2f}x{t_paged/t_ours:>9.2f}x"
              f"{gbs:>8.1f}{gbs/PEAK_GBS*100:>7.1f}%")

    print("-" * 101)
    big = [r for r in rows if r["batch"] >= 32 and r["length"] >= 512]
    if big:
        mp = sum(r["speedup_vs_paged"] for r in big) / len(big)
        mc = sum(r["speedup_vs_contig"] for r in big) / len(big)
        mg = sum(r["gather_cost"] for r in big) / len(big)
        best = max(rows, key=lambda r: r["ours_pct_peak"])
        print(f"at batch 32, context >= 512:")
        print(f"  vs the Phase 2 paged path : {mp:.2f}x")
        print(f"  vs a pre-gathered path    : {mc:.2f}x   (win A: online softmax "
              f"+ no repeat_kv copy)")
        print(f"  the gather alone cost     : {mg:.2f}x   (win B: reading the "
              f"block table in place)")
        mw = sum(r["grouped_win"] for r in big) / len(big)
        print(f"  head-group fusion alone   : {mw:.2f}x   (win C: each KV row "
              f"read once for all 7\n{'':30}query heads instead of once each)")
        print(f"  best useful bandwidth     : {best['ours_pct_peak']:.1f}% of peak "
              f"({best['ours_gbs']:.0f} GB/s) at b{best['batch']} L{best['length']}")
    print("GB/s counts each KV element once. For the per-query-head kernel that "
          "is USEFUL\nbandwidth, not bus traffic -- it issues 7x more; see "
          "--sharing. The grouped kernel\nissues what it uses, so there the "
          "two coincide.")
    print("The grouped kernel needs blocks to fill the card and starts n_rep "
          "behind on block\ncount, so it LOSES below 8 blocks (batch 1 here). "
          "The dispatcher picks on that.")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "kernel_attention.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "peak_bandwidth_gbs": PEAK_GBS,
        "dtype": str(cfg.DTYPE),
        "q_heads": Q_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "runs": args.runs, "warmup": args.warmup,
        "results": rows,
    }, indent=2))

    table = ["| Shape | Phase 2 paged (us) | Pre-gathered (us) | Per-query-head (us) | Head-grouped (us) | Fusion win | vs paged | GB/s | % of peak |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        table.append(f"| b{r['batch']} L{r['length']} | {r['paged_us']:.1f} | "
                     f"{r['contig_us']:.1f} | {r['per_head_us']:.1f} | "
                     f"{r['grouped_us']:.1f} | {r['grouped_win']:.2f}x | "
                     f"**{r['speedup_vs_paged']:.2f}x** | "
                     f"{r['ours_gbs']:.0f} | **{r['ours_pct_peak']:.1f}%** |")
    (cfg.RESULTS_DIR / "kernel_attention.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8")
    print("\nSaved kernel_attention.json and kernel_attention.md")


if __name__ == "__main__":
    main()
