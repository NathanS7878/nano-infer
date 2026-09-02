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

AN HONEST CAVEAT ABOUT THE BANDWIDTH NUMBER
-------------------------------------------
GB/s below is computed from the COMPULSORY bytes — each KV element counted
once. This kernel launches one block per (sequence, query head), so the 7 query
heads sharing a KV head each issue their own reads of the same data. How much of
that duplication reaches DRAM versus being absorbed by L2 depends on whether
those blocks are co-resident. So the figure measures USEFUL bandwidth, not bus
traffic, and a number well under peak may mean duplicated reads rather than
inefficiency. Collapsing the 7 heads into one block is the obvious next
optimization and is recorded as such rather than claimed.

Run:  python -m bench.kernel_attention
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    mod = kernels.load()
    scale = HEAD_DIM ** -0.5
    n_rep = Q_HEADS // KV_HEADS

    print(f"\nFused decode attention microbenchmark — RTX 3070, peak {PEAK_GBS:.0f} GB/s, fp16")
    print(f"GQA {Q_HEADS} query / {KV_HEADS} KV heads, head_dim {HEAD_DIM}, one query token\n")
    print(f"{'shape':>22}{'paged us':>11}{'contig us':>11}{'ours us':>10}"
          f"{'vs paged':>10}{'vs contig':>11}{'ours GB/s':>11}{'% peak':>9}")
    print("-" * 96)

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

        def ours():
            return mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                                lengths, scale)

        t_paged = time_fn(torch_paged, args.runs, args.warmup)
        t_contig = time_fn(torch_contiguous, args.runs, args.warmup)
        t_ours = time_fn(ours, args.runs, args.warmup)

        # compulsory: read K and V once each, for every sequence
        compulsory = 2.0 * batch * length * KV_HEADS * HEAD_DIM * 2
        gbs = compulsory / t_ours / 1e9

        rows.append({
            "batch": batch, "length": length, "label": label,
            "paged_us": t_paged * 1e6, "contig_us": t_contig * 1e6,
            "ours_us": t_ours * 1e6,
            "speedup_vs_paged": t_paged / t_ours,
            "speedup_vs_contig": t_contig / t_ours,
            "gather_cost": t_paged / t_contig,
            "ours_gbs": gbs, "ours_pct_peak": gbs / PEAK_GBS * 100,
            "compulsory_bytes": compulsory,
        })
        print(f"{f'b{batch} L{length}':>22}{t_paged*1e6:>11.1f}{t_contig*1e6:>11.1f}"
              f"{t_ours*1e6:>10.1f}{t_paged/t_ours:>9.2f}x{t_contig/t_ours:>10.2f}x"
              f"{gbs:>11.1f}{gbs/PEAK_GBS*100:>8.1f}%")

    print("-" * 96)
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
        print(f"  best useful bandwidth     : {best['ours_pct_peak']:.1f}% of peak "
              f"({best['ours_gbs']:.0f} GB/s) at b{best['batch']} L{best['length']}")
    print("GB/s counts each KV element once; the 7 query heads sharing a KV head "
          "each issue\ntheir own reads, so this is useful bandwidth, not bus traffic.")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "kernel_attention.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "peak_bandwidth_gbs": PEAK_GBS,
        "dtype": str(cfg.DTYPE),
        "q_heads": Q_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "runs": args.runs, "warmup": args.warmup,
        "results": rows,
    }, indent=2))

    table = ["| Shape | Phase 2 paged (us) | Pre-gathered (us) | Ours (us) | vs paged | vs pre-gathered | GB/s | % of peak |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        table.append(f"| b{r['batch']} L{r['length']} | {r['paged_us']:.1f} | "
                     f"{r['contig_us']:.1f} | {r['ours_us']:.1f} | "
                     f"**{r['speedup_vs_paged']:.2f}x** | {r['speedup_vs_contig']:.2f}x | "
                     f"{r['ours_gbs']:.0f} | **{r['ours_pct_peak']:.1f}%** |")
    (cfg.RESULTS_DIR / "kernel_attention.md").write_text("\n".join(table) + "\n")
    print("\nSaved kernel_attention.json and kernel_attention.md")


if __name__ == "__main__":
    main()
