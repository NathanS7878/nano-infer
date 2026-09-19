"""Divergence attribution in units that mean the same thing in fp16 and bf16.

WHY THIS EXISTS. The Phase 2/3 parity tests used absolute bars calibrated on
Qwen2.5-0.5B in fp16: "a divergence is benign if the reference top-1/top-2 gap
is under 0.05". On Qwen2.5-1.5B in bf16 those bars stopped meaning anything.
bf16 has 7 fraction bits to fp16's 10, so near a logit of 20 its smallest step
is 0.125 -- a gap of 0.05 cannot even be represented -- and the measured drift
between the cached path and the uncached reference is routinely 0.3-1.25
logits. That is Gotcha #16 again: no error metric is universal.

WHY NOT JUST CHECK "GAP <= DRIFT" AT THE FLIP. It is a tautology. If the argmax
flipped from token a to token b, then path[b] >= path[a] while
ref[a] - ref[b] = gap, so the two tokens' logits moved by at least the gap. The
claim with content is about the SIZE of the drift: rounding from a different
matmul shape moves logits by a few ULPs; a real bug (wrong cache slot, wrong
RoPE position, stale KV) moves them by orders of magnitude more.

THE CRITERION. Record the path's own last-position logits while it generates.
Up to and including its first divergence from the reference, its token history
is identical to the reference's -- so its recorded logits there are exactly
what it computes given the reference history, with no teacher-forcing needed.
Compare each against Phase 1's uncached forward on that same history, and
express the max difference in ULPs of the model dtype at that step's top-logit
magnitude. Require it to stay under a bound.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch

from nano_infer import config as cfg
from nano_infer import model as M


def ulp_at(value: float, dtype: torch.dtype = None) -> float:
    """One unit in the last place of `dtype` at magnitude |value|."""
    dtype = dtype or cfg.DTYPE
    mag = max(abs(value), torch.finfo(dtype).tiny)
    return torch.finfo(dtype).eps * 2.0 ** math.floor(math.log2(mag))


@contextmanager
def record_step_logits(fn_name: str):
    """Spy on a module-level forward in nano_infer.model that returns
    last-position logits [batch, vocab] once per generation step."""
    recorded: list[torch.Tensor] = []
    real = getattr(M, fn_name)

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        recorded.append(out.detach().float().clone())
        return out

    setattr(M, fn_name, spy)
    try:
        yield recorded
    finally:
        setattr(M, fn_name, real)


def first_divergence(ref_tokens: torch.Tensor, got_tokens: torch.Tensor):
    ref_tokens, got_tokens = ref_tokens.cpu(), got_tokens.cpu()
    diff = (ref_tokens != got_tokens).nonzero()
    return int(diff[0]) if diff.numel() else None


@torch.no_grad()
def drift_profile(prompt_ids: torch.Tensor, ref_tokens: torch.Tensor,
                  path_logits: list, weights: dict, cf, row: int = 0,
                  upto: int | None = None) -> list[dict]:
    """Per step, how far the path's logits are from Phase 1's uncached forward
    on the same (reference) history.

    prompt_ids  [1, seq] prompt for this row
    ref_tokens  [steps] the reference continuation
    path_logits one [batch, vocab] tensor per generation step, as recorded
    upto        last step to measure (inclusive); defaults to all recorded steps
    """
    last = len(path_logits) - 1 if upto is None else upto
    rows = []
    history = prompt_ids
    dev = prompt_ids.device
    for step in range(last + 1):
        want = M.forward(history, weights, cf)[0, -1].float()
        got = path_logits[step][row].to(want.device)
        top = want.max().item()
        top2 = want.topk(2).values
        drift = (got - want).abs().max().item()
        rows.append({
            "step": step,
            "drift": drift,
            "drift_ulps": drift / ulp_at(top),
            "top_logit": top,
            "ref_gap": (top2[0] - top2[1]).item(),
            "ref_gap_ulps": (top2[0] - top2[1]).item() / ulp_at(top),
        })
        history = torch.cat([history, ref_tokens[step].view(1, 1).to(dev)], dim=1)
    return rows


# Bounds, in ULPs of the model dtype at the reference's top-logit magnitude.
# Set from measurements (PROGRESS 2026-09-16), not tuned to make tests pass:
#   prefill, cached vs uncached last-position logits:  0.5B fp16 ~1u, 1.5B bf16 0.5u
#   cached/paged decode vs Phase 1, before divergence: 0.5B fp16 19.7u, 1.5B bf16 11.4u
#   kernels on vs kernels off, BOTH with fp32 attention scores, before
#   divergence, 62 sequences each, batch 1-32, 64 tokens:
#       0.5B fp16  median 4.5u, p90 6.6u, max 10.5u
#       1.5B bf16  median 3.0u, p90 3.8u, max 5.7u
# The kernel comparison uses fp32 attention scores on purpose. With scores in
# the model dtype, the kernels-off reference on Qwen2.5-1.5B is itself up to 71u
# from an fp32 ground truth (q.k is computed where bf16's resolution is 1,024),
# and the kernel path measured 101u from it while being the more accurate one.
# With fp32 scores that reference is within 4.9u of truth.
# A real bug -- wrong slot, wrong position, wrong KV head -- produces a
# different distribution, hundreds of ULPs or more.
#   packed quantized path vs round-tripped-fp16 path, same grid (proven
#   bit-identical), before divergence, int8 and int4, batch 2 and 4:
#       0.5B fp16  worst 22.5u      1.5B bf16  worst 25.2u
# That last one is wider than the kernel bound on purpose: the fused
# dequant-matmul kernel accumulates a whole 896/1536-long dot product in its own
# order, where the attention kernels only reorder a per-head reduction.
PREFILL_ULPS = 4.0
PATH_DRIFT_ULPS = 32.0
KERNEL_DRIFT_ULPS = 16.0        # ~1.5x the worst controlled measurement (10.5u)
QUANT_PATH_DRIFT_ULPS = 40.0    # ~1.6x the worst controlled measurement (25.2u)


def max_diff_ulps(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max |got - ref| in ULPs of the model dtype at ref's largest |value|."""
    ref = ref.float()
    return ((got.float() - ref).abs().max().item()
            / ulp_at(ref.abs().max().item()))


@torch.no_grad()
def drift_between(logits_a: list, logits_b: list, row: int, upto: int) -> tuple:
    """Worst per-step drift between two recorded paths, steps 0..upto inclusive.
    Returns (worst_ulps, at_step)."""
    worst, at = 0.0, 0
    for s in range(upto + 1):
        u = max_diff_ulps(logits_a[s][row], logits_b[s][row])
        if u > worst:
            worst, at = u, s
    return worst, at
