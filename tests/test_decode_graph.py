"""Static, sync-free decode and its CUDA-graph capture (nano_infer/decode_graph.py).

Three separate claims, tested separately, because they fail for different
reasons and a single end-to-end token comparison would blur them:

  1. CAPTURE IS EXACT. The graph replays the same kernel launches as the static
     step run eagerly, so the two must agree bit for bit — kernels on or off. A
     difference here would mean capture changed the computation (stale buffer,
     wrong stream, a host value baked in at capture time), not numerics.

  2. THE STATIC STEP IS THE SAME ENGINE. Against model.generate_paged:
       - kernels ON: token-identical. The decode kernel reads `lengths` and
         never materialises the score matrix, so the one structural difference
         (the read table is final_len wide from the first step instead of
         growing) cannot reach its arithmetic.
       - kernels OFF: the PyTorch path DOES see the width, as the shape of
         q @ K^T, and a changed matmul shape changes fp16 rounding (Gotcha #2).
         So the bar there is the project's standard: every divergence must be a
         demonstrated near-tie. The attribution was done by experiment, not
         assumed — see test_width_is_the_only_cause_of_divergence.

  3. THE STEP NEVER SYNCHRONISES. That is the property the whole design exists
     for and the precondition for capture. Asserted directly with PyTorch's
     sync-debug mode set to raise, so a future edit that sneaks an `int(...)` or
     `.item()` into the step fails loudly instead of silently re-serialising the
     loop.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nano_infer import config as cfg
from nano_infer import model as M
from nano_infer.cache import PagedKVCache, SlotPlan
from nano_infer.decode_graph import (_StaticDecodeState, _decode_step,
                                     generate_paged_static)

kernels = pytest.importorskip("nano_infer.kernels")

NEAR_TIE_GAP = 0.05        # same bar as tests/test_end_to_end_kernels.py

# (batch, prompt_len, new_tokens). Chosen to cover: batch 1 (per-query-head
# decode kernel), batch >= 4 (head-grouped kernel), a prompt that crosses block
# boundaries, a long context, and the smallest decode (one step).
SHAPES = [(1, 32, 48), (4, 17, 40), (32, 32, 48), (8, 300, 32), (3, 5, 2)]


@pytest.fixture(scope="module")
def loaded():
    weights = M.load_weights()
    cf = M.QwenConfig()
    kernels.load()
    return weights, cf


def _prompt(batch: int, plen: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(1000, 5000, (batch, plen), generator=g).to(cfg.DEVICE)


@pytest.mark.parametrize("use_kernels", [False, True])
@pytest.mark.parametrize("batch,plen,gen", SHAPES)
def test_graph_capture_is_exact(loaded, use_kernels, batch, plen, gen):
    """Claim 1: replaying the graph == running the same step eagerly, bitwise."""
    weights, cf = loaded
    ids = _prompt(batch, plen, seed=batch * 1000 + plen)
    with M.using_kernels(use_kernels):
        eager = generate_paged_static(ids, weights, cf, gen, use_graph=False)
        graph = generate_paged_static(ids, weights, cf, gen, use_graph=True)
    assert graph.shape == (batch, gen)
    assert torch.equal(eager, graph), (
        f"graph replay differs from the eager static step "
        f"({int((eager != graph).sum())} tokens). Capture changed the "
        f"computation -- look for a host value baked in at capture time, a "
        f"buffer reallocated instead of updated in place, or a stream race.")


@pytest.mark.parametrize("batch,plen,gen", SHAPES)
def test_kernels_on_matches_generate_paged_exactly(loaded, batch, plen, gen):
    """Claim 2, kernels on: token-identical to the existing engine."""
    weights, cf = loaded
    ids = _prompt(batch, plen, seed=batch * 1000 + plen)
    with M.using_kernels(True):
        ref = M.generate_paged(ids, weights, cf, gen)
        got = generate_paged_static(ids, weights, cf, gen, use_graph=True)
    assert torch.equal(ref, got), (
        f"{int((ref != got).any(1).sum())} of {batch} sequences diverge with the "
        f"decode kernel on. The kernel should be blind to the read table's "
        f"width, so this is not the known kernels-off near-tie effect.")


def test_kernels_off_divergences_are_near_ties(loaded):
    """Claim 2, kernels off: any divergence from generate_paged is a near-tie.

    Measured on this exact case when it was written: 2 of 32 sequences diverge.
    Sequence 17 at output step 1 with a reference top-1/top-2 gap of 0.0000 (an
    exact tie), sequence 21 at step 62 with a gap of 0.0156 -- both below the
    0.02-0.04 logit shift the width change causes. The bar is on the GAP.

    Each sequence is checked at its FIRST divergent step only. After a flip the
    sequence is conditioned on a different token, so later mismatches are a
    cascade of the first, not independent errors.
    """
    weights, cf = loaded
    b, plen, gen = 32, 32, 64
    g = torch.Generator(device="cpu").manual_seed(0)
    for shape in [(1, 32), (4, 17)]:          # replay the generator to the
        torch.randint(1000, 5000, shape, generator=g)   # originally measured case
    ids = torch.randint(1000, 5000, (b, plen), generator=g).to(cfg.DEVICE)

    recorded = []
    real = M.forward_paged

    def spy(*a, **k):
        out = real(*a, **k)
        recorded.append(out.float().clone())
        return out

    with M.using_kernels(False):
        M.forward_paged = spy
        try:
            ref = M.generate_paged(ids, weights, cf, gen)
        finally:
            M.forward_paged = real
        got = generate_paged_static(ids, weights, cf, gen, use_graph=True)

    diverged = (ref != got).any(1).nonzero().flatten().tolist()
    print(f"\n[static decode, kernels off] {len(diverged)} of {b} sequences diverge")
    for s in diverged:
        t = int((ref[s] != got[s]).nonzero()[0])
        top2 = recorded[t][s].topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"    seq {s}: first divergence at step {t}, reference gap {gap:.4f}")
        assert gap < NEAR_TIE_GAP, (
            f"seq {s} diverged at step {t} with a reference top-1/top-2 gap of "
            f"{gap:.4f}. That is not a near-tie, so it is not explained by the "
            f"read-table width's fp16 rounding.")


def test_width_is_the_only_cause_of_divergence(loaded):
    """Attribute the kernels-off divergence instead of merely bounding it.

    Holds everything fixed except the read table's width, on the same decode
    step, and checks three things in order:
      - the same width twice is bitwise identical (so it is not nondeterminism);
      - width 33 (what generate_paged uses at step 1) vs 96 (what the static
        step uses) changes the layer-0 attention output (so width alone moves
        the arithmetic);
      - the kernel path is blind to width (so kernels-on parity is structural,
        not luck).
    """
    weights, cf = loaded
    b, plen, final = 8, 32, 96

    def layer0_attention(width, use_kernels):
        cache = PagedKVCache(cf.num_layers, b * ((final + 15) // 16) + 4, 16,
                             cf.num_kv_heads, cf.head_dim,
                             dtype=cfg.DTYPE, device=cfg.DEVICE)
        sids = [cache.add_sequence() for _ in range(b)]
        for s in sids:
            cache.ensure_capacity(s, final)
        cos_all, sin_all = M.build_rope_cache(final, cf.head_dim, cf.rope_theta,
                                              device=cfg.DEVICE, dtype=cfg.DTYPE)
        ids = _prompt(b, plen, seed=7)
        with M.using_kernels(use_kernels):
            start = torch.zeros(b, dtype=torch.long, device=cfg.DEVICE)
            first = M.forward_paged(ids, weights, cf, cache, sids, start,
                                    (cos_all, sin_all)).argmax(-1)
            pos = torch.full((b,), plen, device=cfg.DEVICE)
            lengths = pos + 1
            rpos = torch.arange(width, device=cfg.DEVICE).unsqueeze(0).expand(b, width)
            mask = rpos < lengths.unsqueeze(1)
            read = cache._slots(sids, rpos * mask)
            plan = SlotPlan(read.gather(1, pos.unsqueeze(1)).reshape(-1), read, mask)
            allowed = ((torch.arange(width, device=cfg.DEVICE).view(1, 1, 1, width)
                        <= pos.view(b, 1, 1, 1)) & mask.view(b, 1, 1, width))
            x = M.embed_tokens(first.unsqueeze(1), weights)
            h = M._rms(x, weights["model.layers.0.input_layernorm.weight"],
                       cf.rms_norm_eps)
            return M.attention_paged(h, weights, 0, cos_all[pos.unsqueeze(1)],
                                     sin_all[pos.unsqueeze(1)], cf, cache, plan,
                                     allowed, lengths)

    torch_33a = layer0_attention(33, False)
    torch_33b = layer0_attention(33, False)
    torch_96 = layer0_attention(96, False)
    kern_33 = layer0_attention(33, True)
    kern_96 = layer0_attention(96, True)

    shift = (torch_33a.float() - torch_96.float()).abs().max().item()
    print(f"\n[width attribution] PyTorch path 33 vs 96: max abs {shift:.3e}; "
          f"kernel path 33 vs 96 equal: {torch.equal(kern_33, kern_96)}")

    assert torch.equal(torch_33a, torch_33b), "same width twice is not deterministic"
    assert not torch.equal(torch_33a, torch_96), (
        "width no longer changes the PyTorch attention output -- if cuBLAS now "
        "picks the same kernel for both shapes, the kernels-off near-tie test "
        "may be able to become an exact-match test")
    assert torch.equal(kern_33, kern_96), (
        "the decode kernel's output depends on the read table's width; "
        "kernels-on exact parity would then be coincidence, not structure")


@pytest.mark.parametrize("use_kernels", [False, True])
def test_decode_step_never_synchronises(loaded, use_kernels):
    """Claim 3: the step issues no GPU->host synchronisation.

    forward_paged issued 65 per decode step at batch 32 (two int(lengths[i]) per
    sequence plus int(lengths.max())). Deleting them is what made the step
    capturable at all, so this guards the property directly rather than trusting
    that a graph capture would have complained.
    """
    weights, cf = loaded
    b, plen, gen = 4, 20, 8
    final = plen + gen
    cache = PagedKVCache(cf.num_layers, b * ((final + 15) // 16) + 4, 16,
                         cf.num_kv_heads, cf.head_dim,
                         dtype=cfg.DTYPE, device=cfg.DEVICE)
    sids = [cache.add_sequence() for _ in range(b)]
    for s in sids:
        cache.ensure_capacity(s, final)
    cos_all, sin_all = M.build_rope_cache(final, cf.head_dim, cf.rope_theta,
                                          device=cfg.DEVICE, dtype=cfg.DTYPE)
    ids = _prompt(b, plen, seed=3)
    with M.using_kernels(use_kernels):
        start = torch.zeros(b, dtype=torch.long, device=cfg.DEVICE)
        first = M.forward_paged(ids, weights, cf, cache, sids, start,
                                (cos_all, sin_all)).argmax(-1)
        st = _StaticDecodeState(cache, sids, first, plen, gen, final)
        _decode_step(st, weights, cf, cache, cos_all, sin_all)   # warm, unguarded
        torch.cuda.synchronize()

        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(3):
                _decode_step(st, weights, cf, cache, cos_all, sin_all)
        finally:
            torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()


def test_one_new_token_needs_no_decode(loaded):
    """max_new_tokens == 1 is prefill only; no state, no capture, no replay."""
    weights, cf = loaded
    ids = _prompt(2, 9, seed=11)
    timings = {}
    with M.using_kernels(True):
        ref = M.generate_paged(ids, weights, cf, 1)
        got = generate_paged_static(ids, weights, cf, 1, timings=timings)
    assert torch.equal(ref, got)
    assert timings["decode_steps"] == 0 and timings["capture_s"] == 0.0


# ---------------------------------------------------------------------------
# DecodeGraphRunner: capture once, serve many prompts of one shape
# ---------------------------------------------------------------------------

from nano_infer.decode_graph import DecodeGraphRunner  # noqa: E402


@pytest.mark.parametrize("use_kernels", [False, True])
def test_runner_reuse_matches_fresh_runs(loaded, use_kernels):
    """One runner, three DIFFERENT prompts, each equal to a fresh one-shot run.

    This is the test that would catch stale state leaking between prompts. The
    runner reuses its KV pool without zeroing it, on the argument that prefill
    overwrites [0, prompt_len) and every decode step writes its slot before any
    read. If that argument were wrong, the second and third prompts would attend
    to the first prompt's KV and still produce fluent, wrong tokens -- so the
    comparison is against runs that never shared a pool.

    Also checks the returned tensor is a copy: an earlier result must not change
    when the runner serves the next prompt.
    """
    weights, cf = loaded
    batch, plen, gen = 4, 24, 40
    prompts = [_prompt(batch, plen, seed=s) for s in (101, 202, 303)]
    with M.using_kernels(use_kernels):
        runner = DecodeGraphRunner(weights, cf, batch, plen, gen)
        results = [runner.generate(p) for p in prompts]
        first_snapshot = results[0].clone()
        fresh = [generate_paged_static(p, weights, cf, gen) for p in prompts]

    assert runner.captures == 1, f"captured {runner.captures} times, expected once"
    for i, (got, want) in enumerate(zip(results, fresh)):
        assert torch.equal(got, want), (
            f"prompt {i} through a reused runner differs from a fresh run in "
            f"{int((got != want).sum())} tokens -- state from an earlier prompt "
            f"is leaking into the graph")
    assert torch.equal(results[0], first_snapshot), (
        "an earlier result changed after later calls: generate() returned the "
        "runner's output buffer instead of a copy")
    assert not torch.equal(results[0], results[1]), (
        "different prompts produced identical output; the test is not exercising "
        "stale state")


def test_runner_pays_capture_once(loaded):
    """The point of the runner: the second call of a shape has no capture."""
    weights, cf = loaded
    batch, plen, gen = 2, 16, 24
    with M.using_kernels(True):
        runner = DecodeGraphRunner(weights, cf, batch, plen, gen)
        t1, t2 = {}, {}
        runner.generate(_prompt(batch, plen, seed=1), timings=t1)
        runner.generate(_prompt(batch, plen, seed=2), timings=t2)
    print(f"\n[runner] capture first call {t1['capture_s']:.3f} s, "
          f"second call {t2['capture_s']:.3f} s")
    assert t1["capture_s"] > 0.0
    assert t2["capture_s"] == 0.0


def test_runner_refuses_a_kernel_setting_it_was_not_captured_with(loaded):
    """A graph replays the path it recorded. Flipping the global kernel switch
    after capture would silently keep running the old path, so it must raise."""
    weights, cf = loaded
    batch, plen, gen = 2, 16, 8
    with M.using_kernels(True):
        runner = DecodeGraphRunner(weights, cf, batch, plen, gen)
        runner.generate(_prompt(batch, plen, seed=5))
    with M.using_kernels(False):
        with pytest.raises(RuntimeError, match="captured with kernels on"):
            runner.generate(_prompt(batch, plen, seed=6))


def test_runner_refuses_the_wrong_shape(loaded):
    weights, cf = loaded
    runner = DecodeGraphRunner(weights, cf, 2, 16, 8)
    with pytest.raises(ValueError, match="built for"):
        runner.generate(_prompt(3, 16, seed=7))
