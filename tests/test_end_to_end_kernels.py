"""Phase 3 wrap-up: does the engine still produce the right tokens with the
custom kernels switched on?

Individual kernel parity (tests/test_kernels.py) is necessary but not
sufficient. Four kernels are correct in isolation; this file asks whether the
engine wired to use all four still generates the same text. That is a different
question, because errors compose across 24 layers and a decode loop feeds its
own output back in — a divergence at step 3 changes every token after it.

WHAT "THE SAME" MEANS HERE
--------------------------
Gotcha #2 established the project's standard: an exact match is not required
where the structure legitimately differs, but any divergence must be a
demonstrated NEAR-TIE — the two candidate tokens separated by a negligible logit
gap, so the model was genuinely undecided rather than wrong.

TWO of the four kernels are bit-identical to their references and two are not,
which is worth being precise about because the first draft of this file assumed
three were and failed:

  - SwiGLU and RoPE: 0 ULP, 100% exact. Elementwise, no reduction, no
    reordering. They cannot move a token, and the attribution test below
    confirms they contribute exactly zero drift.
  - RMSNorm: <= 2 ULP, >= 99% exact (Gotcha #3). It sums 896 squares through a
    warp-tree reduction while PyTorch uses its own order, and fp addition is not
    associative. Over 24 layers this compounds into a ~4e-02 logit shift — the
    same scale Gotcha #1 documents for eager-vs-SDPA.
  - Decode attention: not bit-identical and cannot be (Gotcha #16).

So divergences are expected to be rare and, where they occur, to be near-ties.
The test reports the count and the gap rather than asserting zero, and the bar is
on the GAP, not on the count. Note also that one flipped token makes every later
token in that sequence differ, so a divergence COUNT is a cascade, not a tally of
independent errors.
"""
from __future__ import annotations

import pytest
import torch

from nano_infer import config as cfg
from nano_infer import model as M

kernels = pytest.importorskip("nano_infer.kernels")

PROMPT_LEN = 12
NEW_TOKENS = 40
NEAR_TIE_GAP = 0.05        # fp16 logit gap below which a flip is a coin toss


@pytest.fixture(scope="module")
def loaded():
    weights = M.load_weights()
    cf = M.QwenConfig()
    kernels.load()          # pay the JIT compile once, outside any timing
    return weights, cf


def _prompt(batch: int, seed: int = 0) -> torch.Tensor:
    """Deterministic in-vocabulary token ids. Real text is not needed — the
    question is whether two code paths agree, not what the model says."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(1000, 5000, (batch, PROMPT_LEN), generator=g).to(cfg.DEVICE)


def test_kernel_toggle_is_off_by_default_and_scoped():
    """The flag must default off, and `using_kernels` must restore it — a
    benchmark that leaked the setting would silently mislabel every later run."""
    assert M.kernels_enabled() is False
    with M.using_kernels(True):
        assert M.kernels_enabled() is True
        with M.using_kernels(False):
            assert M.kernels_enabled() is False
        assert M.kernels_enabled() is True
    assert M.kernels_enabled() is False


def test_phase1_reference_ignores_the_kernel_flag(loaded):
    """`model.py`'s layering rule: the Phase 1 functions are the answer key and
    are never routed through the kernels, even with the flag on. If this fails,
    the project has lost its independent reference."""
    weights, cf = loaded
    torch.manual_seed(0)
    x = torch.randn(2, 9, cf.hidden_size, dtype=cfg.DTYPE, device=cfg.DEVICE)
    w = weights["model.layers.0.input_layernorm.weight"]

    off = M.rms_norm(x, w, cf.rms_norm_eps)
    with M.using_kernels(True):
        on = M.rms_norm(x, w, cf.rms_norm_eps)
        mlp_on = M.mlp(x, weights, 0)                 # default use_kernels=False
    mlp_off = M.mlp(x, weights, 0)

    assert torch.equal(off, on), "rms_norm changed when the kernel flag was set"
    assert torch.equal(mlp_off, mlp_on), "mlp defaulted to the kernel path"


@pytest.mark.parametrize("batch", [1, 4])
def test_paged_generation_matches_with_kernels_on(loaded, batch):
    """The full acceptance question: same tokens out, with all four kernels in."""
    weights, cf = loaded
    ids = _prompt(batch)

    baseline = M.generate_paged(ids, weights, cf, NEW_TOKENS)
    with M.using_kernels(True):
        with_kernels = M.generate_paged(ids, weights, cf, NEW_TOKENS)

    total = baseline.numel()
    mismatch = (baseline != with_kernels)
    n_diff = int(mismatch.sum())
    print(f"\n[e2e paged batch {batch}] {n_diff}/{total} tokens differ "
          f"({n_diff / total * 100:.2f}%)")

    if n_diff == 0:
        return

    # Every divergence must be a demonstrated near-tie. Re-run the reference to
    # the first differing step and measure the top-1/top-2 gap there.
    b_idx, t_idx = [int(i) for i in mismatch.nonzero()[0]]
    prefix = torch.cat([ids[b_idx:b_idx + 1],
                        baseline[b_idx:b_idx + 1, :t_idx]], dim=1)
    from nano_infer.cache import PagedKVCache
    cache = PagedKVCache(cf.num_layers, (prefix.shape[1] + 8) // 16 + 4, 16,
                         cf.num_kv_heads, cf.head_dim,
                         dtype=cfg.DTYPE, device=cfg.DEVICE)
    sid = [cache.add_sequence()]
    rope = M.build_rope_cache(prefix.shape[1] + 2, cf.head_dim, cf.rope_theta,
                              device=cfg.DEVICE, dtype=cfg.DTYPE)
    start = torch.zeros(1, dtype=torch.long, device=cfg.DEVICE)
    step_logits = M.forward_paged(prefix, weights, cf, cache, sid, start, rope)[0]
    top2 = step_logits.float().topk(2).values
    gap = float(top2[0] - top2[1])
    print(f"  first divergence at batch {b_idx} step {t_idx}: "
          f"top-1/top-2 gap {gap:.4f}")

    assert n_diff / total <= 0.10, (
        f"{n_diff / total * 100:.1f}% of tokens diverged — too many to be "
        "explained by near-ties")
    assert gap < NEAR_TIE_GAP, (
        f"divergence at step {t_idx} had a decisive logit gap of {gap:.4f}; "
        "that is a wrong answer, not a coin toss")


def test_decode_kernel_is_not_used_during_prefill(loaded, monkeypatch):
    """Kernel 4 handles exactly ONE query token against the cached past, which is
    what makes the causal mask unnecessary rather than optional. Route prefill
    through it and every prompt token would attend to its own future.

    Asserted structurally by counting calls, not inferred from numerics: a
    numeric check would also have to explain RMSNorm's reduction-order drift
    (see the test below), and conflating the two would make a real routing bug
    look like rounding.
    """
    weights, cf = loaded
    mod = kernels.load()
    calls = []
    real = mod.decode_attention_forward

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "decode_attention_forward", spy)

    from nano_infer.cache import PagedKVCache
    ids = _prompt(2, seed=3)
    cache = PagedKVCache(cf.num_layers, 16, 16, cf.num_kv_heads, cf.head_dim,
                         dtype=cfg.DTYPE, device=cfg.DEVICE)
    sids = [cache.add_sequence() for _ in range(2)]
    rope = M.build_rope_cache(PROMPT_LEN + 4, cf.head_dim, cf.rope_theta,
                              device=cfg.DEVICE, dtype=cfg.DTYPE)

    with M.using_kernels(True):
        # prefill: n = PROMPT_LEN > 1
        start = torch.zeros(2, dtype=torch.long, device=cfg.DEVICE)
        logits = M.forward_paged(ids, weights, cf, cache, sids, start, rope)
        after_prefill = len(calls)

        # one decode step: n == 1
        nxt = logits.argmax(dim=-1).unsqueeze(1)
        start = torch.full((2,), PROMPT_LEN, dtype=torch.long, device=cfg.DEVICE)
        M.forward_paged(nxt, weights, cf, cache, sids, start, rope)
        after_decode = len(calls)

    print(f"\n[kernel 4 routing] prefill calls {after_prefill}, "
          f"decode calls {after_decode - after_prefill} "
          f"(expected 0 and {cf.num_layers})")
    assert after_prefill == 0, (
        f"the decode kernel was called {after_prefill} times during prefill; "
        "prompt tokens would attend to their own future")
    assert after_decode - after_prefill == cf.num_layers, (
        "the decode kernel should run once per layer on a decode step, got "
        f"{after_decode - after_prefill}")


def test_prefill_divergence_is_attributable_to_rmsnorm(loaded, monkeypatch):
    """Enabling the kernels moves prefill logits by ~4e-02. Attribute it.

    SwiGLU and RoPE are bit-identical to their references (0 ULP, 100% exact).
    RMSNorm is NOT and cannot be: it sums 896 squares through a warp-tree
    reduction while PyTorch uses its own order, and floating-point addition is
    not associative. Its bar is <= 2 ULP with >= 99% of elements exact
    (Gotcha #3) — which compounds over 24 layers into a small logit shift,
    exactly the scale Gotcha #1 documents for eager-vs-SDPA.

    The experiment: hold the kernel flag ON but route RMSNorm back to the
    reference. If RMSNorm is the whole story, the difference must vanish
    completely — and it does, which also proves SwiGLU and RoPE contribute
    nothing and that prefill never touches kernel 4.
    """
    weights, cf = loaded
    ids = _prompt(2, seed=3)

    from nano_infer.cache import PagedKVCache

    def prefill_logits():
        cache = PagedKVCache(cf.num_layers, 16, 16, cf.num_kv_heads, cf.head_dim,
                             dtype=cfg.DTYPE, device=cfg.DEVICE)
        sids = [cache.add_sequence() for _ in range(2)]
        rope = M.build_rope_cache(PROMPT_LEN + 2, cf.head_dim, cf.rope_theta,
                                  device=cfg.DEVICE, dtype=cfg.DTYPE)
        start = torch.zeros(2, dtype=torch.long, device=cfg.DEVICE)
        return M.forward_paged(ids, weights, cf, cache, sids, start, rope)

    off = prefill_logits()
    with M.using_kernels(True):
        on = prefill_logits()
    drift = (off.float() - on.float()).abs().max().item()

    # same again, but with RMSNorm forced back onto the reference
    monkeypatch.setattr(M, "_rms", M.rms_norm)
    with M.using_kernels(True):
        on_no_rms = prefill_logits()
    residual = (off.float() - on_no_rms.float()).abs().max().item()

    print(f"\n[prefill drift] all kernels {drift:.3e}; "
          f"with RMSNorm on the reference {residual:.3e}")
    print(f"  argmax tokens unchanged: "
          f"{bool((off.argmax(-1) == on.argmax(-1)).all())}")

    assert residual == 0.0, (
        f"{residual:.3e} of drift remains with RMSNorm on the reference — "
        "SwiGLU/RoPE are supposed to be bit-identical, or prefill is reaching "
        "the decode kernel")
    assert drift < 0.5, (
        f"RMSNorm's reduction-order drift compounded to {drift:.3e} over "
        f"{cf.num_layers} layers, far more than ~1 ULP per layer explains")
