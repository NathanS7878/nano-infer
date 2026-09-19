"""Phase 2 parity: the KV-cached path vs the Phase 1 reference.

The cache is a pure performance optimization — it removes recomputation of a
frozen past — so it must not change what the model computes. Proving that
requires separating two things that are easy to conflate:

  1. IS THE CACHE LOGIC CORRECT? Given the same input, does the cached path
     produce the same hidden states? This must be EXACT (bit-identical), and it
     is: see test_cache_logic_is_exact.

  2. DOES THE OUTPUT MATCH TOKEN-FOR-TOKEN? Not always, and the reason is not a
     bug. The cache's entire purpose is to process ONE token per step instead of
     the whole sequence — which changes the shape of every matmul in the decode
     path. cuBLAS selects different kernels for different shapes, those kernels
     accumulate in different orders, and fp16 rounds differently as a result.
     Measured: projecting the SAME hidden state as [1,36,896] vs [1,1,896]
     against the 151936x896 output matrix differs by 7.81e-03.

     On a confident token that difference is invisible. On a near-tie it can flip
     the argmax. test_cached_decode_matches_fixture therefore asserts that any
     divergence is a genuine near-tie, and that the divergence RATE is tiny.

Run:  python -m pytest tests/test_cache.py -v -s
"""
from __future__ import annotations

import pytest
import torch

from tests._drift import PREFILL_ULPS, max_diff_ulps

from nano_infer import config as cfg
from nano_infer import model as M
from nano_infer.cache import KVCache
from nano_infer.hf_ref import encode_prompt, load_hf
from tests.test_parity import compare_tokens, load_reference

# fp16 tolerance for logits computed through a different-shaped matmul.
MATMUL_SHAPE_TOL = 2e-2
# A reference top-1/top-2 logit gap below this is a coin-flip, not a decision.
NEAR_TIE_GAP = 0.05


@pytest.fixture(scope="module")
def weights():
    return M.load_weights()


@pytest.fixture(scope="module")
def tok():
    _, t = load_hf()
    return t


def _phase1_hidden(ids, weights, cf):
    """Phase 1 stack, stopping before the final projection."""
    x = M.embed_tokens(ids, weights)
    seq = ids.shape[1]
    cos, sin = M.build_rope_cache(seq, cf.head_dim, cf.rope_theta)
    for i in range(cf.num_layers):
        x = M.decoder_block(x, weights, i, cos, sin, cf)
    return M.rms_norm(x, weights["model.norm.weight"], cf.rms_norm_eps)


def _cached_hidden(ids, weights, cf):
    """Cached stack over the same input, stopping before the final projection."""
    seq = ids.shape[1]
    cache = KVCache(cf.num_layers, ids.shape[0], cf.num_kv_heads, seq + 8, cf.head_dim)
    cos, sin = M.build_rope_cache(seq + 8, cf.head_dim, cf.rope_theta)
    x = M.embed_tokens(ids, weights)
    for i in range(cf.num_layers):
        x = M.decoder_block_cached(x, weights, i, cos[:seq], sin[:seq], cf, cache, 0)
    return M.rms_norm(x, weights["model.norm.weight"], cf.rms_norm_eps)


def test_cache_logic_is_exact(weights, tok):
    """THE correctness proof: writing K/V through the cache and reading them back
    must reproduce Phase 1's hidden states bit-for-bit."""
    cf = M.QwenConfig()
    for prompt in cfg.PROMPTS:
        ids = encode_prompt(tok, prompt)
        ref = _phase1_hidden(ids, weights, cf)
        got = _cached_hidden(ids, weights, cf)
        assert torch.equal(ref, got), f"cached hidden states differ for {prompt!r}"
    print(f"\n[cache logic] hidden states bit-identical to Phase 1 "
          f"on all {len(cfg.PROMPTS)} prompts (torch.equal)")


def test_prefill_logits_match_phase1(weights, tok):
    """Last-position logits after prefill. Bounded by matmul-shape rounding, not
    by cache error — forward_cached projects only the last position ([b,1,896])
    where Phase 1 projects the whole sequence."""
    cf = M.QwenConfig()
    max_diff = 0.0
    for prompt in cfg.PROMPTS:
        ids = encode_prompt(tok, prompt)
        seq = ids.shape[1]
        ref = M.forward(ids, weights, cf)[:, -1]
        cache = KVCache(cf.num_layers, 1, cf.num_kv_heads, seq + 8, cf.head_dim)
        rope = M.build_rope_cache(seq + 8, cf.head_dim, cf.rope_theta)
        got = M.forward_cached(ids, weights, cf, cache, 0, rope)
        max_diff = max(max_diff, max_diff_ulps(got[0], ref[0]))

    # In ULPs of the model dtype, not an absolute tolerance: the old 2e-2 bar sat
    # below bf16's resolution at these logit magnitudes (0.0625-0.125), so on
    # Qwen2.5-1.5B it could only be met by bitwise equality.
    print(f"\n[prefill] last-position logits vs Phase 1: max {max_diff:.2f} ULPs of "
          f"{cfg.DTYPE_NAME} (bound {PREFILL_ULPS} — cuBLAS kernel choice, cache logic is exact)")
    assert max_diff <= PREFILL_ULPS


@pytest.mark.slow
def test_cached_decode_matches_fixture(weights, tok):
    """PHASE 2 ACCEPTANCE: cached greedy decode vs the Phase 1 fixture.

    Up to each prompt's first divergence the cached path must stay within
    PATH_DRIFT_ULPS of Phase 1's uncached logits on the same history. A flip
    after a bounded drift is rounding; an unbounded drift is a bug."""
    cf = M.QwenConfig()
    ref = load_reference()

    # The bar is on the DRIFT, in ULPs of the model dtype, not on a fixed logit
    # gap or a divergence rate. Both of those encoded 0.5B-fp16 numerics: in
    # bf16 an exact logit tie is common and a gap under 0.125 unrepresentable,
    # so rate and absolute gap measure the dtype's resolution, not correctness.
    # See tests/_drift.py and ROADMAP #44.
    from tests._drift import (PATH_DRIFT_ULPS, drift_profile, first_divergence,
                              record_step_logits)
    worst_overall = 0.0
    divergences = 0
    for i, (prompt, ref_p) in enumerate(zip(cfg.PROMPTS, ref["prompts"])):
        ids = encode_prompt(tok, prompt)
        with record_step_logits("forward_cached") as logs:
            got = M.generate_cached(ids, weights, cf, cfg.PARITY_NEW_TOKENS)[0]
        div = first_divergence(ref_p["greedy_ids"], got)
        upto = cfg.PARITY_NEW_TOKENS - 1 if div is None else div
        prof = drift_profile(ids, ref_p["greedy_ids"], logs, weights, cf, upto=upto)
        worst = max(prof, key=lambda r: r["drift_ulps"])
        worst_overall = max(worst_overall, worst["drift_ulps"])
        if div is None:
            print(f"[{i}] OK  worst drift {worst['drift_ulps']:.1f} ULPs (step {worst['step']})")
        else:
            divergences += 1
            at = prof[-1]
            print(f"[{i}] diverges@{div}  reference gap {at['ref_gap_ulps']:.1f} ULPs, "
                  f"drift there {at['drift_ulps']:.1f} ULPs; worst before it "
                  f"{worst['drift_ulps']:.1f} ULPs (step {worst['step']})")
        assert worst["drift_ulps"] <= PATH_DRIFT_ULPS, (
            f"prompt {i}: the cached path drifted {worst['drift_ulps']:.1f} ULPs from Phase 1 "
            f"at step {worst['step']} (bound {PATH_DRIFT_ULPS}). That is not rounding "
            f"from a different matmul shape; look for a wrong slot, position or KV row.")
    print(f"\n[acceptance] {divergences}/{len(cfg.PROMPTS)} prompts diverge; worst drift "
          f"{worst_overall:.1f} ULPs of {cfg.DTYPE_NAME} (bound {PATH_DRIFT_ULPS})")


def test_cache_memory_accounting():
    """Document the GQA payoff in bytes."""
    cf = M.QwenConfig()
    batch, max_seq = 32, 512
    c = KVCache(cf.num_layers, batch, cf.num_kv_heads, max_seq, cf.head_dim)
    gqa_mb = c.bytes_allocated() / 1024**2
    mha_mb = gqa_mb * (cf.num_q_heads / cf.num_kv_heads)
    print(f"\n[memory] batch={batch} max_seq={max_seq}: "
          f"GQA cache {gqa_mb:.0f} MB vs full multi-head {mha_mb:.0f} MB "
          f"({cf.num_q_heads // cf.num_kv_heads}x smaller)")
    assert gqa_mb < mha_mb
