"""Phase 4: quantization correctness.

THE BAR IS DIFFERENT HERE, AND IT IS STATED BEFORE THE KERNEL EXISTS
--------------------------------------------------------------------
Kernels 1-3 were held to bit-identity or a 2-ULP bound. None of that applies to
quantization: it is lossy BY DESIGN, so "how close to fp16" is not a pass/fail
question but the entire subject of the phase.

What can still be asserted exactly, and is:

  1. **Round-trip structure.** Dequantizing a quantized tensor must reproduce
     the quantization grid exactly — every dequantized value must be a
     representable point of its own scheme. That is bit-exact and catches
     packing bugs, nibble-order bugs, and scale/zero misalignment.
  2. **Error bounds from first principles.** A quantizer with step `s` has a
     round-to-nearest error of at most s/2, per element, always. That is
     arithmetic, not a tolerance, so it is asserted as an equality-grade bound
     rather than a fudge factor.
  3. **Ordering.** INT8 must be strictly more accurate than group-wise INT4 on
     the same tensor. If it ever is not, something is wrong with one of them.
  4. **Storage.** The claimed bits/weight must match the bytes actually held,
     metadata included. This is where "4x smaller" claims usually quietly cheat.

The QUALITY question — what this costs in perplexity and generated text — is not
a unit test. It is measured in bench/perplexity.py and reported with numbers,
per spec rule 5.
"""
from __future__ import annotations

import pytest
import torch

from nano_infer import config as cfg
from nano_infer import model as M
from nano_infer import quant as Q


def test_int8_dequantizes_onto_its_own_grid():
    """Every dequantized value must be exactly q * scale — no drift from packing
    or dtype juggling."""
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=cfg.DTYPE, device=cfg.DEVICE)
    t = Q.quantize_int8(w)
    deq = t.dequantize()

    expected = (t.q.to(torch.float16) * t.scale).to(cfg.DTYPE)
    assert torch.equal(deq, expected), "INT8 dequantization is not q * scale"
    assert t.q.dtype == torch.int8 and t.q.abs().max() <= 127


def test_int8_error_is_bounded_by_half_a_step():
    """Round-to-nearest cannot err by more than half the quantization step.
    This is arithmetic, not a tolerance — a violation is a bug, not noise."""
    torch.manual_seed(0)
    w = torch.randn(128, 512, dtype=cfg.DTYPE, device=cfg.DEVICE) * 3.0
    t = Q.quantize_int8(w)
    deq = t.dequantize()
    err = (w.float() - deq.float()).abs()
    half_step = t.scale.float() / 2

    # The bound is half a step PLUS one fp16 rounding of the reconstruction,
    # because the dequantized value is itself stored in fp16. Both terms are
    # derived, not tuned: round-to-nearest gives s/2, and fp16 has a 2^-11
    # relative representation error.
    bound = half_step + deq.float().abs() * 2 ** -11 * 1.01
    violations = int((err > bound).sum())
    print(f"\n[int8] max err {err.max():.3e}, max half-step {half_step.max():.3e}, "
          f"violations {violations}")
    assert violations == 0, f"{violations} elements exceeded half a step + fp16 ulp"


def test_int4_packing_round_trips_exactly():
    """Two nibbles per byte, low nibble first. A swapped order still produces
    plausible numbers, so it is checked directly."""
    torch.manual_seed(0)
    q = torch.randint(0, 16, (32, 128), dtype=torch.uint8, device=cfg.DEVICE)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    back = Q.unpack_int4(packed, 128)

    assert torch.equal(q, back), "INT4 pack/unpack is not a round trip"
    assert packed.shape == (32, 64), "packing did not halve the last dimension"
    # explicitly pin the nibble convention
    assert int(packed[0, 0]) == int(q[0, 0]) | (int(q[0, 1]) << 4)


def test_int4_dequantizes_onto_its_own_grid():
    torch.manual_seed(0)
    w = torch.randn(32, 256, dtype=cfg.DTYPE, device=cfg.DEVICE)
    t = Q.quantize_int4(w, group=128)

    q = Q.unpack_int4(t.packed, 256).reshape(32, 2, 128).float()
    zero = t.zero.reshape(32, 2, 1).float()
    scale = t.scale.reshape(32, 2, 1).float()
    expected = ((q - zero) * scale).reshape(32, 256).to(cfg.DTYPE)

    assert torch.equal(t.dequantize(), expected), \
        "INT4 dequantization is not (q - zero) * scale"
    assert int(q.max()) <= 15 and int(q.min()) >= 0, "codes outside 0..15"


def test_int4_error_is_bounded_by_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(64, 512, dtype=cfg.DTYPE, device=cfg.DEVICE) * 2.0
    t = Q.quantize_int4(w, group=128)
    deq = t.dequantize()
    err = (w.float() - deq.float()).abs().reshape(64, 4, 128)
    half_step = t.scale.float().reshape(64, 4, 1) / 2

    bound = half_step + deq.float().abs().reshape(64, 4, 128) * 2 ** -11 * 1.01
    violations = int((err > bound).sum())
    print(f"\n[int4] max err {err.max():.3e}, max half-step {half_step.max():.3e}, "
          f"violations {violations}")
    assert violations == 0, f"{violations} elements exceeded half a step + fp16 ulp"


def test_int4_group_boundaries_use_their_own_scale():
    """Group-wise means each group of 128 gets its OWN scale. A tensor whose
    groups differ wildly in magnitude exposes a kernel that used one scale for
    the whole row — the small group would be crushed to zero."""
    w = torch.zeros(4, 256, dtype=cfg.DTYPE, device=cfg.DEVICE)
    w[:, :128] = 100.0                      # huge first group
    w[:, 128:] = torch.linspace(-0.01, 0.01, 128, device=cfg.DEVICE).to(cfg.DTYPE)

    deq = Q.quantize_int4(w, group=128).dequantize()
    small = deq[:, 128:].float()
    assert small.abs().max() > 1e-3, (
        "the small group was crushed to zero — a single row-wide scale was used "
        "instead of per-group scales")
    rel = (w[:, 128:].float() - small).abs().max() / 0.01
    print(f"\n[int4 groups] small-group relative error {rel:.3f}")
    assert rel < 0.2


def test_int8_is_more_accurate_than_int4():
    """Ordering sanity: 8 bits per weight must beat 4 bits plus metadata."""
    weights = M.load_weights()
    w = weights["model.layers.12.mlp.gate_proj.weight"]

    e8 = (w.float() - Q.quantize_int8(w).dequantize().float()).abs().mean()
    e4 = (w.float() - Q.quantize_int4(w).dequantize().float()).abs().mean()
    print(f"\n[ordering] mean abs err int8 {e8:.3e}, int4 {e4:.3e}, "
          f"ratio {e4 / e8:.1f}x")
    assert e8 < e4, "INT8 is not more accurate than INT4 — check the schemes"


def test_bits_per_weight_counts_the_metadata():
    """The honest storage number. Group-wise INT4 is not 4.0 bits/weight: one
    fp16 scale and one uint8 zero per 128 weights is 3 bytes per 64 packed
    bytes. A claim of 4.0 here would be hiding the metadata."""
    torch.manual_seed(0)
    w = torch.randn(512, 1024, dtype=cfg.DTYPE, device=cfg.DEVICE)
    t = Q.quantize_int4(w, group=128)

    bpw = t.bits_per_weight()
    expected = (t.packed.numel() + t.scale.numel() * 2 + t.zero.numel()) * 8 / w.numel()
    print(f"\n[storage] int4 effective {bpw:.3f} bits/weight "
          f"(payload 4.0 + metadata {bpw - 4.0:.3f})")
    assert abs(bpw - expected) < 1e-9
    assert 4.0 < bpw < 4.5, f"expected just over 4 bits/weight, got {bpw:.3f}"


def test_quantize_weights_leaves_embeddings_and_norms_alone():
    """Only the 2-D projections are quantized. The embedding is tied to the
    output head, so an error there is applied twice; norms and biases are tiny
    and quality-critical. If this ever changes it must be a deliberate decision,
    not a silent one."""
    weights = M.load_weights()
    quantized, stats = Q.quantize_weights(weights, "int8")

    assert torch.equal(quantized["model.embed_tokens.weight"],
                       weights["model.embed_tokens.weight"]), "embedding was quantized"
    assert torch.equal(quantized["model.norm.weight"],
                       weights["model.norm.weight"]), "final norm was quantized"
    assert torch.equal(quantized["model.layers.0.self_attn.q_proj.bias"],
                       weights["model.layers.0.self_attn.q_proj.bias"]), "bias was quantized"
    assert not torch.equal(quantized["model.layers.0.mlp.gate_proj.weight"],
                           weights["model.layers.0.mlp.gate_proj.weight"]), \
        "gate_proj was NOT quantized"

    # 7 projections x 24 layers
    assert stats["quantized_tensors"] == 7 * 24, stats["quantized_tensors"]
    print(f"\n[coverage] {stats['quantized_tensors']} tensors quantized, "
          f"{stats['bytes_quantizable_original']/1e6:.1f} MB of "
          f"{stats['bytes_original']/1e6:.1f} MB "
          f"({stats['bytes_quantizable_original']/stats['bytes_original']*100:.1f}%)")


@pytest.mark.parametrize("mode,lo,hi", [("int8", 1.5, 1.7), ("int4", 2.0, 2.3)])
def test_whole_model_compression_is_what_we_claim(mode, lo, hi):
    """The model-level number, not the per-tensor one. Because the embedding
    stays fp16, INT4 gives ~2.2x on the whole model even though the tensors it
    touches shrink ~3.8x. Both numbers are reported; only one of them is 'the
    model is N times smaller'."""
    weights = M.load_weights()
    _, s = Q.quantize_weights(weights, mode)
    print(f"\n[{mode}] whole model {s['bytes_original']/1e6:.1f} -> "
          f"{s['bytes_quantized']/1e6:.1f} MB = {s['compression']:.2f}x   |   "
          f"quantized tensors alone {s['quantizable_compression']:.2f}x   |   "
          f"{s['bits_per_weight']:.2f} bits/weight   |   "
          f"mean rel err {s['mean_rel_err']:.2e}")
    assert lo < s["compression"] < hi, (
        f"{mode} whole-model compression {s['compression']:.2f}x outside the "
        f"expected {lo}-{hi}x — the set of quantized tensors changed")
