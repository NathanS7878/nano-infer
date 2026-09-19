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

    # The bound is half a step PLUS one rounding of the reconstruction in the
    # model dtype, because the dequantized value is itself stored in it. Both
    # terms are derived, not tuned: round-to-nearest gives s/2, and a 16-bit
    # float has an eps/2 relative representation error -- 2^-11 for fp16,
    # 2^-8 for bf16.
    bound = half_step + deq.float().abs() * (torch.finfo(cfg.DTYPE).eps / 2) * 1.01
    violations = int((err > bound).sum())
    print(f"\n[int8] max err {err.max():.3e}, max half-step {half_step.max():.3e}, "
          f"violations {violations}")
    assert violations == 0, f"{violations} elements exceeded half a step + one {cfg.DTYPE_NAME} rounding"


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

    bound = (half_step + deq.float().abs().reshape(64, 4, 128)
             * (torch.finfo(cfg.DTYPE).eps / 2) * 1.01)
    violations = int((err > bound).sum())
    print(f"\n[int4] max err {err.max():.3e}, max half-step {half_step.max():.3e}, "
          f"violations {violations}")
    assert violations == 0, f"{violations} elements exceeded half a step + one {cfg.DTYPE_NAME} rounding"


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

    # 7 projections per layer: 168 on 0.5B's 24 layers, 196 on 1.5B's 28
    n_layers = M.QwenConfig().num_layers
    assert stats["quantized_tensors"] == 7 * n_layers, stats["quantized_tensors"]
    print(f"\n[coverage] {stats['quantized_tensors']} tensors quantized, "
          f"{stats['bytes_quantizable_original']/1e6:.1f} MB of "
          f"{stats['bytes_original']/1e6:.1f} MB "
          f"({stats['bytes_quantizable_original']/stats['bytes_original']*100:.1f}%)")


# Whole-model compression depends on how much of the model is the tied
# embedding, which stays unquantized -- so the expected range is per model,
# around the measured value, not one range for every checkpoint.
EXPECTED_COMPRESSION = {
    "qwen2.5-0.5b": {"int8": (1.5, 1.7), "int4": (2.0, 2.3)},     # 1.57x, 2.15x
    "qwen2.5-1.5b": {"int8": (1.65, 1.85), "int4": (2.55, 2.8)},  # 1.74x, 2.68x
}


@pytest.mark.parametrize("mode", ["int8", "int4"])
def test_whole_model_compression_is_what_we_claim(mode):
    """The model-level number, not the per-tensor one. Because the embedding
    stays fp16, INT4 gives ~2.2x on the whole model even though the tensors it
    touches shrink ~3.8x. Both numbers are reported; only one of them is 'the
    model is N times smaller'."""
    if cfg.MODEL_SLUG not in EXPECTED_COMPRESSION:
        pytest.skip(f"no measured compression range recorded for {cfg.MODEL_SLUG}")
    lo, hi = EXPECTED_COMPRESSION[cfg.MODEL_SLUG][mode]
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


# --- the fused dequant-matmul kernels ---------------------------------------
#
# These are validated against `Int4Tensor.dequantize() @ x` — the step-1
# reference — and NOT against fp16. That distinction is the whole point:
# quantization error is expected and was measured in bench/perplexity.py, while
# KERNEL error is not expected at all. Comparing against fp16 would fold the two
# together and let a genuine kernel bug hide inside the quantization loss.

kernels = pytest.importorskip("nano_infer.kernels")

# Only the accumulation order differs from the reference (fp32 in the kernel vs
# cuBLAS's own order), so the bar is tight -- and stated in eps of the model
# dtype. It was 5e-3, which is 5.1 eps in fp16 but only 0.64 eps in bf16. Against
# an fp64 truth computed from the exactly-dequantized weights, the kernel is at
# 0.39-0.40 eps and the dequantize-then-cuBLAS reference at 0.47-0.60 eps in
# BOTH dtypes; they differ from each other by at most 0.86 eps.
MATMUL_REL = 5 * torch.finfo(cfg.DTYPE).eps


@pytest.fixture(scope="module")
def kmod():
    return kernels.load()


QUANT_MATMUL_SHAPES = [
    (896, 896),        # q_proj / o_proj
    (4864, 896),       # gate_proj / up_proj
    (896, 4864),       # down_proj
    (128, 896),        # k_proj / v_proj (GQA: only 2 KV heads)
]


@pytest.mark.parametrize("out_f,in_f", QUANT_MATMUL_SHAPES)
@pytest.mark.parametrize("batch", [1, 4, 32])
def test_int4_matmul_matches_the_dequantized_reference(kmod, out_f, in_f, batch):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE) * 0.05
    x = torch.randn(batch, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE)
    t = Q.quantize_int4(w, Q.INT4_GROUP)

    ref = torch.nn.functional.linear(x, t.dequantize())
    got = kmod.int4_matmul(x, t.packed, t.scale, t.zero, in_f, Q.INT4_GROUP)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    rel = ((ref.float() - got.float()).abs().max()
           / ref.float().abs().max().clamp(min=1e-6)).item()
    print(f"\n[int4 matmul {out_f}x{in_f} b{batch}] rel {rel:.2e}")
    assert rel < MATMUL_REL, f"kernel disagrees with its own reference by {rel:.2e}"


@pytest.mark.parametrize("out_f,in_f", QUANT_MATMUL_SHAPES)
@pytest.mark.parametrize("batch", [1, 4, 32])
def test_int8_matmul_matches_the_dequantized_reference(kmod, out_f, in_f, batch):
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE) * 0.05
    x = torch.randn(batch, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE)
    t = Q.quantize_int8(w)

    ref = torch.nn.functional.linear(x, t.dequantize())
    got = kmod.int8_matmul(x, t.q, t.scale)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    rel = ((ref.float() - got.float()).abs().max()
           / ref.float().abs().max().clamp(min=1e-6)).item()
    print(f"\n[int8 matmul {out_f}x{in_f} b{batch}] rel {rel:.2e}")
    assert rel < MATMUL_REL, f"kernel disagrees with its own reference by {rel:.2e}"


def test_int4_kernel_reads_the_nibbles_in_the_packing_order(kmod):
    """A swapped nibble order still produces finite, plausible output — it just
    silently pairs every weight with the wrong input. Pinned with a weight whose
    even and odd columns are deliberately different, against a one-hot input."""
    in_f, out_f = 128, 8
    w = torch.zeros(out_f, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE)
    w[:, 0::2] = 1.0                      # even columns
    w[:, 1::2] = -1.0                     # odd columns
    t = Q.quantize_int4(w, group=128)

    # one-hot on column 0 (an even column) must select the +1 weights
    x = torch.zeros(1, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE)
    x[0, 0] = 1.0
    got = kmod.int4_matmul(x, t.packed, t.scale, t.zero, in_f, 128)
    ref = torch.nn.functional.linear(x, t.dequantize())
    assert torch.allclose(got.float(), ref.float(), atol=1e-2), (
        f"even-column selection wrong: got {got[0,0].item()}, "
        f"reference {ref[0,0].item()} — nibble order is likely swapped")

    # and column 1 (odd) must select the -1 weights
    x.zero_()
    x[0, 1] = 1.0
    got = kmod.int4_matmul(x, t.packed, t.scale, t.zero, in_f, 128)
    ref = torch.nn.functional.linear(x, t.dequantize())
    assert torch.allclose(got.float(), ref.float(), atol=1e-2), \
        "odd-column selection wrong — nibble order is swapped"


def test_int4_kernel_uses_per_group_scales(kmod):
    """Same trap as the reference test: a kernel that used one scale per row
    would crush a small group to zero. Here group 0 is huge and group 1 tiny."""
    in_f, out_f = 256, 4
    w = torch.zeros(out_f, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE)
    w[:, :128] = 50.0
    w[:, 128:] = 0.01
    t = Q.quantize_int4(w, group=128)

    x = torch.zeros(1, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE)
    x[0, 200] = 1.0                        # lands in the small second group
    got = kmod.int4_matmul(x, t.packed, t.scale, t.zero, in_f, 128).float()
    assert got.abs().max() > 1e-3, (
        "the small group produced ~0: the kernel is using a single row-wide "
        "scale instead of per-group scales")
    ref = torch.nn.functional.linear(x, t.dequantize()).float()
    assert torch.allclose(got, ref, atol=1e-3)


def test_quantized_matmul_never_materializes_the_weight(kmod):
    """The defining constraint: unpacking happens in registers, so no fp16 copy
    of W is ever allocated. Measured, not asserted by inspection — a
    dequantize-then-linear implementation would allocate out*in*2 bytes here.
    """
    out_f, in_f = 4864, 896
    torch.manual_seed(0)
    w = torch.randn(out_f, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE) * 0.05
    t = Q.quantize_int4(w, Q.INT4_GROUP)
    x = torch.randn(1, in_f, dtype=cfg.DTYPE, device=cfg.DEVICE)

    kmod.int4_matmul(x, t.packed, t.scale, t.zero, in_f, Q.INT4_GROUP)  # warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    kmod.int4_matmul(x, t.packed, t.scale, t.zero, in_f, Q.INT4_GROUP)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - before

    dequantized_bytes = out_f * in_f * 2
    print(f"\n[no-materialize] peak extra allocation {peak_extra} B; a "
          f"dequantized copy would be {dequantized_bytes} B")
    assert peak_extra < dequantized_bytes // 4, (
        f"allocated {peak_extra} B — that looks like a dequantized weight copy, "
        "which would move more bytes than fp16 and defeat the whole point")


# --- the packed path (what the engine actually runs) ------------------------

@pytest.mark.parametrize("mode", ["int8", "int4"])
def test_packed_weights_land_on_exactly_the_round_trip_grid(mode):
    """THE claim the acceptance table rests on, asserted exactly.

    The table's perplexity column is measured on the round-tripped-fp16 path;
    its speed and VRAM columns are measured on the packed path. That is only
    honest if the two describe the same model -- if the packed weights encoded a
    different grid, the quality number would belong to something else.

    They do not merely agree closely: every quantized tensor reconstructs
    BIT-IDENTICALLY, because both paths call the same quantize_int8/int4 and
    differ only in whether the reconstruction is materialized. So this is an
    exact bar, and it is the right place to put one -- see the sibling test for
    why generation itself cannot carry one.
    """
    weights = M.load_weights()
    round_trip, _ = Q.quantize_weights(weights, mode)
    packed, stats = Q.pack_weights(weights, mode)

    checked, worst, worst_name = 0, 0.0, None
    for name, obj in packed.items():
        if isinstance(obj, torch.Tensor):
            continue                      # left in fp16 on purpose (embedding)
        checked += 1
        diff = (obj.dequantize().float() - round_trip[name].float()).abs().max().item()
        if diff > worst:
            worst, worst_name = diff, name
    print(f"\n[packed {mode}] {checked} quantized tensors; worst reconstruction "
          f"difference from the round-trip path {worst:.3e}"
          + (f" at {worst_name}" if worst_name else "")
          + f"; {stats['bytes_quantized']/1e6:.1f} MB, {stats['compression']:.2f}x")
    assert checked > 0, "pack_weights quantized nothing"
    assert worst == 0.0, (
        f"packed and round-trip weights differ by up to {worst:.3e} at "
        f"{worst_name} — the acceptance table's perplexity column would not "
        "describe the model its speed column measures")


@pytest.mark.parametrize("mode", ["int8", "int4"])
def test_packed_weights_generate_within_a_rounding_of_the_round_trip(kmod, mode):
    """Same grid (above), so the only thing left between the two paths is the
    ORDER OF THE ARITHMETIC -- and that cannot be asserted token-for-token.

    WHY NOT. The round-trip path hands cuBLAS a materialized fp16/bf16 weight;
    the packed path runs the fused dequant-matmul kernel, which unpacks in
    registers and accumulates its dot products in a different order. Same
    numbers, different summation order, so the last bits differ. This test used
    to require identical tokens and passed on 0.5B/fp16 -- on Qwen2.5-1.5B/bf16
    it failed 20/40, not because anything regressed but because bf16 has three
    fewer fraction bits, so the same rounding lands on the far side of more
    argmax ties. Gotcha #16 again: a bar calibrated on one dtype is not a bar.

    THE BAR THAT DOES MEAN SOMETHING. Compare the two paths' logits, up to and
    including the first token they disagree on (before which both have the same
    history, so no teacher-forcing is needed), and require the difference to
    stay within a rounding -- a few tens of ULPs. A real defect in the packed
    path -- wrong nibble order, a row-wide scale where a per-group one belongs,
    a dropped zero point -- moves logits by orders of magnitude more, which is
    what the kernel-level tests above catch directly.
    """
    from tests._drift import (QUANT_PATH_DRIFT_ULPS, drift_between,
                              first_divergence, record_step_logits)
    weights = M.load_weights()
    cf = M.QwenConfig()
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(1000, 5000, (2, 12), generator=g).to(cfg.DEVICE)
    steps = 20

    round_trip, _ = Q.quantize_weights(weights, mode)
    packed, stats = Q.pack_weights(weights, mode)

    with M.using_kernels(True):
        with record_step_logits("forward_paged") as rt_logs:
            a = M.generate_paged(ids, round_trip, cf, steps)
        with record_step_logits("forward_paged") as pk_logs:
            b = M.generate_paged(ids, packed, cf, steps)

    agree = int((a == b).sum())
    print(f"\n[packed {mode}] {agree}/{a.numel()} tokens match the round-trip "
          f"reference; {stats['bytes_quantized']/1e6:.1f} MB, "
          f"{stats['compression']:.2f}x")
    for row in range(ids.shape[0]):
        diverged = first_divergence(a[row], b[row])
        upto = steps - 1 if diverged is None else diverged
        worst, at = drift_between(rt_logs, pk_logs, row, upto)
        print(f"  row {row}: first divergence "
              f"{'none' if diverged is None else diverged}, "
              f"{upto + 1} steps compared, worst {worst:.2f} {cfg.DTYPE_NAME} "
              f"ULPs at step {at}")
        assert worst <= QUANT_PATH_DRIFT_ULPS, (
            f"packed vs round-trip drift {worst:.1f}u at step {at} of row {row} "
            f"exceeds {QUANT_PATH_DRIFT_ULPS}u — too large to be a summation "
            "order difference; the packed path is decoding a different grid")


@pytest.mark.parametrize("mode", ["int8", "int4"])
def test_packed_weights_hold_no_fp16_copy(mode):
    """The memory claim, asserted rather than assumed.

    Quantization only saves VRAM if the fp16 originals are gone. A packed dict
    must contain NO float tensor for any quantizable projection — if one
    survived, the model would be larger than fp16, not smaller.
    """
    weights = M.load_weights()
    packed, stats = Q.pack_weights(weights, mode)

    # match by NAME: is_quantizable() inspects a tensor's rank, and a packed
    # entry is not a tensor at all
    leaked = [n for n, w in packed.items()
              if n.endswith(Q.QUANTIZABLE_SUFFIXES) and torch.is_tensor(w)]
    assert not leaked, f"{len(leaked)} projections still hold fp16 tensors: {leaked[:3]}"

    held = sum(w.nbytes() if Q.is_packed(w) else w.numel() * w.element_size()
               for w in packed.values())
    assert held == stats["bytes_quantized"], "reported size does not match what is held"
    assert held < sum(w.numel() * w.element_size() for w in weights.values())
